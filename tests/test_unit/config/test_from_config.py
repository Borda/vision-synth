"""Comprehensive integration tests for Compose.from_config() classmethod.

Tests cover pipeline construction from TransformSpec lists, backend dispatch, error handling, probability masking,
data_keys forwarding, and fusion verification.

"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest
import torch

import fused_transforms.dispatch.resolver as resolver_mod
from fused_transforms import Compose, FusedCompose, TransformSpec
from fused_transforms._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFromConfigBasicKornia:
    """from_config with backend='kornia' — basic pipeline construction."""

    def test_single_rotation_produces_output(self):
        """Single rotation spec on kornia backend produces output with input shape."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 64, 64)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_two_geometric_specs_fuse(self):
        """Two consecutive geometric specs (rotation + hflip) get fused into one warp.

        Verifies that the fusion planner detects compatible geometric ops and collapses them into a single
        FusedAffineSegment, reflected by n_warps_saved >= 1.

        """
        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}),
            TransformSpec(operation="hflip", params={}),
        ]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 64, 64)
        pipe(image)
        assert pipe.n_warps_saved >= 1, f"Expected fusion, got plan: {pipe.fusion_plan}"

    def test_returns_fused_compose_instance(self):
        """from_config returns a FusedCompose instance, not the base Compose."""
        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = Compose.from_config(specs, backend="kornia")
        assert isinstance(pipe, FusedCompose), f"Expected FusedCompose, got {type(pipe).__name__}"

    def test_fusion_plan_descriptors_populated(self):
        """fusion_plan_descriptors is a non-empty list after construction from specs."""
        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-15.0, 15.0)}),
            TransformSpec(operation="hflip", params={}, prob=0.5),
        ]
        pipe = Compose.from_config(specs, backend="kornia")
        descriptors = pipe.fusion_plan_descriptors
        assert isinstance(descriptors, list)
        assert len(descriptors) >= 1, "Expected at least one segment descriptor"

    def test_output_no_nan(self):
        """Pipeline forward pass produces no NaN values for valid rotation specs."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-45.0, 45.0)}, prob=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(4, 3, 32, 32)
        out = pipe(image)
        assert not torch.isnan(out).any(), "from_config pipeline produced NaN values"


class TestFromConfigEmptySpecs:
    """from_config with empty specs list returns identity pipeline."""

    def test_empty_specs_identity(self):
        """Empty specs list produces an identity pipeline that returns input unchanged."""
        pipe = Compose.from_config([], backend="kornia")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape
        assert torch.allclose(out, image), "Empty specs should produce identity pipeline"

    def test_empty_specs_returns_fused_compose(self):
        """Empty specs still returns a FusedCompose instance (not a different fallback type)"""
        pipe = Compose.from_config([], backend="kornia")
        assert isinstance(pipe, FusedCompose)

    def test_empty_specs_forwards_output_backend(self) -> None:
        """Empty specs pipeline still honors output_backend='numpy' and converts the output.

        Even with no transforms, the output_backend conversion path must run; this guards against an optimization that
        would short-circuit identity pipelines and skip output conversion.

        """
        pipe = Compose.from_config([], backend="kornia", output_backend="numpy")
        image = torch.rand(1, 3, 32, 32)
        out = pipe(image)
        assert isinstance(out, np.ndarray)
        assert out.shape == (32, 32, 3)


class TestFromConfigErrors:
    """Error paths for from_config."""

    @pytest.mark.parametrize(
        "specs_arg",
        [
            pytest.param([TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})], id="with_specs"),
            pytest.param([], id="empty_specs"),
        ],
    )
    def test_invalid_backend_raises(self, specs_arg):
        """Unknown backend raises ValueError regardless of whether specs is empty or not."""
        with pytest.raises(ValueError, match="unknown backend"):
            Compose.from_config(specs_arg, backend="nonexistent_backend")

    def test_invalid_op_raises_at_construction(self):
        """Invalid operation name should raise ValueError at construction, not at forward time.

        Fail-fast: catching the typo in 'definitely_not_a_real_op' at Compose.from_config() time gives a
        clear traceback at the call site, instead of a misleading shape/key error inside a deep forward pass.

        """
        specs = [TransformSpec(operation="definitely_not_a_real_op", params={})]
        with pytest.raises(ValueError, match="unknown operation"):
            Compose.from_config(specs, backend="kornia")

    def test_params_cannot_shadow_transform_probability(self) -> None:
        """Putting 'prob' inside spec.params raises ValueError to prevent shadowing spec.prob.

        TransformSpec.prob is the single source of truth for probability; allowing 'prob' to slip into params would
        create two competing values and silently override the spec-level prob.

        """
        specs = [TransformSpec(operation="hflip", params={"prob": 0.1}, prob=0.5)]
        with pytest.raises(ValueError, match="must not include 'prob'"):
            Compose.from_config(specs, backend="kornia")


class TestFromConfigProbability:
    """Per-transform probability via TransformSpec.prob."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_p_zero_never_applied(self):
        """prob=0.0 large rotation should produce output identical to input.

        A wide rotation range (-90, 90) is chosen so that even a single sampled rotation would visibly change the image;
        if anything slips through with prob=0.0, the assertion will fail.

        """
        specs = [TransformSpec(operation="rotation", params={"degrees": (-90.0, 90.0)}, prob=0.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 64, 64)
        out = pipe(image)
        assert torch.allclose(out, image, atol=1e-5), "prob=0.0 transform should never be applied"

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_p_one_hflip_always_applied(self):
        """prob=1.0 hflip should always flip the image."""
        specs = [TransformSpec(operation="hflip", params={}, prob=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        expected = image.flip(dims=[3])
        assert torch.allclose(out, expected, atol=1e-5), "prob=1.0 hflip must always flip"

    @pytest.mark.parametrize(
        "p_value",
        [
            pytest.param(0.0, id="prob=0.0"),
            pytest.param(0.5, id="prob=0.5"),
            pytest.param(1.0, id="prob=1.0"),
        ],
    )
    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_p_values_produce_valid_shape(self, p_value):
        """Pipeline preserves input shape for prob in {0.0, 0.5, 1.0}"""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=p_value)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestFromConfigTorchVision:
    """from_config with backend='torchvision'."""

    def test_hflip_works(self):
        """Hflip spec on torchvision backend constructs and runs without error."""
        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = Compose.from_config(specs, backend="torchvision")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_rotation_works(self):
        """Rotation spec on torchvision backend constructs and runs without error."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-15.0, 15.0)})]
        pipe = Compose.from_config(specs, backend="torchvision")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_scale_uses_zero_degree_affine_default(self) -> None:
        """Canonical 'scale' op maps to torchvision RandomAffine with degrees=[0,0] and the given scale range.

        TorchVision has no standalone scale transform; we route 'scale' through RandomAffine with rotation disabled. The
        test pins both the synthesized degrees and scale to detect regressions in the resolver translation.

        """
        specs = [TransformSpec(operation="scale", params={"scale": (0.8, 1.2)})]
        pipe = Compose.from_config(specs, backend="torchvision")
        transform = pipe.original_transforms[0]
        assert transform.degrees == [0.0, 0.0]
        assert transform.scale == (0.8, 1.2)
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestFromConfigAlbumentations:
    """from_config with backend='albumentations'."""

    def test_hflip_works(self):
        """Hflip spec on albumentations backend constructs and runs without error."""
        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = Compose.from_config(specs, backend="albumentations")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_rotation_translates_canonical_degrees_to_limit(self) -> None:
        """Canonical 'degrees' param maps to albumentations Rotate.limit field.

        Albumentations uses 'limit' rather than 'degrees' for its rotation range; the resolver must translate the
        canonical kwarg name so users don't have to know backend-specific naming.

        """
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="albumentations")
        transform = pipe.original_transforms[0]
        assert transform.limit == (-10.0, 10.0)

    def test_affine_translates_canonical_degrees_to_rotate(self) -> None:
        """Canonical 'degrees' and 'scale' map to albumentations Affine.rotate and Affine.scale={x,y} dict.

        Albumentations Affine takes 'rotate' (not 'degrees') and 'scale' as either a tuple or per-axis dict. The
        resolver must translate canonical names and expand scalar/tuple scale into the per-axis form albumentations
        expects.

        """
        specs = [TransformSpec(operation="affine", params={"degrees": (-5.0, 5.0), "scale": (0.9, 1.1)})]
        pipe = Compose.from_config(specs, backend="albumentations")
        transform = pipe.original_transforms[0]
        assert transform.rotate == (-5.0, 5.0)
        assert transform.scale == {"x": (0.9, 1.1), "y": (0.9, 1.1)}


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFromConfigScaleKornia:
    """Canonical scale config remains constructible on Kornia."""

    def test_scale_uses_zero_degree_affine_default(self) -> None:
        """Canonical 'scale' op maps to kornia RandomAffine with degrees=0.0 and the given scale range.

        Kornia, like torchvision, lacks a standalone scale transform; the resolver routes 'scale' through RandomAffine
        with rotation explicitly disabled. Asserting against repr() rather than attributes because kornia stores degrees
        inside an internal parameter object that is awkward to inspect directly.

        """
        specs = [TransformSpec(operation="scale", params={"scale": (0.8, 1.2)})]
        pipe = Compose.from_config(specs, backend="kornia")
        transform = pipe.original_transforms[0]
        assert "degrees=0.0" in repr(transform)
        assert "scale=(0.8, 1.2)" in repr(transform)
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFromConfigDataKeys:
    """from_config with data_keys parameter."""

    def test_data_keys_input_mask(self):
        """data_keys=['input', 'mask'] produces a pipeline that returns (image, mask) tuple with shapes preserved.

        Verifies that auxiliary targets (segmentation masks) flow through the same geometric transform as the image and
        that the call signature pipe(img, mask) returns matched-shape outputs as a tuple.

        """
        specs = [TransformSpec(operation="rotation", params={"degrees": (-15.0, 15.0)})]
        pipe = Compose.from_config(specs, backend="kornia", data_keys=["input", "mask"])
        img = torch.rand(2, 3, 32, 32)
        mask = torch.randint(0, 3, (2, 1, 32, 32)).float()
        out = pipe(img, mask)
        assert isinstance(out, tuple), f"Expected tuple with data_keys, got {type(out)}"
        assert len(out) == 2
        out_img, out_mask = out
        assert out_img.shape == img.shape
        assert out_mask.shape == mask.shape


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFromConfigInterpolationPadding:
    """from_config forwards interpolation and padding_mode."""

    @pytest.mark.parametrize("interpolation", ["bilinear", "nearest", "bicubic"])
    def test_interpolation_modes(self, interpolation):
        """All supported interpolation modes (bilinear, nearest, bicubic) construct and run."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="kornia", interpolation=interpolation)
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_padding_modes(self, padding_mode):
        """All supported padding modes (zeros, border, reflection) construct and run."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="kornia", padding_mode=padding_mode)
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape


class TestFromConfigUserWarning:
    """UserWarning emitted when spec.prob != 1.0 and the backend cannot set p= on the transform."""

    def test_p_not_applied_emits_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UserWarning fires when prob≠1.0 and backend transform has no settable prob attribute."""

        class _NoPTransform:
            """Mock: explicitly rejects p= kwarg, has no settable prob attribute (slots)."""

            __slots__ = ()

            def __init__(self, **_kwargs: object) -> None:
                if "p" in _kwargs:
                    raise TypeError("_NoPTransform does not accept 'p' keyword argument")

        def _mock_resolve(transform: str, backend: str) -> type:
            return _NoPTransform

        monkeypatch.setattr(resolver_mod, "resolve_op", _mock_resolve)
        monkeypatch.setattr(resolver_mod, "SUPPORTED_BACKENDS", frozenset({"mock_backend"}))

        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with contextlib.suppress(ValueError, Exception):
                # FusedCompose.__init__ rejects unknown transforms after the warning fires
                Compose.from_config(specs, backend="mock_backend")

        matching = [
            warning
            for warning in recorded
            if issubclass(warning.category, UserWarning) and "could not be applied" in str(warning.message)
        ]
        assert matching, (
            f"Expected UserWarning about p= not applied. Got: {[str(warning.message) for warning in recorded]}"
        )


class TestFromConfigBackendGap:
    """Ops that are valid globally but unsupported by a specific backend."""

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_rotation90_unsupported_by_torchvision_raises(self) -> None:
        """Rotation90 is in SUPPORTED_OPS but TorchVision has no such transform."""
        specs = [TransformSpec(operation="rotation90", params={}, prob=1.0)]
        with pytest.raises(ValueError, match="does not support op"):
            Compose.from_config(specs, backend="torchvision")
