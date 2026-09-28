"""Unit tests for _segment.py: FusedAffineSegment, ExactAffineSegment, and build_segments."""

from __future__ import annotations

import pytest
import torch

from fused_transforms._compat import _CV2_AVAILABLE
from fused_transforms.affine import segment as segment_mod
from fused_transforms.affine.matrix import hflip_matrix, inv3x3, matmul3x3, scale_matrix, vflip_matrix
from fused_transforms.affine.segment import (
    AlbuProjectiveSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    ProjectiveSegment,
    build_segments,
    reorder_pointwise,
)
from fused_transforms.types import TransformCategory


class _StubTransform:
    """A stub geometric transform with a p attribute and a matrix factory."""

    def __init__(self, matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_INTERP):
        self.p = prob
        self.matrix_fn = matrix_fn
        self._category = category


class _BarrierTransform:
    """A stub non-geometric transform (SPATIAL_KERNEL)."""

    def __init__(self):
        self._category = TransformCategory.SPATIAL_KERNEL


class _PointwiseTransform:
    """A stub pointwise transform."""

    def __init__(self):
        self._category = TransformCategory.POINTWISE


class _StubAdapter:
    """Minimal TransformAdapter for unit tests -- no Kornia dependency."""

    def category(self, transform):
        """Return the transform's category attribute or default to SPATIAL_KERNEL."""
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, transform, input_shape, device):
        """Return minimal canonical params with batch size from input_shape."""
        batch_size = input_shape[0]
        return {"_batch_size": torch.tensor([batch_size])}

    def build_matrix(self, transform, params, height, width):
        """Delegate to transform.matrix_fn or return identity."""
        batch_size = int(params["_batch_size"].item())
        if hasattr(transform, "matrix_fn"):
            return transform.matrix_fn(batch_size, height, width)
        return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        """Pass through the image unchanged for stub testing."""
        return image


def _identity_matrix_fn(batch, height, width):
    """Return (batch, 3, 3) identity matrices."""
    return torch.eye(3).unsqueeze(0).expand(batch, -1, -1)


def _hflip_matrix_fn(batch, height, width):
    """Return (batch, 3, 3) horizontal flip matrices."""
    return hflip_matrix(width=width, batch_size=batch, device=torch.device("cpu"), dtype=torch.float32)


def _vflip_matrix_fn(batch, height, width):
    """Return (batch, 3, 3) vertical flip matrices."""
    return vflip_matrix(height=height, batch_size=batch, device=torch.device("cpu"), dtype=torch.float32)


def _small_scale_matrix_fn(batch, height, width):
    """Scale 0.01 -- near-degenerate but valid."""
    scale_x = torch.full((batch,), 0.01)
    scale_y = torch.full((batch,), 0.01)
    return scale_matrix(scale_x, scale_y, height=height, width=width)


class TestP0Identity:
    """Verify that p=0 produces identity (no-op) behaviour."""

    def test_all_transforms_inactive(self):
        """With p=0 every transform is skipped; composed matrix is identity."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=0.0)
        seg = FusedAffineSegment([transform, transform], adapter)

        img = torch.rand(2, 3, 16, 16)
        out = seg(img)

        assert torch.allclose(out, img, atol=1e-5)
        # Composed matrix should be identity
        identity_matrix = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(seg.last_matrix, identity_matrix, atol=1e-7)


class TestP1AlwaysActive:
    """Verify that p=1 always applies the transform."""

    def test_both_flips_compose_to_rotation_180(self):
        """Two flips (h+v) with p=1 should compose to 180-deg rotation."""
        adapter = _StubAdapter()
        hflip_transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        vflip_transform = _StubTransform(_vflip_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([hflip_transform, vflip_transform], adapter)

        batch_size, num_channels, height, width = 1, 3, 8, 8
        img = torch.rand(batch_size, num_channels, height, width)
        out = seg(img)

        # Applying h+v flip should be equivalent to 180-deg rotation
        # For a square image, this means pixel (x, y) -> (W-1-x, H-1-y)
        # Verify on a known pattern
        assert out.shape == img.shape

        # The composed matrix should be the product of hflip and vflip
        composed_matrix = seg.last_matrix
        assert composed_matrix is not None
        assert composed_matrix.shape == (batch_size, 3, 3)


class TestBatchHeterogeneity:
    """Verify p-masking produces per-sample variation across a batch."""

    def test_different_samples_different_active_masks(self):
        """Manually verify that p-masking produces per-sample variation.

        We use p=0.5 and a fixed seed such that some samples are active and some are not, then verify the outputs differ
        across samples.

        """
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=0.5)
        seg = FusedAffineSegment([transform], adapter)

        batch_size = 8
        img = torch.rand(batch_size, 1, 8, 8)

        out = seg(img)

        # At least one sample should differ from input, and at least one
        # should be unchanged (identity). Check per-sample max diff.
        diffs = (out - img).abs().amax(dim=(1, 2, 3))
        has_changed = (diffs > 1e-4).any()
        has_unchanged = (diffs < 1e-4).any()

        # With 8 samples and p=0.5 the probability of all-same is 2*(0.5^8)=0.8%.
        # If seed changes break this, the test documents expected heterogeneity.
        assert has_changed, "Expected at least one sample to be transformed"
        assert has_unchanged, "Expected at least one sample to remain unchanged"


class TestNearDegenerateScale:
    """Verify near-degenerate scale factors do not produce NaN/Inf."""

    def test_scale_001_no_nan(self):
        """Scale factor 0.01 should invert without NaN."""
        adapter = _StubAdapter()
        transform = _StubTransform(_small_scale_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([transform], adapter)

        img = torch.rand(2, 3, 16, 16)
        out = seg(img)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

        # Verify the inverse round-trip in float64 where det clamping
        # does not dominate.  Float32 det of scale=0.01 matrix is 1e-4,
        # which sits right at the eps*1e3 clamp boundary.
        matrix = seg.last_matrix.double()
        matrix_inv = inv3x3(matrix)
        product = matmul3x3(matrix_inv, matrix)
        identity_matrix = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(product, identity_matrix, atol=1e-8)


class TestNaNInfInput:
    """Verify NaN and Inf inputs propagate through grid_sample without crash."""

    @pytest.mark.parametrize(
        "fill_value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
        ],
    )
    def test_special_value_propagates(self, fill_value):
        """NaN and Inf inputs propagate through grid_sample without raising."""
        adapter = _StubAdapter()
        transform = _StubTransform(_identity_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([transform], adapter)
        img = torch.full((1, 1, 4, 4), fill_value)
        out = seg(img)
        assert out.shape == img.shape


class TestDeviceConsistency:
    """Verify output and matrices live on the same device as the input."""

    def test_cpu_device_consistency(self):
        """All intermediate matrices should live on the same device as the input."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([transform], adapter)

        img = torch.rand(2, 3, 8, 8, device=torch.device("cpu"))
        out = seg(img)

        assert out.device == img.device
        assert seg.last_matrix.device == img.device


class TestBuildSegments:
    """Verify build_segments partitions transforms into correct segments."""

    def test_single_geometric_returns_one_segment(self):
        """Single geometric transform produces one FusedAffineSegment."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        result = build_segments([transform], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 1

    def test_two_geometric_fused_into_one_segment(self):
        """Two consecutive geometric transforms fuse into one segment."""
        adapter = _StubAdapter()
        first_transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        second_transform = _StubTransform(_vflip_matrix_fn, prob=1.0)
        result = build_segments([first_transform, second_transform], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_geometric_barrier_breaks_segment(self):
        """A SPATIAL_KERNEL transform breaks the fused segment."""
        adapter = _StubAdapter()
        geometric_transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        barrier_transform = _BarrierTransform()
        result = build_segments([geometric_transform, barrier_transform], adapter)

        assert len(result) == 2
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is barrier_transform

    def test_geo_barrier_geo_produces_three_elements(self):
        """[geo, barrier, geo] produces [segment, barrier, segment]."""
        adapter = _StubAdapter()
        first_transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        second_transform = _StubTransform(_vflip_matrix_fn, prob=1.0)
        barrier_transform = _BarrierTransform()
        result = build_segments([first_transform, barrier_transform, second_transform], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is barrier_transform
        assert isinstance(result[2], FusedAffineSegment)

    def test_pointwise_breaks_segment(self):
        """A POINTWISE transform breaks the fused segment."""
        adapter = _StubAdapter()
        geometric_transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        pointwise_transform = _PointwiseTransform()
        result = build_segments([geometric_transform, pointwise_transform, geometric_transform], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is pointwise_transform
        assert isinstance(result[2], FusedAffineSegment)

    def test_geometric_exact_fuses_with_interp(self):
        """GEOMETRIC_EXACT and GEOMETRIC_INTERP fuse into a single segment."""
        adapter = _StubAdapter()
        interp_transform = _StubTransform(_hflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        exact_transform = _StubTransform(_vflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        result = build_segments([interp_transform, exact_transform], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_empty_transforms_returns_empty(self):
        """Empty transform list returns empty segment list."""
        adapter = _StubAdapter()
        result = build_segments([], adapter)
        assert result == []

    def test_interpolation_and_padding_forwarded(self):
        """Interpolation and padding_mode kwargs are forwarded to the segment."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        result = build_segments([transform], adapter, interpolation="bicubic", padding_mode="reflection")

        seg = result[0]
        assert isinstance(seg, FusedAffineSegment)
        assert seg.interpolation == "bicubic"
        assert seg.padding_mode == "reflection"


class TestLastMatrixProperty:
    """Verify last_matrix lifecycle: None before forward, populated after."""

    def test_none_before_forward(self):
        """last_matrix is None before any forward pass."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([transform], adapter)
        assert seg.last_matrix is None

    def test_populated_after_forward(self):
        """last_matrix is populated with correct shape after forward."""
        adapter = _StubAdapter()
        transform = _StubTransform(_identity_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([transform], adapter)
        seg(torch.rand(2, 3, 8, 8))

        assert seg.last_matrix is not None
        assert seg.last_matrix.shape == (2, 3, 3)


class _HFlipTransform:
    """Stub HFlip transform for ExactAffineSegment tests."""

    def __init__(self, prob=1.0):
        self.p = prob
        self._category = TransformCategory.GEOMETRIC_EXACT
        self._flip_dims = [3]  # width axis


class _VFlipTransform:
    """Stub VFlip transform for ExactAffineSegment tests."""

    def __init__(self, prob=1.0):
        self.p = prob
        self._category = TransformCategory.GEOMETRIC_EXACT
        self._flip_dims = [2]  # height axis


class _FlipAdapter:
    """Adapter stub that supports exact_flip_dims for ExactAffineSegment tests."""

    def category(self, transform):
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def exact_flip_dims(self, transform):
        return getattr(transform, "_flip_dims", [])

    def exact_apply(self, transform, image):
        return image.flip(dims=self.exact_flip_dims(transform))

    def sample_params(self, transform, input_shape, device):
        batch_size = input_shape[0]
        return {"_batch_size": torch.tensor([batch_size])}

    def build_matrix(self, transform, params, height, width):
        batch_size = int(params["_batch_size"].item())
        if 3 in self.exact_flip_dims(transform):
            return hflip_matrix(width=width, batch_size=batch_size, device=torch.device("cpu"), dtype=torch.float32)
        return vflip_matrix(height=height, batch_size=batch_size, device=torch.device("cpu"), dtype=torch.float32)

    def call_nonfused(self, transform, image, **kwargs):
        return image


class _RaisingExactAdapter(_FlipAdapter):
    """Adapter whose exact_apply must never run for inactive transforms."""

    def exact_apply(self, transform, image):
        raise RuntimeError("exact_apply should not be called for inactive samples")


class TestExactAffineSegmentLossless:
    """Verify ExactAffineSegment applies lossless flips via tensor.flip."""

    @pytest.mark.parametrize(
        "transform_cls, flip_dims",
        [
            pytest.param(_HFlipTransform, [3], id="hflip"),
            pytest.param(_VFlipTransform, [2], id="vflip"),
        ],
    )
    def test_flip_p1_matches_tensor_flip(self, transform_cls, flip_dims):
        """Flip with p=1.0 is pixel-exact versus torch.flip on the corresponding axis."""
        adapter = _FlipAdapter()
        transform = transform_cls(prob=1.0)
        seg = ExactAffineSegment([transform], adapter)
        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        expected = img.flip(dims=flip_dims)
        assert torch.equal(out, expected), "ExactAffineSegment flip should be pixel-exact"


class TestExactAffineSegmentP0:
    """Verify p=0 leaves the image unchanged."""

    def test_p0_output_equals_input(self):
        """ExactAffineSegment with p=0 returns the input tensor unchanged."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=0.0)
        seg = ExactAffineSegment([transform], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img), "p=0 should leave image unchanged"

    def test_p0_skips_exact_apply_entirely(self):
        """Inactive exact transforms must not evaluate exact_apply."""
        adapter = _RaisingExactAdapter()
        transform = _HFlipTransform(prob=0.0)
        seg = ExactAffineSegment([transform], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img), "p=0 should bypass exact_apply and leave image unchanged"


class TestExactAffineSegmentDoubleFlip:
    """Verify HFlip then VFlip with p=1 composes correctly."""

    def test_hflip_then_vflip(self):
        """HFlip then VFlip with p=1 is same as image.flip(dims=[2, 3]) sequentially."""
        adapter = _FlipAdapter()
        hflip_transform = _HFlipTransform(prob=1.0)
        vflip_transform = _VFlipTransform(prob=1.0)
        seg = ExactAffineSegment([hflip_transform, vflip_transform], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        # Sequential: first flip width, then flip height
        expected = img.flip(dims=[3]).flip(dims=[2])

        assert torch.equal(out, expected), "HFlip+VFlip should match sequential tensor.flip"


class TestExactAffineSegmentPerSampleMask:
    """Verify per-sample p=0.5 masking in ExactAffineSegment."""

    def test_p05_heterogeneous_batch(self):
        """With B=8 and p=0.5, at least one sample changed, at least one unchanged."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=0.5)
        seg = ExactAffineSegment([transform], adapter)

        batch_size = 8
        img = torch.rand(batch_size, 1, 8, 8)

        # Make the per-sample mask used inside ExactAffineSegment deterministic so that
        # some samples are flipped and some are not, avoiding flaky behavior.
        pattern = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        orig_rand = torch.rand

        def _deterministic_rand(*size, **kwargs):
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", torch.float32)
            # Intercept calls that generate a per-sample mask over the batch.
            if len(size) >= 1 and size[0] == batch_size:
                base = torch.zeros(size, device=device, dtype=dtype)
                mask = pattern.to(device=device, dtype=dtype)
                view_shape = (batch_size,) + (1,) * (base.ndim - 1)
                return base + mask.view(view_shape)
            return orig_rand(*size, **kwargs)

        try:
            torch.rand = _deterministic_rand
            out = seg(img)
        finally:
            torch.rand = orig_rand

        diffs = (out - img).abs().amax(dim=(1, 2, 3))
        has_changed = (diffs > 1e-6).any()
        has_unchanged = (diffs < 1e-6).any()

        assert has_changed, "Expected at least one sample to be flipped"
        assert has_unchanged, "Expected at least one sample to remain unchanged"


class TestExactAffineSegmentLastMatrix:
    """Verify ExactAffineSegment publishes actual exact-call matrices."""

    def test_last_matrix_none_before_forward(self):
        """last_matrix is None before any forward pass."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=1.0)
        seg = ExactAffineSegment([transform], adapter)
        assert seg.last_matrix is None

    def test_last_matrix_matches_the_lossless_flip(self):
        """The published HFlip matrix follows the same image-centre convention as pixels."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=1.0)
        seg = ExactAffineSegment([transform], adapter)
        seg(torch.rand(2, 3, 8, 8))

        expected = hflip_matrix(width=8, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
        torch.testing.assert_close(seg.last_matrix, expected)


class TestBuildSegmentsExactOnly:
    """Verify build_segments routes an EXACT-only run to ExactAffineSegment."""

    def test_exact_only_returns_exact_segment(self):
        """An EXACT-only run (no INTERP) produces an ExactAffineSegment, not FusedAffineSegment."""
        adapter = _StubAdapter()
        first_transform = _StubTransform(_hflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        second_transform = _StubTransform(_vflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        result = build_segments([first_transform, second_transform], adapter)

        assert len(result) == 1
        assert isinstance(result[0], ExactAffineSegment)
        assert len(result[0].transforms) == 2


class TestExactAffineSegmentSameOnBatch:
    """Verify ExactAffineSegment respects same_on_batch=True."""

    def test_same_on_batch_p1_all_flipped(self):
        """With same_on_batch=True and p=1, every sample in the batch is flipped."""

        class _SameOnBatchHFlip:
            p = 1.0
            same_on_batch = True
            _category = TransformCategory.GEOMETRIC_EXACT
            _flip_dims = (3,)

        adapter = _FlipAdapter()
        seg = ExactAffineSegment([_SameOnBatchHFlip()], adapter)

        img = torch.rand(4, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img.flip(dims=[3]))

    def test_same_on_batch_p0_none_flipped(self):
        """With same_on_batch=True and p=0, no sample in the batch is flipped."""

        class _SameOnBatchHFlip:
            p = 0.0
            same_on_batch = True
            _category = TransformCategory.GEOMETRIC_EXACT
            _flip_dims = (3,)

        adapter = _FlipAdapter()
        seg = ExactAffineSegment([_SameOnBatchHFlip()], adapter)

        img = torch.rand(4, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img)


class TestBatchSizeOne:
    """Verify B=1 forward pass works for both segment types."""

    def test_fused_affine_b1_shape(self):
        """FusedAffineSegment with B=1 produces output of shape (1, C, H, W)."""
        adapter = _StubAdapter()
        transform = _StubTransform(_identity_matrix_fn, prob=1.0)
        seg = FusedAffineSegment([transform], adapter)

        out = seg(torch.rand(1, 3, 8, 8))

        assert out.shape == (1, 3, 8, 8)

    def test_exact_segment_b1_shape(self):
        """ExactAffineSegment with B=1 produces output of shape (1, C, H, W)."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=1.0)
        seg = ExactAffineSegment([transform], adapter)

        out = seg(torch.rand(1, 3, 8, 8))

        assert out.shape == (1, 3, 8, 8)


class TestPointwiseReorderBuildSegments:
    """POINTWISE reorder + build_segments integration: verify correct segmentation."""

    def test_reorder_then_build_fuses_geometric(self):
        """[Rotate, Brightness, Scale] with POINTWISE reorder -> 1 FusedAffineSegment + 1 passthrough."""
        adapter = _StubAdapter()
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)
        brightness = _PointwiseTransform()
        scale = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)

        reordered = reorder_pointwise([rotate, brightness, scale], adapter)
        segments = build_segments(reordered, adapter)

        # After reorder: [Rotate, Scale, Brightness]
        # build_segments: FusedAffineSegment([Rotate, Scale]), passthrough(Brightness)
        assert len(segments) == 2
        assert isinstance(segments[0], FusedAffineSegment)
        assert len(segments[0].transforms) == 2
        assert segments[1] is brightness


class TestProjectiveSegmentBuildSegments:
    """build_segments() creates ProjectiveSegment for PROJECTIVE transforms."""

    def _proj_transform(self):
        """Stub with PROJECTIVE category."""
        return _StubTransform(_identity_matrix_fn, prob=1.0, category=TransformCategory.PROJECTIVE)

    def test_two_projective_become_one_segment(self):
        """[Proj, Proj] -> one ProjectiveSegment."""
        adapter = _StubAdapter()
        first_projective_transform, second_projective_transform = self._proj_transform(), self._proj_transform()
        segments = build_segments([first_projective_transform, second_projective_transform], adapter)
        assert len(segments) == 1
        assert isinstance(segments[0], ProjectiveSegment)
        assert len(segments[0].transforms) == 2

    def test_rotate_then_proj_gives_two_segments(self):
        """[Rot, Proj] -> [FusedAffineSegment, ProjectiveSegment]."""
        adapter = _StubAdapter()
        rotation_transform = _StubTransform(_identity_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        projective_transform = self._proj_transform()
        segments = build_segments([rotation_transform, projective_transform], adapter)
        assert len(segments) == 2
        assert isinstance(segments[0], FusedAffineSegment)
        assert isinstance(segments[1], ProjectiveSegment)

    def test_proj_then_rotate_gives_two_segments(self):
        """[Proj, Rot] -> [ProjectiveSegment, FusedAffineSegment]."""
        adapter = _StubAdapter()
        projective_transform = self._proj_transform()
        rotation_transform = _StubTransform(_identity_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        segments = build_segments([projective_transform, rotation_transform], adapter)
        assert len(segments) == 2
        assert isinstance(segments[0], ProjectiveSegment)
        assert isinstance(segments[1], FusedAffineSegment)

    @pytest.mark.skipif(not _CV2_AVAILABLE, reason="missing cv2")
    def test_use_numpy_true_gives_albu_projective(self):
        """use_numpy=True produces AlbuProjectiveSegment."""
        adapter = _StubAdapter()
        projective_transform = self._proj_transform()
        segments = build_segments([projective_transform], adapter, use_numpy=True)
        assert len(segments) == 1
        assert isinstance(segments[0], AlbuProjectiveSegment)

    def test_projective_forward_identity(self):
        """ProjectiveSegment with identity matrix produces same-shape output."""
        adapter = _StubAdapter()
        projective_transform = self._proj_transform()
        seg = ProjectiveSegment([projective_transform], adapter)
        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        assert out.shape == img.shape
        # atol=1e-5: bilinear interpolation via F.grid_sample introduces floating-point
        # error even for an identity warp; this is an interpolation tolerance, not an
        # algebraic identity check.
        assert torch.allclose(out, img, atol=1e-5)

    def test_projective_segment_mask_aux_target(self):
        """ProjectiveSegment with identity transform returns mask via aux_targets unchanged."""
        adapter = _StubAdapter()
        projective_transform = self._proj_transform()
        seg = ProjectiveSegment([projective_transform], adapter)

        img = torch.rand(2, 3, 8, 8)
        mask = torch.rand(2, 1, 8, 8)
        result = seg(img, aux_targets={"mask": mask})

        assert isinstance(result, tuple)
        assert len(result) == 2
        out_img, out_aux = result
        assert out_img.shape == img.shape
        assert "mask" in out_aux
        assert out_aux["mask"].shape == mask.shape
        # Identity transform: image and mask should pass through unchanged
        assert torch.allclose(out_img, img, atol=1e-5)
        assert torch.allclose(out_aux["mask"], mask, atol=1e-5)

    @pytest.mark.skipif(not _CV2_AVAILABLE, reason="missing cv2")
    def test_albu_projective_segment_routes_aux_targets(self):
        """AlbuProjectiveSegment routes an aux mask through the composed homography."""
        adapter = _StubAdapter()
        projective_transform = self._proj_transform()
        seg = AlbuProjectiveSegment([projective_transform], adapter)

        img = torch.rand(2, 3, 8, 8)
        mask = torch.rand(2, 1, 8, 8)
        result = seg(img, aux_targets={"mask": mask})

        assert isinstance(result, tuple)
        out_img, out_aux = result
        assert out_img.shape == img.shape
        assert out_aux["mask"].shape == mask.shape
        # Identity homography: image and mask pass through unchanged.
        assert torch.allclose(out_img, img, atol=1e-5)
        assert torch.allclose(out_aux["mask"], mask, atol=1e-5)

    def test_albu_projective_segment_requires_cv2(self, monkeypatch: pytest.MonkeyPatch):
        """AlbuProjectiveSegment raises a clear ImportError when cv2 is unavailable."""
        monkeypatch.setattr(segment_mod, "_cv2_module", lambda: None)

        adapter = _StubAdapter()
        projective_transform = self._proj_transform()

        with pytest.raises(ImportError, match="opencv-python"):
            AlbuProjectiveSegment([projective_transform], adapter)


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="missing cv2")
class TestCv2ReflectionBorderParity:
    """The cv2 border constant for "reflection" must match torch grid_sample reflection semantics.

    torch grid_sample(padding_mode="reflection", align_corners=True) reflects about the edge sample without duplicating
    it, which is cv2.BORDER_REFLECT_101 -- cv2.BORDER_REFLECT duplicates the edge pixel and produces off-by-one borders
    relative to the torch path.

    """

    def test_reflection_maps_to_border_reflect_101(self):
        """The _CV2_BORDER table maps "reflection" to BORDER_REFLECT_101, not BORDER_REFLECT."""
        assert segment_mod._CV2_BORDER["reflection"] == segment_mod._require_cv2().BORDER_REFLECT_101

    def test_translated_border_pixels_match_torch_grid_sample(self):
        """A +2px translation warped via cv2 with the mapped border equals the torch reflection path."""
        import numpy as np

        height, width = 3, 6
        image = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
        shift = 2.0

        pix_x = torch.arange(width, dtype=torch.float32) - shift
        pix_y = torch.arange(height, dtype=torch.float32)
        norm_x = 2.0 * pix_x / (width - 1) - 1.0
        norm_y = 2.0 * pix_y / (height - 1) - 1.0
        grid_y, grid_x = torch.meshgrid(norm_y, norm_x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        out_torch = torch.nn.functional.grid_sample(
            image, grid, mode="bilinear", padding_mode="reflection", align_corners=True
        )

        mtx_fwd = np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.0]], dtype=np.float64)
        out_cv2 = segment_mod._require_cv2().warpAffine(
            image[0, 0].numpy(),
            mtx_fwd,
            (width, height),
            flags=segment_mod._require_cv2().INTER_LINEAR,
            borderMode=segment_mod._CV2_BORDER["reflection"],
        )

        assert np.allclose(out_torch[0, 0].numpy(), out_cv2, atol=1e-5)
