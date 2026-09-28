"""Tests for CROP_RESIZE_FIXED category: adapters, CropResizeSegment, and build_segments routing.

Covers:
- TransformCategory.CROP_RESIZE_FIXED exists
- category() returns CROP_RESIZE_FIXED for RandomResizedCrop in all three adapters
- crop_resize_matrix: identity case and basic scale
- normalize_matrix_io: same-size case equals normalize_matrix; crop-to-half case
- build_segments produces CropResizeSegment for the torch path
- build_segments produces a passthrough for the albumentations numpy path
- CropResizeSegment forward: output shape matches target size
- CropResizeSegment forward: aux_targets passthrough

Run to verify behaviour and guard against regressions:
    pytest tests/test_unit/test_crop_resize_fusion.py -v

"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fused_transforms._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE
from fused_transforms.affine.matrix import crop_resize_matrix, inv3x3, normalize_matrix, normalize_matrix_io
from fused_transforms.affine.segment import CropResizeSegment, _FusedGeoCropSegment, build_segments
from fused_transforms.compose import FusedCompose
from fused_transforms.types import TransformCategory

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fused_transforms.adapters.kornia import KorniaAdapter

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as tv_trans

    from fused_transforms.adapters.torchvision import TorchVisionAdapter

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

    from fused_transforms.adapters.albumentations import AlbumentationsAdapter


def test_crop_resize_fixed_category_exists():
    """TransformCategory.CROP_RESIZE_FIXED is present in the enum."""
    assert hasattr(TransformCategory, "CROP_RESIZE_FIXED")
    assert TransformCategory.CROP_RESIZE_FIXED.value == "crop_resize_fixed"


class TestCropResizeMatrix:
    """crop_resize_matrix: pixel-space forward affine for crop+resize."""

    def test_identity_full_image(self):
        """Crop covering the full image with the same target size gives identity."""
        height, width = 32, 32
        top = torch.zeros(1)
        left = torch.zeros(1)
        crop_h = torch.full((1,), float(height))
        crop_w = torch.full((1,), float(width))
        target_h = torch.full((1,), float(height))
        target_w = torch.full((1,), float(width))

        mtx = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        assert mtx.shape == (1, 3, 3)
        assert torch.allclose(mtx, torch.eye(3).unsqueeze(0))

    def test_scale_halves_output(self):
        """Crop full image but resize to half: scale factor reflects align_corners endpoint mapping.

        With align_corners=True, the discrete scale factor is (W_out - 1) / (W_in - 1)
        rather than the naive W_out / W_in. For a 64->32 downscale this yields 31/63,
        not 0.5 — this test pins that convention.

        """
        height, width = 64, 64
        top = torch.zeros(1)
        left = torch.zeros(1)
        crop_h = torch.full((1,), float(height))
        crop_w = torch.full((1,), float(width))
        target_h = torch.full((1,), float(height) / 2)
        target_w = torch.full((1,), float(width) / 2)

        mtx = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        # align_corners=True endpoint mapping:
        # sx = (W_out-1)/(W_in-1) = 31/63, sy = 31/63
        scale = 31.0 / 63.0
        expected = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        assert torch.allclose(mtx, expected)

    def test_top_left_offset_translates(self):
        """Crop starting at (top=8, left=8) with same-size output: translation only."""
        top = torch.tensor([8.0])
        left = torch.tensor([8.0])
        crop_h = torch.tensor([32.0])
        crop_w = torch.tensor([32.0])
        target_h = torch.tensor([32.0])
        target_w = torch.tensor([32.0])

        mtx = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        # sx = 1, tx = -8; sy = 1, ty = -8
        expected = torch.tensor([[1.0, 0.0, -8.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        assert torch.allclose(mtx, expected)

    def test_batch_of_two(self):
        """crop_resize_matrix accepts (B,) tensors and returns (B, 3, 3)."""
        batch_size = 2
        top = torch.zeros(batch_size)
        left = torch.zeros(batch_size)
        crop_h = torch.full((batch_size,), 32.0)
        crop_w = torch.full((batch_size,), 32.0)
        target_h = torch.full((batch_size,), 32.0)
        target_w = torch.full((batch_size,), 32.0)

        mtx = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        assert mtx.shape == (batch_size, 3, 3)

    def test_rejects_degenerate_crop_or_target_sizes(self):
        """crop_resize_matrix rejects sizes <= 1 (singular endpoint mapping)."""
        with pytest.raises(ValueError, match="requires crop and target sizes > 1"):
            crop_resize_matrix(
                top=torch.tensor([0.0]),
                left=torch.tensor([0.0]),
                crop_h=torch.tensor([1.0]),
                crop_w=torch.tensor([16.0]),
                target_h=torch.tensor([8.0]),
                target_w=torch.tensor([8.0]),
            )


class TestNormalizeMatrixIO:
    """normalize_matrix_io: same-size case matches normalize_matrix; IO variant."""

    def test_same_size_matches_normalize_matrix(self):
        """When H_in==H_out and W_in==W_out, normalize_matrix_io == normalize_matrix."""
        height, width = 64, 64
        mtx = torch.eye(3).unsqueeze(0)
        normalized_mtx = normalize_matrix(mtx, height=height, width=width)
        nmio = normalize_matrix_io(mtx, height_in=height, width_in=width, height_out=height, width_out=width)
        assert torch.allclose(normalized_mtx, nmio, atol=1e-6)

    def test_crop_to_half_scale_factor(self):
        """normalize_matrix_io for a 2x downscale inverse maps correctly."""
        h_in, w_in = 64, 64
        h_out, w_out = 32, 32

        top = torch.zeros(1)
        left = torch.zeros(1)
        mtx_fwd = crop_resize_matrix(
            top,
            left,
            torch.full((1,), float(h_in)),
            torch.full((1,), float(w_in)),
            torch.full((1,), float(h_out)),
            torch.full((1,), float(w_out)),
        )
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix_io(mtx_inv, height_in=h_in, width_in=w_in, height_out=h_out, width_out=w_out)
        assert mtx_norm.shape == (1, 3, 3)
        assert torch.all(torch.isfinite(mtx_norm))

    def test_crop_resize_matches_explicit_crop_then_interpolate(self):
        """normalize_matrix_io + grid_sample matches explicit crop + interpolate within float tolerance.

        This is the core numerical-equivalence guarantee for the fused crop-resize
        path: replacing a two-step (slice then bilinear resize) operation with a
        single affine-grid sample must produce pixel-identical output up to
        floating-point round-off, otherwise the fusion is silently lossy.

        """
        h_in, w_in = 64, 80
        h_out, w_out = 32, 32
        top, left = 7, 11
        crop_h, crop_w = 40, 50

        image = torch.linspace(0.0, 1.0, h_in * w_in).reshape(1, 1, h_in, w_in).repeat(1, 3, 1, 1)
        ref = F.interpolate(
            image[:, :, top : top + crop_h, left : left + crop_w],
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=True,
        )

        mtx_fwd = crop_resize_matrix(
            top=torch.tensor([float(top)]),
            left=torch.tensor([float(left)]),
            crop_h=torch.tensor([float(crop_h)]),
            crop_w=torch.tensor([float(crop_w)]),
            target_h=torch.tensor([float(h_out)]),
            target_w=torch.tensor([float(w_out)]),
        )
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix_io(mtx_inv, height_in=h_in, width_in=w_in, height_out=h_out, width_out=w_out)
        grid = F.affine_grid(mtx_norm[:, :2, :], [1, 3, h_out, w_out], align_corners=True)
        out = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestKorniaAdapterCategory:
    """KorniaAdapter.category() returns CROP_RESIZE_FIXED for RandomResizedCrop."""

    def test_random_resized_crop_category(self):
        """RandomResizedCrop maps to CROP_RESIZE_FIXED."""
        transform = kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        assert KorniaAdapter.category(transform) == TransformCategory.CROP_RESIZE_FIXED


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestTorchVisionAdapterCategory:
    """TorchVisionAdapter.category() returns CROP_RESIZE_FIXED for RandomResizedCrop."""

    def test_v1_random_resized_crop_category(self):
        """V1 RandomResizedCrop maps to CROP_RESIZE_FIXED."""
        transform = tv_trans.RandomResizedCrop(size=(64, 64))
        assert TorchVisionAdapter.category(transform) == TransformCategory.CROP_RESIZE_FIXED

    def test_v2_random_resized_crop_category(self):
        """V2 RandomResizedCrop maps to CROP_RESIZE_FIXED."""
        transform = tv_trans.v2.RandomResizedCrop(size=(64, 64))
        assert TorchVisionAdapter.category(transform) == TransformCategory.CROP_RESIZE_FIXED


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAlbumentationsAdapterCategory:
    """AlbumentationsAdapter.category() returns CROP_RESIZE_FIXED for RandomResizedCrop."""

    def test_random_resized_crop_category(self):
        """RandomResizedCrop maps to CROP_RESIZE_FIXED."""
        transform = albu.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        assert AlbumentationsAdapter.category(transform) == TransformCategory.CROP_RESIZE_FIXED


class TestBuildSegmentsCropResize:
    """build_segments emits CropResizeSegment (torch) or passthrough (numpy)."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_torch_path_produces_crop_resize_segment(self):
        """Kornia RandomResizedCrop → CropResizeSegment on torch path."""
        transform = kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        segments = build_segments([transform], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_torchvision_v1_torch_path_produces_crop_resize_segment(self):
        """TorchVision v1 RandomResizedCrop → CropResizeSegment on torch path."""
        transform = tv_trans.RandomResizedCrop(size=(64, 64))
        adapter = TorchVisionAdapter()
        segments = build_segments([transform], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_torchvision_v2_torch_path_produces_crop_resize_segment(self):
        """TorchVision v2 RandomResizedCrop → CropResizeSegment on torch path."""
        transform = tv_trans.v2.RandomResizedCrop(size=(64, 64))
        adapter = TorchVisionAdapter()
        segments = build_segments([transform], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)

    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
    def test_albumentations_numpy_path_produces_passthrough(self):
        """Albumentations path emits the raw transform (not CropResizeSegment)."""
        transform = albu.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = AlbumentationsAdapter()
        segments = build_segments([transform], adapter, use_numpy=True)
        assert len(segments) == 1
        assert not isinstance(segments[0], CropResizeSegment)

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_crop_resize_fuses_preceding_geo(self):
        """A preceding GEOMETRIC_INTERP run fuses with the crop into one warp.

        On the torch path a crop-resize immediately after a fusible geometric run composes ``M_crop @ M_geo`` into a
        single ``_FusedGeoCropSegment`` — one ``grid_sample`` at the crop's target size instead of a fused-affine warp
        followed by a separate crop warp.

        """
        rotate_transform = kornia_aug.RandomRotation(degrees=30.0, p=1.0)
        crop_transform = kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        segments = build_segments([rotate_transform, crop_transform], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], _FusedGeoCropSegment)
        assert not isinstance(segments[0], CropResizeSegment)

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_crop_resize_without_preceding_geo_stays_standalone(self):
        """A crop-resize with no pending geometric run stays a standalone CropResizeSegment."""
        crop_transform = kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        segments = build_segments([crop_transform], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)
        assert not isinstance(segments[0], _FusedGeoCropSegment)


class TestCropResizeSegmentForward:
    """CropResizeSegment produces correct output shape."""

    @pytest.fixture
    def kornia_crop_segment(self) -> CropResizeSegment:
        """Return a CropResizeSegment backed by Kornia RandomResizedCrop."""
        pytest.importorskip("kornia", exc_type=ModuleNotFoundError)
        transform = kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        return CropResizeSegment(transform, adapter)

    @pytest.fixture
    def tv_v1_crop_segment(self) -> CropResizeSegment:
        """Return a CropResizeSegment backed by TorchVision v1 RandomResizedCrop."""
        pytest.importorskip("torchvision", exc_type=ModuleNotFoundError)
        transform = tv_trans.RandomResizedCrop(size=(64, 64))
        adapter = TorchVisionAdapter()
        return CropResizeSegment(transform, adapter)

    def test_kornia_output_shape(self, kornia_crop_segment: CropResizeSegment):
        """Kornia CropResizeSegment outputs (B, C, target_H, target_W)."""
        image = torch.rand(2, 3, 128, 128)
        out = kornia_crop_segment(image)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_tv_v1_output_shape(self, tv_v1_crop_segment: CropResizeSegment):
        """TorchVision v1 CropResizeSegment outputs (B, C, target_H, target_W)."""
        image = torch.rand(2, 3, 128, 128)
        out = tv_v1_crop_segment(image)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_output_values_bounded(self, kornia_crop_segment: CropResizeSegment):
        """Output values stay within input range (bilinear interpolation, no padding)."""
        image = torch.rand(1, 3, 128, 128)
        out = kornia_crop_segment(image)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_aux_targets_passthrough(self, kornia_crop_segment: CropResizeSegment):
        """With aux_targets=None returns a tensor; with dict returns (tensor, dict)."""
        image = torch.rand(1, 3, 128, 128)

        out = kornia_crop_segment(image, aux_targets=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([1, 3, 64, 64])

        aux = {"mask": torch.zeros(1, 1, 128, 128)}
        out2, aux_out = kornia_crop_segment(image, aux_targets=aux)
        assert isinstance(out2, torch.Tensor)
        assert aux_out is aux

    def test_batch_size_one(self, kornia_crop_segment: CropResizeSegment):
        """Works correctly with batch size 1."""
        image = torch.rand(1, 3, 128, 128)
        out = kornia_crop_segment(image)
        assert out.shape == torch.Size([1, 3, 64, 64])

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_p_zero_is_ignored(self):
        """CropResizeSegment always applies regardless of p; p=0.0 still crops.

        Unlike random geometric ops where p=0 yields a no-op, RandomResizedCrop changes output resolution. Skipping it
        would leave a shape mismatch in the pipeline, so the segment ignores p and always applies.

        """
        transform = kornia_aug.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0), p=0.0)
        seg = CropResizeSegment(transform, KorniaAdapter())
        image = torch.rand(2, 3, 64, 64)
        out = seg(image)
        assert out.shape == torch.Size([2, 3, 32, 32])


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFusionPlanDescriptors:
    """fusion_plan_descriptors includes crop_resize kind."""

    def test_crop_resize_descriptor_kind(self):
        """Single RandomResizedCrop produces one crop_resize descriptor."""
        transform = kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        pipe = FusedCompose([transform], adapter=KorniaAdapter())
        descs = pipe.fusion_plan_descriptors
        assert len(descs) == 1
        assert descs[0].kind == "crop_resize"
        assert descs[0].n_warps_saved == 0

    def test_full_pipeline_descriptor_kinds(self):
        """Rotation → CropResize → HFlip gives [fused, exact].

        The rotation fuses with the crop-resize into one ``fused`` segment, so no standalone ``crop_resize`` descriptor
        remains; the trailing flip is a separate ``exact`` segment.

        """
        transforms = [
            kornia_aug.RandomRotation(degrees=30.0, p=1.0),
            kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)),
            kornia_aug.RandomHorizontalFlip(p=1.0),
        ]
        pipe = FusedCompose(transforms, adapter=KorniaAdapter())
        kinds = [desc.kind for desc in pipe.fusion_plan_descriptors]
        assert kinds == ["fused", "exact"]

    def test_fused_geo_crop_descriptor_counts_crop(self):
        """[rotate, RandomResizedCrop] fuses to one ``fused`` descriptor saving 1 warp."""
        transforms = [
            kornia_aug.RandomRotation(degrees=30.0, p=1.0),
            kornia_aug.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)),
        ]
        pipe = FusedCompose(transforms, adapter=KorniaAdapter())
        descs = pipe.fusion_plan_descriptors
        assert len(descs) == 1
        assert descs[0].kind == "fused"
        assert descs[0].n_warps_saved == 1
        assert pipe.n_warps_saved == 1


class TestFusedComposeWithCropResize:
    """FusedCompose.forward with a CropResizeSegment changes output shape."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_compose_output_shape(self):
        """Kornia pipeline with RandomResizedCrop outputs target size."""
        transform = kornia_aug.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0))
        pipe = FusedCompose([transform], adapter=KorniaAdapter())
        image = torch.rand(2, 3, 64, 64)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_tv_v1_compose_output_shape(self):
        """TorchVision v1 pipeline with RandomResizedCrop outputs target size."""
        transform = tv_trans.RandomResizedCrop(size=(32, 32))
        pipe = FusedCompose([transform], adapter=TorchVisionAdapter())
        image = torch.rand(2, 3, 64, 64)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_rotation_then_crop_resize_shape(self):
        """Rotation → CropResize pipeline outputs the crop-resize target size."""
        transforms = [
            kornia_aug.RandomRotation(degrees=15.0, p=1.0),
            kornia_aug.RandomResizedCrop(size=(48, 48), scale=(0.5, 1.0)),
        ]
        pipe = FusedCompose(transforms, adapter=KorniaAdapter())
        image = torch.rand(2, 3, 96, 96)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 48, 48])


class _FixedCropAdapter:
    """Stub adapter returning deterministic crop params for coordinate-level testing.

    Every call to sample_params returns the same fixed crop region regardless of input shape or transform. build_matrix
    delegates to crop_resize_matrix.

    """

    def __init__(
        self,
        top: float,
        left: float,
        crop_h: float,
        crop_w: float,
        target_h: float,
        target_w: float,
    ) -> None:
        """Store fixed crop coordinates and output size."""
        self._top = top
        self._left = left
        self._crop_h = crop_h
        self._crop_w = crop_w
        self._target_h = target_h
        self._target_w = target_w

    def sample_params(
        self,
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return fixed crop parameter tensors broadcast to batch_size."""
        batch_size = input_shape[0]
        return {
            "crop_top": torch.full((batch_size,), self._top, dtype=torch.float32, device=device),
            "crop_left": torch.full((batch_size,), self._left, dtype=torch.float32, device=device),
            "crop_h": torch.full((batch_size,), self._crop_h, dtype=torch.float32, device=device),
            "crop_w": torch.full((batch_size,), self._crop_w, dtype=torch.float32, device=device),
            "target_h": torch.full((batch_size,), self._target_h, dtype=torch.float32, device=device),
            "target_w": torch.full((batch_size,), self._target_w, dtype=torch.float32, device=device),
        }

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build forward affine matrix from fixed crop params via crop_resize_matrix."""
        return crop_resize_matrix(
            top=params["crop_top"],
            left=params["crop_left"],
            crop_h=params["crop_h"],
            crop_w=params["crop_w"],
            target_h=params["target_h"],
            target_w=params["target_w"],
        )


class TestCropResizeSegmentDeterministicCoords:
    """CropResizeSegment warps aux targets to mathematically expected pixel positions.

    Geometry: 16x16 input, crop at top=4, left=4, crop_h=8, crop_w=8, target=8x8.
    Forward matrix is a pure translation by (-4, -4) with scale_x = scale_y = 1.0
    because (target-1)/(crop-1) = 7/7 = 1.0.

    """

    @pytest.fixture
    def seg(self) -> CropResizeSegment:
        """Return CropResizeSegment backed by stub adapter with fixed crop params."""
        adapter = _FixedCropAdapter(top=4.0, left=4.0, crop_h=8.0, crop_w=8.0, target_h=8.0, target_w=8.0)
        return CropResizeSegment(transform=object(), adapter=adapter)

    def test_keypoint_maps_to_expected_output(self, seg: CropResizeSegment) -> None:
        """Keypoint at (8, 8) in 16x16 maps to (4, 4) after top=left=4 8x8->8x8 crop."""
        image = torch.rand(1, 3, 16, 16)
        keypoints = torch.tensor([[[8.0, 8.0]]])  # (B=1, N=1, 2) — [x, y]
        _, aux = seg(image, aux_targets={"keypoints": keypoints})
        out_kp = aux["keypoints"]
        assert out_kp.shape == (1, 1, 2)
        assert torch.allclose(out_kp[0, 0], torch.tensor([4.0, 4.0]), atol=1e-4)

    def test_bbox_xyxy_maps_to_expected_output(self, seg: CropResizeSegment) -> None:
        """Bbox [4,4,12,12] maps to [0,0,8,8] after top=left=4 8x8->8x8 crop."""
        image = torch.rand(1, 3, 16, 16)
        boxes = torch.tensor([[[4.0, 4.0, 12.0, 12.0]]])  # (B=1, N=1, 4)
        _, aux = seg(image, aux_targets={"bbox_xyxy": boxes})
        out_boxes = aux["bbox_xyxy"]
        assert out_boxes.shape == (1, 1, 4)
        assert torch.allclose(out_boxes[0, 0], torch.tensor([0.0, 0.0, 8.0, 8.0]), atol=1e-4)
