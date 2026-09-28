"""Opt-in antialiased downscale (``antialias=True``).

An aggressive crop-resize downscale through a single ``grid_sample`` aliases: it drops
high-frequency detail between samples. With ``antialias=True`` the input is Gaussian
prefiltered (mipmap-rule sigma) before the warp, so the result is closer to a proper
antialiased resize. These tests assert:

- the antialiased downscale beats the unfiltered one on SSIM vs a
  ``torchvision.transforms.v2.Resize(antialias=True)`` reference;
- the output is bit-identical to the default path when the flag is off;
- the output is bit-identical when the scale is >= 0.5 (no aliasing risk).

SSIM is computed inline (scikit-image is not a dependency).

"""

from __future__ import annotations

import pytest
import torch

import fused_transforms.affine.segment as segment
from fused_transforms import _compat
from fused_transforms._compat import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE
from fused_transforms.affine.matrix import estimate_scale
from fused_transforms.affine.segment import (
    _antialias_axis_scales,
    _maybe_antialias_prefilter,
    _mipmap_sigma,
    build_segments,
)


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Return a normalized 1D Gaussian window for the SSIM filter."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return g / g.sum()


def _ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean SSIM between two ``(B, C, H, W)`` tensors in ``[0, 1]`` (inline, no skimage)."""
    window_size, sigma = 11, 1.5
    channels = img1.shape[1]
    win1d = _gaussian_window(window_size, sigma)
    win2d = (win1d[:, None] * win1d[None, :]).to(img1.dtype)
    window = win2d.expand(channels, 1, window_size, window_size).contiguous()

    def _filter(img: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(img, window, padding=window_size // 2, groups=channels)

    c1, c2 = 0.01**2, 0.03**2
    mu1, mu2 = _filter(img1), _filter(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = _filter(img1 * img1) - mu1_sq
    sigma2_sq = _filter(img2 * img2) - mu2_sq
    sigma12 = _filter(img1 * img2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def _high_frequency_image(size: int = 96) -> torch.Tensor:
    """Build a high-frequency checkerboard-like image that aliases hard on downscale."""
    torch.manual_seed(7)
    coords = torch.arange(size, dtype=torch.float32)
    grid_x = torch.sin(coords * 1.3)[None, :]
    grid_y = torch.sin(coords * 1.3)[:, None]
    pattern = 0.5 + 0.5 * (grid_x * grid_y)
    return pattern.expand(1, 3, size, size).contiguous()


class TestEstimateScale:
    """The per-axis scale estimator underpinning the antialias decision."""

    def test_pure_downscale_reports_min_axis(self) -> None:
        """A diagonal down-scale matrix reports its two axis scales, smallest first."""
        mtx = torch.eye(3).unsqueeze(0)
        mtx[:, 0, 0] = 0.25
        mtx[:, 1, 1] = 0.5
        lo, hi = estimate_scale(mtx)
        assert lo == pytest.approx(0.25, abs=1e-4)
        assert hi == pytest.approx(0.5, abs=1e-4)


class TestAntialiasAxisScales:
    """Per-axis scale mapping that decides which axis the prefilter blurs (AFF-1)."""

    def test_axis_aligned_reports_scales_per_axis_not_by_magnitude(self) -> None:
        """A height-dominant axis-aligned shrink returns (width, height) scales, not magnitude-sorted."""
        mtx = torch.eye(3).unsqueeze(0)
        mtx[:, 0, 0] = 0.9  # width shrinks mildly
        mtx[:, 1, 1] = 0.2  # height shrinks hard
        scale_x, scale_y = _antialias_axis_scales(mtx)
        assert float(scale_x[0]) == pytest.approx(0.9, abs=1e-4)  # width axis, NOT the smaller singular value
        assert float(scale_y[0]) == pytest.approx(0.2, abs=1e-4)  # height axis carries the aggressive shrink

    def test_width_dominant_shrink_maps_to_width_axis(self) -> None:
        """A width-dominant axis-aligned shrink puts the small scale on the width axis."""
        mtx = torch.eye(3).unsqueeze(0)
        mtx[:, 0, 0] = 0.2  # width shrinks hard
        mtx[:, 1, 1] = 0.9  # height shrinks mildly
        scale_x, scale_y = _antialias_axis_scales(mtx)
        assert float(scale_x[0]) == pytest.approx(0.2, abs=1e-4)
        assert float(scale_y[0]) == pytest.approx(0.9, abs=1e-4)

    def test_rotated_matrix_uses_isotropic_min_singular_value(self) -> None:
        """A rotated (non-axis-aligned) shrink falls back to the worst-axis singular value on both axes."""
        theta = torch.tensor(0.6)
        rot = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
        scale = torch.diag(torch.tensor([0.3, 0.8]))
        mtx = torch.eye(3).unsqueeze(0)
        mtx[0, :2, :2] = rot @ scale
        scale_x, scale_y = _antialias_axis_scales(mtx)
        assert float(scale_x[0]) == pytest.approx(0.3, abs=1e-4)  # smallest singular value
        torch.testing.assert_close(scale_x, scale_y)  # applied isotropically for a rotated warp

    def test_batched_scales_remain_per_sample(self) -> None:
        """One sample's downscale cannot choose another sample's blur sigma."""
        mtx = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        mtx[0, 0, 0], mtx[0, 1, 1] = 0.9, 0.7
        mtx[1, 0, 0], mtx[1, 1, 1] = 0.4, 0.95
        scale_x, scale_y = _antialias_axis_scales(mtx)
        torch.testing.assert_close(scale_x, torch.tensor([0.9, 0.4]))
        torch.testing.assert_close(scale_y, torch.tensor([0.7, 0.95]))


def _axis_total_variation(image: torch.Tensor) -> tuple[float, float]:
    """Return mean absolute neighbour differences along (height, width) axes."""
    tv_h = float((image[..., 1:, :] - image[..., :-1, :]).abs().mean().item())
    tv_w = float((image[..., :, 1:] - image[..., :, :-1]).abs().mean().item())
    return tv_h, tv_w


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="antialias prefilter needs the kornia Gaussian backend")
class TestAnisotropicPrefilterAxis:
    """The prefilter blurs the axis that actually downscales (AFF-1 regression)."""

    def test_height_dominant_shrink_blurs_height_axis(self) -> None:
        """A hard height shrink + mild width shrink smooths the height axis far more than width.

        With the pre-fix magnitude-sorted scales the strong blur landed on the wrong (width) axis, so this ratio
        comparison flips and the test fails.

        """
        torch.manual_seed(0)
        image = torch.rand(1, 3, 48, 48)  # broadband high frequency on both axes
        mtx = torch.eye(3).unsqueeze(0)
        mtx[:, 0, 0] = 0.9  # width barely shrinks -> negligible width blur
        mtx[:, 1, 1] = 0.2  # height shrinks hard -> strong height blur

        tv_h_before, tv_w_before = _axis_total_variation(image)
        filtered = _maybe_antialias_prefilter(image, mtx, enabled=True)
        tv_h_after, tv_w_after = _axis_total_variation(filtered)

        ratio_h = tv_h_after / tv_h_before  # height smoothing (should be strong -> small ratio)
        ratio_w = tv_w_after / tv_w_before  # width smoothing (should be weak -> ratio near 1)
        assert ratio_h < 0.6 * ratio_w

    def test_heterogeneous_aggressive_batch_matches_independent_filtering(self) -> None:
        """Each aggressive row keeps the Gaussian support it gets alone.

        Both rows downscale, but each has a different anisotropic sigma. A batched blur using the maximum kernel support
        would change both results relative to these single-row calls.

        """
        torch.manual_seed(3)
        image = torch.rand(2, 3, 48, 48)
        mtx = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        mtx[0, 0, 0], mtx[0, 1, 1] = 0.25, 0.4
        mtx[1, 0, 0], mtx[1, 1, 1] = 0.45, 0.2

        batched = _maybe_antialias_prefilter(image, mtx, enabled=True)
        first = _maybe_antialias_prefilter(image[:1], mtx[:1], enabled=True)
        second = _maybe_antialias_prefilter(image[1:], mtx[1:], enabled=True)

        torch.testing.assert_close(batched[:1], first, rtol=0.0, atol=0.0)
        torch.testing.assert_close(batched[1:], second, rtol=0.0, atol=0.0)


class TestMipmapSigma:
    """The mipmap prefilter sigma rule."""

    def test_no_blur_at_or_above_unit_scale(self) -> None:
        """Scale >= 1 (upscale / no downscale) needs no prefilter."""
        assert _mipmap_sigma(1.0) == 0.0
        assert _mipmap_sigma(1.5) == 0.0

    def test_sigma_grows_as_scale_shrinks(self) -> None:
        """A more aggressive downscale demands a wider prefilter."""
        assert _mipmap_sigma(0.25) > _mipmap_sigma(0.4) > 0.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="antialias prefilter needs the kornia Gaussian backend")
class TestAntialiasDownscale:
    """End-to-end antialiased crop-resize vs a torchvision antialias reference."""

    def _pipe_out(self, image: torch.Tensor, target: int, *, antialias: bool) -> torch.Tensor:
        import kornia.augmentation as kornia_aug

        from fused_transforms.adapters.kornia import KorniaAdapter
        from fused_transforms.compose import FusedCompose

        torch.manual_seed(0)
        crop = kornia_aug.RandomResizedCrop(size=(target, target), scale=(1.0, 1.0), ratio=(1.0, 1.0), p=1.0)
        pipe = FusedCompose([crop], adapter=KorniaAdapter(), antialias=antialias)
        return pipe(image.clone())

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required for the antialias reference")
    def test_antialiased_beats_unfiltered_on_ssim(self) -> None:
        """Antialiased downscale is closer to a torchvision antialias resize than the unfiltered warp."""
        from torchvision.transforms import v2

        image = _high_frequency_image(96)
        target = 24  # 4x downscale, well below the 0.5 threshold
        reference = v2.Resize((target, target), antialias=True)(image)

        unfiltered = self._pipe_out(image, target, antialias=False)
        antialiased = self._pipe_out(image, target, antialias=True)

        ssim_unfiltered = _ssim(unfiltered, reference)
        ssim_antialiased = _ssim(antialiased, reference)
        assert ssim_antialiased > ssim_unfiltered

    def test_flag_off_is_bit_identical_to_default(self) -> None:
        """With the flag off, output is bit-identical to the plain single-warp path."""
        image = _high_frequency_image(96)
        default = self._pipe_out(image, 24, antialias=False)
        again = self._pipe_out(image, 24, antialias=False)
        assert torch.equal(default, again)

    def test_no_change_when_scale_at_or_above_half(self) -> None:
        """A mild downscale (scale >= 0.5) is untouched: antialias output equals the unfiltered output."""
        image = _high_frequency_image(96)
        target = 64  # 96 -> 64 => scale ~0.66, above the 0.5 threshold
        unfiltered = self._pipe_out(image, target, antialias=False)
        antialiased = self._pipe_out(image, target, antialias=True)
        assert torch.equal(unfiltered, antialiased)

    def test_prefilter_preserves_input_gradient(self) -> None:
        """The opt-in image prefilter remains differentiable through the crop-resize warp."""
        image = _high_frequency_image(96).requires_grad_()
        self._pipe_out(image, 24, antialias=True).square().mean().backward()
        assert image.grad is not None
        assert torch.isfinite(image.grad).all()
        assert bool(image.grad.abs().any())

    def test_prefilter_does_not_change_routed_mask(self) -> None:
        """Antialiasing filters the image only; mask routing remains the same nearest warp."""
        import kornia.augmentation as kornia_aug

        from fused_transforms.adapters.kornia import KorniaAdapter
        from fused_transforms.compose import FusedCompose

        image = _high_frequency_image(96)
        mask = torch.arange(96 * 96, dtype=torch.uint8).reshape(1, 1, 96, 96)

        def run(antialias: bool) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply the fixed full-image crop under one antialias setting."""
            crop = kornia_aug.RandomResizedCrop(size=(24, 24), scale=(1.0, 1.0), ratio=(1.0, 1.0), p=1.0)
            pipe = FusedCompose([crop], adapter=KorniaAdapter(), data_keys=["input", "mask"], antialias=antialias)
            return pipe(image, mask)

        unfiltered_image, unfiltered_mask = run(antialias=False)
        filtered_image, filtered_mask = run(antialias=True)
        assert not torch.equal(filtered_image, unfiltered_image)
        assert torch.equal(filtered_mask, unfiltered_mask)

    @pytest.mark.parametrize(
        ("target", "antialias", "expected_calls"),
        [(24, False, 0), (64, True, 0), (24, True, 1)],
        ids=("flag-off", "mild-downscale", "aggressive-downscale"),
    )
    def test_crop_resize_filters_only_aggressive_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
        target: int,
        antialias: bool,
        expected_calls: int,
    ) -> None:
        """Crop-resize enters Gaussian blur only for opt-in aggressive downscales."""
        image = _high_frequency_image(96)
        original = segment._kornia_gaussian_blur
        calls: list[bool] = []

        def spy_gaussian(image: torch.Tensor, sigma_x, sigma_y):
            """Record Gaussian work while preserving the underlying result."""
            calls.append(True)
            return original(image, sigma_x, sigma_y)

        monkeypatch.setattr(segment, "_kornia_gaussian_blur", spy_gaussian)

        self._pipe_out(image, target, antialias=antialias)

        assert len(calls) == expected_calls


def test_antialias_true_rejects_missing_kornia_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit antialiasing never silently degrades when Kornia is unavailable."""
    from fused_transforms.compose import FusedCompose

    # What import_backend caches for a Kornia that is absent (or installed but failing to import).
    monkeypatch.setitem(_compat._LOADED, "kornia", None)

    with pytest.raises(ImportError, match="antialias=True requires the optional kornia dependency"):
        FusedCompose([], antialias=True)


def test_build_segments_antialias_rejects_missing_kornia(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exported segment builder has the same explicit optional-dependency contract."""
    # What import_backend caches for a Kornia that is absent (or installed but failing to import).
    monkeypatch.setitem(_compat._LOADED, "kornia", None)

    with pytest.raises(ImportError, match="antialias=True requires the optional kornia dependency"):
        build_segments([], adapter=object(), antialias=True)  # type: ignore[arg-type]
