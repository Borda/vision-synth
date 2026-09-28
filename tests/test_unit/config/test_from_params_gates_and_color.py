"""Regression tests for backend-free ``from_params`` probability gates and colour ops.

* Per-op probabilities gate each op family independently and keep the canonical order
  rotation -> scale -> shear -> translate.
* Brightness and contrast change images of any channel count, not only RGB.
* ``brightness=`` and ``contrast=`` also accept an explicit ``(low, high)`` factor range.

"""

from __future__ import annotations

import pytest
import torch

from fused_transforms import Compose


def _layout(pipe: object) -> list[list[str]]:
    return [list(transform.param_specs) for transform in pipe.original_transforms]  # type: ignore[attr-defined]


class TestGeometricGates:
    """``rotation_p`` / ``scale_p`` gate op families independently, in canonical order."""

    def test_gated_families_keep_canonical_order(self) -> None:
        """A translate at probability one still runs after the gated rotation and scale."""
        pipe = Compose.from_params(
            rotation=(-30, 30), scale=(0.8, 1.2), translate_x=(-2.0, 2.0), rotation_p=0.5, scale_p=0.5
        )
        assert _layout(pipe) == [["rotation"], ["scale"], ["translate_x"]]

    def test_ungated_scale_merges_with_later_ungated_ops(self) -> None:
        """Adjacent families at probability one share one transform, after the gated rotation."""
        pipe = Compose.from_params(rotation=(-30, 30), scale=(0.8, 1.2), translate_x=(-2.0, 2.0), rotation_p=0.5)
        assert _layout(pipe) == [["rotation"], ["scale", "translate_x"]]

    def test_default_probabilities_keep_single_transform(self) -> None:
        """Both probabilities at one keep the historical single-transform layout (seeded output unchanged)."""
        pipe = Compose.from_params(rotation=(-30, 30), scale=(0.8, 1.2), translate_x=(-2.0, 2.0))
        assert _layout(pipe) == [["rotation", "scale", "translate_x"]]

    def test_rotation_and_scale_gates_are_independent(self) -> None:
        """Equal probabilities must not merge two families under one Bernoulli draw."""
        batch = 4000
        pipe = Compose.from_params(
            rotation=(10.0, 20.0),
            scale=(1.5, 2.0),
            rotation_p=0.5,
            scale_p=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        _, matrix = pipe(torch.rand(batch, 1, 8, 8), return_matrix=True)
        rotated = matrix[:, 0, 1].abs() > 1e-6
        scaled = (torch.linalg.det(matrix[:, :2, :2]) - 1.0).abs() > 1e-3
        assert abs(rotated.float().mean().item() - 0.5) < 0.05
        assert abs(scaled.float().mean().item() - 0.5) < 0.05
        assert abs((rotated & scaled).float().mean().item() - 0.25) < 0.05


class TestBackendFreeColorChannels:
    """Backend-free brightness/contrast are channel-count agnostic."""

    @pytest.mark.parametrize("channels", [1, 2, 4])
    @pytest.mark.parametrize("op", ["brightness", "contrast"])
    def test_non_rgb_matches_rgb_channel(self, channels: int, op: str) -> None:
        """Every channel of a non-RGB image changes exactly like one channel of the RGB image under the same seed."""
        generator = torch.Generator()
        pipe = Compose.from_params(**{op: 0.5}, generator=generator)
        base = torch.linspace(0.1, 0.9, 64).reshape(1, 1, 8, 8).expand(6, -1, -1, -1)
        generator.manual_seed(0)
        rgb = pipe(base.expand(-1, 3, -1, -1).contiguous())
        generator.manual_seed(0)
        other = pipe(base.expand(-1, channels, -1, -1).contiguous())
        assert other.shape == (6, channels, 8, 8)
        assert not torch.equal(other, base.expand(-1, channels, -1, -1)), "the op must not be a silent no-op"
        torch.testing.assert_close(other, rgb[:, :1].expand(-1, channels, -1, -1))


class TestColorRangeTuples:
    """``brightness`` / ``contrast`` accept a ``(low, high)`` factor range."""

    def test_tuple_equals_float_deviation(self) -> None:
        """``0.2`` and ``(0.8, 1.2)`` describe the same factor range and give identical seeded output."""
        image = torch.rand(4, 3, 8, 8, generator=torch.Generator().manual_seed(3))
        from_float = Compose.from_params(brightness=0.2, contrast=0.3, generator=torch.Generator().manual_seed(7))
        from_tuple = Compose.from_params(
            brightness=(0.8, 1.2), contrast=(0.7, 1.3), generator=torch.Generator().manual_seed(7)
        )
        torch.testing.assert_close(from_tuple(image), from_float(image))

    @pytest.mark.parametrize("value", [(1.2, 0.8), (-0.1, 1.0), (0.5,), (0.1, 0.2, 0.3)])
    def test_invalid_tuple_raises(self, value: tuple[float, ...]) -> None:
        """A reversed, negative or wrongly sized range is refused at construction."""
        with pytest.raises(ValueError, match="brightness"):
            Compose.from_params(brightness=value)  # type: ignore[arg-type]
