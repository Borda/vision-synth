"""Coverage for the TorchVision B=1 CPU-numpy combined-affine matrix builder.

``_affine_matrix_np_b1_tv`` (adapters/torchvision.py) is the sibling of the already-fixed
sin/cos typo in the rotation branch (``_ROTATION_TYPES_FS``), but implements the more complex
``RandomAffine`` composition with shear cross-terms -- and had zero test coverage. Every existing
``RandomAffine`` parity test wraps a single transform, so the multi-transform (``len(transforms) >
1``) B=1 cv2 fast path never fires at runtime either.

Two angles are covered:

- direct builder parity: ``_affine_matrix_np_b1_tv`` against the general torch-tensor
  ``_build_affine_matrix``, across combined param combos exercising both shear cross-terms;
- runtime dispatch: a real ``Compose(...)(image)`` forward that actually executes the function.
  ``sample_and_build_matrix_numpy_b1_tv`` resolves ``RandomAffine``/``RandomRotation`` inline and
  never returns ``None`` for them, so a plain ``Compose`` call never falls through to the two-step
  ``build_matrix_numpy_b1_tv`` -> ``_affine_matrix_np_b1_tv`` path under test. The fused sampler is
  disabled on the live segment to force that fallback.

"""

from __future__ import annotations

import math

import pytest
import torch

from fused_transforms._compat import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import RandomAffine, RandomRotation

    from fused_transforms import Compose, ReorderPolicy
    from fused_transforms.adapters import torchvision as _mod
    from fused_transforms.affine import segment as _segment_mod

pytestmark = pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")


def _smooth_image(height: int = 32, width: int = 48) -> torch.Tensor:
    """Smooth, non-square asymmetric image so center_x/center_y mixups are observable."""
    yy, xx = torch.meshgrid(torch.linspace(0, 3.14, height), torch.linspace(0, 3.14, width), indexing="ij")
    plane = 0.5 + 0.3 * torch.sin(3 * xx) * torch.cos(2 * yy) + 0.15 * xx / 3.14
    return plane.reshape(1, 1, height, width).expand(1, 3, height, width).contiguous()


def _interior(tensor: torch.Tensor, margin: int = 3) -> torch.Tensor:
    """Crop off the border where cv2.warpAffine and grid_sample legitimately differ."""
    return tensor[..., margin:-margin, margin:-margin]


def _affine_params(
    angle_deg: float,
    scale: float,
    shear_x_deg: float,
    shear_y_deg: float,
    translate_x: float,
    translate_y: float,
) -> dict[str, torch.Tensor]:
    """Build a canonical batch_size=1 param dict in the units the two builders share."""
    return {
        "angle_rad": torch.tensor([math.radians(angle_deg)], dtype=torch.float32),
        "scale": torch.tensor([scale], dtype=torch.float32),
        "shear_x_rad": torch.tensor([math.radians(shear_x_deg)], dtype=torch.float32),
        "shear_y_rad": torch.tensor([math.radians(shear_y_deg)], dtype=torch.float32),
        "translate_x": torch.tensor([translate_x], dtype=torch.float32),
        "translate_y": torch.tensor([translate_y], dtype=torch.float32),
    }


class TestAffineMatrixNpB1DirectParity:
    """``_affine_matrix_np_b1_tv`` against the general torch-tensor ``_build_affine_matrix``."""

    @pytest.mark.parametrize(
        ("angle_deg", "scale", "shear_x_deg", "shear_y_deg", "translate_x", "translate_y"),
        [
            pytest.param(15.0, 1.1, 0.0, 0.0, 3.0, -2.0, id="zero-shear"),
            pytest.param(-20.0, 0.9, 15.0, 0.0, 5.0, 4.0, id="x-shear-only"),
            pytest.param(25.0, 1.05, 10.0, -8.0, -4.0, 6.0, id="x-and-y-shear"),
            pytest.param(-33.0, 0.95, -12.0, 7.0, 2.0, -3.0, id="negative-angle-with-shear"),
        ],
    )
    def test_matches_torch_tensor_builder(
        self,
        angle_deg: float,
        scale: float,
        shear_x_deg: float,
        shear_y_deg: float,
        translate_x: float,
        translate_y: float,
    ) -> None:
        """The NumPy builder agrees with the torch-tensor builder for combined affine params."""
        params = _affine_params(angle_deg, scale, shear_x_deg, shear_y_deg, translate_x, translate_y)
        height, width = 32, 48
        center_x, center_y = (width - 1) * 0.5, (height - 1) * 0.5

        mtx_np = _mod._affine_matrix_np_b1_tv(params, center_x, center_y)
        mtx_torch = _mod._build_affine_matrix(params, height, width)[0].double()

        torch.testing.assert_close(torch.from_numpy(mtx_np), mtx_torch, atol=1e-6, rtol=1e-6)


class TestBuildMatrixNumpyB1TvAffineDispatch:
    """``build_matrix_numpy_b1_tv`` dispatches real ``RandomAffine`` instances to the branch under test."""

    def test_wrapper_matches_adapter_build_matrix(self) -> None:
        """The wrapper's ``_AFFINE_TYPES_FS`` branch matches ``TorchVisionAdapter.build_matrix``."""
        height, width = 32, 48
        transform = RandomAffine(
            degrees=(20.0, 20.0), translate=(0.1, 0.05), scale=(0.9, 0.9), shear=(10.0, 10.0, 5.0, 5.0)
        )
        angle, translations, scale_val, shear = RandomAffine.get_params(
            transform.degrees, transform.translate, transform.scale, transform.shear, img_size=(width, height)
        )
        params = {
            "angle_rad": torch.tensor([math.radians(angle)], dtype=torch.float32),
            "translate_x": torch.tensor([float(translations[0])], dtype=torch.float32),
            "translate_y": torch.tensor([float(translations[1])], dtype=torch.float32),
            "scale": torch.tensor([float(scale_val)], dtype=torch.float32),
            "shear_x_rad": torch.tensor([math.radians(shear[0])], dtype=torch.float32),
            "shear_y_rad": torch.tensor([math.radians(shear[1])], dtype=torch.float32),
        }

        mtx_np = _mod.build_matrix_numpy_b1_tv(transform, params, height, width)
        mtx_torch = _mod.TorchVisionAdapter.build_matrix(transform, params, height, width)[0].double()

        torch.testing.assert_close(torch.from_numpy(mtx_np), mtx_torch, atol=1e-6, rtol=1e-6)


class TestNumpyB1FastPathAffineRuntimeDispatch:
    """A real ``Compose(...)(image)`` forward that executes ``_affine_matrix_np_b1_tv`` end to end."""

    @staticmethod
    def _pinned_chain() -> list[object]:
        """A fusible, fully-deterministic 2-op chain: nonzero shear/scale/rotate then a rotation."""
        return [
            RandomAffine(degrees=(20.0, 20.0), translate=(0.0, 0.0), scale=(0.9, 0.9), shear=(10.0, 10.0, 5.0, 5.0)),
            RandomRotation(degrees=(15.0, 15.0)),
        ]

    def test_two_step_fallback_matches_torch_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forcing the two-step fallback agrees on interior pixels with the cv2-disabled torch path."""
        image = _smooth_image()

        with monkeypatch.context() as ctx:
            ctx.setattr(_segment_mod, "_cv2_module", lambda: None)
            pipe_torch = Compose(self._pinned_chain(), reorder=ReorderPolicy.NONE)
            out_torch = pipe_torch(image.clone())

        pipe_cv2 = Compose(self._pinned_chain(), reorder=ReorderPolicy.NONE)
        segment = pipe_cv2._segments[0]
        assert type(segment).__name__ == "FusedAffineSegment"
        # Disable the fused sampler so the forward pass falls through to
        # build_matrix_numpy_b1_tv -> _affine_matrix_np_b1_tv for the RandomAffine leg.
        segment._np_fused_builder = None
        out_cv2 = pipe_cv2(image.clone())

        assert out_cv2.shape == out_torch.shape
        diff = (_interior(out_cv2) - _interior(out_torch)).abs()
        assert diff.max().item() < 1e-2, f"interior max diff {diff.max().item():.5f}"
        assert diff.mean().item() < 1e-3, f"interior mean diff {diff.mean().item():.6f}"
