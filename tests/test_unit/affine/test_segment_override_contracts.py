"""Regression tests for segment-level contracts that must not depend on pipeline shape.

* A one-transform pipeline takes a native fast path; it must still honour the ``Compose``-level ``fill``,
  ``padding_mode`` and ``interpolation`` overrides, exactly as the matrix path of a two-transform pipeline does.
* An exact quarter-turn routes the auxiliary mask together with the image; an integer label mask must come back with
  its own dtype and unchanged label values rather than promoted to the image's floating dtype.

"""

from __future__ import annotations

import pytest
import torch

from fused_transforms import Compose

T = pytest.importorskip("torchvision.transforms.v2")


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"fill": 0.5}, id="fill"),
        pytest.param({"padding_mode": "border"}, id="padding-border"),
        pytest.param({"padding_mode": "reflection"}, id="padding-reflection"),
        pytest.param({"interpolation": "bilinear"}, id="interp-bilinear"),
    ],
)
def test_single_op_fast_path_honours_compose_overrides(override: dict[str, object]) -> None:
    """A one-op pipeline and the same op followed by a no-op rotation give the same output under an override."""
    # Batch of two keeps both pipelines off the batch-of-one OpenCV warp, so both run grid_sample.
    image = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    one_op = Compose([T.RandomRotation((45, 45))], **override)
    two_op = Compose([T.RandomRotation((45, 45)), T.RandomRotation((0, 0))], **override)
    torch.testing.assert_close(one_op(image), two_op(image), atol=1e-5, rtol=0.0)


def test_single_op_fill_reaches_the_corner() -> None:
    """The constant border written by ``fill`` shows up in the corner a 45-degree turn uncovers."""
    out = Compose([T.RandomRotation((45, 45))], fill=0.5)(torch.ones(2, 3, 16, 16))
    torch.testing.assert_close(out[:, :, 0, 0], torch.full((2, 3), 0.5))


@pytest.mark.parametrize("mode", ["reflection", "border", "zeros"])
def test_per_transform_padding_is_not_an_override(mode: str) -> None:
    """``padding_mode="per_transform"`` keeps a one-op pipeline on its native render, bit-identical to the op.

    The per-transform policy writes the transform's own border mode onto the segment. That is the transform's setting,
    not a caller override, so it must not push the op off the native fast path (which would change its pixels).

    """
    kornia_aug = pytest.importorskip("kornia.augmentation")
    image = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(2))
    transform = kornia_aug.RandomAffine(degrees=(15, 15), padding_mode=mode, p=1.0)
    pipe = Compose([transform], padding_mode="per_transform")
    torch.testing.assert_close(pipe(image), transform(image), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.int32, torch.uint8, torch.bool])
def test_exact_quarter_turn_keeps_integer_mask_dtype_and_values(mask_dtype: torch.dtype) -> None:
    """A Kornia ``RandomRotation90`` returns the mask in its own dtype, rotated exactly like the image."""
    kornia_aug = pytest.importorskip("kornia.augmentation")
    high = 2 if mask_dtype is torch.bool else 200
    labels = torch.randint(0, high, (2, 1, 8, 8), generator=torch.Generator().manual_seed(1))
    mask = labels.to(mask_dtype)
    # The image carries the labels too, so the rotated image is the reference for the rotated mask.
    image = labels.to(torch.float32).expand(-1, 3, -1, -1).contiguous()
    pipe = Compose([kornia_aug.RandomRotation90(times=(1, 1), p=1.0)], data_keys=["input", "mask"])
    out_image, out_mask = pipe(image, mask)
    assert out_mask.dtype == mask_dtype
    assert out_image.dtype == torch.float32
    assert not torch.equal(out_mask, mask), "the quarter turn must actually move the labels"
    assert torch.equal(out_mask, out_image[:, :1].round().to(mask_dtype))
