"""Regression tests for three segment contracts that only failed off the common path.

* An exact quarter-turn must return an integer label mask bit-exact for *any* value; stacking it with the image in
  ``float64`` rounded every label above ``2**53``.
* A standalone ``FusedAffineSegment`` pickled before the ``_render_overridden`` flag existed must still run; only
  ``FusedCompose`` used to migrate it, so the bare segment raised ``AttributeError``.
* ``antialias=True`` needs Kornia to *import*, not merely be installed; a broken install used to be accepted and fail
  later, on the first downscale.

"""

from __future__ import annotations

import pickle

import pytest
import torch

from fused_transforms import Compose
from fused_transforms._compat import _KORNIA_AVAILABLE
from fused_transforms.affine.segment import FusedAffineSegment

kornia_aug = pytest.importorskip("kornia.augmentation")
T = pytest.importorskip("torchvision.transforms.v2")


@pytest.mark.parametrize("base", [2**53, torch.iinfo(torch.int64).max - 16 * 16], ids=["2**53", "int64-max"])
@pytest.mark.parametrize("batch", [1, 3])
def test_exact_quarter_turn_keeps_large_integer_labels_exact(base: int, batch: int) -> None:
    """Labels ``base + i`` (``2**53 + 1`` among them, up to int64 max) come back exactly where the pixel ``i`` goes."""
    height, width = 16, 16
    index = torch.arange(height * width, dtype=torch.int64).reshape(1, 1, height, width).expand(batch, 1, -1, -1)
    # The image carries the small pixel index, which float32 holds exactly, as the reference for where each label goes.
    image = index.to(torch.float32).expand(batch, 3, -1, -1).contiguous()
    mask = (index + base).contiguous()
    pipe = Compose([kornia_aug.RandomRotation90(times=(1, 1), p=1.0)], data_keys=["input", "mask"])

    out_image, out_mask = pipe(image, mask)

    assert out_mask.dtype == torch.int64
    expected = out_image[:, :1].round().to(torch.int64) + base
    assert not torch.equal(out_image[:, :1], image[:, :1]), "the quarter turn must actually move the pixels"
    assert torch.equal(out_mask, expected)
    assert int(out_mask.max()) == base + height * width - 1


def test_standalone_segment_pickled_before_the_render_flag_still_runs() -> None:
    """A bare ``FusedAffineSegment`` restored from a pre-flag pickle derives the flag from its own arguments."""
    pipe = Compose([T.RandomRotation((45, 45))])
    segment = pipe._segments[0]
    assert isinstance(segment, FusedAffineSegment)
    image = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    expected = segment(image)
    del segment._render_overridden  # what a pickle written before the flag existed carries

    restored = pickle.loads(pickle.dumps(segment))  # noqa: S301

    assert restored._render_overridden is False
    torch.testing.assert_close(restored(image), expected)


def test_compose_pickled_before_the_render_flag_still_applies_its_own_overrides() -> None:
    """Inside a ``FusedCompose`` the Compose-level overrides still decide, exactly as ``build_segments`` does."""
    pipe = Compose([T.RandomRotation((45, 45))], fill=0.5)
    del pipe._segments[0]._render_overridden

    restored = pickle.loads(pickle.dumps(pipe))  # noqa: S301

    assert restored._segments[0]._render_overridden is True


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="needs kornia installed to simulate a broken install")
def test_antialias_rejects_an_installed_but_unimportable_kornia(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Kornia that is installed but fails to import is refused at construction, not on the first downscale."""
    from fused_transforms import _compat

    monkeypatch.setitem(_compat._LOADED, "kornia", None)

    with pytest.raises(ImportError, match="kornia"):
        Compose([T.RandomRotation((45, 45))], antialias=True)


def test_standalone_legacy_pickle_keeps_its_old_fast_path_rendering() -> None:
    """A pre-flag standalone segment restores with the fast path, whatever padding mode its state carries.

    Under ``padding_mode="per_transform"`` a segment stores the transform's own mode (here ``reflection``); deriving
    the flag from that sent a legacy pickle down the matrix path and changed its pixels.
    """
    pipe = Compose([T.RandomRotation((45, 45))])
    segment = pipe._segments[0]
    image = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    expected = segment(image)
    segment.padding_mode = "reflection"
    del segment._render_overridden

    restored = pickle.loads(pickle.dumps(segment))  # noqa: S301

    assert restored._render_overridden is False
    torch.testing.assert_close(restored(image), expected)


def test_float32_cannot_hold_every_index_of_a_large_canvas() -> None:
    """Why one float32 index plane was not enough on MPS: the first index past ``2**24`` rounds away."""
    assert int(torch.tensor(2**24 + 1, dtype=torch.float32).item()) != 2**24 + 1


def test_exact_quarter_turn_rebuilds_indices_from_two_planes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the plane base shrunk to 16, a 16x16 canvas needs both planes, and large labels still come back exact."""
    from fused_transforms.affine import segment as segment_module

    monkeypatch.setattr(segment_module, "_INDEX_PLANE_BASE", 16)
    index = torch.arange(256, dtype=torch.int64).reshape(1, 1, 16, 16)
    image = index.to(torch.float32).expand(1, 3, -1, -1).contiguous()
    pipe = Compose([kornia_aug.RandomRotation90(times=(1, 1), p=1.0)], data_keys=["input", "mask"])

    out_image, out_mask = pipe(image, index + 2**53)

    assert torch.equal(out_mask, out_image[:, :1].round().to(torch.int64) + 2**53)


def test_exact_quarter_turn_refuses_a_canvas_too_large_to_track(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past what two planes can index the mask is refused with a clear error rather than routed inexactly."""
    from fused_transforms.affine import segment as segment_module

    monkeypatch.setattr(segment_module, "_INDEX_PLANE_BASE", 8)
    pipe = Compose([kornia_aug.RandomRotation90(times=(1, 1), p=1.0)], data_keys=["input", "mask"])

    with pytest.raises(ValueError, match="too large"):
        pipe(torch.rand(1, 3, 16, 16), torch.zeros(1, 1, 16, 16, dtype=torch.int64))
