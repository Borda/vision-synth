"""Measure what a box loses through an affine warp, and what recovering it from the polygon returns.

``transform_bbox_xyxy`` returns the axis-aligned box of the four warped *box* corners, which is its
documented contract and the only thing it can do when a box is all it is given. Under a rotation that
box is looser than the object: the corners sweep out a rectangle whose area includes background the
original box never covered. A caller who also holds the outline -- which every segmentation-task
sample does -- can warp that instead and take its extent, recovering a box that hugs the ink again.

These tests pin the size of that gap against a rasterized oracle, so the recipe in
``docs/datasets/tasks.md`` carries a measured number rather than an assertion of principle. No core
change is involved: the recipe is ``polygon_to_bbox_xyxy(to_pixel_edge(warped_points))`` over a
pipeline declaring ``["input", "keypoints"]``.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fused_transforms import FusedCompose
from fused_transforms._compat import _TORCHVISION_AVAILABLE
from synth_datasets.core.config import Color, SyntheticConfig, Task
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.geometry import polygon_to_bbox_xyxy, to_pixel_edge
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SymbolShape

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import v2 as T

#: Every case here warps through a `torchvision` affine, so the module is skipped whole rather than
#: test by test. The import sits behind the same flag: an unguarded one fails collection, which
#: reports as an error rather than as the skip the extras-free CI legs are entitled to.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision"),
]

IMG_SIZE = 192
#: Objects are drawn in one color, so the warped ink is found by distance to it rather than by
#: difference from the background -- an affine warp also paints a border fill, which is neither.
_INK = np.asarray([255.0, 0.0, 0.0])
_INK_TOLERANCE = 90.0
#: Loose enough to hold across seeds, tight enough that the two routes cannot swap places.
_LOOSE_BOX_CEILING = 0.90
_TIGHT_BOX_FLOOR = 0.90


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Return the intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    areas = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
    return float(overlap / (areas - overlap))


def _ink_box(image: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return the edge-space extent of the warped ink, the oracle both routes are scored against."""
    ink = np.linalg.norm(image.astype(np.float64) - _INK, axis=2) < _INK_TOLERANCE
    if not ink.any():
        return None
    rows, columns = np.where(ink)
    return float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1)


def _measure(shapes: tuple, transform: T.Transform) -> tuple[float, float]:
    """Return mean IoU against the ink for the warped box and for the polygon-derived box."""
    config = SyntheticConfig(
        img_size=IMG_SIZE,
        task=Task.SEGMENTATION,
        shapes=shapes,
        colors=(Color.RED,),
        rotate=False,
        min_objects=1,
        max_objects=1,
        min_size_ratio=0.2,
        max_size_ratio=0.35,
    )
    pipeline = FusedCompose([transform], data_keys=["input", "keypoints", "bbox_xyxy"])
    through_box: list[float] = []
    through_polygon: list[float] = []

    for sample in SyntheticGenerator(config).generate(6, seed=0):
        annotation = sample.annotations[0]
        polygon = np.asarray(annotation.polygon, dtype=np.float64).reshape(-1, 2)
        image = torch.from_numpy(sample.image.copy()).permute(2, 0, 1).float()[None] / 255.0

        out_image, out_points, out_box = pipeline(
            image,
            torch.from_numpy(polygon).float()[None],
            torch.tensor([annotation.bbox_xyxy]).float()[None],
        )

        warped = (out_image[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        ink = _ink_box(warped)
        if ink is None:  # the object left the canvas; nothing to score
            continue
        through_box.append(_iou(tuple(out_box[0, 0].numpy()), ink))
        through_polygon.append(_iou(polygon_to_bbox_xyxy(to_pixel_edge(out_points[0].numpy())), ink))

    assert through_box, "no object survived the warp; the measurement has nothing to report"
    return float(np.mean(through_box)), float(np.mean(through_polygon))


@pytest.mark.parametrize(
    ("family", "shapes"),
    [
        pytest.param("geometric", (PrimitiveShape.SQUARE, PrimitiveShape.TRIANGLE), id="geometric"),
        pytest.param("symbols", tuple(SymbolShape)[:3], id="symbols"),
        pytest.param("animals", tuple(AnimalShape)[:3], id="animals"),
        pytest.param("letters", (LetterShape.A, LetterShape.K, LetterShape.Z), id="letters"),
    ],
)
def test_polygon_recovers_the_box_an_affine_warp_loosens(family: str, shapes: tuple) -> None:
    """Warping the outline yields a box on the ink; warping the box yields one that carries background.

    Every family is checked because how much a box loosens depends on how much of it the object fills:
    a rotated triangle wastes more of its own box than a letter does. The claim being pinned is the
    ordering and its size, not the exact decimal -- what matters to a detector is that the polygon
    route is available and materially tighter whenever a segmentation-task sample is in hand.

    """
    warp = T.RandomAffine(degrees=(37.0, 37.0), scale=(0.85, 0.85))

    through_box, through_polygon = _measure(shapes, warp)

    assert through_box <= _LOOSE_BOX_CEILING, family
    assert through_polygon >= _TIGHT_BOX_FLOOR, family
    assert through_polygon - through_box >= 0.10, family


def test_the_gap_survives_shear() -> None:
    """Adding shear does not close the gap, so the recipe is not specific to a pure rotation.

    Shear is the one warp where the package's own rotated-box handling degrades to a refit, so it is worth pinning that
    the polygon route still lands on the ink there.

    """
    warp = T.RandomAffine(degrees=(37.0, 37.0), scale=(0.85, 0.85), shear=(12.0, 12.0))

    through_box, through_polygon = _measure(tuple(AnimalShape)[:3], warp)

    assert through_box <= _LOOSE_BOX_CEILING
    assert through_polygon >= _TIGHT_BOX_FLOOR
    assert through_polygon - through_box >= 0.10
