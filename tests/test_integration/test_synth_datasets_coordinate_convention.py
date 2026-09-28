"""Coordinate-convention tests: which space each annotation field is in, and that a warp keeps it.

The package transforms boxes in pixel-edge space and points in pixel-centre space (see ``docs/known-limitations.md``). A
generated annotation therefore has to declare a space per field rather than share one: ``bbox_xyxy`` is edge space,
``polygon``/``keypoints``/``obb_corners`` are centre space. Get that wrong and the two agree under identity and disagree
by a full pixel after any reflection or quarter turn, which is the regression these tests exist to catch.

"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from fused_transforms import FusedCompose
from fused_transforms._compat import _TORCHVISION_AVAILABLE
from synth_datasets.core.config import ClassMode, Color, OutputFormat, SyntheticConfig, Task, class_vocabulary
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.export.writers import CocoWriter, YoloWriter
from synth_datasets.families.geometry import PIXEL_CENTRE_OFFSET, to_pixel_edge
from synth_datasets.families.primitives import PrimitiveShape

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import v2 as T

pytestmark = pytest.mark.integration

IMG_SIZE = 128


def _config(**overrides: object) -> SyntheticConfig:
    """Return a one-square-per-image config, the smallest scene with an unambiguous ink extent."""
    base: dict[str, object] = {
        "img_size": IMG_SIZE,
        "task": Task.SEGMENTATION,
        "shapes": (PrimitiveShape.SQUARE,),
        "colors": (Color.RED,),
        "rotate": False,
        "min_objects": 1,
        "max_objects": 1,
    }
    base.update(overrides)
    return SyntheticConfig(**base)  # type: ignore[arg-type]


def _ink_extent(image: np.ndarray, background: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Return ``(x_min, x_max, y_min, y_max)`` pixel indices of every non-background pixel."""
    ink = np.any(image.astype(np.int64) != np.asarray(background, dtype=np.int64), axis=2)
    ys, xs = np.where(ink)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _rasterize(flat_edge_polygon: list[float], size: int) -> np.ndarray:
    """Fill an edge-space outline the way the generator's own canvas does, returning a bool mask."""
    mask = Image.new("1", (size, size), 0)
    points = np.asarray(flat_edge_polygon, dtype=np.float64).reshape(-1, 2)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.array(mask, dtype=bool)


def test_polygon_is_emitted_half_a_pixel_below_the_outline_that_was_drawn() -> None:
    """Adding the offset back to `polygon` reproduces the rasterized shape exactly, pixel for pixel.

    This is the emission half of the convention: `polygon` is the drawn outline shifted into
    pixel-centre space, so shifting it back has to redraw the same ink, not merely a similar blob.

    """
    config = _config()
    generator = SyntheticGenerator(config)

    for sample in generator.generate(5, seed=0):
        annotation = sample.annotations[0]
        redrawn = _rasterize([float(v) for v in to_pixel_edge(np.asarray(annotation.polygon))], IMG_SIZE)
        drawn = np.any(sample.image.astype(np.int64) != np.asarray(config.background.color.rgb, dtype=np.int64), axis=2)
        assert np.array_equal(redrawn, drawn)


def test_bbox_stays_in_edge_space_around_the_polygon() -> None:
    """`bbox_xyxy` is the AABB of the *unshifted* outline, so it sits half a pixel outside `polygon`.

    Deriving it from the shifted polygon instead would drag the box into centre space, where `transform_bbox_xyxy` would
    then conjugate it a second time.

    """
    generator = SyntheticGenerator(_config())

    for sample in generator.generate(5, seed=0):
        annotation = sample.annotations[0]
        points = np.asarray(annotation.polygon, dtype=np.float64).reshape(-1, 2)
        x1, y1, x2, y2 = annotation.bbox_xyxy
        assert x1 == pytest.approx(points[:, 0].min() + PIXEL_CENTRE_OFFSET)
        assert y1 == pytest.approx(points[:, 1].min() + PIXEL_CENTRE_OFFSET)
        assert x2 == pytest.approx(points[:, 0].max() + PIXEL_CENTRE_OFFSET)
        assert y2 == pytest.approx(points[:, 1].max() + PIXEL_CENTRE_OFFSET)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
@pytest.mark.parametrize(
    ("name", "build_transform"),
    [
        pytest.param("identity", lambda: T.RandomHorizontalFlip(p=0.0), id="identity"),
        pytest.param("hflip", lambda: T.RandomHorizontalFlip(p=1.0), id="hflip"),
        pytest.param("vflip", lambda: T.RandomVerticalFlip(p=1.0), id="vflip"),
        pytest.param("rot90", lambda: T.RandomRotation((90.0, 90.0)), id="rot90"),
        pytest.param("rot270", lambda: T.RandomRotation((270.0, 270.0)), id="rot270"),
    ],
)
def test_polygon_and_box_both_track_the_ink_through_a_reflection(name: str, build_transform: Callable) -> None:
    """Polygon and box stay on the warped ink under every exact reflection and quarter turn.

    An exact flip or quarter turn permutes pixels without resampling, so the warped ink is as crisp
    as the original and the warped outline has to redraw it exactly -- not approximately. The
    regression this pins: with both fields in one space, one of them travelled through the wrong
    matrix and landed a full pixel off the ink, while identity and generic-angle rotations hid the
    error inside the interpolation band.

    """
    config = _config()
    generator = SyntheticGenerator(config)
    pipeline = FusedCompose([build_transform()], data_keys=["input", "keypoints", "bbox_xyxy"])

    for sample in generator.generate(5, seed=0):
        annotation = sample.annotations[0]
        polygon = np.asarray(annotation.polygon, dtype=np.float64).reshape(-1, 2)
        image = torch.from_numpy(sample.image.copy()).permute(2, 0, 1).float()[None] / 255.0

        out_image, out_points, out_box = pipeline(
            image,
            torch.from_numpy(polygon).float()[None],
            torch.tensor([annotation.bbox_xyxy]).float()[None],
        )

        warped = (out_image[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        ink = np.any(warped.astype(np.int64) != np.asarray(config.background.color.rgb, dtype=np.int64), axis=2)
        points = to_pixel_edge(out_points[0].numpy())  # back to the ink's own edge space
        assert np.array_equal(_rasterize([float(v) for v in points.reshape(-1)], IMG_SIZE), ink), name

        # The box is one pixel wider than the outline is long: its edges bound the ink pixels that
        # the outline's own endpoints fall inside.
        x_min, x_max, y_min, y_max = _ink_extent(warped, config.background.color.rgb)
        box = out_box[0, 0].numpy()
        for lo, hi, axis in ((x_min, x_max, 0), (y_min, y_max, 1)):
            assert lo <= box[axis] < lo + 1, name
            assert hi < box[axis + 2] <= hi + 1, name


def test_coco_segmentation_ring_redraws_the_exported_image(tmp_path) -> None:
    """An exported COCO ring is in edge space: redrawing it reproduces the ink of its own image.

    The writers convert the point fields back at the file boundary, so a consumer reading ``segmentation`` and ``bbox``
    out of one record reads them in one coordinate system.

    """
    config = _config()
    samples = list(SyntheticGenerator(config).generate(3, seed=0))
    vocabulary = class_vocabulary(ClassMode.SHAPE, config.shapes, config.colors)
    CocoWriter(Task.SEGMENTATION, vocabulary).write({"train": samples}, tmp_path)

    records = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_image = {image["id"]: image["file_name"] for image in records["images"]}
    assert len(records["annotations"]) == len(samples)

    for record in records["annotations"]:
        index = int(by_image[record["image_id"]].removeprefix("img_").removesuffix(".jpg"))
        redrawn = _rasterize(record["segmentation"][0], IMG_SIZE)
        drawn = np.any(
            samples[index].image.astype(np.int64) != np.asarray(config.background.color.rgb, dtype=np.int64), axis=2
        )
        assert np.array_equal(redrawn, drawn)
        x, y, width, height = record["bbox"]
        assert (x, y, x + width, y + height) == pytest.approx(samples[index].annotations[0].bbox_xyxy)


def test_yolo_segmentation_row_matches_the_coco_ring(tmp_path) -> None:
    """Both writers export the same edge-space ring, so neither carries a convention of its own."""
    config = _config()
    samples = list(SyntheticGenerator(config).generate(2, seed=0))
    vocabulary = class_vocabulary(ClassMode.SHAPE, config.shapes, config.colors)
    YoloWriter(Task.SEGMENTATION, vocabulary).write({"train": samples}, tmp_path)

    for index, sample in enumerate(samples):
        row = (tmp_path / "labels" / "train" / f"img_{index:06d}.txt").read_text(encoding="utf-8").split()
        exported = np.asarray([float(token) for token in row[1:]], dtype=np.float64) * IMG_SIZE
        expected = np.asarray(sample.annotations[0].polygon, dtype=np.float64) + PIXEL_CENTRE_OFFSET
        assert exported == pytest.approx(expected, abs=1e-3)


def test_output_format_enum_still_names_both_writers() -> None:
    """Guard the import above: the convention tests cover every format the package exports."""
    assert {OutputFormat.COCO, OutputFormat.YOLO} == set(OutputFormat)
