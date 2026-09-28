"""Rectangular ``(width, height)`` canvases, end to end.

A square ``img_size`` stays a plain int everywhere and draws bit-identical pixels (``test_baseline_digests.py`` pins
that). These tests cover the other case: every axis-dependent step — backgrounds, degradations, placement, the boundary
check, keypoint visibility, the occluder stencil, both writers, the streaming dataset and the command line — reads width
from ``W`` and height from ``H``, and nothing assumes they are equal.

"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from synth_datasets import (
    JPEG,
    Background,
    ColorCast,
    Contrast,
    GaussianBlur,
    GaussianNoise,
    GradientBackground,
    ImageBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    Quantize,
    SolidBackground,
    SyntheticConfig,
    SyntheticGenerator,
    Task,
    TextureBackground,
    Vignette,
    generate_dataset,
)
from synth_datasets.core.generator import _boundary_overlap
from synth_datasets.export.writers import YoloWriter
from synth_datasets.families.primitives import PrimitiveShape

WIDE = (96, 48)
TALL = (48, 96)
_SHAPES = ("duck", "camel", "fish")


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("img_size", "expected"),
    [
        pytest.param(64, (64, 64), id="int"),
        pytest.param((96, 48), (96, 48), id="tuple"),
        pytest.param([96, 48], (96, 48), id="list"),
        pytest.param((64, 64), (64, 64), id="square-tuple"),
    ],
)
def test_canvas_size_reads_back_width_then_height(img_size, expected):
    assert SyntheticConfig(img_size=img_size).canvas_size == expected


def test_a_list_img_size_is_stored_as_a_tuple_so_the_config_stays_hashable():
    config = SyntheticConfig(img_size=[96, 48])
    assert config.img_size == (96, 48)
    hash(config)


@pytest.mark.parametrize(
    ("img_size", "match"),
    [
        pytest.param(0, "positive", id="zero"),
        pytest.param(-4, "positive", id="negative"),
        pytest.param((96, 0), "positive", id="zero-height"),
        pytest.param((-1, 48), "positive", id="negative-width"),
        pytest.param((96,), "two values", id="one-value"),
        pytest.param((96, 48, 3), "two values", id="three-values"),
        pytest.param((96.0, 48), "int", id="float"),
        pytest.param(64.0, "int", id="scalar-float"),
        pytest.param("64", "int", id="string"),
        pytest.param(True, "int", id="bool"),
        pytest.param((True, 48), "int", id="bool-in-tuple"),
    ],
)
def test_img_size_rejects_malformed_values(img_size, match):
    with pytest.raises(ValueError, match=match):
        SyntheticConfig(img_size=img_size)


# --------------------------------------------------------------------------- backgrounds and degradations


def _picture_dir(tmp_path):
    rng = np.random.default_rng(0)
    # One file smaller than the canvas on one axis only, so the upscale has to honour both axes.
    for name, (width, height) in {"small.png": (120, 30), "large.png": (200, 150)}.items():
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / name)
    return tmp_path


def _backgrounds(tmp_path) -> list[Background]:
    return [
        SolidBackground(),
        GradientBackground(),
        GradientBackground(direction=0.3),
        GradientBackground(radial=True),
        NoiseBackground(),
        ImpulseNoiseBackground(),
        TextureBackground(),
        TextureBackground(quantize=4),
        ImageBackground(_picture_dir(tmp_path)),
        ImageBackground(tmp_path, grayscale=True),
    ]


@pytest.mark.parametrize("canvas", [WIDE, TALL], ids=["wide", "tall"])
def test_every_background_renders_height_by_width(tmp_path, canvas):
    width, height = canvas
    for background in _backgrounds(tmp_path):
        rng = np.random.default_rng(0) if background.consumes_randomness else None
        pixels, _source = background.render_with_source(rng, canvas)
        assert pixels.shape == (height, width, 3), type(background).__name__
        assert pixels.dtype == np.uint8


def test_a_radial_gradient_is_centred_on_both_axes():
    pixels = GradientBackground(radial=True).render(None, WIDE).astype(int)
    # Symmetric about the vertical and horizontal centre lines, darkest (first stop) at the middle.
    np.testing.assert_array_equal(pixels, pixels[:, ::-1])
    np.testing.assert_array_equal(pixels, pixels[::-1, :])
    assert pixels[24, 48].sum() < pixels[0, 0].sum()


def test_texture_features_keep_their_size_on_a_rectangular_canvas():
    """The lattice gets cells per axis, so a wide canvas shows more features across rather than stretched ones.

    A square ``(cells, cells)`` lattice resized to 96x48 would stretch every feature to twice its height, halving the
    horizontal gradient against the vertical one; per-axis cells keep the two about equal.

    """
    wide = TextureBackground(octaves=1, frequency=8.0, amplitude=100.0).render(np.random.default_rng(0), WIDE)
    field = wide[..., 0].astype(float)
    ratio = np.abs(np.diff(field, axis=1)).mean() / np.abs(np.diff(field, axis=0)).mean()
    assert 0.75 < ratio < 1.33


@pytest.mark.parametrize(
    "step",
    [GaussianNoise(), GaussianBlur(), JPEG(), Contrast(), ColorCast(), Vignette(), Quantize()],
    ids=lambda step: type(step).__name__,
)
def test_every_degradation_keeps_a_rectangular_image_s_shape(step):
    image = np.full((48, 96, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0) if step.consumes_randomness else None
    assert step.apply(image, rng).shape == (48, 96, 3)


def test_vignette_is_centred_on_both_axes():
    out = Vignette(strength=0.5).apply(np.full((48, 96, 3), 200, dtype=np.uint8), None).astype(int)
    np.testing.assert_array_equal(out, out[:, ::-1])
    np.testing.assert_array_equal(out, out[::-1, :])


class _RecordingBackground(Background):
    """A user subclass written against the plain-int signature."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    @property
    def consumes_randomness(self) -> bool:
        return False

    def render(self, rng, img_size):
        self.seen.append(img_size)
        width, height = (img_size, img_size) if isinstance(img_size, int) else img_size
        return np.full((height, width, 3), 90, dtype=np.uint8)


@pytest.mark.parametrize(
    ("img_size", "passed"),
    [
        pytest.param(32, 32, id="int"),
        pytest.param((32, 32), 32, id="square-tuple"),
        pytest.param(WIDE, WIDE, id="wide"),
    ],
)
def test_a_background_receives_an_int_on_square_canvases_and_width_height_otherwise(img_size, passed):
    background = _RecordingBackground()
    next(SyntheticGenerator(SyntheticConfig(img_size=img_size, background=background)).generate(1, seed=0))
    assert background.seen == [passed]


# --------------------------------------------------------------------------- generator


def _config(task: Task, canvas, **overrides) -> SyntheticConfig:
    fields = {
        "img_size": canvas,
        "task": task,
        "shapes": _SHAPES,
        "min_objects": 2,
        "max_objects": 5,
        "boundary_tolerance": 0.0,
        "occluders": 2,
        "distractors": 2,
    }
    return SyntheticConfig(**{**fields, **overrides})


@pytest.mark.parametrize("canvas", [WIDE, TALL], ids=["wide", "tall"])
@pytest.mark.parametrize("task", list(Task), ids=lambda task: task.value)
def test_every_annotation_stays_inside_a_rectangular_canvas(task, canvas):
    width, height = canvas
    config = _config(task, canvas)
    samples = list(SyntheticGenerator(config).generate(12, seed=3))
    far = []
    for sample in samples:
        assert sample.image.shape == (height, width, 3)
        assert (sample.width, sample.height) == canvas
        assert sample.scene.occluder_mask.shape == (height, width)
        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox_xyxy
            assert 0.0 <= x1 <= x2 <= width
            assert 0.0 <= y1 <= y2 <= height
            assert _boundary_overlap(ann.bbox_xyxy, canvas) == 0.0
            # The polygon is in pixel-centre coordinates, half a pixel below the edge-space box.
            xs, ys = ann.polygon[0::2], ann.polygon[1::2]
            assert min(xs) >= -0.5
            assert max(xs) <= width - 0.5
            assert min(ys) >= -0.5
            assert max(ys) <= height - 0.5
            for x, y, visibility in ann.keypoints or ():
                if visibility:
                    assert 0.0 <= x < width
                    assert 0.0 <= y < height
            far.append(x2 if width > height else y2)
    # Placement uses the long axis too: some object lies beyond the short side.
    assert max(far) > min(canvas)


def test_object_size_is_a_ratio_of_the_shorter_side():
    fixed = {"shapes": (PrimitiveShape.SQUARE,), "rotate": False, "min_size_ratio": 0.25, "max_size_ratio": 0.25}

    def box_sides(img_size):
        config = SyntheticConfig(img_size=img_size, **fixed)
        sample = next(SyntheticGenerator(config).generate(1, seed=0))
        return {
            (round(a.bbox_xyxy[2] - a.bbox_xyxy[0], 6), round(a.bbox_xyxy[3] - a.bbox_xyxy[1], 6))
            for a in sample.annotations
        }

    assert box_sides(WIDE) == box_sides(TALL) == box_sides(48)


def test_boundary_overlap_clips_each_axis_to_its_own_extent():
    # A 10x10 box at x 90..100 on a 96-wide canvas: 4 of 10 columns outside, every row inside a 48-high canvas.
    assert _boundary_overlap((90.0, 10.0, 100.0, 20.0), WIDE) == pytest.approx(0.4)
    # The same box on the tall canvas is fully outside horizontally.
    assert _boundary_overlap((90.0, 10.0, 100.0, 20.0), TALL) == pytest.approx(1.0)


# --------------------------------------------------------------------------- writers


def test_coco_records_width_and_height_per_axis(tmp_path):
    counts = generate_dataset(tmp_path, 4, fmt="coco", img_size=WIDE, seed=0, shapes=_SHAPES, task="segmentation")
    for split in counts:
        doc = json.loads((tmp_path / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        for record in doc["images"]:
            assert (record["width"], record["height"]) == WIDE
            with Image.open(tmp_path / split / record["file_name"]) as image:
                assert image.size == WIDE
        for ann in doc["annotations"]:
            x, y, w, h = ann["bbox"]
            assert x + w <= WIDE[0]
            assert y + h <= WIDE[1]


def test_yolo_normalises_x_by_width_and_y_by_height():
    config = _config(Task.DETECTION, WIDE, occluders=0, distractors=0)
    sample = next(SyntheticGenerator(config).generate(1, seed=0))
    writer = YoloWriter(Task.DETECTION, SyntheticGenerator(config).vocabulary)
    for ann in sample.annotations:
        x1, y1, x2, y2 = ann.bbox_xyxy
        expected = [(x1 + x2) / 2 / 96, (y1 + y2) / 2 / 48, (x2 - x1) / 96, (y2 - y1) / 48]
        row = [float(token) for token in writer._label_row(ann, sample.width, sample.height).split()[1:]]
        np.testing.assert_allclose(row, expected, atol=1e-6)


@pytest.mark.parametrize("task", ["detection", "obb", "segmentation", "keypoints"])
def test_yolo_labels_stay_normalised_on_a_rectangular_canvas(tmp_path, task):
    generate_dataset(tmp_path, 4, fmt="yolo", img_size=TALL, seed=1, shapes=_SHAPES, task=task)
    rows = [row for path in (tmp_path / "labels").rglob("*.txt") for row in path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert all(0.0 <= float(token) <= 2.0 for token in row.split()[1:])  # visibility flags are 0/1/2
    for path in (tmp_path / "images").rglob("*.jpg"):
        with Image.open(path) as image:
            assert image.size == TALL


# --------------------------------------------------------------------------- dataset and CLI


def test_a_rectangular_config_round_trips_through_the_iterable_dataset():
    pytest.importorskip("torch")
    from synth_datasets.export.datasets import SyntheticIterableDataset

    for dataset in (
        SyntheticIterableDataset(num_images=3, img_size=WIDE, seed=0),
        SyntheticIterableDataset(num_images=3, config=SyntheticConfig(img_size=WIDE), seed=0),
    ):
        assert dataset.config.canvas_size == WIDE
        samples = list(dataset)
        assert len(samples) == 3
        assert all(sample.image.shape == (48, 96, 3) for sample in samples)


def test_cli_accepts_a_width_height_img_size(tmp_path):
    pytest.importorskip("fire")
    from synth_datasets import cli

    out = tmp_path / "ds"
    cli.main(["generate", str(out), "3", "--fmt", "yolo", "--img_size", "96,48", "--seed", "0"])
    images = list((out / "images").rglob("*.jpg"))
    assert images
    for path in images:
        with Image.open(path) as image:
            assert image.size == WIDE
