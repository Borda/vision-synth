"""Generator tests: reproducibility, placement constraints, and class modes."""

from __future__ import annotations

import numpy as np
import pytest

from synth_datasets.core.config import (
    DEFAULT_SHAPES,
    ClassMode,
    Color,
    SyntheticConfig,
    Task,
    class_names,
)
from synth_datasets.core.generator import SyntheticGenerator, _boundary_overlap, _visible_keypoints
from synth_datasets.families import ALL_SHAPES
from synth_datasets.families.animals import ANIMAL_KEYPOINT_NAMES, AnimalShape
from synth_datasets.families.geometry import bbox_iou
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SYMBOL_KEYPOINT_NAMES, SymbolShape

from ._obb_pose import rebuild_error


def _generate(**kwargs: object) -> tuple:  # type: ignore[type-arg]
    config = SyntheticConfig(**kwargs)
    return SyntheticGenerator(config).sample(np.random.default_rng(123)), config


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray-casting (even-odd rule) point-in-polygon test."""
    x, y = point
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside


def test_image_shape_and_dtype() -> None:
    sample, _config = _generate(img_size=96)
    assert sample.image.shape == (96, 96, 3)
    assert sample.image.dtype == np.uint8
    assert sample.width == 96
    assert sample.height == 96


def test_object_count_within_range() -> None:
    sample, _ = _generate(img_size=128, min_objects=2, max_objects=5)
    assert 2 <= len(sample.annotations) <= 5


def test_same_seed_is_identical() -> None:
    config = SyntheticConfig(img_size=64)
    gen = SyntheticGenerator(config)
    a = gen.sample(np.random.default_rng(7))
    b = gen.sample(np.random.default_rng(7))
    assert np.array_equal(a.image, b.image)
    assert [ann.bbox_xyxy for ann in a.annotations] == [ann.bbox_xyxy for ann in b.annotations]


def test_different_seed_differs() -> None:
    config = SyntheticConfig(img_size=64)
    gen = SyntheticGenerator(config)
    a = gen.sample(np.random.default_rng(1))
    b = gen.sample(np.random.default_rng(2))
    assert not np.array_equal(a.image, b.image)


def test_placement_respects_boundary_tolerance() -> None:
    sample, config = _generate(img_size=128, min_objects=5, max_objects=10)
    for ann in sample.annotations:
        assert _boundary_overlap(ann.bbox_xyxy, config.img_size) <= config.boundary_tolerance + 1e-9


def test_kept_boxes_respect_overlap_threshold() -> None:
    sample, config = _generate(img_size=128, min_objects=5, max_objects=10)
    boxes = [ann.bbox_xyxy for ann in sample.annotations]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert bbox_iou(boxes[i], boxes[j]) <= config.overlap_iou + 1e-9


def test_annotation_fields_are_consistent() -> None:
    sample, _ = _generate(img_size=96, min_objects=3, max_objects=3)
    for ann in sample.annotations:
        assert len(ann.obb_corners) == 8
        assert len(ann.polygon) >= 6
        x1, y1, x2, y2 = ann.bbox_xyxy
        assert x2 > x1
        assert y2 > y1


def test_generate_yields_lazily() -> None:
    import types

    gen = SyntheticGenerator(SyntheticConfig(img_size=32))
    stream = gen.generate(3, seed=0)
    assert isinstance(stream, types.GeneratorType)


def test_generate_yields_exact_count() -> None:
    gen = SyntheticGenerator(SyntheticConfig(img_size=32))
    assert len(list(gen.generate(5, seed=0))) == 5


def test_generate_same_seed_is_deterministic() -> None:
    gen = SyntheticGenerator(SyntheticConfig(img_size=32))
    first = [ann.bbox_xyxy for s in gen.generate(3, seed=7) for ann in s.annotations]
    second = [ann.bbox_xyxy for s in gen.generate(3, seed=7) for ann in s.annotations]
    assert first == second


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ClassMode.SHAPE, {s.value for s in (*PrimitiveShape, *AnimalShape, *SymbolShape, *LetterShape)}),
        (ClassMode.COLOR, {c.value for c in Color}),
        (
            ClassMode.SHAPE_COLOR,
            {
                f"{c.value}_{s.value}"
                for s in (*PrimitiveShape, *AnimalShape, *SymbolShape, *LetterShape)
                for c in Color
            },
        ),
    ],
)
def test_class_names_belong_to_mode(mode: ClassMode, expected: set[str]) -> None:
    config = SyntheticConfig(img_size=128, min_objects=8, max_objects=10, class_mode=mode)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    vocab = set(class_names(mode, ALL_SHAPES))
    assert vocab == expected
    # Indexed through the run's *own* narrowed vocabulary, the one every writer declares -- not
    # `class_names(mode, ALL_SHAPES)` above, which spans the full 49 shapes and would agree here only because
    # the default geometric shapes happen to lead it (the coincidence that hid a real id mismatch).
    run_names = class_names(mode, config.shapes)
    for ann in sample.annotations:
        assert ann.class_name in vocab
        assert run_names[ann.class_id] == ann.class_name


def test_sample_reaches_requested_count_under_pressure() -> None:
    # min==max fixes num_objects; a tight-but-feasible scene forces retries yet must still
    # place every object rather than silently dropping any.
    config = SyntheticConfig(
        img_size=96, min_objects=8, max_objects=8, min_size_ratio=0.16, max_size_ratio=0.26, overlap_iou=0.05
    )
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    assert len(sample.annotations) == 8


def test_sample_retry_is_reproducible_for_a_seed() -> None:
    config = SyntheticConfig(
        img_size=96, min_objects=8, max_objects=8, min_size_ratio=0.16, max_size_ratio=0.26, overlap_iou=0.05
    )
    gen = SyntheticGenerator(config)
    first = [ann.bbox_xyxy for ann in gen.sample(np.random.default_rng(3)).annotations]
    second = [ann.bbox_xyxy for ann in gen.sample(np.random.default_rng(3)).annotations]
    assert first == second


#: Pinned to `AnimalShape` directly (not derived as "every non-default shape") now that a third,
#: non-geometric family exists: a derived `not in DEFAULT_SHAPES` filter would silently sweep every
#: `SymbolShape` into a fixture whose name and tests are animal-specific.
ANIMAL_SHAPES = tuple(AnimalShape)


def test_default_config_draws_only_the_original_four_shapes() -> None:
    """A config that never mentions `shapes` still produces only the pre-animal vocabulary.

    This is the compatibility guarantee behind appending to `Shape`: existing seeded callers must keep getting
    square/rectangle/triangle/circle and nothing else.

    """
    config = SyntheticConfig(img_size=192, min_objects=12, max_objects=12, class_mode=ClassMode.SHAPE)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    drawn = {ann.class_name for ann in sample.annotations}
    assert drawn <= {s.value for s in DEFAULT_SHAPES}


@pytest.mark.parametrize("shape", ANIMAL_SHAPES)
def test_single_animal_shape_is_the_only_one_drawn(shape: AnimalShape | PrimitiveShape) -> None:
    """Restricting `cfg.shapes` to one animal makes every annotation carry that class.

    This is the documented opt-in for the animal family; a sampler still reading the full enum would leak geometric
    shapes into a dataset the caller asked to be animals-only.

    """
    config = SyntheticConfig(img_size=160, min_objects=4, max_objects=6, shapes=(shape,))
    sample = SyntheticGenerator(config).sample(np.random.default_rng(11))
    assert {ann.class_name for ann in sample.annotations} == {shape.value}


def test_default_config_draws_from_all_three_colors() -> None:
    """A config that never mentions `colors` can still produce all three colors.

    Compatibility guarantee for the `colors` field, mirroring `test_default_config_draws_only_the_original_four_shapes`
    for `shapes`: an untouched config keeps sampling from the full `Color` vocabulary.

    """
    config = SyntheticConfig(img_size=192, min_objects=12, max_objects=12, class_mode=ClassMode.COLOR)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    drawn = {ann.class_name for ann in sample.annotations}
    assert drawn == {c.value for c in Color}


def test_single_color_override_is_the_only_one_drawn() -> None:
    """Restricting `cfg.colors` to one color makes every annotation carry that color's class.

    Mirrors `test_single_animal_shape_is_the_only_one_drawn` for `shapes`: a sampler still reading the full `Color` enum
    would leak the other two colors into a dataset the caller asked to be single-color-only.

    """
    config = SyntheticConfig(
        img_size=160, min_objects=4, max_objects=6, class_mode=ClassMode.COLOR, colors=(Color.BLUE,)
    )
    sample = SyntheticGenerator(config).sample(np.random.default_rng(11))
    assert {ann.class_name for ann in sample.annotations} == {Color.BLUE.value}


def test_animal_shapes_respect_boundary_tolerance() -> None:
    """Animal placements obey the same off-canvas budget as the geometric shapes.

    Animals are far less convex than a square, so this confirms the rejection test still works on a box derived from a
    concave outline rather than only on well-behaved ones.

    """
    config = SyntheticConfig(img_size=192, min_objects=5, max_objects=8, shapes=ANIMAL_SHAPES)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(4))
    for ann in sample.annotations:
        assert _boundary_overlap(ann.bbox_xyxy, config.img_size) <= config.boundary_tolerance + 1e-9


def test_animal_shapes_respect_overlap_threshold() -> None:
    """Kept animal boxes stay under `overlap_iou` pairwise.

    Elongated silhouettes (whale, crocodile) have large bounding boxes relative to their filled area, which is exactly
    the case where a broken IoU guard would let objects pile up.

    """
    config = SyntheticConfig(img_size=192, min_objects=5, max_objects=8, shapes=ANIMAL_SHAPES)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(4))
    boxes = [ann.bbox_xyxy for ann in sample.annotations]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert bbox_iou(boxes[i], boxes[j]) <= config.overlap_iou + 1e-9


def test_animal_shapes_are_seed_deterministic() -> None:
    """Two runs of the same seed over the animal vocabulary agree pixel- and label-wise.

    Animal outlines come from lookup tables rather than trigonometry, so this pins that the table path introduces no
    ordering or floating-point nondeterminism of its own.

    """
    config = SyntheticConfig(img_size=128, min_objects=3, max_objects=5, shapes=ANIMAL_SHAPES)
    gen = SyntheticGenerator(config)
    first = gen.sample(np.random.default_rng(21))
    second = gen.sample(np.random.default_rng(21))
    assert np.array_equal(first.image, second.image)
    assert [(a.class_name, a.bbox_xyxy, a.polygon) for a in first.annotations] == [
        (b.class_name, b.bbox_xyxy, b.polygon) for b in second.annotations
    ]


def test_animal_annotations_carry_a_multi_vertex_polygon() -> None:
    """An animal annotation exports its full outline, not a four-corner approximation.

    Segmentation labels are taken straight from the drawn polygon, so an animal collapsing to its box would produce
    masks that no longer match the rendered pixels.

    """
    config = SyntheticConfig(img_size=192, min_objects=3, max_objects=3, shapes=(AnimalShape.ELEPHANT,))
    sample = SyntheticGenerator(config).sample(np.random.default_rng(2))
    for ann in sample.annotations:
        assert len(ann.polygon) >= 2 * 15
        assert len(ann.obb_corners) == 8


def test_sample_raises_when_min_objects_unreachable() -> None:
    # Near-full shapes with zero tolerated overlap and full-containment cannot fit min_objects.
    config = SyntheticConfig(
        img_size=32,
        min_objects=8,
        max_objects=8,
        min_size_ratio=0.9,
        max_size_ratio=1.0,
        overlap_iou=0.0,
        boundary_tolerance=0.0,
        max_placement_attempts=5,
        rotate=False,
    )
    gen = SyntheticGenerator(config)
    with pytest.raises(RuntimeError, match="min_objects"):
        gen.sample(np.random.default_rng(0))


def test_absent_limb_keypoints_get_zero_visibility_even_fully_inside_canvas() -> None:
    """A whale's absent hind limbs are v=0 even when the whole object is on-canvas.

    Visibility ``0`` normally means "clipped off the canvas frame" — the one case `animal_shapes.animal_keypoints`
    cannot fake a coordinate for is a landmark the animal never had at all (no NaN-vs-clipped special case in
    `_visible_keypoints`: both fall to the same `else` branch because ``0.0 <= nan`` is `False`, per the generator's own
    docstring).

    """
    config = SyntheticConfig(
        img_size=256,
        min_objects=1,
        max_objects=1,
        min_size_ratio=0.2,
        max_size_ratio=0.2,
        task=Task.KEYPOINTS,
        shapes=(AnimalShape.WHALE,),
    )
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    (ann,) = sample.annotations
    triples = dict(zip(ANIMAL_KEYPOINT_NAMES, ann.keypoints, strict=True))
    for name in ("hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"):
        assert triples[name] == (0.0, 0.0, 0)
    for name in ("mouth", "head", "body_top", "tail", "front_elbow_left", "front_limb_left"):
        assert triples[name][2] == 2


def test_visible_keypoints_clips_off_canvas_points_and_zeroes_nan() -> None:
    """`_visible_keypoints` zeroes every point outside `[0, img_size)` on either axis, and any NaN point.

    This is the function's primary documented contract, exercised directly rather than only incidentally through a
    fully-on-canvas placement: an in-bounds point, negative-x, y-beyond, x-beyond, a point exactly at the `img_size`
    boundary (pinning the half-open `x < img_size` semantics — the boundary itself is *not* visible), and a NaN point.

    Visibility is decided on the incoming edge-space coordinates, while a surviving point is reported in pixel-centre
    space, so the visible triple sits half a pixel below its input and the zeroed placeholders stay exactly zero.

    """
    img_size = 100
    points = np.array([
        [50.0, 50.0],  # clearly inside
        [-5.0, 50.0],  # negative x
        [50.0, 150.0],  # y beyond img_size
        [150.0, 50.0],  # x beyond img_size
        [100.0, 50.0],  # exactly at the img_size boundary — half-open, so not visible
        [np.nan, np.nan],  # absent landmark
    ])

    triples = _visible_keypoints(points, img_size)

    assert triples[0] == (49.5, 49.5, 2)
    for triple in triples[1:]:
        assert triple == (0.0, 0.0, 0)


def test_task_choice_does_not_perturb_the_seeded_scene() -> None:
    """Switching the config `task` between DETECTION and KEYPOINTS draws the identical scene for a fixed seed.

    The module docstring and `_attempt_placement` both promise that landmarks are a pure function of the placement
    already sampled and consume no extra randomness, so a seed must reproduce byte-identical pixels and boxes whatever
    the configured task — only whether each annotation additionally carries keypoints should differ.

    """
    common = {
        "img_size": 128,
        "min_objects": 3,
        "max_objects": 3,
        "shapes": (AnimalShape.DUCK, AnimalShape.ELEPHANT, AnimalShape.GIRAFFE),
    }
    detection_config = SyntheticConfig(task=Task.DETECTION, **common)
    keypoints_config = SyntheticConfig(task=Task.KEYPOINTS, **common)
    detection_sample = SyntheticGenerator(detection_config).sample(np.random.default_rng(17))
    keypoints_sample = SyntheticGenerator(keypoints_config).sample(np.random.default_rng(17))

    assert np.array_equal(detection_sample.image, keypoints_sample.image)
    assert [ann.bbox_xyxy for ann in detection_sample.annotations] == [
        ann.bbox_xyxy for ann in keypoints_sample.annotations
    ]
    for ann in detection_sample.annotations:
        assert ann.keypoints is None
    for ann in keypoints_sample.annotations:
        assert len(ann.keypoints) == 16


def test_asymmetry_jitter_default_leaves_placement_unchanged() -> None:
    """`asymmetry_jitter=0.0` (the default) draws the identical scene as before the knob existed.

    The extra RNG draw for `skew` is gated on `cfg.asymmetry_jitter` being truthy, so a config that never sets it must
    consume the exact same RNG sequence as one that omits the field entirely.

    """
    common = {"img_size": 128, "min_objects": 4, "max_objects": 4, "rotate": True}
    default_config = SyntheticConfig(**common)
    explicit_zero_config = SyntheticConfig(asymmetry_jitter=0.0, **common)
    default_sample = SyntheticGenerator(default_config).sample(np.random.default_rng(42))
    explicit_sample = SyntheticGenerator(explicit_zero_config).sample(np.random.default_rng(42))
    assert np.array_equal(default_sample.image, explicit_sample.image)


def test_asymmetry_jitter_never_skews_a_circle() -> None:
    """`circle` is excluded from skew even when `asymmetry_jitter` is set, matching its rotation exclusion.

    A circle never rotates either (it is rotation-invariant by design), so an always-unrotated skew would bias every
    drawn circle toward the same absolute image direction instead of varying with a random orientation like every other
    shape's skew does.

    """
    config = SyntheticConfig(
        img_size=128,
        min_objects=6,
        max_objects=6,
        shapes=(PrimitiveShape.CIRCLE,),
        asymmetry_jitter=0.45,
    )
    sample = SyntheticGenerator(config).sample(np.random.default_rng(9))
    for ann in sample.annotations:
        poly = np.array(ann.polygon).reshape(-1, 2)
        center_x = (min(x for x, _ in poly.tolist()) + max(x for x, _ in poly.tolist())) / 2.0
        center_y = (min(y for _, y in poly.tolist()) + max(y for _, y in poly.tolist())) / 2.0
        radii = np.hypot(poly[:, 0] - center_x, poly[:, 1] - center_y)
        assert radii.max() - radii.min() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(DEFAULT_SHAPES, id="geometric"),
        pytest.param(tuple(AnimalShape)[:6], id="animals"),
        pytest.param(tuple(SymbolShape), id="symbols"),
        pytest.param(tuple(LetterShape)[:12], id="letters"),
    ],
)
def test_exported_obb_replaces_its_own_shape(shapes: tuple) -> None:  # type: ignore[type-arg]
    """Every exported `obb_corners` turns its shape's reference outline back onto the object it annotates.

    An OBB annotation is only worth anything if a consumer can read a pose out of it, so this is that read performed
    literally: take the upright reference outline for the annotation's own class, turn and scale and place it by what
    the four exported corners say, and require it to land on the exported polygon. Unlike the geometry-level version in
    `test_geometry.py`, this runs on what the generator actually hands out — so it also covers the placement path that
    produced both fields, and would catch a polygon and a box that were derived from different transforms of the shape.
    The guarantee is for unskewed shapes: `asymmetry_jitter` stays at its `0.0` default here on purpose, since a skewed
    instance is no longer a rigid copy of its reference and could not be rebuilt from one.

    """
    config = SyntheticConfig(
        img_size=256,
        min_objects=3,
        max_objects=5,
        min_size_ratio=0.15,
        max_size_ratio=0.3,
        rotate=True,
        class_mode=ClassMode.SHAPE,
        shapes=shapes,
    )
    # Enough images that every shape in the roster is drawn at several unrelated angles: a tie-break that only flips
    # over part of the circle would otherwise slip through a handful of poses untouched.
    annotations = [ann for sample in SyntheticGenerator(config).generate(24, seed=7) for ann in sample.annotations]
    assert annotations, "the placement budget produced nothing to check"
    worst = max(
        rebuild_error(
            ann.class_name,
            np.array(ann.obb_corners).reshape(-1, 2),
            np.array(ann.polygon).reshape(-1, 2),
        )
        for ann in annotations
    )
    assert worst < 1e-6, f"worst rebuild is {worst:.3g}px off its own annotation"


def test_asymmetry_jitter_keeps_base_landmarks_on_the_skewed_symbol() -> None:
    """A symbol's `base_left`/`base_right` landmarks stay inside its own outline once both are skewed.

    `_attempt_placement` draws one `skew` value and passes it to both `shape_outline` and `symbol_keypoints`; if a
    future edit ever threaded a different value to one of the two calls, a landmark that moves with skew (unlike the on-
    axis `center`) would drift outside the outline it is meant to annotate.

    """
    config = SyntheticConfig(
        img_size=256,
        min_objects=1,
        max_objects=1,
        min_size_ratio=0.4,
        max_size_ratio=0.4,
        task=Task.KEYPOINTS,
        shapes=(SymbolShape.HOUSE,),
        asymmetry_jitter=0.45,
        rotate=False,
    )
    sample = SyntheticGenerator(config).sample(np.random.default_rng(3))
    (ann,) = sample.annotations
    poly = np.array(ann.polygon).reshape(-1, 2)
    triples = dict(zip(SYMBOL_KEYPOINT_NAMES, ann.keypoints, strict=True))
    for name in ("base_left", "base_right"):
        x, y, visibility = triples[name]
        assert visibility == 2
        assert _point_in_polygon((x, y), poly)
