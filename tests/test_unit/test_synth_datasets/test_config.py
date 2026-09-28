"""Config validation and class-vocabulary tests (no Pillow required)."""

from __future__ import annotations

import pytest

from synth_datasets.core.config import (
    DEFAULT_COLORS,
    DEFAULT_SHAPES,
    ClassMode,
    Color,
    Fill,
    SplitRatios,
    SyntheticConfig,
    Task,
    class_id,
    class_names,
    class_vocabulary,
)
from synth_datasets.families import ALL_SHAPES
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SymbolShape


def test_split_ratios_default_sums_to_one() -> None:
    """Default split ratios sum to one."""
    assert SplitRatios().to_dict() == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_split_ratios_drop_zero_split() -> None:
    """Zero split is dropped from dictionary."""
    assert SplitRatios(0.8, 0.2, 0.0).to_dict() == {"train": 0.8, "val": 0.2}


def test_split_ratios_reject_bad_sum() -> None:
    """Invalid split ratios that don't sum to 1 are rejected."""
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(0.5, 0.2, 0.1)


def test_split_ratios_reject_negative() -> None:
    """Negative split ratios are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        SplitRatios(1.2, -0.2, 0.0)


def test_synthetic_config_rejects_inverted_object_range() -> None:
    """Inverted min/max object range is rejected."""
    with pytest.raises(ValueError, match="min_objects <= max_objects"):
        SyntheticConfig(min_objects=5, max_objects=2)


def test_synthetic_config_rejects_inverted_size_ratio() -> None:
    """Inverted min/max size ratio is rejected."""
    with pytest.raises(ValueError, match="min_size_ratio"):
        SyntheticConfig(min_size_ratio=0.4, max_size_ratio=0.2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"overlap_iou": -0.1}, "overlap_iou"),
        ({"overlap_iou": 1.5}, "overlap_iou"),
        ({"boundary_tolerance": -0.2}, "boundary_tolerance"),
        ({"boundary_tolerance": 2.0}, "boundary_tolerance"),
        ({"max_placement_attempts": 0}, "max_placement_attempts"),
        ({"max_placement_attempts": -5}, "max_placement_attempts"),
        ({"asymmetry_jitter": -0.1}, "asymmetry_jitter"),
        ({"asymmetry_jitter": 0.5}, "asymmetry_jitter"),
        ({"asymmetry_jitter": 0.8}, "asymmetry_jitter"),
    ],
)
def test_synthetic_config_rejects_out_of_range_placement_knobs(kwargs: dict[str, float], match: str) -> None:
    """Out-of-range placement knobs are rejected."""
    with pytest.raises(ValueError, match=match):
        SyntheticConfig(**kwargs)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_synthetic_config_accepts_boundary_values(value: float) -> None:
    """Boundary values for overlap and tolerance are accepted."""
    config = SyntheticConfig(overlap_iou=value, boundary_tolerance=value, max_placement_attempts=1)
    assert config.overlap_iou == value
    assert config.max_placement_attempts == 1


@pytest.mark.parametrize("value", [0.0, 0.15, 0.499])
def test_synthetic_config_accepts_asymmetry_jitter_within_range(value: float) -> None:
    """`asymmetry_jitter` accepts its full half-open range `[0, 0.5)`, including both edges tested."""
    assert SyntheticConfig(asymmetry_jitter=value).asymmetry_jitter == value


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ClassMode.SHAPE, [s.value for s in (*PrimitiveShape, *AnimalShape, *SymbolShape, *LetterShape)]),
        (ClassMode.COLOR, [c.value for c in Color]),
    ],
)
def test_class_names(mode: ClassMode, expected: list[str]) -> None:
    """Class names match expected values for mode."""
    assert class_names(mode, ALL_SHAPES) == expected


def test_shape_class_vocabulary_is_pinned_in_order() -> None:
    """The 49 shape class names, pinned as a literal in their exact class-id order.

    A class id is this list's index, and seeded runs are documented as byte-identical across releases, so reordering or
    renaming a member silently relabels every previously exported dataset. Derived assertions cannot catch that — only a
    literal can.

    """
    assert class_names(ClassMode.SHAPE, ALL_SHAPES) == [
        "square",
        "rectangle",
        "triangle",
        "circle",
        "duck",
        "elephant",
        "giraffe",
        "fish",
        "rabbit",
        "camel",
        "eagle",
        "penguin",
        "whale",
        "kangaroo",
        "flamingo",
        "crocodile",
        "kite",
        "trapezoid",
        "house",
        "arrow",
        "cross",
        "teardrop",
        "anchor",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]


def test_class_names_shape_color_product_size() -> None:
    """Shape-color class count is product of shape and color counts."""
    assert len(class_names(ClassMode.SHAPE_COLOR, ALL_SHAPES)) == (
        len(PrimitiveShape) + len(AnimalShape) + len(SymbolShape) + len(LetterShape)
    ) * len(Color)


def test_class_names_shape_color_pins_shape_major_color_minor_order() -> None:
    """Shape-color class names follow shape-major, color-minor order.

    SHAPE_COLOR carries the same class-id-is-list-index guarantee as SHAPE (see class_names docstring): reordering
    silently relabels every exported dataset. A count-only check would still pass if shape-major/color-minor nesting
    changed, so pin the actual sequence for the first 12 (pre-animal-shape) ids.

    """
    assert class_names(ClassMode.SHAPE_COLOR, ALL_SHAPES)[:12] == [
        "red_square",
        "green_square",
        "blue_square",
        "red_rectangle",
        "green_rectangle",
        "blue_rectangle",
        "red_triangle",
        "green_triangle",
        "blue_triangle",
        "red_circle",
        "green_circle",
        "blue_circle",
    ]


def test_class_names_shapes_param_narrows_shape_mode_to_the_given_family() -> None:
    """Passing `shapes=DEFAULT_SHAPES` under `ClassMode.SHAPE` returns just the 4 geometric names.

    Golden fixtures that pin a category count (e.g. a frozen detection-task output) need a vocabulary that stays a
    stable 4 regardless of how many animal shapes later get added to the full `Shape` union; the opt-in `shapes`
    parameter is how they ask for that without touching `class_id`'s full-vocabulary behavior.

    """
    assert class_names(ClassMode.SHAPE, DEFAULT_SHAPES) == ["square", "rectangle", "triangle", "circle"]


def test_class_names_shapes_param_narrows_shape_color_mode() -> None:
    """Passing `shapes` under `ClassMode.SHAPE_COLOR` restricts the product to that shape family.

    Mirrors the `ClassMode.SHAPE` narrowing but exercises the shape-major/color-minor product path, which builds its
    class names from the same `universe` rather than the module-level `(*PrimitiveShape, *AnimalShape)` tuple.

    """
    assert class_names(ClassMode.SHAPE_COLOR, (AnimalShape.DUCK,)) == ["red_duck", "green_duck", "blue_duck"]


def test_class_names_shapes_param_ignored_under_color_mode() -> None:
    """`ClassMode.COLOR` returns the same 3 colors whether or not `shapes` is passed.

    `ClassMode.COLOR` never depends on the shape vocabulary, so a caller narrowing `shapes` for a `SHAPE`-mode fixture
    and reusing the same argument for a `COLOR`-mode one must not see it silently drop colors.

    """
    assert class_names(ClassMode.COLOR, DEFAULT_SHAPES) == class_names(ClassMode.COLOR, ALL_SHAPES)


def test_class_names_requires_an_explicit_shape_vocabulary() -> None:
    """`shapes` has no default, so "which vocabulary" is always stated rather than assumed.

    It used to default to the full 49-shape span while `SyntheticConfig` defaulted to the four primitives, so the
    obvious call for a default config returned a vocabulary twelve times too large — silently, since the ids still
    resolved against it. Requiring the argument removes the mismatch instead of documenting it.

    """
    with pytest.raises(TypeError, match="shapes"):
        class_names(ClassMode.SHAPE)  # type: ignore[call-arg]


def test_class_id_narrows_with_the_same_shapes_class_names_does() -> None:
    """`class_id(..., shapes=)` indexes the vocabulary `class_names(..., shapes=)` returns.

    The two are one contract: a written dataset's `category_id` must resolve against the `categories` block beside it.
    Checked on a symbol, whose global id (16) differs from its narrowed one (0) — a geometric shape cannot show the
    difference, since the four geometric shapes lead the full vocabulary.

    """
    narrowed = (SymbolShape.KITE, SymbolShape.HOUSE)

    assert class_id(SymbolShape.HOUSE, Color.RED, ClassMode.SHAPE, ALL_SHAPES) == class_names(
        ClassMode.SHAPE, ALL_SHAPES
    ).index("house")
    assert class_id(SymbolShape.HOUSE, Color.RED, ClassMode.SHAPE, narrowed) == 1


@pytest.mark.parametrize("mode", list(ClassMode))
def test_a_bare_string_class_mode_selects_the_same_vocabulary_as_its_member(mode: ClassMode) -> None:
    """A plain `"shape"` behaves exactly like `ClassMode.SHAPE` in both vocabulary functions.

    `ClassMode` is a str-Enum, so a bare string compares *and hashes* equal to its member while failing the `is` tests
    both functions branch on — and `class_id`'s per-vocabulary cache is keyed on that equal hash. Left uncoerced,
    whichever spelling ran first won the cache entry and the other silently read the wrong naming out of it, which
    surfaced as an order-dependent `KeyError` from a `SyntheticIterableDataset(class_mode="shape")` stream.

    """
    assert class_names(mode.value, ALL_SHAPES) == class_names(mode, ALL_SHAPES)
    assert class_id(PrimitiveShape.CIRCLE, Color.GREEN, mode.value, ALL_SHAPES) == class_id(
        PrimitiveShape.CIRCLE, Color.GREEN, mode, ALL_SHAPES
    )


def test_config_normalizes_a_bare_string_class_mode_to_the_member() -> None:
    """`SyntheticConfig(class_mode="shape")` stores the `ClassMode` member, not the raw string.

    `SyntheticIterableDataset` forwards `**config_kwargs` verbatim, so the string spelling is public API and reaches the
    config unconverted. Coercing once here is what lets every downstream `is ClassMode.X` branch stay sound.

    """
    config = SyntheticConfig(class_mode="shape")

    assert config.class_mode is ClassMode.SHAPE


def test_class_id_round_trips() -> None:
    """Class ID round-trips work for all modes, shapes, and colors."""
    for mode in ClassMode:
        for shape in (*PrimitiveShape, *AnimalShape, *SymbolShape, *LetterShape):
            for color in Color:
                idx = class_id(shape, color, mode, ALL_SHAPES)
                assert 0 <= idx < len(class_names(mode, ALL_SHAPES))


def test_shapes_defaults_to_the_four_geometric_shapes() -> None:
    """An untouched config still draws only square/rectangle/triangle/circle.

    Appending the animals to `Shape` would otherwise change what every existing seeded caller generates; pinning the
    default here is the guard that kept that upgrade non-breaking.

    """
    assert SyntheticConfig().shapes == (
        PrimitiveShape.SQUARE,
        PrimitiveShape.RECTANGLE,
        PrimitiveShape.TRIANGLE,
        PrimitiveShape.CIRCLE,
    )
    assert SyntheticConfig().shapes == DEFAULT_SHAPES


def test_default_shapes_keeps_the_original_class_ids() -> None:
    """The default vocabulary still occupies class ids 0-3 of the full `Shape` order.

    Datasets generated before the animals landed carry those ids in their label files, so the enum had to grow by
    appending rather than by re-ordering.

    """
    assert [class_id(shape, Color.RED, ClassMode.SHAPE, ALL_SHAPES) for shape in DEFAULT_SHAPES] == [0, 1, 2, 3]


def test_shapes_accepts_an_animal_override() -> None:
    """A custom tuple is stored verbatim and restricts the drawable vocabulary.

    This is the opt-in path documented for animal shapes; the config must not silently widen or reorder what the caller
    asked for.

    """
    config = SyntheticConfig(shapes=(AnimalShape.GIRAFFE, AnimalShape.DUCK))
    assert config.shapes == (AnimalShape.GIRAFFE, AnimalShape.DUCK)


def test_shapes_rejects_an_empty_tuple() -> None:
    """An empty vocabulary is refused at construction rather than at draw time.

    With no shape to sample the generator would fail deep inside the placement loop with an opaque index error, long
    after the real mistake was made.

    """
    with pytest.raises(ValueError, match="at least one Shape"):
        SyntheticConfig(shapes=())


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param((AnimalShape.DUCK, None), id="none"),
        pytest.param((AnimalShape.DUCK, 3), id="int"),
    ],
)
def test_shapes_rejects_non_shape_elements(bad: tuple[object, ...]) -> None:
    """Anything that is neither a `Shape` member nor a shape name is refused.

    A shape *name* is accepted and resolved to its member (see `test_shapes_accepts_shape_names`); what is refused is
    anything else, rather than letting it surface as an opaque lookup failure downstream.

    """
    with pytest.raises(ValueError, match="only Shape members"):
        SyntheticConfig(shapes=bad)


def test_colors_defaults_to_every_color() -> None:
    """An untouched config still draws from all three colors.

    Mirrors `test_shapes_defaults_to_the_four_geometric_shapes`: adding the `colors` field must not change what an
    existing seeded caller generates.

    """
    assert SyntheticConfig().colors == DEFAULT_COLORS
    assert [fill.label for fill in SyntheticConfig().colors] == ["red", "green", "blue"]


def test_colors_accepts_a_restricted_override() -> None:
    """A custom tuple is stored verbatim and restricts the drawable color palette.

    Mirrors `test_shapes_accepts_an_animal_override`: the config must not silently widen or reorder what the caller
    asked for.

    """
    config = SyntheticConfig(colors=(Color.BLUE, Color.RED))
    assert config.colors == (Fill.parse(Color.BLUE), Fill.parse(Color.RED))


def test_colors_rejects_an_empty_tuple() -> None:
    """An empty color palette is refused at construction rather than at draw time.

    Mirrors `test_shapes_rejects_an_empty_tuple`: with no color to sample the generator would otherwise fail deep inside
    the placement loop with an opaque index error.

    """
    with pytest.raises(ValueError, match="at least one Color"):
        SyntheticConfig(colors=())


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param((Color.RED, None), id="none"),
        pytest.param((1, 2, 3), id="bare-ints"),
    ],
)
def test_colors_rejects_non_color_elements(bad: tuple[object, ...]) -> None:
    """Anything that is neither a `Color` member, a colour name, a `Fill`, nor a valid RGB triple is refused.

    Colour names and custom `(r, g, b)` triples are accepted, but widening the field must not turn it into "accept
    anything": `None` or a bare sequence of ints still fails at construction rather than deep in the generator.

    """
    with pytest.raises(ValueError, match="Color member or an"):
        SyntheticConfig(colors=bad)


@pytest.mark.parametrize(
    ("colors", "expected"),
    [
        pytest.param(("red", "blue"), (Color.RED, Color.BLUE), id="names"),
        pytest.param(("Red", "BLUE"), (Color.RED, Color.BLUE), id="any-case"),
        pytest.param((Color.GREEN, "red", Color.BLUE), (Color.GREEN, Color.RED, Color.BLUE), id="mixed-spellings"),
        pytest.param("green", (Color.GREEN,), id="lone-name"),
    ],
)
def test_colors_accept_colour_names(colors: object, expected: tuple[Color, ...]) -> None:
    """A colour name resolves to its `Color` member, so a YAML file or `--colors red,blue` can spell one.

    A lone string is one name rather than a sequence of characters: iterating `"green"` would yield `g r e e n`.

    """
    assert [fill.rgb for fill in SyntheticConfig(colors=colors).colors] == [color.rgb for color in expected]


def test_named_colours_keep_their_class_names() -> None:
    config = SyntheticConfig(class_mode="color", colors=("blue", "red"))
    assert [fill.label for fill in config.colors] == ["blue", "red"]


def test_an_unknown_colour_name_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match=r"unknown colour name 'purple'.*red, green, blue"):
        SyntheticConfig(colors=("red", "purple"))
    with pytest.raises(ValueError, match="unknown colour name '#ff0000'"):
        Fill.parse("#ff0000")


def test_a_colour_name_parses_to_the_member_s_fill_not_a_bare_string() -> None:
    """Under the `str` mixin `"red" == Color.RED`, so a name must become the member's `Fill`, not pass as-is."""
    assert Fill.parse("red") == Fill.parse(Color.RED)


@pytest.mark.parametrize(
    "bad_shapes",
    [
        pytest.param(DEFAULT_SHAPES, id="all-geometric"),
        pytest.param((PrimitiveShape.SQUARE,), id="single-geometric"),
        pytest.param((AnimalShape.DUCK, PrimitiveShape.SQUARE), id="mixed-animal-and-geometric"),
    ],
)
def test_keypoints_task_rejects_shapes_without_a_keypoint_table(bad_shapes: tuple[object, ...]) -> None:
    """`Task.KEYPOINTS` refuses any shape tuple that includes a geometric shape.

    Only `AnimalShape` and `SymbolShape` members carry a landmark table; pairing `Task.KEYPOINTS` with a geometric shape
    (alone or mixed with a keypoint-bearing shape) would otherwise reach the generator and fail deep inside landmark
    lookup instead of at construction.

    """
    with pytest.raises(ValueError, match="have no keypoint table"):
        SyntheticConfig(task=Task.KEYPOINTS, shapes=bad_shapes)


def test_keypoints_task_rejects_shapes_mixing_two_keypoint_families() -> None:
    """`Task.KEYPOINTS` refuses a shape tuple that mixes `AnimalShape` and `SymbolShape` members.

    Each family carries its own fixed-width landmark schema (16 animal names vs. 7 symbol names), and a dataset carries
    exactly one dataset-wide `kpt_shape`/category `keypoints` list, so mixing the two families — even though every shape
    individually has a table — is still unrepresentable and must be rejected at construction.

    """
    with pytest.raises(ValueError, match=r"families.*only one landmark schema"):
        SyntheticConfig(task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK, SymbolShape.KITE))


def test_task_accepts_its_string_form_and_normalizes_it() -> None:
    """`task="keypoints"` is coerced to the member, the same way `class_mode` already was.

    The config used to reject a bare string, which was safe only because `generate_dataset` normalized the task before
    ever constructing a config. With the config the task's sole owner, the string spelling arrives here directly and is
    the documented one — so it is coerced at this boundary, which also means every downstream `is Task.KEYPOINTS` test
    stays valid.

    """
    config = SyntheticConfig(task="keypoints", shapes=tuple(AnimalShape))

    assert config.task is Task.KEYPOINTS


def test_task_rejects_a_value_naming_no_task() -> None:
    """A string that is not a `Task` value is refused at construction rather than downstream.

    Coercion must not become "accept anything": an unknown task would otherwise reach the writer and pick no branch at
    all, emitting a detection dataset for a task the caller never asked for.

    """
    with pytest.raises(ValueError, match="not a valid Task"):
        SyntheticConfig(task="segmentation_v2")


def test_colors_accepts_a_raw_rgb_triple() -> None:
    """A fill may be a plain `(r, g, b)` triple, not only one of the three named colors.

    `background` always took an arbitrary RGB while object fills took only the enum — an asymmetry with no defensible
    reason, since both end up as a Pillow fill. A caller wanting a yellow object had no path at all.

    """
    config = SyntheticConfig(colors=((255, 215, 0),))

    assert config.colors == (Fill(rgb=(255, 215, 0)),)


def test_a_raw_fill_is_labelled_by_its_hex_value() -> None:
    """A custom fill still yields a well-defined class name, derived from its hex value.

    `ClassMode.COLOR` and `ClassMode.SHAPE_COLOR` name classes after the fill, so an unnamed color needs *some* stable
    label — hex is the one choice that neither invents a color name nor collides with the three that have one.

    """
    names = class_names(ClassMode.SHAPE_COLOR, (PrimitiveShape.SQUARE,), ((255, 215, 0), Color.RED))

    assert names == ["ffd700_square", "red_square"]


@pytest.mark.parametrize("mode", [ClassMode.COLOR, ClassMode.SHAPE_COLOR])
def test_color_classes_reject_distinct_labels_for_the_same_rgb(mode: ClassMode) -> None:
    """A named and raw fill with identical pixels cannot be separate classes."""
    colors = (Color.RED, (255, 0, 0))

    with pytest.raises(ValueError, match="duplicate RGB"):
        class_vocabulary(mode, (PrimitiveShape.SQUARE,), colors)
    with pytest.raises(ValueError, match="duplicate RGB"):
        SyntheticConfig(class_mode=mode, shapes=(PrimitiveShape.SQUARE,), colors=colors)


@pytest.mark.parametrize(
    ("mode", "shapes", "colors", "reason"),
    [
        pytest.param(
            ClassMode.SHAPE, (PrimitiveShape.SQUARE, PrimitiveShape.SQUARE), (Color.RED,), "duplicate shape", id="shape"
        ),
        pytest.param(ClassMode.COLOR, (PrimitiveShape.SQUARE,), (Color.RED, Color.RED), "duplicate RGB", id="color"),
        pytest.param(
            ClassMode.SHAPE_COLOR,
            (PrimitiveShape.SQUARE, PrimitiveShape.SQUARE),
            (Color.RED,),
            "duplicate shape",
            id="shape-color-shape",
        ),
        pytest.param(
            ClassMode.SHAPE_COLOR,
            (PrimitiveShape.SQUARE,),
            (Color.RED, Color.RED),
            "duplicate RGB",
            id="shape-color-color",
        ),
    ],
)
def test_config_rejects_repeated_class_entries(mode: ClassMode, shapes: tuple, colors: tuple, reason: str) -> None:
    """Repeated class-defining entries must not create unreachable IDs."""
    with pytest.raises(ValueError, match=reason):
        SyntheticConfig(class_mode=mode, shapes=shapes, colors=colors)


def test_class_ignored_pool_can_repeat_as_a_sampling_weight() -> None:
    """A repeated ignored factor changes draw frequency without adding classes."""
    square = PrimitiveShape.SQUARE
    shape_named = SyntheticConfig(class_mode=ClassMode.SHAPE, shapes=(square,), colors=(Color.RED, Color.RED))
    color_named = SyntheticConfig(class_mode=ClassMode.COLOR, shapes=(square, square), colors=(Color.RED,))

    assert shape_named.colors == (Fill.parse(Color.RED), Fill.parse(Color.RED))
    assert class_vocabulary(shape_named.class_mode, shape_named.shapes, shape_named.colors).names == ["square"]
    assert color_named.shapes == (square, square)
    assert class_vocabulary(color_named.class_mode, color_named.shapes, color_named.colors).names == ["red"]


def test_vocabulary_rejects_duplicate_class_names_from_custom_fills() -> None:
    """Two distinct fills cannot publish the same class name and hide one ID."""
    with pytest.raises(ValueError, match="duplicate class name"):
        class_vocabulary(
            ClassMode.COLOR,
            (PrimitiveShape.SQUARE,),
            (Color.RED, Fill(rgb=(1, 2, 3), name="red")),
        )


def test_split_ratios_accepts_arbitrary_split_names() -> None:
    """Splits are not limited to train/val/test; any named set summing to 1 works.

    The three standard names were hardcoded, so a fourth calibration split was impossible and a bare train/test pair had
    to be spelled as `val=0.0`.

    """
    ratios = SplitRatios.custom({"train": 0.6, "calib": 0.2, "test": 0.2})

    assert ratios.to_dict() == {"train": 0.6, "calib": 0.2, "test": 0.2}


def test_custom_split_ratios_still_must_sum_to_one() -> None:
    """Opening up the names does not relax the arithmetic that makes a split a split.

    Fractions that do not sum to 1 would silently drop or duplicate images across the generated splits.

    """
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        SplitRatios.custom({"train": 0.6, "test": 0.2})


def test_color_members_stay_plain_strings_despite_the_rgb_payload() -> None:
    """Carrying RGB through `__new__` must not change what a `Color` member *is*.

    The payload is attached by overriding `__new__`, which is the one place a str-Enum's value could quietly stop being
    its own string. Everything downstream depends on it still being one: class names are rendered from `.value`, a
    member is used as a dict key against plain strings, and a written dataset serializes the name.

    """
    assert Color.RED.value == "red"
    assert Color.RED == "red"
    assert Color("red") is Color.RED
    assert tuple(Color) == (Color.RED, Color.GREEN, Color.BLUE)
    assert (Color.RED.rgb, Color.GREEN.rgb, Color.BLUE.rgb) == ((255, 0, 0), (0, 128, 0), (0, 0, 255))


def test_fill_parse_accepts_every_spelling_and_is_idempotent() -> None:
    """`Fill.parse` is the single boundary the fill union is unpacked at, so it must absorb its own output.

    `SyntheticConfig` normalizes at construction and `class_vocabulary` normalizes again on whatever it is handed —
    typically a config's already-normalized tuple. A `parse` that wrapped or rejected a `Fill` would make that second
    pass either lossy or an error.

    """
    named, raw = Fill.parse(Color.GREEN), Fill.parse((255, 215, 0))

    assert (named.rgb, named.name) == ((0, 128, 0), "green")
    assert (raw.rgb, raw.name) == ((255, 215, 0), None)
    assert Fill.parse(named) is named
    assert Fill.parse(raw) is raw


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param((255, 0), id="two-channels"),
        pytest.param((255, 0, 0, 0), id="four-channels"),
        pytest.param((256, 0, 0), id="channel-above-range"),
        pytest.param((-1, 0, 0), id="negative-channel"),
        pytest.param((255.0, 0, 0), id="float-channel"),
        pytest.param([255, 0, 0], id="list-not-tuple"),
    ],
)
def test_fill_rejects_anything_that_is_not_an_8_bit_triple(bad: object) -> None:
    """A malformed fill raises where it is written, not deep inside the generator's draw call."""
    with pytest.raises(ValueError, match="Color member or an"):
        Fill.parse(bad)  # type: ignore[arg-type]


def test_fill_is_hashable_so_it_can_key_the_vocabulary_cache() -> None:
    """`_build_vocabulary` is `lru_cache`d on its `(mode, shapes, colors)` tuple, so fills must hash.

    An unhashable fill would not fail a type check — it would raise `TypeError` on the first call that builds a
    vocabulary, i.e. on every real run.

    """
    assert Fill.parse(Color.RED) == Fill.parse(Color.RED)
    assert len({Fill.parse(Color.RED), Fill.parse(Color.RED), Fill.parse((255, 215, 0))}) == 2
