"""Annotation construction-time validation of the landmark table against its schema."""

from __future__ import annotations

import pytest

from synth_datasets.core.keypoints import KeypointSchema
from synth_datasets.core.sample import Annotation
from synth_datasets.families.animals import ANIMAL_KEYPOINT_SCHEMA
from synth_datasets.families.letters import LETTER_KEYPOINT_SCHEMA
from synth_datasets.families.symbols import SYMBOL_KEYPOINT_SCHEMA

_BOX = (0.0, 0.0, 10.0, 10.0)
_POLYGON = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]


def _annotation(
    keypoints: tuple[tuple[float, float, int], ...] | None,
    schema: KeypointSchema | None = ANIMAL_KEYPOINT_SCHEMA,
) -> Annotation:
    """Build an annotation whose only interesting fields are its landmark table and that table's schema."""
    return Annotation(
        class_id=4,
        class_name="duck",
        polygon=_POLYGON,
        bbox_xyxy=_BOX,
        keypoints=keypoints,
        keypoint_schema=None if keypoints is None else schema,
    )


def _table(schema: KeypointSchema, visibility: int = 2) -> tuple[tuple[float, float, int], ...]:
    """Return a full-length table for ``schema`` with every landmark carrying the same visibility flag."""
    return tuple((1.0, 2.0, visibility) for _ in schema.names)


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(ANIMAL_KEYPOINT_SCHEMA, id="animals-16"),
        pytest.param(SYMBOL_KEYPOINT_SCHEMA, id="symbols-7"),
        pytest.param(LETTER_KEYPOINT_SCHEMA, id="letters-15"),
    ],
)
def test_accepts_a_table_matching_its_schema(schema: KeypointSchema) -> None:
    """A full-width table for any registered family is stored verbatim.

    This is the shape the generator emits for every `Task.KEYPOINTS` object, so the guard must let the normal case
    through untouched — a validator that also rewrote or reordered the table would break the landmark-to-name
    correspondence it exists to protect. Every family is covered because the annotation now carries the schema it was
    built against, so no family is privileged by being the one whose width is assumed.

    """
    table = _table(schema)

    ann = _annotation(table, schema)

    assert ann.keypoints == table
    assert ann.keypoint_schema is schema


def test_accepts_no_table_at_all() -> None:
    """`keypoints=None` stays the documented "not a keypoints task" state and is not validated.

    Every non-keypoint task constructs annotations this way, so `None` must not be mistaken for a zero-length table and
    rejected.

    """
    assert _annotation(None).keypoints is None


def test_rejects_a_table_with_no_schema() -> None:
    """Landmarks without a schema are refused rather than guessed at from their width.

    The width used to be the identifier: a 16-triple table meant animals, 7 meant symbols, 15 meant letters. That only
    worked while the registered families happened to have distinct counts, and it left the annotation unable to say
    which family it belonged to — so a writer had to be told separately, and the two could disagree. An annotation now
    either carries its schema or is not a keypoint annotation.

    """
    with pytest.raises(ValueError, match="without a keypoint_schema"):
        Annotation(
            class_id=4,
            class_name="duck",
            polygon=_POLYGON,
            bbox_xyxy=_BOX,
            keypoints=_table(ANIMAL_KEYPOINT_SCHEMA),
        )


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="single-triple"),
        pytest.param(len(ANIMAL_KEYPOINT_SCHEMA.names) - 1, id="one-short"),
        pytest.param(len(ANIMAL_KEYPOINT_SCHEMA.names) + 1, id="one-too-many"),
        # 15 is the letter schema's own width. Under the old length-keyed lookup this count was
        # *accepted* for an animal annotation, which is why the case had to be written against the
        # symbol width instead; checked against a named schema it is simply wrong, as it always was.
        pytest.param(len(LETTER_KEYPOINT_SCHEMA.names), id="another-familys-width"),
    ],
)
def test_rejects_a_table_of_the_wrong_length(count: int) -> None:
    """A table that is not exactly one triple per schema name raises at construction.

    A wrong-length table is read positionally downstream — a COCO `keypoints` array of any length still parses and a
    YOLO pose row is split by position — so every landmark past the gap would be silently attributed to the wrong name
    rather than failing.

    """
    with pytest.raises(ValueError, match="to match the schema"):
        _annotation(tuple((1.0, 2.0, 2) for _ in range(count)))


@pytest.mark.parametrize(
    "visibility",
    [
        pytest.param(3, id="above-range"),
        pytest.param(-1, id="negative"),
        pytest.param(255, id="pixel-value"),
        pytest.param(1.0, id="float-equal-to-a-valid-flag"),
        pytest.param(True, id="bool-equal-to-a-valid-flag"),
    ],
)
def test_rejects_a_visibility_outside_the_coco_flags(visibility: float | bool) -> None:
    """A visibility that is not one of COCO's 0/1/2 flags raises and names the offending landmark.

    Visibility is an index into a three-value scale, not a magnitude; a stray value passes straight through the writers
    into the emitted dataset, where a consumer reads it as an unknown flag. `1.0` and `True` are included because both
    compare equal to the int `1` in a plain membership check, so a validator that only checks membership would let them
    through — `True` would then reach the YOLO writer and serialize as the word "True" instead of a numeric flag.

    """
    with pytest.raises(ValueError, match="visibility"):
        _annotation(_table(ANIMAL_KEYPOINT_SCHEMA, visibility))


@pytest.mark.parametrize("visibility", [0, 1, 2])
def test_accepts_every_coco_visibility_flag(visibility: int) -> None:
    """All three COCO flags are valid, including the `1` the generator never emits.

    The generator only produces 0 and 2, but 1 ("labeled but not visible") is part of the COCO schema; rejecting it
    would refuse a legitimate table built by hand or by a downstream tool.

    """
    table = _table(ANIMAL_KEYPOINT_SCHEMA, visibility)

    assert _annotation(table).keypoints == table


def test_names_the_offending_landmark_in_the_error() -> None:
    """The error identifies which landmark carries the bad flag, not just that one does.

    With 16 same-shaped triples, an error that only says "a visibility is wrong" leaves the caller to bisect the table
    by hand.

    """
    valid = _table(ANIMAL_KEYPOINT_SCHEMA)
    table = (*valid[:2], (1.0, 2.0, 9), *valid[3:])

    with pytest.raises(ValueError, match=ANIMAL_KEYPOINT_SCHEMA.names[2]):
        _annotation(table)


def test_obb_corners_is_derived_from_the_polygon() -> None:
    """The oriented box comes from the polygon on access, matching what a direct computation gives.

    It used to be a constructor field the generator filled for every object, which meant detection, segmentation and
    keypoint runs all paid for a per-object box derivation and then never read the result — measured at 75% of
    generation time when it was a convex-hull scan. Deriving it must produce the identical box, or the saving came at
    the cost of the answer.

    """
    from synth_datasets.families import shape_outline
    from synth_datasets.families.geometry import polygon_to_obb

    outline = shape_outline("rectangle", center=(50.0, 50.0), size=30.0, angle=0.7)
    ann = Annotation(0, "rectangle", [float(v) for v in outline.reshape(-1)], (0.0, 0.0, 100.0, 100.0), angle=0.7)

    expected = [float(v) for v in polygon_to_obb(outline, 0.7).reshape(-1)]

    assert ann.obb_corners == expected


def test_obb_corners_is_computed_once_and_cached() -> None:
    """Repeated access returns the same list object rather than recomputing the hull each time.

    A caller iterating annotations to write an OBB dataset touches this per object; recomputing per access would undo
    the saving for the one task that does read it.

    """
    ann = Annotation(0, "square", _POLYGON, _BOX)

    assert ann.obb_corners is ann.obb_corners
