"""Symbol table validity: simplicity, unit convention, landmark schema, and placement."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from synth_datasets.core.keypoints import KeypointSchema
from synth_datasets.families import shape_outline
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import (
    SYMBOL_KEYPOINT_FLIP_IDX,
    SYMBOL_KEYPOINT_NAMES,
    SYMBOL_KEYPOINT_SCHEMA,
    SYMBOL_KEYPOINT_SKELETON,
    SYMBOL_KEYPOINTS,
    SYMBOL_NAMES,
    SYMBOL_POLYGONS,
    SymbolShape,
    symbol_keypoints,
)

#: Which `SYMBOL_KEYPOINT_NAMES` slots each symbol actually uses, hand-reviewed and pinned as a literal — see the
#: module docstring's per-shape table in `symbols.py`. Only `center` (index 0) is guaranteed across every symbol.
PINNED_PRESENT_SLOTS: dict[str, frozenset[str]] = {
    "kite": frozenset({"center", "apex", "tail", "flank_left", "flank_right"}),
    "trapezoid": frozenset({"center", "flank_left", "flank_right", "base_left", "base_right"}),
    "house": frozenset({"center", "apex", "flank_left", "flank_right", "base_left", "base_right"}),
    "arrow": frozenset({"center", "apex", "tail", "flank_left", "flank_right"}),
    "cross": frozenset({"center", "apex", "tail", "flank_left", "flank_right"}),
    "teardrop": frozenset({"center", "apex", "tail", "flank_left", "flank_right"}),
    "anchor": frozenset({"center", "apex", "tail", "flank_left", "flank_right"}),
}


def _area_centroid(points: np.ndarray) -> np.ndarray:
    """Independent shoelace-formula centroid, kept separate from `landmarks._polygon_centroid`."""
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def _cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the z-component of `(a - origin) x (b - origin)`."""
    return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))


def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """Whether collinear point `r` lies within the bounding box of segment `p`-`q`."""
    return bool(min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Whether closed segments `p1p2` and `p3p4` share at least one point (CLRS predicate)."""
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 < 0 and d3 * d4 < 0:
        return True
    collinear_hits = (
        (d1 == 0 and _on_segment(p3, p4, p1)),
        (d2 == 0 and _on_segment(p3, p4, p2)),
        (d3 == 0 and _on_segment(p1, p2, p3)),
        (d4 == 0 and _on_segment(p1, p2, p4)),
    )
    return any(collinear_hits)


def _self_intersections(points: np.ndarray) -> list[tuple[int, int]]:
    """Return every pair of non-adjacent closed-polygon edge indices that touch or cross."""
    count = len(points)
    edges = [(points[i], points[(i + 1) % count]) for i in range(count)]
    pairs = [(i, j) for i in range(count) for j in range(i + 2, count - (1 if i == 0 else 0))]
    return [(i, j) for i, j in pairs if _segments_intersect(*edges[i], *edges[j])]


def _rasterize(name: str, size: float, canvas: int, angle: float = 0.0) -> np.ndarray:
    """Fill one symbol outline into a boolean canvas exactly as the generator would."""
    image = Image.new("L", (canvas, canvas), 0)
    poly = shape_outline(name, center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    return np.asarray(image) > 0


def _rounded_pixel_is_filled(mask: np.ndarray, x: float, y: float) -> bool:
    """Whether the rounded `(x, y)` lands on a filled pixel of `mask`."""
    row, col = round(float(y)), round(float(x))
    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
        return False
    return bool(mask[row, col])


def test_every_symbol_member_has_a_table() -> None:
    """`SymbolShape` and the loaded outline/keypoint tables are the same set, in both directions."""
    assert {shape.value for shape in SymbolShape} == set(SYMBOL_POLYGONS)
    assert {shape.value for shape in SymbolShape} == set(SYMBOL_KEYPOINTS)


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_table_is_a_simple_polygon(name: str) -> None:
    """No two non-adjacent outline edges touch or cross.

    A self-intersecting outline renders as an unpredictable bow-tie under Pillow's fill and makes the segmentation mask
    disagree with the rasterized pixels, so simplicity is the load-bearing invariant of a hand-authored table.

    """
    assert _self_intersections(SYMBOL_POLYGONS[name]) == []


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_table_has_no_repeated_vertex(name: str) -> None:
    """Every outline vertex is distinct, so the polygon has no pinch point."""
    unique = np.unique(SYMBOL_POLYGONS[name], axis=0)
    assert len(unique) == len(SYMBOL_POLYGONS[name])


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_table_area_centroid_is_the_origin(name: str) -> None:
    """Every outline is centred on its own area centroid, the unit-space invariant `_normalized_pair` enforces.

    Not the vertex mean: several symbols (the arrow's barbs, the anchor's flukes) bunch vertices
    unevenly, so only the area centroid (center of mass) reliably lands on the origin.

    """
    assert np.allclose(_area_centroid(SYMBOL_POLYGONS[name]), [0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_table_larger_extent_is_unit(name: str) -> None:
    """Every outline's larger extent is exactly `1`, the scale `_base_polygon` multiplies by `size`."""
    poly = SYMBOL_POLYGONS[name]
    extent = (poly.max(axis=0) - poly.min(axis=0)).max()
    assert extent == pytest.approx(1.0)


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_keypoint_table_holds_seven_points(name: str) -> None:
    """Each symbol carries exactly one `(x, y)` per schema name.

    YOLO's pose format declares a single dataset-wide `kpt_shape`, so a table of a different length cannot be
    represented at all — it has to be caught here.

    """
    assert SYMBOL_KEYPOINTS[name].shape == (len(SYMBOL_KEYPOINT_NAMES), 2)


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_center_is_never_absent(name: str) -> None:
    """`center` (index 0) is the one slot every symbol must carry a real value for."""
    center_x, _center_y = SYMBOL_KEYPOINTS[name][0]
    assert not np.isnan(center_x)


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_present_slots_match_the_pinned_matrix(name: str) -> None:
    """Each symbol's present-vs-absent slot set matches a literal, hand-reviewed pin, not just "whatever came out".

    The absent-slot matrix is decided per symbol when its raw vertex/landmark literal is authored — pinning it as a
    literal means an edit that silently changes which symbol has, say, a `tail` is a diff a reviewer sees.

    """
    table = SYMBOL_KEYPOINTS[name]
    present = {key for key, (x, _y) in zip(SYMBOL_KEYPOINT_NAMES, table, strict=True) if not np.isnan(x)}
    assert present == PINNED_PRESENT_SLOTS[name]


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_keypoints_are_distinct_positions(name: str) -> None:
    """No two *present* landmarks of a symbol sit on the same point.

    Two coincident landmarks would train a model on contradictory targets for two different names.

    """
    table = SYMBOL_KEYPOINTS[name]
    present = [(round(float(x), 6), round(float(y), 6)) for x, y in table if not np.isnan(x)]
    assert len(set(present)) == len(present)


@pytest.mark.parametrize("angle", [0.0, 0.9, -1.4, 2.6])
@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_keypoints_lie_inside_the_rasterized_silhouette(name: str, angle: float) -> None:
    """Every landmark falls on a filled pixel of the symbol it annotates, at any rotation.

    Every present landmark here is deliberately pulled inward from its outline vertex (see `symbols.py`'s `_RAW`
    docstring) rather than placed exactly on the boundary, precisely so this holds at every angle — a rasterizer's
    scanline fill can exclude the exact boundary pixel from whichever direction the fill sweeps, and which vertex that
    is changes with the rotation angle.

    """
    size, canvas = 200.0, 260
    mask = _rasterize(name, size, canvas, angle=angle)
    points = symbol_keypoints(SymbolShape(name), center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    outside = [
        key
        for key, (x, y) in zip(SYMBOL_KEYPOINT_NAMES, points, strict=True)
        if not np.isnan(x) and not _rounded_pixel_is_filled(mask, x, y)
    ]
    assert outside == []


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_keypoint_table_is_frozen_against_mutation(name: str) -> None:
    """The shared landmark table rejects in-place writes."""
    with pytest.raises(ValueError, match="read-only"):
        SYMBOL_KEYPOINTS[name][0, 0] = 99.0


@pytest.mark.parametrize("name", SYMBOL_NAMES)
def test_placed_keypoints_do_not_alias_the_table(name: str) -> None:
    """`symbol_keypoints` returns a fresh writable array rather than a view of the constant."""
    points = symbol_keypoints(SymbolShape(name), center=(0.0, 0.0), size=10.0)
    points[0, 0] += 1.0  # must not raise
    assert points.base is not SYMBOL_KEYPOINTS[name]


def test_symbol_keypoints_rejects_a_shape_without_a_table() -> None:
    """A geometric shape (no keypoint table at all) is refused with a clear message."""
    with pytest.raises(ValueError, match="no keypoint table"):
        symbol_keypoints(PrimitiveShape.SQUARE, center=(0.0, 0.0), size=10.0)


def test_flip_idx_is_pinned() -> None:
    """`SYMBOL_KEYPOINT_FLIP_IDX` is the literal left/right swap the module docstring documents.

    `center`/`apex`/`tail` (indices 0-2) sit on the symmetry axis and map to themselves; `flank_left`/`flank_right` (3,
    4) and `base_left`/`base_right` (5, 6) swap.

    """
    assert SYMBOL_KEYPOINT_FLIP_IDX == (0, 1, 2, 4, 3, 6, 5)


def test_skeleton_is_a_star_from_center() -> None:
    """Every skeleton edge connects `center` (index 0) to exactly one other slot, each exactly once.

    A star topology is what makes every optional slot a leaf: an absent slot drops exactly its own edge and orphans
    nothing, the same property the animal skeleton relies on.

    """
    assert sorted(SYMBOL_KEYPOINT_SKELETON) == [(0, i) for i in range(1, len(SYMBOL_KEYPOINT_NAMES))]


def test_symbol_keypoint_schema_bundles_the_family() -> None:
    """`SYMBOL_KEYPOINT_SCHEMA` carries the same names/skeleton/flip_idx as the standalone module constants."""
    assert (
        KeypointSchema(
            names=SYMBOL_KEYPOINT_NAMES,
            skeleton=SYMBOL_KEYPOINT_SKELETON,
            flip_idx=SYMBOL_KEYPOINT_FLIP_IDX,
            shape_values=SYMBOL_NAMES,
        )
        == SYMBOL_KEYPOINT_SCHEMA
    )


def test_every_symbol_svg_carries_a_matching_skeleton_group() -> None:
    """Each packaged symbol SVG holds a `skeleton` group whose lines connect `center` to every other keypoint.

    The animal and letter assets both show their edge topology in the file itself (a `skeleton` group, the letters'
    `strokes` graph), so a human opening any asset sees shape, dots, and edges together; this pins that the symbol
    family stays consistent with them. The group is a visualization aid regenerated by the editor, never read by the
    loader — the check guards the authored files, not runtime behavior.

    """
    import xml.etree.ElementTree as ET

    from synth_datasets.families.svgio import svg_tag, zoo_attr
    from synth_datasets.families.symbols import _ASSET

    for name in SYMBOL_NAMES:
        root = ET.fromstring((_ASSET / f"{name}.svg").read_text())  # noqa: S314 - our own packaged asset
        circles_group = root.find(f"{svg_tag('g')}[@id='keypoints']")
        assert circles_group is not None, name
        coords = {
            circle.get(zoo_attr("name")): (float(circle.get("cx")), float(circle.get("cy")))
            for circle in circles_group.findall(svg_tag("circle"))
        }
        skeleton = root.find(f"{svg_tag('g')}[@id='skeleton']")
        assert skeleton is not None, f"{name}.svg has no skeleton group"
        edges = {
            ((float(line.get("x1")), float(line.get("y1"))), (float(line.get("x2")), float(line.get("y2"))))
            for line in skeleton.findall(svg_tag("line"))
        }
        expected = {(coords["center"], coords[kpt]) for kpt in coords if kpt != "center"}
        assert edges == expected, f"{name}.svg skeleton lines do not match its keypoints"
