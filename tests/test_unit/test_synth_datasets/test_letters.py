"""Letter outline validity: single-polygon simplicity, node coverage, keypoint schema, symmetry."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from synth_datasets.core.keypoints import KeypointSchema
from synth_datasets.families import shape_outline
from synth_datasets.families.letters import (
    LETTER_COUNTER_GAP,
    LETTER_KEYPOINT_FLIP_IDX,
    LETTER_KEYPOINT_NAMES,
    LETTER_KEYPOINT_SCHEMA,
    LETTER_KEYPOINT_SKELETON,
    LETTER_KEYPOINTS,
    LETTER_NAMES,
    LETTER_POLYGONS,
    LETTER_STROKE_WIDTH,
    LetterShape,
    letter_keypoints,
)
from synth_datasets.families.primitives import PrimitiveShape

#: Which `LETTER_KEYPOINT_NAMES` slots each letter actually uses, hand-reviewed and pinned as a
#: literal from the letter's own authored stroke graph — see `letters.json`'s `strokes` table.
PINNED_USED_NODES: dict[str, frozenset[str]] = {
    "a": frozenset({"top_mid", "lower_left", "lower_right", "bottom_left", "bottom_right"}),
    "b": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "upper_right",
        "mid_left",
        "mid_right",
        "lower_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "c": frozenset({"top_left", "top_mid", "top_right", "mid_left", "bottom_left", "bottom_mid", "bottom_right"}),
    "d": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "upper_right",
        "mid_right",
        "lower_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "e": frozenset({"top_left", "top_right", "mid_left", "mid_right", "bottom_left", "bottom_right"}),
    "f": frozenset({"top_left", "top_right", "mid_left", "mid_right", "bottom_left"}),
    "g": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "mid_left",
        "mid_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "h": frozenset({"top_left", "top_right", "mid_left", "mid_right", "bottom_left", "bottom_right"}),
    "i": frozenset({"top_left", "top_mid", "top_right", "bottom_left", "bottom_mid", "bottom_right"}),
    "j": frozenset({
        "top_left",
        "top_right",
        "lower_left",
        "lower_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "k": frozenset({"top_left", "top_right", "mid_left", "bottom_left", "bottom_right"}),
    "l": frozenset({"top_left", "bottom_left", "bottom_right"}),
    "m": frozenset({"top_left", "top_right", "center", "bottom_left", "bottom_right"}),
    "n": frozenset({"top_left", "top_right", "bottom_left", "bottom_right"}),
    "o": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "mid_left",
        "mid_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "p": frozenset({"top_left", "top_mid", "top_right", "upper_right", "mid_left", "mid_right", "bottom_left"}),
    "q": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "mid_left",
        "mid_right",
        "lower_mid",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "r": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "upper_right",
        "mid_left",
        "mid_right",
        "bottom_left",
        "bottom_right",
    }),
    "s": frozenset({
        "top_left",
        "top_mid",
        "top_right",
        "upper_left",
        "center",
        "lower_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    }),
    "t": frozenset({"top_left", "top_mid", "top_right", "bottom_mid"}),
    "u": frozenset({"top_left", "top_right", "lower_left", "lower_right", "bottom_left", "bottom_right"}),
    "v": frozenset({"top_left", "top_right", "bottom_mid"}),
    "w": frozenset({"top_left", "top_mid", "top_right", "lower_left", "lower_right"}),
    "x": frozenset({"top_left", "top_right", "center", "bottom_left", "bottom_right"}),
    "y": frozenset({"top_left", "top_right", "center", "bottom_mid"}),
    "z": frozenset({
        "top_left",
        "top_right",
        "mid_left",
        "center",
        "mid_right",
        "bottom_left",
        "bottom_right",
    }),
}

#: Letters whose regular block form is exactly invariant under a 180-degree turn about its own
#: center, and which therefore place one node off its default grid slot to break that symmetry —
#: see `letters.py`'s module docstring and `letters.json`'s `nodes` table.
SYMMETRY_RISK_LETTERS: tuple[str, ...] = ("b", "d", "h", "i", "n", "o", "s", "x", "z")

#: Letters with an enclosed counter — a cycle in the stroke graph — and therefore a `_CUTS` entry.
#: `b` has two independent bowls, so it needs two cuts; every other one needs exactly one.
COUNTER_LETTERS: dict[str, int] = {"a": 1, "b": 2, "d": 1, "o": 1, "p": 1, "q": 1, "r": 1}


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the z-component of `(a - origin) x (b - origin)`."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


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


def _area_centroid(points: np.ndarray) -> np.ndarray:
    """Independent shoelace-formula centroid, kept separate from `landmarks._polygon_centroid`."""
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def _rasterize(name: str, size: float, canvas: int, angle: float = 0.0) -> np.ndarray:
    """Fill one letter's outline into a boolean canvas exactly as the generator would."""
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


def _point_in_polygon(polygon: np.ndarray, x: float, y: float) -> bool:
    """Whether `(x, y)` lies inside the closed `polygon` ring, by ray casting along +x."""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) / (y2 - y1) * (x2 - x1):
            inside = not inside
    return inside


def _is_180_invariant(table: np.ndarray, edges: frozenset[tuple[int, int]]) -> bool:
    """Whether the letter's nodes and `edges` are together unchanged by a 180-degree turn about its own center.

    A letter free to place its nodes anywhere (see `letters._NODES`) cannot be checked by permuting grid slots, so this
    matches geometrically instead: rotate every used node a half turn about their centroid — the only center such a
    symmetry could have — and require both that each rotated node lands on another used node and that the induced
    permutation maps the edge set onto itself.

    """
    used = sorted({index for edge in edges for index in edge})
    center = table[used].mean(axis=0)
    mapping: dict[int, int] = {}
    for index in used:
        landed = [other for other in used if np.allclose(table[other], 2.0 * center - table[index], atol=1e-9)]
        if not landed:
            return False
        mapping[index] = landed[0]
    return {tuple(sorted((mapping[a], mapping[b]))) for a, b in edges} == edges


def _cycle_rank(name: str) -> int:
    """Return the letter's stroke graph's cycle rank: edges - nodes + connected components."""
    from synth_datasets.families.letters import _LETTER_STROKES

    edges = _LETTER_STROKES[name]
    nodes = {index for edge in edges for index in edge}
    parent = {node: node for node in nodes}

    def _find(node: int) -> int:
        while parent[node] != node:
            node = parent[node]
        return node

    for a, b in edges:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb
    components = len({_find(node) for node in nodes})
    return len(edges) - len(nodes) + components


def test_every_letter_member_has_a_table() -> None:
    """`LetterShape` and the loaded outline/keypoint tables are the same set, in both directions."""
    assert {shape.value for shape in LetterShape} == set(LETTER_POLYGONS)
    assert {shape.value for shape in LetterShape} == set(LETTER_KEYPOINTS)
    assert len(LetterShape) == 26


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_outline_is_a_single_simple_polygon(name: str) -> None:
    """No two non-adjacent outline edges touch or cross.

    A self-intersecting outline renders as an unpredictable bow-tie under Pillow's fill and makes the segmentation mask
    disagree with the rasterized pixels — the same invariant `test_symbols.py`'s equivalent test guards, now load-
    bearing for letters too since each one must be exactly one polygon, not a pile of disjoint stroke ribbons.

    """
    assert _self_intersections(LETTER_POLYGONS[name]) == []


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_outline_area_centroid_is_the_origin(name: str) -> None:
    """Every outline is centred on its own area centroid, the unit-space invariant `_normalized_pair` enforces."""
    assert np.allclose(_area_centroid(LETTER_POLYGONS[name]), [0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_outline_larger_extent_is_unit(name: str) -> None:
    """Every outline's larger extent is exactly `1`, the scale `_base_polygon` multiplies by `size`."""
    poly = LETTER_POLYGONS[name]
    extent = (poly.max(axis=0) - poly.min(axis=0)).max()
    assert extent == pytest.approx(1.0)


@pytest.mark.parametrize(("name", "expected_cuts"), sorted(COUNTER_LETTERS.items()))
def test_counter_letter_cut_count_equals_its_cycle_rank(name: str, expected_cuts: int) -> None:
    """A letter with `n` independent enclosed counters names exactly `n` cuts in `letters.json`.

    Each cut opens exactly one cycle (see `letters.py`'s module docstring); naming fewer than the graph's own cycle rank
    would leave an un-opened counter, which `_stroke_outline` would catch by tracing more than one face — this pins the
    expected count directly instead of relying on that failure mode.

    """
    from synth_datasets.families.letters import _CUTS

    assert len(_CUTS[name]) == expected_cuts == _cycle_rank(name)


@pytest.mark.parametrize("name", [name for name in LETTER_NAMES if name not in COUNTER_LETTERS])
def test_non_counter_letter_has_no_cuts_and_zero_cycle_rank(name: str) -> None:
    """A letter with no enclosed counter has an acyclic stroke graph and no `_CUTS` entry."""
    from synth_datasets.families.letters import _CUTS

    assert name not in _CUTS
    assert _cycle_rank(name) == 0


def test_letter_o_counter_rasterizes_unfilled() -> None:
    """`o`'s center pixel is unfilled: the hairline-slit cut opens the ring without losing the hole."""
    mask = _rasterize("o", size=150.0, canvas=200)
    assert not mask[100, 100]


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_keypoint_table_holds_fifteen_points(name: str) -> None:
    """Each letter carries exactly one `(x, y)` per grid slot.

    YOLO's pose format declares a single dataset-wide `kpt_shape`, so a table of a different length cannot be
    represented at all — it has to be caught here.

    """
    assert LETTER_KEYPOINTS[name].shape == (len(LETTER_KEYPOINT_NAMES), 2)


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_used_nodes_match_the_pinned_matrix(name: str) -> None:
    """Each letter's present-vs-absent grid slot set matches a literal, hand-reviewed pin.

    The used-node set is decided per letter when its stroke graph is authored in `letters.json` — pinning it as a
    literal means an edit that silently drops or adds a node is a diff a reviewer sees, the same guard
    `test_symbols.py`'s `PINNED_PRESENT_SLOTS` gives the symbol family.

    """
    table = LETTER_KEYPOINTS[name]
    present = {key for key, (x, _y) in zip(LETTER_KEYPOINT_NAMES, table, strict=True) if not np.isnan(x)}
    assert present == PINNED_USED_NODES[name]


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_keypoints_are_distinct_positions(name: str) -> None:
    """No two *present* keypoints of a letter sit on the same point.

    Two coincident keypoints would train a model on contradictory targets for two different names.

    """
    table = LETTER_KEYPOINTS[name]
    present = [(round(float(x), 6), round(float(y), 6)) for x, y in table if not np.isnan(x)]
    assert len(set(present)) == len(present)


@pytest.mark.parametrize("angle", [0.0, 0.9, -1.4, 2.6])
@pytest.mark.parametrize("name", LETTER_NAMES)
def test_keypoints_lie_inside_the_rasterized_outline(name: str, angle: float) -> None:
    """Every keypoint falls on a filled pixel of the letter's own drawn outline, at any rotation.

    The rasterized counterpart of `test_skeleton_edges_lie_inside_the_outline` below: wrapping the outline around the
    skeleton leaves every node half a stroke width inside the fill, including a stroke's free end, which sits at the
    center of its own round cap rather than on a flat cap's edge. That clearance is what makes the placement survive
    rasterization at an arbitrary angle — the fragility `test_symbols.py`'s equivalent test guards against for landmarks
    placed near a polygon boundary.

    """
    size, canvas = 200.0, 260
    mask = _rasterize(name, size, canvas, angle=angle)
    points = letter_keypoints(LetterShape(name), center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    outside = [
        key
        for key, (x, y) in zip(LETTER_KEYPOINT_NAMES, points, strict=True)
        if not np.isnan(x) and not _rounded_pixel_is_filled(mask, x, y)
    ]
    assert outside == []


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_skeleton_edges_lie_inside_the_outline(name: str) -> None:
    """Every edge joining two of a letter's keypoints runs entirely inside the letter's own polygon.

    The property the whole design exists for: a viewer connecting a letter's keypoints in skeleton order must trace
    that letter *through its ink*, never across empty space. It holds by construction — the outline is the set of
    points within half a stroke width of the skeleton (`letters._stroke_outline`) — so a failure here means the wrap
    broke, not that a letterform is badly drawn. The band around a cut edge's midpoint is the one documented exception:
    that is where the hairline slit opening a counter crosses, checked for its width by
    `test_counter_slit_is_hairline_next_to_the_stroke_width`.

    """
    from synth_datasets.families.letters import _CUTS, _LETTER_STROKES

    polygon = LETTER_POLYGONS[name]
    points = LETTER_KEYPOINTS[name]
    cuts = _CUTS.get(name, {})
    outside = [
        (a, b, round(t, 3))
        for a, b in _LETTER_STROKES[name]
        for t in np.linspace(0.0, 1.0, 51)
        if not _within_slit(cuts, a, b, t)
        and not _point_in_polygon(polygon, *(points[a] + t * (points[b] - points[a])))
    ]
    assert outside == []


def _within_slit(cuts: dict[tuple[int, int], float], a: int, b: int, t: float) -> bool:
    """Whether ``t`` along edge ``a -> b`` falls in the band a counter-opening slit spans."""
    at = cuts.get((min(a, b), max(a, b)))
    if at is None:
        return False
    return abs(t - (at if a < b else 1.0 - at)) <= 0.1


def test_no_two_letters_share_a_silhouette() -> None:
    """No two letters rasterize to near-identical filled shapes.

    Detection, segmentation, and OBB all see only the silhouette, so two classes drawn as the same shape are two
    contradictory labels for one object — unlearnable, and invisible to every other test here, which checks each letter
    on its own. `d` and `o` were exactly that before their bowls were shaped through `letters.json`'s `nodes` table:
    both traced the same rounded rectangle and differed only in how many nodes were spaced along it.

    """
    grid = 64
    masks = {name: _rasterize(name, grid * 0.9, grid) for name in LETTER_NAMES}
    collisions = [
        (first, second, round(iou, 3))
        for index, first in enumerate(LETTER_NAMES)
        for second in LETTER_NAMES[index + 1 :]
        if (iou := _mask_iou(masks[first], masks[second])) > _SILHOUETTE_IOU_LIMIT
    ]
    assert collisions == []


#: Highest silhouette overlap two different letters may reach — see the two tests either side of it.
_SILHOUETTE_IOU_LIMIT = 0.85
#: Degrees between sampled relative rotations in `test_no_two_letters_share_a_silhouette_at_any_angle`.
#: Fine enough to land within a few degrees of any collision: an overlap peak is broad, because a
#: letter rotated slightly off its worst angle still covers most of the shape it collides with.
_ROTATION_STEP_DEG = 15


def test_no_two_letters_share_a_silhouette_at_any_angle() -> None:
    """No two letters rasterize alike at *any* relative rotation, not merely upright.

    `test_no_two_letters_share_a_silhouette` compares letters standing upright, but the generator draws every object at
    a random angle — so a pair that coincides at some relative rotation is exactly as unlearnable as `d` and `o` were
    standing still, and nothing was checking for it. Two pairs did: an upside-down `v` was `a`'s lambda with the
    crossbar taken away (0.85), and an upside-down `t` was a serifed `i` missing one bar (0.82). `v` now narrows at the
    top, and `i` carries even full-width serifs instead of a narrow top one, so it reads as a whole I-beam rather than
    as a `t` with something added. Both were fixed on the letter that could afford it: dropping `t`'s crossbar to
    lowercase height would have separated the pair just as well, but back when the OBB was the minimum-area box a thin
    cross's near-diamond hull leaned that box 15 to 22 degrees off upright, so the fix landed on `v` and `i` instead.

    Letters are drawn at one shared size, so the comparison is between shapes as the generator would place them; the
    size leaves room for a letter turned onto its diagonal to still fit the canvas.

    """
    grid = 64
    size = grid * 0.62
    angles = range(0, 360, _ROTATION_STEP_DEG)
    turned = {
        (name, deg): _rasterize(name, size, grid, np.radians(float(deg))) for name in LETTER_NAMES for deg in angles
    }
    collisions = [
        (first, second, deg, round(iou, 3))
        for index, first in enumerate(LETTER_NAMES)
        for second in LETTER_NAMES[index + 1 :]
        for deg in angles
        if (iou := _mask_iou(turned[first, 0], turned[second, deg])) > _SILHOUETTE_IOU_LIMIT
    ]
    assert collisions == []


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return the intersection-over-union of two boolean rasterized masks of the same size."""
    union = int(np.logical_or(first, second).sum())
    return 0.0 if union == 0 else float(np.logical_and(first, second).sum()) / union


def test_counter_cuts_sit_at_the_bottom_of_their_counter() -> None:
    """Each counter opens where its own pen stroke would lift, pinned edge and position together.

    Where a counter opens is a letterform decision, not a free choice. A bowl hung off a stem (`b`, `d`, `p`, `r`)
    breaks on the edge leaving that stem and hard against it, so the bowl reads as a curve just touching a vertical
    line. A free-standing ring breaks along its bottom — left of centre for `o`, and bottom-right for `q`, beside where
    its tail joins. `a` breaks at its crossbar, the bottom of its triangular counter. Pinned as a literal, the same way
    `PINNED_USED_NODES` is: the geometry tests confirm any cut set produces a valid single ring, so only this pin can
    catch a cut that is legal but placed somewhere visually wrong.

    """
    from synth_datasets.families.letters import _CUTS

    assert {name: dict(sorted(edges.items())) for name, edges in _CUTS.items()} == {
        "a": {(9, 11): 0.5},  # crossbar — the bottom of A's triangular counter
        "b": {(6, 8): 0.24, (12, 13): 0.42},  # both bowls, as near the stem they leave as fits
        "d": {(12, 13): 0.4},  # bowl leaving the stem's foot
        "o": {(12, 13): 0.5},  # free-standing ring: bottom, left of centre
        "p": {(6, 8): 0.27},
        "q": {(13, 14): 0.5},  # bottom right, beside where the tail joins
        "r": {(6, 8): 0.27},
    }


def test_counter_slit_is_hairline_next_to_the_stroke_width() -> None:
    """The counter-opening slit is an order of magnitude thinner than the stroke it cuts through.

    The slit is the single place a skeleton edge leaves the fill, so its width is the size of that exception. Pinning
    the ratio keeps a future stroke-width change from quietly turning the hairline into a visible pen lift — and keeps
    the two flat-capped stubs closer together than a round cap's own diameter, which is why they must cap flat.

    """
    assert LETTER_COUNTER_GAP <= LETTER_STROKE_WIDTH / 8.0


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_no_letter_is_invariant_under_180_degree_reflection(name: str) -> None:
    """No letter's stroke graph is unchanged by a 180-degree turn about its own center.

    A letter whose nodes and edges map onto themselves under that turn has keypoint-identity ambiguity under the
    generator's continuous random rotation — the risk `letters.py`'s module docstring documents and the free node
    positions in `letters.json`'s `nodes` table exist to break. Checked on the authored geometry
    (`letters.raw_letter_nodes`), not the wrapped outline or its normalization.

    """
    from synth_datasets.families.letters import _LETTER_STROKES, raw_letter_nodes

    edges = frozenset(tuple(sorted(edge)) for edge in _LETTER_STROKES[name])
    assert not _is_180_invariant(raw_letter_nodes(name), edges)


def test_symmetry_risk_letters_matches_the_documented_set() -> None:
    """`SYMMETRY_RISK_LETTERS` is exactly the module docstring's nine letters, no more, no fewer.

    A future edit that adds an accidentally-symmetric letterform (like the `B`/`D` bumps this design
    originally had before nudging) without updating the docstring's claimed set would otherwise pass
    silently, since `test_no_letter_is_invariant_under_180_degree_reflection` only checks the final,
    already-nudged state.

    """
    assert set(SYMMETRY_RISK_LETTERS) == {"b", "d", "h", "i", "n", "o", "s", "x", "z"}


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_skeleton_by_value_covers_every_letter(name: str) -> None:
    """`LETTER_KEYPOINT_SCHEMA.skeleton_for` returns the letter's own edges, not the family fallback.

    Every letter's stroke topology genuinely differs (that is what makes it that letter), so the per-letter
    `skeleton_by_value` override — not the shared `skeleton` fallback — must answer this for every member; see
    `KeypointSchema.skeleton_for`.

    """
    from synth_datasets.families.letters import _LETTER_STROKES

    own_edges = frozenset(tuple(sorted(edge)) for edge in _LETTER_STROKES[name])
    schema_edges = frozenset(tuple(sorted(edge)) for edge in LETTER_KEYPOINT_SCHEMA.skeleton_for(name))
    assert schema_edges == own_edges


def test_keypoint_table_is_frozen_against_mutation() -> None:
    """The shared keypoint table rejects in-place writes."""
    with pytest.raises(ValueError, match="read-only"):
        LETTER_KEYPOINTS["a"][0, 0] = 99.0


def test_outline_table_is_frozen_against_mutation() -> None:
    """The shared outline table rejects in-place writes."""
    with pytest.raises(ValueError, match="read-only"):
        LETTER_POLYGONS["a"][0, 0] = 99.0


@pytest.mark.parametrize("name", LETTER_NAMES)
def test_placed_keypoints_do_not_alias_the_table(name: str) -> None:
    """`letter_keypoints` returns a fresh writable array rather than a view of the constant."""
    points = letter_keypoints(LetterShape(name), center=(0.0, 0.0), size=10.0)
    points[0, 0] += 1.0  # must not raise
    assert points.base is not LETTER_KEYPOINTS[name]


def test_letter_keypoints_rejects_a_shape_without_a_table() -> None:
    """A geometric shape (no keypoint table at all) is refused with a clear message."""
    with pytest.raises(ValueError, match="no keypoint table"):
        letter_keypoints(PrimitiveShape.SQUARE, center=(0.0, 0.0), size=10.0)


def test_flip_idx_is_the_grid_column_mirror() -> None:
    """`LETTER_KEYPOINT_FLIP_IDX` swaps each row's left/right column, keeps the middle column fixed."""
    expected = tuple(3 * (i // 3) + (2 - i % 3 if i % 3 != 1 else 1) for i in range(len(LETTER_KEYPOINT_NAMES)))
    assert expected == LETTER_KEYPOINT_FLIP_IDX
    assert LETTER_KEYPOINT_FLIP_IDX == (2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12)


def test_letter_keypoint_schema_bundles_the_family() -> None:
    """`LETTER_KEYPOINT_SCHEMA` carries the same names/skeleton/flip_idx/shape_values as the module constants."""
    from synth_datasets.families.letters import _LETTER_STROKES

    assert (
        KeypointSchema(
            names=LETTER_KEYPOINT_NAMES,
            skeleton=LETTER_KEYPOINT_SKELETON,
            flip_idx=LETTER_KEYPOINT_FLIP_IDX,
            shape_values=LETTER_NAMES,
            skeleton_by_value=dict(_LETTER_STROKES),
        )
        == LETTER_KEYPOINT_SCHEMA
    )


def test_stroke_width_is_a_positive_fraction() -> None:
    """`LETTER_STROKE_WIDTH` is a sane, positive fraction of the grid's own row span."""
    assert 0.0 < LETTER_STROKE_WIDTH < 0.5
