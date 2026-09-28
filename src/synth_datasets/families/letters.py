"""Analytic capital-letter outlines: a fourth shape family, one polygon per letter.

Twenty-six capital letters, ``A``-``Z``, each authored the way you would sketch one — as a set of
**keypoints and the edges between them** — and then turned into something drawable by wrapping that
skeleton in a pen stroke of constant width (:func:`_stroke_outline`). The result is **one simple,
single-ring polygon** per letter (:data:`LETTER_POLYGONS`), exactly the kind of outline
:mod:`~synth_datasets.families.animals` and :mod:`~synth_datasets.families.symbols` hand-author: a
pile of disjoint ribbon quads, one per stroke, would give a detector several disconnected blobs
under one class label, and a single outline never does.

Authoring skeleton-first rather than outline-first is what makes the landmarks trustworthy. Because
the polygon *is* the set of points within half a stroke width of the skeleton, **every keypoint and
every edge between two keypoints lies strictly inside the letter**, with half a stroke width of
clearance — no keypoint can land on the boundary or outside the ink, and no skeleton edge can cut
across empty space. :data:`LETTER_KEYPOINT_SCHEMA`'s ``skeleton_by_value`` reports exactly the edges
the outline was wrapped around, via :data:`_LETTER_STROKES`, so drawing a letter's own keypoints and
joining them in order sketches that letter inside its own filled shape.

**Round joins and caps.** Every stroke tip is capped with a semicircle and every convex corner
rounded to an arc, so a letter has no sharp point anywhere on it and a tip's node sits at the
center of its own cap rather than on the boundary. Only the concave side of a turn stays angular —
there the two strokes' bodies already cover the corner, and rounding it would bulge the outline
outward across the inside of the turn.

**Straight strokes and arcs.** An edge may itself be curved: :data:`_BULGES` gives it a signed
sagitta as a fraction of its own chord, and :func:`_edge_polyline` samples the resulting circular
arc into short links before the wrap sees it, so the machinery needs no notion of curvature at all.
That is what makes ``o``/``c``/``g``/``s``'s bowls read as bowls rather than octagons, while the
letters that genuinely have no curve in a block face (``a e f h i k l m n t v w x y z``) stay
straight throughout. Two constraints bound how deep an arc may go, both of them consequences of
promises made above rather than matters of taste: the *annotated* skeleton joins two keypoints with
a straight line, so an edge bowed past :data:`_MAX_SAGITTA` would let its own chord escape the stroke;
and a cut edge must stay straight, since splitting a sampled arc into two hairline-separated flat
caps is where the geometry is least forgiving (see :func:`_split_polyline`). Both are rejected at
load time rather than left to a rendering surprise.

**Counters via a hairline slit.** Seven letters (``a``, ``b``, ``d``, ``o``, ``p``, ``q``, ``r``)
have an enclosed counter — a cycle in the stroke graph. A single ring cannot enclose a hole
directly, so :data:`_CUTS` names one graph edge per independent cycle; :func:`_stroke_outline`
splits that edge at its midpoint into two flat-capped stubs separated by
:data:`LETTER_COUNTER_GAP`, which opens the cycle (the trace no longer finds a separate inner face)
while leaving a slit far too thin to read as a gap.

*Where* a counter opens is an authoring decision, not a free one, so a cut may name a position along
its edge as well as the edge itself. A bowl hung off a stem (``b``, ``d``, ``p``, ``r``) breaks on
the edge leaving that stem and as near to it as fits, so the bowl reads as a curve just touching a
vertical line. A free-standing ring breaks along its bottom — left of centre for ``o``, bottom-right
for ``q`` beside where its tail joins. ``a`` breaks at its crossbar, the bottom of its triangular
counter. ``test_counter_cuts_sit_at_the_bottom_of_their_counter`` pins edge and position together.
"As near as fits" has a floor: a stub shorter than :data:`_MIN_STUB` half-widths is swallowed by the
stroke it branches off, and :func:`_reject_tight_cuts` says so rather than letting the ring fold.

The cut is purely a rendering-time transform;
:data:`_LETTER_STROKES` (the graph :attr:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for`
reports) is untouched by it, so a counter letter's skeleton still connects straight across the cut
— that hairline is the one place, anywhere in the family, where a skeleton edge leaves the fill.

**The node grid, and free positions.** Fifteen named slots give every letter the same landmark
vocabulary (a fixed count per dataset is a hard requirement of both output formats). Their default
coordinates form a regular 3-column x 5-row grid, but a letter is free to place any of its own nodes
anywhere it likes through :data:`_NODES` — the grid is a convenient default, not a constraint on the
letterform. The slots, in :data:`LETTER_KEYPOINT_NAMES` order:

=========== =========== ===========
top_left     top_mid     top_right
upper_left   upper_mid   upper_right
mid_left     center      mid_right
lower_left   lower_mid   lower_right
bottom_left  bottom_mid  bottom_right
=========== =========== ===========

A letter carries **only the nodes its shape actually needs**: wherever three consecutive keypoints
sit on one straight run, the middle one says nothing a viewer could not read off its neighbours, so
it is not authored at all. That is why a plain stem is two keypoints rather than five, and why the
ones that remain are exactly the corners, junctions, ends, and the points along a curve. An unused
slot is ``(nan, nan)`` in
:data:`LETTER_KEYPOINTS`, the same optional-landmark contract
:mod:`~synth_datasets.families.animals` and :mod:`~synth_datasets.families.symbols` already use.
:data:`LETTER_KEYPOINT_FLIP_IDX` swaps each row's left/right slot and holds the middle column fixed
— a property of the slot *naming* rather than of any one letter, unlike the hand-verified mappings
either of those families carries. Note that a mirrored letter is generally a different letter (or no
letter at all), so a horizontal flip is not a label-preserving augmentation here the way it is for an
animal silhouette; the field is published for format completeness.

**Rotational-symmetry authoring rule.** A letter whose nodes and edges are together invariant under a
180-degree rotation about its own center has keypoint-identity ambiguity under the generator's
continuous random rotation — the shape looks the same upside down, so which node is which becomes
unrecoverable. It is the same reason :class:`~synth_datasets.families.primitives.PrimitiveShape` carries no
keypoint table for ``SQUARE``/``CIRCLE`` and :class:`~synth_datasets.families.symbols.SymbolShape`
has no plain triangle. Checking every letter found nine that are exactly invariant as regular block
letterforms: ``B``, ``D``, ``H``, ``I``, ``N``, ``O``, ``S``, ``X``, ``Z`` — the same set real
handwriting calls "look the same upside down". Each breaks the symmetry by moving one node off its
default slot in :data:`_NODES`, the same technique
:attr:`~synth_datasets.families.symbols.SymbolShape.KITE`'s unequal diagonal lengths already use.
:data:`_CUTS` does not affect this check: it is a rendering-only graph transform, not part of the
logical ``(nodes, edges)`` a viewer or this check ever sees.

Pure NumPy, no image-library dependency, and no import from
:mod:`~synth_datasets.families.animals` or :mod:`~synth_datasets.families.symbols`: this is a third
sibling under :mod:`~synth_datasets.core.keypoints`, importing only
:mod:`~synth_datasets.families.geometry`. Tables are keyed by the plain
:class:`LetterShape` *values*.

Examples:
    ```pycon
    >>> from synth_datasets.families.letters import LETTER_POLYGONS, LetterShape
    >>> len(LETTER_POLYGONS)
    26
    >>> LETTER_POLYGONS["i"].shape[1]
    2
    >>> LetterShape("a")
    <LetterShape.A: 'a'>

    ```

"""

from __future__ import annotations

import math
from importlib.resources import files
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from synth_datasets.core.keypoints import KeypointSchema, _normalized_pair
from synth_datasets.families.geometry import place_points
from synth_datasets.families.shape_enum import ShapeEnum
from synth_datasets.families.svgio import read_graph_document

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable

    from numpy.typing import NDArray

_NAN = float("nan")
#: Placeholder for a grid slot a given letter has no node at.
_ABSENT: tuple[float, float] = (_NAN, _NAN)


class LetterShape(ShapeEnum):
    """Capital-letter outline vocabulary (definition order is the letter class order).

    Twenty-six capitals, each a single simple polygon derived from a small straight-stroke graph on
    the shared node grid described in the module docstring. Lower-case only in ``A``-``Z``.

    Examples:
        ```pycon
        >>> from synth_datasets.families.letters import LetterShape
        >>> len(LetterShape)
        26
        >>> LetterShape("z")
        <LetterShape.Z: 'z'>

        ```

    """

    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"
    G = "g"
    H = "h"
    I = "i"  # noqa: E741 - the letter I, not the ambiguous-name lint's usual loop-counter case
    J = "j"
    K = "k"
    L = "l"
    M = "m"
    N = "n"
    O = "o"  # noqa: E741 - the letter O
    P = "p"
    Q = "q"
    R = "r"
    S = "s"
    T = "t"
    U = "u"
    V = "v"
    W = "w"
    X = "x"
    Y = "y"
    Z = "z"


#: Letter names in :class:`LetterShape` declaration order.
LETTER_NAMES: tuple[str, ...] = tuple(shape.value for shape in LetterShape)

#: Landmark names for :attr:`~synth_datasets.core.config.Task.KEYPOINTS` under the letter
#: family: the 15-node grid, row-major (see the module docstring's table). No slot is mandatory —
#: unlike :mod:`~synth_datasets.families.symbols`'s ``center``, the grid has no single node every
#: letter touches (e.g. ``V`` never uses ``center``).
LETTER_KEYPOINT_NAMES: tuple[str, ...] = (
    "top_left",
    "top_mid",
    "top_right",
    "upper_left",
    "upper_mid",
    "upper_right",
    "mid_left",
    "center",
    "mid_right",
    "lower_left",
    "lower_mid",
    "lower_right",
    "bottom_left",
    "bottom_mid",
    "bottom_right",
)

#: Grid node index constants, purely for readability in :data:`_LETTER_STROKES`, :data:`_CUTS`, and
#: :data:`_NODES` below — these are positions into :data:`LETTER_KEYPOINT_NAMES`, not exported.
_TL, _TM, _TR = 0, 1, 2
_UL, _UM, _UR = 3, 4, 5
_ML, _CTR, _MR = 6, 7, 8
_LL, _LM, _LR = 9, 10, 11
_BL, _BM, _BR = 12, 13, 14

#: Canonical unit-ish grid coordinates, row-major, screen orientation (``+y`` down, matching every
#: other family). Columns narrower than rows spread (a letter is taller than it is wide); a letter's
#: own outline is later renormalized to the package's unit convention regardless of these raw values.
_COL_X: tuple[float, float, float] = (-0.62, 0.0, 0.62)
_ROW_Y: tuple[float, float, float, float, float] = (-1.0, -0.5, 0.0, 0.5, 1.0)
_GRID_XY: tuple[tuple[float, float], ...] = tuple(
    (_COL_X[index % 3], _ROW_Y[index // 3]) for index in range(len(LETTER_KEYPOINT_NAMES))
)
#: The grid's own larger raw extent (row span) — the reference :data:`LETTER_STROKE_WIDTH` and
#: :data:`LETTER_COUNTER_GAP` scale against, before any per-letter renormalization.
_GRID_EXTENT: float = _ROW_Y[-1] - _ROW_Y[0]

#: Horizontal-flip permutation: each row's left/right column (indices ``3*row`` and ``3*row + 2``)
#: swap, the middle column (``3*row + 1``) maps to itself. A property of the grid's own coordinates
#: (any node in column 0 mirrors to column 2 at the same row) rather than of any one letter's shape,
#: unlike :data:`~synth_datasets.families.symbols.SYMBOL_KEYPOINT_FLIP_IDX`.
LETTER_KEYPOINT_FLIP_IDX: tuple[int, ...] = tuple(
    3 * (index // 3) + (2 - index % 3 if index % 3 != 1 else 1) for index in range(len(LETTER_KEYPOINT_NAMES))
)

#: Packaged asset holding the per-letter stroke graph (which node pairs are joined by a stroke), the
#: counter-opening cuts, and the free node positions described in the module docstring.
#: The single source of truth :func:`_stroke_outline` (rendering) and
#: :data:`LETTER_KEYPOINT_SCHEMA`'s ``skeleton_by_value`` (annotation) both read from. A letter's
#: *used* nodes are exactly the indices appearing in its own edge list; every other slot is absent
#: (NaN) in :data:`LETTER_KEYPOINTS`.
#: Directory holding the packaged letter documents, resolved through :mod:`importlib.resources` so
#: it works from a source checkout and from an installed wheel alike.
_ASSET: Traversable = files("synth_datasets") / "assets" / "letters"

#: The document frame letters are authored in: a 1000x1000 canvas with the grid origin at its centre
#: and one grid unit spanning :data:`_DOC_SCALE` pixels. Matches the zoo documents' canvas, so every
#: family is edited at the same scale by the same tool.
_DOC_SCALE: float = 400.0
_DOC_MID: float = 500.0


#: Shortest a cut stub may be, in multiples of half the stroke width — see :func:`_split_polyline`.
_MIN_STUB: float = 1.2

#: Deepest an arc may bow, as a fraction of *half the stroke width* — see :func:`_edge_polyline`.
#: The bound is on the arc's absolute height rather than on its ratio to its own chord, because it is
#: really a bound on how far the edge's straight skeleton chord strays from the ink: that distance is
#: exactly the arc's height, whatever the chord's length. Checked in :func:`_load`, where the node
#: positions the height depends on are finally known.
_MAX_SAGITTA: float = 0.7


#: Node name to grid index, the inverse of :data:`LETTER_KEYPOINT_NAMES`. Documents name their nodes
#: (``"top_mid"``), while the stroke machinery indexes them, so the reader converts once here.
_NODE_INDEX: dict[str, int] = {name: index for index, name in enumerate(LETTER_KEYPOINT_NAMES)}


def _read_letter_document(
    name: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[int, tuple[float, float]],
]:
    """Read one packaged letter document into its stroke, bulge, cut, and node-position tables.

    A letter is stored as a graph rather than an outline: the silhouette is generated by stroking
    these edges (see :func:`_stroke_outline`), which is what keeps
    :data:`LETTER_STROKE_WIDTH` and :data:`LETTER_COUNTER_GAP` tunable after the fact and keeps a
    node a single draggable point instead of a consequence baked into hundreds of outline vertices.
    That is also why letters keep their own document shape while animals and symbols share the
    outline one — the schema follows the data, not the other way round.

    Args:
        name: Letter value, i.e. the ``<name>.svg`` stem in the packaged ``letters`` directory.

    Returns:
        The stroke edges as index pairs, the non-zero bulges by edge, the cuts by *sorted* edge with
        the slit position measured from the lower index, and every used node's position in the
        authored grid frame.

    Raises:
        ValueError: If a cut names an edge the letter's own stroke graph does not have, sits outside
            that edge, or names a bowed one. Endpoint and provenance validation happens in the
            shared reader.

    """
    nodes, raw_strokes, raw_cuts, _provenance = read_graph_document(_ASSET, name, LETTER_KEYPOINT_NAMES)
    positions = {
        _NODE_INDEX[node]: ((x - _DOC_MID) / _DOC_SCALE, (y - _DOC_MID) / _DOC_SCALE) for node, (x, y) in nodes.items()
    }
    strokes = tuple((_NODE_INDEX[start], _NODE_INDEX[end]) for start, end, _ in raw_strokes)
    bulges = {(_NODE_INDEX[start], _NODE_INDEX[end]): bulge for start, end, bulge in raw_strokes if bulge != 0.0}
    known_edges = {(min(i, j), max(i, j)) for i, j in strokes}
    # A cut names where along its edge the slit sits, as a fraction from the endpoint it lists first;
    # stored keyed by the sorted edge, so the fraction is always measured from the lower index.
    cuts: dict[tuple[int, int], float] = {}
    for start, end, at in raw_cuts:
        lo, hi = _NODE_INDEX[start], _NODE_INDEX[end]
        cuts[(min(lo, hi), max(lo, hi))] = at if lo < hi else 1.0 - at
    unknown = set(cuts) - known_edges
    if unknown:
        raise ValueError(f"letter {name!r} cut(s) name edge(s) {unknown!r} not in its strokes")
    outside = {edge: at for edge, at in cuts.items() if not 0.0 < at < 1.0}
    if outside:
        raise ValueError(f"letter {name!r} cut position(s) lie outside (0, 1): {outside!r}")
    bowed = {edge for edge, value in bulges.items() if (min(edge), max(edge)) in cuts and value}
    if bowed:
        raise ValueError(
            f"letter {name!r} cuts bowed edge(s) {bowed!r}; a cut edge must stay straight "
            "(see _split_polyline) — bow a neighbouring edge instead, or move the cut"
        )
    return strokes, bulges, cuts, positions


def _read_letter_data() -> tuple[
    dict[str, tuple[tuple[int, int], ...]],
    dict[str, dict[tuple[int, int], float]],
    dict[str, dict[tuple[int, int], float]],
    dict[str, dict[int, tuple[float, float]]],
]:
    """Read every packaged letter document into the per-letter stroke, bulge, cut, and node tables."""
    strokes: dict[str, tuple[tuple[int, int], ...]] = {}
    bulges: dict[str, dict[tuple[int, int], float]] = {}
    cuts: dict[str, dict[tuple[int, int], float]] = {}
    positions: dict[str, dict[int, tuple[float, float]]] = {}
    for name in LETTER_NAMES:
        strokes[name], bulges[name], letter_cuts, positions[name] = _read_letter_document(name)
        # Only letters that actually have counters get a cuts entry, so membership in this table
        # keeps meaning "this letter has a counter to open" rather than "this letter exists".
        if letter_cuts:
            cuts[name] = letter_cuts
    return strokes, bulges, cuts, positions


_LETTER_STROKES, _BULGES, _CUTS, _NODES = _read_letter_data()

#: Stroke width :func:`_stroke_outline` offsets each edge by, as a fraction of the grid's own row
#: span (:data:`_GRID_EXTENT`) — the family analog of
#: :data:`~synth_datasets.families.geometry.RECT_ASPECT`. Applied before a letter's outline is
#: renormalized to the package's unit convention, so the final on-screen thickness is close to but
#: not exactly this fraction of the letter's own drawn extent (the outline grows slightly larger
#: than the bare node grid once stroke width is added).
LETTER_STROKE_WIDTH: float = 0.16

#: How far apart the two stubs of a counter-opening cut sit, as the same grid-row-span fraction
#: :data:`LETTER_STROKE_WIDTH` uses — an order of magnitude smaller, so the slit it leaves in a
#: counter letter's outline (``a``, ``b``, ``d``, ``o``, ``p``, ``q``, ``r``) reads as a hairline
#: rather than a visible gap, and so the stretch of skeleton edge crossing it is negligible. It is
#: also what forces those two stubs to cap flat: two semicircles this close together would overlap
#: and cross the ring (see :func:`_stroke_outline`).
LETTER_COUNTER_GAP: float = 0.01

#: Every distinct stroke edge across the whole alphabet, deduplicated and sorted — the family-wide
#: fallback :attr:`~synth_datasets.core.keypoints.KeypointSchema.skeleton`. A consumer that
#: reads only this field (rather than
#: :meth:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for`) sees the union of every
#: letter's strokes; the per-letter accuracy lives in ``skeleton_by_value`` instead.
LETTER_KEYPOINT_SKELETON: tuple[tuple[int, int], ...] = tuple(
    sorted({(min(edge), max(edge)) for edges in _LETTER_STROKES.values() for edge in edges})
)


#: Angular resolution of a rounded join or cap, in radians — a quarter-turn corner becomes six
#: segments, a stroke tip's semicircle twelve. Small enough to read as a smooth curve at any render
#: size this package draws at, large enough to keep a letter's ring in the low hundreds of vertices.
_ARC_STEP: float = math.pi / 12.0

#: How close two outline points may be, as a fraction of the stroke width, before :func:`_deduplicated`
#: merges them into one. Well below any real feature, well above the slivers a miter landing beside an
#: arc sample leaves behind.
_MERGE_FRACTION: float = 0.05

#: Below this ``|cross|`` two consecutive offset lines are treated as parallel — a concave join that
#: close to a straight line has no usable intersection, so it falls back to the two raw endpoints.
_PARALLEL_EPS: float = 1e-9


def _stroke_outline(
    nodes: NDArray[np.float64],
    edges: tuple[tuple[int, int], ...],
    bulges: Mapping[tuple[int, int], float],
    cuts: Mapping[tuple[int, int], float],
    width: float,
    gap: float,
) -> NDArray[np.float64]:
    """Wrap a stroke graph in a single round-joined, round-capped outline of thickness ``width``.

    The letter is authored as keypoints and the edges between them; this is the step that turns that
    skeleton into something drawable, by wrapping it in a pen stroke of constant width — the exact
    inverse of tracing keypoints out of a hand-drawn outline. Because the outline *is* the set of
    points within ``width / 2`` of the skeleton, **every skeleton edge lies strictly inside the
    polygon by construction**, with ``width / 2`` of clearance, which is the property that makes the
    keypoints and their connecting edges meaningful to look at. (The single exception is an edge
    named in ``cuts``; see below.)

    Mechanically it is a face trace of the graph's planar embedding: at every node, incident edges
    are ordered by angle, and the walk always continues on the next edge clockwise in that fixed
    rotational order from the one it arrived on — the standard way to trace one face's boundary
    given a rotation system. A tree (no cycles) has exactly one face, so this single walk covers
    every edge twice (once per side) and closes into one simple ring.

    Each visited half-edge is offset ``width / 2`` to its own left, and the two offset lines meeting
    at a node are joined by whichever of two shapes the union of the strokes actually has there:

    * **Convex side** (the walk sweeps at most half a turn around the node) — a circular **arc** of
      radius ``width / 2`` centered on the node, sampled every :data:`_ARC_STEP`. A stroke's free end
      is the limiting case, an exact half turn, so it caps as a **semicircle**: no letter has a sharp
      point anywhere on it, and the node itself sits at the cap's own center, a full ``width / 2``
      inside the fill rather than on the boundary.
    * **Concave side** (more than half a turn) — the single **miter** point where the two offset
      lines actually cross, since the strokes' filled bodies already cover the node's whole disk
      there. Rounding that side instead would bulge the outline *outward* across the inside of the
      turn and self-intersect the ring; bevelling it does the same. A razor-thin concave angle would
      send a miter shooting far from the node, but no letter has one.

    A cycle (an enclosed counter — see the module docstring) yields more than one face, so ``cuts``
    names edges to open first: a cut edge becomes two stubs, each stopping ``gap / 2`` short of the
    true edge midpoint, rather than one edge joining its two nodes. That removes exactly the cycle
    the edge closed without moving any other node, leaving a single walk that dips into the counter
    through a hairline slit and back out. A stub's own free end caps **flat**, not round: two
    semicircles ``gap`` apart would overlap and cross each other, since ``gap`` is deliberately far
    smaller than ``width``. The slit is the one place a skeleton edge leaves the fill, for the ``gap``
    of its length that the slit spans.

    Args:
        nodes: ``(15, 2)`` node table (see :data:`_GRID_XY`); only rows ``edges`` reference are
            read, so unused (NaN) rows never enter the computation.
        edges: The letter's stroke graph, as index pairs into ``nodes``.
        bulges: Per-edge arc depth, keyed by the authored ``(u, v)`` direction — see
            :func:`_edge_polyline`. An edge naming none is straight.
        cuts: Where to open each cut edge, as ``{(min, max): fraction}`` per :data:`_CUTS`.
        width: Stroke width, in ``nodes``' own units.
        gap: Cut-stub separation, in ``nodes``' own units — see :data:`LETTER_COUNTER_GAP`.

    Returns:
        A single, simple, closed ``(num_vertices, 2)`` ring.

    Raises:
        ValueError: If the graph (after opening ``cuts``) is not a single connected tree — either a
            disconnected component or a cycle ``cuts`` failed to fully open, both of which would
            trace more than one face.

    """
    half = width / 2.0
    positions, adjacency, flat_capped = _opened_graph(nodes, edges, bulges, cuts, gap, half)
    order = {
        node: [n for _, n in sorted((_angle(positions[n] - positions[node]), n) for n in neighbors)]
        for node, neighbors in adjacency.items()
    }
    neighbor_index = {node: {n: i for i, n in enumerate(neighbors)} for node, neighbors in order.items()}

    def _next_half_edge(u: int, v: int) -> tuple[int, int]:
        neighbors = order[v]
        idx = neighbor_index[v][u]
        return v, neighbors[(idx - 1) % len(neighbors)]

    total_half_edges = sum(len(neighbors) for neighbors in adjacency.values())
    start = next(node for node, neighbors in adjacency.items() if neighbors)
    current = (start, adjacency[start][0])
    visited: set[tuple[int, int]] = set()
    walk: list[tuple[int, int]] = []
    while current not in visited:
        visited.add(current)
        walk.append(current)
        current = _next_half_edge(*current)
    if len(visited) != total_half_edges:
        raise ValueError(
            f"stroke graph traced {len(visited)} of {total_half_edges} half-edges; it is not a single "
            "connected tree (a disconnected component or an un-cut cycle remains)"
        )

    ring_points: list[NDArray[np.float64]] = []
    for k, (u, v) in enumerate(walk):
        w = walk[(k + 1) % len(walk)][1]
        ring_points.extend(_join_points(positions, u, v, w, half, flat_capped=v in flat_capped))
    return _deduplicated(np.array(ring_points, dtype=np.float64), width)


def _opened_graph(
    nodes: NDArray[np.float64],
    edges: tuple[tuple[int, int], ...],
    bulges: Mapping[tuple[int, int], float],
    cuts: Mapping[tuple[int, int], float],
    gap: float,
    half: float,
) -> tuple[dict[int, NDArray[np.float64]], dict[int, list[int]], set[int]]:
    """Return ``(positions, adjacency, flat_capped)`` for ``edges``, curved and cut as authored.

    Every edge becomes a *chain* of nodes: a straight edge is the single link it already was, while a curved one (see
    :data:`_BULGES`) is subdivided along its arc into short straight links, which is what lets the rest of
    :func:`_stroke_outline` wrap a curve while having no notion of curvature at all. Those interior nodes are synthetic,
    invisible to :data:`LETTER_KEYPOINTS` and to the reported skeleton, which still name only the authored endpoints.

    A cut edge's chain is broken at its own arclength midpoint into two stubs ``gap`` apart instead of joined through;
    see :func:`_stroke_outline`. ``flat_capped`` collects exactly those two stub tips, the only nodes whose free end is
    capped with a straight line rather than a semicircle.

    """
    used = {index for edge in edges for index in edge}
    positions: dict[int, NDArray[np.float64]] = {index: nodes[index] for index in used}
    adjacency: dict[int, list[int]] = {index: [] for index in used}
    flat_capped: set[int] = set()
    next_id = max(used) + 1

    def _link(first: int, second: int) -> None:
        adjacency[first].append(second)
        adjacency[second].append(first)

    def _extend(anchor: int, points: list[NDArray[np.float64]]) -> int:
        """Chain ``points`` onto ``anchor`` through fresh synthetic nodes, returning the last one."""
        nonlocal next_id
        for point in points:
            positions[next_id] = point
            adjacency[next_id] = []
            _link(anchor, next_id)
            anchor = next_id
            next_id += 1
        return anchor

    for u, v in edges:
        chain = _edge_polyline(positions[u], positions[v], bulges.get((u, v), 0.0))
        if (min(u, v), max(u, v)) not in cuts:
            _link(_extend(u, chain[1:-1]), v)
            continue
        at = cuts[min(u, v), max(u, v)] if u < v else 1.0 - cuts[v, u]
        head, tail = _split_polyline(chain, gap, at, half)
        flat_capped.add(_extend(u, head[1:]))
        flat_capped.add(_extend(v, tail[1:]))
    return positions, adjacency, flat_capped


def _edge_polyline(start: NDArray[np.float64], end: NDArray[np.float64], bulge: float) -> list[NDArray[np.float64]]:
    """Return the points of one authored edge, from ``start`` to ``end`` inclusive.

    ``bulge`` is the arc's sagitta as a signed fraction of the chord it spans — the arc's height at its own midpoint,
    positive to the left of ``start -> end``. Zero (the default, and what every edge naming no bulge gets) leaves the
    edge the straight two-point segment it was. A curve is sampled at the same :data:`_ARC_STEP` resolution the rounded
    joins use, so a curve and the caps at its ends read as one continuous stroke.

    Keeping ``|bulge|`` modest is a design constraint, not taste: the annotated skeleton joins two keypoints with a
    *straight* line, so an edge bowed far enough for its chord to leave the stroke would break the family's central
    promise. Roughly ``0.21`` is a quarter turn, where the chord still lies within the stroke for these letter sizes;
    ``test_skeleton_edges_lie_inside_the_outline`` samples exactly that chord, so an over-deep arc is caught rather than
    shipped.

    """
    chord = end - start
    length = float(np.hypot(*chord))
    sagitta = bulge * length
    if abs(sagitta) < _PARALLEL_EPS:
        return [start, end]
    radius = (length**2 / 4.0 + sagitta**2) / (2.0 * sagitta)
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / length
    center = (start + end) / 2.0 + normal * (sagitta - radius)
    begin = _angle(start - center)
    # Two arcs join any two points on a circle; take the one running through the bowed midpoint.
    span = (_angle(end - center) - begin) % (2.0 * math.pi)
    if (_angle((start + end) / 2.0 + normal * sagitta - center) - begin) % (2.0 * math.pi) > span:
        span -= 2.0 * math.pi
    steps = max(1, math.ceil(abs(span) / _ARC_STEP))
    return [
        center + abs(radius) * np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        for theta in (begin + span * step / steps for step in range(steps + 1))
    ]


def _split_polyline(
    chain: list[NDArray[np.float64]], gap: float, at: float, half: float
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Split the straight ``chain`` into two stubs ``gap`` apart, ``at`` of the way along it.

    Returns the stub running forward from ``chain[0]`` and the one running *backward* from ``chain[-1]``, so each is
    ordered outward from the authored node it hangs off.

    A cut edge is required to be straight (:func:`_read_letter_data` rejects a bowed one), which is what keeps this two
    lines rather than a special case per curve. Splitting a *sampled arc* is deceptively fragile: the midpoint can land
    near a sample, leaving the two stubs ending on segments a whole :data:`_ARC_STEP` apart in direction, and since a
    flat cap is a full stroke width long while ``gap`` is a hairline, a few degrees of tilt is enough for the two caps
    to cross and self-intersect the ring. Curving the one edge a counter opens buys nothing visible — the slit sits
    there — so the rule is simply that it stays straight.

    """
    start, end = chain[0], chain[-1]
    span = end - start
    length = float(np.hypot(*span))
    # However close to an end ``at`` asks for, a stub has to emerge from whatever stroke it branches
    # off: one shorter than that stroke's own half width is swallowed whole, and the ring then traces
    # a boundary through the inside of another stroke and crosses itself. This is what stops a slit
    # sitting flush against a stem rather than just clear of it.
    clearance = (_MIN_STUB * half + gap / 2.0) / length
    point = start + span * min(max(at, clearance), 1.0 - clearance)
    tangent = span / length
    return [start, point - tangent * (gap / 2.0)], [end, point + tangent * (gap / 2.0)]


def _join_points(
    positions: dict[int, NDArray[np.float64]], u: int, v: int, w: int, half: float, *, flat_capped: bool
) -> list[NDArray[np.float64]]:
    """Return the outline points joining half-edge ``u -> v`` to half-edge ``v -> w``, around ``v``.

    ``sweep`` is how far the boundary turns around ``v`` between the two offset lines: an exact half turn at a stroke's
    free end (where ``w`` is ``u``), zero for a dead-straight pass-through, and anything in between at a corner. Past a
    half turn the join is on the concave side of the turn and resolves to a miter instead of an arc — see
    :func:`_stroke_outline` for why.

    """
    origin = positions[v]
    angle_in = _angle(positions[u] - origin)
    sweep = (angle_in - _angle(positions[w] - origin) - math.pi) % (2.0 * math.pi)
    if sweep > math.pi:
        miter = _miter_point(positions, u, v, w, half)
        if miter is not None:
            return [miter]
    segments = 1 if flat_capped else max(1, math.ceil(sweep / _ARC_STEP))
    start = angle_in - math.pi / 2.0
    return [
        origin + half * np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        for theta in (start - sweep * step / segments for step in range(segments + 1))
    ]


def _miter_point(
    positions: dict[int, NDArray[np.float64]], u: int, v: int, w: int, half: float
) -> NDArray[np.float64] | None:
    """Return where the two offset lines around ``v`` cross, or ``None`` when they are parallel."""
    p_in, d_in = _offset_line(positions, u, v, half)
    p_out, d_out = _offset_line(positions, v, w, half)
    turn = float(d_in[0] * d_out[1] - d_in[1] * d_out[0])
    if abs(turn) < _PARALLEL_EPS:
        return None
    diff = p_out - p_in
    crossing: NDArray[np.float64] = p_in + d_in * ((diff[0] * d_out[1] - diff[1] * d_out[0]) / turn)
    return crossing


def _offset_line(
    positions: dict[int, NDArray[np.float64]], u: int, v: int, half: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``u -> v`` offset line's start point and the half-edge's raw direction."""
    direction = positions[v] - positions[u]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64) / float(np.hypot(*direction))
    return positions[u] + normal * half, direction


def _deduplicated(ring: NDArray[np.float64], width: float) -> NDArray[np.float64]:
    """Merge points closer than :data:`_MERGE_FRACTION` of ``width``, and close the ring.

    The obvious job is dropping the exactly-coincident points a straight pass-through emits. The tolerance is scaled to
    the stroke width rather than left at floating-point noise for the less obvious one: where a rounded join's miter
    lands a hair from an arc sample, the two leave a sliver a thousandth of a stroke wide, and a sliver that happens to
    step backwards is a genuine self-intersection of the ring — visually nothing, but enough to fail the simplicity
    that every downstream consumer relies on. Two outline points that close together cannot be told apart at any size
    this package renders, so merging them is free.

    """
    tolerance = width * _MERGE_FRACTION
    keep = [0]
    for i in range(1, len(ring)):
        if float(np.hypot(*(ring[i] - ring[keep[-1]]))) > tolerance:
            keep.append(i)
    if float(np.hypot(*(ring[keep[-1]] - ring[keep[0]]))) <= tolerance:
        keep.pop()
    return ring[keep]


def _angle(delta: NDArray[np.float64]) -> float:
    """Return the angle of ``delta`` in radians, for sorting :func:`_stroke_outline`'s neighbors."""
    return float(np.arctan2(delta[1], delta[0]))


def _load() -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Build the per-letter outline polygon and its keypoint table.

    The outline is wrapped around the very nodes the keypoint table reports (see
    :func:`_stroke_outline`) and both pass through one shared normalization (see
    :func:`~synth_datasets.core.keypoints._normalized_pair`), so a keypoint can never drift off
    the ink the way authoring an outline and its landmarks separately could — and needs no
    correction afterwards, since every node sits a full half stroke-width inside the fill.

    """
    raw_width = LETTER_STROKE_WIDTH * _GRID_EXTENT
    raw_gap = LETTER_COUNTER_GAP * _GRID_EXTENT
    polygons: dict[str, NDArray[np.float64]] = {}
    keypoint_tables: dict[str, NDArray[np.float64]] = {}
    for name in LETTER_NAMES:
        raw_nodes = raw_letter_nodes(name)
        _reject_deep_arcs(name, raw_nodes, raw_width / 2.0)
        _reject_tight_cuts(name, raw_nodes, raw_width / 2.0, raw_gap)
        outline = _stroke_outline(
            raw_nodes,
            _LETTER_STROKES[name],
            _BULGES.get(name, {}),
            _CUTS.get(name, {}),
            raw_width,
            raw_gap,
        )
        outline_points = [(float(x), float(y)) for x, y in outline]
        raw_points = [(float(x), float(y)) for x, y in raw_nodes]
        polygon, keypoints = _normalized_pair(outline_points, raw_points, LETTER_KEYPOINT_NAMES)
        keypoints.setflags(write=False)
        polygons[name] = polygon
        keypoint_tables[name] = keypoints
    return polygons, keypoint_tables


def _reject_deep_arcs(name: str, nodes: NDArray[np.float64], half: float) -> None:
    """Raise if any of ``name``'s arcs bows further from its own chord than :data:`_MAX_SAGITTA` allows.

    Args:
        name: A :class:`LetterShape` value.
        nodes: That letter's authored node table, from :func:`raw_letter_nodes`.
        half: Half the stroke width, in ``nodes``' own units.

    Raises:
        ValueError: If an edge's arc height exceeds ``_MAX_SAGITTA * half``, which would put the
            straight skeleton chord between its two keypoints outside the stroke that draws it.

    """
    deep = {
        edge: round(height, 4)
        for edge, bulge in _BULGES.get(name, {}).items()
        if (height := abs(bulge) * float(np.hypot(*(nodes[edge[1]] - nodes[edge[0]])))) > _MAX_SAGITTA * half
    }
    if deep:
        raise ValueError(
            f"letters.json letter {name!r} bows edge(s) {deep!r} more than {_MAX_SAGITTA} of the stroke's half width "
            f"({_MAX_SAGITTA * half:.4f}); the straight skeleton chord between those keypoints would leave the ink "
            "(see _edge_polyline). Shorten the edge or shallow the bow."
        )


def _reject_tight_cuts(name: str, nodes: NDArray[np.float64], half: float, gap: float) -> None:
    """Raise if any of ``name``'s cuts sits too near an end of its own edge to leave a drawable stub.

    :func:`_split_polyline` clamps a position it cannot honour, which would otherwise mean the asset
    quietly says one thing and the outline does another. Checking here turns that into an error that
    names the nearest position the edge can actually carry.

    Args:
        name: A :class:`LetterShape` value.
        nodes: That letter's authored node table, from :func:`raw_letter_nodes`.
        half: Half the stroke width, in ``nodes``' own units.
        gap: The slit width, likewise.

    Raises:
        ValueError: If a cut leaves a stub shorter than ``_MIN_STUB`` half-widths, which the stroke it
            branches off would swallow whole (see :func:`_split_polyline`).

    """
    tight = {}
    for (low, high), at in _CUTS.get(name, {}).items():
        clearance = (_MIN_STUB * half + gap / 2.0) / float(np.hypot(*(nodes[high] - nodes[low])))
        if not clearance <= at <= 1.0 - clearance:
            tight[low, high] = (at, round(clearance, 3), round(1.0 - clearance, 3))
    if tight:
        raise ValueError(
            f"letters.json letter {name!r} cuts too close to an edge end: {tight!r} (position, then the range that "
            "edge can carry). A stub shorter than the stroke it branches off is swallowed by it and the outline "
            "self-intersects; move the cut inward, or put it on a longer edge."
        )


def raw_letter_nodes(name: str) -> NDArray[np.float64]:
    """Return one letter's authored node positions, before any normalization.

    The letter's own :data:`_NODES` entry overrides the shared grid slot outright for whichever nodes
    it names (see the module docstring's free-position rule); every other used node keeps its
    :data:`_GRID_XY` default, and a node the letter's stroke graph never touches is ``(nan, nan)``.

    This is the frame a letter is *authored* in — the one the rotational-symmetry rule is checked
    against. :data:`LETTER_KEYPOINTS` reports these same nodes after normalization, so the two differ
    only by the uniform scale-and-translate :func:`_normalized_pair` applies.

    Args:
        name: A :class:`LetterShape` value.

    Returns:
        A fresh ``(15, 2)`` array in :data:`_GRID_XY`'s own units, in
        :data:`LETTER_KEYPOINT_NAMES` order.

    Examples:
        ```pycon
        >>> from synth_datasets.families.letters import raw_letter_nodes
        >>> raw_letter_nodes("i").shape
        (15, 2)

        ```

    """
    used = {index for edge in _LETTER_STROKES[name] for index in edge}
    free = _NODES.get(name, {})
    return np.array(
        [
            (free.get(index, _GRID_XY[index]) if index in used else _ABSENT)
            for index in range(len(LETTER_KEYPOINT_NAMES))
        ],
        dtype=np.float64,
    )


_POLYGONS, _KEYPOINTS = _load()

#: Outline polygon per letter :class:`~synth_datasets.core.config.Shape` *value*, centered on
#: its area centroid with larger extent ``1`` — the same unit convention
#: :data:`~synth_datasets.families.animals.ANIMAL_POLYGONS` and
#: :data:`~synth_datasets.families.symbols.SYMBOL_POLYGONS` use, so
#: :func:`~synth_datasets.families.shape_outline` needs no letter-specific handling.
LETTER_POLYGONS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_POLYGONS)

#: Keypoint table per letter :class:`~synth_datasets.core.config.Shape` *value*, in
#: :data:`LETTER_KEYPOINT_NAMES` order, in the same frame as :data:`LETTER_POLYGONS`. Every entry is
#: a read-only ``(15, 2)`` array; scale a copy rather than mutating it. A row is ``(nan, nan)`` for a
#: slot the letter does not use. A present row is exactly the node the outline was wrapped around
#: (see :func:`raw_letter_nodes`), needing no correction: the wrap leaves every node half a stroke
#: width inside the fill, so a keypoint is never on the boundary or outside it.
LETTER_KEYPOINTS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_KEYPOINTS)

#: The complete keypoint schema for every :class:`LetterShape` — the one artifact
#: :func:`~synth_datasets.core.config.keypoint_schema_for` and the writers need to describe a
#: letter ``Task.KEYPOINTS`` run.
LETTER_KEYPOINT_SCHEMA = KeypointSchema(
    names=LETTER_KEYPOINT_NAMES,
    skeleton=LETTER_KEYPOINT_SKELETON,
    flip_idx=LETTER_KEYPOINT_FLIP_IDX,
    shape_values=LETTER_NAMES,
    skeleton_by_value=MappingProxyType(dict(_LETTER_STROKES)),
)


def letter_keypoints(
    shape: LetterShape, center: tuple[float, float], size: float, angle: float = 0.0, skew: float = 0.0
) -> NDArray[np.float64]:
    """Place one letter's keypoint table into image coordinates.

    The table is looked up in :data:`LETTER_KEYPOINTS`, scaled, skewed, rotated, and translated
    exactly as :func:`~synth_datasets.families.shape_outline` treats
    :data:`LETTER_POLYGONS` — mirroring
    :func:`~synth_datasets.families.animals.animal_keypoints` and
    :func:`~synth_datasets.families.symbols.symbol_keypoints`.

    Args:
        shape: A :class:`LetterShape` member.
        center: Target center ``(x, y)`` in pixels — the same value passed to
            :func:`~synth_datasets.families.shape_outline`.
        size: Bounding size in pixels — likewise.
        angle: Rotation in radians about the shape center — likewise.
        skew: Signed fraction narrowing one pre-rotation half — likewise; see
            :attr:`~synth_datasets.core.config.SyntheticConfig.asymmetry_jitter`.

    Returns:
        ``(15, 2)`` float array of keypoint coordinates in image pixels, ordered by
        :data:`LETTER_KEYPOINT_NAMES`. A row is ``(nan, nan)`` for a grid slot the letter does not
        use.

    Raises:
        ValueError: If ``shape`` has no keypoint table.

    Examples:
        ```pycon
        >>> from synth_datasets.families.letters import LetterShape, letter_keypoints
        >>> points = letter_keypoints(LetterShape.X, center=(50.0, 50.0), size=20.0)
        >>> points.shape
        (15, 2)

        ```

    """
    table = LETTER_KEYPOINTS.get(shape.value)
    if table is None:
        known = ", ".join(LETTER_KEYPOINTS)
        raise ValueError(f"shape {shape.value!r} has no keypoint table; expected one of {known}")
    return place_points(table * size, center, angle, skew)
