"""Animal table validity: simplicity, unit convention, archetype, landmarks, provenance."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from synth_datasets.core.keypoints import _normalized, _normalized_pair
from synth_datasets.families import shape_outline
from synth_datasets.families.animals import (
    _OPTIONAL_KEYPOINTS,
    _REQUIRED_KEYPOINTS,
    _ZOO,
    ANIMAL_KEYPOINT_NAMES,
    ANIMAL_KEYPOINTS,
    ANIMAL_NAMES,
    ANIMAL_POLYGONS,
    ANIMAL_SOURCES,
    AnimalShape,
    _read_svg,
    animal_keypoints,
)
from synth_datasets.families.geometry import polygon_to_bbox_xyxy
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.svgio import (
    _SVG_NS,
    _ZOO_NS,
    parse_path_d,
    read_named_circles,
    reject_transforms,
    svg_tag,
    zoo_attr,
)


def _area_centroid(points: np.ndarray) -> np.ndarray:
    """Independent shoelace-formula centroid, kept separate from `landmarks._polygon_centroid`."""
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


ZOO_ANIMAL_NAMES = ANIMAL_NAMES

#: `AnimalShape`'s declaration order, pinned as a literal so a reordering of its members — which would
#: silently renumber every class id — is a deliberate, reviewed edit rather than an unnoticed drift.
PINNED_ANIMAL_ORDER = (
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
)

#: Per-animal set of landmark names this specific silhouette lacks (a NaN row in
#: :data:`ANIMAL_KEYPOINTS`), pinned as a literal so a change here is a deliberate edit, not an
#: incidental side effect of re-authoring an SVG.
ABSENT_KEYPOINTS = {
    "duck": frozenset(),
    "elephant": frozenset(),
    "giraffe": frozenset(),
    "fish": frozenset({"ear", "hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"}),
    "rabbit": frozenset(),
    "camel": frozenset(),
    "eagle": frozenset(),
    "penguin": frozenset(),
    "whale": frozenset({"ear", "hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"}),
    "kangaroo": frozenset(),
    "flamingo": frozenset(),
    "crocodile": frozenset(),
}

# Pillow includes boundary pixels when filling polygons. At the test scale the whale's and the
# crocodile's long, narrow bodies carry the largest rasterization-vs-shoelace area bias; the other
# silhouettes remain within the general 5% bound. The bias shrinks as the same outline is
# rasterized at higher scale.
RASTER_AREA_REL_TOLERANCE = {"whale": 0.08, "crocodile": 0.08}

# CC0 1.0 and the Public Domain Mark are the only provenances the packaged art may carry: both
# place the work in the public domain worldwide, so redistributing it inside a wheel is unencumbered.
PUBLIC_DOMAIN_LICENSES = {
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
}

# Width-to-height bands for the documented silhouette archetypes, bracketing the proportions of the
# public-domain reference art each outline was traced from. They are wide enough to absorb a
# re-trace but narrow enough that an animal losing its archetype (e.g. the giraffe no longer being
# tall-thin) fails instead of silently degrading.
ARCHETYPE_ASPECT_BANDS = {
    "duck": (0.58, 0.78),  # upright-bird
    "elephant": (1.65, 2.15),  # bulky-quadruped
    "giraffe": (0.66, 0.90),  # tall-thin
    "fish": (2.27, 2.95),  # streamlined
    "rabbit": (1.03, 1.35),  # compact-eared
    "camel": (1.10, 1.45),  # bulky-quadruped-humped
    "eagle": (0.39, 0.52),  # upright-bird
    "penguin": (0.55, 0.74),  # upright-bird
    "whale": (4.00, 5.25),  # streamlined-aquatic-large
    "kangaroo": (1.47, 1.92),  # hopping-marsupial
    "flamingo": (0.50, 0.67),  # long-legged-wader
    "crocodile": (3.42, 4.48),  # sprawling-reptile
}


def _cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the z-component of ``(a - origin) x (b - origin)``."""
    return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))


def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """Whether collinear point ``r`` lies within the bounding box of segment ``p``-``q``."""
    return bool(min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Whether closed segments ``p1p2`` and ``p3p4`` share at least one point (CLRS predicate)."""
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
    # Skip each edge's own neighbours (they legitimately share a vertex); trimming the last
    # index for i == 0 keeps the closing edge from being compared against edge 0.
    pairs = [(i, j) for i in range(count) for j in range(i + 2, count - (1 if i == 0 else 0))]
    return [(i, j) for i, j in pairs if _segments_intersect(*edges[i], *edges[j])]


def _shoelace_area(points: np.ndarray) -> float:
    """Return the absolute polygon area via the shoelace formula."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rasterize(name: str, size: float, canvas: int, angle: float = 0.0) -> np.ndarray:
    """Fill one animal outline into a boolean canvas exactly as the generator would."""
    image = Image.new("L", (canvas, canvas), 0)
    poly = shape_outline(name, center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    return np.asarray(image) > 0


def _rounded_pixel_is_filled(mask: np.ndarray, x: float, y: float) -> bool:
    """Whether the rounded ``(x, y)`` lands on a filled pixel of ``mask``.

    A coordinate that rounds outside the canvas counts as unfilled rather than being indexed directly:
    a negative index would silently wrap to the opposite edge under NumPy's indexing rules, and an
    index past the far edge would raise `IndexError` instead of failing the drift assertion cleanly.

    """
    row, col = round(float(y)), round(float(x))
    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
        return False
    return bool(mask[row, col])


def test_every_animal_member_has_a_table() -> None:
    """`AnimalShape` and the loaded outline tables are the same set, in both directions.

    Guards the two halves of the vocabulary drifting apart: a member without a document would raise only at draw time,
    deep inside a generation run, and a document without a member would ship dead weight in the wheel.

    """
    assert {shape.value for shape in AnimalShape} == set(ANIMAL_POLYGONS)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_is_a_simple_polygon(name: str) -> None:
    """No two non-adjacent outline edges touch or cross.

    A self-intersecting outline renders as an unpredictable bow-tie under Pillow's fill and makes the segmentation mask
    disagree with the rasterized pixels, so simplicity is the load-bearing invariant of a hand-authored table.

    """
    assert _self_intersections(ANIMAL_POLYGONS[name]) == []


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_has_no_repeated_vertex(name: str) -> None:
    """Every outline vertex is distinct, so the polygon has no pinch point.

    A duplicated vertex is the degenerate case the intersection predicate cannot flag as a crossing yet still produces a
    zero-width neck in the filled shape.

    """
    unique = np.unique(ANIMAL_POLYGONS[name], axis=0)
    assert len(unique) == len(ANIMAL_POLYGONS[name])


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_area_centroid_is_the_origin(name: str) -> None:
    """The outline is centred on its area centroid (center of mass), not its vertex mean.

    `shape_outline` translates the base outline by the requested centre without re-centring, so a
    table whose centroid drifts would place objects off their annotated centre. A vertex mean would
    not do here: a traced silhouette's vertices are unevenly spread along its edges (dense around a
    curved neck, sparse along a straight back), so only the area centroid reliably lands on the
    outline's true visual middle.

    """
    assert _area_centroid(ANIMAL_POLYGONS[name]) == pytest.approx([0.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_larger_extent_is_unit(name: str) -> None:
    """The larger of the two extents is exactly 1, so a `size` argument bounds the shape.

    Placement rejection and the size-ratio knobs both assume `size` is the true bounding extent; a table scaled
    differently would silently break `min_size_ratio`/`max_size_ratio`.

    """
    poly = ANIMAL_POLYGONS[name]
    extents = poly.max(axis=0) - poly.min(axis=0)
    assert max(extents) == pytest.approx(1.0)
    assert min(extents) > 0.0


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_vertex_count_is_in_the_authoring_band(name: str) -> None:
    """Each outline carries enough vertices to read as an animal without bloating labels.

    Segmentation labels emit every vertex, so an over-detailed table inflates label files while an under-detailed one
    stops being recognizable — this pins both ends of that trade-off.

    """
    assert 150 <= len(ANIMAL_POLYGONS[name]) <= 300


@pytest.mark.parametrize(("name", "band"), sorted(ARCHETYPE_ASPECT_BANDS.items()))
def test_table_matches_its_documented_archetype(name: str, band: tuple[float, float]) -> None:
    """Each silhouette keeps the width-to-height ratio its archetype is documented with.

    The twelve animals are chosen to be mutually distinguishable by gross proportion; coordinate tuning that pushes one
    into another's band would erode that separability.

    """
    poly = ANIMAL_POLYGONS[name]
    width, height = poly.max(axis=0) - poly.min(axis=0)
    low, high = band
    assert low <= width / height <= high


def test_every_animal_has_an_archetype_band() -> None:
    """Every roster member carries a documented aspect band, so a new animal cannot skip the check.

    `test_table_matches_its_documented_archetype` above parametrizes over `ARCHETYPE_ASPECT_BANDS.items()`, not over
    `ANIMAL_NAMES`, so an animal added without a matching band entry would silently generate zero test cases for it
    rather than fail; this closes that gap by asserting the two sets are exactly the same.

    """
    assert set(ARCHETYPE_ASPECT_BANDS) == set(ANIMAL_NAMES)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_is_frozen_against_mutation(name: str) -> None:
    """The shared table rejects in-place writes.

    Tables are module-level constants handed to every caller; a consumer mutating one would corrupt every later sample
    in the process, which is exactly the failure this prevents.

    """
    with pytest.raises(ValueError, match="read-only"):
        ANIMAL_POLYGONS[name][0, 0] = 99.0


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_scaled_polygon_does_not_alias_the_table(name: str) -> None:
    """`shape_outline` returns a fresh writable array rather than a view of the constant.

    Downstream code freely mutates the returned polygon (rotation, translation); aliasing the frozen table would either
    raise or silently corrupt the vocabulary.

    """
    poly = shape_outline(name, center=(0.0, 0.0), size=10.0)
    poly[0, 0] += 1.0  # must not raise
    assert poly.base is not ANIMAL_POLYGONS[name]


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_polygon_scales_and_translates_like_a_geometric_shape(name: str) -> None:
    """`shape_outline` bounds an animal by `size` and centres it on the requested point.

    This is the contract the placement loop depends on: it derives the candidate box from the
    polygon and rejects on that box, so a mis-scaled animal would break boundary handling.

    """
    poly = shape_outline(name, center=(120.0, 80.0), size=40.0)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    assert max(x2 - x1, y2 - y1) == pytest.approx(40.0)
    assert _area_centroid(poly) == pytest.approx([120.0, 80.0], abs=1e-6)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_rasterized_fill_matches_the_analytic_area(name: str) -> None:
    """Pillow's filled pixel count agrees with the outline's shoelace area.

    A self-intersecting or degenerate outline fills to a different area than its signed area predicts, so this is an
    end-to-end check that the annotation matches the drawn pixels.

    """
    size, canvas = 240.0, 320
    filled = int(_rasterize(name, size, canvas).sum())
    expected = _shoelace_area(ANIMAL_POLYGONS[name] * size)
    tolerance = RASTER_AREA_REL_TOLERANCE.get(name, 0.05)
    assert filled == pytest.approx(expected, rel=tolerance)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_silhouette_survives_a_small_rotated_draw(name: str) -> None:
    """A rotated animal at the smallest realistic size still rasterizes to visible pixels.

    `min_size_ratio` defaults to 0.1, so on a 320 px canvas the generator routinely draws 32 px animals at arbitrary
    angles; a thin outline that vanished there would emit empty labels.

    """
    image = Image.new("L", (64, 64), 0)
    poly = shape_outline(name, center=(32.0, 32.0), size=32.0, angle=0.7)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    assert int((np.asarray(image) > 0).sum()) > 0


def test_normalized_rejects_a_degenerate_outline() -> None:
    """Fewer than three points cannot describe an outline and is refused up front.

    The helper runs at import time, so a malformed table must fail loudly at module load rather than produce an unusable
    constant that only breaks during generation.

    """
    with pytest.raises(ValueError, match="at least 3"):
        _normalized([(0.0, 0.0), (1.0, 1.0)])


def test_normalized_rejects_a_zero_extent_outline() -> None:
    """An outline whose vertices coincide is refused instead of dividing by zero.

    Guards the scaling step: a zero extent would otherwise yield NaN coordinates that propagate
    silently into every annotation derived from the polygon.

    """
    with pytest.raises(ValueError, match="zero extent"):
        _normalized([(2.0, 2.0), (2.0, 2.0), (2.0, 2.0)])


def test_normalized_centres_and_scales_a_raw_outline() -> None:
    """A raw authoring-scale outline comes back centred with its larger extent equal to 1.

    This is the guarantee that lets the tables be tuned visually in convenient coordinates without any hand-maintained
    normalization, which is where drift would otherwise creep in. The ``(offset, extent)`` frame comes back alongside
    the polygon so a caller mapping a second table into the same frame need not re-measure the outline; it is pinned
    here because `_normalized_pair` maps every landmark through exactly these two numbers.

    """
    poly, offset, extent = _normalized([(10.0, 10.0), (30.0, 10.0), (30.0, 20.0), (10.0, 20.0)])
    assert poly.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert (poly.max(axis=0) - poly.min(axis=0)) == pytest.approx([1.0, 0.5])
    assert offset == pytest.approx([20.0, 15.0])
    assert extent == pytest.approx(20.0)


def test_zoo_directory_holds_exactly_the_declared_animals() -> None:
    """The packaged SVG files and the loader's animal list are the same set.

    The files are data, not code, so a rename or a missed addition would otherwise surface only when a generation run
    reached that animal; comparing the directory listing to the declared names catches it at test time.

    """
    stems = {entry.name.removesuffix(".svg") for entry in _ZOO.iterdir() if entry.name.endswith(".svg")}
    assert stems == set(ZOO_ANIMAL_NAMES)


def test_zoo_animal_order_matches_the_enum() -> None:
    """The packaged directory and `AnimalShape`'s own declaration order both match the enum, independently.

    `Traversable.iterdir()` makes no ordering guarantee, so the directory side is sorted before comparison — this is a
    genuine round trip through the filesystem, not the prior self-comparison of `ANIMAL_NAMES` against its own alias,
    which could never fail. Class ids come from the enum's declaration order, so that order is pinned separately as a
    literal: an accidental reshuffling of `AnimalShape`'s members would silently renumber every category, and a set
    comparison alone would not catch it.

    """
    stems = sorted(entry.name.removesuffix(".svg") for entry in _ZOO.iterdir() if entry.name.endswith(".svg"))
    assert stems == sorted(shape.value for shape in AnimalShape)
    assert tuple(shape.value for shape in AnimalShape) == PINNED_ANIMAL_ORDER


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_document_records_public_domain_provenance(name: str) -> None:
    """Every packaged silhouette carries a source URL and a public-domain license.

    The artwork is redistributed inside the wheel, so provenance is a licensing obligation rather than documentation
    polish: an entry that lost its license field, or gained a share-alike one, must fail the build.

    """
    source = ANIMAL_SOURCES[name]
    assert set(source) >= {"origin", "title", "license", "note"}
    assert source["origin"].startswith("https://")
    assert source["license"] in PUBLIC_DOMAIN_LICENSES


def test_every_animal_has_a_keypoint_table() -> None:
    """The outline and landmark vocabularies cover exactly the same animals.

    `Task.KEYPOINTS` looks a landmark table up by the shape it just drew, so an animal with an outline but no table
    would fail mid-generation rather than at configuration time.

    """
    assert sorted(ANIMAL_KEYPOINTS) == sorted(ANIMAL_POLYGONS)


def test_keypoint_colors_are_consistent_across_every_document() -> None:
    """One fill per landmark name, the same in all twelve documents, and never shared between names.

    The colors are a navigation aid: knowing that blue is always the neck only works if no document disagrees and no
    two landmarks look alike. Derived from the packaged assets rather than pinned against a constant, because the
    library itself never reads a fill — only the authoring editor does.

    """
    fills: dict[str, set[str]] = {}
    for name in ANIMAL_NAMES:
        root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - our own packaged asset, not untrusted input
        circles_group = root.find(f"{svg_tag('g')}[@id='keypoints']")
        assert circles_group is not None
        circles = circles_group.findall(svg_tag("circle"))
        assert circles
        for circle in circles:
            kpt_name = circle.get(zoo_attr("name"))
            fill_color = circle.get("fill")
            if kpt_name is not None and fill_color is not None:
                fills.setdefault(kpt_name, set()).add(fill_color)

    inconsistent = {key: sorted(value) for key, value in fills.items() if len(value) != 1}
    assert inconsistent == {}
    # fish and whale omit the ear and the four hind landmarks, so only the mandatory names are guaranteed present
    assert set(fills) >= set(ANIMAL_KEYPOINT_NAMES) - _OPTIONAL_KEYPOINTS
    used = [next(iter(value)) for value in fills.values()]
    assert len(set(used)) == len(used)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoint_table_holds_sixteen_points(name: str) -> None:
    """Each animal carries exactly one `(x, y)` per schema name.

    YOLO's pose format declares a single dataset-wide `kpt_shape`, so a table of a different length cannot be
    represented at all — it has to be caught here.

    """
    assert ANIMAL_KEYPOINTS[name].shape == (len(ANIMAL_KEYPOINT_NAMES), 2)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoints_are_distinct_positions(name: str) -> None:
    """No two *present* landmarks of an animal sit on the same point.

    Two coincident landmarks would train a model on contradictory targets for two different names; it is also the exact
    symptom of a table authored by copying a neighbouring row. Restricted to present rows: `hash(nan)` is identity-based
    since Python 3.10, so an absent (all-NaN) row would otherwise count as trivially "distinct" without this filter, and
    the test would check one row fewer than it appears to for an animal with no hind limb.

    """
    table = ANIMAL_KEYPOINTS[name]
    present = [(round(float(x), 6), round(float(y), 6)) for x, y in table if not np.isnan(x)]
    assert len(set(present)) == len(present)


@pytest.mark.parametrize("angle", [0.0, 0.9])
@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoints_lie_inside_the_rasterized_silhouette(name: str, angle: float) -> None:
    """Every landmark falls on a filled pixel of the animal it annotates, rotated or not.

    This is the load-bearing property of a hand-placed table: a landmark that drifts off the silhouette teaches a model
    a point that is not on the object. Rasterizing rather than testing the polygon analytically also proves the
    landmark pipeline applies the same scale/rotate/translate as the outline pipeline.

    """
    size, canvas = 240.0, 320
    mask = _rasterize(name, size, canvas, angle=angle)
    points = animal_keypoints(AnimalShape(name), center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    outside = [
        key
        for key, (x, y) in zip(ANIMAL_KEYPOINT_NAMES, points, strict=True)
        if not np.isnan(x) and not _rounded_pixel_is_filled(mask, x, y)
    ]
    assert outside == []


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoint_table_is_frozen_against_mutation(name: str) -> None:
    """The shared landmark table rejects in-place writes.

    Like the outlines, these tables are module-level constants handed to every caller; a consumer mutating one would
    corrupt every later sample in the process.

    """
    with pytest.raises(ValueError, match="read-only"):
        ANIMAL_KEYPOINTS[name][0, 0] = 99.0


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_placed_keypoints_do_not_alias_the_table(name: str) -> None:
    """`animal_keypoints` returns a fresh writable array rather than a view of the constant.

    Callers are free to translate or clip the returned points; aliasing the frozen table would either raise or silently
    corrupt the vocabulary for the rest of the process.

    """
    points = animal_keypoints(AnimalShape(name), center=(0.0, 0.0), size=10.0)
    points[0, 0] += 1.0  # must not raise
    assert points.base is not ANIMAL_KEYPOINTS[name]


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_placed_keypoints_stay_within_the_placed_polygon(name: str) -> None:
    """Landmarks land inside the bounding box of the outline placed with the same arguments.

    The generator derives the annotation box from the polygon and the landmarks from the table; if the two pipelines
    disagreed about centre, scale or rotation, the points would sit outside the very box they are exported with.

    """
    center, size, angle = (120.0, 80.0), 40.0, 0.9
    poly = shape_outline(name, center=center, size=size, angle=angle)
    points = animal_keypoints(AnimalShape(name), center=center, size=size, angle=angle)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    present = points[~np.isnan(points[:, 0])]
    # A hand-placed extremity landmark (`nose`, `tail`, either limb) sits close to the silhouette
    # boundary, so integer authoring coordinates can leave it a hair proud of the outline. The
    # tolerance scales with `size` to stay a fixed fraction of the shape: it absorbs that
    # quantization while still failing loudly on the thing this test is for, a centre/scale/rotation
    # disagreement between the two pipelines, which would put points whole pixels outside the box.
    tol = 1e-3 * size
    assert np.all(present[:, 0] >= x1 - tol)
    assert np.all(present[:, 0] <= x2 + tol)
    assert np.all(present[:, 1] >= y1 - tol)
    assert np.all(present[:, 1] <= y2 + tol)


def test_animal_keypoints_rejects_a_shape_without_a_table() -> None:
    """A geometric shape has no landmark table and is refused with a listing of the ones that do.

    A square is four-fold symmetric, so a fixed landmark on it carries no stable identity; failing loudly here is what
    turns that modelling fact into an actionable error instead of a `KeyError`.

    """
    with pytest.raises(ValueError, match="no keypoint table"):
        animal_keypoints(PrimitiveShape.SQUARE, center=(0.0, 0.0), size=10.0)


def test_normalized_pair_rejects_a_wrong_sized_landmark_table() -> None:
    """A document whose landmark table is not exactly sixteen points is refused at load time.

    The loader runs at import, so a malformed zoo file must fail loudly on import rather than produce an animal that
    breaks only once a keypoints run reaches it.

    """
    outline = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="exactly 16"):
        _normalized_pair(outline, [(0.5, 0.5), (1.0, 0.5)], ANIMAL_KEYPOINT_NAMES)


def test_normalized_pair_maps_landmarks_through_the_outline_frame() -> None:
    """Landmarks are centred and scaled by the *outline's* mean and extent, not their own.

    Normalizing the sixteen points independently would re-centre them on their own mean and detach them from the
    silhouette; this pins the shared-frame contract that keeps a landmark on the animal.

    """
    outline = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    corners = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    landmarks_in = [(10.0, 5.0), *corners, *corners, *corners, *corners[:3]]  # 1 centre + 15 corner repeats = 16
    polygon, landmarks = _normalized_pair(outline, landmarks_in, ANIMAL_KEYPOINT_NAMES)
    assert landmarks[0] == pytest.approx([0.0, 0.0])  # outline centre maps to the origin
    assert landmarks[1:5] == pytest.approx(polygon)  # corners map onto the normalized corners


def test_normalized_pair_rejects_a_half_nan_landmark_row() -> None:
    """A landmark with exactly one NaN coordinate is refused rather than silently treated as absent.

    A real absence (no hind limbs on a whale) is NaN in *both* coordinates; a single NaN is a parser bug — reading only
    one of `cx`/`cy` off a malformed circle, say — and must fail loudly rather than propagate as a half-broken point.

    """
    outline = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    landmarks = [(float("nan"), 5.0), *([(1.0, 1.0)] * 15)]
    with pytest.raises(ValueError, match="NaN"):
        _normalized_pair(outline, landmarks, ANIMAL_KEYPOINT_NAMES)


def test_absent_keypoints_match_the_pinned_matrix() -> None:
    """Each animal's absent-landmark set matches a literal, hand-reviewed pin, not just "whatever came out".

    The absent-slot matrix is decided per animal when its SVG is authored — pinning it as a literal means an edit that
    silently changes which animal has a hind limb is a diff a reviewer sees, not a value that just drifts.

    """
    for name in ANIMAL_NAMES:
        table = ANIMAL_KEYPOINTS[name]
        absent = {ANIMAL_KEYPOINT_NAMES[i] for i in range(len(ANIMAL_KEYPOINT_NAMES)) if np.isnan(table[i, 0])}
        assert absent == ABSENT_KEYPOINTS[name]
        assert absent <= _OPTIONAL_KEYPOINTS  # only the ear and the four hind-leg points may ever be absent


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_svg_is_well_formed(name: str) -> None:
    """Every packaged SVG parses without error and has the SVG root tag.

    A hand-edit in Inkscape (or a bug in the authoring script) that produces invalid XML must fail here rather than
    surface as an opaque `ElementTree.ParseError` deep inside module import.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    assert root.tag == svg_tag("svg")


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_svg_vertices_lie_inside_the_viewbox(name: str) -> None:
    """Every outline vertex and every keypoint circle sits within the declared `0 0 1000 1000` viewBox.

    A vertex outside the viewBox would still parse and load fine but renders as a clipped or invisible artifact in any
    SVG viewer — an authoring-time regression this only-at-load-time loader cannot otherwise catch.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    (path_el,) = root.findall(svg_tag("path"))
    points = parse_path_d(path_el.get("d"), name)
    for x, y in points:
        assert 0 <= x <= 1000
        assert 0 <= y <= 1000
    group = root.find(f"{svg_tag('g')}[@id='keypoints']")
    if group is not None:
        for circle in group.findall(svg_tag("circle")):
            cx = circle.get("cx")
            cy = circle.get("cy")
            if cx is not None and cy is not None:
                assert 0 <= float(cx) <= 1000
                assert 0 <= float(cy) <= 1000


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_svg_title_matches_zoo_title(name: str) -> None:
    """The visible `<title>` element and the machine-read `zoo:title` attribute agree.

    Both carry the same provenance text by design (`<title>` for a human hovering in a viewer, `zoo:title` for the
    loader) — letting them drift would mean the file lies to whichever reader looks at the "wrong" one.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    title_el = root.find(svg_tag("title"))
    assert title_el is not None
    assert title_el.text == root.get(zoo_attr("title"))


def test_read_svg_rejects_a_document_without_exactly_one_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zoo document with zero `<path>` elements is refused, naming the count found.

    `_read_svg` expects exactly one outline path per animal; an authoring mistake that drops the path (or,
    symmetrically, duplicates it into a second `<path>`) must fail loudly at load time rather than surface as an empty
    or ambiguous outline deep inside a generation run.

    """
    monkeypatch.setattr("synth_datasets.families.animals._ZOO", tmp_path)
    (tmp_path / "test.svg").write_text(f'<svg xmlns="{_SVG_NS}"><g id="keypoints"></g></svg>', encoding="utf-8")
    with pytest.raises(ValueError, match=r"exactly one <path>, found 0"):
        _read_svg("test")


def test_read_svg_rejects_a_document_missing_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A document with a valid closed path but no `zoo:` provenance attributes is refused, naming what's missing.

    Provenance (`origin`/`title`/`license`/`note`) is a licensing obligation for redistributed public-domain art, not
    documentation polish, so a document authored without it must fail at load time instead of shipping an animal with no
    attribution trail.

    """
    monkeypatch.setattr("synth_datasets.families.animals._ZOO", tmp_path)
    (tmp_path / "test.svg").write_text(
        f'<svg xmlns="{_SVG_NS}"><path d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/></svg>', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"missing the key\(s\)"):
        _read_svg("test")


def test_parse_path_d_rejects_a_curve_command() -> None:
    """A `C`/`S`/`Q`/`T`/`A` command is refused with the Inkscape-flatten hint.

    The authoring script and this loader only ever speak straight lines; a curve reaching the parser means someone drew
    with the pen tool instead of flattening first, and the message should say so.

    """
    with pytest.raises(ValueError, match="curve command"):
        parse_path_d("M 0 0 C 1 1 2 2 3 3 Z", "test")


def test_parse_path_d_rejects_a_second_subpath() -> None:
    """A second `M`/`m` mid-path (a second subpath) is refused; a zoo outline is exactly one closed path."""
    with pytest.raises(ValueError, match="second subpath"):
        parse_path_d("M 0 0 L 1 1 Z M 5 5 L 6 6 Z", "test")


def test_parse_path_d_rejects_an_unclosed_path() -> None:
    """A path missing its trailing `Z`/`z` is refused rather than silently treated as closed."""
    with pytest.raises(ValueError, match="not closed"):
        parse_path_d("M 0 0 L 1 1 L 2 2", "test")


def test_parse_path_d_rejects_an_unsupported_command() -> None:
    """A recognised-but-unsupported letter (not curve, not M/L/H/V/Z) is refused, not silently skipped."""
    with pytest.raises(ValueError, match="unsupported command"):
        parse_path_d("M 0 0 L 1 1 B 2 2 Z", "test")


def test_parse_path_d_rejects_coordinates_after_the_close() -> None:
    """A coordinate pair trailing the closing `Z`/`z` is refused rather than absorbed as another vertex.

    `Z` takes no arguments, so a trailing pair is malformed data an editor should never emit. Left unguarded it fell
    through every branch of the coordinate dispatch to the relative-lineto default and grew the outline by a phantom
    vertex — a silently reshaped silhouette rather than a load-time error.

    """
    with pytest.raises(ValueError, match="after the closing Z"):
        parse_path_d("M 0 0 L 1 1 Z 2 2", "test")


def test_parse_path_d_rejects_a_moveto_without_a_coordinate_pair() -> None:
    """A moveto immediately followed by another command letter is refused, not silently skipped.

    Defense-in-depth against a hand-edited or third-party document: this package's own writer always pairs `M` with
    coordinates. Unguarded, the argument-less moveto simply produced no vertex, so the outline came back one point
    short — a quietly reshaped silhouette instead of a load-time error naming the offending token.

    """
    with pytest.raises(ValueError, match="moveto without a coordinate pair"):
        parse_path_d("M L 1 1 L 2 2 Z", "test")


def test_parse_path_d_rejects_path_data_ending_mid_coordinate_pair() -> None:
    """Path data stopping between the x and the y of a pair is refused by name, not by IndexError.

    A truncated `d` attribute — a copy-paste that lost its tail, or an exporter that flushed short — used to index one
    token past the end and surface as a bare `IndexError` naming no document, breaking the load-time contract every
    sibling failure mode here honours.

    """
    with pytest.raises(ValueError, match="ends mid-coordinate-pair"):
        parse_path_d("M 0 0 L 1 1 L 2", "test")


def test_parse_path_d_rejects_a_command_where_a_coordinate_belongs() -> None:
    """A command letter standing in for the second half of a pair is refused by name, not by a bare float() error.

    `M 0 0 L 1 1 L 2 Z` reads as a lineto whose y is the close command. Unguarded, `float("Z")` raised `could not
    convert string to float: 'Z'` — a ValueError that names neither the document nor the offending construct, so the
    animal that failed to load stayed anonymous.

    """
    with pytest.raises(ValueError, match="inside a coordinate pair"):
        parse_path_d("M 0 0 L 1 1 L 2 Z", "test")


@pytest.mark.parametrize(
    "d",
    [
        pytest.param("M 10 10 L 20 10 L 20 20 L 10 20 Z", id="absolute"),
        pytest.param("m 10 10 l 10 0 l 0 10 l -10 0 z", id="relative"),
        pytest.param("M 10 10 H 20 V 20 H 10 Z", id="horizontal-vertical"),
        pytest.param("M 10 10 20 10 20 20 10 20 Z", id="implicit-lineto"),
    ],
)
def test_parse_path_d_tolerant_forms_agree(d: str) -> None:
    """Absolute, relative, H/V, and implicit-lineto forms of the same square parse to the same vertices.

    Inkscape's default save is relative with H/V shorthand; the authoring script always writes absolute M/L/Z. Both must
    round-trip to the same geometry, or a save-in-Inkscape-then-reload cycle silently reshapes the outline.

    """
    expected = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    assert parse_path_d(d, "test") == expected


def test_reject_transforms_flags_the_offending_element() -> None:
    """A `transform` attribute anywhere in the document is refused, naming the element and the Inkscape fix."""
    root = ET.fromstring(f'<svg xmlns="{_SVG_NS}"><path transform="translate(1,1)" d="M 0 0 Z"/></svg>')  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    with pytest.raises(ValueError, match="transform"):
        reject_transforms(root, "test")


def test_read_keypoints_rejects_an_unknown_zoo_name() -> None:
    """A `zoo:name` outside the sixteen-name schema is refused rather than silently ignored."""
    root = ET.fromstring(  # noqa: S314 - fixed literal XML, not untrusted input
        f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}">'
        f'<g id="keypoints"><circle zoo:name="wing" cx="1" cy="1"/></g></svg>'
    )
    with pytest.raises(ValueError, match="unknown"):
        read_named_circles(root, "test", "keypoints", ANIMAL_KEYPOINT_NAMES, _REQUIRED_KEYPOINTS)


def test_read_keypoints_rejects_a_duplicate_zoo_name() -> None:
    """Two circles claiming the same `zoo:name` are refused; only one landmark per name is meaningful."""
    root = ET.fromstring(  # noqa: S314 - fixed literal XML, not untrusted input
        f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">'
        f'<circle zoo:name="mouth" cx="1" cy="1"/><circle zoo:name="mouth" cx="2" cy="2"/>'
        "</g></svg>"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_named_circles(root, "test", "keypoints", ANIMAL_KEYPOINT_NAMES, _REQUIRED_KEYPOINTS)


def test_read_keypoints_rejects_a_missing_mandatory_landmark() -> None:
    """A document missing a mandatory (non-optional) landmark is refused, listing what is missing.

    The reader is shared with the symbol and letter families now, so it says "point" rather than "landmark" — the
    animals' anatomical word for the same thing.

    """
    root = ET.fromstring(  # noqa: S314 - fixed literal XML, not untrusted input
        f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">'
        f'<circle zoo:name="mouth" cx="1" cy="1"/></g></svg>'
    )
    with pytest.raises(ValueError, match="missing the point"):
        read_named_circles(root, "test", "keypoints", ANIMAL_KEYPOINT_NAMES, _REQUIRED_KEYPOINTS)


@pytest.mark.parametrize(
    ("kp_name", "attrs"),
    [
        pytest.param("mouth", 'cy="1"', id="mandatory-without-cx"),
        pytest.param("mouth", 'cx="1"', id="mandatory-without-cy"),
        pytest.param("mouth", "", id="mandatory-without-either"),
        pytest.param("hind_knee_left", "", id="optional-without-either"),
    ],
)
def test_read_keypoints_rejects_a_landmark_without_coordinates(kp_name: str, attrs: str) -> None:
    """A circle carrying a `zoo:name` but no `cx`/`cy` is refused at parse time, optional names included."""
    others = [name for name in ANIMAL_KEYPOINT_NAMES if name not in _OPTIONAL_KEYPOINTS and name != kp_name]
    circles = "".join(f'<circle zoo:name="{other}" cx="1" cy="1"/>' for other in others)
    circles += f'<circle zoo:name="{kp_name}" {attrs}/>'
    root = ET.fromstring(f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">{circles}</g></svg>')  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    # The mandatory-landmark guard only tests key presence, so a NaN-defaulted circle passed it.
    with pytest.raises(ValueError, match="missing cx/cy"):
        read_named_circles(root, "test", "keypoints", ANIMAL_KEYPOINT_NAMES, _REQUIRED_KEYPOINTS)


def test_read_keypoints_allows_missing_optional_landmarks() -> None:
    """A document carrying every mandatory landmark but none of the optional ones (ear, hind legs) still loads."""
    mandatory = [name for name in ANIMAL_KEYPOINT_NAMES if name not in _OPTIONAL_KEYPOINTS]
    circles = "".join(f'<circle zoo:name="{name}" cx="1" cy="1"/>' for name in mandatory)
    root = ET.fromstring(f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">{circles}</g></svg>')  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    present = read_named_circles(root, "test", "keypoints", ANIMAL_KEYPOINT_NAMES, _REQUIRED_KEYPOINTS)
    assert set(present) == set(mandatory)
