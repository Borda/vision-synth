"""Geometry tests: polygon construction, rotation, and box derivation."""

from __future__ import annotations

import numpy as np
import pytest

from synth_datasets.families import shape_outline
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.geometry import bbox_iou, polygon_to_bbox_xyxy, polygon_to_obb, rotate_polygon
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SymbolShape

from ._obb_pose import rebuild_error

#: Every drawable shape across all four families — the OBB contract is family-agnostic, so the
#: box-derivation tests below sweep the whole vocabulary rather than one family's roster.
ALL_SHAPE_VALUES = [s.value for s in (*PrimitiveShape, *AnimalShape, *SymbolShape, *LetterShape)]


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _area_centroid(points: np.ndarray) -> np.ndarray:
    """Independent shoelace-formula centroid, kept separate from `landmarks._polygon_centroid`."""
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def _rect_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


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
    pairs = [(i, j) for i in range(count) for j in range(i + 2, count - (1 if i == 0 else 0))]
    return [(i, j) for i, j in pairs if _segments_intersect(*edges[i], *edges[j])]


@pytest.mark.parametrize("shape", [s.value for s in (*PrimitiveShape, *AnimalShape, *SymbolShape)])
def test_shape_outline_is_centered(shape: str) -> None:
    """Polygon shapes are centered at the specified point, by area centroid (not vertex mean).

    A traced or hand-authored outline can have vertices bunched unevenly along its edges (an arrow's barbs, an animal's
    long neck), so the vertex mean of a placed polygon need not sit on the requested center — only its area centroid
    (center of mass) is guaranteed to.

    """
    poly = shape_outline(shape, center=(50.0, 50.0), size=20.0)
    assert poly.ndim == 2
    assert poly.shape[1] == 2
    centroid = _area_centroid(poly)
    assert np.allclose(centroid, [50.0, 50.0], atol=1e-6)


def test_square_bbox_exact() -> None:
    """Square bbox computation returns exact coordinate tuple."""
    poly = shape_outline("square", center=(5.0, 5.0), size=4.0)
    assert polygon_to_bbox_xyxy(poly) == (3.0, 3.0, 7.0, 7.0)


def test_rectangle_is_non_square() -> None:
    """Rectangle bounding box has different width and height."""
    poly = shape_outline("rectangle", center=(0.0, 0.0), size=10.0)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    assert (x2 - x1) != pytest.approx(y2 - y1)


@pytest.mark.parametrize("shape", ["square", "rectangle", "triangle"])
def test_rotation_preserves_area(shape: str) -> None:
    """Polygon rotation preserves area to floating point precision."""
    poly = shape_outline(shape, center=(30.0, 30.0), size=12.0)
    rotated = rotate_polygon(poly, np.pi / 5, center=(30.0, 30.0))
    assert _polygon_area(rotated) == pytest.approx(_polygon_area(poly), rel=1e-6)


def test_zero_skew_matches_the_unskewed_polygon() -> None:
    """`skew=0.0` is a true no-op, byte-identical to omitting the argument.

    This is the property that keeps every pre-existing seeded configuration's output unchanged:
    `SyntheticConfig.asymmetry_jitter` defaults to `0.0`, so the generator must never observe a difference from before
    the knob existed.

    """
    plain = shape_outline("square", center=(10.0, 10.0), size=8.0, angle=0.3)
    skewed = shape_outline("square", center=(10.0, 10.0), size=8.0, angle=0.3, skew=0.0)
    assert np.array_equal(plain, skewed)


def test_positive_skew_narrows_only_the_right_half_of_a_square() -> None:
    """A positive `skew` scales the `x > 0` half toward the axis, leaving the `x < 0` half fixed."""
    poly = shape_outline("square", center=(0.0, 0.0), size=10.0, skew=0.3)
    xs = sorted(poly[:, 0])
    assert xs[:2] == pytest.approx([-5.0, -5.0])  # left edge untouched
    assert xs[2:] == pytest.approx([3.5, 3.5])  # right edge narrowed by 30%


def test_negative_skew_narrows_only_the_left_half_of_a_square() -> None:
    """A negative `skew` narrows the `x < 0` half instead, mirroring the positive case."""
    poly = shape_outline("square", center=(0.0, 0.0), size=10.0, skew=-0.3)
    xs = sorted(poly[:, 0])
    assert xs[:2] == pytest.approx([-3.5, -3.5])  # left edge narrowed by 30%
    assert xs[2:] == pytest.approx([5.0, 5.0])  # right edge untouched


@pytest.mark.parametrize("shape", [s.value for s in (*PrimitiveShape, *AnimalShape, *SymbolShape)])
@pytest.mark.parametrize("skew", [0.49, -0.49])
def test_skew_near_its_upper_bound_keeps_every_shape_simple(shape: str, skew: float) -> None:
    """Every shape stays a simple (non-self-intersecting) polygon at the edge of the allowed skew range.

    `SyntheticConfig.asymmetry_jitter` caps at just under `0.5`; this pins that the cap is actually safe for every shape
    this package draws, not just the ones exercised by example configs.

    """
    poly = shape_outline(shape, center=(0.0, 0.0), size=10.0, skew=skew)
    assert _self_intersections(poly) == []


def test_obb_at_zero_angle_equals_aabb() -> None:
    """With no rotation the oriented box is exactly the axis-aligned box, corner for corner."""
    poly = shape_outline("rectangle", center=(40.0, 40.0), size=16.0, angle=0.0)
    corners = polygon_to_obb(poly)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    expected = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    assert corners.shape == (4, 2)
    np.testing.assert_allclose(corners, expected)


@pytest.mark.parametrize("shape", ALL_SHAPE_VALUES)
def test_obb_area_is_angle_invariant(shape: str) -> None:
    """The box area never changes as the shape turns — it is the upright-frame box carried rigidly.

    The minimum-area box this replaced had the same property, but the axis-aligned *detection* box does not: this is
    what separates `polygon_to_obb`'s output from a lazily re-fit AABB at each pose.

    """
    center, size = (0.0, 0.0), 100.0
    reference_area = _polygon_area(polygon_to_obb(shape_outline(shape, center=center, size=size, angle=0.0)))
    for angle_deg in (10, 47, 91, 133, 179, 250, 311):
        angle = np.radians(float(angle_deg))
        area = _polygon_area(polygon_to_obb(shape_outline(shape, center=center, size=size, angle=angle), angle))
        assert area == pytest.approx(reference_area, rel=1e-9), f"{shape} at {angle_deg} degrees"


@pytest.mark.parametrize("shape", ALL_SHAPE_VALUES)
def test_obb_contains_its_polygon(shape: str) -> None:
    """Every polygon vertex sits inside (or on) the oriented box, at every pose.

    The upright-frame box is a bounding box by construction only in the de-rotated frame; this pins that the rotate-
    back places it around the image-frame polygon too — a pivot mismatch between the de-rotation and the re-rotation
    would break exactly this.

    """
    center, size = (200.0, 200.0), 100.0
    for angle_deg in (0, 10, 47, 91, 133, 179, 250, 311):
        angle = np.radians(float(angle_deg))
        poly = shape_outline(shape, center=center, size=size, angle=angle)
        corners = polygon_to_obb(poly, angle)
        # Express vertices in the box's own edge basis; containment is a per-axis interval check.
        u = corners[1] - corners[0]
        v = corners[3] - corners[0]
        rel = poly - corners[0]
        for axis in (u, v):
            along = rel @ axis / float(axis @ axis)
            assert along.min() >= -1e-9, f"{shape} at {angle_deg} degrees"
            assert along.max() <= 1.0 + 1e-9, f"{shape} at {angle_deg} degrees"


def test_circle_obb_equals_aabb() -> None:
    """A circle is rotation-invariant, is never rotated by the generator, and gets the plain AABB."""
    poly = shape_outline("circle", center=(20.0, 20.0), size=10.0)
    obb_area = _polygon_area(polygon_to_obb(poly))
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area == pytest.approx(aabb_area, rel=1e-9)


@pytest.mark.parametrize("shape", ALL_SHAPE_VALUES)
def test_obb_places_the_reference_shape_back_onto_the_drawn_one(shape: str) -> None:
    """Turning the upright reference shape by what its OBB says lands it back on the drawn shape.

    Every other OBB test here checks a property *of the box* — its corner count, that it rotates rigidly, that it
    contains the outline. None checks the thing the box actually promises a consumer: that it describes the pose of the
    object inside it. This closes the loop by discarding the drawn polygon's own coordinates entirely, rebuilding it
    from the reference outline plus the four box corners (see `_obb_pose.rebuild_error`), and requiring the rebuild to
    land on the original to float precision.

    """
    center, size = (200.0, 200.0), 100.0
    for angle_deg in (10, 47, 91, 133, 179, 250, 311):
        angle = np.radians(float(angle_deg))
        drawn = shape_outline(shape, center=center, size=size, angle=angle)
        error = rebuild_error(shape, polygon_to_obb(drawn, angle), drawn)
        assert error < 1e-9 * size, f"{shape} at {angle_deg} degrees is off by {error:.3g}px"


@pytest.mark.parametrize("shape", ALL_SHAPE_VALUES)
def test_obb_is_rotation_equivariant(shape: str) -> None:
    """The box rotates rigidly with the shape: box(angle) is exactly box(0) turned by angle.

    The minimum-area box this replaced could only promise this up to a tie-break between equal-area candidates; the
    upright-frame box has no candidates to tie — the de-rotated outline is angle-independent to float precision, so the
    corners match a rigid rotation of the upright box exactly, in the same corner order.

    """
    center, size = (0.0, 0.0), 100.0
    reference = polygon_to_obb(shape_outline(shape, center=center, size=size, angle=0.0))
    for angle_deg in (10, 47, 91, 133, 179, 250, 311):
        angle = np.radians(float(angle_deg))
        actual = polygon_to_obb(shape_outline(shape, center=center, size=size, angle=angle), angle)
        expected = rotate_polygon(reference, angle, center=center)
        np.testing.assert_allclose(actual, expected, atol=1e-9, err_msg=f"{shape} at {angle_deg} degrees")


def test_unknown_shape_raises() -> None:
    """Unknown shape name raises ValueError."""
    with pytest.raises(ValueError, match="unknown shape"):
        shape_outline("hexagon", center=(0.0, 0.0), size=4.0)


def test_bbox_iou_known_value() -> None:
    """Bbox IOU for two overlapping boxes matches expected value."""
    assert bbox_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_bbox_iou_disjoint_is_zero() -> None:
    """Bbox IOU for disjoint boxes is zero."""
    assert bbox_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
