"""Family-agnostic polygon math: placement, rotation, and box derivation.

Pure NumPy, no image-library dependency and no knowledge of which shape families exist. Polygons are
``(num_points, 2)`` float arrays of ``(x, y)`` pixel coordinates. Boxes are derived from the polygon
so annotations always match the rasterized pixels.

The analytic shape family that used to live here moved to
:mod:`~synth_datasets.families.primitives`, and the per-family outline dispatch moved to
:mod:`~synth_datasets.families`. What is left is the math every family shares — which is
what lets each family module import this one without the import cycle the old arrangement dodged
with deferred imports.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from synth_datasets.families.geometry import polygon_to_bbox_xyxy
    >>> square = np.array([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]])
    >>> polygon_to_bbox_xyxy(square)
    (3.0, 3.0, 7.0, 7.0)

    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _rotation_matrix(angle: float) -> NDArray[np.float64]:
    """Return the 2x2 counter-clockwise rotation matrix for ``angle`` radians."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


def rotate_polygon(
    points: NDArray[np.float64], angle: float, center: tuple[float, float] = (0.0, 0.0)
) -> NDArray[np.float64]:
    """Rotate ``points`` by ``angle`` radians about ``center``.

    Args:
        points: ``(num_points, 2)`` array of ``(x, y)`` coordinates.
        angle: Rotation angle in radians (counter-clockwise).
        center: Pivot point ``(x, y)``.

    Returns:
        Rotated ``(num_points, 2)`` array.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.families.geometry import rotate_polygon
        >>> pts = np.array([[1.0, 0.0]])
        >>> out = rotate_polygon(pts, np.pi / 2)
        >>> bool(np.allclose(out, [[0.0, 1.0]]))
        True

        ```

    """
    pivot = np.asarray(center, dtype=np.float64)
    rotated: NDArray[np.float64] = (points - pivot) @ _rotation_matrix(angle).T + pivot
    return rotated


def _skewed(points: NDArray[np.float64], skew: float) -> NDArray[np.float64]:
    """Narrow one half of origin-centered ``points`` toward the vertical axis.

    Every shape in this package but :attr:`PrimitiveShape.CIRCLE` is drawn mirror-symmetric about its
    own local vertical axis (before rotation), so its oriented bounding box would otherwise show
    identical margins on both sides of that axis — see
    :attr:`~synth_datasets.core.config.SyntheticConfig.asymmetry_jitter`. This breaks that
    symmetry per placed instance instead.

    Args:
        points: ``(num_points, 2)`` array already centered on the origin, in the shape's own
            pre-rotation frame — "left" and "right" are only meaningful there, which is why this
            runs before :func:`place_points` applies ``angle``. NaN rows (absent landmarks) pass through
            unchanged: a comparison against NaN is always false, so the selection mask below never
            selects one.
        skew: Signed fraction narrowing one half. Positive narrows the ``x > 0`` half, negative the
            ``x < 0`` half; magnitude is the fractional narrowing (``0.2`` == 20% narrower).

    Returns:
        A new ``(num_points, 2)`` array; ``points`` itself is never mutated.

    """
    narrowed: NDArray[np.float64] = points.copy()
    on_narrowed_side = (points[:, 0] * np.sign(skew)) > 0.0
    narrowed[on_narrowed_side, 0] *= 1.0 - abs(skew)
    return narrowed


def place_points(
    points: NDArray[np.float64], center: tuple[float, float], angle: float, skew: float = 0.0
) -> NDArray[np.float64]:
    """Skew, rotate, and translate origin-centered ``points`` onto ``center``, in that order.

    The single placement implementation every family shares — outlines via
    :func:`~synth_datasets.families.shape_outline`, landmarks via each family's own
    ``*_keypoints`` — so an outline and its landmarks can never drift apart. Skew runs first because
    "left"/"right" are only meaningful in the shape's own pre-rotation frame.

    Args:
        points: ``(num_points, 2)`` array already scaled and centered on the origin.
        center: Target center ``(x, y)`` in pixels.
        angle: Rotation in radians applied about the origin (i.e. about the shape center).
        skew: Signed fraction (see :func:`_skewed`) narrowing one pre-rotation half. ``0.0`` (the
            default) skips the step entirely, so existing callers are unaffected.

    Returns:
        ``(num_points, 2)`` float array in image coordinates.

    """
    if skew:
        points = _skewed(points, skew)
    if angle:
        points = points @ _rotation_matrix(angle).T
    return points + np.asarray(center, dtype=np.float64)


#: Distance between a pixel's edge-space and centre-space coordinate. Pillow fills pixel
#: ``floor(x)`` for a vertex at ``x``, so an outline authored for the rasterizer is in *edge* space
#: (pixel ``i`` spans ``[i, i + 1)``), while the matrices in
#: :mod:`~fused_transforms.targets` resample images -- and therefore move keypoints -- in *centre*
#: space (pixel ``i`` is the point ``i``). The two differ by half a pixel each way, which becomes a
#: full pixel under any reflection, so a point field has to declare which space it is in.
PIXEL_CENTRE_OFFSET = 0.5


def to_pixel_centre(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert edge-space coordinates to the pixel-centre space the point transforms assume.

    Args:
        points: Coordinates in edge space, any shape ending in the coordinate axis.

    Returns:
        The same array shifted by ``-`` :data:`PIXEL_CENTRE_OFFSET`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.families.geometry import to_pixel_centre
        >>> to_pixel_centre(np.array([[3.0, 4.0]]))
        array([[2.5, 3.5]])

        ```

    """
    return np.asarray(points, dtype=np.float64) - PIXEL_CENTRE_OFFSET


def to_pixel_edge(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert pixel-centre coordinates back to the edge space the rasterizer and writers use.

    Inverse of :func:`to_pixel_centre`; the dataset writers apply it so an exported COCO or YOLO
    file stays in one space with the ``bbox`` beside it.

    Args:
        points: Coordinates in pixel-centre space, any shape ending in the coordinate axis.

    Returns:
        The same array shifted by ``+`` :data:`PIXEL_CENTRE_OFFSET`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.families.geometry import to_pixel_edge
        >>> to_pixel_edge(np.array([[2.5, 3.5]]))
        array([[3., 4.]])

        ```

    """
    return np.asarray(points, dtype=np.float64) + PIXEL_CENTRE_OFFSET


def polygon_to_bbox_xyxy(points: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box ``(x_min, y_min, x_max, y_max)`` of a polygon.

    Args:
        points: ``(num_points, 2)`` array of coordinates.

    Returns:
        Bounding box tuple in pixels.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.families.geometry import polygon_to_bbox_xyxy
        >>> polygon_to_bbox_xyxy(np.array([[1.0, 2.0], [3.0, 5.0], [0.0, 4.0]]))
        (0.0, 2.0, 3.0, 5.0)

        ```

    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))


def polygon_to_obb(points: NDArray[np.float64], angle: float = 0.0) -> NDArray[np.float64]:
    """Return the oriented bounding box aligned with the shape's own upright frame.

    The box is the polygon's axis-aligned bounding box *in the shape's pre-rotation frame*,
    carried rigidly through the placement rotation: the polygon is de-rotated by ``-angle``
    about its centroid, its min/max extents taken, and the four corners rotated back by
    ``angle``. Every shape in this package is authored upright (mirror-symmetric about its
    local vertical axis where it has a symmetry at all — see
    :mod:`~synth_datasets.families.symbols`), so the box's sides always run along and across
    that upright axis, matching how a human would draw the box around the object.

    This deliberately is *not* the minimum-area rectangle. A minimum-area box must lie flush
    to a convex-hull edge, so for a shape with no horizontal or vertical hull edge in its
    upright pose (``kite``, ``arrow``, ``teardrop``) it sits tilted against the shape's own
    symmetry axis — geometrically tighter, but visually wrong as a pose annotation.

    Args:
        points: ``(num_points, 2)`` array of polygon coordinates in the image frame.
        angle: Rotation in radians that was applied when the polygon was placed (see
            :func:`place_points`). **Defaults to ``0.0``, which returns the plain axis-aligned
            box** — a rotated polygon passed without its angle gets a loose upright box, not
            the tight oriented one, so real callers must pass the placement angle through.

    Returns:
        ``(4, 2)`` array of corner coordinates in order (a rigid rotation of
        ``(min, min) → (max, min) → (max, max) → (min, max)`` in the de-rotated frame).

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.families.geometry import polygon_to_obb
        >>> corners = polygon_to_obb(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]))
        >>> corners.shape
        (4, 2)
        >>> corners[2] - corners[0]
        array([2., 1.])

        ```

    """
    if not angle:
        x1, y1, x2, y2 = polygon_to_bbox_xyxy(points)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    pivot = points.mean(axis=0)
    upright = (points - pivot) @ _rotation_matrix(-angle).T
    min_xy = upright.min(axis=0)
    max_xy = upright.max(axis=0)
    aligned = np.array(
        [
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ],
        dtype=np.float64,
    )
    corners: NDArray[np.float64] = aligned @ _rotation_matrix(angle).T + pivot
    return corners


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Return the intersection-over-union of two axis-aligned xyxy boxes.

    Args:
        box_a: First box ``(x_min, y_min, x_max, y_max)``.
        box_b: Second box ``(x_min, y_min, x_max, y_max)``.

    Returns:
        IoU in ``[0, 1]``; ``0.0`` when the boxes do not overlap.

    Examples:
        ```pycon
        >>> from synth_datasets.families.geometry import bbox_iou
        >>> bbox_iou((0, 0, 2, 2), (1, 1, 3, 3))
        0.14285714285714285

        ```

    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)
