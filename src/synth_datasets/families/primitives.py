"""The analytic shape family: outlines computed from ``size`` rather than read from an asset.

This is the fourth shape family alongside :mod:`~synth_datasets.families.animals`,
:mod:`~synth_datasets.families.symbols`, and :mod:`~synth_datasets.families.letters`, and the only
one with no packaged artwork behind it — a square is four lines of NumPy, not a traced silhouette.
It is also the only family with no landmark table, which is deliberate rather than an omission: a
square is 4-fold symmetric and a circle rotation-invariant, so a fixed landmark on either has no
identity a model could learn.

Splitting the family out of :mod:`~synth_datasets.families.geometry` leaves that module holding only
family-agnostic math (rotation, box derivation, convex hull), which is what lets
:mod:`~synth_datasets.families` import every family uniformly without the import cycle the
old arrangement needed deferred imports to dodge.

Examples:
    ```pycon
    >>> from synth_datasets.families.primitives import PrimitiveShape, primitive_outline
    >>> [shape.value for shape in PrimitiveShape]
    ['square', 'rectangle', 'triangle', 'circle']
    >>> primitive_outline("square", 2.0).tolist()
    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]

    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from synth_datasets.families.shape_enum import ShapeEnum

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Number of vertices used to approximate a circle outline.
CIRCLE_POINTS = 32

#: Height-to-width ratio for the ``rectangle`` shape (non-square, so its OBB is oriented).
RECT_ASPECT = 0.5


class PrimitiveShape(ShapeEnum):
    """Analytically computed shape vocabulary (definition order is the class order).

    Computed from ``size`` rather than looked up in a table. ``RECTANGLE`` is deliberately
    non-square and every shape but ``CIRCLE`` takes a per-shape rotation, so oriented bounding
    boxes carry real orientation variety; ``CIRCLE`` is rotation-invariant, so its OBB collapses to
    the axis-aligned box. None of them carries a landmark table: a square is 4-fold symmetric and a
    circle rotation-invariant, so a fixed landmark on them has no identity a model could learn.

    Attributes:
        SQUARE: Axis-aligned equal-sided quadrilateral.
        RECTANGLE: Non-square quadrilateral.
        TRIANGLE: Equilateral triangle, apex up — mirror-symmetric about its vertical axis
            like every other upright shape in the package. Its 3-fold rotational symmetry means
            the silhouette alone cannot distinguish rotations 120 degrees apart; the exported
            oriented box still records the drawn angle exactly.
        CIRCLE: Polygon-approximated circle.

    Examples:
        ```pycon
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> [shape.value for shape in PrimitiveShape]
        ['square', 'rectangle', 'triangle', 'circle']

        ```

    """

    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"


def primitive_outline(value: str, size: float) -> NDArray[np.float64]:
    """Return the origin-centered outline for one :class:`PrimitiveShape` value.

    Shares the unit convention every asset-backed family's table already uses — centered on the
    area centroid, larger extent equal to ``size`` — so
    :func:`~synth_datasets.families.shape_outline` can treat analytic and table-backed
    families identically.

    Args:
        value: A :class:`PrimitiveShape` value: ``"square"``, ``"rectangle"``, ``"triangle"``, or
            ``"circle"``.
        size: Bounding size (side / diameter / larger extent) in pixels.

    Returns:
        ``(num_points, 2)`` float array centered at the origin.

    Raises:
        ValueError: If ``value`` names no :class:`PrimitiveShape` member.

    Examples:
        ```pycon
        >>> from synth_datasets.families.primitives import primitive_outline
        >>> primitive_outline("circle", 2.0).shape
        (32, 2)

        ```

    """
    half = size / 2.0
    if value == PrimitiveShape.SQUARE.value:
        return np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float64)
    if value == PrimitiveShape.RECTANGLE.value:
        h = half * RECT_ASPECT
        return np.array([[-half, -h], [half, -h], [half, h], [-half, h]], dtype=np.float64)
    if value == PrimitiveShape.TRIANGLE.value:
        # Equilateral, apex up, side == size, centered on the centroid (which for a triangle is
        # the vertex mean), matching every other shape's centered-at-origin convention.
        height = size * np.sqrt(3.0) / 2.0
        return np.array([[0.0, -2.0 * height / 3.0], [half, height / 3.0], [-half, height / 3.0]], dtype=np.float64)
    if value == PrimitiveShape.CIRCLE.value:
        angles = np.linspace(0.0, 2.0 * np.pi, CIRCLE_POINTS, endpoint=False)
        return np.stack([half * np.cos(angles), half * np.sin(angles)], axis=1).astype(np.float64)
    known = ", ".join(shape.value for shape in PrimitiveShape)
    raise ValueError(f"unknown primitive shape {value!r}; expected one of {known}")
