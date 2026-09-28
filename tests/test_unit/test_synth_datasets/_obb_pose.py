"""Rebuild a shape from its oriented bounding box alone, shared by the geometry and generator tests.

An OBB annotation claims to say where an object is, how big it is, and how it is turned. The way to check that claim is
to act on it: take the shape's own upright reference outline, turn and scale and place it by nothing but the four
corners, and see whether it lands back on the object. Both `test_geometry.py` (against `shape_outline`'s own output) and
`test_generator.py` (against what the generator actually exports) make that check, so the rebuild itself lives here
rather than in either.

Not a test module — it defines no tests and pytest collects nothing from it.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synth_datasets.families import shape_outline
from synth_datasets.families.geometry import polygon_to_obb, rotate_polygon


def box_heading(corners: NDArray[np.float64]) -> float:
    """Return the angle in radians of the box's first edge, as `polygon_to_obb` ordered its corners."""
    edge = corners[1] - corners[0]
    return float(np.arctan2(edge[1], edge[0]))


def box_area(corners: NDArray[np.float64]) -> float:
    """Return the area of a ``(4, 2)`` box given as four consecutive corners."""
    return float(np.hypot(*(corners[1] - corners[0])) * np.hypot(*(corners[2] - corners[1])))


def rebuild_error(shape: str, corners: NDArray[np.float64], drawn: NDArray[np.float64]) -> float:
    """Return how far `shape`'s reference outline, placed by `corners` alone, lands from `drawn`.

    Scale comes from the box's area against the reference box's, position from the two box centers, and rotation from
    the box's heading measured against the reference box's own heading (zero for the upright-frame box, kept in the
    formula so the rebuild stays correct for any box convention that orders its corners the same way).

    A box is unchanged by a half turn, and a square one by a quarter turn, so the pose it pins down is inherently only
    known up to a quarter turn. All four are tried and the closest reported, which is exactly the ambiguity a consumer
    reading the box would face — and it is not a loophole: for any shape that is not itself quarter-turn symmetric,
    only the true pose lands anywhere near the object.

    Args:
        shape: Shape name, as `shape_outline` accepts it.
        corners: The object's OBB as a ``(4, 2)`` array of corners.
        drawn: The object's own outline as an ``(num_points, 2)`` array, in the same pixel frame.

    Returns:
        The smallest achievable maximum vertex displacement, in the same units as `drawn`.

    """
    reference = shape_outline(shape, center=(0.0, 0.0), size=1.0)
    reference_box = polygon_to_obb(reference)
    scale = np.sqrt(box_area(corners) / box_area(reference_box))
    heading = box_heading(corners) - box_heading(reference_box)
    best = np.inf
    for quarter in range(4):
        turn = heading + quarter * np.pi / 2.0
        anchor = rotate_polygon(reference_box.mean(axis=0)[None, :] * scale, turn)
        rebuilt = rotate_polygon(reference * scale, turn) + (corners.mean(axis=0) - anchor)
        best = min(best, float(np.linalg.norm(rebuilt - drawn, axis=1).max()))
    return best
