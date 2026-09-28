"""Keypoint-schema type and outline/landmark normalization shared by every keypoint family.

:class:`KeypointSchema` is the one artifact a keypoint-bearing shape family ( :mod:`~synth_datasets.families.animals`,
:mod:`~synth_datasets.families.symbols`) hands to the rest of the pipeline: a fixed landmark-name list, the skeleton
edges a viewer draws between them, and the permutation a horizontal flip applies.
:func:`~synth_datasets.core.config.class_names` and the writers never need to know how a family's outline or landmark
table was built, only its schema.

The normalization helpers below place an outline at a zero area centroid (center of mass, not the vertex mean — see
:func:`_polygon_centroid`) and a unit larger-extent, and map a landmark table through *the outline's own* transform so a
landmark can never drift off the silhouette it annotates. They live here, underneath every shape family, so
:mod:`~synth_datasets.families.animals` (SVG-traced) and :mod:`~synth_datasets.families.symbols` (analytic) can both use
them without either importing the other.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class KeypointSchema:
    """The fixed landmark vocabulary one keypoint-bearing shape family draws its tables from.

    Args:
        names: Landmark names, in the order every table, annotation, and label row uses.
        skeleton: Visualization-only edges as index pairs into ``names`` (COCO viewers connect the
            dots with it; nothing in generation, writing, or validation depends on it).
        flip_idx: The landmark each index becomes under a horizontal flip — Ultralytics'
            ``flip_idx`` convention. Must be a self-inverse permutation of ``range(len(names))``:
            flipping twice returns every point to itself.
        shape_values: Every :class:`~synth_datasets.core.config.Shape` *value* this schema's
            family draws from (e.g. every :class:`~synth_datasets.families.animals.AnimalShape`
            value). The writers use this to tell which ``class_names`` categories belong to this
            keypoint-bearing family — see
            :func:`~synth_datasets.export.writers._keypoint_eligible_names` — independently of
            which landmark names or skeleton the family itself uses.
        skeleton_by_value: Per-``shape_values``-entry skeleton override, for a family whose members
            do not all share one topology (e.g.
            :mod:`~synth_datasets.families.letters`, where each letter's stroke edges *are* its
            drawn geometry, not just a viewer aid). ``None`` (the default) means every member shares
            ``skeleton`` — the behavior every family before ``letters`` relies on, unaffected by
            this field existing. When given, it need not cover every ``shape_values`` entry; a
            missing key falls back to ``skeleton`` the same way ``None`` would.

    Raises:
        ValueError: If ``flip_idx`` is not exactly a self-inverse permutation of
            ``range(len(names))``, a ``skeleton`` edge indexes outside ``names``, a
            ``skeleton_by_value`` key is not in ``shape_values`` or one of its edges indexes outside
            ``names``, or ``shape_values`` is empty.

    Examples:
        ```pycon
        >>> from synth_datasets.core.keypoints import KeypointSchema
        >>> schema = KeypointSchema(names=("a", "b"), skeleton=((0, 1),), flip_idx=(1, 0), shape_values=("x",))
        >>> schema.kpt_shape
        2
        >>> schema.skeleton_for("x")
        ((0, 1),)

        ```

    """

    names: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...]
    flip_idx: tuple[int, ...]
    shape_values: tuple[str, ...]
    skeleton_by_value: Mapping[str, tuple[tuple[int, int], ...]] | None = None

    def __post_init__(self) -> None:
        """Validate ``flip_idx``, ``skeleton``, and ``skeleton_by_value`` (when given)."""
        if not self.shape_values:
            raise ValueError("shape_values must name at least one shape value, got an empty sequence")
        n = len(self.names)
        if len(self.flip_idx) != n:
            raise ValueError(f"flip_idx must hold exactly {n} entries (one per name), got {len(self.flip_idx)}")
        if sorted(self.flip_idx) != list(range(n)):
            raise ValueError(f"flip_idx must be a permutation of range({n}), got {self.flip_idx!r}")
        if any(self.flip_idx[i] != i and self.flip_idx[self.flip_idx[i]] != i for i in range(n)):
            raise ValueError(f"flip_idx must be self-inverse (flipping twice is the identity), got {self.flip_idx!r}")
        self._validate_edges("skeleton", self.skeleton, n)
        if self.skeleton_by_value is not None:
            unknown = [value for value in self.skeleton_by_value if value not in self.shape_values]
            if unknown:
                raise ValueError(f"skeleton_by_value key(s) {unknown!r} are not in shape_values {self.shape_values!r}")
            for value, edges in self.skeleton_by_value.items():
                self._validate_edges(f"skeleton_by_value[{value!r}]", edges, n)

    def _validate_edges(self, label: str, edges: tuple[tuple[int, int], ...], n: int) -> None:
        """Raise if any ``edges`` pair indexes outside ``range(n)``."""
        bad_edges = [edge for edge in edges if not all(0 <= idx < n for idx in edge)]
        if bad_edges:
            raise ValueError(f"{label} edges must index into names[0:{n}], got out-of-range edge(s) {bad_edges!r}")

    def skeleton_for(self, shape_value: str) -> tuple[tuple[int, int], ...]:
        """Return the skeleton edges to draw for one ``shape_values`` member.

        Args:
            shape_value: A value from :attr:`shape_values`.

        Returns:
            ``skeleton_by_value[shape_value]`` when :attr:`skeleton_by_value` is given and holds an
            entry for it; :attr:`skeleton` otherwise — the single shared topology every
            pre-``letters`` family uses.

        """
        if self.skeleton_by_value is not None and shape_value in self.skeleton_by_value:
            return self.skeleton_by_value[shape_value]
        return self.skeleton

    @property
    def kpt_shape(self) -> int:
        """Return the landmark count (Ultralytics' ``kpt_shape`` first dimension)."""
        return len(self.names)


def _polygon_centroid(points: NDArray[np.float64], extent: float) -> NDArray[np.float64]:
    """Return a simple polygon's area centroid (center of mass), via the shoelace formula.

    Unlike the vertex mean, this is invariant to how densely vertices are spread along an edge — an
    outline with many points bunched along one edge and few along another (an arrow's barbs, an
    anchor's flukes) still centers on its true visual middle rather than drifting toward whichever
    edge happens to carry more vertices.

    Args:
        points: ``(num_points, 2)`` simple (non-self-intersecting) outline array, in either winding
            direction.
        extent: The outline's larger bounding-box extent, from :func:`_frame` — used only to scale
            the degenerate-area threshold below, so it need not be recomputed here.

    Returns:
        The ``(x, y)`` area centroid.

    Raises:
        ValueError: If the polygon's signed area is ~0 relative to its extent (collinear or
            near-collinear vertices — an outline with no real interior).

    """
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = float(cross.sum()) / 2.0
    if abs(area) < 1e-9 * extent**2:
        raise ValueError("outline has ~zero area (collinear vertices); cannot locate a center of mass")
    cx = float(((x + x_next) * cross).sum()) / (6.0 * area)
    cy = float(((y + y_next) * cross).sum()) / (6.0 * area)
    return np.array([cx, cy], dtype=np.float64)


def _frame(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Return the ``(offset, extent)`` that normalize an outline into unit space.

    Args:
        points: ``(num_points, 2)`` raw outline array.

    Returns:
        The area centroid (center of mass) to subtract and the larger extent to divide by.

    Raises:
        ValueError: If the outline collapses to a point, or has ~zero area (see
            :func:`_polygon_centroid`).

    """
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if extent <= 0.0:
        raise ValueError("outline has zero extent; every vertex is identical")
    offset = _polygon_centroid(points, extent)
    return offset, extent


def _normalized(
    vertices: Sequence[tuple[float, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Center a raw outline on its area centroid and scale its larger extent to ``1``.

    Args:
        vertices: ``(x, y)`` outline points in any convenient scale, ordered along the outline
            (winding direction is irrelevant).

    Returns:
        The read-only ``(num_points, 2)`` float array with zero area centroid (center of mass) and
        a maximum extent of exactly ``1``, followed by the ``(offset, extent)`` frame it was
        normalized through. The frame is handed back so a caller mapping a second table into the
        same frame — see :func:`_normalized_pair` — reuses this measurement instead of calling
        :func:`_frame` on the same outline a second time. The array is frozen because it is shared
        by every caller; consumers scale it into a fresh array rather than mutating it.

    Raises:
        ValueError: If the outline has fewer than three points, collapses to a point, or has ~zero
            area (see :func:`_polygon_centroid`).

    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"an outline needs at least 3 (x, y) points, got array of shape {points.shape}")
    offset, extent = _frame(points)
    scaled: NDArray[np.float64] = (points - offset) / extent
    scaled.setflags(write=False)
    return scaled, offset, extent


def _normalized_pair(
    outline: Sequence[tuple[float, float]],
    landmarks: Sequence[tuple[float, float]],
    names: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize an outline and map its landmarks through the *outline's* transform.

    Landmarks are authored in the same coordinates as the outline they annotate, so they must be
    centred and scaled by the outline's own mean and extent — normalizing them independently would
    re-centre them on their own mean and detach them from the silhouette. Pairing the two tables in
    one call is what makes that impossible to get wrong.

    Args:
        outline: ``(x, y)`` outline points; see :func:`_normalized`.
        landmarks: ``len(names)`` ``(x, y)`` landmarks, in ``names`` order, in the same coordinates
            as ``outline``. An absent optional landmark is ``(nan, nan)``.
        names: The landmark names ``landmarks`` is ordered by — used only to size-check the table
            and to name an offending landmark in an error, never stored.

    Returns:
        The read-only normalized ``(num_points, 2)`` outline and the read-only ``(len(names), 2)``
        landmark table. Unlike the outline, the landmark table is **not** self-normalized: its mean
        is not the origin and its extent is not ``1``. NaN rows pass through unchanged: this call
        only ever measures the *outline* (via :func:`_frame`), never the landmark table, so a NaN
        landmark can never poison the offset/extent it and every other landmark are mapped through.

    Raises:
        ValueError: If the outline is degenerate (see :func:`_normalized`), the landmark table does
            not hold exactly ``len(names)`` ``(x, y)`` points, or a row has exactly one NaN
            coordinate (a parser bug — a real absence is NaN in both).

    """
    polygon, offset, extent = _normalized(outline)
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (len(names), 2):
        raise ValueError(
            f"a keypoint table needs exactly {len(names)} (x, y) landmarks, got array of shape {points.shape}"
        )
    nan_mask = np.isnan(points)
    half_nan = nan_mask.any(axis=1) & ~nan_mask.all(axis=1)
    if half_nan.any():
        bad = [names[int(i)] for i in np.nonzero(half_nan)[0]]
        raise ValueError(f"keypoint(s) {bad} have exactly one NaN coordinate; an absent landmark must be NaN in both")
    mapped: NDArray[np.float64] = (points - offset) / extent
    mapped.setflags(write=False)
    return polygon, mapped
