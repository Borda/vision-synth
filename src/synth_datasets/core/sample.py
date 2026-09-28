"""Format-agnostic in-memory representation of one synthetic image.

A :class:`Sample` carries the rendered image plus every annotation field a writer
could need (polygon, axis-aligned box, oriented box, and — for the keypoints task —
landmarks). Writers pick the subset that their target task requires, so the generator
never needs to know the output format.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from synth_datasets.families.geometry import polygon_to_obb

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from synth_datasets.core.keypoints import KeypointSchema

#: The COCO landmark visibility flags a triple may carry: ``0`` "not labeled", ``1`` "labeled but not
#: visible", ``2`` "labeled and visible". Spelled as the accepted set rather than a ``0 <= v <= 2``
#: range test so a value that is numerically in range but not a flag (``1.5``) is rejected too.
_KEYPOINT_VISIBILITIES: frozenset[int] = frozenset({0, 1, 2})


@dataclass(frozen=True)
class Annotation:
    """One object instance with all task representations precomputed.

    Coordinates are absolute pixel values in the image frame. Polygons and OBB corners are flat
    ``[x1, y1, x2, y2, ...]`` lists. The oriented box is *derived* from the polygon on first access
    (see :attr:`obb_corners`) rather than stored, so the tasks that never read it never pay for it.

    A landmark table is validated against its own schema on construction (see
    :meth:`__post_init__`), so every consumer — the writers,
    :class:`~synth_datasets.export.datasets.SyntheticIterableDataset`, and any third-party code
    reading a :class:`Sample` — can rely on the width without re-checking it. Validating here rather
    than in a writer is what makes that guarantee hold for consumers that never touch a writer.

    Args:
        class_id: Zero-based class index (see :func:`~synth_datasets.core.config.class_names`).
        class_name: Human-readable class label.
        polygon: Filled-shape outline as a flat pixel-coordinate list, in **pixel-centre** space --
            the space :func:`~fused_transforms.targets.transform_keypoints` moves a point field
            through. :attr:`obb_corners`, derived from it, is in the same space.
        bbox_xyxy: Axis-aligned box ``(x_min, y_min, x_max, y_max)`` in pixels, in **pixel-edge**
            space -- the space :func:`~fused_transforms.targets.transform_bbox_xyxy` assumes, so a
            full ``(H, W)`` canvas spans ``[0, W] x [0, H]``. The two conventions differ by half a
            pixel each way and are not interchangeable; see
            :data:`~synth_datasets.families.geometry.PIXEL_CENTRE_OFFSET`. Both dataset writers
            convert the point fields back to edge space at the file boundary, so an exported COCO or
            YOLO file carries one convention throughout.
        angle: Rotation in radians the shape was placed with (counter-clockwise, ``0.0`` for an
            unrotated or rotation-invariant shape). Carried so :attr:`obb_corners` can derive the
            oriented box in the shape's own upright frame instead of re-guessing the pose from the
            polygon.
        keypoints: Landmarks as ``(x, y, visibility)`` triples in ``keypoint_schema`` order, or
            ``None`` for any task other than
            :attr:`~synth_datasets.core.config.Task.KEYPOINTS`. Visibility follows COCO: ``2``
            for a point inside the canvas, ``0`` for one clipped away by the frame — a ``0`` point
            carries ``(0.0, 0.0)`` rather than its off-canvas coordinates. Visible coordinates are
            in pixel-centre space like ``polygon``; the zeroed placeholder is a flag value and
            carries no convention.
        keypoint_schema: The keypoint-bearing family ``keypoints`` was drawn from, which names and
            sizes the table. Required whenever ``keypoints`` is given, ``None`` otherwise. Carrying
            it here is what lets any consumer — the writers,
            :class:`~synth_datasets.export.datasets.SyntheticIterableDataset`, third-party code —
            interpret a table without being told separately which family produced it.

    Raises:
        ValueError: If ``keypoints`` is given without a ``keypoint_schema``, holds a number of
            triples other than the schema's ``kpt_shape``, or carries a visibility outside COCO's
            ``{0, 1, 2}``.

    Examples:
        ```pycon
        >>> from synth_datasets.families.animals import ANIMAL_KEYPOINT_SCHEMA
        >>> from synth_datasets.core.sample import Annotation
        >>> ann = Annotation(0, "square", [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ...                  (0.0, 0.0, 2.0, 2.0))
        >>> ann.class_name
        'square'
        >>> ann.keypoints is None
        True
        >>> table = tuple((1.0, 2.0, 2) for _ in ANIMAL_KEYPOINT_SCHEMA.names)
        >>> duck = Annotation(4, "duck", [], (0.0, 0.0, 2.0, 2.0), keypoints=table,
        ...                   keypoint_schema=ANIMAL_KEYPOINT_SCHEMA)
        >>> duck.keypoints[0]
        (1.0, 2.0, 2)

        ```

    """

    class_id: int
    class_name: str
    polygon: list[float]
    bbox_xyxy: tuple[float, float, float, float]
    angle: float = 0.0
    keypoints: tuple[tuple[float, float, int], ...] | None = None
    keypoint_schema: KeypointSchema | None = None

    @cached_property
    def obb_corners(self) -> list[float]:
        """Return the upright-frame oriented box as four corners, flat ``[x1, y1, ..., x4, y4]``.

        The box is the shape's axis-aligned box in its own pre-rotation frame, rotated by
        :attr:`angle` — its sides run along and across the shape's upright (symmetry) axis, not
        the minimum-area rectangle's hull-edge direction (see
        :func:`~synth_datasets.families.geometry.polygon_to_obb`).

        Derived from :attr:`polygon` on first access rather than stored. It used to be computed for
        every object at generation time, which meant every detection, segmentation and keypoint run
        paid for a convex hull plus a rotating-calipers scan per object and then never read the
        result — measured at **75% of generation time** on a mixed-family run. Deriving it here costs
        exactly the same for an OBB run, and nothing for the three tasks that do not want it.

        Returns:
            The eight corner coordinates, or an empty list when :attr:`polygon` holds fewer than
            three points and has no oriented box to speak of.

        Examples:
            ```pycon
            >>> from synth_datasets.core.sample import Annotation
            >>> square = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]
            >>> ann = Annotation(0, "square", square, (0.0, 0.0, 2.0, 2.0))
            >>> len(ann.obb_corners)
            8

            ```

        """
        points = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 3:
            return []
        return [float(value) for value in polygon_to_obb(points, self.angle).reshape(-1)]

    def __post_init__(self) -> None:
        """Reject a landmark table that does not match the schema it claims to follow.

        A short, long, or mis-flagged table is silently lossy downstream rather than loud: a COCO
        ``keypoints`` array of the wrong length still parses, and a YOLO pose row of the wrong width
        is read positionally, so both mislabel every landmark after the first missing one instead of
        failing. Catching it at construction turns that into an error at the point the bad table was
        built.

        The schema is carried rather than inferred. It used to be looked up by *table length* —
        which worked only because the registered families happened to have distinct landmark counts,
        and left an annotation unable to say which family it belonged to, so a writer had to be told
        separately and the two could disagree.

        Raises:
            ValueError: If ``keypoints`` is given without a ``keypoint_schema``, if it holds a
                number of triples other than the schema's ``kpt_shape``, or if a triple's visibility
                is not a COCO flag.

        """
        if self.keypoints is None:
            return
        if self.keypoint_schema is None:
            raise ValueError(
                f"keypoints were given without a keypoint_schema; pass the family's schema so the "
                f"{len(self.keypoints)} triples can be named and validated"
            )
        names = self.keypoint_schema.names
        if len(self.keypoints) != len(names):
            raise ValueError(
                f"keypoints must hold exactly {len(names)} (x, y, visibility) triples to match the "
                f"schema, got {len(self.keypoints)}"
            )
        for name, (_x, _y, visibility) in zip(names, self.keypoints, strict=True):
            if type(visibility) is not int or visibility not in _KEYPOINT_VISIBILITIES:
                raise ValueError(
                    f"keypoint {name!r} has visibility {visibility!r}, but COCO allows only "
                    f"{sorted(_KEYPOINT_VISIBILITIES)}: 0 not labeled, 1 labeled but hidden, 2 labeled and visible"
                )


@dataclass(frozen=True, eq=False)
class SceneRecord:
    """What the renderer knows about a scene that its annotations do not say.

    The side-car for everything that describes a whole image rather than one object in it. Hanging a
    field per feature off :class:`Sample` would grow its public surface every time the generator
    learns something new, and an untyped ``dict`` would be unlike every other field in the module;
    one typed record means later additions land here and touch no consumer.

    ``eq=False`` is load-bearing rather than stylistic. A generated ``__eq__`` compares field tuples,
    so two populated records would raise ``ValueError: The truth value of an array with more than one
    element is ambiguous`` instead of returning a bool, and ``hash()`` would raise ``TypeError``.
    Falling back to identity comparison is well defined for every record and is what a raster
    side-car should offer anyway: comparing two masks is the caller's business, not this type's.

    Args:
        occluder_mask: ``(height, width)`` boolean raster, ``True`` where an occluder covers the
            pixel, or ``None`` when no occluder was asked for. ``None`` rather than an all-``False``
            array on purpose, so "no occluders" is distinguishable from "occluders that happened to
            miss". The buffer is marked read-only before it is stored: ``frozen`` stops the field
            being rebound and does nothing to stop a caller mutating what it points at, and a mutated
            mask would silently disagree with the visibility flags already computed from it.
        background_source: Identifier of the image a photographic background was cropped from, or
            ``None`` for a procedural one.

    Examples:
        ```pycon
        >>> from synth_datasets.core.sample import SceneRecord
        >>> SceneRecord().occluder_mask is None
        True

        ```

    """

    occluder_mask: NDArray[np.bool_] | None = None
    background_source: str | None = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a record and re-clear the mask's writability, which pickling does not carry.

        Args:
            state: The instance dictionary being restored.

        ``ndarray.__reduce__`` does not preserve the ``WRITEABLE`` flag, so a mask that was read-only
        when it was written comes back writable — and this is the ordinary path, not an exotic one:
        :class:`~synth_datasets.export.datasets.SyntheticIterableDataset` is a torch
        ``IterableDataset``, so every sample crosses a pickle boundary under ``num_workers > 0``.
        Without this the freeze would be a guarantee that held only in the process that made it.

        Restoring through ``__dict__`` rather than ``__init__``: unpickling a dataclass bypasses the
        constructor, so a ``__post_init__`` freeze would never run.

        """
        self.__dict__.update(state)
        if self.occluder_mask is not None:
            self.occluder_mask.flags.writeable = False


#: The record every sample carries until something fills one in. Shared rather than built per sample,
#: which is safe precisely because it is empty and frozen: both its fields are immutable scalars, so
#: no sample can reach another through it. Never replace this with a ``default_factory`` "for
#: safety" — that would allocate per sample and buy nothing.
_EMPTY_SCENE = SceneRecord()


@dataclass(frozen=True)
class Sample:
    """A rendered image and its annotations.

    Args:
        image: RGB image, shape ``(height, width, 3)``, dtype ``uint8``. Writable exactly when the
            run configured a ``degrade`` chain: without one the array is the renderer's own read-only
            view onto Pillow's buffer, which is what it has always been and costs no copy, while a
            chain produces a buffer the sample owns. Copy it before writing if you need to write.
        annotations: Object annotations, one per drawn shape.
        width: Image width in pixels.
        height: Image height in pixels.
        scene: What the renderer knows about the scene as a whole — see :class:`SceneRecord`.
            Defaults to a shared empty record, so every existing construction keeps working and a
            consumer that never asks about the scene never notices it.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.core.sample import Sample
        >>> img = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> Sample(img, [], width=4, height=4).width
        4
        >>> Sample(img, [], width=4, height=4).scene.background_source is None
        True

        ```

    """

    image: NDArray[Any]
    annotations: list[Annotation]
    width: int
    height: int
    scene: SceneRecord = _EMPTY_SCENE
