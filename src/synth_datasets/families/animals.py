"""Animal-silhouette outline and landmark tables, loaded from the packaged ``assets/animals`` files.

Twelve side-profile silhouettes — duck, elephant, giraffe, fish, rabbit, camel, eagle, penguin,
whale, kangaroo, flamingo, crocodile — traced from public-domain reference art (see
:data:`ANIMAL_SOURCES`) rather than guessed by hand, so each one is recognizable at preview size.
Every animal ships as one SVG document,
``synth_datasets/assets/animals/<animal>.svg``, holding its outline path, its sixteen landmarks (as a
``<g id="keypoints">`` of ``<circle>`` elements) and its provenance (as ``zoo:``-namespaced root
attributes); the artwork can therefore be opened, inspected and corrected in any SVG editor without
touching Python. This module is only the loader that turns those documents into NumPy tables.

Each outline is a **simple** (non-self-intersecting) polygon in the unit space
:func:`~synth_datasets.families.geometry._base_polygon` uses for the geometric shapes: center of mass
at the origin and the larger of the two extents scaled to ``1``, so multiplying by a pixel ``size``
yields a shape bounded by ``size`` pixels. Coordinates are in screen orientation — ``+x`` right,
``+y`` **down** — matching Pillow's raster axes, so every animal renders upright and faces left.
The stored values are already normalized; they are passed through :func:`_normalized` again at load
time, which makes the invariant impossible to break by editing an SVG file.

:data:`ANIMAL_KEYPOINTS` carries the sixteen landmarks in :data:`ANIMAL_KEYPOINT_NAMES` order — an
anatomical ``mouth``/``eye``/``ear``/``head``/``neck``/``body_top``/``body_bottom``/``tail`` chain
plus two-segment front limbs (``front_elbow_*`` then ``front_limb_*``: paws, wings, or
fins/flippers) and the optional two-segment hind legs (``hind_knee_*`` then ``hind_limb_*``) — that
:attr:`~synth_datasets.core.config.Task.KEYPOINTS` annotates.
Landmarks are mapped through *their outline's* transform, so the two tables live in one frame: a
landmark sits **inside or on** its silhouette and, unlike an outline table, is neither centred on the
origin nor scaled to unit extent by itself.

**Absent keypoints.** ``ear`` and the four hind-leg names (``hind_knee_*``, ``hind_limb_*``) are the
only ones an animal may omit — a whale's or a fish's silhouette shows pectoral fins/flippers as its
front limbs but no hind legs, and neither taxon has an external ear to annotate at all: an omitted
``<circle>`` in the source SVG becomes a ``NaN`` row — ``(nan, nan)`` — in the returned table rather
than a faked point. Every other name is mandatory; a missing one is a load-time
:class:`ValueError`, not a silent gap. The NaN encoding was
chosen because the rest of the pipeline already handles it for free: a NaN coordinate compares
``False`` against every bound check downstream (canvas-visibility, placement clipping), so it falls
through to "not labeled" exactly like a point clipped off-canvas — see
:func:`~synth_datasets.core.generator._visible_keypoints` and
:func:`~synth_datasets.export.writers._keypoint_triples`. This module's own :func:`_frame` only
ever measures the **outline**, never the landmark table, so a NaN landmark can never poison the
outline's centring/scaling math — that invariant is what makes the encoding safe.

Unlike the geometric shapes these outlines are asymmetric and each belongs to a distinct silhouette
archetype, so the classes stay separable at a glance and every outline point keeps an unambiguous
identity under rotation.

Pure NumPy plus the stdlib XML parser — no Pillow, no torch, and no import from
:mod:`synth_datasets.core.config`: the configuration layer imports *this* module for the animal
half of its shape vocabulary, never the other way round. Tables are keyed by the plain
:class:`AnimalShape` *values*.

Examples:
    ```pycon
    >>> from synth_datasets.families.animals import ANIMAL_KEYPOINTS, ANIMAL_POLYGONS
    >>> len(ANIMAL_POLYGONS)
    12
    >>> sorted(ANIMAL_POLYGONS)[:6]
    ['camel', 'crocodile', 'duck', 'eagle', 'elephant', 'fish']
    >>> duck = ANIMAL_POLYGONS["duck"]
    >>> duck.shape[1]
    2
    >>> round(float((duck.max(axis=0) - duck.min(axis=0)).max()), 9)
    1.0
    >>> ANIMAL_KEYPOINTS["duck"].shape
    (16, 2)
    >>> sorted(ANIMAL_KEYPOINTS) == sorted(ANIMAL_POLYGONS)
    True

    ```

"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from synth_datasets.core.keypoints import KeypointSchema, _normalized_pair
from synth_datasets.families.geometry import place_points
from synth_datasets.families.shape_enum import ShapeEnum
from synth_datasets.families.svgio import read_outline_document

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from numpy.typing import NDArray


class AnimalShape(ShapeEnum):
    """Animal silhouette vocabulary (definition order is the animal class order).

    Twelve fixed side-profile silhouettes traced from public-domain reference art. Each is
    asymmetric and belongs to a distinct silhouette archetype, so the classes stay separable at a
    glance and every outline point keeps an unambiguous identity under rotation — the property a
    landmark needs and a square or circle cannot offer. Every member has a ``<value>.svg`` document
    in the packaged ``assets/animals`` directory and therefore a landmark table, which is what makes
    :attr:`~synth_datasets.core.config.Task.KEYPOINTS` well-defined for exactly this enum.

    Attributes:
        DUCK: Compact duck silhouette with an S-curved neck and a beak.
        ELEPHANT: Bulky elephant silhouette with a trunk, a large ear, and thick legs.
        GIRAFFE: Tall, thin giraffe silhouette with a very long neck and thin legs.
        FISH: Streamlined fish silhouette with a forked tail fin.
        RABBIT: Compact rabbit silhouette with long upright ears.
        CAMEL: Humped camel silhouette on four long legs.
        EAGLE: Perched eagle silhouette with a hooked beak and a long tail.
        PENGUIN: Upright penguin silhouette with flippers and webbed feet.
        WHALE: Streamlined whale silhouette with a pectoral flipper and a tail fluke.
        KANGAROO: Hopping kangaroo silhouette with a heavy tail and one large hind foot.
        FLAMINGO: Long-legged flamingo silhouette with an S-curved neck.
        CROCODILE: Low, elongated crocodile silhouette with a long snout and sprawled legs.

    Examples:
        ```pycon
        >>> from synth_datasets.families.animals import AnimalShape
        >>> len(AnimalShape)
        12
        >>> AnimalShape("duck")
        <AnimalShape.DUCK: 'duck'>

        ```

    """

    DUCK = "duck"
    ELEPHANT = "elephant"
    GIRAFFE = "giraffe"
    FISH = "fish"
    RABBIT = "rabbit"
    CAMEL = "camel"
    EAGLE = "eagle"
    PENGUIN = "penguin"
    WHALE = "whale"
    KANGAROO = "kangaroo"
    FLAMINGO = "flamingo"
    CROCODILE = "crocodile"


#: Animal names in :class:`AnimalShape` declaration order — the enum's values, kept as a plain tuple
#: because the loader and its tables are keyed by name rather than by member.
ANIMAL_NAMES: tuple[str, ...] = tuple(shape.value for shape in AnimalShape)

#: Landmark names for :attr:`~synth_datasets.core.config.Task.KEYPOINTS`, in the order every
#: keypoint table, annotation, and label row uses — and the order landmarks are read out of an
#: animal document. One shared anatomical schema across all animals: Ultralytics' YOLO pose format carries
#: a single dataset-wide ``kpt_shape``, so a per-class name list is not representable.
#: The ``front_limb_*`` pair covers whatever the animal actually has at that slot — paws, wings, or
#: flippers/fins; ``left`` is the limb nearer the viewer (fully visible), ``right`` the far one — a
#: documented convention, since a side-profile silhouette cannot truly tell left from right. Every
#: limb is articulated in two points, proximal before distal: ``front_elbow_*`` (elbow, wing wrist,
#: or flipper bend) then ``front_limb_*`` (the paw/wing tip/fin tip), and ``hind_knee_*`` (the
#: knee/hock bend) then ``hind_limb_*`` (the foot) — a limb's bend is the most visible pose cue on a
#: silhouette. ``ear`` and all four hind points are optional — see :data:`_OPTIONAL_KEYPOINTS`.
ANIMAL_KEYPOINT_NAMES: tuple[str, ...] = (
    "mouth",
    "eye",
    "ear",
    "head",
    "neck",
    "body_top",
    "body_bottom",
    "tail",
    "front_elbow_left",
    "front_elbow_right",
    "front_limb_left",
    "front_limb_right",
    "hind_knee_left",
    "hind_knee_right",
    "hind_limb_left",
    "hind_limb_right",
)

#: The only landmark names a document may omit; every other name in :data:`ANIMAL_KEYPOINT_NAMES` is
#: mandatory and a missing one is a load-time :class:`ValueError`. ``ear`` earns its place next to the
#: hind legs on the same anatomical ground: a fish and a whale have no external ear to point at, so the
#: alternative is inventing a spot on the head and labelling it visible.
_OPTIONAL_KEYPOINTS: frozenset[str] = frozenset({
    "ear",
    "hind_knee_left",
    "hind_knee_right",
    "hind_limb_left",
    "hind_limb_right",
})

#: The landmarks every animal must carry — the family's vocabulary minus the optional ones. Derived
#: rather than listed so the two can never disagree.
_REQUIRED_KEYPOINTS: tuple[str, ...] = tuple(name for name in ANIMAL_KEYPOINT_NAMES if name not in _OPTIONAL_KEYPOINTS)

#: Skeleton edges as index pairs into :data:`ANIMAL_KEYPOINT_NAMES`: ``mouth-head``, ``eye-head``,
#: ``ear-head``, the ``head-neck-body_top-body_bottom-tail`` chain, a two-segment
#: ``body_top-front_elbow-front_limb`` chain per front limb, and a two-segment
#: ``body_bottom-hind_knee-hind_limb`` chain per hind leg. 15 edges over 16 nodes is a spanning tree
#: whose optional points sit at the ends of their chains — ``ear`` is a leaf on ``head``, the hind
#: points hang off ``body_bottom`` — so an absent ear drops exactly its one edge, an absent hind leg
#: exactly its own two, and neither orphans anything. Purely a visualization aid (COCO viewers connect
#: the dots with it); nothing in generation, writing, or validation depends on it.
ANIMAL_KEYPOINT_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (5, 8),
    (5, 9),
    (8, 10),
    (9, 11),
    (6, 12),
    (6, 13),
    (12, 14),
    (13, 15),
)

_ZOO: Traversable = files("synth_datasets") / "assets" / "animals"


def _read_svg(name: str) -> tuple[list[tuple[float, float]], dict[str, tuple[float, float]], dict[str, str]]:
    """Read one packaged animal document through the shared reader.

    Args:
        name: Animal name, i.e. the ``<name>.svg`` stem in the packaged ``assets/animals`` directory.

    Returns:
        The outline vertices, the present keypoints (by name), and the provenance attributes.

    """
    return read_outline_document(_ZOO, name, ANIMAL_KEYPOINT_NAMES, _REQUIRED_KEYPOINTS)


def _load() -> tuple[
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.float64]],
    dict[str, dict[str, str]],
]:
    """Load every packaged animal into the outline, landmark and provenance tables."""
    polygons: dict[str, NDArray[np.float64]] = {}
    keypoints: dict[str, NDArray[np.float64]] = {}
    sources: dict[str, dict[str, str]] = {}
    for name in ANIMAL_NAMES:
        outline, present, source = _read_svg(name)
        table = [present.get(key, (np.nan, np.nan)) for key in ANIMAL_KEYPOINT_NAMES]
        polygons[name], keypoints[name] = _normalized_pair(outline, table, ANIMAL_KEYPOINT_NAMES)
        sources[name] = source
    return polygons, keypoints, sources


_POLYGONS, _KEYPOINTS, _SOURCES = _load()

#: Outline table per animal :class:`~synth_datasets.core.config.Shape` *value*.
#: Every entry is unit-normalized and read-only; scale a copy rather than mutating it. The table
#: itself is a read-only view as well, so ``ANIMAL_POLYGONS["duck"] = other`` raises instead of
#: silently repointing an outline every consumer in the process shares.
ANIMAL_POLYGONS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_POLYGONS)

#: Landmark table per animal :class:`~synth_datasets.core.config.Shape` *value*, in
#: ``config.KEYPOINT_NAMES`` order. Every entry is a read-only ``(16, 2)`` array in its outline's
#: unit frame, so landmarks lie inside or on the silhouette; scale a copy rather than mutating it. A
#: row is ``(nan, nan)`` for a landmark the animal does not have — the ``ear`` row and all four hind
#: rows for a fish or a whale. Like :data:`ANIMAL_POLYGONS`, the table itself is a read-only view,
#: not just the arrays in it.
ANIMAL_KEYPOINTS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_KEYPOINTS)

#: Provenance per animal: ``origin`` (source page), ``title`` (what the art depicts), ``license``,
#: ``attribution`` (credit, not required by CC0/PDM) and a ``note`` on how the art was processed.
#: Frozen at both levels — neither the animal-to-record mapping nor an individual record can be
#: rewritten, because a licence field edited in place would misdescribe artwork already in the wheel.
ANIMAL_SOURCES: Mapping[str, Mapping[str, str]] = MappingProxyType({
    animal: MappingProxyType(source) for animal, source in _SOURCES.items()
})

#: Identity permutation: this schema's ``left``/``right`` are viewer-relative, not the animal's own
#: left/right (a side-profile silhouette cannot truly tell them apart — see
#: :data:`ANIMAL_KEYPOINT_NAMES`), so mirroring a side profile never turns a near limb into a far
#: one and every landmark maps to itself under a horizontal flip.
ANIMAL_KEYPOINT_FLIP_IDX: tuple[int, ...] = tuple(range(len(ANIMAL_KEYPOINT_NAMES)))

#: The complete keypoint schema for every :class:`AnimalShape` — the one artifact
#: :func:`~synth_datasets.core.config.keypoint_schema_for` and the writers need to describe an
#: animal ``Task.KEYPOINTS`` run.
ANIMAL_KEYPOINT_SCHEMA = KeypointSchema(
    names=ANIMAL_KEYPOINT_NAMES,
    skeleton=ANIMAL_KEYPOINT_SKELETON,
    flip_idx=ANIMAL_KEYPOINT_FLIP_IDX,
    shape_values=ANIMAL_NAMES,
)


def animal_keypoints(
    shape: AnimalShape, center: tuple[float, float], size: float, angle: float = 0.0, skew: float = 0.0
) -> NDArray[np.float64]:
    """Place one animal's landmark table into image coordinates.

    The table is looked up in :data:`ANIMAL_KEYPOINTS`, scaled, skewed, rotated, and translated
    exactly as :func:`~synth_datasets.families.shape_outline` treats the matching outline,
    so passing the same ``center``, ``size``, ``angle``, and ``skew`` to both puts every landmark on
    the silhouette that was drawn. No randomness is involved: the result is a pure function of the
    placement the generator already sampled.

    Args:
        shape: An :class:`AnimalShape` member. The geometric shapes have no landmark table at all
            (a square's 4-fold symmetry gives a fixed landmark no stable identity), which is why
            this signature names the animal enum rather than the shape union.
        center: Target center ``(x, y)`` in pixels — the same value passed to ``shape_outline``.
        size: Bounding size in pixels — the same value passed to ``shape_outline``.
        angle: Rotation in radians about the shape center — likewise.
        skew: Signed fraction narrowing one pre-rotation half — likewise; see
            :attr:`~synth_datasets.core.config.SyntheticConfig.asymmetry_jitter`.

    Returns:
        ``(16, 2)`` float array of landmark coordinates in image pixels, ordered by
        :data:`ANIMAL_KEYPOINT_NAMES`. Points may fall outside the canvas; clipping is the caller's
        decision. A row is ``(nan, nan)`` for a landmark the animal does not have (a fish's or a
        whale's ear and hind legs); NaN propagates through unchanged since scaling and translation
        are row-independent arithmetic.

    Raises:
        ValueError: If ``shape`` has no keypoint table.

    Examples:
        ```pycon
        >>> from synth_datasets.families.animals import AnimalShape, animal_keypoints
        >>> points = animal_keypoints(AnimalShape.DUCK, center=(50.0, 50.0), size=20.0)
        >>> points.shape
        (16, 2)

        ```

    """
    table = ANIMAL_KEYPOINTS.get(shape.value)
    if table is None:
        known = ", ".join(ANIMAL_KEYPOINTS)
        raise ValueError(f"shape {shape.value!r} has no keypoint table; expected one of {known}")
    # The stored table is frozen, so multiplying returns a fresh writable array, never an alias.
    return place_points(table * size, center, angle, skew)
