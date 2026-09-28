"""Build the unit-space geometry snapshot every shape family is pinned against.

The snapshot is the refactor safety net: it records what each of the 49 shapes looks like *before* the registry
refactor, the naming rename, and the JSON-to-SVG asset migration, so any of those silently moving a vertex or a landmark
shows up as a failing comparison rather than as a subtly different dataset months later.

Everything is sampled in unit space (``center=(0, 0)``, ``size=1.0``, ``angle=0.0``, ``skew=0.0``) because that is where
the family assets actually live; placement is a separate, already-tested transform and pinning it here would only add
rotation noise to the diff.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synth_datasets.families import shape_outline
from synth_datasets.families.animals import AnimalShape, animal_keypoints
from synth_datasets.families.letters import LetterShape, letter_keypoints
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SymbolShape, symbol_keypoints

#: Keypoint accessor per family. Spelled out here rather than taken from
#: :data:`~synth_datasets.families.SHAPE_FAMILIES` on purpose: the snapshot is the check
#: on the registry, so reading the registry to build it would make the two agree by construction
#: and prove nothing.
_KEYPOINT_FNS = {
    AnimalShape: animal_keypoints,
    SymbolShape: symbol_keypoints,
    LetterShape: letter_keypoints,
}

_ORIGIN = (0.0, 0.0)


def build_golden() -> dict[str, NDArray[np.float64]]:
    """Return the ``key -> array`` snapshot of every shape's unit-space outline and landmarks.

    Returns:
        A mapping with a ``poly:<value>`` entry per shape and a ``kpts:<value>`` entry per shape
        belonging to a keypoint-bearing family. NaN landmark rows (an absent optional point) are
        kept as NaN; the comparison side uses ``equal_nan``.

    """
    golden: dict[str, NDArray[np.float64]] = {}
    for family in (PrimitiveShape, AnimalShape, SymbolShape, LetterShape):
        keypoint_fn = _KEYPOINT_FNS.get(family)
        for shape in family:
            golden[f"poly:{shape.value}"] = shape_outline(shape.value, _ORIGIN, 1.0)
            if keypoint_fn is not None:
                golden[f"kpts:{shape.value}"] = keypoint_fn(shape, _ORIGIN, 1.0, 0.0, 0.0)
    return golden
