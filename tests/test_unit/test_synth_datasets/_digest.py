"""Fold a sample's pixels and labels into a digest that is a property of the code, not of the CPU.

Split out of :mod:`._baseline` so it can be run against a checkout that predates the background and degradation modules:
that module imports them to describe its feature matrix, this one imports nothing beyond
:class:`~synth_datasets.core.sample.Sample`. Regenerating the pre-feature half of the snapshot means copying this file
into a worktree at the older commit, which only works while it stays importable there.

The digest deliberately treats its two inputs differently, because they do not share a failure mode.

Pixels enter exactly. A rasterizer maps a continuous vertex onto a discrete grid, which quantizes away vertex noise long
before it reaches a byte: skewing ``np.cos``/``np.sin`` by a relative ``1e-6`` — ten orders of magnitude above the drift
any real machine introduces, and enough to move a label by ``2.8e-05`` px — moves **no pixel at all** across the whole
matrix. Exact bytes are therefore both the most sensitive signal available and a portable one.

Labels enter quantized to :data:`_LABEL_GRID`. They are the half that does not survive a change of machine: ``numpy``
promises no bitwise-identical ``sin``/``cos`` across SIMD paths and builds, a single ULP there shifts a polygon
coordinate by ``~1e-14`` px, and a full-precision ``float64`` hash turns that into a different digest. That is not
hypothetical — it is what turned every CI job red on a snapshot taken on a developer's arm64 machine while CI runs
x86_64. Rounding onto a grid ten orders coarser than the drift and eight orders finer than anything a reader would call
a label change keeps the sensitivity that matters and drops the sensitivity that does not.

"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to a type checker
    from synth_datasets.core.sample import Sample

#: Grid every label float is rounded onto before it enters a digest, in pixels (and in radians for
#: an angle). Sits between two scales that are far apart: floating-point drift between machines is
#: around ``1e-14``, and the smallest label change a reader would ever call real is far above
#: ``1e-4``. Anything in that gap works; the value is stated once so the two ends stay visible.
_LABEL_GRID = 1e-4


def quantized(values: object) -> bytes:
    """Return a flat float sequence snapped to :data:`_LABEL_GRID` as little-endian ``int64`` bytes.

    Args:
        values: Anything :func:`numpy.asarray` accepts as a float sequence, including an empty one.

    Returns:
        Each value divided by the grid and rounded to the nearest integer, as ``<i8`` bytes. Integers
        rather than rounded floats on purpose: a rounded float still carries a binary representation
        that two libms can disagree about, where an integer count of grid steps cannot.

    """
    steps = np.rint(np.asarray(values, dtype=np.float64).reshape(-1) / _LABEL_GRID)
    return steps.astype("<i8").tobytes()


def feed(digest: hashlib._Hash, sample: Sample) -> None:
    """Fold one sample's pixels and every label field it carries into a running digest.

    Args:
        digest: The hash object to update in place.
        sample: The sample to absorb.

    Each field is length-prefixed by the update order alone: the annotation count and the landmark
    count both enter as their own byte blocks, so two samples that differ only in how their fields
    split cannot collide.

    """
    digest.update(sample.image.tobytes())
    digest.update(quantized([sample.width, sample.height, len(sample.annotations)]))
    for annotation in sample.annotations:
        digest.update(quantized([annotation.class_id]))
        digest.update(quantized([annotation.angle]))
        digest.update(annotation.class_name.encode("utf-8"))
        digest.update(quantized(annotation.polygon))
        digest.update(quantized(annotation.bbox_xyxy))
        keypoints = annotation.keypoints or ()
        digest.update(quantized([len(keypoints)]))
        digest.update(quantized([value for triple in keypoints for value in triple]))


def stream_digest(config: object, seed: int, stream_length: int) -> str:
    """Return the hex digest of one configuration's seeded sample stream.

    Args:
        config: The configuration to generate from.
        seed: The seed handed to :meth:`SyntheticGenerator.generate`.
        stream_length: How many samples to fold in.

    Returns:
        The SHA-256 hex digest over the whole stream.

    """
    from synth_datasets.core.generator import SyntheticGenerator

    digest = hashlib.sha256()
    for sample in SyntheticGenerator(config).generate(stream_length, seed=seed):  # type: ignore[arg-type]
        feed(digest, sample)
    return digest.hexdigest()
