"""Build the byte-level snapshot the generator's pixel and label output is pinned against.

:mod:`._golden` pins *unit-space geometry* — what a shape looks like before it is placed. This
snapshot pins the other end: the pixels and labels a fully configured run actually emits. The two
answer different questions, and only this one can refuse a change that moves a placement, a fill, or
a visibility flag while every outline stays put.

It exists because the backgrounds/degradations work adds knobs that must be provably inert when
unset. A test comparing two configurations built from the *same* working tree — which is what
``test_asymmetry_jitter_default_leaves_placement_unchanged`` does, correctly, for its own narrower
question — cannot detect a regression that moved both. Only a digest committed before the change
can. This module is therefore snapshotted first, from untouched code, and any later commit that
legitimately moves a byte regenerates it in the same change that moved it.

Regenerate deliberately, never incidentally::

    FUSE_REGEN_BASELINE=1 uv run pytest tests/test_unit/test_data/test_baseline_digests.py

The digest covers more than the pixels: the class id, the outline, the axis-aligned box, the angle
and the landmark table all feed it, so a run that renders identically but relabels an object still
fails. How each half enters — pixels exactly, labels onto a grid — is :mod:`._digest`'s subject, and
the reason the two are treated differently is worth reading before regenerating anything.

"""

from __future__ import annotations

from synth_datasets.content.backgrounds import (
    GradientBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    TextureBackground,
)
from synth_datasets.content.degradations import (
    JPEG,
    ColorCast,
    Contrast,
    GaussianBlur,
    GaussianNoise,
    Quantize,
    Vignette,
)
from synth_datasets.core.config import ClassMode, Color, SyntheticConfig, Task
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.letters import LetterShape

from ._digest import stream_digest

#: Samples drawn per configuration. More than one on purpose: a single sample cannot see a knob that
#: perturbs the shared :class:`numpy.random.Generator` across a stream, which is exactly the failure
#: mode the side-stream RNG contract exists to prevent.
_STREAM_LENGTH = 3

#: The configurations the digest covers, as ``name -> (config, seed)``. Spelled out one by one rather
#: than built as a product: the product would silently include ``Task.KEYPOINTS`` with primitive
#: shapes, which :class:`SyntheticConfig` rejects outright, and small explicit cases are easier to
#: extend than a filtered cartesian sweep.
#:
#: **These entries were snapshotted before any background, degradation, distractor or occluder knob
#: existed**, so they carry a guarantee the entries below cannot: that the knobs are inert when
#: unset. Regenerating one is a claim that the *old* behaviour legitimately moved, which is a much
#: larger claim than regenerating a feature entry. Never add a configuration that sets a new knob
#: here — it would dilute exactly that distinction.
_MATRIX: dict[str, tuple[SyntheticConfig, int]] = {
    "detection-shape-primitives-64": (SyntheticConfig(img_size=64), 0),
    "detection-shape-primitives-192": (SyntheticConfig(img_size=192), 0),
    "detection-color-primitives-64": (SyntheticConfig(img_size=64, class_mode=ClassMode.COLOR), 1),
    "detection-shapecolor-primitives-64": (SyntheticConfig(img_size=64, class_mode=ClassMode.SHAPE_COLOR), 1),
    "segmentation-shape-primitives-64": (SyntheticConfig(img_size=64, task=Task.SEGMENTATION), 0),
    "obb-shape-primitives-64": (SyntheticConfig(img_size=64, task=Task.OBB), 2),
    "keypoints-shape-animals-64": (SyntheticConfig(img_size=64, task=Task.KEYPOINTS, shapes=tuple(AnimalShape)), 0),
    "keypoints-shape-animals-192": (SyntheticConfig(img_size=192, task=Task.KEYPOINTS, shapes=tuple(AnimalShape)), 3),
    "detection-shape-animals-64": (SyntheticConfig(img_size=64, shapes=tuple(AnimalShape)), 4),
    "detection-shape-letters-64": (SyntheticConfig(img_size=64, shapes=tuple(LetterShape)), 5),
    "detection-norotate-primitives-64": (SyntheticConfig(img_size=64, rotate=False), 0),
    "detection-jitter-primitives-64": (SyntheticConfig(img_size=64, asymmetry_jitter=0.2), 0),
    "detection-dense-small-primitives-64": (
        SyntheticConfig(img_size=64, min_objects=3, max_objects=8, min_size_ratio=0.05, max_size_ratio=0.2),
        6,
    ),
    "detection-customfill-primitives-64": (SyntheticConfig(img_size=64, colors=((255, 215, 0), Color.RED)), 7),
}


#: Configurations exercising the knobs added by the backgrounds-and-difficulty work. These were
#: snapshotted from the code that implements them, so they prove **stability**, not inertness: a
#: digest here moving means a rendering change, deliberate or not, and a reviewer decides which. That
#: is a weaker guarantee than :data:`_MATRIX`'s and a necessary one — nothing in that matrix touches
#: :mod:`~synth_datasets.content.backgrounds` or
#: :mod:`~synth_datasets.content.degradations` at all, so a regression inside either module moved no
#: digest before these existed.
#:
#: :class:`~synth_datasets.content.backgrounds.ImageBackground` is deliberately absent: it reads
#: files this package does not ship, so pinning it would pin a fixture directory rather than the
#: renderer. Its own tests cover it against pictures they write themselves.
_FEATURE_MATRIX: dict[str, tuple[SyntheticConfig, int]] = {
    "bg-gradient-sampled-64": (SyntheticConfig(img_size=64, background=GradientBackground()), 0),
    "bg-gradient-radial-64": (
        SyntheticConfig(img_size=64, background=GradientBackground(radial=True, stops=((20, 20, 60), (200, 200, 240)))),
        1,
    ),
    "bg-noise-64": (SyntheticConfig(img_size=64, background=NoiseBackground(sigma=24.0)), 0),
    "bg-impulse-64": (SyntheticConfig(img_size=64, background=ImpulseNoiseBackground(amount=0.08)), 2),
    "bg-texture-64": (SyntheticConfig(img_size=64, background=TextureBackground(octaves=3, frequency=6.0)), 0),
    "bg-texture-quantized-64": (
        SyntheticConfig(img_size=64, background=TextureBackground(octaves=2, frequency=4.0, quantize=5)),
        3,
    ),
    "degrade-chain-64": (
        SyntheticConfig(
            img_size=64,
            degrade=(GaussianBlur(radius=0.8), JPEG(quality=55), Contrast(factor=0.6), Quantize(levels=12)),
        ),
        0,
    ),
    "degrade-noise-cast-vignette-64": (
        SyntheticConfig(
            img_size=64,
            degrade=(GaussianNoise(sigma=9.0), ColorCast(gain=(1.15, 1.0, 0.85)), Vignette(strength=0.4)),
        ),
        4,
    ),
    "distractors-64": (SyntheticConfig(img_size=64, min_objects=2, max_objects=4, distractors=5), 0),
    "occluders-keypoints-96": (
        SyntheticConfig(
            img_size=96,
            task=Task.KEYPOINTS,
            shapes=tuple(AnimalShape)[:4],
            min_objects=3,
            max_objects=3,
            min_size_ratio=0.2,
            max_size_ratio=0.35,
            occluders=5,
        ),
        0,
    ),
    "everything-96": (
        SyntheticConfig(
            img_size=96,
            background=TextureBackground(octaves=2, frequency=5.0),
            min_objects=2,
            max_objects=4,
            distractors=4,
            occluders=3,
            degrade=(GaussianBlur(radius=0.5), JPEG(quality=70)),
        ),
        5,
    ),
}


def baseline_names() -> tuple[str, ...]:
    """Return the configuration names the snapshot covers, in matrix order.

    Returns:
        The keys :func:`build_baseline` produces, both matrices together. Exposed separately so the
        check can parametrize over them without generating every image at collection time.

    """
    return tuple(_MATRIX) + tuple(_FEATURE_MATRIX)


def build_baseline() -> dict[str, str]:
    """Return the ``configuration name -> hex digest`` snapshot over the whole matrix.

    Returns:
        One SHA-256 hex digest per entry of :data:`_MATRIX` and :data:`_FEATURE_MATRIX`, taken over
        a :data:`_STREAM_LENGTH`-sample stream from that configuration's own seed. The two matrices
        answer different questions — see each one's own note — and are kept apart for that reason
        even though they share a file.

    """
    return {
        name: stream_digest(config, seed, _STREAM_LENGTH)
        for name, (config, seed) in (*_MATRIX.items(), *_FEATURE_MATRIX.items())
    }
