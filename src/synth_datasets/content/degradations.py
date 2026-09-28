"""Pointwise effects baked into a rendered image after every shape has been drawn.

A degradation changes pixel *values* and never pixel *positions*, which is what keeps the generator's
labels exact for the pixels it emits: noise, blur, a JPEG round-trip, a contrast or colour shift and
a vignette all leave a box where it was. Anything geometric — an elastic warp, a perspective change,
a motion smear — is deliberately absent and belongs to
:class:`~fused_transforms.FusedCompose`, which is what this package fuses third-party transforms
for in the first place.

The overlap with that pipeline is smaller than it looks. A `degrade` tuple describes pixels **baked
into an on-disk dataset** by :func:`~synth_datasets.generate_dataset` — a fixed property of
the data, replayable from its seed — where a transform in a training loop resamples every epoch. The
two answer different questions and a dataset can carry both.

Like a background, a degradation is a type rather than a mode string, draws only from the generator's
side stream, and declares whether it draws at all through :attr:`Degradation.consumes_randomness`.
Each step takes ``uint8`` and returns ``uint8``, so quantisation error accumulates between steps
rather than being carried in float to the end — which is what a real camera pipeline does, and what
keeps every effect independently testable.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from synth_datasets.content.degradations import Contrast, Quantize
    >>> image = np.full((4, 4, 3), 200, dtype=np.uint8)
    >>> int(Quantize(levels=2).apply(image, None).max())
    255
    >>> Contrast(factor=0.5).apply(image, None).shape
    (4, 4, 3)

    ```

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from synth_datasets._render import require_stream, to_uint8

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Widest JPEG quality Pillow encodes meaningfully. Above 95 the file grows without a visible gain,
#: which Pillow's own documentation says outright, so the ceiling is the library's rather than ours.
_MAX_JPEG_QUALITY = 95


class Degradation(ABC):
    """One pointwise effect applied to a finished image, in the order the tuple lists it.

    Implement :meth:`apply` and, when the effect has no randomness of its own, override :attr:`consumes_randomness` —
    every parameter here is a fixed scalar by design, so most effects draw nothing and only the ones sampling a field
    per pixel do.

    """

    @property
    def consumes_randomness(self) -> bool:
        """Return whether :meth:`apply` draws from the generator it is handed.

        Defaults to ``False``, the opposite of a background's default: a degradation is configured by explicit scalars,
        so drawing is the exception rather than the rule.

        """
        return False

    @abstractmethod
    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Return the degraded image.

        Args:
            image: ``(height, width, 3)`` ``uint8`` RGB pixels, already carrying every earlier step.
            rng: The side stream, or ``None`` when :attr:`consumes_randomness` is ``False``.

        Returns:
            A ``uint8`` image of the same shape, in a buffer the caller owns. The Pillow-backed steps
            build theirs with :func:`numpy.array` rather than :func:`numpy.asarray` for that reason:
            an ``asarray`` of a Pillow image is a read-only view onto the library's own memory, so a
            chain ending on one would hand back a sample whose writability depended on which effect
            happened to run last.

        """


@dataclass(frozen=True)
class GaussianNoise(Degradation):
    """Additive per-pixel Gaussian noise over the finished image.

    Costs a model edge localisation and small-object recall: the noise floor sits on the object and
    the canvas alike, so a small shape's boundary stops being the strongest local gradient.

    Args:
        sigma: Per-channel standard deviation in 8-bit units.

    Raises:
        ValueError: If ``sigma`` is negative.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import GaussianNoise
        >>> GaussianNoise(sigma=10.0).apply(np.zeros((4, 4, 3), np.uint8), np.random.default_rng(0)).dtype
        dtype('uint8')

        ```

    """

    sigma: float = 8.0

    def __post_init__(self) -> None:
        """Reject a negative spread."""
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    @property
    def consumes_randomness(self) -> bool:
        """Return ``True``: one Gaussian field is drawn per image."""
        return True

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Add one Gaussian field over the whole image, drawn in a single call."""
        stream = require_stream(rng, type(self).__name__)
        noise = stream.standard_normal(image.shape).astype(np.float32) * float(self.sigma)
        return to_uint8(image.astype(np.float32) + noise)


@dataclass(frozen=True)
class GaussianBlur(Degradation):
    """A Gaussian blur of a fixed radius.

    The step that costs corner keypoints and, on a square-ish shape, the oriented box's angle: a
    blurred corner is no longer a corner, and the pose has to be inferred from the silhouette.

    Args:
        radius: Blur radius in pixels, as Pillow's :class:`PIL.ImageFilter.GaussianBlur` reads it.

    Raises:
        ValueError: If ``radius`` is negative.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import GaussianBlur
        >>> edge = np.zeros((8, 8, 3), np.uint8)
        >>> edge[:, 4:] = 255
        >>> bool(GaussianBlur(radius=1.5).apply(edge, None)[0, 3, 0] > 0)
        True

        ```

    """

    radius: float = 1.0

    def __post_init__(self) -> None:
        """Reject a negative radius."""
        if self.radius < 0:
            raise ValueError(f"radius must be non-negative, got {self.radius}")

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Blur through Pillow, which owns the kernel this package does not reimplement."""
        from PIL import Image, ImageFilter

        blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=float(self.radius)))
        return np.array(blurred, dtype=np.uint8)


@dataclass(frozen=True)
class JPEG(Degradation):
    """A JPEG encode/decode round-trip at a fixed quality.

    Block artefacts around a thin letter stroke are the point, and they are realistic rather than
    synthetic: anything that came out of a camera pipeline carries them.

    Args:
        quality: Encoder quality from 1 to 95. Pillow treats anything above 95 as wasted file size
            rather than added fidelity, so that is the ceiling.

    Raises:
        ValueError: If ``quality`` falls outside ``[1, 95]``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import JPEG
        >>> JPEG(quality=30).apply(np.full((16, 16, 3), 128, np.uint8), None).shape
        (16, 16, 3)

        ```

    """

    quality: int = 75

    def __post_init__(self) -> None:
        """Reject a quality the encoder would not honour."""
        if not 1 <= self.quality <= _MAX_JPEG_QUALITY:
            raise ValueError(f"quality must be within [1, {_MAX_JPEG_QUALITY}], got {self.quality}")

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Encode to an in-memory JPEG and decode it back, keeping whatever the codec did."""
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=int(self.quality))
        buffer.seek(0)
        return np.array(Image.open(buffer).convert("RGB"), dtype=np.uint8)


@dataclass(frozen=True)
class Contrast(Degradation):
    """A contrast scaling around the image's own mean luminance.

    Under :attr:`~synth_datasets.core.config.ClassMode.COLOR` this is the knob that makes classes
    converge toward each other, since every fill moves toward the same grey.

    Args:
        factor: ``1.0`` leaves the image alone, below 1 flattens it, above 1 stretches it.

    Raises:
        ValueError: If ``factor`` is negative.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import Contrast
        >>> flat = np.stack([np.full((4, 4), v, np.uint8) for v in (0, 255, 128)], axis=2)
        >>> int(Contrast(factor=0.0).apply(flat, None).std())
        0

        ```

    """

    factor: float = 0.7

    def __post_init__(self) -> None:
        """Reject a negative factor, which has no meaning as a contrast scaling."""
        if self.factor < 0:
            raise ValueError(f"factor must be non-negative, got {self.factor}")

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Scale every channel toward the image's mean grey, which is what Pillow's enhancer does."""
        from PIL import Image, ImageEnhance

        enhanced = ImageEnhance.Contrast(Image.fromarray(image)).enhance(float(self.factor))
        return np.array(enhanced, dtype=np.uint8)


@dataclass(frozen=True)
class ColorCast(Degradation):
    """A per-channel gain, the way a white balance error looks.

    What it costs is red-versus-green separation: two fills that were far apart in hue sit closer
    together once one channel has been lifted and another pulled down.

    Args:
        gain: One multiplier per channel, in ``(r, g, b)`` order.

    Raises:
        ValueError: If ``gain`` does not hold exactly three non-negative multipliers.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import ColorCast
        >>> grey = np.full((2, 2, 3), 100, np.uint8)
        >>> ColorCast(gain=(1.2, 1.0, 0.8)).apply(grey, None)[0, 0].tolist()
        [120, 100, 80]

        ```

    """

    gain: tuple[float, float, float] = (1.1, 1.0, 0.9)

    def __post_init__(self) -> None:
        """Reject anything that is not three non-negative multipliers."""
        gain = tuple(self.gain)
        if len(gain) != 3 or any(value < 0 for value in gain):
            raise ValueError(f"gain must hold three non-negative per-channel multipliers, got {self.gain!r}")
        object.__setattr__(self, "gain", gain)

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Multiply each channel by its own gain and clip once."""
        return to_uint8(image.astype(np.float32) * np.asarray(self.gain, dtype=np.float32))


@dataclass(frozen=True)
class Vignette(Degradation):
    """A radial darkening toward the corners.

    Interacts with ``boundary_tolerance`` on purpose: an object allowed to sit half off the frame now
    also sits in the darkest part of it, which is the combination a border-region detector fails on.

    Args:
        strength: Fraction of brightness removed at the corners; ``0.0`` leaves the image alone.

    Raises:
        ValueError: If ``strength`` falls outside ``[0, 1]``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import Vignette
        >>> flat = np.full((9, 9, 3), 200, np.uint8)
        >>> out = Vignette(strength=0.5).apply(flat, None)
        >>> bool(out[0, 0, 0] < out[4, 4, 0])
        True

        ```

    """

    strength: float = 0.3

    def __post_init__(self) -> None:
        """Reject a strength outside the unit interval."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be within [0, 1], got {self.strength}")

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Scale each pixel by a quadratic falloff from the centre to the corners."""
        height, width = image.shape[:2]
        rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
        centre_y, centre_x = (height - 1) / 2.0, (width - 1) / 2.0
        distance = np.hypot(rows - centre_y, columns - centre_x)
        normalized = distance / max(float(np.hypot(centre_y, centre_x)), 1e-6)
        falloff = 1.0 - float(self.strength) * normalized**2
        return to_uint8(image.astype(np.float32) * falloff[..., None])


@dataclass(frozen=True)
class Quantize(Degradation):
    """A reduction to a fixed number of evenly spaced levels per channel.

    A flat fill becomes banded, which is the cheap stand-in for a low-bit-depth sensor and the one
    degradation that makes a gradient background visibly wrong rather than merely dimmer.

    Args:
        levels: Number of levels per channel, from 2 to 256.

    Raises:
        ValueError: If ``levels`` falls outside ``[2, 256]``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.degradations import Quantize
        >>> ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)[..., None].repeat(3, axis=2)
        >>> len(np.unique(Quantize(levels=4).apply(ramp, None)))
        4

        ```

    """

    levels: int = 16

    def __post_init__(self) -> None:
        """Reject a level count that would not quantise anything or would exceed the channel."""
        if not 2 <= self.levels <= 256:
            raise ValueError(f"levels must be within [2, 256], got {self.levels}")

    def apply(self, image: NDArray[np.uint8], rng: np.random.Generator | None) -> NDArray[np.uint8]:
        """Snap each channel to the nearest of ``levels`` evenly spaced values."""
        steps = int(self.levels) - 1
        return to_uint8(np.rint(image.astype(np.float32) / 255.0 * steps) / steps * 255.0)
