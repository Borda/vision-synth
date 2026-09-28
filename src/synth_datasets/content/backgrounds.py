"""Canvas fillers the synthetic generator draws its objects on top of.

A background is a *type*, not a mode string. :class:`Background` is abstract and every concrete
subclass carries exactly the parameters its own mode needs, with defaults, and renders itself. That
removes the question a ``mode="noise"`` field would create — which parameters are legal together —
because a class that has no ``sigma`` cannot be given one, and the constructor says so rather than a
hand-written cross-field check. It is also the extension point: a third party's own
:class:`Background` works with no registration, matching the
:func:`~synth_datasets.export.writers.register_writer` culture already in the package.

Every background renders from a **side stream**, never from the generator's own placement stream, so
turning one on cannot move an object. :attr:`Background.consumes_randomness` is what
:class:`~synth_datasets.core.generator.SyntheticGenerator` asks before taking that side stream at
all, which is why a background that draws nothing is handed ``None`` and must not touch it.

Numpy only at module scope, plus Pillow inside the one method that needs a resampler: importing this
module is as cheap as importing :mod:`~synth_datasets.core.config`, which it deliberately does
not force to grow heavier.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from synth_datasets.content.backgrounds import NoiseBackground, SolidBackground
    >>> SolidBackground((10, 20, 30)).render(None, 4).shape
    (4, 4, 3)
    >>> canvas = NoiseBackground(sigma=8.0).render(np.random.default_rng(0), 8)
    >>> canvas.dtype
    dtype('uint8')

    ```

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import numpy as np

from synth_datasets._render import require_stream, to_uint8
from synth_datasets.core.config import ColorLike, Fill, as_canvas_size

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL.Image import Image as PILImage

#: What :meth:`Background.render` receives: a square side, or ``(width, height)``.
CanvasSize = int | tuple[int, int]

#: The grey every mode falls back to, and the fill the generator drew before backgrounds were types.
#: Spelled once here so the default of five dataclasses cannot drift apart.
DEFAULT_BASE: tuple[int, int, int] = (128, 128, 128)

#: Highest sigma :class:`NoiseBackground` may use while the rasterized-ink oracle in
#: ``tests/test_unit/test_data/test_bbox_from_polygon.py`` stays sound. That oracle finds ink by
#: distance to pure red under a tolerance of 90, and Gaussian noise is unbounded, so its tail decides
#: this rather than its mean. Measured on a 192x192x3 canvas around ``(128, 128, 128)``: 0 false-ink
#: pixels at sigma 8, 16 and 32, then 36 at 48 and 248 at 64. A single stray pixel widens the
#: oracle's box, so the bound is the last value that produced none.
INK_SAFE_SIGMA = 32.0

#: File suffixes :class:`ImageBackground` will open, lowercased.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})

#: Directories whose listing is remembered. A handful covers any real program; the bound keeps a
#: caller that builds paths programmatically from retaining one listing per path forever.
_DIRECTORY_CACHE_SIZE = 32


def _rgb(color: ColorLike) -> NDArray[np.float32]:
    """Return a fill as a ``(3,)`` float array, validating the spelling on the way through."""
    return np.asarray(Fill.parse(color).rgb, dtype=np.float32)


class Background(ABC):
    """A canvas filler whose subclass *is* the mode, so there is no mode field.

    Implement :meth:`render` and, when the mode draws nothing, override :attr:`consumes_randomness` — the generator
    reads it to decide whether to take a side stream at all, and a background that claims to draw nothing is handed
    ``None`` in place of one.

    """

    @property
    def consumes_randomness(self) -> bool:
        """Return whether :meth:`render` draws from the generator it is handed.

        Defaults to ``True``, which is always safe: a side stream that is taken and not used costs one cheap child and
        moves nothing, while a background that draws from a stream it said it would not need would receive ``None`` and
        fail loudly rather than quietly.

        """
        return True

    @abstractmethod
    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return the canvas this background fills, as ``(height, width, 3)`` ``uint8``.

        Args:
            rng: The side stream to draw from, or ``None`` when :attr:`consumes_randomness` is
                ``False`` for this instance. It is never the generator's placement stream.
            img_size: The canvas side as an ``int`` when it is square, or ``(width, height)`` when
                it is not. The generator passes an ``int`` for every square canvas, so a subclass
                that only ever serves square configs may treat it as one; the built-ins normalize
                both spellings through :func:`~synth_datasets.core.config.as_canvas_size`.

        Returns:
            A writable, C-contiguous RGB canvas.

        """

    def render_with_source(
        self, rng: np.random.Generator | None, img_size: CanvasSize
    ) -> tuple[NDArray[np.uint8], str | None]:
        """Return the canvas together with whatever names where its pixels came from.

        Args:
            rng: The side stream, exactly as :meth:`render` takes it.
            img_size: The canvas size, exactly as :meth:`render` takes it.

        Returns:
            The canvas and a provenance string, or ``None`` when the mode is procedural and there is
            nothing outside the process to name.

        This is what the generator calls, and what ends up in
        :attr:`~synth_datasets.core.sample.SceneRecord.background_source`. It is not abstract: a
        procedural mode has no source, so the default delegates to :meth:`render` and reports
        ``None``, which means a third-party background implementing only :meth:`render` keeps
        working. Only a mode reading files outside the package overrides it.

        """
        return self.render(rng, img_size), None


@dataclass(frozen=True)
class SolidBackground(Background):
    """One flat colour across the whole canvas — the behaviour that predates this module.

    Args:
        color: The fill, as a :class:`~synth_datasets.core.config.Color`, an ``(r, g, b)``
            triple, or a :class:`~synth_datasets.core.config.Fill`. Normalized to a ``Fill`` at
            construction, like every other fill in the package.

    Examples:
        ```pycon
        >>> from synth_datasets.content.backgrounds import SolidBackground
        >>> SolidBackground((20, 30, 40)).render(None, 2)[0, 0].tolist()
        [20, 30, 40]

        ```

    """

    color: ColorLike = DEFAULT_BASE

    def __post_init__(self) -> None:
        """Normalize the fill once, so every later read holds a :class:`Fill`."""
        object.__setattr__(self, "color", Fill.parse(self.color))

    @property
    def consumes_randomness(self) -> bool:
        """Return ``False``: a flat fill has nothing to draw."""
        return False

    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return a canvas of one repeated colour, ignoring ``rng`` entirely.

        Built with :func:`numpy.full` rather than a broadcast view made contiguous: at ``img_size=1`` every axis is
        length one, so the broadcast result is *already* flagged C-contiguous and :func:`numpy.ascontiguousarray` hands
        it straight back — read-only, which breaks the writability half of the base class's contract at exactly one
        canvas size.

        """
        width, height = as_canvas_size(img_size)
        return np.full((height, width, 3), _rgb(self.color).astype(np.uint8), dtype=np.uint8)


@dataclass(frozen=True)
class GradientBackground(Background):
    """A linear or radial ramp between two stops.

    Breaks the "background is one value" assumption a colour-mode classifier can otherwise lean on:
    the same object fill now sits on a different local intensity depending on where it landed.

    Args:
        stops: The two ends of the ramp. For a linear ramp the first is at the low end of the
            direction axis; for a radial one it is at the canvas centre and the second at the
            corners.
        direction: Ramp angle in radians for a linear ramp, or ``None`` to sample one per image.
            Ignored entirely when ``radial`` is set, which has no direction to speak of.
        radial: Ramp outward from the centre rather than across the canvas.

    Raises:
        ValueError: If ``stops`` does not hold exactly two fills.

    Examples:
        ```pycon
        >>> from synth_datasets.content.backgrounds import GradientBackground
        >>> ramp = GradientBackground(direction=0.0).render(None, 4)
        >>> bool(ramp[0, 0, 0] < ramp[0, -1, 0])
        True

        ```

    """

    stops: tuple[ColorLike, ColorLike] = ((64, 64, 64), (192, 192, 192))
    direction: float | None = None
    radial: bool = False

    def __post_init__(self) -> None:
        """Normalize both stops and reject a ramp that does not have exactly two ends."""
        if len(tuple(self.stops)) != 2:
            raise ValueError(f"stops must hold exactly two fills, got {len(tuple(self.stops))}")
        object.__setattr__(self, "stops", tuple(Fill.parse(stop) for stop in self.stops))

    @property
    def consumes_randomness(self) -> bool:
        """Return whether the ramp angle still has to be sampled.

        Only a linear ramp with no fixed ``direction`` draws. A radial ramp is centred and has no angle at all, so it
        draws nothing whatever ``direction`` says — which is why the draw count is keyed on this property rather than on
        ``direction`` alone.

        """
        return self.direction is None and not self.radial

    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return the ramp, interpolating between the stops in float and rounding once."""
        first, second = (_rgb(stop) for stop in self.stops)
        return to_uint8(first + self._ramp(rng, img_size)[..., None] * (second - first))

    def _ramp(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.float32]:
        """Return the scalar field the stops are interpolated over, normalized to ``[0, 1]``.

        The linear field is rescaled by its own extent rather than by a closed form, so the ramp spans both stops
        exactly at every angle instead of compressing toward the diagonals.

        """
        width, height = as_canvas_size(img_size)
        rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
        if self.radial:
            distance = np.hypot(rows - (height - 1) / 2.0, columns - (width - 1) / 2.0)
            return np.asarray(distance / max(float(distance.max()), 1e-6), dtype=np.float32)
        if self.direction is not None:
            angle = float(self.direction)
        else:
            angle = float(require_stream(rng, type(self).__name__).uniform(0.0, 2.0 * math.pi))
        field = math.cos(angle) * columns + math.sin(angle) * rows
        span = float(field.max() - field.min())
        return np.asarray((field - field.min()) / max(span, 1e-6), dtype=np.float32)


@dataclass(frozen=True)
class NoiseBackground(Background):
    """Per-pixel Gaussian noise around a base colour.

    The first knob that makes a small object genuinely hard: it removes the trivial edge detector a
    flat canvas hands a model for free.

    Args:
        base: The colour the noise is centred on.
        sigma: Per-channel standard deviation in 8-bit units. Keep it at or below
            :data:`INK_SAFE_SIGMA` for any run scored by a rasterized-ink oracle; see that constant
            for the measurement.

    Raises:
        ValueError: If ``sigma`` is negative.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.backgrounds import NoiseBackground
        >>> NoiseBackground(sigma=4.0).render(np.random.default_rng(0), 4).shape
        (4, 4, 3)

        ```

    """

    base: ColorLike = DEFAULT_BASE
    sigma: float = 16.0

    def __post_init__(self) -> None:
        """Normalize the base fill and reject a negative spread."""
        object.__setattr__(self, "base", Fill.parse(self.base))
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return the base colour plus one Gaussian field, added rather than multiplied."""
        stream = require_stream(rng, type(self).__name__)
        width, height = as_canvas_size(img_size)
        noise = stream.standard_normal((height, width, 3)).astype(np.float32) * float(self.sigma)
        return to_uint8(_rgb(self.base) + noise)


@dataclass(frozen=True)
class ImpulseNoiseBackground(Background):
    """Salt-and-pepper pixels scattered over a base colour.

    The same axis as :class:`NoiseBackground` but heavy-tailed, and the reason it is a separate
    difficulty step: impulse pixels survive a blur that erases a Gaussian field.

    Args:
        base: The colour the surviving pixels keep.
        amount: Fraction of pixels replaced, in ``[0, 1]``.
        salt_ratio: Of those, the fraction set white rather than black, in ``[0, 1]``.

    Raises:
        ValueError: If ``amount`` or ``salt_ratio`` falls outside ``[0, 1]``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.backgrounds import ImpulseNoiseBackground
        >>> canvas = ImpulseNoiseBackground(amount=0.5).render(np.random.default_rng(0), 16)
        >>> bool((canvas == 255).any() and (canvas == 0).any())
        True

        ```

    """

    base: ColorLike = DEFAULT_BASE
    amount: float = 0.05
    salt_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Normalize the base fill and reject fractions outside the unit interval."""
        object.__setattr__(self, "base", Fill.parse(self.base))
        for name, value in (("amount", self.amount), ("salt_ratio", self.salt_ratio)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")

    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return the base colour with a fraction of pixels replaced outright by black or white.

        Replacement, not addition: adding a fixed salt value to an arbitrary base would not produce endpoint pixels,
        which is the whole point of impulse noise and the reason it survives a blur.

        """
        stream = require_stream(rng, type(self).__name__)
        width, height = as_canvas_size(img_size)
        canvas = np.broadcast_to(_rgb(self.base), (height, width, 3)).copy()
        selected = stream.random((height, width)) < float(self.amount)
        salt = stream.random((height, width)) < float(self.salt_ratio)
        canvas[selected & salt] = 255.0
        canvas[selected & ~salt] = 0.0
        return to_uint8(canvas)


@dataclass(frozen=True)
class TextureBackground(Background):
    """Value noise at a chosen spatial frequency, optionally posterized.

    The first background that puts *structure* at object scale, so a false positive becomes possible
    rather than merely a matter of contrast.

    Args:
        base: The colour the field deviates around.
        amplitude: Maximum deviation from ``base`` in 8-bit units — an upper bound rather than a
            value reached. The octaves are summed and divided by their weight total, so a
            multi-octave field spans less than the full ``[-1, 1]`` and its peak falls short of
            ``amplitude`` by however much the octaves failed to align; at ``octaves=5`` the largest
            deviation measured over eight seeds was 46 of a requested 48. Only ``octaves=1`` reaches
            the bound.
        frequency: Lattice cells across the image at the first octave.
        octaves: Number of octaves summed, each at twice the frequency and half the weight.
        quantize: Number of levels the normalized ``[-1, 1]`` range is divided into before the field
            is mapped to colour, or ``None`` for a continuous field. The count of levels a rendered
            canvas actually shows is this or fewer, for the reason given under ``amplitude``: a
            multi-octave field does not span the whole range, so some levels have nothing in them.
            ``octaves=1`` shows exactly this many.

    Raises:
        ValueError: If ``amplitude`` is negative, ``frequency`` is not positive, ``octaves`` is below
            1, or ``quantize`` is given and below 2.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.content.backgrounds import TextureBackground
        >>> TextureBackground(octaves=2).render(np.random.default_rng(0), 32).shape
        (32, 32, 3)

        ```

    """

    base: ColorLike = DEFAULT_BASE
    amplitude: float = 48.0
    frequency: float = 8.0
    octaves: int = 3
    quantize: int | None = None

    def __post_init__(self) -> None:
        """Normalize the base fill and reject a lattice that cannot be built."""
        object.__setattr__(self, "base", Fill.parse(self.base))
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {self.amplitude}")
        if self.frequency <= 0:
            raise ValueError(f"frequency must be positive, got {self.frequency}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be at least 1, got {self.octaves}")
        if self.quantize is not None and self.quantize < 2:
            raise ValueError(f"quantize must be at least 2 levels when given, got {self.quantize}")

    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return the base colour plus the summed octaves, added rather than multiplied.

        Additive on purpose: a multiplicative field would make the effective contrast depend on
        ``base``, so the same ``amplitude`` would mean different things on a dark and a light canvas.

        """
        field = self._field(require_stream(rng, type(self).__name__), img_size)
        return to_uint8(_rgb(self.base) + float(self.amplitude) * field[..., None])

    def _field(self, rng: np.random.Generator, img_size: CanvasSize) -> NDArray[np.float32]:
        """Return the summed, normalized and optionally posterized value-noise field in ``[-1, 1]``.

        One uniform lattice is drawn per octave and upsampled bicubically, so the draw count is the octave count
        exactly. Weights halve per octave and the sum is divided by their total, which is what bounds the result rather
        than a second pass over the data.

        """
        from PIL import Image

        width, height = as_canvas_size(img_size)
        short = min(width, height)
        field = np.zeros((height, width), dtype=np.float32)
        weights = 0.0
        for octave in range(self.octaves):
            # ``frequency`` counts features across the shorter side; the longer side gets proportionally more cells,
            # so a rectangular canvas shows more features rather than stretched ones. On a square canvas both axes
            # get the same count, and the draw is the same ``(cells, cells)`` lattice it always was.
            features = float(self.frequency) * 2**octave
            cells_x = math.ceil(features * (width / short)) + 1
            cells_y = math.ceil(features * (height / short)) + 1
            lattice = rng.uniform(-1.0, 1.0, size=(cells_y, cells_x)).astype(np.float32)
            upsampled = Image.fromarray(lattice, mode="F").resize((width, height), Image.Resampling.BICUBIC)
            weight = 0.5**octave
            field += np.asarray(upsampled, dtype=np.float32) * weight
            weights += weight
        field = np.clip(field / weights, -1.0, 1.0)
        if self.quantize is None:
            return field
        levels = int(self.quantize) - 1
        return np.asarray(np.rint((field + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0, dtype=np.float32)


@lru_cache(maxsize=_DIRECTORY_CACHE_SIZE)
def _scan(image_dir: Path) -> tuple[Path, ...]:
    """Return the readable images in a directory, sorted, cached per directory for the process.

    Args:
        image_dir: Directory to list.

    Returns:
        Every readable image under the directory, recursively, in sorted order so a seed selects the
        same file on every machine rather than in whatever order the filesystem happens to return.

        Three filters, and the last one is the point: the suffix must name an image format, the path
        must be a *file* — a directory named ``shots.png`` otherwise counted as one — and Pillow must
        be able to parse the header. Without that last check the promise of "readable" was only a
        promise about the filename, and a corrupt file was accepted at construction and then raised
        :class:`PIL.UnidentifiedImageError` from inside rendering, far from the directory that caused
        it.

    Raises:
        ValueError: If the path is not a directory, or holds no image this package can open.

    Cached because the alternative is a directory listing per generated image, which turns a cheap
    background into a syscall-bound one. The cost is that a file added to the directory mid-process
    is not picked up; a background reading a directory that changes underneath it has no reproducible
    meaning anyway, which is the reason to prefer the stale listing here.

    """
    if not image_dir.is_dir():
        raise ValueError(f"ImageBackground.image_dir must be an existing directory, got {image_dir}")
    from PIL import Image, UnidentifiedImageError

    candidates = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    readable = []
    for path in candidates:
        try:
            with Image.open(path) as probe:
                probe.verify()
        except (OSError, UnidentifiedImageError):
            continue
        readable.append(path)
    files = tuple(readable)
    if not files:
        raise ValueError(
            f"ImageBackground.image_dir holds no image this package can open: {image_dir} "
            f"(searched recursively for {', '.join(sorted(_IMAGE_SUFFIXES))}; "
            f"{len(candidates)} matched by name but none could be parsed)"
        )
    return files


@dataclass(frozen=True)
class ImageBackground(Background):
    """Random crops of the caller's own photographs.

    The only mode that reads the outside world, and the only one whose parameter has no default:
    there is nothing to ship a default directory from, and a silently empty one is the failure worth
    refusing outright rather than rendering as black.

    What it buys is real texture statistics — the spatial correlations, gradients and clutter of
    photographs — without any labelling cost, since the labels still come from the shapes drawn on
    top. Which file and which crop were used is reported through
    :attr:`~synth_datasets.core.sample.SceneRecord.background_source`, so a sample can be traced
    back to what it stood on.

    Args:
        image_dir: Directory of images to crop from. Required; scanned and validated at construction.
        grayscale: Drop the colour of the crop, leaving its structure. Useful when the run's classes
            are colour-named and a photographic canvas would otherwise compete with them.

    Raises:
        ValueError: If ``image_dir`` is not a directory or holds no readable image.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from pathlib import Path
        >>> from PIL import Image
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     Image.fromarray(np.full((40, 40, 3), 90, np.uint8)).save(Path(folder) / "a.png")
        ...     canvas, source = ImageBackground(Path(folder)).render_with_source(np.random.default_rng(0), 16)
        >>> canvas.shape, source
        ((16, 16, 3), 'a.png')

        ```

    """

    image_dir: Path
    grayscale: bool = False

    def __post_init__(self) -> None:
        """Normalize the directory to a :class:`~pathlib.Path` and refuse one with nothing in it."""
        object.__setattr__(self, "image_dir", Path(self.image_dir))
        _scan(self.image_dir)

    def render(self, rng: np.random.Generator | None, img_size: CanvasSize) -> NDArray[np.uint8]:
        """Return one random crop, discarding which file it came from."""
        return self.render_with_source(rng, img_size)[0]

    def render_with_source(
        self, rng: np.random.Generator | None, img_size: CanvasSize
    ) -> tuple[NDArray[np.uint8], str | None]:
        """Return one random crop and the name of the file it was taken from.

        Three draws, always in this order and always all three: the file index, then the crop's left and top offsets.
        The offsets are drawn even when only one position is possible, so the draw count does not depend on how large
        the chosen file happens to be.

        """
        from PIL import Image

        stream = require_stream(rng, type(self).__name__)
        files = _scan(self.image_dir)
        chosen = files[int(stream.integers(len(files)))]
        with Image.open(chosen) as opened:
            picture = opened.convert("L" if self.grayscale else "RGB")
            width, height = as_canvas_size(img_size)
            picture = _at_least(picture, width, height)
            left = int(stream.integers(picture.width - width + 1))
            top = int(stream.integers(picture.height - height + 1))
            crop = picture.crop((left, top, left + width, top + height)).convert("RGB")
        source = PurePath(chosen.relative_to(self.image_dir)).as_posix()
        return np.array(crop, dtype=np.uint8), source


def _at_least(picture: PILImage, width: int, height: int) -> PILImage:
    """Return a picture at least ``width`` wide and ``height`` high, upscaling proportionally if not.

    A file smaller than the canvas has no crop to give, and refusing it would make the mode depend on the caller pre-
    sizing a directory. Scaling the short side up keeps the aspect ratio, so the texture statistics the mode exists for
    are stretched rather than distorted.

    """
    if picture.width >= width and picture.height >= height:
        return picture
    from PIL import Image

    # The larger of the two per-axis factors; on a square canvas this is ``side / min(picture sides)``.
    scale = max(width / picture.width, height / picture.height)
    size = (max(width, round(picture.width * scale)), max(height, round(picture.height * scale)))
    return picture.resize(size, Image.Resampling.BICUBIC)
