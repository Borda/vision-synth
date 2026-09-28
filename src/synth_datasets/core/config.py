"""Configuration primitives for synthetic dataset generation.

Defines the task/format/class-mode enums, the class vocabulary derived from shapes and colors, the
split ratios, and the :class:`SyntheticConfig` knob bundle consumed by
:class:`~synth_datasets.core.generator.SyntheticGenerator`.

Which shape families exist is **not** decided here — that is
:mod:`~synth_datasets.families`, which this module reads. Everything below is about how a
run is configured and how its classes are numbered, never about what a duck looks like.

No torch and no Pillow: this module imports with only the base dependencies installed. It does pull
in the shape families transitively, which parse the packaged assets at import time (NumPy plus the
stdlib XML parser).

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from numbers import Integral
from typing import TYPE_CHECKING

from synth_datasets.families import (
    DEFAULT_SHAPES,
    Shape,
    describe_keypoint_mismatch,
    keypoint_schema_for,
    resolve_shape,
)

if TYPE_CHECKING:
    from synth_datasets.content.backgrounds import Background
    from synth_datasets.content.degradations import Degradation

_SPLIT_SUM_TOL = 1e-6

#: Bound on the per-(mode, shapes) vocabulary and id-map caches. Real programs use a handful of
#: combinations; a bound keeps a program that builds shape tuples programmatically from retaining
#: one map per distinct tuple for the life of the process.
_VOCABULARY_CACHE_SIZE = 64


class Color(str, Enum):
    """The named fill vocabulary; each member carries its own 8-bit RGB payload.

    A closed set of three, which is what :attr:`ClassMode.COLOR` numbers its classes from. A run is
    not limited to them: a fill may equally be a raw ``(r, g, b)`` triple, so a yellow object needs
    no change here. Both spellings normalize to a :class:`Fill` at the boundary, and that is the
    single type everything downstream holds; ``background`` and object fills accept the same spellings.

    The RGB triple is stored on the member itself, through ``__new__``, rather than looked up in a
    table rebuilt on every access. Members stay ordinary strings either way: ``Color.RED == "red"``
    holds, ``Color("red")`` parses, and JSON serializes a member as its value.

    Attributes:
        RED: ``(255, 0, 0)``.
        GREEN: ``(0, 128, 0)``.
        BLUE: ``(0, 0, 255)``.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import Color
        >>> Color.GREEN.rgb
        (0, 128, 0)
        >>> Color("red") is Color.RED
        True

        ```

    """

    #: The member's RGB payload, set by ``__new__``; read through :attr:`rgb`.
    _rgb: tuple[int, int, int]

    RED = "red", (255, 0, 0)
    GREEN = "green", (0, 128, 0)
    BLUE = "blue", (0, 0, 255)

    # PYI034 wants ``Self`` here, which needs ``typing_extensions`` on Python 3.10 -- an
    # undeclared dependency for one annotation. ``Color`` is exact anyway: the class is final in
    # practice, since an Enum with members cannot be subclassed.
    def __new__(cls, value: str, rgb: tuple[int, int, int]) -> Color:  # noqa: PYI034
        """Build a member that *is* its own value string and carries its RGB payload alongside."""
        color = str.__new__(cls, value)
        color._value_ = value
        color._rgb = rgb
        return color

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Return the 8-bit RGB fill tuple for this color."""
        return self._rgb


def _color_by_name(name: str) -> Color:
    """Return the :class:`Color` a case-insensitive name spells.

    Raises:
        ValueError: If ``name`` names no :class:`Color`; the message lists every valid name.

    """
    key = name.strip().lower()
    for color in Color:
        if color.value == key:
            return color
    valid = ", ".join(color.value for color in Color)
    raise ValueError(f"unknown colour name {name!r}; valid names are {valid}")


@dataclass(frozen=True)
class Fill:
    """One object fill: the RGB triple to draw with, plus the name it came from when it had one.

    The single fill type everything past the config boundary holds. Every accepted spelling — a
    :class:`Color`, its name, or a raw ``(r, g, b)`` triple — normalizes to this one type, so no
    consumer takes a union apart again for an RGB or a label.

    Args:
        rgb: The ``(r, g, b)`` triple to draw with; three integers in ``[0, 255]``.
        name: The :class:`Color` name this fill came from, or ``None`` for a raw triple. Only
            :attr:`label` reads it.

    Raises:
        ValueError: If ``rgb`` is not a tuple of three integers in ``[0, 255]``.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import Color, Fill
        >>> Fill.parse(Color.BLUE).rgb
        (0, 0, 255)
        >>> Fill.parse((255, 215, 0)).label
        'ffd700'

        ```

    """

    rgb: tuple[int, int, int]
    name: str | None = None

    def __post_init__(self) -> None:
        """Reject anything that is not three 8-bit channels."""
        if (
            not isinstance(self.rgb, tuple)
            or len(self.rgb) != 3
            or not all(isinstance(channel, int) and 0 <= channel <= 255 for channel in self.rgb)
        ):
            raise ValueError(f"fill must be a Color member or an (r, g, b) tuple of 0-255 integers, got {self.rgb!r}")

    @property
    def label(self) -> str:
        """Return the class-name label this fill contributes.

        A named fill labels itself; a raw triple has no name, so it is labelled by its hex value —
        ``(255, 215, 0)`` becomes ``"ffd700"``. That keeps :attr:`ClassMode.COLOR` and
        :attr:`ClassMode.SHAPE_COLOR` well defined for custom fills without inventing color names.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import Color, Fill
            >>> Fill.parse(Color.RED).label, Fill.parse((255, 215, 0)).label
            ('red', 'ffd700')

            ```

        """
        if self.name is not None:
            return self.name
        red, green, blue = self.rgb
        return f"{red:02x}{green:02x}{blue:02x}"

    @classmethod
    def parse(cls, color: ColorLike) -> Fill:
        """Normalize any accepted spelling of a fill into a :class:`Fill`.

        The one boundary where the :data:`ColorLike` union is unpacked. Every signature that takes a
        caller-supplied fill runs it through here once and holds the result.

        Args:
            color: A :class:`Color` member, its name in any case (``"red"``, ``"Blue"``), an
                ``(r, g, b)`` tuple, or an existing :class:`Fill`.

        Returns:
            The normalized fill — ``color`` itself when it already is one.

        Raises:
            ValueError: If ``color`` is a string naming no :class:`Color` (the message lists the
                valid names), or is neither a name, a :class:`Color`, nor a valid RGB triple.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import Color, Fill
            >>> Fill.parse(Color.BLUE)
            Fill(rgb=(0, 0, 255), name='blue')
            >>> Fill.parse((255, 215, 0))
            Fill(rgb=(255, 215, 0), name=None)
            >>> Fill.parse("Blue") == Fill.parse(Color.BLUE)
            True

            ```

        """
        if isinstance(color, Fill):
            return color
        if isinstance(color, Color):
            return cls(rgb=color.rgb, name=color.value)
        if isinstance(color, str):
            return cls.parse(_color_by_name(color))
        return cls(rgb=color)


#: A fill as a *caller* may spell it: a named :class:`Color` or its name, a raw 8-bit ``(r, g, b)``
#: triple, or an already-normalized :class:`Fill`. Only :meth:`Fill.parse` and the public signatures that feed
#: it accept the union — everything past that boundary holds a :class:`Fill`.
ColorLike = Color | str | tuple[int, int, int] | Fill

#: Fills drawn when :attr:`SyntheticConfig.colors` is not overridden — the full :class:`Color`
#: vocabulary, normalized, i.e. the behavior that predated the field existing.
DEFAULT_COLORS: tuple[Fill, ...] = tuple(Fill.parse(color) for color in Color)

#: Fills unlabelled clutter is drawn from when :attr:`SyntheticConfig.distractor_colors` is not
#: given. Six neutrals and secondaries deliberately outside the :class:`Color` vocabulary, so the
#: default configuration — whose ``colors`` is all three named members — still has a complement to
#: draw clutter from. Taking the complement of ``colors`` against :data:`DEFAULT_COLORS` instead
#: would be empty out of the box, which would make ``distractors=3`` refuse on a stock config.
#: The complement is compared on the RGB triple rather than on the whole :class:`Fill`, because a
#: fill also carries a name: ordinary set subtraction would leave ``Fill((96, 106, 120))`` available
#: as clutter for a user who had claimed that exact triple under a name of their own, painting
#: unlabelled pixels in a colour a labelled class owns — invisible to the vocabulary, and fatal
#: under :attr:`ClassMode.COLOR`.
DISTRACTOR_PALETTE: tuple[Fill, ...] = (
    Fill(rgb=(96, 106, 120), name="slate"),
    Fill(rgb=(198, 178, 134), name="sand"),
    Fill(rgb=(124, 90, 62), name="brown"),
    Fill(rgb=(48, 132, 132), name="teal"),
    Fill(rgb=(128, 72, 128), name="plum"),
    Fill(rgb=(112, 124, 56), name="olive"),
)


class Task(str, Enum):
    """Annotation task the dataset targets.

    Attributes:
        DETECTION: Axis-aligned bounding boxes only.
        SEGMENTATION: Bounding boxes plus filled polygon masks.
        OBB: Oriented (rotated) bounding boxes as four corner points.
        KEYPOINTS: Bounding boxes plus the named landmarks of whichever keypoint-bearing family
            the run draws from — animals (16 anatomical points), symbols (7), or letters (15 grid
            nodes). A run must stay within one such family, since a dataset declares one landmark
            schema; :func:`~synth_datasets.families.keypoint_schema_for` is what resolves
            it. Points a shape does not have (a whale's hind limbs, a grid slot a letter never
            touches) are absent rather than faked — see each family's module for the NaN-row
            contract.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import Task
        >>> Task.OBB.value
        'obb'
        >>> Task("keypoints") is Task.KEYPOINTS
        True

        ```

    """

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    OBB = "obb"
    KEYPOINTS = "keypoints"


class OutputFormat(str, Enum):
    """On-disk dataset layout.

    Attributes:
        COCO: Roboflow-style ``<split>/_annotations.coco.json`` plus images.
        YOLO: ``images/<split>`` + ``labels/<split>`` + ``data.yaml``.

    """

    COCO = "coco"
    YOLO = "yolo"


class ClassMode(str, Enum):
    """How object classes are derived from shape and color.

    Attributes:
        SHAPE: One class per shape in the run's own ``shapes`` vocabulary — the four primitives
            by default, up to all 49 across every family.
        COLOR: One class per fill in the run's own ``colors``, named by :attr:`Fill.label`.
        SHAPE_COLOR: Cartesian product of the two, named ``"<color>_<shape>"``.

    """

    SHAPE = "shape"
    COLOR = "color"
    SHAPE_COLOR = "shape_color"


@dataclass(frozen=True)
class ClassEntry:
    """One class in a run's vocabulary, with the shape and color it was derived from kept intact.

    The point of this type is that the writers never have to reconstruct structure from a class
    *name*. A ``ClassMode.SHAPE_COLOR`` name is built by concatenation (``"red_duck"``), and the
    writers used to recover the shape half by splitting on the first underscore — which is correct
    only for as long as no shape value and no color value contains an underscore. Carrying the pair
    through instead makes that whole class of bug impossible.

    Args:
        index: The class id — this entry's position in its vocabulary.
        name: The rendered class name, as it appears in a COCO ``categories`` block or a YOLO
            ``names`` list.
        shape: The shape this class was derived from, or ``None`` under
            :attr:`ClassMode.COLOR`, whose classes name no specific shape.
        color: The color this class was derived from, or ``None`` under :attr:`ClassMode.SHAPE`,
            whose classes name no specific color.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> entry = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,)).entries[0]
        >>> entry.index, entry.name, entry.color is None
        (0, 'square', True)

        ```

    """

    index: int
    name: str
    shape: Shape | None
    color: Fill | None


@dataclass(frozen=True)
class ClassVocabulary:
    """The ordered classes one run declares, and the ``(shape, color) -> id`` map over them.

    A dataset's ids are local to its own vocabulary: :class:`SyntheticConfig` narrows ``shapes``,
    and both the ``categories``/``names`` block a writer emits and the ids the generator stamps come
    from this one object. That is what keeps a written dataset internally consistent — at the cost
    of an id meaning different things in a narrowed run and a full-vocabulary one. **Compare two
    runs by class name, never by raw id.**

    Args:
        class_mode: The naming rule these entries were built under.
        entries: The classes, in id order.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, Color, class_vocabulary
        >>> from synth_datasets.families.animals import AnimalShape
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (AnimalShape.DUCK, AnimalShape.CAMEL))
        >>> vocab.names
        ['duck', 'camel']
        >>> vocab.id_of(AnimalShape.CAMEL, Color.RED)
        1

        ```

    """

    class_mode: ClassMode
    entries: tuple[ClassEntry, ...]

    @property
    def names(self) -> list[str]:
        """Return the class names in id order — the list a writer declares verbatim."""
        return [entry.name for entry in self.entries]

    def id_of(self, shape: Shape, color: ColorLike) -> int:
        """Return the class id for a (shape, color) pair under this vocabulary's naming rule.

        Args:
            shape: The :data:`~synth_datasets.families.Shape` member drawn.
            color: The fill drawn — a :class:`Color` member or a raw ``(r, g, b)`` triple.

        Returns:
            The zero-based class id, indexing into :attr:`entries`.

        Raises:
            KeyError: If the pair names no class in this vocabulary — under a shape-naming mode,
                that means ``shape`` was not among the shapes the vocabulary was built from.

        """
        return _id_map(self)[_class_key(shape, Fill.parse(color), self.class_mode)]


@lru_cache(maxsize=_VOCABULARY_CACHE_SIZE)
def _id_map(vocabulary: ClassVocabulary) -> dict[str, int]:
    """Return the cached ``class name -> id`` map for one vocabulary.

    :meth:`ClassVocabulary.id_of` runs once per placed object, so scanning the entries linearly there would cost a pass
    per object. Bounded rather than unbounded: a program that builds vocabularies programmatically would otherwise
    retain one map per distinct shape tuple forever. Module-private and never handed out — callers must not mutate it.

    """
    return {entry.name: entry.index for entry in vocabulary.entries}


def class_vocabulary(
    class_mode: ClassMode, shapes: Iterable[Shape], colors: Iterable[ColorLike] = DEFAULT_COLORS
) -> ClassVocabulary:
    """Build the ordered class vocabulary for a class mode over a specific shape vocabulary.

    ``shapes`` is required rather than defaulted. It used to default to the full 49-shape
    vocabulary while :class:`SyntheticConfig` defaulted to the four primitives, so the obvious call
    for a default config returned a vocabulary twelve times too large — and silently, since the ids
    still resolved. Requiring the argument removes the mismatch instead of documenting it; pass
    :data:`~synth_datasets.families.ALL_SHAPES` for the full vocabulary.

    Args:
        class_mode: The selected :class:`ClassMode`.
        shapes: The shapes a run draws from, in the order they should be numbered.
        colors: The fills a run draws from, in the order they are numbered. Defaults to the
            three named :class:`Color` members. Only the color-naming modes read it.
            :attr:`ClassMode.COLOR` ignores them, since its classes never depend on the shape
            vocabulary.
        colors: The fills a run draws from, in the order they are numbered. Defaults to the
            three named :class:`Color` members. Only the color-naming modes read it.

    Returns:
        The :class:`ClassVocabulary` for that combination.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, class_vocabulary
        >>> from synth_datasets.families import ALL_SHAPES, DEFAULT_SHAPES
        >>> class_vocabulary(ClassMode.SHAPE, DEFAULT_SHAPES).names
        ['square', 'rectangle', 'triangle', 'circle']
        >>> len(class_vocabulary(ClassMode.SHAPE, ALL_SHAPES).entries)
        49
        >>> class_vocabulary(ClassMode.COLOR, DEFAULT_SHAPES).names
        ['red', 'green', 'blue']
        >>> class_vocabulary(ClassMode.SHAPE_COLOR, DEFAULT_SHAPES).names[:2]
        ['red_square', 'green_square']

        ```

    """
    # ``ClassMode`` is a str-Enum, so a bare ``"shape"`` compares and hashes equal to the member yet
    # fails every ``is`` test below -- silently selecting the wrong naming rather than raising.
    return _build_vocabulary(ClassMode(class_mode), tuple(shapes), tuple(Fill.parse(color) for color in colors))


@lru_cache(maxsize=_VOCABULARY_CACHE_SIZE)
def _build_vocabulary(class_mode: ClassMode, shapes: tuple[Shape, ...], colors: tuple[Fill, ...]) -> ClassVocabulary:
    """Return the vocabulary for an already-normalized mode and shape tuple.

    Split from :func:`class_vocabulary` purely so the result can be cached on a hashable key:
    :meth:`ClassVocabulary.id_of` runs once per placed object, so rebuilding the entry list per call would cost a full
    pass over the vocabulary for every shape drawn.

    """
    if class_mode is ClassMode.COLOR:
        pairs: list[tuple[Shape | None, Fill | None]] = [(None, color) for color in colors]
    elif class_mode is ClassMode.SHAPE:
        pairs = [(shape, None) for shape in shapes]
    else:
        pairs = [(shape, color) for shape in shapes for color in colors]
    entries = tuple(
        ClassEntry(index=index, name=_render_name(shape, color, class_mode), shape=shape, color=color)
        for index, (shape, color) in enumerate(pairs)
    )
    return ClassVocabulary(class_mode=class_mode, entries=entries)


def class_names(
    class_mode: ClassMode, shapes: Iterable[Shape], colors: Iterable[ColorLike] = DEFAULT_COLORS
) -> list[str]:
    """Return the ordered class-name vocabulary for a class mode and shape vocabulary.

    Thin convenience over :func:`class_vocabulary` for callers that only want the names — the list
    index is the class id, for the same ``shapes``.

    Args:
        class_mode: The selected :class:`ClassMode`.
        shapes: The shapes a run draws from, in the order they should be numbered.
        colors: The fills a run draws from, in the order they are numbered. Defaults to the
            three named :class:`Color` members. Only the color-naming modes read it.

    Returns:
        Class names in id order.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, class_names
        >>> from synth_datasets.families import ALL_SHAPES, DEFAULT_SHAPES
        >>> class_names(ClassMode.SHAPE, DEFAULT_SHAPES)
        ['square', 'rectangle', 'triangle', 'circle']
        >>> class_names(ClassMode.SHAPE, ALL_SHAPES)[4:8]
        ['duck', 'elephant', 'giraffe', 'fish']
        >>> class_names(ClassMode.SHAPE, ALL_SHAPES)[23:27]
        ['a', 'b', 'c', 'd']
        >>> len(class_names(ClassMode.SHAPE, ALL_SHAPES))
        49

        ```

    """
    return class_vocabulary(class_mode, shapes, colors).names


def class_id(
    shape: Shape,
    color: ColorLike,
    class_mode: ClassMode,
    shapes: Iterable[Shape],
    colors: Iterable[ColorLike] = DEFAULT_COLORS,
) -> int:
    """Return the class id for a (shape, color) pair under a class mode and shape vocabulary.

    Args:
        shape: The :data:`~synth_datasets.families.Shape` member.
        color: The :class:`Color` member.
        class_mode: The selected :class:`ClassMode`.
        shapes: The shape vocabulary to index into, in order.
        colors: The fills a run draws from, in the order they are numbered. Defaults to the
            three named :class:`Color` members. Only the color-naming modes read it.

    Returns:
        Zero-based class id indexing into ``class_names(class_mode, shapes)``.

    Raises:
        KeyError: If ``shape`` is not in ``shapes`` (under a shape-naming mode).

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, Color, class_id
        >>> from synth_datasets.families import ALL_SHAPES
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.families.symbols import SymbolShape
        >>> class_id(PrimitiveShape.TRIANGLE, Color.RED, ClassMode.SHAPE, ALL_SHAPES)
        2
        >>> class_id(PrimitiveShape.TRIANGLE, Color.RED, ClassMode.COLOR, ALL_SHAPES)
        0
        >>> class_id(SymbolShape.KITE, Color.RED, ClassMode.SHAPE, ALL_SHAPES)
        16
        >>> class_id(SymbolShape.KITE, Color.RED, ClassMode.SHAPE, (SymbolShape.KITE,))
        0

        ```

    """
    return class_vocabulary(class_mode, shapes, colors).id_of(shape, color)


def _render_name(shape: Shape | None, color: Fill | None, class_mode: ClassMode) -> str:
    """Return the displayed class name for one (shape, color) pair under a class mode.

    Exactly one of ``shape`` and ``color`` may be ``None``, and only for the mode that ignores it:
    :attr:`ClassMode.SHAPE` names no color, :attr:`ClassMode.COLOR` names no shape.

    """
    if class_mode is ClassMode.SHAPE:
        return str(shape.value) if shape is not None else ""
    if class_mode is ClassMode.COLOR:
        return color.label if color is not None else ""
    return f"{color.label}_{shape.value}" if shape is not None and color is not None else ""


def _class_key(shape: Shape, color: Fill, class_mode: ClassMode) -> str:
    """Return the lookup key for a drawn (shape, color) pair — the name its class was rendered under."""
    return _render_name(shape, color, class_mode)


def _resolve_shape_names(values: Iterable[object] | str) -> tuple[object, ...]:
    """Return ``values`` as a tuple with every shape name replaced by its member; see ``SyntheticConfig``."""
    items = (values,) if isinstance(values, str) else tuple(values)
    return tuple(resolve_shape(item) if isinstance(item, str) else item for item in items)


#: Characters that would let a split name reach outside ``output_dir`` or change meaning across platforms: both path
#: separators (a config file must mean the same on POSIX and Windows), the Windows drive separator, and NUL.
_UNSAFE_SPLIT_CHARS = frozenset("/\\:\x00")

#: Characters Windows forbids in a file name beyond :data:`_UNSAFE_SPLIT_CHARS`: ``< > " | ? *`` and every control
#: character.
_WINDOWS_INVALID_CHARS = frozenset('<>"|?*' + "".join(chr(code) for code in range(1, 32)))

#: Prefix the writers keep for their own entries in ``output_dir`` — the ``.vision-synth.json`` manifest and the
#: ``.vision-synth-staging-*`` directory a replacing write stages in — so no split may start with it.
_RESERVED_SPLIT_PREFIX = ".vision-synth"

#: Device names Windows reserves in every directory, whatever the case and whatever follows the first dot
#: (``aux.txt`` and ``nul.tar.gz`` are reserved too): a split directory named after one cannot be created there.
_WINDOWS_RESERVED_NAMES = frozenset({
    "con",
    "prn",
    "aux",
    "nul",
    *(f"{device}{digit}" for device in ("com", "lpt") for digit in "123456789\u00b9\u00b2\u00b3"),
})


def as_canvas_size(img_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return a canvas size as ``(width, height)``, validating it.

    The one place the two spellings of :attr:`SyntheticConfig.img_size` are unpacked: a plain ``int`` is a square
    side, a pair is ``(width, height)`` — Pillow's order, not numpy's ``(rows, columns)``. Every built-in
    :class:`~synth_datasets.content.backgrounds.Background` normalizes its ``img_size`` argument through here.

    Args:
        img_size: A positive ``int``, or a pair of positive ``int`` values ``(width, height)``.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: If ``img_size`` is not a positive ``int`` or a pair of them. ``bool`` is refused though it
            subclasses ``int``: ``img_size=True`` is a mistake, not a one-pixel canvas.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import as_canvas_size
        >>> as_canvas_size(64), as_canvas_size((96, 48))
        ((64, 64), (96, 48))

        ```

    """
    if _is_int(img_size):
        values: tuple[object, ...] = (img_size, img_size)
    elif isinstance(img_size, (tuple, list)):
        values = tuple(img_size)
        if len(values) != 2:
            raise ValueError(f"img_size as a sequence must hold two values (width, height), got {img_size!r}")
    else:
        raise ValueError(f"img_size must be an int or a (width, height) pair of ints, got {img_size!r}")
    if not all(_is_int(value) for value in values):
        raise ValueError(f"img_size must be an int or a (width, height) pair of ints, got {img_size!r}")
    width, height = (int(value) for value in values)  # type: ignore[call-overload]
    if width <= 0 or height <= 0:
        raise ValueError(f"img_size must be positive, got {img_size!r}")
    return width, height


def _is_int(value: object) -> bool:
    """Return whether ``value`` is an integer and not a ``bool`` (numpy integers included)."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate_split_name(name: object) -> str:
    r"""Return ``name`` unchanged if it can name a split directory, i.e. is one plain path component.

    Every writer turns a split name into a directory under ``output_dir``, so a name such as ``"../escaped"`` used to
    create ``escaped/`` beside the output directory rather than inside it. :class:`SplitRatios` checks its names here
    at construction, and the built-in writers check again before writing, for a caller who builds ``splits`` by hand.

    Args:
        name: The candidate split name.

    Returns:
        ``name``, known to be a non-empty string that is not ``.`` or ``..``, holds no ``/``, ``\\``, ``:`` or NUL, does
        not end in a dot or a space, holds none of ``< > " | ? *`` or a control character, is not a Windows device
        name (``CON``, ``NUL``, ``aux.txt``, ``COM1``, ``LPT¹``…), and does not start with ``.vision-synth``.

    Raises:
        ValueError: If ``name`` is not such a string.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import validate_split_name
        >>> validate_split_name("holdout")
        'holdout'
        >>> validate_split_name("../escaped")
        Traceback (most recent call last):
        ...
        ValueError: split name '../escaped' is not one plain path component; ...

        ```

    """
    if not isinstance(name, str):
        raise ValueError(f"split name must be a string, got {type(name).__name__} {name!r}")
    if name in ("", ".", "..") or not _UNSAFE_SPLIT_CHARS.isdisjoint(name):
        raise ValueError(
            f"split name {name!r} is not one plain path component; each split is written to its own directory under "
            "output_dir, so use a name such as 'train' or 'holdout' (not empty, '.', '..', or holding '/', '\\', ':')"
        )
    if (
        name.endswith((".", " "))
        or not _WINDOWS_INVALID_CHARS.isdisjoint(name)
        or name.partition(".")[0].rstrip(" ").lower() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"split name {name!r} cannot name a directory on Windows, which reserves device names such as 'CON', "
            "'NUL', 'COM1', 'LPT\u00b9' or 'aux.txt', forbids the characters < > \" | ? * and control characters, and "
            "strips a trailing dot or space; use a name such as 'train' or 'holdout'"
        )
    # casefold: on a case-insensitive filesystem '.VISION-SYNTH.JSON' is the manifest.
    if name.casefold().startswith(_RESERVED_SPLIT_PREFIX):
        raise ValueError(
            f"split name {name!r} starts with {_RESERVED_SPLIT_PREFIX!r} (in any case), which the writers keep for "
            "their manifest and their staging and backup directories; use a name such as 'train' or 'holdout'"
        )
    return name


@dataclass(frozen=True)
class SplitRatios:
    """Dataset split fractions; must be non-negative and sum to ~1.

    The three standard splits are the constructor's arguments because they are what almost every
    caller wants. They are not a *limit*: :meth:`custom` takes any names at all, so a fourth
    calibration split or a bare train/test pair needs no change here. The class used to hardcode
    exactly ``train``/``val``/``test``, which meant "train and test only" had to be spelled as
    ``val=0.0`` and a fourth split was simply impossible.

    Args:
        train: Training fraction.
        val: Validation fraction.
        test: Test fraction.

    Raises:
        ValueError: If any fraction is negative or they do not sum to 1.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import SplitRatios
        >>> SplitRatios().to_dict()
        {'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> SplitRatios(0.8, 0.2, 0.0).to_dict()
        {'train': 0.8, 'val': 0.2}
        >>> SplitRatios.custom({"train": 0.6, "calib": 0.2, "test": 0.2}).to_dict()
        {'train': 0.6, 'calib': 0.2, 'test': 0.2}

        ```

    """

    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    #: Set by :meth:`custom` to override the three named fields entirely. ``None`` (the default)
    #: means the fields above are the splits, which is what every ordinary construction wants.
    named: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate split names, non-negativity, and unit sum across whichever splits are in play."""
        for name in self.named or ():
            validate_split_name(name)
        for name, value in self._items():
            if value < 0:
                raise ValueError(f"split ratio {name!r} must be non-negative, got {value}")
        total = sum(value for _, value in self._items())
        if abs(total - 1.0) > _SPLIT_SUM_TOL:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")

    @classmethod
    def custom(cls, splits: Mapping[str, float]) -> SplitRatios:
        """Build split ratios over arbitrary split names.

        Args:
            splits: ``name -> fraction``, in the order the splits should be written. Fractions must
                be non-negative and sum to 1, exactly as for the standard three.

        Returns:
            The :class:`SplitRatios` carrying those splits.

        Raises:
            ValueError: If ``splits`` is empty, a name is not one plain path component (see
                :func:`validate_split_name`), or its fractions are negative or do not sum to 1.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import SplitRatios
            >>> SplitRatios.custom({"train": 0.9, "holdout": 0.1}).to_dict()
            {'train': 0.9, 'holdout': 0.1}

            ```

        """
        if not splits:
            raise ValueError("splits must name at least one split, got an empty mapping")
        return cls(named=dict(splits))

    def _items(self) -> tuple[tuple[str, float], ...]:
        """Return the ``(name, fraction)`` pairs in play, custom ones taking precedence."""
        if self.named is not None:
            return tuple(self.named.items())
        return (("train", self.train), ("val", self.val), ("test", self.test))

    def to_dict(self) -> dict[str, float]:
        """Return the non-zero splits as an ordered ``name -> fraction`` mapping."""
        return {name: value for name, value in self._items() if value > 0}


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs controlling one synthetic image's content.

    Args:
        img_size: Canvas size in pixels: an ``int`` for a square canvas, or a ``(width, height)``
            pair for a rectangular one. :attr:`canvas_size` reads either back as ``(width, height)``;
            images come out as ``(height, width, 3)`` arrays.
        min_objects: Minimum objects drawn per image (inclusive).
        max_objects: Maximum objects drawn per image (inclusive).
        min_size_ratio: Minimum object size as a fraction of the canvas's shorter side.
        max_size_ratio: Maximum object size as a fraction of the canvas's shorter side.
        overlap_iou: Reject a candidate whose IoU with any kept box exceeds this.
        boundary_tolerance: Max fraction of a box allowed outside the canvas.
        max_placement_attempts: Retry cap per object before giving up.
        background: What fills the canvas before any object is drawn. Either a plain fill — a
            :class:`Color` or its name (``"red"``), an ``(r, g, b)`` triple, or a :class:`Fill` — or a
            :class:`~synth_datasets.content.backgrounds.Background` such as
            :class:`~synth_datasets.content.backgrounds.NoiseBackground` or
            :class:`~synth_datasets.content.backgrounds.TextureBackground`. A plain fill is
            normalized to a :class:`~synth_datasets.content.backgrounds.SolidBackground` at
            construction, so ``config.background`` always reads back as a background object and the
            default draws a flat grey canvas. Only the three :class:`Color` names are accepted as
            strings; other Pillow colour strings such as ``"white"`` or ``"#204080"`` are rejected —
            pass an ``(r, g, b)`` triple instead. A background renders from its
            own side stream, never from the placement stream, so switching one on never moves an
            object at a fixed seed.
        rotate: Apply a random rotation to each polygonal shape.
        asymmetry_jitter: Max fraction, in ``[0, 0.5)``, by which one randomly chosen half of a
            shape — left or right of its own local vertical axis, before rotation — is narrowed,
            drawn independently per placed object. ``0.0`` (the default) disables it and leaves
            every existing seeded configuration's output unchanged. Every shape this package draws
            except :attr:`~synth_datasets.families.primitives.PrimitiveShape.CIRCLE` is mirror-symmetric
            about that axis in its canonical orientation, so its oriented bounding box would
            otherwise always show identical left/right margins; a nonzero value breaks that with
            per-instance variety instead — real oriented objects (vehicles, ships) are rarely that
            symmetric. ``circle`` is always excluded: it never rotates either, so an unrotated skew
            would bias every instance toward the same absolute image direction rather than varying
            with a random orientation. Applies to the polygon and, under :attr:`Task.KEYPOINTS`, the
            landmark table together, so a shape and its keypoints never drift apart.
        class_mode: How classes are derived (see :class:`ClassMode`).
        shapes: Shapes the generator may draw, sampled uniformly. Defaults to
            :data:`DEFAULT_SHAPES`; pass e.g. ``(AnimalShape.DUCK, AnimalShape.GIRAFFE)`` to draw
            animal silhouettes instead, ``tuple(AnimalShape)`` for every animal, ``tuple(SymbolShape)``
            for every symbol, ``tuple(LetterShape)`` for every letter, or
            ``(*PrimitiveShape, *AnimalShape, *SymbolShape, *LetterShape)`` for the full mixed vocabulary.
            A shape may also be spelled by its name — ``("duck", "giraffe")``, or a single ``"duck"`` —
            which is how a YAML file or the ``vision-synth`` command line can name one; each name is
            resolved to its member at construction (see
            :func:`~synth_datasets.families.resolve_shape`), so ``config.shapes`` always reads back
            as members. ``distractor_shapes`` accepts names the same way.
            Restricting this **does** renumber classes: the vocabulary a run declares and the ids
            it stamps both narrow to exactly these shapes, in this order (see :func:`class_names`),
            so a symbols-only run numbers its symbols from ``0`` rather than from their offset into
            the full :class:`Shape` enum. Compare runs by class name, not by raw id.
        colors: Fills the generator may draw, sampled uniformly. Each may be spelled as a
            :class:`Color` member, its name in any case (``"red"``), a raw ``(r, g, b)`` triple, or a
            :class:`Fill`; all four are
            normalized to :class:`Fill` at construction, so ``config.colors`` reads back as
            ``Fill`` objects whichever spelling went in. Defaults to :data:`DEFAULT_COLORS` (all
            three named colors); pass e.g. ``(Color.RED,)`` to draw only red objects, or
            ``((255, 215, 0),)`` for a custom yellow. Under :attr:`ClassMode.COLOR` and
            :attr:`ClassMode.SHAPE_COLOR` restricting this **does** renumber classes, exactly as
            ``shapes`` does: the color axis narrows to exactly these fills, in this order, so
            ``colors=(Color.BLUE,)`` declares one color class, ``blue``, with id ``0``. Under
            :attr:`ClassMode.SHAPE` there is no color axis, so the ids never depend on this.
        task: Annotation task the generated samples target, as a :class:`Task` or its string value
            (``"detection"``, ``"segmentation"``, ``"obb"``, ``"keypoints"``). This is the **only**
            place a run's task is set — :func:`~synth_datasets.generate_dataset` reads it
            from here rather than taking its own argument, so the generator and the writer can never
            disagree about it. Only :attr:`Task.KEYPOINTS` changes what the generator computes (it
            adds the landmark table); the other tasks all read the same polygon/box fields, so they
            differ at write time only.

    Raises:
        ValueError: On non-positive sizes, inverted min/max ranges, an ``overlap_iou`` or
            ``boundary_tolerance`` outside ``[0, 1]``, ``max_placement_attempts`` below 1, an
            ``asymmetry_jitter`` outside ``[0, 0.5)``,
            a ``shapes`` tuple that is empty, names an unknown shape (the message lists every valid
            name), or holds an element that is neither a :class:`Shape` nor a name, a ``colors``
            tuple that is empty or holds an element that is no valid fill, a ``task`` naming no
            :class:`Task`, or a :attr:`Task.KEYPOINTS` task combined with a ``shapes`` tuple
            that does not belong entirely to one keypoint-bearing family (see
            :func:`keypoint_schema_for`) — a :class:`~synth_datasets.families.primitives.PrimitiveShape`
            mixed in, or two of :class:`~synth_datasets.families.animals.AnimalShape`,
            :class:`~synth_datasets.families.symbols.SymbolShape`, and
            :class:`~synth_datasets.families.letters.LetterShape` mixed together, since only one
            landmark schema can describe a dataset.

    Examples:
        ```pycon
        >>> from synth_datasets.families.animals import AnimalShape
        >>> from synth_datasets.core.config import ClassMode, Color, SyntheticConfig, Task, class_names
        >>> SyntheticConfig(img_size=128).img_size
        128
        >>> SyntheticConfig(shapes=(AnimalShape.DUCK, AnimalShape.CAMEL)).shapes
        (<AnimalShape.DUCK: 'duck'>, <AnimalShape.CAMEL: 'camel'>)
        >>> SyntheticConfig(task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK,)).task
        <Task.KEYPOINTS: 'keypoints'>
        >>> SyntheticConfig(colors=(Color.RED,)).colors
        (Fill(rgb=(255, 0, 0), name='red'),)
        >>> SyntheticConfig(colors=((255, 215, 0),)).colors[0].label
        'ffd700'
        >>> blue_only = SyntheticConfig(class_mode=ClassMode.COLOR, colors=(Color.BLUE,))
        >>> class_names(blue_only.class_mode, blue_only.shapes, blue_only.colors)
        ['blue']
        >>> SyntheticConfig(shapes=("duck", "camel")).shapes
        (<AnimalShape.DUCK: 'duck'>, <AnimalShape.CAMEL: 'camel'>)

        ```

    """

    img_size: int | tuple[int, int] = 640
    min_objects: int = 1
    max_objects: int = 10
    min_size_ratio: float = 0.1
    max_size_ratio: float = 0.3
    overlap_iou: float = 0.1
    boundary_tolerance: float = 0.05
    max_placement_attempts: int = 100
    background: ColorLike | Background = (128, 128, 128)
    degrade: tuple[Degradation, ...] = ()
    distractors: int = 0
    occluders: int = 0
    distractor_shapes: tuple[Shape, ...] | None = None
    distractor_colors: tuple[Fill, ...] | None = None
    rotate: bool = True
    asymmetry_jitter: float = 0.0
    class_mode: ClassMode = ClassMode.SHAPE
    shapes: tuple[Shape, ...] = DEFAULT_SHAPES
    colors: tuple[Fill, ...] = DEFAULT_COLORS
    task: Task = Task.DETECTION

    def __post_init__(self) -> None:
        """Normalize the scalar enums, then validate the numeric knobs and the vocabulary."""
        # ``SyntheticIterableDataset`` forwards ``**config_kwargs`` verbatim, so a documented
        # ``class_mode="shape"`` arrives here as a bare string. ``ClassMode`` is a str-Enum, so that
        # string compares *and hashes* equal to the member while failing every ``is`` identity test
        # the module uses to branch on it -- the silent half of the trap `_validate_vocabulary`
        # rejects outright for `shapes`/`colors`/`task`. Rejecting is not an option here (the string
        # form is public API), so coerce once, at the only boundary that sees the raw value.
        object.__setattr__(self, "class_mode", ClassMode(self.class_mode))
        # ``task`` needs the same treatment for the same reason, and now more than ever: it used to
        # be normalized by ``generate_dataset`` before ever reaching here, so this class could
        # afford to reject a bare string. With the config the task's sole owner, ``task="keypoints"``
        # arrives here raw and is the documented spelling -- coerce it rather than reject it.
        object.__setattr__(self, "task", Task(self.task))
        self._normalize_shapes()
        self._normalize_colors()
        self._normalize_background()
        self._validate_degradations()
        self._normalize_distractor_pools()
        as_canvas_size(self.img_size)
        if isinstance(self.img_size, list):
            # A YAML file spells a pair as a list; store the tuple so the frozen config stays hashable.
            object.__setattr__(self, "img_size", tuple(self.img_size))
        if not 1 <= self.min_objects <= self.max_objects:
            raise ValueError(f"require 1 <= min_objects <= max_objects, got {self.min_objects}, {self.max_objects}")
        if not 0 < self.min_size_ratio <= self.max_size_ratio <= 1:
            raise ValueError(
                f"require 0 < min_size_ratio <= max_size_ratio <= 1, got {self.min_size_ratio}, {self.max_size_ratio}"
            )
        if not 0 <= self.overlap_iou <= 1:
            raise ValueError(f"overlap_iou must be within [0, 1], got {self.overlap_iou}")
        if not 0 <= self.boundary_tolerance <= 1:
            raise ValueError(f"boundary_tolerance must be within [0, 1], got {self.boundary_tolerance}")
        if self.max_placement_attempts < 1:
            raise ValueError(f"max_placement_attempts must be >= 1, got {self.max_placement_attempts}")
        if self.distractors < 0:
            raise ValueError(f"distractors must be non-negative, got {self.distractors}")
        if self.occluders < 0:
            raise ValueError(f"occluders must be non-negative, got {self.occluders}")
        if not 0.0 <= self.asymmetry_jitter < 0.5:
            raise ValueError(f"asymmetry_jitter must be within [0, 0.5), got {self.asymmetry_jitter}")
        self._validate_vocabulary()

    def _normalize_shapes(self) -> None:
        """Resolve shape names in :attr:`shapes` and :attr:`distractor_shapes` to their enum members.

        A YAML file or a command line can only spell a shape as a string, so a name is public API here the way
        ``task="keypoints"`` is. It is resolved once, at this boundary, rather than compared as a string later:
        ``Shape`` is a str-Enum, so ``"duck"`` compares equal to ``AnimalShape.DUCK`` while failing every identity
        and ``type(shape)`` lookup downstream. A lone string — a name, or a bare member, which is a string too — is
        one shape rather than a sequence to iterate, since iterating ``"duck"`` would yield the letters ``d u c k``.
        Anything that is neither a member nor a name is left for :meth:`_validate_vocabulary` to reject.

        Raises:
            ValueError: If a string names no shape; the message lists every valid name.

        """
        object.__setattr__(self, "shapes", _resolve_shape_names(self.shapes))
        if self.distractor_shapes is not None:
            object.__setattr__(self, "distractor_shapes", _resolve_shape_names(self.distractor_shapes))

    def _normalize_colors(self) -> None:
        """Replace :attr:`colors` with the normalized :class:`Fill` tuple it stands for.

        Same reasoning as the ``class_mode`` and ``task`` coercions above, applied to a sequence:
        a fill is public API in several spellings, so the union is unpacked once here rather than at
        every point that later needs an RGB triple or a class-name label. A colour name such as
        ``"red"`` resolves to its :class:`Color` member here, so no bare string reaches the
        generator, where under the :class:`str` mixin it would compare equal to the member while
        failing every identity test. A lone string — a name, or a bare member — is one colour
        rather than a sequence to iterate, since iterating ``"red"`` would yield ``r e d``.

        Raises:
            ValueError: If ``colors`` is empty, names an unknown colour, or holds anything that is
                not a :class:`Color`, a colour name, a :class:`Fill`, or a valid ``(r, g, b)`` triple.

        """
        colors = (self.colors,) if isinstance(self.colors, str) else self.colors
        if not colors:
            raise ValueError("colors must name at least one Color, got an empty sequence")
        object.__setattr__(self, "colors", tuple(Fill.parse(value) for value in colors))

    def _normalize_background(self) -> None:
        """Replace :attr:`background` with the :class:`Background` it stands for.

        Same boundary-normalization reasoning as :meth:`_normalize_colors`, applied to the canvas:
        the field accepts a bare fill *or* a background object, and everything past construction
        holds a background. A bare fill becomes a
        :class:`~synth_datasets.content.backgrounds.SolidBackground`, which draws the same flat
        canvas as the bare fill, pixel for pixel.

        The import is deferred rather than made at module scope because
        :mod:`~synth_datasets.content.backgrounds` imports this module for its fill union; the cycle
        is real and broken here, at the one runtime point that needs the concrete class.

        Raises:
            ValueError: If ``background`` is neither a :class:`Background` nor a valid fill.

        """
        from synth_datasets.content.backgrounds import Background, SolidBackground

        if isinstance(self.background, Background):
            return
        object.__setattr__(self, "background", SolidBackground(color=Fill.parse(self.background)))

    def _validate_degradations(self) -> None:
        """Reject a ``degrade`` tuple holding anything that is not a :class:`Degradation`.

        The effects are applied in order to the finished image, so a stray tuple element would
        surface as an attribute error deep inside rendering rather than at the point the bad tuple
        was written. The import is deferred for the same cycle-breaking reason as
        :meth:`_normalize_background`'s.

        Raises:
            ValueError: If any element is not a
                :class:`~synth_datasets.content.degradations.Degradation`.

        """
        from synth_datasets.content.degradations import Degradation

        object.__setattr__(self, "degrade", tuple(self.degrade))
        invalid = [step for step in self.degrade if not isinstance(step, Degradation)]
        if invalid:
            raise ValueError(f"degrade must contain only Degradation instances, got {invalid!r}")

    def _normalize_distractor_pools(self) -> None:
        """Normalize an explicitly given distractor pool, and refuse a clutter request nothing can fill.

        Only what the caller actually passed is normalized. The complements stay *derived*, on
        :attr:`resolved_distractor_shapes` and :attr:`resolved_distractor_colors`, rather than being
        written back over the fields — writing them back made the defaults survive a
        :func:`dataclasses.replace` that changed what they were derived from, so
        ``replace(config, shapes=(AnimalShape.DUCK,))`` kept a pool still containing ``DUCK`` and
        clutter was drawn as unlabelled duplicates of a real class, with no error anywhere. That
        idiom is live in this repository, so the failure was reachable rather than theoretical.

        Raises:
            ValueError: If clutter was asked for and either pool resolves to nothing, naming which of
                the two was empty.

        """
        if self.distractor_shapes is not None:
            object.__setattr__(self, "distractor_shapes", tuple(self.distractor_shapes))
        if self.distractor_colors is not None:
            object.__setattr__(self, "distractor_colors", tuple(Fill.parse(f) for f in self.distractor_colors))
        if not (self.distractors or self.occluders):
            return
        pools = (
            ("distractor_shapes", self.resolved_distractor_shapes),
            ("distractor_colors", self.resolved_distractor_colors),
        )
        for name, pool in pools:
            if not pool:
                raise ValueError(
                    f"clutter was asked for (distractors={self.distractors}, occluders={self.occluders}) "
                    f"but {name} resolved to an empty pool; "
                    f"pass {name}= explicitly, or narrow shapes/colors so a complement remains"
                )

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Return the canvas as ``(width, height)`` in pixels, whichever way :attr:`img_size` spells it.

        Examples:
            ```pycon
            >>> from synth_datasets import SyntheticConfig
            >>> SyntheticConfig(img_size=64).canvas_size, SyntheticConfig(img_size=(96, 48)).canvas_size
            ((64, 64), (96, 48))

            ```

        """
        return as_canvas_size(self.img_size)

    @property
    def resolved_background(self) -> Background:
        """Return the canvas filler, typed as the :class:`Background` the field always holds.

        :attr:`background` declares the union its *constructor* accepts, because a dataclass builds
        ``__init__`` from the annotation. Everything past ``__post_init__`` holds a background, but
        a type checker reading the annotation cannot know that, and rejects
        ``config.background.render(...)`` on the three fill members of the union. This property is
        where that invariant is stated once, so no caller has to assert or cast it.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import SyntheticConfig
            >>> type(SyntheticConfig(img_size=32, background=(10, 20, 30)).resolved_background).__name__
            'SolidBackground'

            ```

        Raises:
            TypeError: If the field somehow holds a fill, which only a write that bypasses
                ``__post_init__`` can produce.

        """
        from synth_datasets.content.backgrounds import Background

        if not isinstance(self.background, Background):  # pragma: no cover - __post_init__ normalizes it
            raise TypeError(f"background holds {type(self.background).__name__}, not a Background")
        return self.background

    @property
    def resolved_distractor_shapes(self) -> tuple[Shape, ...]:
        """Return the shapes clutter is actually drawn from.

        :attr:`distractor_shapes` when it was given, otherwise the complement of :attr:`shapes`
        against :data:`~synth_datasets.families.ALL_SHAPES`, so clutter never wears the
        silhouette of a class. Derived on read rather than stored, which is what keeps it correct
        after a :func:`dataclasses.replace` that changed ``shapes``.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import SyntheticConfig
            >>> from synth_datasets.families.primitives import PrimitiveShape
            >>> config = SyntheticConfig(img_size=32, shapes=(PrimitiveShape.SQUARE,))
            >>> PrimitiveShape.SQUARE in config.resolved_distractor_shapes
            False

            ```

        """
        from synth_datasets.families import ALL_SHAPES

        if self.distractor_shapes is not None:
            return self.distractor_shapes
        return tuple(shape for shape in ALL_SHAPES if shape not in self.shapes)

    @property
    def resolved_distractor_colors(self) -> tuple[Fill, ...]:
        """Return the fills clutter is actually drawn from.

        :attr:`distractor_colors` when it was given, otherwise :data:`DISTRACTOR_PALETTE` minus any
        entry whose RGB triple :attr:`colors` already claims. Compared on the triple rather than on
        the whole :class:`Fill`, since a fill also carries a name and a user may claim a palette
        colour under one of their own. Derived on read, for the same reason as
        :attr:`resolved_distractor_shapes`.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import DISTRACTOR_PALETTE, Fill, SyntheticConfig
            >>> config = SyntheticConfig(img_size=32, colors=(Fill(rgb=DISTRACTOR_PALETTE[0].rgb),))
            >>> DISTRACTOR_PALETTE[0] in config.resolved_distractor_colors
            False

            ```

        """
        if self.distractor_colors is not None:
            return self.distractor_colors
        claimed = {fill.rgb for fill in self.colors}
        return tuple(fill for fill in DISTRACTOR_PALETTE if fill.rgb not in claimed)

    def _validate_vocabulary(self) -> None:
        """Reject an unusable shape/color tuple, a non-:class:`Task` task, or an unannotatable pairing.

        Split out of :meth:`__post_init__` so neither routine outgrows the project's complexity
        budget; it carries every check that reads the enum-valued fields rather than the numbers.

        Raises:
            ValueError: If ``shapes`` is empty or holds a non-:class:`Shape` element, ``colors`` is
                empty or holds a non-:class:`Color` element, ``task`` is not a :class:`Task` member,
                or a :attr:`Task.KEYPOINTS` task is paired with a ``shapes`` tuple that does not
                belong entirely to one keypoint-bearing family.

        """
        if not self.shapes:
            raise ValueError("shapes must name at least one Shape, got an empty sequence")
        # ``Shape`` is a str-Enum, so a bare "duck" compares equal to AnimalShape.DUCK yet is not an
        # instance -- reject it here rather than let it surface as an opaque lookup failure.
        invalid = [value for value in self.shapes if not isinstance(value, Shape)]
        if invalid:
            raise ValueError(f"shapes must contain only Shape members, got {invalid!r}")
        if self.task is Task.KEYPOINTS and keypoint_schema_for(self.shapes) is None:
            # ``keypoint_schema_for`` collapses "no landmark table at all", "two families mixed",
            # and "empty" into one ``None``; the registry knows which it was, so the reason is asked
            # for rather than re-derived here.
            reason = describe_keypoint_mismatch(self.shapes)
            raise ValueError(f"task {Task.KEYPOINTS.value!r} needs one keypoint schema, but {reason}")
