"""The shape-family registry: the one place that knows which shape families exist.

Every family — :mod:`~synth_datasets.families.primitives`,
:mod:`~synth_datasets.families.animals`, :mod:`~synth_datasets.families.symbols`,
:mod:`~synth_datasets.families.letters` — contributes exactly one :class:`ShapeFamily` entry to
:data:`SHAPE_FAMILIES`, and every other module in the package consults that tuple instead of
naming the families itself.

Adding a family means exactly one edit: write the module and append one :class:`ShapeFamily`
here. :data:`Shape` needs no second edit: it is the shared base class
:class:`~synth_datasets.families.shape_enum.ShapeEnum`, which a type checker reads directly off
each family's own declaration.

Examples:
    ```pycon
    >>> from synth_datasets.families import SHAPE_FAMILIES, family_of, shape_outline
    >>> [family.name for family in SHAPE_FAMILIES]
    ['primitives', 'animals', 'symbols', 'letters']
    >>> from synth_datasets.families.animals import AnimalShape
    >>> family_of(AnimalShape.DUCK).name
    'animals'
    >>> shape_outline("square", center=(0.0, 0.0), size=2.0).shape
    (4, 2)

    ```

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from synth_datasets.families.animals import ANIMAL_KEYPOINT_SCHEMA, ANIMAL_POLYGONS, AnimalShape, animal_keypoints
from synth_datasets.families.geometry import place_points
from synth_datasets.families.letters import LETTER_KEYPOINT_SCHEMA, LETTER_POLYGONS, LetterShape, letter_keypoints
from synth_datasets.families.primitives import PrimitiveShape, primitive_outline
from synth_datasets.families.shape_enum import ShapeEnum
from synth_datasets.families.symbols import SYMBOL_KEYPOINT_SCHEMA, SYMBOL_POLYGONS, SymbolShape, symbol_keypoints

if TYPE_CHECKING:
    from collections.abc import Iterable
    from enum import Enum

    from numpy.typing import NDArray

    from synth_datasets.core.keypoints import KeypointSchema

#: Any drawable shape, as a static type *and* as a runtime check: every family's enum derives from
#: :class:`~synth_datasets.families.shape_enum.ShapeEnum`, so ``isinstance(value, Shape)`` accepts
#: any member type and still tells a bare ``"duck"`` string apart — which is how :func:`resolve_shape`
#: knows a name needs resolving and a member does not. This was
#: a hand-written ``PrimitiveShape | AnimalShape | ...`` union until the base class replaced it,
#: which is what reduced adding a family to a single edit site. It is not iterable — use
#: :data:`ALL_SHAPES` for the full vocabulary.
Shape = ShapeEnum


class PlaceKeypoints(Protocol):
    """The signature every family's landmark placer shares.

    ``shape`` is :class:`~typing.Any` rather than :data:`Shape` on purpose: each family's placer accepts only its *own*
    member type, and a callable taking a narrower parameter is not assignable to one declared over the whole union. The
    registry only ever calls a placer with a shape of its own family — :func:`place_keypoints` looks the family up by
    ``type(shape)`` — so the guarantee holds at the call site rather than in the annotation.

    """

    def __call__(
        self,
        shape: Any,  # noqa: ANN401 - see the class docstring: each family's placer accepts only its own
        # member type, and a callable over the narrower type is not assignable to one declared over the union
        center: tuple[float, float],
        size: float,
        angle: float,
        skew: float,
    ) -> NDArray[np.float64]:
        """Return the placed ``(num_keypoints, 2)`` landmark table for one shape."""
        ...


def _table_outline(table: Mapping[str, NDArray[np.float64]]) -> Callable[[str, float], NDArray[np.float64]]:
    """Return an outline accessor reading one asset-backed family's unit-space polygon table.

    The stored table is frozen (read-only), so multiplying by ``size`` returns a fresh writable
    array and never aliases the packaged constant.

    Args:
        table: The family's ``shape value -> unit-space outline`` mapping.

    Returns:
        A ``(value, size) -> outline`` callable matching :attr:`ShapeFamily.base_outline`, raising
        :class:`KeyError` for a value the family does not own — callers reach it through
        :func:`shape_outline`, which checks family membership first.

    """
    return lambda value, size: table[value] * size


@dataclass(frozen=True)
class ShapeFamily:
    """One shape family's contribution to the drawable vocabulary.

    Args:
        name: The family's short name, used in error messages and diagnostics (``"animals"``).
        members: Every member of the family's enum, in declaration order — which is also the order
            :func:`~synth_datasets.core.config.class_names` numbers them in.
        base_outline: Returns the origin-centered ``(num_points, 2)`` outline for one member
            *value* at a given size. Analytic for :mod:`~synth_datasets.families.primitives`, a
            table lookup for every asset-backed family; the two are interchangeable here precisely
            because both share the unit convention (area centroid at the origin, larger extent
            equal to ``size``).
        keypoint_schema: The family's landmark vocabulary, or ``None`` for a family whose members
            carry no landmarks (:class:`~synth_datasets.families.primitives.PrimitiveShape`). A
            family with a schema must also supply ``place_keypoints``, and vice versa.
        place_keypoints: Places the family's landmark table through the same skew/rotate/translate
            pipeline its outline goes through, or ``None`` for a family with no landmarks.

    Raises:
        ValueError: If ``members`` is empty, or if exactly one of ``keypoint_schema`` and
            ``place_keypoints`` is given — a family either has landmarks or does not.

    Examples:
        ```pycon
        >>> from synth_datasets.families import SHAPE_FAMILIES
        >>> primitives = SHAPE_FAMILIES[0]
        >>> primitives.name, primitives.has_keypoints
        ('primitives', False)
        >>> SHAPE_FAMILIES[1].member_type.__name__
        'AnimalShape'

        ```

    """

    name: str
    members: tuple[Shape, ...]
    base_outline: Callable[[str, float], NDArray[np.float64]]
    keypoint_schema: KeypointSchema | None = None
    place_keypoints: PlaceKeypoints | None = None

    def __post_init__(self) -> None:
        """Reject an empty family or a half-declared landmark capability."""
        if not self.members:
            raise ValueError(f"shape family {self.name!r} must have at least one member")
        if (self.keypoint_schema is None) != (self.place_keypoints is None):
            raise ValueError(
                f"shape family {self.name!r} must declare both keypoint_schema and place_keypoints, or neither"
            )

    @property
    def member_type(self) -> type[Enum]:
        """Return the family's enum class — the type ``isinstance`` and ``type(shape)`` see."""
        return type(self.members[0])

    @property
    def has_keypoints(self) -> bool:
        """Return whether this family's members carry a landmark table."""
        return self.keypoint_schema is not None

    @property
    def values(self) -> tuple[str, ...]:
        """Return every member's string value, in declaration order."""
        return tuple(str(member.value) for member in self.members)


#: Every shape family, in the order their classes are numbered by
#: :func:`~synth_datasets.core.config.class_names`. Append here to add a family — and extend
#: :data:`Shape` on the same change.
SHAPE_FAMILIES: tuple[ShapeFamily, ...] = (
    ShapeFamily(name="primitives", members=tuple(PrimitiveShape), base_outline=primitive_outline),
    ShapeFamily(
        name="animals",
        members=tuple(AnimalShape),
        base_outline=_table_outline(ANIMAL_POLYGONS),
        keypoint_schema=ANIMAL_KEYPOINT_SCHEMA,
        place_keypoints=animal_keypoints,
    ),
    ShapeFamily(
        name="symbols",
        members=tuple(SymbolShape),
        base_outline=_table_outline(SYMBOL_POLYGONS),
        keypoint_schema=SYMBOL_KEYPOINT_SCHEMA,
        place_keypoints=symbol_keypoints,
    ),
    ShapeFamily(
        name="letters",
        members=tuple(LetterShape),
        base_outline=_table_outline(LETTER_POLYGONS),
        keypoint_schema=LETTER_KEYPOINT_SCHEMA,
        place_keypoints=letter_keypoints,
    ),
)

#: Every drawable shape, across every family, in class-id order. This is the full vocabulary
#: :func:`~synth_datasets.core.config.class_names` numbers when a run is not narrowed.
ALL_SHAPES: tuple[Shape, ...] = tuple(member for family in SHAPE_FAMILIES for member in family.members)

#: The shapes a :class:`~synth_datasets.core.config.SyntheticConfig` draws when ``shapes`` is not
#: overridden — the analytic family alone, i.e. the vocabulary that predates every asset-backed one.
DEFAULT_SHAPES: tuple[Shape, ...] = SHAPE_FAMILIES[0].members

_BY_TYPE: dict[type, ShapeFamily] = {family.member_type: family for family in SHAPE_FAMILIES}
_BY_VALUE: dict[str, ShapeFamily] = {value: family for family in SHAPE_FAMILIES for value in family.values}
#: Every shape keyed by its name (its string value), for :func:`resolve_shape`. One flat map is enough because
#: values are unique across families, which :data:`_BY_VALUE` above already relies on.
_BY_NAME: dict[str, Shape] = {str(shape.value): shape for shape in ALL_SHAPES}


def resolve_shape(shape: Shape | str) -> Shape:
    """Return the shape a name stands for; a :data:`Shape` member is returned unchanged.

    A YAML file or a command line can only spell a shape as a string, so
    :class:`~synth_datasets.core.config.SyntheticConfig` resolves names through here at construction. The member,
    not the equal string, is what every downstream lookup needs: under the ``str`` mixin ``"duck"`` compares equal to
    ``AnimalShape.DUCK`` but is not one, and :func:`family_of` looks a shape up by its type.

    Args:
        shape: A :data:`Shape` member, or its name such as ``"duck"``, ``"square"`` or ``"a"``.

    Returns:
        The member.

    Raises:
        ValueError: If ``shape`` is a string naming no shape; the message lists every valid name, family by family.

    Examples:
        ```pycon
        >>> from synth_datasets.families import resolve_shape
        >>> resolve_shape("duck")
        <AnimalShape.DUCK: 'duck'>
        >>> resolve_shape("dragon")
        Traceback (most recent call last):
        ...
        ValueError: unknown shape name 'dragon'; valid names are primitives: ...

        ```

    """
    if isinstance(shape, Shape):
        return shape
    member = _BY_NAME.get(shape)
    if member is None:
        valid = "; ".join(f"{family.name}: {', '.join(family.values)}" for family in SHAPE_FAMILIES)
        raise ValueError(f"unknown shape name {shape!r}; valid names are {valid}")
    return member


def family_of(shape: Shape) -> ShapeFamily:
    """Return the family a shape member belongs to.

    Args:
        shape: Any :data:`Shape` member.

    Returns:
        The owning :class:`ShapeFamily`.

    Raises:
        KeyError: If ``shape`` is not a member of any registered family — which for a genuine
            :data:`Shape` member is impossible, and for a bare string is the intended failure.

    Examples:
        ```pycon
        >>> from synth_datasets.families import family_of
        >>> from synth_datasets.families.symbols import SymbolShape
        >>> family_of(SymbolShape.KITE).name
        'symbols'

        ```

    """
    return _BY_TYPE[type(shape)]


def base_outline(value: str, size: float) -> NDArray[np.float64]:
    """Return the origin-centered outline for any shape value, from any family.

    The single dispatch point that replaced the per-family if-chain: a new family becomes reachable
    here the moment it is registered, with no edit to this function.

    Args:
        value: A :data:`Shape` value — ``"square"``, ``"duck"``, ``"kite"``, ``"a"``, and so on.
        size: Bounding size (side / diameter / larger extent) in pixels.

    Returns:
        ``(num_points, 2)`` float array centered at the origin.

    Raises:
        ValueError: If ``value`` names no shape in any registered family.

    Examples:
        ```pycon
        >>> from synth_datasets.families import base_outline
        >>> base_outline("triangle", 6.0).shape
        (3, 2)

        ```

    """
    family = _BY_VALUE.get(value)
    if family is None:
        known = ", ".join(shape.value for shape in ALL_SHAPES)
        raise ValueError(f"unknown shape {value!r}; expected one of {known}")
    return family.base_outline(value, size)


def shape_outline(
    value: str, center: tuple[float, float], size: float, angle: float = 0.0, skew: float = 0.0
) -> NDArray[np.float64]:
    """Build the skewed, rotated, translated outline for any shape value, from any family.

    The drawing entry point. It replaced ``geometry.shape_outline``, whose name said "polygon" while
    :attr:`~synth_datasets.core.sample.Annotation.polygon` means the *flat* coordinate list a
    writer emits — two different things one word away from each other.

    Args:
        value: A :data:`Shape` value — ``"square"``, ``"duck"``, ``"kite"``, ``"a"``, and so on.
        center: Target center ``(x, y)`` in pixels.
        size: Bounding size in pixels.
        angle: Rotation in radians applied about the shape center.
        skew: Signed fraction narrowing one pre-rotation half — see
            :attr:`~synth_datasets.core.config.SyntheticConfig.asymmetry_jitter`. ``0.0`` (the
            default) leaves the outline unchanged.

    Returns:
        ``(num_points, 2)`` float array in image coordinates.

    Raises:
        ValueError: If ``value`` names no shape in any registered family.

    Examples:
        ```pycon
        >>> from synth_datasets.families import shape_outline
        >>> shape_outline("triangle", center=(10.0, 10.0), size=6.0).shape
        (3, 2)

        ```

    """
    return place_points(base_outline(value, size), center, angle, skew)


def keypoint_schema_for(shapes: Iterable[Shape]) -> KeypointSchema | None:
    """Return the schema shared by every shape in ``shapes``, when there is exactly one.

    Args:
        shapes: The shapes a run draws from — typically
            :attr:`~synth_datasets.core.config.SyntheticConfig.shapes`.

    Returns:
        The :class:`~synth_datasets.core.keypoints.KeypointSchema` every shape shares, or
        ``None`` when ``shapes`` spans no single keypoint-bearing family. Use
        :func:`describe_keypoint_mismatch` to find out *which* of those cases it was.

    Examples:
        ```pycon
        >>> from synth_datasets.families.animals import AnimalShape
        >>> from synth_datasets.families import keypoint_schema_for
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> keypoint_schema_for((AnimalShape.DUCK, AnimalShape.CAMEL)).kpt_shape
        16
        >>> keypoint_schema_for((PrimitiveShape.SQUARE,)) is None
        True

        ```

    """
    families = {type(shape) for shape in shapes}
    if len(families) != 1:
        return None
    return _BY_TYPE[families.pop()].keypoint_schema if families else None


def describe_keypoint_mismatch(shapes: Iterable[Shape]) -> str:
    """Return a caller-facing explanation of why ``shapes`` names no single keypoint schema.

    :func:`keypoint_schema_for` collapses three distinct situations into one ``None``; the config
    validator needs to tell them apart to write a useful message, and re-deriving the distinction at
    the call site is exactly the duplication this function removes.

    Args:
        shapes: The shapes that failed :func:`keypoint_schema_for`.

    Returns:
        A sentence naming the specific problem: shapes with no landmark table at all, a mix of two
        keypoint-bearing families, or an empty vocabulary.

    Examples:
        ```pycon
        >>> from synth_datasets.families import describe_keypoint_mismatch
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> describe_keypoint_mismatch((PrimitiveShape.SQUARE,)).startswith("['square'] have no keypoint table")
        True

        ```

    """
    shapes = tuple(shapes)
    if not shapes:
        return "the shape vocabulary is empty, so it names no keypoint family"
    # Shapes with no table at all are the primary failure and are reported first; only a *pure* mix
    # of two keypoint-bearing families (nothing table-less present) falls through to the second case.
    unsupported = [shape.value for shape in shapes if not _BY_TYPE[type(shape)].has_keypoints]
    if unsupported:
        supported = ", ".join(value for family in SHAPE_FAMILIES if family.has_keypoints for value in family.values)
        return f"{unsupported} have no keypoint table; restrict shapes to a keypoint-bearing family: {supported}"
    families = sorted({_BY_TYPE[type(shape)].name for shape in shapes})
    return f"shapes mix the {families} families; a dataset carries only one landmark schema"


def place_keypoints(
    shape: Shape, center: tuple[float, float], size: float, angle: float, skew: float
) -> NDArray[np.float64] | None:
    """Place one shape's landmark table, or return ``None`` for a family with no landmarks.

    Args:
        shape: The shape being placed.
        center: Target center ``(x, y)`` in pixels.
        size: Bounding size in pixels.
        angle: Rotation in radians about the shape center.
        skew: Signed asymmetry fraction; see
            :attr:`~synth_datasets.core.config.SyntheticConfig.asymmetry_jitter`.

    Returns:
        The placed ``(num_keypoints, 2)`` table, or ``None`` when ``shape``'s family carries none.

    """
    family = _BY_TYPE[type(shape)]
    if family.place_keypoints is None:
        return None
    return family.place_keypoints(shape, center, size, angle, skew)
