"""The base class every shape family's enum inherits, and nothing else.

Its whole job is to make "is this a drawable shape?" answerable by one type rather than by a
hand-maintained union. :data:`~synth_datasets.families.Shape` used to be spelled
``PrimitiveShape | AnimalShape | SymbolShape | LetterShape``, which meant adding a family was a
two-site edit — append to the registry *and* extend the union — with the second site easy to forget
and slow to fail. With every family enum deriving from :class:`ShapeEnum`, the union is the base
class, so the registry is the only place a new family has to be named.

The class is deliberately **empty**. Two reasons, both hard:

* Behavior like ``shape.outline(size)`` would need
  :mod:`~synth_datasets.families`, which imports every family module, which would import
  this one — an import cycle. The family-aware operations stay free functions in that module, where
  they can see the registry.
* Enum members share a namespace with the class's own attributes, so any method or property named
  here permanently forbids a shape value of that name. An empty base forbids nothing.

Examples:
    ```pycon
    >>> from synth_datasets.families.animals import AnimalShape
    >>> from synth_datasets.families.shape_enum import ShapeEnum
    >>> isinstance(AnimalShape.DUCK, ShapeEnum)
    True
    >>> isinstance("duck", ShapeEnum)
    False

    ```

"""

from __future__ import annotations

from enum import Enum


class ShapeEnum(str, Enum):
    """Common base of every shape family's enum; carries identity, no behavior.

    Subclassing a *memberless* :class:`~enum.Enum` is the one form of enum inheritance Python
    allows, which is what makes this work: :class:`ShapeEnum` declares no members, so each family
    is free to declare its own. The :class:`str` mixin is inherited too, so members keep comparing
    equal to their own values (``AnimalShape.DUCK == "duck"``) and serialize as plain strings.

    ``isinstance(value, ShapeEnum)`` is the runtime membership test the config validator uses. It
    accepts any family's member and still rejects a bare ``"duck"`` string, which is the trap worth
    catching: under the :class:`str` mixin that string compares *and hashes* equal to the member
    while failing every identity test the package branches on.

    Examples:
        ```pycon
        >>> from synth_datasets.families.shape_enum import ShapeEnum
        >>> from synth_datasets.families.symbols import SymbolShape
        >>> issubclass(SymbolShape, ShapeEnum)
        True
        >>> SymbolShape.KITE == "kite"
        True
        >>> list(ShapeEnum)
        []

        ```

    """
