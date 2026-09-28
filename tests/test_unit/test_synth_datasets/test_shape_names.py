"""``SyntheticConfig`` accepts shape *names* wherever it accepts shape members.

A YAML file or a command line can only spell a shape as a string, so a config built from either used to have to map
every name to its enum member itself. The config now resolves names at its own boundary and stores the real members.

"""

from __future__ import annotations

import pytest

from synth_datasets import SyntheticConfig
from synth_datasets.families import ALL_SHAPES, resolve_shape
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.primitives import PrimitiveShape


def test_shapes_accepts_shape_names():
    """Each name resolves to its member, so every downstream ``is`` comparison still holds."""
    config = SyntheticConfig(shapes=("duck", PrimitiveShape.SQUARE, "a"))

    assert config.shapes == (AnimalShape.DUCK, PrimitiveShape.SQUARE, LetterShape("a"))
    assert [type(shape) for shape in config.shapes] == [AnimalShape, PrimitiveShape, LetterShape]


def test_shapes_accepts_one_bare_name():
    """A lone string is one name, never iterated into single letters (``"duck"`` is not ``d, u, c, k``)."""
    assert SyntheticConfig(shapes="duck").shapes == (AnimalShape.DUCK,)


def test_unknown_shape_name_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown shape name 'dragon'") as info:
        SyntheticConfig(shapes=("dragon",))
    assert "duck" in str(info.value)
    assert "square" in str(info.value)


def test_distractor_shapes_accept_names_too():
    config = SyntheticConfig(distractors=1, distractor_shapes=("camel",))
    assert config.distractor_shapes == (AnimalShape.CAMEL,)


def test_every_shape_resolves_from_its_own_value():
    assert all(resolve_shape(str(shape.value)) is shape for shape in ALL_SHAPES)
    assert all(resolve_shape(shape) is shape for shape in ALL_SHAPES)
