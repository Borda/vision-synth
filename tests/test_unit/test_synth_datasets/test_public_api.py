"""Public-API surface tests for the `synth_datasets` package."""

from __future__ import annotations


def test_subpackage_exports_public_names() -> None:
    """The `synth_datasets` package re-exports the documented public API.

    The facade carries what a dataset-building caller needs — the generator, the config, the writers, the vocabulary
    helpers — and no longer carries five names per shape family. Family specifics like `animal_keypoints` now come from
    the family's own module, which is what stops this surface growing every time a family is added.

    """
    from synth_datasets import (
        ALL_SHAPES,
        DEFAULT_SHAPES,
        SHAPE_FAMILIES,
        CocoWriter,
        SyntheticConfig,
        SyntheticGenerator,
        YoloWriter,
        generate_dataset,
        get_writer,
        shape_outline,
    )
    from synth_datasets.families.animals import animal_keypoints

    assert all(
        callable(obj)
        for obj in (generate_dataset, get_writer, SyntheticGenerator, SyntheticConfig, shape_outline, animal_keypoints)
    )
    assert len(ALL_SHAPES) == sum(len(family.members) for family in SHAPE_FAMILIES)
    assert CocoWriter.__name__ == "CocoWriter"
    assert YoloWriter.__name__ == "YoloWriter"
    assert isinstance(DEFAULT_SHAPES, tuple)
    assert DEFAULT_SHAPES


def test_all_exported_names_resolve() -> None:
    """Every name in `__all__` is reachable as an attribute of the package.

    `__all__` is hand-maintained alongside the imports that populate it; a name added to one but not the other would
    still import cleanly and only fail for a caller doing `from synth_datasets import *` or introspecting the module, so
    this checks the two stay in sync directly. `SyntheticIterableDataset` is excluded: it is the one torch-dependent
    name in this namespace, resolved lazily via module `__getattr__`, and genuinely does not resolve without torch — see
    `test_lazy_torch_dependent_name_needs_torch` for its contract.

    """
    import synth_datasets

    checked = [name for name in synth_datasets.__all__ if name != "SyntheticIterableDataset"]
    missing = [name for name in checked if not hasattr(synth_datasets, name)]
    assert missing == []


def test_lazy_torch_dependent_name_needs_torch() -> None:
    """`SyntheticIterableDataset` resolves when torch is installed, and only then.

    It is the one name in `synth_datasets.__all__` that needs torch (it wraps `torch.utils.data.IterableDataset`); every
    other export is torch-free. This pins that contract explicitly rather than letting it fall out of
    `test_all_exported_names_resolve`.

    """
    import importlib.util

    import synth_datasets

    if importlib.util.find_spec("torch") is None:
        import pytest

        with pytest.raises(ModuleNotFoundError, match="torch"):
            _ = synth_datasets.SyntheticIterableDataset
    else:
        assert synth_datasets.SyntheticIterableDataset.__name__ == "SyntheticIterableDataset"


def test_every_registered_family_derives_from_the_shape_base() -> None:
    """`Shape` is the shared base class, so the registry is the only place a family is named.

    `Shape` used to be a hand-written `PrimitiveShape | AnimalShape | SymbolShape | LetterShape` union that had to be
    extended alongside `SHAPE_FAMILIES`. Missing that second edit left the new family drawable but invisible to
    `SyntheticConfig`'s validation, which rejects anything failing `isinstance(value, Shape)` — so the family's own
    members would have been refused as if they were bare strings. Deriving every family enum from one base makes the two
    impossible to desynchronize, and this asserts the property the union used to provide by hand.

    """
    from synth_datasets import ALL_SHAPES, SHAPE_FAMILIES, Shape

    assert all(issubclass(family.member_type, Shape) for family in SHAPE_FAMILIES)
    assert all(isinstance(shape, Shape) for shape in ALL_SHAPES)


def test_the_shape_base_still_rejects_a_bare_string() -> None:
    """A shape *value* is not a `Shape`, even though the str mixin makes it compare equal to one.

    This is the check `SyntheticConfig._validate_vocabulary` relies on to catch `shapes=("duck",)`. Under the `str`
    mixin `"duck" == AnimalShape.DUCK` and both hash alike, so a membership or equality test would let the string
    through and it would only fail much later, in a registry lookup keyed by `type(shape)`.

    """
    from synth_datasets import Shape
    from synth_datasets.families.animals import AnimalShape

    assert AnimalShape.DUCK == "duck"
    assert not isinstance("duck", Shape)
