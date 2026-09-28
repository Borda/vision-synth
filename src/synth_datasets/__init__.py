"""Synthetic detection / segmentation / OBB / keypoint dataset generation.

Draw colored shapes on a canvas and export **COCO** or **YOLO** datasets for detection,
segmentation, oriented-bounding-box, or keypoint tasks. This is a standalone generation utility: no
dataset loaders, no model, no training loop.

The shape vocabulary is assembled from independent families —
:mod:`~synth_datasets.families.primitives` (analytic), :mod:`~synth_datasets.families.animals`
(traced silhouettes), :mod:`~synth_datasets.families.symbols`, and
:mod:`~synth_datasets.families.letters` (stroke figures) — registered in
:mod:`~synth_datasets.families`. Reach for a family's own module when you want its
specifics; this namespace exports the pieces a dataset-building caller needs.

This package is standalone and torch-free: ``import synth_datasets`` never touches
:mod:`fused_transforms`, so it installs and imports without the ``torch`` extra.
:class:`SyntheticIterableDataset` is the only torch-dependent name in this namespace, deliberately
left out of :data:`__all__` and resolved lazily on first attribute access instead — when
``synth_datasets`` is imported directly, even a torch-installed environment only pays that import
cost when the name is actually used. This is the only import path for the generator: the former
``fused_transforms.data`` facade and ``fused_transforms.generate_dataset`` alias are both removed,
since reaching either one ran the augmentation package's eager import and required ``torch``
regardless.

Examples:
    ```pycon
    >>> import tempfile
    >>> from synth_datasets import generate_dataset
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     counts = generate_dataset(tmp, num_images=10, fmt="yolo", task="detection", seed=0)
    ...     sorted(counts)
    ['test', 'train', 'val']

    ```

"""

from __future__ import annotations

import importlib
import itertools
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from synth_datasets.content.backgrounds import (
    Background,
    GradientBackground,
    ImageBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    SolidBackground,
    TextureBackground,
)
from synth_datasets.content.degradations import (
    JPEG as JPEG,
)
from synth_datasets.content.degradations import (
    ColorCast,
    Contrast,
    Degradation,
    GaussianBlur,
    GaussianNoise,
    Quantize,
    Vignette,
)
from synth_datasets.core.config import (
    DISTRACTOR_PALETTE,
    ClassEntry,
    ClassMode,
    ClassVocabulary,
    Color,
    Fill,
    OutputFormat,
    SplitRatios,
    SyntheticConfig,
    Task,
    class_id,
    class_names,
    class_vocabulary,
)
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.core.keypoints import KeypointSchema
from synth_datasets.core.sample import Annotation, Sample, SceneRecord
from synth_datasets.export.writers import CocoWriter, DatasetWriter, YoloWriter, get_writer, register_writer
from synth_datasets.families import (
    ALL_SHAPES,
    DEFAULT_SHAPES,
    SHAPE_FAMILIES,
    Shape,
    ShapeFamily,
    family_of,
    keypoint_schema_for,
    shape_outline,
)
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SymbolShape

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    # Explicit re-export (`as` same name): keeps this name resolvable to static type checkers even
    # though it is intentionally left out of __all__ below -- see the module docstring and __getattr__.
    from synth_datasets.export.datasets import SyntheticIterableDataset as SyntheticIterableDataset

try:
    # `synth_datasets` ships inside the `vision-synth` distribution (see pyproject.toml's
    # packages.find.include), not its own -- this lookup breaks if synth_datasets is ever split into
    # its own distribution and must be repointed at that distribution's name then.
    __version__ = version("vision-synth")
except PackageNotFoundError:  # pragma: no cover - only hit for an unbuilt/uninstalled checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "ALL_SHAPES",
    "DEFAULT_SHAPES",
    "DISTRACTOR_PALETTE",
    "JPEG",
    "SHAPE_FAMILIES",
    "AnimalShape",
    "Annotation",
    "Background",
    "ClassEntry",
    "ClassMode",
    "ClassVocabulary",
    "CocoWriter",
    "Color",
    "ColorCast",
    "Contrast",
    "DatasetWriter",
    "Degradation",
    "Fill",
    "GaussianBlur",
    "GaussianNoise",
    "GradientBackground",
    "ImageBackground",
    "ImpulseNoiseBackground",
    "KeypointSchema",
    "LetterShape",
    "NoiseBackground",
    "OutputFormat",
    "PrimitiveShape",
    "Quantize",
    "Sample",
    "SceneRecord",
    "Shape",
    "ShapeFamily",
    "SolidBackground",
    "SplitRatios",
    "SymbolShape",
    "SyntheticConfig",
    "SyntheticGenerator",
    "Task",
    "TextureBackground",
    "Vignette",
    "YoloWriter",
    "class_id",
    "class_names",
    "class_vocabulary",
    "family_of",
    "generate_dataset",
    "get_writer",
    "keypoint_schema_for",
    "register_writer",
    "shape_outline",
]

#: Names resolved on first access rather than at import. :class:`SyntheticIterableDataset` is here
#: to keep ``import synth_datasets`` free of torch — measured at ~440 ms of the ~480 ms this
#: package used to cost, imposed on every caller including the many who only write a dataset to disk.
_LAZY: dict[str, str] = {"SyntheticIterableDataset": "synth_datasets.export.datasets"}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module-level attribute access is untyped by nature
    """Resolve a lazily-imported public name on first access (:pep:`562`).

    Args:
        name: The attribute being looked up on this module.

    Returns:
        The resolved object. :data:`_LAZY` names resolve by attribute access exactly like the rest
        of this namespace's public surface; they are simply imported late, and -- unlike the rest --
        deliberately excluded from :data:`__all__` so ``from synth_datasets import *`` stays
        torch-free.

    Raises:
        AttributeError: If ``name`` is neither exported nor deferred.
        ModuleNotFoundError: If resolving ``name`` requires :mod:`torch` and it is not installed; the
            re-raised error carries an actionable ``pip install torch`` message instead of the raw one.

    """
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # Exact match, not a prefix/split check: a missing submodule *within* torch (e.g. torch.jit)
        # is a different problem and must not be mislabelled as torch itself being absent.
        if exc.name != "torch":
            raise
        raise ModuleNotFoundError(
            f"{name!r} requires the 'torch' package, which is not installed: run `pip install torch`.",
            name="torch",
        ) from exc
    return getattr(module, name)


def __dir__() -> list[str]:
    """List module attributes for ``dir()`` and tab-completion, :data:`_LAZY` names included (:pep:`562`).

    Returns:
        Sorted names combining this module's regular globals, :data:`__all__`, and the deferred names
        in :data:`_LAZY` -- the latter are not assigned in the module namespace until :func:`__getattr__`
        resolves them, so without this override ``dir(synth_datasets)`` would omit them even though
        they are reachable via attribute access.

    """
    return sorted({*globals(), *__all__, *_LAZY})


def _assign_splits(num_images: int, split_ratios: SplitRatios) -> dict[str, int]:
    """Return per-split counts summing to ``num_images`` using largest remainders.

    Equal remainders are awarded in the configured split order.

    Args:
        num_images: Total images to distribute.
        split_ratios: Fractions per split.

    Returns:
        Ordered ``split -> count`` mapping whose values sum to ``num_images``.

    """
    ratios = split_ratios.to_dict()
    total_ratio = sum(ratios.values())
    exact = {name: frac * num_images / total_ratio for name, frac in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = num_images - sum(counts.values())
    # SplitRatios tolerates tiny sum drift; normalize above so that drift cannot create a negative remainder.
    by_remainder = sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in by_remainder[:remainder]:
        counts[name] += 1
    return {name: count for name, count in counts.items() if count > 0}


def generate_dataset(
    output_dir: str | Path,
    num_images: int,
    fmt: OutputFormat | str = OutputFormat.COCO,
    split_ratios: SplitRatios | None = None,
    seed: int | None = None,
    config: SyntheticConfig | None = None,
    *,
    overwrite: bool = False,
    **config_kwargs: Any,  # noqa: ANN401 - forwarded verbatim to SyntheticConfig
) -> dict[str, int]:
    """Generate a synthetic dataset on disk and return per-split image counts.

    Everything about an image's *content* — task, class mode, shapes, colors, size — is a
    :class:`SyntheticConfig` field, reachable either through ``config`` or through ``config_kwargs``.
    Only ``fmt`` and ``split_ratios``, which describe the on-disk layout rather than the pixels, are
    parameters here.

    ``task`` and ``class_mode`` are config fields with a single owner: ``task="keypoints"`` passed
    here arrives as a config field, so the generator and the writer always read the same task.

    Args:
        output_dir: Destination directory (created if absent).
        num_images: Total number of images to generate across all splits.
        fmt: Output layout: ``"coco"``, ``"yolo"`` (or an :class:`OutputFormat`), or any key
            registered with :func:`register_writer`.
        split_ratios: Train/val/test fractions; defaults to 70/20/10.
        seed: Seed for reproducible generation; ``None`` uses fresh entropy.
        config: Full :class:`SyntheticConfig`. When given, ``config_kwargs`` must be empty — a
            config already says everything they would.
        overwrite: Replace an earlier dataset's files instead of refusing. The writer deletes
            the paths this run writes (for COCO, each ``<split>/``; for YOLO, ``images/<split>``,
            ``labels/<split>`` and ``data.yaml``) plus every path an earlier run recorded in the
            ``.vision-synth.json`` manifest, so a split an earlier run wrote and this run does not
            (say, ``test`` when this run's ratios leave it empty) is removed too. Nothing else
            under ``output_dir`` is touched, and nothing is deleted through a symlink or outside
            ``output_dir``. The new dataset is written into a staging directory first and the
            earlier one is replaced only once it is complete, so a failure while generating
            leaves the earlier dataset as it was (see :meth:`DatasetWriter.write_replacing`). Without it, a
            populated destination is refused, since writing into it would mix the two datasets.
        **config_kwargs: :class:`SyntheticConfig` fields (``task``, ``class_mode``, ``img_size``,
            ``shapes``, ``colors``, …) used to build the config when ``config`` is not supplied.

    Returns:
        Ordered mapping of split name to the number of images written.

    Raises:
        ValueError: If ``output_dir`` holds a staging or backup directory an interrupted overwrite left,
            if ``num_images`` is not a positive integer, if both a ``config`` and
            ``config_kwargs`` were supplied, if ``seed`` is negative, if no writer is registered
            for ``fmt``, or if ``overwrite`` would delete outside ``output_dir`` or through a
            symlink.
        FileExistsError: If ``overwrite`` is false and a path the writer would write is already
            populated. Raised before any sample is generated or any file written.
        RuntimeError: If ``overwrite`` is true and the swap of the new dataset into place fails.
            It is rolled back first, by renaming only. When the earlier dataset is fully back, the
            emptied ``.vision-synth-backup-*`` directory is removed; otherwise it is kept and named.
            A ``KeyboardInterrupt`` during the swap is rolled back the same way and re-raised.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from synth_datasets import generate_dataset
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     generate_dataset(tmp, num_images=10, fmt="coco", task="segmentation",
        ...                      img_size=64, seed=1)
        {'train': 7, 'val': 2, 'test': 1}

        ```

    """
    if isinstance(num_images, bool) or not isinstance(num_images, Integral) or num_images < 1:
        raise ValueError(f"num_images must be a positive integer, got {num_images}")
    if config is not None and config_kwargs:
        raise ValueError(
            f"pass either a config or its fields as keywords, not both; got config plus {sorted(config_kwargs)}"
        )
    split_ratios = split_ratios or SplitRatios()
    config = config if config is not None else SyntheticConfig(**config_kwargs)

    # ``get_writer`` resolves an OutputFormat member, its value, or any registered key; coercing
    # through ``OutputFormat(fmt)`` first used to reject every custom format register_writer allows.
    # ``SyntheticConfig`` already guarantees a single keypoint-bearing family under Task.KEYPOINTS,
    # so the schema is None only for tasks that never read it.
    writer: DatasetWriter = get_writer(
        fmt,
        config.task,
        class_vocabulary(config.class_mode, config.shapes, config.colors),
        keypoint_schema=keypoint_schema_for(config.shapes),
    )
    counts = _assign_splits(num_images, split_ratios)
    # ``generate`` is lazy, so a bad seed would otherwise surface only on the first sample — after
    # ``overwrite`` had already deleted the earlier dataset. Seeding a throwaway generator checks it now.
    np.random.default_rng(seed)

    # Stream a single lazy sample source through per-split islice views so only one Sample is
    # materialized at a time. The writer must consume the splits in insertion order and exactly once
    # each; DatasetWriter.write documents that contract, since these views share one iterator.
    generator = SyntheticGenerator(config)
    sample_stream = generator.generate(num_images, seed=seed)
    splits: dict[str, Iterable[Sample]] = {
        split: itertools.islice(sample_stream, count) for split, count in counts.items()
    }
    if overwrite:
        # Staged: the earlier dataset is replaced only once the new one is completely written, so a
        # failure while generating (a bad value that surfaces only on the first sample) loses nothing.
        writer.write_replacing(splits, output_dir)
    else:
        # Refuse before a single sample is generated or file written.
        writer.prepare_output(counts, output_dir)
        writer.write(splits, output_dir)
    return counts
