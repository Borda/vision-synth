"""The ``vision-synth`` command line.

One command, ``generate``, over :func:`~synth_datasets.generate_dataset`::

    vision-synth generate ./shapes-ds 1000 --fmt yolo --task obb --shapes duck,camel --img_size 256 --seed 0

``output_dir`` and ``num_images`` are positional; ``--fmt``, ``--split_ratios``, ``--seed`` and ``--overwrite`` are
:func:`~synth_datasets.generate_dataset`'s own options, and every other ``--flag`` is a
:class:`~synth_datasets.SyntheticConfig` field. Shapes and colours are given by name, comma-separated. An invalid value,
an unknown flag (the message lists the valid ones) or an already populated output directory ends the run with
one line on stderr and exit status 1.

The argument parser is ``fire``, which ships with the ``cli`` extra (``pip install "vision-synth[cli]"``) and is
imported only when :func:`main` runs, so importing this module — like importing :mod:`synth_datasets` — needs neither
``fire`` nor ``torch``.

"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import synth_datasets
from synth_datasets import SplitRatios, SyntheticConfig, generate_dataset
from synth_datasets.core.config import Fill
from synth_datasets.families.shape_enum import ShapeEnum

_INSTALL_HINT = 'the vision-synth command line needs the "cli" extra: pip install "vision-synth[cli]"'

#: The split ratios accepted on the command line: a ``SplitRatios``, a ``{name: fraction}`` mapping, or a
#: ``(train, val, test)`` triple.
SplitSpec = SplitRatios | Mapping[str, float] | Sequence[float]


def _as_names(names: str | Sequence[str]) -> tuple[str, ...]:
    """Return names as a tuple, splitting a comma-separated string (``"duck,camel"``)."""
    if isinstance(names, str):
        return tuple(name.strip() for name in names.split(",") if name.strip())
    return tuple(names)


def _cli_message(exc: Exception) -> str:
    """Return ``exc`` as one line, naming the ``--overwrite`` flag where the library names its keyword."""
    message = " ".join(str(exc).split())
    if isinstance(exc, FileExistsError):
        reason = message.partition("; pass overwrite=True")[0]
        message = f"{reason}; pass --overwrite to replace it, or choose an empty output directory"
    return f"vision-synth: error: {message}"


#: The type each of the command's own options accepts; every other option is a ``SyntheticConfig`` field, checked
#: against that field's annotation.
#: One colour as the command line spells it: a name, or an ``(r, g, b)`` triple (a list works too).
_ONE_COLOR = str | tuple[int, int, int] | Fill
#: A colour option as the command line spells it: comma-separated names, one triple, or a list of names and triples.
_COLORS_INPUT = str | tuple[int, int, int] | tuple[_ONE_COLOR, ...] | None
#: A shape option as the command line spells it: comma-separated names, or a list of names.
_SHAPES_INPUT = str | tuple[str | ShapeEnum, ...] | None

_OWN_OPTIONS: dict[str, object] = {
    "output_dir": str | Path,
    "num_images": int,
    "fmt": str,
    "split_ratios": SplitRatios | dict[str, float] | tuple[float, ...] | None,
    "seed": int | None,
    "overwrite": bool,
    "shapes": _SHAPES_INPUT,
    "colors": _COLORS_INPUT,
}

#: ``SyntheticConfig`` fields whose annotation is the *normalized* type, checked against their input spellings instead:
#: the annotation ``tuple[Fill, ...]`` would refuse ``--distractor_colors "[(255,215,0)]"``, which the config accepts.
_INPUT_HINTS: dict[str, object] = {
    "distractor_colors": _COLORS_INPUT,
    "distractor_shapes": _SHAPES_INPUT,
}

#: Exact checks for the scalar annotations; ``bool`` is refused where a number is expected, although it is an ``int``.
_SCALAR_CHECKS: dict[object, Callable[[object], bool]] = {
    bool: lambda value: isinstance(value, bool),
    int: lambda value: isinstance(value, Integral) and not isinstance(value, bool),
    float: lambda value: isinstance(value, Real) and not isinstance(value, bool),
    type(None): lambda value: value is None,
}

#: What each scalar annotation is called in an error message.
_SCALAR_NAMES: dict[object, str] = {
    bool: "true or false",
    int: "an integer",
    float: "a number",
    str: "a string",
    type(None): "None",
    Path: "a path",
}


def _matches(value: object, hint: object) -> bool:
    """Return whether a parsed option ``value`` has the type ``hint`` annotates.

    A name stands for an enum member or a colour (``SyntheticConfig`` resolves it, and rejects an unknown one), and a
    list stands for a tuple, since ``fire`` parses ``[1,2]`` as a list.

    """
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        return any(_matches(value, arg) for arg in typing.get_args(hint))
    if origin in (tuple, dict):
        return _matches_container(value, origin, typing.get_args(hint))
    if hint in _SCALAR_CHECKS:
        return _SCALAR_CHECKS[hint](value)
    named = isinstance(value, str) and isinstance(hint, type) and issubclass(hint, (Enum, Fill))
    return named or (isinstance(hint, type) and isinstance(value, hint))


def _matches_container(value: object, origin: object, args: tuple[object, ...]) -> bool:
    """Return whether ``value`` is a ``tuple[...]`` (given as a tuple or list) or ``dict[...]`` of the ``args``."""
    if origin is dict:
        return isinstance(value, Mapping) and all(
            _matches(key, args[0]) and _matches(item, args[1]) for key, item in value.items()
        )
    if not isinstance(value, (tuple, list)):
        return False
    if len(args) == 2 and args[1] is Ellipsis:
        return all(_matches(item, args[0]) for item in value)
    return len(value) == len(args) and all(_matches(item, arg) for item, arg in zip(value, args, strict=True))


def _describe(hint: object) -> str:
    """Return a short description of ``hint`` for an error message."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        return " or ".join(_describe(arg) for arg in args)
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return f"a list of {_describe(args[0])} values"
    if origin is tuple:
        return f"{len(args)} values ({', '.join(_describe(arg) for arg in args)})"
    if origin is dict:
        return f"a mapping of {_describe(args[0])} to {_describe(args[1])}"
    if hint in _SCALAR_NAMES:
        return _SCALAR_NAMES[hint]
    return f"a {getattr(hint, '__name__', hint)}"


def _check_flags(own: Mapping[str, object], config_fields: Mapping[str, Any]) -> None:
    """Refuse an unknown option or a wrongly typed value by name, before anything is generated or deleted.

    Args:
        own: The command's own options by name, each checked against :data:`_OWN_OPTIONS`.
        config_fields: The remaining options, each checked against its ``SyntheticConfig`` field's annotation.

    Raises:
        ValueError: If an option names no :class:`~synth_datasets.SyntheticConfig` field (the message lists the
            valid ones), or a value does not have its option's type (the message names the option).

    """
    fields = {field.name for field in dataclasses.fields(SyntheticConfig) if field.init}
    unknown = [f"--{name}" for name in config_fields if name not in fields]
    if unknown:
        raise ValueError(
            f"unknown option {', '.join(unknown)}; the options are {', '.join(f'--{name}' for name in _OWN_OPTIONS)} "
            f"and the SyntheticConfig fields: {', '.join(sorted(fields))}"
        )
    # The annotations name classes config.py imports only for type checking; the package namespace resolves them.
    hints = typing.get_type_hints(SyntheticConfig, localns=vars(synth_datasets))
    checks = [*((name, value, _OWN_OPTIONS[name]) for name, value in own.items())]
    checks += [(name, value, _INPUT_HINTS.get(name, hints[name])) for name, value in config_fields.items()]
    for name, value, hint in checks:
        if not _matches(value, hint):
            raise ValueError(f"--{name} expects {_describe(hint)}, got {value!r}")


def _is_triple(value: object) -> bool:
    """Return whether ``value`` spells one ``(r, g, b)`` colour: three integers, as a tuple or a list."""
    return isinstance(value, (tuple, list)) and len(value) == 3 and all(_SCALAR_CHECKS[int](item) for item in value)


def _as_colors(colors: object) -> object:
    """Return a colour option as the tuple of colours ``SyntheticConfig`` takes, from any spelling it was given in.

    Comma-separated names are split, a lone ``(r, g, b)`` triple becomes a one-colour tuple, and a list triple becomes
    a tuple, since :meth:`~synth_datasets.core.config.Fill.parse` reads a triple only as a tuple.

    """
    if isinstance(colors, str):
        return _as_names(colors)
    if _is_triple(colors):
        return (tuple(colors),)  # type: ignore[arg-type]
    if isinstance(colors, (tuple, list)):
        return tuple(tuple(item) if isinstance(item, list) else item for item in colors)
    return colors


def _as_split_ratios(split_ratios: SplitSpec | None) -> SplitRatios | None:
    """Return ``split_ratios`` as a :class:`SplitRatios`, or ``None`` for the 70/20/10 default.

    Raises:
        ValueError: If a sequence is not exactly ``(train, val, test)``.

    """
    if split_ratios is None or isinstance(split_ratios, SplitRatios):
        return split_ratios
    if isinstance(split_ratios, Mapping):
        return SplitRatios.custom(split_ratios)
    values = tuple(split_ratios)
    if len(values) != 3:
        raise ValueError(
            f"split_ratios as a sequence must be (train, val, test), got {values!r}; "
            "pass a mapping such as {'train': 0.9, 'holdout': 0.1} for other splits"
        )
    return SplitRatios(*values)


def generate(
    output_dir: str | Path,
    num_images: int,
    fmt: str = "coco",
    split_ratios: SplitSpec | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    shapes: str | Sequence[str] | None = None,
    colors: str | Sequence[object] | None = None,
    **config_fields: Any,  # noqa: ANN401 - forwarded verbatim to SyntheticConfig
) -> dict[str, int]:
    """Generate a dataset on disk and return its per-split image counts — the ``generate`` command.

    Args:
        output_dir: Destination directory.
        num_images: Total number of images across all splits.
        fmt: ``"coco"``, ``"yolo"``, or any key registered with :func:`~synth_datasets.register_writer`.
        split_ratios: A ``{name: fraction}`` mapping or a ``(train, val, test)`` triple; ``None`` means 70/20/10.
        seed: Seed for a reproducible dataset; ``None`` uses fresh entropy.
        overwrite: Replace an earlier dataset in ``output_dir``; see :func:`~synth_datasets.generate_dataset`.
        shapes: Shape names, as a sequence or one comma-separated string such as ``"duck,camel"``.
        colors: Colours, as one comma-separated string of names (``red``, ``green``, ``blue``, any case), one
            ``(r, g, b)`` triple, or a sequence of names and triples. ``--distractor_colors`` takes the same spellings,
            and ``--distractor_shapes`` the same as ``shapes``.
        **config_fields: Any other :class:`~synth_datasets.SyntheticConfig` field, e.g. ``task="obb"``.

    Returns:
        Ordered mapping of split name to the number of images written.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from synth_datasets.cli import generate
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     generate(tmp, 3, fmt="yolo", img_size=32, seed=0, shapes="duck,camel")
        {'train': 2, 'val': 1}

        ```

    """
    # Checked as given, then normalized: the check reads each option's input spellings, not the config's types.
    own = {"output_dir": output_dir, "num_images": num_images, "fmt": fmt, "split_ratios": split_ratios}
    _check_flags({**own, "seed": seed, "overwrite": overwrite, "shapes": shapes, "colors": colors}, config_fields)
    if shapes is not None:
        config_fields["shapes"] = _as_names(shapes)
    if colors is not None:
        config_fields["colors"] = _as_colors(colors)
    if config_fields.get("distractor_shapes") is not None:
        config_fields["distractor_shapes"] = _as_names(config_fields["distractor_shapes"])
    if config_fields.get("distractor_colors") is not None:
        config_fields["distractor_colors"] = _as_colors(config_fields["distractor_colors"])
    if isinstance(config_fields.get("background"), list):
        config_fields["background"] = tuple(config_fields["background"])
    return generate_dataset(
        output_dir,
        num_images,
        fmt=fmt,
        split_ratios=_as_split_ratios(split_ratios),
        seed=seed,
        overwrite=overwrite,
        **config_fields,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``vision-synth`` command line.

    Args:
        argv: The arguments after the program name; ``None`` reads them from ``sys.argv``.

    Raises:
        SystemExit: If ``fire`` is not installed, carrying the command that installs the ``cli`` extra; or, with
            a one-line message on stderr and exit status 1, if a value is invalid or the output directory already
            holds a dataset and ``--overwrite`` was not given. An unknown flag is reported with the list of valid
            ones, and a wrongly typed value with the flag it was given to.

    """
    try:
        fire = importlib.import_module("fire")
    except ModuleNotFoundError as exc:
        # Exact match: a module missing *inside* an installed fire is a different problem.
        if exc.name != "fire":
            raise
        raise SystemExit(_INSTALL_HINT) from exc
    try:
        fire.Fire({"generate": generate}, command=None if argv is None else list(argv), name="vision-synth")
    except (FileExistsError, ValueError) as exc:
        # A bad value or a populated destination is the caller's to fix, not a crash: one line, no traceback.
        print(_cli_message(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
