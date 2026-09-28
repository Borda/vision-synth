"""Serialize generated samples to COCO or YOLO dataset layouts.

Both writers consume the same format-agnostic
:class:`~synth_datasets.core.sample.Sample` objects and select which
annotation fields to emit based on the requested
:class:`~synth_datasets.core.config.Task`.

COCO layout (Roboflow-style)::

    <out>/<split>/img_000000.jpg
    <out>/<split>/_annotations.coco.json

YOLO layout (Ultralytics-style)::

    <out>/images/<split>/img_000000.jpg
    <out>/labels/<split>/img_000000.txt
    <out>/data.yaml

Either writer also records the paths it wrote in ``<out>/.vision-synth.json``; ``overwrite`` deletes only what that
manifest lists and what the new run itself writes, never anything inferred from a file name.

COCO has no native oriented-box field, so for :attr:`Task.OBB` the four corners are
stored as a 4-point ``segmentation`` polygon alongside the axis-aligned ``bbox``.

For :attr:`Task.KEYPOINTS` both writers emit the landmark block in addition to the box: COCO gains
a ``segmentation`` polygon (as for :attr:`Task.SEGMENTATION`) plus per-category ``keypoints``/``skeleton``
and per-annotation ``keypoints``/``num_keypoints``, and YOLO appends ``x y v`` triples to the detection
row and declares ``kpt_shape`` plus a horizontal-flip mapping ``flip_idx`` in ``data.yaml``. An
annotation that carries no landmarks — one
generated for a different task and then handed to a keypoint writer — is written as an all-zero,
visibility-``0`` ("not labeled") table rather than a short record, so every row and record still
matches the schema the task declares.

"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from PIL import Image

from synth_datasets.core.config import (
    _RESERVED_SPLIT_PREFIX,
    ClassVocabulary,
    OutputFormat,
    Task,
    validate_split_name,
)
from synth_datasets.families.geometry import PIXEL_CENTRE_OFFSET

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from synth_datasets.core.config import ClassEntry
    from synth_datasets.core.keypoints import KeypointSchema
    from synth_datasets.core.sample import Annotation, Sample

_IMAGE_STEM = "img_{index:06d}"


def _covers(entry: ClassEntry, schema: KeypointSchema) -> bool:
    """Return whether ``schema`` can describe the landmarks of the class ``entry`` names.

    :class:`~synth_datasets.core.config.ClassEntry` carries the shape itself, so the test reads it directly
    rather than parsing the class name: does this class name a shape of the run's own keypoint family?

    A ``ClassMode.COLOR`` entry names no shape at all (``entry.shape is None``) yet is still drawn as whichever family
    the run was restricted to, so it is always covered.

    """
    return entry.shape is None or str(entry.shape.value) in schema.shape_values


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive ``[lo, hi]`` range."""
    return max(lo, min(hi, value))


def _clamp_flat(flat: list[float], img_w: float, img_h: float) -> list[float]:
    """Clamp a flat ``[x1, y1, ...]`` coordinate list to the image extent."""
    return [_clamp(v, 0.0, img_w) if i % 2 == 0 else _clamp(v, 0.0, img_h) for i, v in enumerate(flat)]


def _edge_flat(flat: list[float]) -> list[float]:
    """Convert a flat pixel-centre coordinate list to the edge space an exported file uses.

    An :class:`~synth_datasets.core.sample.Annotation` carries outlines, oriented-box corners
    and landmarks in pixel-centre space, because that is the space the point transforms in
    :mod:`~fused_transforms.targets` move them through. ``bbox_xyxy`` stays in edge space for the
    same reason -- that is the space its own transform assumes. A COCO or YOLO file has no room for
    two conventions: its ``segmentation`` ring, its landmarks and its ``bbox`` are read as one
    coordinate system, and that system is edge space, so the point fields are converted back here,
    at the file boundary, and nowhere else.

    Args:
        flat: ``[x1, y1, x2, y2, ...]`` coordinates in pixel-centre space.

    Returns:
        The same list shifted into edge space.

    """
    return [v + PIXEL_CENTRE_OFFSET for v in flat]


def _keypoint_triples(
    ann: Annotation, img_w: float, img_h: float, schema: KeypointSchema
) -> list[tuple[float, float, int]]:
    """Return one ``(x, y, visibility)`` landmark triple per keypoint, clamped to the image.

    Args:
        ann: The annotation to read landmarks from.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        schema: The active run's keypoint schema — its ``names`` order and count are what an
            annotation without landmarks falls back to.

    Returns:
        One triple per name in ``schema.names``, in that order. A visible point is converted from
        the annotation's pixel-centre space to the file's edge space (see :func:`_edge_flat`) and
        clamped to the image extent like every other coordinate field; an invisible one keeps the
        zeroed placeholder coordinates, unshifted, rather than being clamped into a spurious corner
        position — ``(0.0, 0.0)`` there is a flag value, not a location. An annotation without
        landmarks yields the all-zero, "not labeled" table — see the module docstring.

    """
    if ann.keypoints is None:
        return [(0.0, 0.0, 0)] * len(schema.names)
    return [
        (_clamp(x + PIXEL_CENTRE_OFFSET, 0.0, img_w), _clamp(y + PIXEL_CENTRE_OFFSET, 0.0, img_h), visibility)
        if visibility > 0
        else (0.0, 0.0, visibility)
        for x, y, visibility in ann.keypoints
    ]


def _save_image(image: NDArray[Any], path: Path) -> None:
    """Write an RGB ``uint8`` array to ``path`` as JPEG."""
    Image.fromarray(image).save(path, quality=95)


# Strict filesystem metadata. Every check below reads "absent" only from FileNotFoundError or NotADirectoryError and
# lets any other error (permission, I/O) propagate: pathlib's predicates and os.path.lexists turn such errors into
# False on some Python versions, and an unreadable path taken for a missing one would be skipped, overwritten, or
# reported as safe. None of these follow symlinks.


def _lstat(path: Path) -> os.stat_result | None:
    """Return ``path``'s own metadata, or ``None`` only when nothing is there; any other error propagates."""
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _exists(path: Path) -> bool:
    """Return whether ``path`` is there, a dangling symlink included."""
    return _lstat(path) is not None


def _is_dir(path: Path) -> bool:
    """Return whether ``path`` is a real directory (a symlink to one is not)."""
    info = _lstat(path)
    return info is not None and stat.S_ISDIR(info.st_mode)


def _is_file(path: Path) -> bool:
    """Return whether ``path`` is a regular file (a symlink to one is not)."""
    info = _lstat(path)
    return info is not None and stat.S_ISREG(info.st_mode)


def _is_symlink(path: Path) -> bool:
    """Return whether ``path`` is a symlink."""
    info = _lstat(path)
    return info is not None and stat.S_ISLNK(info.st_mode)


def _iterdir(path: Path) -> list[Path]:
    """Return ``path``'s entries in name order; every error, a missing directory included, propagates."""
    with os.scandir(path) as entries:
        return sorted(path / entry.name for entry in entries)


def _walk(path: Path) -> list[tuple[Path, bool]]:
    """Return every entry below ``path`` with whether it is a real directory; any traversal error propagates.

    Unlike ``os.walk`` or ``Path.rglob``, nothing unreadable is skipped, so an empty result proves emptiness.

    """
    found: list[tuple[Path, bool]] = []
    pending = [path]
    while pending:
        current = pending.pop()
        for entry in _iterdir(current):
            is_dir = _is_dir(entry)
            found.append((entry, is_dir))
            if is_dir:
                pending.append(entry)
    return found


def _is_populated(path: Path) -> bool:
    """Return whether ``path`` holds anything a new dataset would mix with.

    A directory counts only once it has an entry, so an empty split directory made ahead of time never blocks a run.
    Anything else that exists — a file, or a symlink, which is judged as itself and never followed — counts.

    """
    if _is_dir(path):
        return bool(_iterdir(path))
    return _exists(path)


#: The file, at the output root, in which a writer records the paths it wrote. It is the only evidence
#: ``overwrite`` accepts that a path belongs to an earlier run: a file name such as ``_annotations.coco.json``
#: is not, since a hand-curated dataset beside the output carries the same one.
MANIFEST_NAME = ".vision-synth.json"

#: Name prefix of the directory, inside the output directory, that a replacing write stages the new dataset in. Split
#: names starting with ``.vision-synth`` are refused, so neither this nor the manifest can collide with a split.
STAGING_PREFIX = ".vision-synth-staging-"

#: Name prefix of the directory, inside the output directory, that a replacing write moves the earlier dataset's
#: paths into while it swaps the new one in. One left behind by an interrupted run holds that earlier dataset.
BACKUP_PREFIX = ".vision-synth-backup-"

#: Name prefix a staging or backup directory is renamed to once the swap is verified complete — the commit point.
#: One left behind holds only the superseded earlier dataset (or a staging directory's empty folders).
DISCARD_PREFIX = ".vision-synth-discard-"

#: The ownership marker written into every staging and backup directory this tool creates. A reserved-looking
#: directory without a valid one was not created by vision-synth, and is never described as disposable.
OWNER_NAME = ".vision-synth-owner.json"

#: Where a refusal or rollback error points the user for manual recovery steps.
_RECOVERY_DOCS = "See 'Recovering from an interrupted overwrite' in the annotation-formats docs"

#: At most this many recorded paths are spelled out in a refusal; past it, the message points at the marker.
_LISTED_PATHS = 10


def _tool_version() -> str:
    """Return the installed vision-synth version, recorded in ownership markers."""
    try:
        return version("vision-synth")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _write_json_atomically(path: Path, doc: dict[str, Any]) -> None:
    """Write ``doc`` to ``path`` through a temporary sibling and one rename, so a reader never sees half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _make_owned_dir(root: Path, prefix: str, kind: str, **fields: object) -> Path:
    """Create a fresh ``<prefix><token>`` directory in ``root`` and mark it as this tool's with an ownership marker."""
    token = secrets.token_hex(8)
    path = root / f"{prefix}{token}"
    path.mkdir()
    doc = {"kind": kind, "token": token, "version": _tool_version(), "output_dir": str(root.resolve()), **fields}
    try:
        _write_json_atomically(path / OWNER_NAME, doc)
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)  # created by this call and still empty of anything else
        raise
    return path


def _to_discard(root: Path, path: Path, prefix: str) -> Path:
    """Rename the owned directory ``path`` to its ``.vision-synth-discard-<token>`` name and return the new path."""
    discard = root / f"{DISCARD_PREFIX}{path.name[len(prefix) :]}"
    os.replace(path, discard)
    return discard


def _remove_owned(path: Path) -> None:
    """Remove an owned directory's contents, then its marker, then itself, so it stays identified until empty."""
    for child in _iterdir(path):
        if child.name == OWNER_NAME:
            continue
        if _is_dir(child):
            shutil.rmtree(child)
        else:
            child.unlink()
    (path / OWNER_NAME).unlink()
    path.rmdir()


@dataclass(frozen=True)
class _Marker:
    """What reading a reserved directory's ownership marker found.

    Attributes:
        doc: The marker, when it is present, readable and valid for the directory.
        unreadable: Why it could not be read, when an ``OSError`` stopped the read. Ownership is then unknown, which is
            neither "created by vision-synth" nor "not".

    A marker that is absent, or parsed but wrong, leaves both unset: the directory was not created by vision-synth.

    """

    doc: dict[str, Any] | None = None
    unreadable: str | None = None


def _owner_marker(path: Path, prefix: str) -> _Marker:
    """Read the ownership marker of the reserved directory ``path`` named with ``prefix``; never raises ``OSError``.

    Valid means: a regular JSON file whose ``kind`` fits the prefix (a discard directory may hold either kind) and
    whose ``token`` is exactly the directory's name after the prefix.

    """
    marker = path / OWNER_NAME
    try:
        if not _is_dir(path) or not _is_file(marker):
            return _Marker()
        with open(marker, encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError:
        return _Marker()
    except OSError as err:
        return _Marker(unreadable=str(err))
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return _Marker()
    kinds = {STAGING_PREFIX: ("staging",), BACKUP_PREFIX: ("backup",), DISCARD_PREFIX: ("staging", "backup")}[prefix]
    valid = isinstance(doc, dict) and doc.get("kind") in kinds and doc.get("token") == path.name[len(prefix) :]
    return _Marker(doc=doc if valid else None)


def _listing(paths: list[str], key: str, marker: Path) -> str:
    """Spell out ``paths`` for a message, or count them and point at the ``marker`` key that records them all."""
    if len(paths) > _LISTED_PATHS:
        shown = ", ".join(paths[:_LISTED_PATHS])
        return f"{shown}, and {len(paths) - _LISTED_PATHS} more of the '{key}' paths in {marker} (same rule)"
    return ", ".join(paths)


@dataclass(frozen=True)
class _RestorePlan:
    """What restoring the earlier dataset from a backup takes, path by path, as found on disk now.

    Attributes:
        replace: Replaced paths the backup holds a copy of, with something at the path now — the interrupted run's
            version: remove it, then move the backup copy there.
        move: Replaced paths the backup holds a copy of, with nothing at the path: move the backup copy there.
        keep: Replaced paths the backup holds no copy of: the original was never moved and is still in place.
        remove: Added paths present now with no backup copy — only the interrupted run can have put them there.
        suspect: Added paths the backup does hold a copy of, which should not happen: left alone and reported.
        unknown: Recorded paths whose state could not be read (a permission or I/O error): no step is guessed.

    """

    replace: list[str]
    move: list[str]
    keep: list[str]
    remove: list[str]
    suspect: list[str]
    unknown: list[str]


def _restore_plan(backup: Path, marker: dict[str, Any]) -> _RestorePlan:
    """Return the per-path steps that restore the earlier dataset from ``backup``, judged from disk right now.

    A replaced path is only ever replaced when the backup holds its copy: otherwise the one at the path is the
    original, never moved, and deleting it would lose it. An added path is only ever removed when present and not
    backed up, since only the interrupted run can have put it there.

    """
    root = backup.parent
    plan = _RestorePlan(replace=[], move=[], keep=[], remove=[], suspect=[], unknown=[])
    for key in ("replaced", "added"):
        for rel in (rel for rel in marker.get(key, []) if isinstance(rel, str)):
            try:
                in_backup, in_place = _exists(backup / rel), _exists(root / rel)
            except OSError:
                plan.unknown.append(rel)
                continue
            _plan_path(plan, key, rel, in_backup=in_backup, in_place=in_place)
    return plan


def _plan_path(plan: _RestorePlan, key: str, rel: str, *, in_backup: bool, in_place: bool) -> None:
    """File one recorded path into its :class:`_RestorePlan` group from whether it is in the backup and in place."""
    if key == "added":
        if in_backup:
            plan.suspect.append(rel)
        elif in_place:
            plan.remove.append(rel)
    elif not in_backup:
        plan.keep.append(rel)
    else:
        (plan.replace if in_place else plan.move).append(rel)


def _backup_advice(backup: Path, marker: dict[str, Any]) -> str:
    """Return the restore steps for ``backup`` as one sentence per non-empty group of :func:`_restore_plan`."""
    plan = _restore_plan(backup, marker)
    record = backup / OWNER_NAME
    groups = [
        (
            plan.replace,
            "replaced",
            f"remove the interrupted run's version, then move the copy from {backup.name}/ there",
        ),
        (plan.move, "replaced", f"move the copy from {backup.name}/ there"),
        (plan.keep, "replaced", "original still in place (no backup copy): leave it"),
        (plan.remove, "added", "added by the interrupted run: remove it"),
        (plan.suspect, "added", "recorded as added yet backed up, which should not happen: leave it and report it"),
        (plan.unknown, "replaced", "state unknown (its metadata could not be read): leave it and check it by hand"),
    ]
    steps = [f"{action}: {_listing(paths, key, record)}" for paths, key, action in groups if paths]
    return (
        f"{backup.name}: the overwrite did not commit, and the backup holds what was moved of the earlier dataset. "
        "Replace a path only when the backup holds its copy; otherwise the original is still in place. Path by path, "
        f"as found now — {'; '.join(steps) or 'nothing to do'}"
    )


def _leftover_advice(entry: Path, prefix: str, backup_found: bool) -> str:
    """Return what ``entry``, a reserved-prefix directory in the output root, is and the safe thing to do with it."""
    marker = _owner_marker(entry, prefix)
    if marker.unreadable is not None:
        return (
            f"{entry.name}: ownership unknown (marker unreadable: {marker.unreadable}); do nothing to this directory "
            "until its marker can be read, then run again to learn what it is"
        )
    if marker.doc is None:
        return (
            f"{entry.name} was not created by vision-synth (it has no valid {OWNER_NAME}): rename it or move it out "
            "of the output directory"
        )
    if prefix == DISCARD_PREFIX:
        return (
            f"{entry.name}: the overwrite completed and the new dataset is in place; this holds only the superseded "
            "earlier dataset (or empty staging folders), safe to delete when no longer needed"
        )
    if prefix == STAGING_PREFIX:
        tail = " once the earlier dataset is restored from the backup" if backup_found else ""
        return f"{entry.name}: only generated files that were never moved into place; safe to delete{tail}"
    return _backup_advice(entry, marker.doc)


def _refuse_leftovers(root: Path) -> None:
    """Refuse to write while ``root`` holds a staging, backup or discard directory, saying what each one is.

    Nothing is removed here, and each directory is classified from its prefix and its ownership marker rather than
    guessed from its name: a directory without a valid marker was not created by vision-synth and is only ever
    advised to be renamed or moved.

    Raises:
        ValueError: If a direct entry of ``root`` is named with a staging, backup or discard prefix, in any case.

    """
    try:
        entries = _iterdir(root)  # follows a symlinked output directory, as writing into it does
    except (FileNotFoundError, NotADirectoryError):
        return
    prefixes = (STAGING_PREFIX, BACKUP_PREFIX, DISCARD_PREFIX)
    found = sorted(
        (entry, next(prefix for prefix in prefixes if entry.name.casefold().startswith(prefix)))
        for entry in entries
        if entry.name.casefold().startswith(prefixes)
    )
    if found:
        backup_found = any(
            prefix == BACKUP_PREFIX and _owner_marker(entry, prefix).doc is not None for entry, prefix in found
        )
        advice = "; ".join(_leftover_advice(entry, prefix, backup_found) for entry, prefix in found)
        raise ValueError(
            f"refusing to write into {root}: it holds directories an interrupted overwrite may have left. {advice}. "
            f"Then run again. {_RECOVERY_DOCS}"
        )


def _refuse_reserved(root: Path, paths: Iterable[Path]) -> None:
    """Refuse if any path to replace has a component named with the reserved ``.vision-synth`` prefix, in any case.

    Such a path could be this tool's own staging or backup directory, or a split an older release let through before
    the prefix was reserved; acting on it could move or remove the wrong thing.

    Raises:
        ValueError: If a path under ``root`` has such a component.

    """
    for path in paths:
        if any(part.casefold().startswith(_RESERVED_SPLIT_PREFIX) for part in path.relative_to(root).parts):
            raise ValueError(
                f"refusing to replace {path}: names starting with {_RESERVED_SPLIT_PREFIX!r} are reserved for the "
                "manifest and the staging, backup and discard directories; move it out of the way by hand, then run "
                "again"
            )


def _read_manifest(root: Path) -> tuple[str, ...]:
    """Return the relative paths an earlier run recorded in ``root``'s manifest, or none when there is no manifest.

    Raises:
        ValueError: If the manifest is a symlink or anything but a regular file (a directory, say), or is not a JSON
            object whose ``paths`` is a list of non-empty strings. A manifest that cannot be read in full is refused
            rather than half-trusted, and one that could not be written back is refused before any work starts.

    """
    path = root / MANIFEST_NAME
    try:
        info = _lstat(path)
    except OSError as err:
        # Not absence: an unreadable manifest may list splits this run would otherwise stop owning.
        raise ValueError(f"cannot check the dataset manifest {path}: {err}; fix access to it") from err
    if info is None:
        return ()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"refusing to use {path}: the dataset manifest is a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"refusing to use {path}: the dataset manifest is not a regular file; move it out of the way")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"cannot read the dataset manifest {path}: {err}; fix or delete it") from err
    paths = doc.get("paths") if isinstance(doc, dict) else None
    if not isinstance(paths, list) or not all(isinstance(rel, str) and rel for rel in paths):
        raise ValueError(f"the dataset manifest {path} has no list of relative 'paths'; fix or delete it")
    for rel in paths:
        # Only the normalized relative form a writer records is trusted: ``.``, ``..``, an empty component, an absolute
        # path or a Windows separator could each name the output root itself or something outside it.
        if rel.startswith("/") or "\\" in rel or ":" in rel or any(part in ("", ".", "..") for part in rel.split("/")):
            raise ValueError(
                f"the dataset manifest {path} lists {rel!r}, which is not a normalized relative path inside the "
                "output directory; fix or delete it"
            )
    return tuple(paths)


def _check_replaceable(root: Path, targets: Iterable[Path]) -> None:
    """Refuse unless moving every one of ``targets`` out stays inside ``root`` and never goes through a symlink.

    All targets are checked before any is moved, so a refusal leaves the output directory exactly as it was.

    Raises:
        ValueError: If a target resolves outside ``root`` or to ``root`` itself, or a directory between ``root`` and a
            target is a symlink.

    """
    resolved_root = Path(os.path.realpath(root, strict=True))
    for target in targets:
        parts = target.relative_to(root).parts
        current = root
        for part in parts[:-1]:
            current = current / part
            if _is_symlink(current):
                raise ValueError(
                    f"refusing to replace {target}: {current} is a symlink, which could lead outside {root}"
                )
        # With no symlink above it, only the target itself can lead elsewhere; strict, so no error reads as a path.
        resolved = resolved_root.joinpath(*parts)
        if _is_symlink(target):
            try:
                resolved = Path(os.path.realpath(target, strict=True))
            except FileNotFoundError as err:
                raise ValueError(f"refusing to replace {target}: it is a symlink whose target is missing") from err
        if not parts or resolved == resolved_root or not resolved.is_relative_to(resolved_root):
            raise ValueError(f"refusing to replace {target}: it resolves outside the output directory {root}, or to it")


def _plan_moves(
    staging: Path, root: Path, replaced: set[Path], listings: _Listings | None = None
) -> list[tuple[Path, Path]]:
    """Return the ``(staged, destination)`` renames that put a staged dataset in place once ``replaced`` is gone.

    A staged directory whose destination is a real directory that survives is merged entry by entry; anything else
    moves whole, onto a destination that is absent or being replaced — never over a surviving file, which this run
    does not own. Planned before anything moves, so a refusal leaves the earlier dataset intact.

    Raises:
        ValueError: If a staged entry would land on a surviving entry that is not a directory it can merge into.

    """
    moves: list[tuple[Path, Path]] = []
    pending = [(staging, root)]
    while pending:
        src_dir, dst_dir = pending.pop()
        for src in _iterdir(src_dir):
            if src_dir == staging and src.name == OWNER_NAME:
                continue  # the staging directory's own marker is never promoted
            dst = dst_dir / src.name
            # Judged by the entry the filesystem finds, so a respelled alias of a replaced path counts as replaced.
            stored = _on_disk(root, dst, listings)
            if stored is not None and stored in replaced:
                stored = None  # being replaced, so free to move onto
            if stored is None:
                moves.append((src, dst))
            elif _is_dir(src) and _is_dir(stored):
                pending.append((src, stored))
            else:
                raise ValueError(
                    f"refusing to replace {dst}: it is not this writer's, and the new {src.name!r} cannot be moved "
                    "over it; move it out of the way"
                )
    return moves


class _Listings:
    """Directory listings read once per planning phase, so planning costs one read per directory, not per path.

    Planning runs before anything moves, so what a directory held when first read is what it still holds. The inode
    index, needed only to resolve an alias, is built on first use.

    """

    def __init__(self) -> None:
        self._names: dict[Path, frozenset[str] | None] = {}
        self._inodes: dict[Path, dict[tuple[int, int], list[str]]] = {}

    def names(self, parent: Path) -> frozenset[str] | None:
        """Return the names ``parent`` lists, or ``None`` when it does not exist or is not a directory.

        Any other failure to read it (a permission or I/O error) propagates: reading an unreadable directory as empty
        would plan its files as new paths and skip their backup.

        """
        if parent not in self._names:
            try:
                self._names[parent] = frozenset(os.listdir(parent))
            except (FileNotFoundError, NotADirectoryError):
                self._names[parent] = None
        return self._names[parent]

    def by_inode(self, parent: Path) -> dict[tuple[int, int], list[str]]:
        """Return ``parent``'s entries grouped by ``(st_dev, st_ino)``, in name order."""
        if parent not in self._inodes:
            index: dict[tuple[int, int], list[str]] = {}
            for name in sorted(self.names(parent) or ()):
                stat = os.lstat(parent / name)
                index.setdefault((stat.st_dev, stat.st_ino), []).append(name)
            self._inodes[parent] = index
        return self._inodes[parent]


def _stored_name(parent: Path, part: str, listings: _Listings | None = None) -> str | None:
    """Return the name ``parent / part`` is stored under, or ``None`` when nothing is there.

    That is ``part`` itself, unless the filesystem matched it to an entry spelled differently — a case-insensitive one
    finds ``train`` under ``Train``, and a normalizing one ``é`` under its decomposed form. The alias is read from the
    filesystem (the listed entry with the same inode), never assumed from the names, so a case-sensitive filesystem
    keeps ``Train`` and ``train`` apart. ``listings`` shares directory reads across one planning phase.

    """
    listings = listings if listings is not None else _Listings()
    names = listings.names(parent)
    if names is None:
        return None
    if part in names:
        return part
    try:
        wanted = os.lstat(parent / part)
    except (FileNotFoundError, NotADirectoryError):
        return None
    same = listings.by_inode(parent).get((wanted.st_dev, wanted.st_ino), [])
    folded = unicodedata.normalize("NFC", part).casefold()
    spelled = [name for name in same if unicodedata.normalize("NFC", name).casefold() == folded]
    return (spelled or same or [part])[0]


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _on_disk(root: Path, path: Path, listings: _Listings | None = None) -> Path | None:
    """Return ``path`` under ``root`` spelled as the filesystem stores it, or ``None`` if it does not exist."""
    current = root
    for part in path.relative_to(root).parts:
        name = _stored_name(current, part, listings)
        if name is None:
            return None
        current = current / name
    return current


def _outermost_present(root: Path, paths: Iterable[Path], listings: _Listings | None = None) -> list[Path]:
    """Return the ``paths`` that exist, as stored, with aliases and duplicates merged and any under another dropped.

    Two spellings the filesystem resolves to one entry (``images/train`` and ``images/Train`` on a case-insensitive one)
    are one path to replace, so it is backed up once.

    """
    stored = (_on_disk(root, path, listings) for path in paths)
    present = list(dict.fromkeys(path for path in stored if path is not None))
    # A set lookup per ancestor, not a scan of every other path: linear in paths times depth.
    kept = set(present)
    return [path for path in present if kept.isdisjoint(path.parents)]


def _swap(root: Path, targets: list[Path], moves: list[tuple[Path, Path]], backup: Path, staging: Path) -> None:
    """Rename ``targets`` into ``backup``, then each staged entry into place; on any exception, roll back from disk.

    The rollback reads where each entry is now instead of trusting a list kept while renaming, so an interrupt that
    lands between a rename and its bookkeeping cannot hide a moved path. It only ever renames, and never deletes a
    file from the backup: once a re-listing shows every earlier path back in place, the emptied backup directory is
    removed with ``os.rmdir`` alone; otherwise it is kept and the error names it. The staging directory, which holds
    only new files, is removed only once every earlier path is found either back in place or in the backup.

    Raises:
        RuntimeError: From the ``Exception`` that stopped the swap, naming ``backup`` and saying whether the earlier
            dataset was fully restored.
        BaseException: A ``KeyboardInterrupt``, ``SystemExit`` or similar is re-raised as it is, after the rollback.

    """
    try:
        for target in targets:
            saved = backup / target.relative_to(root)
            saved.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, saved)
        for src, dst in moves:
            os.replace(src, dst)
    except BaseException as err:
        try:
            _roll_back(root, targets, moves, backup)
            # Re-listed, not remembered: an interrupt may have landed between a rename and anything that tracked it.
            copies = [backup / target.relative_to(root) for target in targets]
            accounted = all(_exists(target) or _exists(copy) for target, copy in zip(targets, copies, strict=True))
            restored = all(_exists(target) and not _exists(copy) for target, copy in zip(targets, copies, strict=True))
        except BaseException as rollback_err:
            # Unknown state (a failed rename, or a path whose metadata cannot be read) keeps everything where it is.
            _raise_naming(
                rollback_err, f"replacing the dataset in {root} failed ({err!r}) and so did the rollback", backup
            )
        if not _prune_owned(staging) and accounted:
            shutil.rmtree(staging, ignore_errors=True)
        # A complete restore leaves the backup holding only its marker and the directories it was built with; the
        # marker goes and os.rmdir removes those, refusing anything else, so a backup that still holds a file is kept.
        kept = None if restored and _prune_owned(backup) else backup
        state = "is back in place" if restored else "is NOT fully back in place"
        _raise_naming(err, f"replacing the dataset in {root} failed ({err!r}); the earlier dataset {state}", kept)


def _roll_back(root: Path, targets: list[Path], moves: list[tuple[Path, Path]], backup: Path) -> None:
    """Undo whatever part of a swap happened, judged from disk: promoted entries go back, backed-up paths return."""
    for src, dst in reversed(moves):
        if _exists(dst) and not _exists(src):
            os.replace(dst, src)
    for target in reversed(targets):
        saved = backup / target.relative_to(root)
        if _exists(saved) and not _exists(target):
            os.replace(saved, target)


def _prune_owned(path: Path) -> bool:
    """Remove the owned directory ``path`` if it holds nothing but its marker and empty directories.

    The marker is unlinked only after a strict walk of the whole directory — one that stops on any unreadable entry
    instead of skipping it — shows it is the one file left; the directories then go by ``os.rmdir`` alone. Any error
    keeps everything and returns ``False``.

    """
    try:
        entries = _walk(path)
    except OSError:
        return False
    if any(not is_dir and entry != path / OWNER_NAME for entry, is_dir in entries):
        return False
    try:
        (path / OWNER_NAME).unlink(missing_ok=True)
    except OSError:
        return False
    return _prune_empty(path, [entry for entry, is_dir in entries if is_dir])


def _prune_empty(path: Path, directories: list[Path]) -> bool:
    """Remove ``directories`` (all below ``path``) deepest first, then ``path``; return whether ``path`` is gone.

    Uses ``os.rmdir`` alone, which refuses a non-empty directory, so it can never delete a file.

    """
    try:
        for directory in sorted(directories, key=lambda entry: len(entry.parts), reverse=True):
            os.rmdir(directory)
        os.rmdir(path)
        return not _exists(path)
    except OSError:
        return False


def _raise_naming(err: BaseException, message: str, backup: Path | None) -> NoReturn:
    """Re-raise ``err`` so the error names the kept ``backup``, if one was kept.

    An ``Exception`` becomes a ``RuntimeError`` carrying the message, chained to it. Anything else — a
    ``KeyboardInterrupt``, a ``SystemExit`` — is re-raised as it is, so it keeps meaning what it means, with the message
    attached as a note where the interpreter supports notes.

    """
    marker = _owner_marker(backup, BACKUP_PREFIX) if backup is not None else _Marker()
    if backup is not None:
        steps = _backup_advice(backup, marker.doc) if marker.doc is not None else f"see {backup / OWNER_NAME}"
        message = f"{message}. The backup is kept at {backup}. {steps}"
    if isinstance(err, Exception):
        raise RuntimeError(message) from err
    if hasattr(err, "add_note"):
        err.add_note(message)
    raise err


#: The one annotation file a COCO split directory holds.
_COCO_JSON = "_annotations.coco.json"


class DatasetWriter(ABC):
    """Base class for dataset serializers.

    Args:
        task: The annotation task determining which fields are emitted.
        vocabulary: The classes to declare, in id order — each entry keeps the shape and color it
            was derived from, which is how a writer tells which categories its keypoint schema
            covers without parsing their names.
        keypoint_schema: The keypoint family a :attr:`~synth_datasets.core.config.Task.KEYPOINTS`
            run draws from; required for that task and ignored for every other. It used to default
            to the animal schema for backward compatibility, which meant a directly-constructed
            symbol or letter pose writer silently emitted a 16-landmark animal header over 7- or
            15-landmark rows. There is no safe default, so there is none.

    """

    def __init__(self, task: Task, vocabulary: ClassVocabulary, keypoint_schema: KeypointSchema | None = None) -> None:
        """Store the task and vocabulary, rejecting a keypoints task with no schema to write.

        Raises:
            ValueError: If ``task`` is :attr:`~synth_datasets.core.config.Task.KEYPOINTS` and no
                ``keypoint_schema`` was given.

        """
        if task is Task.KEYPOINTS and keypoint_schema is None:
            raise ValueError(
                "Task.KEYPOINTS needs a keypoint_schema naming the family being written; pass the "
                "one keypoint_schema_for(config.shapes) returns"
            )
        self.task = task
        self.vocabulary = vocabulary
        self.class_names = vocabulary.names
        self.keypoint_schema = keypoint_schema

    @property
    def schema(self) -> KeypointSchema:
        """Return the keypoint schema, which the constructor guarantees for a keypoints task.

        Every landmark-writing path reaches the schema through here rather than through the
        optional attribute, so the "a keypoints writer always has one" invariant is stated once and
        checked, instead of being asserted implicitly at four call sites.

        Raises:
            ValueError: If no schema was supplied — only reachable by mutating the attribute after
                construction, since the constructor rejects a keypoints task without one.

        """
        if self.keypoint_schema is None:
            raise ValueError("this writer has no keypoint schema; it was not built for Task.KEYPOINTS")
        return self.keypoint_schema

    def owned_paths(self, split_names: Iterable[str], output_dir: Path) -> tuple[Path, ...]:
        """Return the paths under ``output_dir`` this writer creates for ``split_names`` — a subclass hook.

        :meth:`prepare_output` refuses to write while any of them is populated, and :meth:`write_replacing` swaps
        them out, together with whatever an earlier run recorded in the :data:`MANIFEST_NAME` manifest and nothing
        else under ``output_dir``. The base declares none, so a third-party writer that does not override this is
        neither blocked nor replaced: it keeps whatever behavior it had.

        Args:
            split_names: The splits about to be written, each already checked to be one plain path component.
            output_dir: The destination root.

        Returns:
            Every file or directory the writer produces for those splits.

        """
        return ()

    def record_output(self, split_names: Iterable[str], output_dir: str | Path) -> None:
        """Record in ``output_dir``'s :data:`MANIFEST_NAME` manifest the paths this writer now owns there.

        The built-in writers call this once a write completes. The manifest keeps an earlier run's entries that still
        exist (a split this run left alone is still that tool's own) and adds this run's :meth:`owned_paths`, so a
        later ``overwrite`` can remove a split this run did not write without guessing from file names.

        Args:
            split_names: The splits just written.
            output_dir: The destination root.

        """
        root = Path(output_dir)
        names = list(split_names)
        kept = [rel for rel in _read_manifest(root) if _exists(root / rel)]
        owned = [path.relative_to(root).as_posix() for path in self.owned_paths(names, root)]
        fmt = next((key for key, cls in _WRITERS.items() if cls is type(self)), type(self).__name__)
        doc = {"format": fmt, "splits": names, "paths": list(dict.fromkeys([*kept, *owned]))}
        (root / MANIFEST_NAME).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def prepare_output(self, split_names: Iterable[str], output_dir: str | Path) -> None:
        """Make sure writing ``split_names`` under ``output_dir`` cannot mix two datasets; it never deletes anything.

        Refuses while any of this run's :meth:`owned_paths` already holds files, since writing ``img_000000..`` over
        a larger earlier run would keep its higher-numbered files beside the new ones. The built-in writers call this
        before consuming a single sample; to replace an earlier dataset, use :meth:`write_replacing`.

        Args:
            split_names: The splits about to be written.
            output_dir: The destination root.

        Raises:
            ValueError: If a split name is not one plain path component (see
                :func:`~synth_datasets.core.config.validate_split_name`), if the manifest cannot be read, or if
                ``output_dir`` holds a staging or backup directory an interrupted overwrite left.
            FileExistsError: If an owned path is a file or a non-empty directory.

        """
        names = [validate_split_name(name) for name in split_names]
        root = Path(output_dir)
        owned = self.owned_paths(names, root)
        _read_manifest(root)
        _refuse_leftovers(root)
        blocking = [path.relative_to(root).as_posix() for path in owned if _is_populated(path)]
        if blocking:
            raise FileExistsError(
                f"refusing to write into {root} ({', '.join(blocking)}): an earlier dataset is already there "
                "and the two runs' files would mix; pass overwrite=True (to generate_dataset, or call "
                "write_replacing instead of write) to replace this writer's files, leaving everything else under the "
                "output directory alone, or choose an empty output directory"
            )

    def write_replacing(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write ``splits`` over an earlier dataset under ``output_dir``, removing it only once the new one is done.

        What ``overwrite=True`` runs. The new dataset is written by :meth:`write` into a fresh
        ``.vision-synth-staging-*`` directory inside ``output_dir``. Only after that succeeds are the paths to replace —
        this run's :meth:`owned_paths`, every path the earlier :data:`MANIFEST_NAME` manifest lists, and the manifest
        itself — checked (inside ``output_dir``, never through a symlink) and swapped: each is renamed into a fresh
        ``.vision-synth-backup-*`` directory, then each staged entry is renamed into place, all on one filesystem.
        Only after every staged entry is confirmed at its target does the overwrite commit, by renaming the backup and
        staging directories to ``.vision-synth-discard-*``; they are removed after that. Both carry a
        :data:`OWNER_NAME` marker, and the backup's records the relative paths the swap replaces and adds.
        Any exception during the swap, ``KeyboardInterrupt`` included, rolls it back by renaming only, reading the
        state from disk. If a re-listing shows the earlier dataset fully back, the emptied backup directory is removed
        with ``os.rmdir`` alone; otherwise it is kept, whatever it holds, and the error names it. A staging or backup
        directory left by an interrupted run makes this refuse (see :func:`_refuse_leftovers`); it is never removed
        automatically.

        Args:
            splits: Mapping of split name to its samples; consumed as :meth:`write` documents.
            output_dir: The destination root (created if absent).

        Raises:
            ValueError: If a split name is not one plain path component, the manifest cannot be read, an interrupted
                run left a staging or backup directory, a path to replace has a reserved ``.vision-synth`` name,
                resolves outside ``output_dir`` or lies under a symlinked directory, or a staged entry cannot be moved
                without replacing what survives. Nothing under ``output_dir`` has changed when it is raised.
            RuntimeError: If the swap fails (after its rollback; the message names the kept backup and says whether
                the earlier dataset is fully back), or the new dataset is in place but a leftover cannot be removed.

        """
        root = Path(output_dir)
        names = [validate_split_name(name) for name in splits]
        recorded = _read_manifest(root)
        _refuse_leftovers(root)
        candidates = [*self.owned_paths(names, root), *(root / rel for rel in recorded), root / MANIFEST_NAME]
        _refuse_reserved(root, candidates[:-1])
        root.mkdir(parents=True, exist_ok=True)
        staging: Path | None = None
        backup: Path | None = None
        try:
            staging = _make_owned_dir(root, STAGING_PREFIX, "staging")
            self.write(splits, staging)
            listings = _Listings()  # one read per directory for the whole planning phase
            targets = _outermost_present(root, candidates, listings)
            _check_replaceable(root, targets)
            moves = _plan_moves(staging, root, set(targets), listings)
            # Recorded before anything moves: what the swap replaces, and — the only paths recovery may ever tell
            # anyone to remove — the destinations that do not exist yet, which the swap will add.
            replaced = [target.relative_to(root).as_posix() for target in targets]
            added = [dst.relative_to(root).as_posix() for _, dst in moves if _on_disk(root, dst, listings) is None]
            backup = _make_owned_dir(root, BACKUP_PREFIX, "backup", replaced=replaced, added=added)
        except BaseException:
            # Nothing has moved yet: only directories this call created are removed.
            for created in (staging, backup):
                if created is not None:
                    shutil.rmtree(created, ignore_errors=True)
            raise
        _swap(root, targets, moves, backup, staging)
        try:
            missing = [dst for src, dst in moves if not _exists(dst) or _exists(src)]
        except OSError as err:
            # The swap cannot be verified, so it did not commit: keep both directories and name the backup.
            _raise_naming(err, f"replacing the dataset in {root} could not be verified", backup)
        if missing:
            steps = _backup_advice(backup, _owner_marker(backup, BACKUP_PREFIX).doc or {})
            raise RuntimeError(
                f"after replacing the dataset in {root}, {missing[0]} is not in place; the overwrite did not commit "
                f"and the earlier dataset is kept in {backup}. {steps}. {_RECOVERY_DOCS}"
            )
        # Commit point: both directories are renamed to discard before anything is removed, so from here on a
        # leftover can only be read as "the new dataset is complete; this is the superseded copy".
        discards = [_to_discard(root, backup, BACKUP_PREFIX), _to_discard(root, staging, STAGING_PREFIX)]
        try:
            for discard in discards:
                _remove_owned(discard)
        except OSError as err:
            raise RuntimeError(
                f"the new dataset is in place in {root}, but removing a superseded copy failed ({err}); the "
                f"{DISCARD_PREFIX}* directories left hold only that copy and are safe to delete. {_RECOVERY_DOCS}"
            ) from err

    @abstractmethod
    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write all splits under ``output_dir``.

        **Consume each split exactly once, in the order given.** The splits
        :func:`~synth_datasets.generate_dataset` passes are lazy views over a *single*
        shared sample stream, so iterating them out of order, twice, or partially does not merely
        repeat work — it silently redistributes samples between splits or empties them. An
        implementation that needs a split more than once must materialize it itself, accepting the
        memory that costs.

        Args:
            splits: Mapping of split name to its samples, in the order they must be consumed.
            output_dir: Destination root directory (created if absent).

        """


class CocoWriter(DatasetWriter):
    """Write a COCO-format dataset, one JSON per split.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, Task, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.export.writers import CocoWriter
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,))
        >>> CocoWriter(Task.DETECTION, vocab).task.value
        'detection'

        ```

    """

    def owned_paths(self, split_names: Iterable[str], output_dir: Path) -> tuple[Path, ...]:
        """Return each split's ``<output_dir>/<split>/`` directory — images and its JSON both live there."""
        return tuple(output_dir / split for split in split_names)

    def _annotation_dict(self, ann: Annotation, ann_id: int, image_id: int, img_w: int, img_h: int) -> dict[str, Any]:
        """Build one COCO annotation record, clamping geometry to the image extent."""
        x1 = _clamp(ann.bbox_xyxy[0], 0, img_w)
        y1 = _clamp(ann.bbox_xyxy[1], 0, img_h)
        x2 = _clamp(ann.bbox_xyxy[2], 0, img_w)
        y2 = _clamp(ann.bbox_xyxy[3], 0, img_h)
        width, height = x2 - x1, y2 - y1
        record = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann.class_id + 1,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        if self.task is Task.SEGMENTATION:
            record["segmentation"] = [_clamp_flat(_edge_flat(ann.polygon), img_w, img_h)]
        elif self.task is Task.OBB:
            record["segmentation"] = [_clamp_flat(_edge_flat(ann.obb_corners), img_w, img_h)]
        elif self.task is Task.KEYPOINTS:
            record["segmentation"] = [_clamp_flat(_edge_flat(ann.polygon), img_w, img_h)]
            triples = _keypoint_triples(ann, img_w, img_h, self.schema)
            record["keypoints"] = [value for triple in triples for value in triple]
            record["num_keypoints"] = sum(1 for *_, visibility in triples if visibility > 0)
        return record

    def _categories(self) -> list[dict[str, Any]]:
        """Build the category records, adding the keypoint schema to every category it covers.

        Under ``ClassMode.SHAPE`` or ``ClassMode.SHAPE_COLOR`` naming the vocabulary can span shapes
        outside the run's own keypoint family — every primitive-shape category always, plus every
        category of another keypoint-bearing family when one is active. Decorating those with a
        schema they can never produce a matching annotation for would misdescribe the dataset to any
        COCO consumer, so only the categories :func:`_covers` accepts are decorated.

        A category's ``skeleton`` prefers
        :meth:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for` — the letter family's
        per-letter stroke edges — falling back to the family-wide ``skeleton`` for a bare-color
        category or a family whose members all share one topology (animals, symbols).

        """
        categories: list[dict[str, Any]] = [
            {"id": entry.index + 1, "name": entry.name, "supercategory": "none"} for entry in self.vocabulary.entries
        ]
        if self.task is not Task.KEYPOINTS or self.keypoint_schema is None:
            return categories
        schema = self.keypoint_schema
        for entry, category in zip(self.vocabulary.entries, categories, strict=True):
            if not _covers(entry, schema):
                continue
            category["keypoints"] = list(schema.names)
            skeleton = schema.skeleton if entry.shape is None else schema.skeleton_for(str(entry.shape.value))
            # COCO skeleton edges are 1-based indices into the category's own keypoint list.
            category["skeleton"] = [[i + 1, j + 1] for i, j in skeleton]
        return categories

    def _coco_doc(self, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap image and annotation records into a COCO document with categories."""
        categories = self._categories()
        return {
            "info": {"description": "vision-synth synthetic dataset"},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Stream each split to ``<output_dir>/<split>/`` in a single pass over its samples.

        Image pixels are written as they are produced and never held in memory. The COCO schema,
        however, emits one JSON document per split, so lightweight per-image and per-annotation
        metadata records (no pixels) accumulate for the duration of the split and are serialized
        once the split is exhausted: memory is O(n) in the split's image and annotation counts, not
        constant. For a constant-memory path use the YOLO writer (one label file per image) or the
        in-memory :class:`~synth_datasets.export.datasets.SyntheticIterableDataset`.

        Raises:
            ValueError: If a split name is not one plain path component.
            FileExistsError: If a split directory already holds files; see :meth:`prepare_output`.

        """
        output_dir = Path(output_dir)
        self.prepare_output(splits, output_dir)
        for split, samples in splits.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            ann_id = 1
            for image_id, sample in enumerate(samples):
                stem = _IMAGE_STEM.format(index=image_id)
                _save_image(sample.image, split_dir / f"{stem}.jpg")
                images.append({
                    "id": image_id,
                    "file_name": f"{stem}.jpg",
                    "width": sample.width,
                    "height": sample.height,
                })
                for ann in sample.annotations:
                    annotations.append(self._annotation_dict(ann, ann_id, image_id, sample.width, sample.height))
                    ann_id += 1
            doc = self._coco_doc(images, annotations)
            (split_dir / _COCO_JSON).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        self.record_output(splits, output_dir)


class YoloWriter(DatasetWriter):
    """Write a YOLO-format dataset with normalized labels and a ``data.yaml``.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, Task, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.export.writers import YoloWriter
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,))
        >>> YoloWriter(Task.OBB, vocab).task.value
        'obb'

        ```

    """

    def owned_paths(self, split_names: Iterable[str], output_dir: Path) -> tuple[Path, ...]:
        """Return each split's ``images/<split>`` and ``labels/<split>`` directories, plus the shared ``data.yaml``."""
        names = tuple(split_names)
        return (
            *(output_dir / "images" / split for split in names),
            *(output_dir / "labels" / split for split in names),
            output_dir / "data.yaml",
        )

    @staticmethod
    def _box_coords(ann: Annotation, width: int, height: int) -> list[float]:
        """Return the normalized ``cx cy w h`` of the box clipped to the image extent."""
        # Clamp corners to the image extent before deriving cx/cy/w/h so an edge-crossing box's
        # label matches its clipped visible box (consistent with CocoWriter._annotation_dict).
        x1 = _clamp(ann.bbox_xyxy[0], 0.0, width)
        y1 = _clamp(ann.bbox_xyxy[1], 0.0, height)
        x2 = _clamp(ann.bbox_xyxy[2], 0.0, width)
        y2 = _clamp(ann.bbox_xyxy[3], 0.0, height)
        return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]

    def _keypoint_tokens(self, ann: Annotation, width: int, height: int) -> list[str]:
        """Return the trailing ``x y v`` tokens of a pose row, three per landmark.

        Coordinates are normalized and clamped like every other coordinate; the visibility flag is an index into COCO's
        scale, so it is written as a plain integer and never normalized.

        """
        tokens: list[str] = []
        for x, y, visibility in _keypoint_triples(ann, float(width), float(height), self.schema):
            tokens += [f"{_clamp(x / width, 0.0, 1.0):.6f}", f"{_clamp(y / height, 0.0, 1.0):.6f}", str(visibility)]
        return tokens

    def _label_row(self, ann: Annotation, width: int, height: int) -> str:
        """Format one YOLO label row for the writer's task, clamping coordinates to ``[0, 1]``."""
        if self.task in (Task.DETECTION, Task.KEYPOINTS):
            # A pose row is a detection row plus the landmark block (Ultralytics' order).
            coords = self._box_coords(ann, width, height)
        else:
            flat = _edge_flat(ann.polygon if self.task is Task.SEGMENTATION else ann.obb_corners)
            coords = [v / width if i % 2 == 0 else v / height for i, v in enumerate(flat)]
        tokens = [str(ann.class_id), *(f"{_clamp(c, 0.0, 1.0):.6f}" for c in coords)]
        if self.task is Task.KEYPOINTS:
            tokens += self._keypoint_tokens(ann, width, height)
        return " ".join(tokens)

    def _write_split(self, split: str, samples: Iterable[Sample], output_dir: Path) -> None:
        """Write images and label files for one split."""
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            stem = _IMAGE_STEM.format(index=index)
            _save_image(sample.image, img_dir / f"{stem}.jpg")
            rows = [self._label_row(ann, sample.width, sample.height) for ann in sample.annotations]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def _data_yaml(self, splits: dict[str, Iterable[Sample]]) -> str:
        """Build the ``data.yaml`` contents referencing present splits."""
        if not splits:  # defensive backstop; write() is the primary guard
            raise ValueError("YoloWriter requires at least one split, got an empty mapping")
        lines = ["path: .", f"train: images/{'train' if 'train' in splits else next(iter(splits))}"]
        if "val" in splits:
            lines.append("val: images/val")
        if "test" in splits:
            lines.append("test: images/test")
        lines.append(f"nc: {len(self.class_names)}")
        if self.task is Task.KEYPOINTS:
            # Ultralytics carries one dataset-wide (num_keypoints, dims) shape, hence one shared
            # landmark schema for every class; dims is 3 because each point ships its visibility.
            lines.append(f"kpt_shape: [{self.schema.kpt_shape}, 3]")
            # ``flip_idx`` names the landmark each one becomes under a horizontal flip — see
            # KeypointSchema.flip_idx for what makes each family's mapping (identity for animals,
            # a genuine left/right swap for symbols) correct.
            lines.append(f"flip_idx: {list(self.schema.flip_idx)}")
        lines.append("names:")
        lines.extend(f"  {i}: {name}" for i, name in enumerate(self.class_names))
        return "\n".join(lines) + "\n"

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write images, labels, and ``data.yaml`` under ``output_dir``.

        Raises:
            ValueError: If ``splits`` is empty (checked before any output directory is created), or a split name is
                not one plain path component.
            FileExistsError: If a split's image or label directory, or ``data.yaml``, already holds files; see
                :meth:`prepare_output`.

        """
        if not splits:
            raise ValueError("YoloWriter requires at least one split, got an empty mapping")
        output_dir = Path(output_dir)
        self.prepare_output(splits, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, samples in splits.items():
            self._write_split(split, samples, output_dir)
        (output_dir / "data.yaml").write_text(self._data_yaml(splits), encoding="utf-8")
        self.record_output(splits, output_dir)


#: Writer class per output format. A dispatch table rather than an if/else so a third party can add
#: a format (Pascal VOC, CVAT, a house schema) without forking this module — see
#: :func:`register_writer`. The two built-ins register themselves below.
_WRITERS: dict[str, type[DatasetWriter]] = {}


def register_writer(fmt: OutputFormat | str, writer: type[DatasetWriter]) -> None:
    """Register the writer class serving one output format.

    Args:
        fmt: The format key. An :class:`~synth_datasets.core.config.OutputFormat` member for the
            built-ins, or any string for a custom format — :func:`get_writer` accepts both, and so does
            :func:`~synth_datasets.generate_dataset`, so a caller can pass ``fmt="voc"`` straight to it
            once registered.
        writer: A concrete :class:`DatasetWriter` subclass.

    Raises:
        TypeError: If ``writer`` is not a :class:`DatasetWriter` subclass.

    Examples:
        ```pycon
        >>> from synth_datasets.export.writers import YoloWriter, register_writer
        >>> class UltralyticsWriter(YoloWriter):
        ...     pass
        >>> register_writer("ultralytics", UltralyticsWriter)

        ```

    """
    if not (isinstance(writer, type) and issubclass(writer, DatasetWriter)):
        raise TypeError(f"writer must be a DatasetWriter subclass, got {writer!r}")
    _WRITERS[fmt.value if isinstance(fmt, OutputFormat) else str(fmt)] = writer


register_writer(OutputFormat.COCO, CocoWriter)
register_writer(OutputFormat.YOLO, YoloWriter)


def get_writer(
    fmt: OutputFormat | str, task: Task, vocabulary: ClassVocabulary, keypoint_schema: KeypointSchema | None = None
) -> DatasetWriter:
    """Return the writer registered for an output format.

    Args:
        fmt: Target format — an :class:`~synth_datasets.core.config.OutputFormat` member, its
            string value, or any key passed to :func:`register_writer`.
        task: Annotation task to emit.
        vocabulary: The classes to declare, in id order; see :class:`DatasetWriter`.
        keypoint_schema: The keypoint family a :attr:`~synth_datasets.core.config.Task.KEYPOINTS`
            run draws from; see :class:`DatasetWriter`. Required for that task, ignored otherwise.

    Returns:
        A concrete :class:`DatasetWriter`.

    Raises:
        ValueError: If no writer is registered for ``fmt``.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, OutputFormat, Task, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.export.writers import get_writer
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,))
        >>> type(get_writer(OutputFormat.YOLO, Task.DETECTION, vocab)).__name__
        'YoloWriter'

        ```

    """
    key = fmt.value if isinstance(fmt, OutputFormat) else str(fmt)
    writer = _WRITERS.get(key)
    if writer is None:
        raise ValueError(f"no writer registered for format {key!r}; known formats: {sorted(_WRITERS)}")
    return writer(task, vocabulary, keypoint_schema)
