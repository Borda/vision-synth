"""``overwrite=True`` deletes only what this tool demonstrably wrote, and only inside the output directory.

* Ownership is recorded, not inferred: the writers list what they wrote in a ``.vision-synth.json`` manifest at the
  output root, and stale-split cleanup removes only paths that manifest lists. A curated COCO dataset sitting beside the
  output (any directory holding an ``_annotations.coco.json``) used to be deleted as if it were an old split.
* Nothing is deleted through a symlink: a symlinked ``images/`` used to redirect ``rmtree`` outside the output root.
* Every input is validated before anything is deleted: ``seed=-1`` used to fail only once the lazy sample stream was
  iterated, after the earlier dataset was already gone.
* Split names that Windows cannot create (``CON``, ``aux.txt``, a trailing dot or space) are refused up front.
* The command line reports an unknown flag or a wrongly typed value as one line naming the flag.

"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from synth_datasets import (
    DEFAULT_SHAPES,
    ClassMode,
    CocoWriter,
    SplitRatios,
    SyntheticConfig,
    SyntheticGenerator,
    Task,
    YoloWriter,
    class_vocabulary,
    generate_dataset,
)
from synth_datasets.core.config import validate_split_name
from synth_datasets.export import writers

_COMMON = {"img_size": 32, "seed": 0, "task": "detection"}
_MANIFEST = ".vision-synth.json"


def _tree(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*"))


def test_coco_overwrite_keeps_a_sibling_dataset_it_did_not_write(tmp_path):
    out = tmp_path / "ds"
    generate_dataset(out, num_images=4, fmt="coco", **_COMMON)
    curated = out / "curated"
    curated.mkdir()
    (curated / "_annotations.coco.json").write_text("{}", encoding="utf-8")
    (curated / "img_0001.jpg").write_bytes(b"hand-labelled")

    generate_dataset(out, num_images=4, fmt="coco", overwrite=True, **_COMMON)

    assert (curated / "img_0001.jpg").read_bytes() == b"hand-labelled"
    assert (curated / "_annotations.coco.json").is_file()


def test_writers_record_what_they_wrote_in_a_manifest(tmp_path):
    counts = generate_dataset(tmp_path, num_images=4, fmt="yolo", **_COMMON)

    manifest = json.loads((tmp_path / _MANIFEST).read_text(encoding="utf-8"))
    assert manifest["format"] == "yolo"
    assert manifest["splits"] == list(counts)
    assert set(manifest["paths"]) == {
        *(f"images/{split}" for split in counts),
        *(f"labels/{split}" for split in counts),
        "data.yaml",
    }


def test_without_a_manifest_overwrite_removes_only_this_runs_own_paths(tmp_path):
    generate_dataset(tmp_path, num_images=8, fmt="coco", split_ratios=SplitRatios(0.5, 0.25, 0.25), **_COMMON)
    (tmp_path / _MANIFEST).unlink()

    counts = generate_dataset(
        tmp_path, num_images=8, fmt="coco", split_ratios=SplitRatios(0.5, 0.5, 0.0), overwrite=True, **_COMMON
    )

    assert set(counts) == {"train", "val"}
    # Nothing records that ``test/`` came from this tool any more, so it is left alone.
    assert (tmp_path / "test" / "_annotations.coco.json").is_file()


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("../victim", id="parent"),
        pytest.param(".", id="root-itself"),
        pytest.param("train/..", id="root-via-dotdot"),
        pytest.param("/etc", id="absolute"),
    ],
)
def test_a_manifest_entry_outside_the_output_dir_is_refused_before_deleting_anything(tmp_path, entry):
    out = tmp_path / "ds"
    generate_dataset(out, num_images=4, fmt="coco", **_COMMON)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep me", encoding="utf-8")
    manifest = json.loads((out / _MANIFEST).read_text(encoding="utf-8"))
    manifest["paths"].append(entry)
    (out / _MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    before = _tree(out)

    with pytest.raises(ValueError, match=r"outside|inside"):
        generate_dataset(out, num_images=4, fmt="coco", overwrite=True, **_COMMON)

    assert (victim / "keep.txt").read_text(encoding="utf-8") == "keep me"
    assert _tree(out) == before


def test_yolo_overwrite_never_deletes_through_a_symlinked_directory(tmp_path):
    outside = tmp_path / "outside"
    (outside / "train").mkdir(parents=True)
    (outside / "train" / "keep.txt").write_text("keep me", encoding="utf-8")
    out = tmp_path / "ds"
    out.mkdir()
    (out / "images").symlink_to(outside, target_is_directory=True)
    (out / "data.yaml").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        generate_dataset(out, num_images=2, fmt="yolo", overwrite=True, **_COMMON)

    assert (outside / "train" / "keep.txt").read_text(encoding="utf-8") == "keep me"
    assert (out / "data.yaml").read_text(encoding="utf-8") == "stale"


@pytest.mark.parametrize("fmt", ["coco", "yolo"])
def test_an_invalid_seed_is_refused_before_the_earlier_dataset_is_deleted(tmp_path, fmt):
    generate_dataset(tmp_path, num_images=4, fmt=fmt, **_COMMON)
    before = _tree(tmp_path)

    with pytest.raises(ValueError, match=r"seed|negative"):
        generate_dataset(tmp_path, num_images=4, fmt=fmt, overwrite=True, **{**_COMMON, "seed": -1})

    assert _tree(tmp_path) == before


@pytest.mark.parametrize(
    "name",
    ["CON", "con", "PRN", "Aux", "NUL", "aux.txt", "nul.tar.gz", "COM1", "com9", "LPT1", "lpt9", "train.", "val "],
)
def test_split_names_windows_cannot_create_are_refused(name):
    with pytest.raises(ValueError, match="split name"):
        validate_split_name(name)


@pytest.mark.parametrize("name", ["console", "com10", "lpt", "auxiliary", "nullable", "train.v2", "hold out"])
def test_split_names_that_merely_resemble_reserved_ones_are_accepted(name):
    assert validate_split_name(name) == name


@pytest.mark.parametrize(
    ("args", "flag"),
    [
        pytest.param(["--min_objects", "bad"], "--min_objects", id="int-field"),
        pytest.param(["--min_size_ratio", "tiny"], "--min_size_ratio", id="float-field"),
        pytest.param(["--seed", "bad"], "--seed", id="seed"),
        pytest.param(["--overwrite", "yes"], "--overwrite", id="overwrite"),
    ],
)
def test_cli_names_the_flag_of_a_wrongly_typed_value(tmp_path, capsys, args, flag):
    pytest.importorskip("fire")
    from synth_datasets import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["generate", str(tmp_path / "ds"), "2", "--img_size", "32", *args])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1
    assert flag in err
    assert not (tmp_path / "ds").exists()


def test_cli_lists_the_valid_fields_for_an_unknown_flag(tmp_path, capsys):
    pytest.importorskip("fire")
    from synth_datasets import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["generate", str(tmp_path / "ds"), "2", "--img_size", "32", "--bogus_knob", "3"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1
    assert "--bogus_knob" in err
    assert "min_objects" in err


# --- Round 3: overwrite is transactional; the manifest name is reserved; every CLI flag is type-checked. ---


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Map every path under ``root`` to its bytes (``None`` for a directory), for a byte-for-byte comparison."""
    return {path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes() for path in root.rglob("*")}


def _staging_dirs(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.name.startswith(".vision-synth-staging"))


@pytest.mark.parametrize("fmt", ["coco", "yolo"])
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"distractors": 0.5}, id="fractional-distractors"),
        pytest.param({"occluders": 1.5}, id="fractional-occluders"),
        pytest.param({"min_objects": 2.5, "max_objects": 2.5}, id="unplaceable-fractional-count"),
    ],
)
def test_an_overwrite_that_fails_during_generation_keeps_the_earlier_dataset(tmp_path, fmt, bad):
    """A config that only fails once samples are drawn leaves the earlier dataset byte-for-byte and no staging dir.

    The new dataset is written into a staging directory first, so the old one is deleted only after the new one is
    complete; before, ``overwrite`` deleted first and the lazy failure then lost both.

    """
    generate_dataset(tmp_path, num_images=4, fmt=fmt, **_COMMON)
    before = _snapshot(tmp_path)

    with pytest.raises((TypeError, RuntimeError)):
        generate_dataset(tmp_path, num_images=4, fmt=fmt, overwrite=True, **{**_COMMON, **bad})

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("fmt", ["coco", "yolo"])
def test_a_successful_overwrite_leaves_no_staging_dir_and_the_new_manifest(tmp_path, fmt):
    """After a replacing run the root holds the new dataset and its manifest, and no staging directory."""
    generate_dataset(tmp_path, num_images=8, fmt=fmt, split_ratios=SplitRatios(0.5, 0.25, 0.25), **_COMMON)

    counts = generate_dataset(
        tmp_path, num_images=4, fmt=fmt, split_ratios=SplitRatios(0.5, 0.5, 0.0), overwrite=True, **_COMMON
    )

    assert _staging_dirs(tmp_path) == []
    manifest = json.loads((tmp_path / _MANIFEST).read_text(encoding="utf-8"))
    assert manifest["splits"] == list(counts) == ["train", "val"]
    assert not (tmp_path / "test").exists()
    assert not (tmp_path / "images" / "test").exists()


def test_a_manifest_that_is_a_directory_is_refused_before_any_work(tmp_path):
    """A directory where the manifest belongs is neither treated as absent nor written over after the export."""
    (tmp_path / _MANIFEST).mkdir()
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="manifest"):
        generate_dataset(tmp_path, num_images=2, fmt="coco", **_COMMON)

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("name", [".vision-synth.json", ".vision-synth-staging-1234", ".vision-synth"])
def test_split_names_that_collide_with_the_manifest_or_staging_are_refused(name):
    """A split directory named like the manifest or a staging directory would collide with it."""
    with pytest.raises(ValueError, match="split name"):
        SplitRatios.custom({name: 1.0})


@pytest.mark.parametrize(
    "name",
    [
        "COM¹",
        "com²",
        "LPT³",
        "lpt¹.txt",
        "a<b",
        "a>b",
        'a"b',
        "a|b",
        "a?b",
        "a*b",
        "a\tb",
        "a\x1fb",
        "a\nb",
    ],
)
def test_split_names_with_other_windows_reserved_forms_are_refused(name):
    """Superscript COM/LPT devices, the characters Windows forbids in a name, and control characters are refused."""
    with pytest.raises(ValueError, match="split name"):
        validate_split_name(name)


@pytest.mark.parametrize(
    ("args", "flag"),
    [
        pytest.param(["--min_objects", "None"], "--min_objects", id="none-for-int"),
        pytest.param(["--distractors", "0.5"], "--distractors", id="float-for-int"),
        pytest.param(["--shapes", "5"], "--shapes", id="int-for-names"),
        pytest.param(["--colors", "[1,2]"], "--colors", id="ints-for-names"),
        pytest.param(["--split_ratios", "5"], "--split_ratios", id="scalar-split-ratios"),
        pytest.param(["--seed", "[1]"], "--seed", id="list-seed"),
        pytest.param(["--fmt", "3"], "--fmt", id="int-fmt"),
        pytest.param(["--rotate", "None"], "--rotate", id="none-for-bool"),
        pytest.param(["--img_size", "[1,2,3]"], "--img_size", id="triple-img-size"),
        pytest.param(["--degrade", "5"], "--degrade", id="scalar-degrade"),
    ],
)
def test_cli_names_the_flag_of_any_misused_value(tmp_path, capsys, args, flag):
    """None, a list or a scalar where the flag takes something else is one line naming that flag, not a traceback."""
    pytest.importorskip("fire")
    from synth_datasets import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["generate", str(tmp_path / "ds"), "2", "--img_size", "32", *args])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1
    assert flag in err
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize("writer_cls", [CocoWriter, YoloWriter])
def test_a_replacing_write_that_fails_mid_split_leaves_the_earlier_dataset_and_no_staging(tmp_path, writer_cls):
    """Samples already staged when the stream breaks are discarded with the staging dir; the earlier dataset stays."""
    generate_dataset(tmp_path, num_images=4, fmt=writer_cls.__name__[:4].lower(), **_COMMON)
    before = _snapshot(tmp_path)
    generator = SyntheticGenerator(SyntheticConfig(img_size=32))

    def breaks_after_two():
        yield from generator.generate(2, seed=1)
        raise RuntimeError("stream broke")

    with pytest.raises(RuntimeError, match="stream broke"):
        writer_cls(Task.DETECTION, class_vocabulary(ClassMode.SHAPE, DEFAULT_SHAPES)).write_replacing(
            {"train": breaks_after_two()}, tmp_path
        )

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--distractors", "1", "--distractor_shapes", "duck"], id="one-distractor-shape"),
        pytest.param(["--distractors", "1", "--distractor_shapes", "duck,camel"], id="distractor-shapes"),
        pytest.param(["--distractors", "1", "--distractor_colors", "red,blue"], id="distractor-colors"),
        pytest.param(["--img_size", "48,32"], id="rectangular-img-size"),
        pytest.param(["--rotate", "False"], id="bool-field"),
    ],
)
def test_cli_accepts_well_typed_values_the_type_check_must_not_block(tmp_path, args):
    """Names given as one comma-separated string, a size pair and a bool pass the boundary check and generate."""
    pytest.importorskip("fire")
    from synth_datasets import cli

    cli.main(["generate", str(tmp_path / "ds"), "2", "--img_size", "32", "--seed", "0", *args])

    assert (tmp_path / "ds" / _MANIFEST).is_file()


# --- Rounds 4-5: backup-then-swap; nothing is auto-deleted; a failed swap is rolled back from what is on disk. ---

#: A three-split YOLO dataset replaced by a two-split one moves this many earlier paths into the backup
#: (images/ and labels/ for train, val and test, data.yaml, the manifest) ...
_BACKUPS = 8
#: ... and promotes this many staged entries (images/ and labels/ for train and val, data.yaml, the manifest).
_PROMOTIONS = 6
#: Renames before the swap: the staging and backup ownership markers, each written atomically.
_BEFORE_SWAP = 2
#: Renames after it: the backup, then staging, renamed to ``.vision-synth-discard-*`` once the swap is verified.
_AFTER_SWAP = 2
_RESERVED = (".vision-synth-staging", ".vision-synth-backup", ".vision-synth-discard")
_OWNER = ".vision-synth-owner.json"


class _FailOnCall:
    """Wrap ``real`` so the calls numbered in ``fail_at`` raise ``error``, just before the real call or just after it.

    ``after=True`` is the interrupt that lands once a rename has happened but before the code that made it could record
    it.

    """

    def __init__(self, real, *fail_at, error=OSError, after=False):
        self.real = real
        self.fail_at = set(fail_at)
        self.error = error
        self.after = after
        self.calls = 0

    def __call__(self, *args, **kwargs):
        index = self.calls
        self.calls += 1
        if index in self.fail_at and not self.after:
            raise self.error(f"injected failure at call {index}")
        result = self.real(*args, **kwargs)
        if index in self.fail_at and self.after:
            raise self.error(f"injected failure after call {index}")
        return result


def _three_split_yolo(root: Path) -> None:
    generate_dataset(root, num_images=8, fmt="yolo", split_ratios=SplitRatios(0.5, 0.25, 0.25), **_COMMON)


def _replace_with_two_splits(root: Path, fmt: str = "yolo") -> dict[str, int]:
    return generate_dataset(
        root, num_images=8, fmt=fmt, split_ratios=SplitRatios(0.5, 0.5, 0.0), overwrite=True, **_COMMON
    )


def _leftovers(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.name.casefold().startswith(_RESERVED))


def _outside_leftovers(root: Path) -> dict[str, bytes | None]:
    """Snapshot ``root`` without the staging and backup directories."""
    return {rel: data for rel, data in _snapshot(root).items() if not rel.casefold().startswith(_RESERVED)}


def _lost_files(root: Path, before: dict[str, bytes | None]) -> list[str]:
    """Return the files of ``before`` found byte-for-byte neither in place under ``root`` nor in a backup there."""
    backups = [root / name for name in _leftovers(root) if name.startswith(".vision-synth-backup")]
    homes = [root, *backups]
    return [
        rel
        for rel, data in before.items()
        if data is not None and not any((home / rel).is_file() and (home / rel).read_bytes() == data for home in homes)
    ]


def test_a_replacing_write_swaps_through_a_backup_and_leaves_nothing_behind(tmp_path, monkeypatch):
    """Every earlier path is renamed into the backup and every staged entry into place; on success nothing is left.

    Pins the rename count the fault-injection cases below index into.

    """
    _three_split_yolo(tmp_path)
    counter = _FailOnCall(os.replace)
    monkeypatch.setattr(os, "replace", counter)

    _replace_with_two_splits(tmp_path)

    assert counter.calls == _BEFORE_SWAP + _BACKUPS + _PROMOTIONS + _AFTER_SWAP
    assert list(tmp_path.rglob(".vision-synth-owner*")) == []
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        pytest.param(OSError, RuntimeError, id="oserror"),
        pytest.param(KeyboardInterrupt, KeyboardInterrupt, id="ctrl-c"),
    ],
)
@pytest.mark.parametrize("after", [pytest.param(False, id="before-rename"), pytest.param(True, id="after-rename")])
@pytest.mark.parametrize("fail_at", range(_BACKUPS + _PROMOTIONS))
def test_a_swap_stopped_at_any_rename_is_rolled_back_completely_and_the_next_write_proceeds(
    tmp_path, monkeypatch, error, raised, after, fail_at
):
    """Whatever stops the swap, wherever, every earlier file is back in place and no reserved directory is left.

    A rollback verified complete by re-listing removes the emptied backup with ``os.rmdir`` alone, which cannot delete a
    file, so the next write is not blocked by an empty leftover.

    ``after-rename`` is the case the in-memory bookkeeping missed: a Ctrl+C between a rename into the backup and the
    line recording it left that path untracked, and the rollback then deleted the backup holding its only copy.

    """
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(os, "replace", _FailOnCall(os.replace, _BEFORE_SWAP + fail_at, error=error, after=after))

    with pytest.raises(raised):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _lost_files(tmp_path, before) == []
    assert _snapshot(tmp_path) == before
    assert _replace_with_two_splits(tmp_path) == {"train": 4, "val": 4}


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize(
    "fail_at",
    [
        pytest.param(_BEFORE_SWAP + 3, id="backing-up-the-manifest"),
        pytest.param(_BEFORE_SWAP + 5, id="promoting-a-whole-split-dir"),
        pytest.param(_BEFORE_SWAP + 6, id="promoting-the-manifest"),
    ],
)
def test_a_coco_swap_rolls_back_whole_directory_promotions(tmp_path, monkeypatch, fail_at, error):
    """COCO splits are promoted as whole directories rather than merged; a stopped swap still loses nothing.

    Four backups (train, val, test, manifest) then three promotions (train, val, manifest).

    """
    generate_dataset(tmp_path, num_images=8, fmt="coco", split_ratios=SplitRatios(0.5, 0.25, 0.25), **_COMMON)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(os, "replace", _FailOnCall(os.replace, fail_at, error=error, after=True))

    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        _replace_with_two_splits(tmp_path, fmt="coco")

    monkeypatch.undo()
    assert _outside_leftovers(tmp_path) == before


def test_a_failed_rollback_keeps_the_backup_and_names_it_and_the_next_run_refuses(tmp_path, monkeypatch):
    """If putting the earlier dataset back fails too, its files stay in the backup, which is named, never deleted."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    # Promotion 1 fails, then so does the rollback's first rename (moving promotion 0 back out).
    monkeypatch.setattr(
        os, "replace", _FailOnCall(os.replace, _BEFORE_SWAP + _BACKUPS + 1, _BEFORE_SWAP + _BACKUPS + 2)
    )

    with pytest.raises(RuntimeError, match=r"rollback.*\.vision-synth-backup-") as exc_info:
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    backups = [name for name in _leftovers(tmp_path) if name.startswith(".vision-synth-backup")]
    assert len(backups) == 1
    assert backups[0] in str(exc_info.value)
    assert _lost_files(tmp_path, before) == []
    with pytest.raises(ValueError, match=backups[0]):
        _replace_with_two_splits(tmp_path)


def test_a_cleanup_failure_after_the_swap_reports_it_and_keeps_the_new_dataset(tmp_path, monkeypatch):
    """Once every entry is confirmed in place the new dataset stands; a failed removal is reported, not hidden."""
    _three_split_yolo(tmp_path)
    monkeypatch.setattr(shutil, "rmtree", _FailOnCall(shutil.rmtree, 0))

    with pytest.raises(RuntimeError, match="new dataset is in place"):
        _replace_with_two_splits(tmp_path)

    manifest = json.loads((tmp_path / _MANIFEST).read_text(encoding="utf-8"))
    assert manifest["splits"] == ["train", "val"]
    assert not (tmp_path / "images" / "test").exists()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize(
    "leftover",
    [".vision-synth-backup-k3x9", ".vision-synth-staging-k3x9", ".Vision-Synth-Staging-k3x9"],
)
def test_a_leftover_staging_or_backup_dir_is_refused_by_name_and_kept(tmp_path, overwrite, leftover):
    """An interrupted run's leftovers cannot be told apart from anything else safely, so a run stops and names them."""
    (tmp_path / leftover / "images" / "train").mkdir(parents=True)
    (tmp_path / leftover / "images" / "train" / "img_000000.jpg").write_bytes(b"earlier")
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match=leftover):
        generate_dataset(tmp_path, num_images=2, fmt="yolo", overwrite=overwrite, **_COMMON)

    assert _snapshot(tmp_path) == before


def _list_in_manifest(root: Path, rel: str) -> None:
    manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
    manifest["paths"].append(rel)
    (root / _MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("overwrite", [False, True])
def test_an_old_split_named_like_a_staging_dir_is_refused_not_deleted(tmp_path, overwrite):
    """A split an older release let through as ``.Vision-Synth-Staging-1`` is listed as owned, and still kept."""
    generate_dataset(tmp_path, num_images=4, fmt="coco", **_COMMON)
    (tmp_path / ".Vision-Synth-Staging-1").mkdir()
    (tmp_path / ".Vision-Synth-Staging-1" / "_annotations.coco.json").write_text("{}", encoding="utf-8")
    _list_in_manifest(tmp_path, ".Vision-Synth-Staging-1")
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match=r"\.Vision-Synth-Staging-1"):
        generate_dataset(tmp_path, num_images=4, fmt="coco", overwrite=overwrite, **_COMMON)

    assert _snapshot(tmp_path) == before


def test_a_manifest_path_with_a_reserved_component_is_refused_before_anything_moves(tmp_path):
    """A listed path with a ``.vision-synth`` component anywhere is not acted on, even below the root."""
    generate_dataset(tmp_path, num_images=4, fmt="yolo", **_COMMON)
    (tmp_path / "images" / ".VISION-synth-old").mkdir()
    _list_in_manifest(tmp_path, "images/.VISION-synth-old")
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="reserved"):
        generate_dataset(tmp_path, num_images=4, fmt="yolo", overwrite=True, **_COMMON)

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "name", [".VISION-SYNTH.JSON", ".Vision-Synth-Staging-1", ".vIsIoN-sYnTh", ".VISION-synth-backup-x"]
)
def test_mixed_case_forms_of_the_reserved_prefix_are_refused(name):
    """A case-insensitive filesystem would put these on top of the manifest or a staging/backup dir."""
    with pytest.raises(ValueError, match="split name"):
        validate_split_name(name)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--colors", "red,blue"], id="colors-names"),
        pytest.param(["--colors", "[(255,215,0)]"], id="colors-list-of-triples"),
        pytest.param(["--colors", "(255,215,0)"], id="colors-one-triple"),
        pytest.param(["--colors", "[red,(0,0,255)]"], id="colors-mixed-list"),
        pytest.param(["--distractors", "1", "--distractor_colors", "[(255,215,0)]"], id="distractor-colors-triples"),
        pytest.param(["--distractors", "1", "--distractor_colors", "(255,215,0)"], id="distractor-colors-one-triple"),
        pytest.param(["--distractors", "1", "--distractor_colors", "green"], id="distractor-colors-name"),
        pytest.param(["--background", "red"], id="background-name"),
        pytest.param(["--background", "(255,215,0)"], id="background-triple"),
        pytest.param(["--background", "[255,215,0]"], id="background-list"),
        pytest.param(["--shapes", "[duck,camel]"], id="shapes-list"),
        pytest.param(["--distractors", "1", "--distractor_shapes", "[duck,camel]"], id="distractor-shapes-list"),
        pytest.param(["--img_size", "[48,32]"], id="img-size-list"),
        pytest.param(["--split_ratios", "{train:1}"], id="split-ratios-int-mapping"),
        pytest.param(["--split_ratios", "[0.5,0.5,0]"], id="split-ratios-list"),
    ],
)
def test_cli_accepts_every_documented_spelling(tmp_path, args):
    """Each input spelling the options document passes the boundary check; normalization happens after it."""
    pytest.importorskip("fire")
    from synth_datasets import cli

    cli.main(["generate", str(tmp_path / "ds"), "2", "--img_size", "32", "--seed", "0", *args])

    assert (tmp_path / "ds" / _MANIFEST).is_file()


# --- Round 6: ownership markers, a discard phase, and refusals built from what is actually found. ---


def _reference_new_dataset(tmp_path: Path) -> dict[str, bytes | None]:
    """The dataset an uninterrupted replace leaves, generated in a separate directory for byte comparison."""
    ref = tmp_path / "reference"
    _three_split_yolo(ref)
    _replace_with_two_splits(ref)
    return _snapshot(ref)


def _marker(directory: Path) -> dict:
    return json.loads((directory / _OWNER).read_text(encoding="utf-8"))


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.is_symlink() or path.exists():
        path.unlink()


def _refusal(root: Path) -> str:
    with pytest.raises(ValueError, match="refusing to write") as exc_info:
        generate_dataset(root, num_images=2, fmt="yolo", **_COMMON)
    return str(exc_info.value)


class _FailWhen:
    """Wrap ``real`` so its first call whose destination name starts with ``prefix`` raises, just before or after."""

    def __init__(self, real, prefix, after=False):
        self.real = real
        self.prefix = prefix
        self.after = after
        self.fired = False

    def __call__(self, src, dst, *args, **kwargs):
        hit = not self.fired and Path(dst).name.startswith(self.prefix)
        if hit and not self.after:
            self.fired = True
            raise KeyboardInterrupt(f"injected before renaming onto {dst}")
        result = self.real(src, dst, *args, **kwargs)
        if hit:
            self.fired = True
            raise KeyboardInterrupt(f"injected after renaming onto {dst}")
        return result


@pytest.mark.parametrize("after", [pytest.param(False, id="before-rename"), pytest.param(True, id="after-rename")])
@pytest.mark.parametrize("fail_at", range(_BEFORE_SWAP))
def test_ctrl_c_while_marking_leaves_the_earlier_dataset_and_nothing_else(tmp_path, monkeypatch, fail_at, after):
    """An interrupt while a staging or backup marker is written, before anything moved, is cleaned up by this call."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(os, "replace", _FailOnCall(os.replace, fail_at, error=KeyboardInterrupt, after=after))

    with pytest.raises(KeyboardInterrupt):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _snapshot(tmp_path) == before


def test_ctrl_c_before_the_commit_rename_leaves_a_backup_that_restores_the_earlier_dataset(tmp_path, monkeypatch):
    """Before the backup is renamed to discard the run did not commit; the documented steps restore it exactly."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    index = _BEFORE_SWAP + _BACKUPS + _PROMOTIONS
    monkeypatch.setattr(os, "replace", _FailOnCall(os.replace, index, error=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert "did not commit" in _refusal(tmp_path)
    _apply_restore_plan(tmp_path)
    assert _outside_leftovers(tmp_path) == before


@pytest.mark.parametrize(
    ("step", "after"),
    [
        pytest.param(0, True, id="after-backup-to-discard"),
        pytest.param(1, False, id="before-staging-to-discard"),
        pytest.param(1, True, id="after-staging-to-discard"),
    ],
)
def test_ctrl_c_after_the_commit_rename_leaves_the_complete_new_dataset(tmp_path, monkeypatch, step, after):
    """Once the backup is renamed to discard the new dataset is complete in place, and the refusal says so."""
    reference = _reference_new_dataset(tmp_path)
    out = tmp_path / "out"
    _three_split_yolo(out)
    index = _BEFORE_SWAP + _BACKUPS + _PROMOTIONS + step
    monkeypatch.setattr(os, "replace", _FailOnCall(os.replace, index, error=KeyboardInterrupt, after=after))

    with pytest.raises(KeyboardInterrupt):
        _replace_with_two_splits(out)

    monkeypatch.undo()
    message = _refusal(out)
    assert _outside_leftovers(out) == reference
    assert "completed" in message
    assert "did not commit" not in message


@pytest.mark.parametrize("fail_at", range(4))
def test_ctrl_c_while_removing_the_discarded_copies_keeps_the_new_dataset(tmp_path, monkeypatch, fail_at):
    """An interrupt during the final removals leaves only marked discard dirs beside the complete new dataset."""
    reference = _reference_new_dataset(tmp_path)
    out = tmp_path / "out"
    _three_split_yolo(out)
    monkeypatch.setattr(shutil, "rmtree", _FailOnCall(shutil.rmtree, fail_at, error=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        _replace_with_two_splits(out)

    monkeypatch.undo()
    assert _outside_leftovers(out) == reference
    assert {name[:21] for name in _leftovers(out)} == {".vision-synth-discard"}
    message = _refusal(out)
    assert "completed" in message
    assert "not created by vision-synth" not in message


def test_the_backup_marker_lists_exactly_the_paths_the_run_adds(tmp_path, monkeypatch):
    """Going from two splits to three adds only the new split's image and label dirs; nothing else is named added."""
    generate_dataset(tmp_path, num_images=8, fmt="yolo", split_ratios=SplitRatios(0.5, 0.5, 0.0), **_COMMON)
    monkeypatch.setattr(os, "replace", _FailWhen(os.replace, ".vision-synth-discard"))

    with pytest.raises(KeyboardInterrupt):
        generate_dataset(
            tmp_path, num_images=8, fmt="yolo", split_ratios=SplitRatios(0.5, 0.25, 0.25), overwrite=True, **_COMMON
        )

    monkeypatch.undo()
    (backup,) = [path for path in tmp_path.iterdir() if path.name.startswith(".vision-synth-backup")]
    marker = _marker(backup)
    assert marker["kind"] == "backup"
    assert sorted(marker["added"]) == ["images/test", "labels/test"]
    assert "images/test" in _refusal(tmp_path)


@pytest.mark.parametrize("name", [".vision-synth-staging-x", ".vision-synth-backup-x", ".Vision-Synth-Discard-x"])
def test_an_unmarked_reserved_dir_is_unrecognized_and_never_called_disposable(tmp_path, name):
    """Without a valid ownership marker a reserved-looking dir is not this tool's: rename or move it, never delete."""
    (tmp_path / name).mkdir()
    (tmp_path / name / "keep.txt").write_text("not vision-synth's", encoding="utf-8")
    before = _snapshot(tmp_path)

    message = _refusal(tmp_path)

    assert "not created by vision-synth" in message
    assert "delete" not in message.lower()
    assert _snapshot(tmp_path) == before


def test_a_legacy_split_with_a_reserved_name_is_unrecognized_and_untouched(tmp_path):
    """A split an older release wrote as ``.Vision-Synth-Staging-1`` is refused as unrecognized, never as
    disposable."""
    generate_dataset(tmp_path, num_images=4, fmt="coco", **_COMMON)
    (tmp_path / ".Vision-Synth-Staging-1").mkdir()
    (tmp_path / ".Vision-Synth-Staging-1" / "_annotations.coco.json").write_text("{}", encoding="utf-8")
    _list_in_manifest(tmp_path, ".Vision-Synth-Staging-1")
    before = _snapshot(tmp_path)

    message = _refusal(tmp_path)

    assert "not created by vision-synth" in message
    assert "delete" not in message.lower()
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("name", [".vision-synth-discard", ".VISION-SYNTH-DISCARD-1"])
def test_the_discard_prefix_is_reserved_for_split_names(name):
    """A split cannot take the name the committed-phase directories use."""
    with pytest.raises(ValueError, match="split name"):
        validate_split_name(name)


# --- Round 7: the backup's restore steps are computed per path from what is on disk at refusal time. ---


def _apply_restore_plan(root: Path) -> None:
    """Carry out the backup's restore steps mechanically, from the structured plan the refusal message is built
    from."""
    (backup,) = [path for path in root.iterdir() if path.name.startswith(".vision-synth-backup")]
    plan = writers._restore_plan(backup, _marker(backup))
    for rel in plan.replace:
        _remove_path(root / rel)
        os.replace(backup / rel, root / rel)
    for rel in plan.move:
        os.replace(backup / rel, root / rel)
    for rel in plan.remove:
        _remove_path(root / rel)
    assert plan.suspect == []


def _die_instead_of_rolling_back(*args, **kwargs):
    """Stand-in for a process killed after the swap stopped: no rollback runs at all."""
    raise KeyboardInterrupt("process died before rolling back")


@pytest.mark.parametrize("after", [pytest.param(False, id="before-rename"), pytest.param(True, id="after-rename")])
@pytest.mark.parametrize("fail_at", range(_BACKUPS + _PROMOTIONS))
def test_following_the_restore_steps_after_a_kill_mid_swap_loses_nothing(tmp_path, monkeypatch, fail_at, after):
    """Killed at any swap rename with no rollback, the refusal's per-path steps rebuild the earlier dataset exactly.

    A path not yet backed up still holds its only original; the old guidance deleted it before moving a backup copy that
    did not exist.

    """
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(writers, "_roll_back", _die_instead_of_rolling_back)
    monkeypatch.setattr(
        os, "replace", _FailOnCall(os.replace, _BEFORE_SWAP + fail_at, error=KeyboardInterrupt, after=after)
    )

    with pytest.raises(KeyboardInterrupt):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _lost_files(tmp_path, before) == []
    assert "did not commit" in _refusal(tmp_path)
    _apply_restore_plan(tmp_path)
    assert _outside_leftovers(tmp_path) == before


#: Stopping the swap just before its last promotion leaves 5 promotions and 8 backups for the rollback to undo.
_ROLLBACK_RENAMES = _PROMOTIONS - 1 + _BACKUPS


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
@pytest.mark.parametrize("rollback_step", range(_ROLLBACK_RENAMES))
def test_following_the_restore_steps_after_a_failed_rollback_loses_nothing(tmp_path, monkeypatch, rollback_step, error):
    """A rollback stopped at any of its renames leaves a mixed state; the per-path steps still restore it exactly."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    last_promotion = _BEFORE_SWAP + _BACKUPS + _PROMOTIONS - 1
    failing = _FailOnCall(os.replace, last_promotion, last_promotion + 1 + rollback_step, error=error)
    monkeypatch.setattr(os, "replace", failing)

    with pytest.raises((error, RuntimeError)):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _lost_files(tmp_path, before) == []
    assert "did not commit" in _refusal(tmp_path)
    _apply_restore_plan(tmp_path)
    assert _outside_leftovers(tmp_path) == before


def test_a_path_the_backup_has_no_copy_of_is_reported_as_still_in_place(tmp_path, monkeypatch):
    """Killed before the first backup rename, every replaced path is still the original, and the message says so."""
    _three_split_yolo(tmp_path)
    monkeypatch.setattr(writers, "_roll_back", _die_instead_of_rolling_back)
    monkeypatch.setattr(os, "replace", _FailOnCall(os.replace, _BEFORE_SWAP, error=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    message = _refusal(tmp_path)
    assert "still in place" in message
    assert "images/train" in message
    assert "remove the interrupted run's version" not in message


def test_an_added_path_with_a_backup_copy_is_left_alone_and_reported(tmp_path, monkeypatch):
    """An added path never existed before, so a backup copy of one is a bug: it is neither removed nor restored."""
    generate_dataset(tmp_path, num_images=8, fmt="yolo", split_ratios=SplitRatios(0.5, 0.5, 0.0), **_COMMON)
    monkeypatch.setattr(os, "replace", _FailWhen(os.replace, ".vision-synth-discard"))
    with pytest.raises(KeyboardInterrupt):
        generate_dataset(
            tmp_path, num_images=8, fmt="yolo", split_ratios=SplitRatios(0.5, 0.25, 0.25), overwrite=True, **_COMMON
        )
    monkeypatch.undo()
    (backup,) = [path for path in tmp_path.iterdir() if path.name.startswith(".vision-synth-backup")]
    (backup / "images" / "test").mkdir(parents=True)

    plan = writers._restore_plan(backup, _marker(backup))

    assert plan.suspect == ["images/test"]
    assert "images/test" not in plan.remove
    assert "should not happen" in _refusal(tmp_path)


# --- Round 8: a split respelled only in case is one entry on a case-insensitive filesystem. ---


def _tmp_is_case_insensitive() -> bool:
    """Probe the temp filesystem once, at collection: does ``PROBE`` find a directory created as ``probe``?"""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "probe").mkdir()
        return (Path(tmp) / "PROBE").exists()


_skip_case_sensitive_fs = pytest.mark.skipif(
    not _tmp_is_case_insensitive(), reason="needs a case-insensitive temp filesystem (default macOS/Windows)"
)
_LOWER = SplitRatios.custom({"train": 0.5, "val": 0.5})
_UPPER = SplitRatios.custom({"Train": 0.5, "val": 0.5})


@_skip_case_sensitive_fs
@pytest.mark.parametrize(("fmt", "split_dir"), [("yolo", "images"), ("coco", ".")])
def test_respelling_a_split_in_case_replaces_it_and_records_the_new_spelling(tmp_path, fmt, split_dir):
    """``train`` then ``Train`` names one directory here; the overwrite replaces it once and stores ``Train``."""
    reference = tmp_path / "reference"
    generate_dataset(reference, num_images=4, fmt=fmt, split_ratios=_UPPER, **_COMMON)
    out = tmp_path / "out"
    generate_dataset(out, num_images=4, fmt=fmt, split_ratios=_LOWER, **_COMMON)

    counts = generate_dataset(out, num_images=4, fmt=fmt, split_ratios=_UPPER, overwrite=True, **_COMMON)

    assert counts == {"Train": 2, "val": 2}
    assert "Train" in os.listdir(out / split_dir)
    assert "train" not in os.listdir(out / split_dir)
    assert _snapshot(out) == _snapshot(reference)
    assert _leftovers(out) == []


@_skip_case_sensitive_fs
@pytest.mark.parametrize("after", [pytest.param(False, id="before-rename"), pytest.param(True, id="after-rename")])
@pytest.mark.parametrize("fail_at", range(6))
def test_a_stopped_case_respelling_rolls_back_to_the_old_spelling(tmp_path, monkeypatch, fail_at, after):
    """Ctrl+C anywhere in a ``train`` to ``Train`` swap restores the earlier dataset exactly, spelled ``train``."""
    generate_dataset(tmp_path, num_images=4, fmt="coco", split_ratios=_LOWER, **_COMMON)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(
        os, "replace", _FailOnCall(os.replace, _BEFORE_SWAP + fail_at, error=KeyboardInterrupt, after=after)
    )

    with pytest.raises(KeyboardInterrupt):
        generate_dataset(tmp_path, num_images=4, fmt="coco", split_ratios=_UPPER, overwrite=True, **_COMMON)

    monkeypatch.undo()
    assert _snapshot(tmp_path) == before
    assert "train" in os.listdir(tmp_path)


def _casefold_alias(parent: Path, part: str, listings=None) -> str | None:
    """Stand-in for a case-insensitive filesystem's lookup, usable on any filesystem."""
    return next((name for name in os.listdir(parent) if name.casefold() == part.casefold()), None)


def test_aliases_are_planned_as_one_entry_when_the_filesystem_says_so(tmp_path, monkeypatch):
    """With the lookup reporting ``images/Train`` as ``images/train``, it is backed up once and promoted whole."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    staging = tmp_path / ".vision-synth-staging-t"
    (staging / "images" / "Train").mkdir(parents=True)
    monkeypatch.setattr(writers, "_stored_name", _casefold_alias)

    targets = writers._outermost_present(tmp_path, [tmp_path / "images" / "Train", tmp_path / "images" / "train"])
    moves = writers._plan_moves(staging, tmp_path, set(targets))

    assert targets == [tmp_path / "images" / "train"]
    assert moves == [(staging / "images" / "Train", tmp_path / "images" / "Train")]


def test_distinct_spellings_stay_distinct_when_the_filesystem_keeps_them_apart(tmp_path, monkeypatch):
    """A lookup that finds only exact names (a case-sensitive filesystem) never merges ``Train`` into ``train``."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    monkeypatch.setattr(
        writers, "_stored_name", lambda parent, part, listings=None: part if part in os.listdir(parent) else None
    )

    targets = writers._outermost_present(tmp_path, [tmp_path / "images" / "Train", tmp_path / "images" / "train"])

    assert targets == [tmp_path / "images" / "train"]


# --- Round 9: planning reads each directory once, however many paths the manifest lists. ---


class _CountCalls:
    """Wrap ``real`` and count its calls."""

    def __init__(self, real):
        self.real = real
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.real(*args, **kwargs)


def _list_every_file_in_manifest(root: Path) -> None:
    """Make the manifest file-level, as a custom writer's could be: one entry per file it wrote."""
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    for rel in files:
        if rel != _MANIFEST:
            _list_in_manifest(root, rel)


def _directory_reads_for_a_file_level_overwrite(root: Path, num_images: int, monkeypatch) -> int:
    common = {**_COMMON, "fmt": "yolo", "split_ratios": SplitRatios.custom({"train": 1.0})}
    generate_dataset(root, num_images=num_images, **common)
    _list_every_file_in_manifest(root)
    listdir, scandir = _CountCalls(os.listdir), _CountCalls(os.scandir)
    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "scandir", scandir)

    generate_dataset(root, num_images=num_images, overwrite=True, **common)

    monkeypatch.undo()
    return listdir.calls + scandir.calls


def test_planning_reads_each_directory_once_however_many_paths_are_listed(tmp_path, monkeypatch):
    """Directory reads stay bounded by the number of directories: 800 listed files cost what 200 do.

    Planning used to list a path's parent once per path, so a file-level manifest made it quadratic.

    """
    small = _directory_reads_for_a_file_level_overwrite(tmp_path / "small", 200, monkeypatch)
    large = _directory_reads_for_a_file_level_overwrite(tmp_path / "large", 800, monkeypatch)

    # Root, images/, labels/, the split dirs, staging, backup and discard mirrors: a few dozen reads at most.
    assert large <= 40
    assert large - small <= 2


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_an_unreadable_directory_aborts_planning_instead_of_reading_as_absent(tmp_path, monkeypatch, error):
    """A directory that cannot be listed is an error, never an empty one.

    Reading it as absent would plan the listed files as new paths, skip their backup, and lose the originals if the swap
    then failed.

    """
    common = {**_COMMON, "fmt": "yolo", "split_ratios": SplitRatios.custom({"train": 1.0})}
    generate_dataset(tmp_path, num_images=4, **common)
    _list_every_file_in_manifest(tmp_path)
    before = _snapshot(tmp_path)
    unreadable = tmp_path / "images" / "train"
    real_listdir = os.listdir

    def listdir(path="."):
        if Path(path) == unreadable:
            raise error(13, "Permission denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", listdir)
    with pytest.raises(error):
        generate_dataset(tmp_path, num_images=4, overwrite=True, **common)
    monkeypatch.undo()

    assert _leftovers(tmp_path) == []
    assert _snapshot(tmp_path) == before


# --- Round 10: metadata errors are never read as absence; unknown state stops the run instead of guessing. ---

_DENIED = [
    pytest.param(PermissionError(errno.EACCES, "Permission denied"), id="eacces"),
    pytest.param(OSError(errno.EIO, "Input/output error"), id="eio"),
]


class _DenyLstat:
    """``os.lstat`` that raises ``error`` for paths containing every one of ``fragments`` once ``armed``."""

    def __init__(self, real, error, *fragments, armed=True):
        self.real = real
        self.error = error
        self.fragments = fragments
        self.armed = armed

    def __call__(self, path, *args, **kwargs):
        if self.armed and all(fragment in os.fspath(path) for fragment in self.fragments):
            raise self.error
        return self.real(path, *args, **kwargs)


class _ArmingFailure(_FailOnCall):
    """Raise ``KeyboardInterrupt`` at one ``os.replace`` call and arm ``denier`` from then on: the rollback's reads."""

    def __init__(self, real, fail_at, denier):
        super().__init__(real, fail_at, error=KeyboardInterrupt)
        self.denier = denier

    def __call__(self, *args, **kwargs):
        if self.calls in self.fail_at:
            self.denier.armed = True
        return super().__call__(*args, **kwargs)


@pytest.mark.parametrize("error", _DENIED)
@pytest.mark.parametrize("fail_at", range(4))
def test_a_rollback_that_cannot_stat_a_staged_path_stops_instead_of_moving_an_original(
    tmp_path, monkeypatch, fail_at, error
):
    """An unreadable staged ``data.yaml`` is unknown, not absent: the rollback stops and moves nothing onto it.

    Reading it as absent made the rollback move the original ``data.yaml`` into staging, where nothing looked.

    """
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    denier = _DenyLstat(os.lstat, error, ".vision-synth-staging", "data.yaml", armed=False)
    monkeypatch.setattr(os, "lstat", denier)
    monkeypatch.setattr(os, "replace", _ArmingFailure(os.replace, _BEFORE_SWAP + fail_at, denier))

    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _lost_files(tmp_path, before) == []
    assert [name[:20] for name in _leftovers(tmp_path) if name.startswith(".vision-synth-backup")]


@pytest.mark.parametrize("error", _DENIED)
def test_an_unstatable_manifest_stops_the_run_instead_of_reading_as_absent(tmp_path, monkeypatch, error):
    """A manifest whose metadata cannot be read is not a missing one: its splits are not dropped from ownership."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(os, "lstat", _DenyLstat(os.lstat, error, _MANIFEST))
    monkeypatch.setattr(os, "stat", _DenyLstat(os.stat, error, _MANIFEST))

    with pytest.raises((ValueError, OSError)):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _snapshot(tmp_path) == before


def test_the_manifest_check_does_not_depend_on_pathlib_predicates(tmp_path, monkeypatch):
    """Python 3.14's pathlib reads a stat error as 'absent'; with its predicates made to do so, the run still stops."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    denied = PermissionError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(os, "lstat", _DenyLstat(os.lstat, denied, _MANIFEST))
    monkeypatch.setattr(os, "stat", _DenyLstat(os.stat, denied, _MANIFEST))
    for predicate in ("exists", "is_file", "is_dir", "is_symlink"):
        real = getattr(Path, predicate)
        monkeypatch.setattr(Path, predicate, _swallowing(real))

    with pytest.raises((ValueError, OSError)):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _snapshot(tmp_path) == before


def _swallowing(predicate):
    """Return ``predicate`` with 3.14 semantics: any ``OSError`` reads as ``False``."""

    def swallowed(self, *args, **kwargs):
        try:
            return predicate(self, *args, **kwargs)
        except OSError:
            return False

    return swallowed


@pytest.mark.parametrize("error", _DENIED)
def test_pruning_keeps_the_marker_when_a_subdirectory_cannot_be_read(tmp_path, monkeypatch, error):
    """The marker goes only after a complete walk shows nothing but empty dirs; an unreadable subdir proves nothing."""
    owned = tmp_path / ".vision-synth-backup-t"
    (owned / "images" / "train").mkdir(parents=True)
    (owned / "images" / "train" / "img_000000.jpg").write_bytes(b"earlier")
    (owned / _OWNER).write_text("{}", encoding="utf-8")
    real_scandir = os.scandir

    def scandir(path="."):
        if os.fspath(path).endswith(os.sep + "images"):
            raise error
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)

    pruned = writers._prune_owned(owned)

    monkeypatch.undo()
    assert pruned is False
    assert (owned / _OWNER).is_file()
    assert (owned / "images" / "train" / "img_000000.jpg").read_bytes() == b"earlier"


def _interrupted_backup(root: Path) -> Path:
    """Leave a genuine backup behind: a run killed right after its first backup rename, with no rollback."""
    _three_split_yolo(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(writers, "_roll_back", _die_instead_of_rolling_back)
        patch.setattr(os, "replace", _FailOnCall(os.replace, _BEFORE_SWAP, error=KeyboardInterrupt, after=True))
        with pytest.raises(KeyboardInterrupt):
            _replace_with_two_splits(root)
    (backup,) = [path for path in root.iterdir() if path.name.startswith(".vision-synth-backup")]
    return backup


def test_an_unreadable_marker_is_ownership_unknown_not_unrecognized(tmp_path, monkeypatch):
    """A marker that exists but cannot be opened says nothing either way: no action is suggested on its directory."""
    _interrupted_backup(tmp_path)
    real_open = open

    # Denied through ``open`` rather than ``chmod(0)``: Windows only sets a read-only flag and root ignores the mode,
    # so the file would stay readable there and the test would check nothing.
    def denying_open(file, *args, **kwargs):
        if os.fspath(file).endswith(_OWNER):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(writers, "open", denying_open, raising=False)

    message = _refusal(tmp_path)

    assert "ownership unknown" in message
    assert "not created by vision-synth" not in message
    assert _lost_files(tmp_path, _snapshot(tmp_path)) == []


def test_a_marker_read_io_error_is_ownership_unknown(tmp_path, monkeypatch):
    """An I/O error opening the marker is reported with its reason, never as an unrelated or disposable directory."""
    backup = _interrupted_backup(tmp_path)
    real_open = open

    def failing_open(file, *args, **kwargs):
        if os.fspath(file).endswith(_OWNER):
            raise OSError(errno.EIO, "Input/output error")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(writers, "open", failing_open, raising=False)

    message = _refusal(tmp_path)

    assert "ownership unknown" in message
    assert "Input/output error" in message
    assert backup.name in message


@pytest.mark.parametrize("error", _DENIED)
def test_the_restore_plan_reports_an_unstatable_path_as_unknown(tmp_path, monkeypatch, error):
    """A recorded path whose state cannot be read is neither replaced, moved, kept nor removed: it is unknown."""
    backup = _interrupted_backup(tmp_path)
    monkeypatch.setattr(os, "lstat", _DenyLstat(os.lstat, error, os.sep + "labels" + os.sep + "val"))

    plan = writers._restore_plan(backup, _marker(backup))
    message = _refusal(tmp_path)

    assert plan.unknown == ["labels/val"]
    assert "labels/val" not in plan.replace + plan.move + plan.keep + plan.remove
    assert "state unknown" in message


@pytest.mark.parametrize("error", _DENIED)
def test_planning_stops_on_an_unstatable_target(tmp_path, monkeypatch, error):
    """A target whose metadata cannot be read stops the planning phase; nothing has moved."""
    _three_split_yolo(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(os, "lstat", _DenyLstat(os.lstat, error, str(tmp_path / "images")))

    with pytest.raises(OSError, match=r"Permission denied|Input/output error"):
        _replace_with_two_splits(tmp_path)

    monkeypatch.undo()
    assert _snapshot(tmp_path) == before
