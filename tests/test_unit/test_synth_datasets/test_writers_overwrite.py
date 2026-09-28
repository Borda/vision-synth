"""Writers refuse to mix a new dataset into an old one unless told to replace it.

A second ``generate_dataset`` into the same directory used to overwrite ``img_000000..`` in place and leave every
higher-numbered file of the first run behind: 20 images then 5 left 14 in ``images/train`` and a ``data.yaml``
describing neither run. The writers now check the paths they own before consuming a single sample.

"""

from __future__ import annotations

import json
import shutil
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

_COMMON = {"img_size": 32, "seed": 0, "task": "detection"}


def _tree(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*"))


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_rerun_into_populated_dir_refuses_and_leaves_it_untouched(tmp_path, fmt):
    generate_dataset(tmp_path, num_images=20, fmt=fmt, **_COMMON)
    before = _tree(tmp_path)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_dataset(tmp_path, num_images=5, fmt=fmt, **_COMMON)

    assert _tree(tmp_path) == before


def test_yolo_overwrite_leaves_exactly_the_new_run_and_keeps_unrelated_files(tmp_path):
    generate_dataset(tmp_path, num_images=20, fmt="yolo", **_COMMON)
    (tmp_path / "README.md").write_text("keep me", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "note.txt").write_text("keep me too", encoding="utf-8")
    (tmp_path / "images" / "extra").mkdir()
    (tmp_path / "images" / "extra" / "note.txt").write_text("no run recorded images/extra", encoding="utf-8")

    counts = generate_dataset(tmp_path, num_images=10, fmt="yolo", overwrite=True, **_COMMON)

    for split, count in counts.items():
        assert len(list((tmp_path / "images" / split).iterdir())) == count
        assert len(list((tmp_path / "labels" / split).iterdir())) == count
    # Ownership comes from the manifest, which lists images/<split>, not the whole images/ tree.
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == sorted([*counts, "extra"])
    assert (tmp_path / "images" / "extra" / "note.txt").read_text(encoding="utf-8") == "no run recorded images/extra"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "keep me"
    assert (tmp_path / "notes" / "note.txt").read_text(encoding="utf-8") == "keep me too"


_THREE_SPLITS = SplitRatios(0.5, 0.25, 0.25)
_TWO_SPLITS = SplitRatios(0.5, 0.5, 0.0)


def test_yolo_overwrite_removes_a_split_the_new_run_does_not_write(tmp_path):
    generate_dataset(tmp_path, num_images=8, fmt="yolo", split_ratios=_THREE_SPLITS, **_COMMON)
    assert (tmp_path / "images" / "test").is_dir()

    counts = generate_dataset(tmp_path, num_images=8, fmt="yolo", split_ratios=_TWO_SPLITS, overwrite=True, **_COMMON)

    assert set(counts) == {"train", "val"}
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == ["train", "val"]
    assert sorted(path.name for path in (tmp_path / "labels").iterdir()) == ["train", "val"]
    assert "test:" not in (tmp_path / "data.yaml").read_text(encoding="utf-8")


def test_coco_overwrite_removes_a_split_the_new_run_does_not_write(tmp_path):
    generate_dataset(tmp_path, num_images=8, fmt="coco", split_ratios=_THREE_SPLITS, **_COMMON)
    (tmp_path / "other" / "deep").mkdir(parents=True)
    (tmp_path / "other" / "deep" / "_annotations.coco.json").write_text("{}", encoding="utf-8")
    (tmp_path / "other" / "keep.txt").write_text("keep me", encoding="utf-8")
    (tmp_path / "README.md").write_text("keep me", encoding="utf-8")

    counts = generate_dataset(tmp_path, num_images=8, fmt="coco", split_ratios=_TWO_SPLITS, overwrite=True, **_COMMON)

    assert set(counts) == {"train", "val"}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".vision-synth.json",
        "README.md",
        "other",
        "train",
        "val",
    ]
    assert (tmp_path / "other" / "keep.txt").read_text(encoding="utf-8") == "keep me"
    assert (tmp_path / "other" / "deep" / "_annotations.coco.json").is_file()


def test_a_stale_split_alone_does_not_block_a_rerun_without_overwrite(tmp_path):
    generate_dataset(tmp_path, num_images=8, fmt="coco", split_ratios=_THREE_SPLITS, **_COMMON)
    for split in ("train", "val"):
        shutil.rmtree(tmp_path / split)

    counts = generate_dataset(tmp_path, num_images=8, fmt="coco", split_ratios=_TWO_SPLITS, **_COMMON)

    assert set(counts) == {"train", "val"}
    assert (tmp_path / "test" / "_annotations.coco.json").is_file()


def test_coco_overwrite_leaves_exactly_the_new_run_and_keeps_unrelated_files(tmp_path):
    generate_dataset(tmp_path, num_images=20, fmt="coco", **_COMMON)
    (tmp_path / "README.md").write_text("keep me", encoding="utf-8")

    counts = generate_dataset(tmp_path, num_images=10, fmt="coco", overwrite=True, **_COMMON)

    for split, count in counts.items():
        doc = json.loads((tmp_path / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        assert len(doc["images"]) == count
        assert len(list((tmp_path / split).glob("*.jpg"))) == count
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "keep me"


def test_empty_split_dirs_and_unrelated_files_are_not_a_conflict(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "unrelated.txt").write_text("x", encoding="utf-8")

    counts = generate_dataset(tmp_path, num_images=3, fmt="yolo", **_COMMON)

    assert len(list((tmp_path / "images" / "train").iterdir())) == counts["train"]


def test_a_stray_data_yaml_blocks_yolo_until_overwrite(tmp_path):
    (tmp_path / "data.yaml").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"data\.yaml"):
        generate_dataset(tmp_path, num_images=2, fmt="yolo", **_COMMON)

    generate_dataset(tmp_path, num_images=2, fmt="yolo", overwrite=True, **_COMMON)
    assert (tmp_path / "data.yaml").read_text(encoding="utf-8").startswith("path: .")


@pytest.mark.parametrize("writer_cls", [YoloWriter, CocoWriter])
def test_direct_write_refuses_before_consuming_any_sample(tmp_path, writer_cls):
    vocabulary = class_vocabulary(ClassMode.SHAPE, DEFAULT_SHAPES)
    generator = SyntheticGenerator(SyntheticConfig(img_size=32))
    writer_cls(Task.DETECTION, vocabulary).write({"train": generator.generate(2, seed=0)}, tmp_path)
    consumed: list[int] = []

    def watched():
        for sample in generator.generate(2, seed=1):
            consumed.append(1)
            yield sample

    with pytest.raises(FileExistsError, match="overwrite=True"):
        writer_cls(Task.DETECTION, vocabulary).write({"train": watched()}, tmp_path)
    assert consumed == []


@pytest.mark.parametrize("writer_cls", [YoloWriter, CocoWriter])
@pytest.mark.parametrize("split", ["../escaped", "a/b", "..", ""])
def test_writers_refuse_split_names_that_leave_the_output_dir(tmp_path, writer_cls, split):
    vocabulary = class_vocabulary(ClassMode.SHAPE, DEFAULT_SHAPES)
    generator = SyntheticGenerator(SyntheticConfig(img_size=32))
    out = tmp_path / "ds"

    with pytest.raises(ValueError, match="split name"):
        writer_cls(Task.DETECTION, vocabulary).write({split: generator.generate(1, seed=0)}, out)
    assert not (tmp_path / "escaped").exists()
