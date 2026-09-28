"""COCO writer tests: JSON validity, categories, and per-task segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from synth_datasets.core.config import ClassMode, Color, SyntheticConfig, Task, class_vocabulary
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.core.keypoints import KeypointSchema
from synth_datasets.export.writers import CocoWriter
from synth_datasets.families import ALL_SHAPES, keypoint_schema_for
from synth_datasets.families.animals import ANIMAL_KEYPOINT_NAMES, ANIMAL_KEYPOINT_SCHEMA, AnimalShape
from synth_datasets.families.letters import LETTER_KEYPOINT_SCHEMA, LetterShape
from synth_datasets.families.symbols import SYMBOL_KEYPOINT_NAMES, SYMBOL_KEYPOINT_SCHEMA, SymbolShape


def _write(
    tmp_path: Path,
    writer_task: Task,
    count: int = 4,
    seed: int = 5,
    keypoint_schema: KeypointSchema | None = None,
    full_vocabulary: bool = False,
    **config_kwargs: object,
) -> tuple:  # type: ignore[type-arg]
    """Generate a seeded dataset, write it as COCO, and return its parsed JSON, samples, and class names.

    Every knob the individual tests vary — sample count, seed, and any `SyntheticConfig` field such as `shapes`,
    `img_size`, or `task` — is a keyword here, so a test states only what makes it different from the others. The
    writer's `task` is positional and separate from the config's: the writer formats whatever the samples already carry,
    so a detection-configured stream can still be written out as segmentation or OBB. `keypoint_schema` is only relevant
    to `Task.KEYPOINTS` runs and defaults to whichever family the config's own `shapes` name.

    The writer's vocabulary is narrowed to the config's own `shapes`, exactly as `generate_dataset` narrows it, so the
    ids the generator stamps index the very `categories` block written beside them. `full_vocabulary=True` hands the
    writer the full 49-shape list instead — for the one test that needs categories the run never draws.

    """
    settings = {"img_size": 96, "min_objects": 2, "max_objects": 4, **config_kwargs}
    config = SyntheticConfig(**settings)
    generator = SyntheticGenerator(config)
    rng = np.random.default_rng(seed)
    samples = [generator.sample(rng) for _ in range(count)]
    # `shapes` is required now, so "the full vocabulary" is spelled out as ALL_SHAPES rather than
    # signalled by omitting the argument.
    vocabulary = class_vocabulary(config.class_mode, ALL_SHAPES if full_vocabulary else config.shapes)
    names = vocabulary.names
    # A keypoints *writer* over a non-keypoints *config* is a case several tests here exercise on
    # purpose. The config's own shapes then name no schema, so the family has to be stated — which
    # is exactly what removing the writer's implicit animal default forces, and why the choice is
    # visible here instead of hidden in the writer.
    schema = keypoint_schema if keypoint_schema is not None else keypoint_schema_for(config.shapes)
    if schema is None and writer_task is Task.KEYPOINTS:
        schema = ANIMAL_KEYPOINT_SCHEMA
    writer_kwargs = {} if schema is None else {"keypoint_schema": schema}
    CocoWriter(writer_task, vocabulary, **writer_kwargs).write({"train": samples}, tmp_path)
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    return doc, samples, names


@pytest.mark.parametrize("task", [Task.DETECTION, Task.SEGMENTATION, Task.OBB])
def test_json_is_parseable_and_complete(tmp_path: Path, task: Task) -> None:
    doc, samples, names = _write(tmp_path, task)
    assert len(doc["images"]) == len(samples)
    assert len(doc["annotations"]) == sum(len(s.annotations) for s in samples)
    assert len(doc["categories"]) == len(names)
    # A stray per-category `keypoints` schema on a non-pose dataset would go unnoticed otherwise.
    assert all("keypoints" not in category for category in doc["categories"])


def test_category_ids_are_one_based(tmp_path: Path) -> None:
    """COCO category IDs are one-based."""
    doc, _, names = _write(tmp_path, Task.DETECTION)
    assert [c["id"] for c in doc["categories"]] == list(range(1, len(names) + 1))
    for ann in doc["annotations"]:
        assert 1 <= ann["category_id"] <= len(names)


def test_bbox_area_positive(tmp_path: Path) -> None:
    """Bounding box areas are positive."""
    doc, _, _ = _write(tmp_path, Task.DETECTION)
    for ann in doc["annotations"]:
        assert ann["area"] > 0
        assert len(ann["bbox"]) == 4


def test_detection_has_no_segmentation(tmp_path: Path) -> None:
    """Detection task does not include segmentation."""
    doc, _, _ = _write(tmp_path, Task.DETECTION)
    assert all("segmentation" not in ann for ann in doc["annotations"])


def test_segmentation_polygon_present(tmp_path: Path) -> None:
    """Segmentation task includes polygon."""
    doc, _, _ = _write(tmp_path, Task.SEGMENTATION)
    for ann in doc["annotations"]:
        assert "segmentation" in ann
        assert len(ann["segmentation"][0]) >= 6


def test_obb_stores_four_corner_polygon(tmp_path: Path) -> None:
    """OBB task stores four corner polygon."""
    doc, _, _ = _write(tmp_path, Task.OBB)
    for ann in doc["annotations"]:
        assert len(ann["segmentation"][0]) == 8


def test_keypoints_includes_segmentation_polygon(tmp_path: Path) -> None:
    """KEYPOINTS task includes the same outline polygon as SEGMENTATION.

    Real COCO person-keypoint annotations carry both `keypoints` and `segmentation`; a COCO consumer that gates instance
    retention on a parseable `segmentation` ring (as lit-YOLOs' `CocoDetectionDataset` does per its crowd/RLE-exclusion
    policy) would silently drop every keypoint annotation without this.

    """
    doc, _, _ = _write(tmp_path, Task.KEYPOINTS, task=Task.KEYPOINTS, shapes=tuple(AnimalShape))
    assert doc["annotations"]
    for ann in doc["annotations"]:
        assert "segmentation" in ann
        assert len(ann["segmentation"][0]) >= 6


def test_images_written_to_disk(tmp_path: Path) -> None:
    """Generated images are written to disk."""
    _write(tmp_path, Task.DETECTION)
    jpgs = sorted((tmp_path / "train").glob("*.jpg"))
    assert len(jpgs) == 4


@pytest.mark.parametrize("task", list(Task))
def test_image_ids_match_files(tmp_path: Path, task: Task) -> None:
    """Image IDs in JSON match files on disk."""
    doc, _, _ = _write(tmp_path, task)
    for image in doc["images"]:
        assert (tmp_path / "train" / image["file_name"]).exists()


@pytest.mark.parametrize("task", list(Task))
def test_animal_shape_dataset_round_trips(tmp_path: Path, task: Task) -> None:
    """A giraffe-only dataset writes valid COCO JSON for every task without writer changes.

    Animal outlines are concave and carry many more vertices than a square; this is the regression guard that the writer
    stays purely format-agnostic and needed no special casing.

    """
    doc, samples, names = _write(
        tmp_path, task, count=3, seed=31, img_size=192, max_objects=3, shapes=(AnimalShape.GIRAFFE,)
    )
    assert len(doc["annotations"]) == sum(len(s.annotations) for s in samples)
    giraffe_id = names.index(AnimalShape.GIRAFFE.value) + 1  # COCO category ids are 1-based
    assert {ann["category_id"] for ann in doc["annotations"]} == {giraffe_id}
    for ann in doc["annotations"]:
        assert ann["area"] > 0
        if task is Task.SEGMENTATION:
            assert len(ann["segmentation"][0]) >= 2 * 15
        elif task is Task.OBB:
            assert len(ann["segmentation"][0]) == 8
        elif task is Task.KEYPOINTS:
            assert len(ann["segmentation"][0]) >= 2 * 15


def test_keypoints_task_declares_the_sixteen_point_schema_and_fifteen_edge_skeleton(tmp_path: Path) -> None:
    """Animal categories carry all 16 landmark names and the 15-edge skeleton, 1-based.

    COCO viewers connect the dots via `skeleton`, which indexes into `keypoints` starting at 1 (not 0); getting that
    off-by-one wrong draws every edge one landmark short.

    """
    doc, _samples, names = _write(
        tmp_path,
        Task.KEYPOINTS,
        count=3,
        seed=11,
        img_size=192,
        max_objects=3,
        task=Task.KEYPOINTS,
        shapes=tuple(AnimalShape),
    )
    animal_values = {shape.value for shape in AnimalShape}
    animal_categories = [c for c, name in zip(doc["categories"], names, strict=True) if name in animal_values]
    assert animal_categories
    for category in animal_categories:
        assert category["keypoints"] == list(ANIMAL_KEYPOINT_NAMES)
        assert len(category["skeleton"]) == 15
        for i, j in category["skeleton"]:
            assert 1 <= i <= 16
            assert 1 <= j <= 16


def test_keypoints_task_leaves_categories_outside_the_run_family_without_a_keypoint_schema(tmp_path: Path) -> None:
    """Only the run's own keypoint-bearing family is decorated — every category outside it stays undecorated.

    A `CocoWriter` is handed whatever vocabulary its caller chooses, and a full-vocabulary one (`class_vocabulary` over
    `ALL_SHAPES`) covers categories the run's own family never includes: every geometric-shape category always, plus
    every category of the *other* keypoint-bearing family. A category the writer never draws a matching annotation for
    must not claim a landmark schema it can't back — checked here for an animal run (geometric categories undecorated)
    and a symbol run (both geometric *and* animal categories undecorated). `generate_dataset` narrows the vocabulary and
    so never hits this path, but the writer is public and must stay correct without it.

    """
    doc, _samples, names = _write(
        tmp_path,
        Task.KEYPOINTS,
        count=3,
        seed=11,
        img_size=192,
        max_objects=3,
        task=Task.KEYPOINTS,
        shapes=tuple(AnimalShape),
        full_vocabulary=True,
    )
    animal_values = {shape.value for shape in AnimalShape}
    geometric_categories = [c for c, name in zip(doc["categories"], names, strict=True) if name not in animal_values]
    assert geometric_categories
    for category in geometric_categories:
        assert "keypoints" not in category
        assert "skeleton" not in category

    symbol_doc, _symbol_samples, symbol_names = _write(
        tmp_path / "symbols",
        Task.KEYPOINTS,
        count=3,
        seed=11,
        img_size=192,
        max_objects=3,
        task=Task.KEYPOINTS,
        shapes=tuple(SymbolShape),
        keypoint_schema=SYMBOL_KEYPOINT_SCHEMA,
        full_vocabulary=True,
    )
    symbol_values = {shape.value for shape in SymbolShape}
    outside_symbol_family = [
        c for c, name in zip(symbol_doc["categories"], symbol_names, strict=True) if name not in symbol_values
    ]
    assert outside_symbol_family
    for category in outside_symbol_family:
        assert "keypoints" not in category
        assert "skeleton" not in category
    symbol_categories = [
        c for c, name in zip(symbol_doc["categories"], symbol_names, strict=True) if name in symbol_values
    ]
    assert symbol_categories
    for category in symbol_categories:
        assert category["keypoints"] == list(SYMBOL_KEYPOINT_NAMES)
        assert len(category["skeleton"]) == 6


def test_keypoints_color_class_mode_categories_carry_the_keypoint_schema(tmp_path: Path) -> None:
    """Under `ClassMode.COLOR` the categories are bare colors, and each one still declares the landmark schema.

    A color names no shape family, so a predicate asking "is this an animal?" answers no for `red`/`green`/`blue` and
    strips the schema off a dataset made entirely of animals: every annotation would ship a 16-landmark `keypoints`
    array while no category declared what those landmarks are, leaving a COCO consumer nothing to name them by. Asking
    "is this a geometric shape?" instead keeps the schema attached, which is the case this guards.

    """
    doc, _samples, names = _write(
        tmp_path,
        Task.KEYPOINTS,
        count=3,
        seed=11,
        img_size=192,
        max_objects=3,
        task=Task.KEYPOINTS,
        class_mode=ClassMode.COLOR,
        shapes=(AnimalShape.DUCK, AnimalShape.GIRAFFE),
    )
    assert names == [color.value for color in Color]
    for category in doc["categories"]:
        assert category["keypoints"] == list(ANIMAL_KEYPOINT_NAMES)
        assert len(category["skeleton"]) == 15
    # Every emitted annotation must land in a category that declares the schema its triples follow.
    decorated = {category["id"] for category in doc["categories"] if "keypoints" in category}
    assert doc["annotations"]
    assert {ann["category_id"] for ann in doc["annotations"]} <= decorated


def test_keypoints_num_keypoints_excludes_absent_landmarks(tmp_path: Path) -> None:
    """`num_keypoints` counts only `v>0` triples, so a whale's absent ear and hind-leg points are excluded.

    A whale's silhouette shows pectoral flippers but no hind legs, and a whale has no external ear, so it carries 11
    present landmarks (16 minus `ear` and the four `hind_knee_*`/`hind_limb_*` points); `num_keypoints` must reflect
    that, not the full schema length, or a consumer would expect a triple that never arrives.

    """
    doc, _samples, _names = _write(
        tmp_path,
        Task.KEYPOINTS,
        count=2,
        seed=3,
        img_size=192,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=(AnimalShape.WHALE,),
    )
    absent_names = {"ear", "hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"}
    for ann in doc["annotations"]:
        assert len(ann["keypoints"]) == 16 * 3
        assert ann["num_keypoints"] == 11
        flat = ann["keypoints"]
        triples = {name: flat[3 * i : 3 * i + 3] for i, name in enumerate(ANIMAL_KEYPOINT_NAMES)}
        for name in absent_names:
            assert triples[name] == [0.0, 0.0, 0]
        for name, triple in triples.items():
            if name in absent_names:
                continue
            assert triple[2] == 2


def test_letter_shape_dataset_emits_a_single_segmentation_ring(tmp_path: Path) -> None:
    """A letter's `segmentation` is one ring, exactly like every other single-polygon family.

    A letter is one outline polygon (see `synth_datasets.families.letters`), not a pile of disjoint stroke ribbons, so
    this must behave identically to a symbol or animal run rather than needing any letter-specific handling.

    """
    doc, _samples, names = _write(
        tmp_path,
        Task.SEGMENTATION,
        count=3,
        seed=7,
        img_size=192,
        max_objects=3,
        shapes=(LetterShape.X,),
    )
    assert doc["annotations"]
    x_id = names.index(LetterShape.X.value) + 1
    for ann in doc["annotations"]:
        assert ann["category_id"] == x_id
        assert len(ann["segmentation"]) == 1
        assert len(ann["segmentation"][0]) >= 6


def test_letter_categories_carry_their_own_per_letter_skeleton(tmp_path: Path) -> None:
    """Two different letter categories under one run get two different `skeleton` edge lists.

    Every member of the animal and symbol families shares one topology, so their category `skeleton` is identical across
    the whole family; a letter's stroke topology genuinely differs per letter (that's what makes it that letter), so
    `_category_shape_value`/`skeleton_for` must resolve each category to its own edges, not the family-wide fallback.

    """
    doc, _samples, names = _write(
        tmp_path,
        Task.KEYPOINTS,
        count=4,
        seed=9,
        img_size=192,
        max_objects=4,
        task=Task.KEYPOINTS,
        shapes=(LetterShape.I, LetterShape.X),
        keypoint_schema=LETTER_KEYPOINT_SCHEMA,
    )
    categories = {c["name"]: c for c in doc["categories"] if c["name"] in {"i", "x"}}
    assert set(categories) == {"i", "x"}
    assert len(categories["i"]["skeleton"]) == 5  # LetterShape.I has 5 stroke edges (stem + two serif bars)
    assert len(categories["x"]["skeleton"]) == 4  # LetterShape.X has 4 stroke edges
    assert categories["i"]["skeleton"] != categories["x"]["skeleton"][: len(categories["i"]["skeleton"])]
    # Every category still declares the shared 15-name landmark list, regardless of its own topology.
    assert names  # sanity: class_names resolved for this run
    for category in categories.values():
        assert len(category["keypoints"]) == 15


def test_keypoints_writer_task_with_non_keypoints_config_emits_the_all_zero_table(tmp_path: Path) -> None:
    """A KEYPOINTS-task writer over a plain detection-configured stream emits the zeroed placeholder table.

    The writer's `task` and the config's `task` are independent knobs (see `_write`'s docstring): the config here stays
    at the default `Task.DETECTION` with the four geometric shapes, so `SyntheticGenerator` never computes a landmark
    table and every `ann.keypoints` is `None`. `_keypoint_triples`' `ann.keypoints is None` fallback is what has to
    produce a well-formed all-zero, "not labeled" table here instead of raising or emitting a short list.

    """
    doc, samples, _names = _write(tmp_path, Task.KEYPOINTS)
    assert doc["annotations"]
    assert all(ann.keypoints is None for sample in samples for ann in sample.annotations)
    for ann in doc["annotations"]:
        assert ann["keypoints"] == [0.0, 0.0, 0] * 16
        assert ann["num_keypoints"] == 0
