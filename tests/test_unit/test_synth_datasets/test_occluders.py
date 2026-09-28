"""Pin the one addition that is allowed to touch a label, and the side-car that carries the evidence.

An occluder is drawn over a labelled object, so a landmark under it is no longer visible. COCO has a flag for exactly
that — ``1``, labelled but not visible — and the polygon keeps describing the whole object, which is what "amodal"
means. These tests hold both halves plus the record that publishes the mask, so a consumer wanting a modal mask has
something to subtract.

"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from synth_datasets.core.config import SyntheticConfig, Task
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.core.sample import Annotation, Sample, SceneRecord
from synth_datasets.families.animals import AnimalShape


def _keypoint_config(occluders: int) -> SyntheticConfig:
    """Return a keypoints run dense enough that some landmark lands under an occluder."""
    return SyntheticConfig(
        img_size=96,
        task=Task.KEYPOINTS,
        shapes=(AnimalShape.DUCK, AnimalShape.CAMEL, AnimalShape.GIRAFFE),
        min_objects=3,
        max_objects=3,
        min_size_ratio=0.2,
        max_size_ratio=0.35,
        occluders=occluders,
    )


def test_a_default_sample_carries_an_empty_scene_record() -> None:
    """Every sample has a scene record, and a run with no occluders fills none of its fields in.

    `None` rather than an all-`False` mask is the contract: a consumer can tell "no occluders were asked for" from
    "occluders were drawn and happened to miss this object".

    """
    sample = next(iter(SyntheticGenerator(SyntheticConfig(img_size=48)).generate(1, seed=0)))

    assert sample.scene.occluder_mask is None
    assert sample.scene.background_source is None


def test_a_sample_can_still_be_built_positionally() -> None:
    """The new field is last and defaulted, so every existing construction keeps working."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    sample = Sample(image, [], 4, 4)

    assert sample.scene.occluder_mask is None


def test_two_empty_records_are_distinct_objects_but_one_shared_default() -> None:
    """`SceneRecord` compares by identity, and the empty default is shared rather than rebuilt.

    A generated `__eq__` over an ndarray field raises `ValueError` instead of returning a bool, so identity comparison
    is the only well-defined choice; the shared default is safe because the record is empty and frozen, and it saves an
    allocation per sample.

    """
    first = next(iter(SyntheticGenerator(SyntheticConfig(img_size=32)).generate(1, seed=0)))
    second = next(iter(SyntheticGenerator(SyntheticConfig(img_size=32)).generate(1, seed=1)))

    assert first.scene is second.scene
    assert SceneRecord() != SceneRecord()


def test_a_populated_record_neither_raises_on_comparison_nor_hashes() -> None:
    """Identity comparison holds for a record carrying a raster, which a generated `__eq__` would not."""
    record = SceneRecord(occluder_mask=np.zeros((4, 4), dtype=bool))

    assert record == record
    assert record != SceneRecord(occluder_mask=np.zeros((4, 4), dtype=bool))


def test_the_mask_buffer_is_read_only() -> None:
    """A caller cannot mutate the mask, which would leave it disagreeing with the flags drawn from it.

    `frozen=True` stops the field being rebound and does nothing about the buffer it points at, so the array is marked
    read-only explicitly before it is published.

    """
    sample = next(iter(SyntheticGenerator(_keypoint_config(occluders=4)).generate(1, seed=0)))

    with pytest.raises(ValueError, match="read-only"):
        sample.scene.occluder_mask[0, 0] = True


def test_the_mask_has_the_canvas_shape_and_covers_something() -> None:
    """The published mask is a canvas-sized boolean raster with the occluders actually marked in it."""
    config = _keypoint_config(occluders=5)

    sample = next(iter(SyntheticGenerator(config).generate(1, seed=2)))

    assert sample.scene.occluder_mask.shape == (config.img_size, config.img_size)
    assert sample.scene.occluder_mask.dtype == np.bool_
    assert sample.scene.occluder_mask.any()


def test_occluders_demote_a_landmark_rather_than_dropping_it() -> None:
    """A covered landmark keeps its coordinates and takes COCO visibility 1, not 0.

    This is the whole point of the feature: the point is still labelled, the model is simply told it
    cannot be seen. Dropping it to 0 would throw away a coordinate the renderer knows exactly.

    """
    samples = list(SyntheticGenerator(_keypoint_config(occluders=8)).generate(4, seed=0))

    triples = [triple for s in samples for a in s.annotations for triple in a.keypoints]
    demoted = [triple for triple in triples if triple[2] == 1]

    assert demoted, "no landmark ended up under an occluder; the fixture is not exercising the path"
    assert all(x != 0.0 or y != 0.0 for x, y, _ in demoted)


def test_no_landmark_is_demoted_without_occluders() -> None:
    """Visibility 1 appears only because an occluder put it there, never on its own."""
    samples = list(SyntheticGenerator(_keypoint_config(occluders=0)).generate(4, seed=0))

    assert all(triple[2] in (0, 2) for s in samples for a in s.annotations for triple in a.keypoints)


def test_an_already_hidden_landmark_is_never_promoted() -> None:
    """A point the frame clipped stays at 0 whatever the mask says, since the mask cannot reach it."""
    samples = list(SyntheticGenerator(_keypoint_config(occluders=8)).generate(4, seed=1))

    zeroed = [triple for s in samples for a in s.annotations for triple in a.keypoints if triple[2] == 0]

    assert all(triple[:2] == (0.0, 0.0) for triple in zeroed)


def test_occluders_leave_polygons_and_boxes_untouched() -> None:
    """A COCO polygon describes the object, not its visible part, so an occluder moves neither field.

    That is what makes the exported mask amodal, and why `scene.occluder_mask` is published at all — a consumer wanting
    a modal mask subtracts it rather than asking the generator for one.

    """
    plain = list(SyntheticGenerator(_keypoint_config(occluders=0)).generate(3, seed=0))
    occluded = list(SyntheticGenerator(_keypoint_config(occluders=6)).generate(3, seed=0))

    assert [[a.bbox_xyxy for a in s.annotations] for s in plain] == [
        [a.bbox_xyxy for a in s.annotations] for s in occluded
    ]
    assert [[a.polygon for a in s.annotations] for s in plain] == [[a.polygon for a in s.annotations] for s in occluded]


def test_a_demoted_landmark_is_accepted_by_the_annotation_contract() -> None:
    """Visibility 1 is a valid COCO flag, so `Annotation` takes it without special-casing."""
    from synth_datasets.families.animals import ANIMAL_KEYPOINT_SCHEMA

    table = tuple((1.0, 2.0, 1) for _ in ANIMAL_KEYPOINT_SCHEMA.names)

    ann = Annotation(0, "duck", [], (0.0, 0.0, 2.0, 2.0), keypoints=table, keypoint_schema=ANIMAL_KEYPOINT_SCHEMA)

    assert ann.keypoints[0][2] == 1


def test_occluders_are_byte_reproducible_for_a_seed() -> None:
    """Two runs of one seed cover the same pixels, so a demoted flag is replayable."""
    config = _keypoint_config(occluders=5)

    first = next(iter(SyntheticGenerator(config).generate(1, seed=9)))
    second = next(iter(SyntheticGenerator(config).generate(1, seed=9)))

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.scene.occluder_mask, second.scene.occluder_mask)


def test_the_coco_writer_counts_a_demoted_landmark(tmp_path) -> None:
    """COCO's `num_keypoints` counts every labelled point, visible or not, so a demoted one still counts.

    The writers needed no change for this — `_keypoint_triples` branches on `visibility > 0` and the count does the same
    — and that is exactly the claim worth checking rather than assuming, since a writer that silently dropped visibility
    1 would export a structurally valid, quietly lossy file.

    """
    import json

    from synth_datasets import generate_dataset

    generate_dataset(tmp_path, num_images=6, fmt="coco", config=_keypoint_config(occluders=8), seed=0)
    records = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())["annotations"]

    flags = [record["keypoints"][2::3] for record in records]
    assert any(1 in row for row in flags), "no demoted landmark reached the exported file"
    counted = [sum(1 for flag in row if flag > 0) for row in flags]
    assert [record["num_keypoints"] for record in records] == counted


def test_the_mask_stays_read_only_across_a_pickle_round_trip() -> None:
    """The freeze must survive a process boundary, which every DataLoader worker crosses.

    `ndarray.__reduce__` does not carry the WRITEABLE flag, so without `SceneRecord.__setstate__` the mask came back
    writable — and `SyntheticIterableDataset` is a torch `IterableDataset`, so this is the ordinary path under
    `num_workers > 0`, not an exotic one. A guarantee that held only in the process that made it would be worse than
    none, since the docstring promises otherwise.

    """
    sample = next(iter(SyntheticGenerator(_keypoint_config(occluders=4)).generate(1, seed=0)))

    revived = pickle.loads(pickle.dumps(sample))  # noqa: S301 - round-tripping our own object

    assert revived.scene.occluder_mask.flags["WRITEABLE"] is False
    assert np.array_equal(revived.scene.occluder_mask, sample.scene.occluder_mask)
