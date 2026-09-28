---
title: Synthetic data experiments
description: Choose the synthetic-data experiment that answers your question, wire the exported labels into a trainer, and read what a passing run does and does not establish.
---

# Synthetic data experiments

`synth_datasets` supplies labelled images; your framework supplies the model, optimizer, loss, and metrics. That division decides which questions a synthetic run can answer.

This page routes you to the right experiment and covers the one step every route shares: decoding exported labels correctly. The runnable recipes live in [Prototyping and convergence checks](../datasets/prototyping.md).

## Which experiment answers your question

| Your question                               | Experiment                                                                            | Where                                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Is my loader and target conversion correct? | Export 20 small single-object images and decode them with your own loader             | [Export a small dataset](../datasets/prototyping.md#1-export-a-small-dataset)                                 |
| Can my training path fit anything at all?   | Overfit a fixed set of 8–32 easy images                                               | [Check convergence on fixed data](../datasets/prototyping.md#2-check-convergence-on-fixed-data)               |
| Which nuisance hurts my model?              | Change one config field and compare the same checkpoint across conditions             | [Increase difficulty deliberately](../datasets/prototyping.md#3-increase-difficulty-deliberately)             |
| Does my loader keep up with the model?      | Stream from `SyntheticIterableDataset` and measure throughput separately from quality | [Stream when disk export is unnecessary](../datasets/prototyping.md#4-stream-when-disk-export-is-unnecessary) |
| Which knob combinations are hard?           | Read the three published bands and their training-free statistics                     | [Difficulty bands](../datasets/difficulty.md)                                                                 |
| Which task and format do I need?            | Pick one of detection, segmentation, OBB, or keypoints and one of COCO or YOLO        | [Choose your computer vision task](../datasets/index.md#choose-your-computer-vision-task)                     |

Two of these are frequently conflated. A single checkpoint tested across harder scenes measures robustness; retraining separately at each difficulty measures learnability. Keep them distinct.

## Decode the exported labels before training

Both formats describe the same vocabulary, and both index it differently on disk: COCO categories are 1-based, YOLO classes are 0-based, and the in-memory `class_names` ordering is the 0-based one. A silent off-by-one here looks like a model that learned the wrong classes.

This example exports the same seeded dataset in both formats and checks the index bases against the shared vocabulary.

```python
import json
import tempfile
from pathlib import Path

from synth_datasets import DEFAULT_SHAPES, class_names, generate_dataset

names = class_names("shape", DEFAULT_SHAPES)

with tempfile.TemporaryDirectory() as out_dir:
    root = Path(out_dir)
    for fmt in ("coco", "yolo"):
        generate_dataset(
            root / fmt,
            num_images=10,
            img_size=96,
            fmt=fmt,
            task="detection",
            class_mode="shape",
            min_objects=1,
            max_objects=1,
            seed=5,
        )

    coco = json.loads((root / "coco" / "train" / "_annotations.coco.json").read_text())
    coco_names = [c["name"] for c in sorted(coco["categories"], key=lambda entry: entry["id"])]
    coco_ids = sorted(c["id"] for c in coco["categories"])
    yolo_classes = sorted({int(p.read_text().split()[0]) for p in (root / "yolo" / "labels").rglob("*.txt")})

    # COCO category order matches the in-memory vocabulary, offset by one.
    assert coco_names == names
    assert coco_ids == list(range(1, len(names) + 1))
    # YOLO rows index the same vocabulary directly.
    assert all(0 <= index < len(names) for index in yolo_classes)

print(len(names), coco_ids[0], names[coco_ids[0] - 1])
```

<details>
<summary>Vocabulary size, lowest COCO category id, and the class it names</summary>

```
4 1 square
```

</details>

Coordinate conventions differ between memory and disk. On an in-memory `Sample`, polygons and oriented-box corners are in pixel-centre space and axis-aligned boxes in pixel-edge space; the COCO and YOLO writers convert every point field to pixel-edge space at the file boundary, so an exported file uses one convention throughout. COCO carries OBB corners inside `segmentation`. See [Annotation formats](../datasets/outputs.md) before writing a target converter.

## Keep splits and seeds stable

Changing `num_images` or the split ratios in a seeded export can move samples between splits. Freeze an exported evaluation set for comparisons, and avoid reusing one seed and configuration for both training and validation — that replays the same samples. Record seeds, the full configuration, package and dependency versions, and any source background files; a seed alone is not a cross-version reproducibility guarantee.

Background, clutter, and degradation knobs each draw from a side stream of their own, so switching one on at a fixed seed cannot move an object. Size, shape, object-count, and placement changes carry no such promise.

## What a passing run establishes

A tiny-set pass shows the tested training path can fit those examples. Held-out synthetic results measure performance under the chosen generator distribution. Neither establishes accuracy on photographs or production data — use real held-out data for that decision, and keep synthetic runs as fast, reproducible development checks.

To augment generated samples with a fused geometric pipeline, continue to [Generate and augment](generate-and-augment.md).
