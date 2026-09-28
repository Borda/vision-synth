---
title: Synthetic data generation for computer vision
description: Generate synthetic COCO and YOLO datasets for computer vision prototyping, model convergence checks, and controlled difficulty experiments across four annotation tasks.
---

# Synthetic data generation for computer vision

`synth_datasets` draws labelled shapes for computer vision experiments. Export COCO or YOLO files for your trainer, or stream samples into a PyTorch DataLoader. Use it to prototype a pipeline, check whether a model can overfit a tiny dataset, and measure sensitivity to smaller objects, clutter, occlusion, or degraded images. The package generates data; your application owns training and evaluation.

`synth_datasets` is a standalone top-level package: it never imports the `fused_transforms` augmentation engine and needs no `torch` install for direct generation. It is also the only import path — the old `fused_transforms.data` facade and the `fused_transforms.generate_dataset` alias are both removed, so use `synth_datasets.families.geometry` and the other `synth_datasets.*` modules directly. `SyntheticIterableDataset` and PyTorch `DataLoader` integration require the `torch` extra.

**Start with [Prototyping and convergence checks](prototyping.md)** for runnable recipes and an experiment sequence: check the loader, fit fixed easy samples, evaluate held-out images, then increase difficulty.

It supports two annotation formats — **COCO** and **YOLO** — across four tasks — **detection**, **segmentation**, **oriented bounding box (OBB)**, and **keypoints / pose**.

Choose this generator when you need labelled test data without collecting or annotating images: a repeatable loader test, a small training experiment, or a controlled robustness evaluation. Objects are drawn primitives, animal silhouettes, symbols, and letters. Photorealistic scene generation and real-world accuracy validation are outside its scope.

How hard the samples are to read is set by ordinary config fields rather than by a difficulty preset: `background` picks the canvas, `degrade` bakes camera effects into the pixels, and `distractors` / `occluders` add unlabelled shapes under and over the labelled ones. Each is pictured in [Customization and extension](customization.md), and [Difficulty bands](difficulty.md) combines them into three suggested settings.

## Install

Rendering uses [Pillow](https://python-pillow.github.io/) and NumPy, both base dependencies — nothing extra to install, and no `torch` required:

```bash
pip install vision-synth
```

## Quickstart

```python
import tempfile

from synth_datasets import generate_dataset

with tempfile.TemporaryDirectory() as out_dir:
    counts = generate_dataset(
        out_dir,
        num_images=100,
        fmt="coco",
        task="detection",
        class_mode="shape",
        seed=0,
    )

print(counts)
```

<details>
<summary>Per-split image counts</summary>

```
{'train': 70, 'val': 20, 'test': 10}
```

</details>

Pass a real path instead of the temporary directory to keep the dataset. The same call with `fmt="yolo"` writes an Ultralytics-style layout; `generate_dataset` returns the number of images written per split.

## Choose your computer vision task

Keep the same `generate_dataset` call and select these arguments. Both formats support every listed task.

| You need                   | Arguments                                     | Generated labels                                          |
| -------------------------- | --------------------------------------------- | --------------------------------------------------------- |
| Object detection           | `task="detection"`                            | Axis-aligned boxes and classes                            |
| Instance segmentation      | `task="segmentation"`                         | One polygon per object and classes                        |
| Oriented object detection  | `task="obb"`                                  | Four oriented-box corners and classes                     |
| Keypoint / pose estimation | `task="keypoints", shapes=tuple(AnimalShape)` | Boxes, classes, and the animal landmark/visibility schema |

For the pose recipe, import `AnimalShape` from `synth_datasets.families.animals`. Symbols and letters also support pose, each with its own schema; use one family per keypoint dataset. The default geometric primitives support detection, segmentation, and OBB.

Use `fmt="coco"` for per-split `_annotations.coco.json` files or `fmt="yolo"` for normalized text labels plus `data.yaml`. Set `class_mode="shape"` to predict shape names; `"color"` and `"shape_color"` select other class vocabularies. Coordinate conventions and the COCO OBB representation are documented in [Annotation formats](outputs.md).

### Runnable YOLO recipes for all four tasks

This example creates ten images for each task, checks the exported labels and dataset manifest, then cleans up. Use a persistent output directory for a training run.

```python
import tempfile
from pathlib import Path

from synth_datasets import generate_dataset
from synth_datasets.families.animals import AnimalShape

with tempfile.TemporaryDirectory() as out_dir:
    for task in ("detection", "segmentation", "obb", "keypoints"):
        destination = Path(out_dir) / task
        task_options = {"shapes": tuple(AnimalShape)} if task == "keypoints" else {}
        counts = generate_dataset(
            destination,
            num_images=10,
            img_size=128,
            fmt="yolo",
            task=task,
            class_mode="shape",
            min_objects=1,
            max_objects=1,
            seed=0,
            **task_options,
        )
        assert counts == {"train": 7, "val": 2, "test": 1}
        assert (destination / "data.yaml").is_file()
        labels = sorted((destination / "labels").rglob("*.txt"))
        assert len(labels) == 10
        for label in labels:
            rows = label.read_text().splitlines()
            assert len(rows) == 1
            fields = rows[0].split()
            if task == "segmentation":
                assert len(fields) >= 7 and (len(fields) - 1) % 2 == 0
            else:
                assert len(fields) == {"detection": 5, "obb": 9, "keypoints": 53}[task]
        print(task, sum(counts.values()))
```

<details>
<summary>Images exported for each YOLO task</summary>

```
detection 10
segmentation 10
obb 10
keypoints 10
```

</details>

## Adjust the dataset to your experiment

| Requirement                    | Set or use                                                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Repeat the same data           | `seed=0`; keep configuration and environment fixed                                                                    |
| Change dataset size            | `num_images=...` (total across splits)                                                                                |
| Change image resolution        | `img_size=...` (an `int` side, or `(width, height)`)                                                                  |
| Increase difficulty            | Object size, `background`, `degrade`, `distractors`, `occluders`; see [difficulty bands](difficulty.md)               |
| Choose classes and silhouettes | `shapes`, `colors`, `class_mode`; see [shape families](shapes.md)                                                     |
| Generate from the shell        | `vision-synth generate` (the `cli` extra); see [command line](outputs.md#command-line)                                |
| Feed a custom training loop    | `SyntheticGenerator` or `SyntheticIterableDataset`; see [streaming](outputs.md#in-memory-streaming-and-training-feed) |

All generation APIs accept either a full `SyntheticConfig` or its individual fields. With `generate_dataset`, pass one form only: `config=...` cannot be combined with content keywords such as `task=` or `img_size=`. Dataset generation does not train a model or schedule a curriculum; follow [Prototyping and convergence checks](prototyping.md) for those experiments.

## In this section

- [Prototyping and convergence checks](prototyping.md) — small exports, tiny-set overfitting, held-out evaluation, controlled difficulty changes, and streaming inputs for agents and developers.
- [Shape families](shapes.md) — the four vocabularies (geometric, animals, symbols, letters), the visual shape reference, and how to select shapes and colors.
- [Tasks and keypoints](tasks.md) — the four annotation tasks and every family's keypoint schema.
- [Annotation formats](outputs.md) — COCO and YOLO on-disk layouts, plus the in-memory streaming training feed.
- [Difficulty bands](difficulty.md) — suggested easy/moderate/hard knob combinations, pictured, and the training-free statistics that rank them.
- [Customization and extension](customization.md) — reproducibility knobs, custom splits and fills, registering new families and writers, and editing the packaged assets.
