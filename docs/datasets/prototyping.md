---
title: Prototyping and convergence checks
description: Prototype computer vision pipelines with synthetic COCO or YOLO data, check model convergence on a fixed tiny dataset, and scale difficulty with controlled experiments.
---

# Prototype a model with synthetic data

Use `vision-synth` to generate labelled images before collecting a real dataset: check your loader and targets, try to overfit a tiny training set, then measure performance as scenes become harder. It supplies the data; your training framework supplies the model, optimizer, loss, and metrics. No training framework beyond the package's base dependencies is needed to generate images.

## Choose a starting point

| Goal                                  | Data recipe                                                    | Check in your application                                                                                              |
| ------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Prototype a loader or model interface | 20 small images, one object each, `fmt="coco"` or `fmt="yolo"` | Load images, draw decoded labels, run a forward/backward step; loss and gradients must be finite.                      |
| Check whether training can converge   | Reuse a fixed set of 8–32 easy images                          | Training loss falls and predictions fit those same images. This tests the training path, not generalization.           |
| Check synthetic generalization        | Keep a separate validation set fixed while training            | Track held-out task metrics, per-class errors, and training/validation loss together.                                  |
| Find sensitivity to a nuisance        | Change only background, blur, clutter, or object size          | Compare the same checkpoint across fixed evaluation conditions.                                                        |
| Try curriculum learning               | Train on progressively harder configurations                   | Compare against a fixed-difficulty baseline at the same training budget. Advancement is controlled by your trainer.    |
| Scale the sample count                | Increase `num_images` or use `SyntheticIterableDataset`        | Measure loader throughput and memory separately from model quality. More samples do not themselves make scenes harder. |

These counts are starting budgets, not measured convergence requirements. Choose acceptance thresholds and a step limit for your model before running an experiment.

## 1. Export a small dataset

Install with `pip install vision-synth`. This complete example writes and checks a 20-image COCO detection dataset. Replace the temporary directory with a new output path to retain it for your trainer.

```python
import json
import tempfile
from pathlib import Path

from synth_datasets import generate_dataset

with tempfile.TemporaryDirectory() as out_dir:
    counts = generate_dataset(
        out_dir,
        num_images=20,
        fmt="coco",
        task="detection",
        class_mode="shape",
        img_size=128,
        min_objects=1,
        max_objects=1,
        min_size_ratio=0.20,
        max_size_ratio=0.35,
        rotate=False,
        seed=17,
    )
    root = Path(out_dir)
    for split, count in counts.items():
        labels = json.loads((root / split / "_annotations.coco.json").read_text())
        assert len(labels["images"]) == count
        assert len(labels["annotations"]) == count
        for item in labels["images"]:
            assert (root / split / item["file_name"]).is_file()

print(counts)
```

<details>
<summary>Image counts for the prototype dataset</summary>

```
{'train': 14, 'val': 4, 'test': 2}
```

</details>

Use `fmt="yolo"` for a `data.yaml` plus image/label directories. Change `task` to `segmentation` or `obb` for polygons or oriented boxes. For `task="keypoints"`, also choose shapes from one keypoint-bearing family: animals, symbols, or letters. Geometric primitives do not supply a pose schema. See [Tasks and keypoints](tasks.md) and [Annotation formats](outputs.md) before wiring targets into a model: COCO categories are 1-based, YOLO classes are 0-based, and COCO OBB corners are carried in `segmentation`.

## 2. Check convergence on fixed data

1. Inspect a few images with labels decoded by your actual loader. Confirm class names, box coordinates, image scaling, and any keypoint visibility handling.
2. Repeatedly train on the same tiny training subset. Initially disable random training augmentation and keep generated images fixed; fresh images on every pass would test a different question.
3. Record training loss and predictions at fixed step intervals. A decreasing loss alone is insufficient: boxes, masks, or landmarks must fit the targets too.
4. If fitting fails, inspect target conversion, loss inputs, gradients, learning rate, and model capacity before increasing data difficulty.
5. Once the tiny-set check passes, train on a larger set and evaluate on separate fixed validation images. Reserve the test split for the final comparison.

Use your task's metrics: box AP for detection, mask IoU/AP for segmentation, oriented-box IoU/AP for OBB, and a metric consistent with the family's landmark and visibility schema for pose. The package does not prescribe a universal loss threshold, step count, or guaranteed accuracy.

Keep split membership stable: changing `num_images` or split ratios in a seeded export can move samples between splits. Freeze an exported evaluation set for comparisons. For separate generation calls, avoid reusing the same seed and configuration for training and validation, which would replay samples. Record seeds, full configuration, package/dependency versions, and any source background files; a seed alone is not a cross-version reproducibility guarantee.

## 3. Increase difficulty deliberately

There is no `difficulty=` argument or automatic curriculum scheduler. Use `SyntheticConfig` fields. Start with one change at a time so a metric change has an interpretable cause; use the [suggested difficulty bands](difficulty.md) when you want combined stress conditions.

This paired evaluation recipe preserves the shape vocabulary and object placement while adding background noise. It checks that the pixels change and boxes remain aligned before you evaluate a model on the two conditions.

```python
from dataclasses import replace

import numpy as np

from synth_datasets import NoiseBackground, SyntheticConfig, SyntheticGenerator

baseline = SyntheticConfig(
    img_size=128,
    task="detection",
    class_mode="shape",
    min_objects=1,
    max_objects=1,
    min_size_ratio=0.20,
    max_size_ratio=0.35,
)
noisy = replace(baseline, background=NoiseBackground(sigma=12))
plain_samples = SyntheticGenerator(baseline).generate(4, seed=23)
noisy_samples = SyntheticGenerator(noisy).generate(4, seed=23)

paired = 0
for plain, changed in zip(plain_samples, noisy_samples, strict=True):
    for original, modified in zip(plain.annotations, changed.annotations, strict=True):
        assert original.bbox_xyxy == modified.bbox_xyxy
        assert original.class_id == modified.class_id
    assert not np.array_equal(plain.image, changed.image)
    paired += 1
assert paired == 4
print(paired)
```

<details>
<summary>Paired evaluation images with unchanged boxes</summary>

```
4
```

</details>

| Change                            | Config fields                                                 | Interpretation                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Background variation              | `background=NoiseBackground(...)` or `TextureBackground(...)` | Test separation from the canvas.                                                                                                        |
| Smaller targets                   | `min_size_ratio`, `max_size_ratio`, `img_size`                | Track object size in pixels (ratios are of the shorter canvas side); resizing by the trainer can erase small details.                   |
| More objects                      | `min_objects`, `max_objects`, `overlap_iou`                   | Increase scene density; placement constraints can yield fewer objects than requested.                                                   |
| Background clutter                | `distractors`, optional distractor shape/color pools          | Add unlabelled shapes behind targets.                                                                                                   |
| Partial occlusion                 | `occluders`                                                   | Boxes and polygons retain full-object geometry; covered keypoints become visibility `1`. These are not visible-only segmentation masks. |
| Camera effects                    | `degrade=(GaussianBlur(...), JPEG(...))`                      | Bake effects into pixels; generation does not resample them at each training step.                                                      |
| More classes or finer silhouettes | `shapes`, `class_mode`                                        | Changes the learning task and can renumber classes; keep vocabulary fixed within a curriculum.                                          |

Background, clutter, and degradation side streams preserve placement at a fixed seed. Size, shape, object-count, and placement changes do not carry that promise. The easy/moderate/hard bands change several factors, including vocabulary: their training-free statistics do not establish model convergence or an accuracy ranking.

For a curriculum, construct the next configuration in your trainer after an explicit validation criterion or epoch budget. Evaluate each stage on the same frozen suite and report regressions on easier cases. A single checkpoint tested across harder scenes measures robustness; retraining separately at each difficulty measures learnability. Keep those experiments distinct.

## 4. Stream when disk export is unnecessary

`SyntheticGenerator(config).generate(n, seed=...)` yields individual `Sample` objects. `SyntheticIterableDataset` integrates that stream with PyTorch; it does not convert samples to a model-specific target dictionary.

```python
import numpy as np
from torch.utils.data import DataLoader

from synth_datasets import SyntheticIterableDataset

dataset = SyntheticIterableDataset(
    num_images=8,
    img_size=64,
    task="detection",
    class_mode="shape",
    seed=31,
)
loader = DataLoader(dataset, batch_size=4, collate_fn=list, num_workers=0)
batch = next(iter(loader))
assert len(batch) == 4
assert all(sample.image.shape == (64, 64, 3) for sample in batch)
assert all(sample.image.dtype == np.uint8 for sample in batch)
print(batch[0].image.shape, batch[0].image.dtype)
```

<details>
<summary>Image layout before model preprocessing</summary>

```
(64, 64, 3) uint8
```

</details>

Convert the HWC RGB arrays and ragged annotations to your trainer's tensor and target contract. Keep `collate_fn=list` until that conversion is defined. For fresh streams each epoch, build a new dataset and loader with `epoch=...`; there is no `set_epoch()` method. See [distributed ranks and epochs](outputs.md#distributed-ranks-and-epochs) for worker seeding, per-rank counts, and persistent-worker limits.

Direct generation and YOLO export keep sample storage bounded; COCO export accumulates annotation metadata per split. DataLoader batches and worker prefetching add memory. See [streaming memory contracts](outputs.md#in-memory-streaming-and-training-feed) before sizing a large run.

## What a passing experiment establishes

A tiny-set pass shows that the tested training path can fit those examples. Held-out synthetic results measure performance under the chosen generator distribution. Neither establishes accuracy on photographs or production data. Use real held-out data for that decision, and keep synthetic runs as fast, reproducible development checks.
