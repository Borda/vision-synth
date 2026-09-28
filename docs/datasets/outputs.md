---
title: Annotation formats
description: COCO and YOLO dataset layouts, plus the in-memory streaming training feed.
---

# Annotation formats

## COCO output

Layout (Roboflow-style, one JSON per split):

```text
shapes_coco/
  train/
    img_000000.jpg
    img_000001.jpg
    _annotations.coco.json
  val/  …
  test/ …
```

`_annotations.coco.json` follows the standard COCO object-detection schema. Category ids are **1-based**:

```json
{
  "info": { "description": "vision-synth synthetic dataset" },
  "licenses": [],
  "categories": [
    { "id": 1, "name": "square", "supercategory": "none" },
    { "id": 2, "name": "rectangle", "supercategory": "none" }
  ],
  "images": [
    { "id": 0, "file_name": "img_000000.jpg", "width": 640, "height": 640 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 3,
      "bbox": [x, y, w, h],
      "area": 902.0,
      "iscrowd": 0
    }
  ]
}
```

Per task:

- **detection** — `bbox` `[x, y, w, h]` and `area`; no `segmentation` key.
- **segmentation** — adds `"segmentation": [[x1, y1, x2, y2, …]]`, the filled-shape polygon.
- **obb** — stores the four oriented-box corners as a 4-point `"segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]]` alongside the axis-aligned `bbox`. COCO has no native oriented-box field, so the corner polygon is the OBB carrier.

All coordinates are in pixels and clamped to the image extent.

## YOLO output

Layout (Ultralytics-style):

```text
shapes_yolo/
  images/{train,val,test}/img_000000.jpg
  labels/{train,val,test}/img_000000.txt
  data.yaml
```

`data.yaml` lists only the splits that were written:

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 4
names:
  0: square
  1: rectangle
  2: triangle
  3: circle
```

Each label file has one row per object. Class ids are **0-based**; all coordinates are normalized to `[0, 1]` and clamped:

| Task           | Row format                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `detection`    | `cls cx cy w h`                                                                                   |
| `segmentation` | `cls x1 y1 x2 y2 … xn yn`                                                                         |
| `obb`          | `cls x1 y1 x2 y2 x3 y3 x4 y4`                                                                     |
| `keypoints`    | `cls cx cy w h` + one `x y v` triple per schema landmark (53 tokens animal, 26 symbol, 50 letter) |

Example detection rows (`labels/train/img_000000.txt`):

```text
2 0.512000 0.334000 0.180000 0.210000
0 0.744000 0.618000 0.150000 0.150000
```

## In-memory streaming and training feed

Generation and writing are decoupled: `SyntheticGenerator` produces `Sample` objects, and writers persist them. Image **pixels** always stream: when you iterate `SyntheticGenerator` or drive a writer directly, only one `Sample` is materialized at a time, so you can feed a training loop with no disk round-trip and generation never holds more than one image in memory. This one-sample guarantee covers direct generator and writer iteration only — wrapping the source in a batching or multi-worker `DataLoader` (see below) is the exception. Label bookkeeping depends on the format: the YOLO writer emits one label file per image and the in-memory `SyntheticIterableDataset` retains nothing, so both stay memory-bounded regardless of `num_images`. The COCO writer emits a single JSON document per split, so it retains lightweight per-image and per-annotation metadata records (no pixels) in memory — O(n) in the split's image and annotation counts — until that split is written.

Iterate samples directly (no I/O):

```python
# phmdoctest:skip
import numpy as np
from synth_datasets import SyntheticConfig, SyntheticGenerator

gen = SyntheticGenerator(SyntheticConfig(img_size=256))
for sample in gen.generate(1000, seed=0):  # lazy: one Sample at a time
    train_step(sample.image, sample.annotations)
```

Or plug straight into a PyTorch `DataLoader` via `SyntheticIterableDataset` (exported from `synth_datasets.export.datasets`). This wrapper requires the `torch` extra; direct generation and iteration through `SyntheticGenerator` do not. Because object annotations are ragged (a variable number per image), pass a custom `collate_fn`; each batch is then a `list[Sample]`:

```python
from torch.utils.data import DataLoader

from synth_datasets import SyntheticIterableDataset

ds = SyntheticIterableDataset(num_images=8, img_size=64, class_mode="shape", seed=0)
loader = DataLoader(ds, batch_size=4, collate_fn=list)

print(sum(len(batch) for batch in loader))
```

<details>
<summary>Total samples yielded across the DataLoader batches</summary>

```
8
```

</details>

Set `num_workers>0` for multi-process loading: `SyntheticIterableDataset` is worker-shard aware, so each worker generates a disjoint, deterministically-seeded slice of `num_images`. Note that a `DataLoader` relaxes the one-sample bound: a batch materializes up to `batch_size` samples at once, prefetching holds `prefetch_factor` batches per worker, and each of `num_workers` workers materializes its own sample concurrently — so in-memory peak scales with `batch_size × num_workers`, not with a single image. When writing to disk, `generate_dataset` streams the same single sample source through per-split views, so image pixels never accumulate; peak memory then depends on the writer — bounded for YOLO, and O(n) COCO metadata (per the note above) for COCO.

### Distributed ranks and epochs

`SyntheticIterableDataset` treats `num_images` as a per-rank count. Pass the process `rank`, total `world_size`, and an immutable `epoch` when constructing it; the dataset does not inspect `torch.distributed` and has no `set_epoch()` method. Rank, epoch, and worker id are part of the seeded stream namespace, so equal seeds remain reproducible within one rank and epoch while different ranks and epochs receive different streams.

Build a fresh dataset and `DataLoader` for every epoch. Keep `persistent_workers=False`; persistent workers retain the old dataset instance and therefore cannot observe a new immutable epoch. This recipe also shows the per-rank count: each rank yields ten samples, so a two-rank job yields twenty total samples per epoch.

```python
from torch.utils.data import DataLoader

from synth_datasets import SyntheticIterableDataset

rank, world_size = 1, 2
persistent_workers = False
if persistent_workers:
    raise ValueError("build a fresh DataLoader per epoch with persistent_workers=False")


def loader_for_epoch(epoch: int) -> DataLoader:
    dataset = SyntheticIterableDataset(
        num_images=10,
        img_size=32,
        class_mode="shape",
        seed=7,
        rank=rank,
        world_size=world_size,
        epoch=epoch,
    )
    return DataLoader(
        dataset,
        batch_size=2,
        collate_fn=list,
        num_workers=2,
        persistent_workers=persistent_workers,
    )


for epoch in range(2):
    loader = loader_for_epoch(epoch)
    print(epoch, len(loader.dataset), sum(len(batch) for batch in loader))
```

<details>
<summary>Per-rank samples for two fresh epochs</summary>

```
0 10 10
1 10 10
```

</details>

The dataset intentionally does not promise mid-epoch replay, persistent-worker epoch propagation, topology-independent streams, or accelerator behavior. If persistent workers are required by an application, the application must own an explicit epoch-aware worker protocol rather than changing this dataset's immutable `epoch` property.
