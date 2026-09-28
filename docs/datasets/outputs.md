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

`data.yaml` lists the standard `train`, `val`, and `test` splits that were written:

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

Custom split names still produce `images/<split>/` and `labels/<split>/` directories. They are not additional `data.yaml` keys because the training layout uses the standard keys; point a consumer at the custom image directory explicitly. If `train` is absent, the first written split supplies the `train` path in `data.yaml`.

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

## Writing into an existing directory

Both writers refuse to write into a destination that already holds a dataset, so two runs never mix silently. `generate_dataset` raises `FileExistsError` before generating a single sample when a path the writer owns for the run's splits is populated: `<split>/` for COCO; `images/<split>`, `labels/<split>`, and `data.yaml` for YOLO.

### Replacing a dataset with `overwrite=True`

Every run records the paths it wrote in a `.vision-synth.json` manifest at the root of the output directory. `overwrite=True` replaces this run's own paths plus every path the manifest lists, so a split an earlier run wrote and this run does not is removed too. Every other file under the output directory is left alone. Ownership is never guessed from file names: a hand-curated COCO folder beside the output keeps its `_annotations.coco.json`, and without a manifest only this run's own paths are replaced. Paths are compared as the filesystem stores them: on a case-insensitive filesystem, respelling a split from `train` to `Train` replaces the one directory once, and the manifest records `Train`; on a case-sensitive one the two stay distinct.

The replacement is staged, never deleted first:

1. The new dataset, manifest included, is written into a `.vision-synth-staging-*` directory inside the output directory. A failure here removes only that staging directory; nothing else has moved.
2. A `.vision-synth-backup-*` directory is created, recording which paths the swap replaces and which it adds. Each path of the earlier dataset is renamed into it, and each staged entry is renamed into place, all on the same filesystem.
3. Once every staged entry is confirmed in place, the overwrite commits: the backup and staging directories are renamed to `.vision-synth-discard-*`, and only then removed.

Every staging and backup directory carries a `.vision-synth-owner.json` ownership marker (its kind, a random token matching the directory name, the tool version and the output directory), written atomically when the directory is created.

If the swap itself fails — a failed rename, or Ctrl+C — it is rolled back by renaming only, judged from what is on disk. Once a fresh listing shows every earlier path back in place, the emptied backup directory is removed with `os.rmdir`, which cannot delete a file, and the next run proceeds. The library never deletes anything it did not create in the same call.

Checks run before anything moves. `overwrite` refuses with `ValueError`, deleting nothing, when a path resolves outside the output directory or lies under a symlinked directory, when `.vision-synth.json` is not a regular file, or when a path to replace has a name starting with `.vision-synth` in any case. Split names must be single path components that Windows can also create — not `CON`, `NUL`, `aux.txt`, `COM1` or `LPT¹`, none of `< > " | ? *` or a control character, not ending in a dot or space — and must not start with `.vision-synth`, which the manifest and the staging, backup and discard directories use.

### Recovering from an interrupted overwrite

A run killed mid-overwrite, or one whose rollback could not finish, can leave reserved directories behind. Every later write into that output directory refuses with an error that classifies each one it finds, by its name prefix and its ownership marker, and gives the step below. The library never removes them itself.

| Found                                                                                 | What it means                                                                                                                                          | Safe action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.vision-synth-discard-*` with a valid marker                                         | The overwrite completed; the new dataset is in place. The directory holds only the superseded earlier dataset, or a staging directory's empty folders. | Delete it when you no longer need the earlier copy.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `.vision-synth-backup-*` with a valid marker                                          | The overwrite did not commit. The earlier dataset is inside at its original relative paths.                                                            | Replace a path only when its backup copy exists; otherwise the original is still in place. The refusal checks every path the marker records against the disk and lists what to do with each: where the backup has a copy and something sits at the path, remove that (the interrupted run's version) and move the copy there — removing first, because a plain `mv` onto an existing directory would nest the copy inside it; where the backup has a copy and the path is empty, move the copy there; where the backup has no copy, leave the path alone. An `added` path is removed only when it is present and has no backup copy; one that has a backup copy is left alone and reported. Each group is listed inline up to ten paths, the rest counted with a pointer to the marker. Then delete the emptied backup. |
| `.vision-synth-staging-*` with a valid marker                                         | Only generated files that were never moved into place.                                                                                                 | Delete it; if a backup is also there, restore from the backup first.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Any of these names without a valid marker                                             | Not created by vision-synth: an unrelated directory, or a split an older release allowed.                                                              | Rename it or move it out of the output directory. Do not delete it on this tool's say-so.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Any of these names whose marker exists but cannot be read (a permission or I/O error) | Ownership unknown: it may be a backup holding the only copy of the earlier dataset, or something else entirely.                                        | Do nothing to it. Fix access to the marker, then run again to learn what it is.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

Recovery only ever names paths the marker recorded; it never tells you to remove a path it did not create. A recorded path whose state cannot be read is listed as "state unknown" with no step for it, and an unreadable path never counts as missing: the overwrite, its rollback and the clean-up all stop instead of guessing.

## Command line

The `cli` extra installs a `vision-synth` command over `generate_dataset`:

```bash
pip install "vision-synth[cli]"
vision-synth generate ./shapes-ds 1000 --fmt yolo --task obb --shapes duck,camel --img_size 640,480 --seed 0
```

`output_dir` and `num_images` are positional; `--fmt`, `--split_ratios`, `--seed`, and `--overwrite` are `generate_dataset` options, and every other `--flag` sets the `SyntheticConfig` field of the same name. `--shapes` and `--colors` take comma-separated names. An invalid value, a wrongly typed value (the message names its flag), an unknown flag (the message lists the valid ones), or a populated output directory is reported as one line on stderr with exit status 1.

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

`SyntheticIterableDataset` treats `num_images` as a per-rank count. Pass an immutable `epoch` when constructing it; `rank` and `world_size` default to the default `torch.distributed` process group when one is initialized at construction (and to `0` and `1` otherwise), and can be passed explicitly. The dataset has no `set_epoch()` method. Every stream is seeded from `numpy.random.SeedSequence(seed, spawn_key=(rank, epoch, worker_id))`, so equal seeds stay reproducible within one rank and epoch, different ranks, epochs, and workers receive different streams, and adjacent seeds never overlap.

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
