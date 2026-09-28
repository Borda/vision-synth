---
title: Dataset generation API
description: Generated reference for synth_datasets generation entry points, configuration, tasks, formats, and class vocabularies.
---

# Dataset generation API

`generate_dataset` writes a split dataset to disk. `SyntheticGenerator` yields samples in memory. Both read the same `SyntheticConfig`, so a configuration proven on one path transfers to the other.

Everything on this page is importable from the `synth_datasets` root namespace and needs no `torch` install. The one exception is `SyntheticIterableDataset`, documented at the end of this page.

For narrative guides, start with [Synthetic data generation](../datasets/index.md); for the annotation objects, backgrounds, degradations, shape families, and writers, see [Dataset scenes and outputs API](datasets-scenes.md).

## Entry points

### `generate_dataset`

::: synth_datasets.generate_dataset
    options:
        show_root_heading: true
        show_source: false

### `SyntheticGenerator`

::: synth_datasets.SyntheticGenerator
    options:
        show_root_heading: true
        show_source: false

### Command line

`vision-synth generate` (the `cli` extra) calls this function; see [command line](../datasets/outputs.md#command-line).

::: synth_datasets.cli.generate
    options:
        show_root_heading: true
        show_source: false

## Configuration

`SyntheticConfig` is the single configuration object. `generate_dataset` accepts either a full config or its individual fields, but not both in one call.

::: synth_datasets.SyntheticConfig
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.SplitRatios
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.core.config.as_canvas_size
    options:
        show_root_heading: true
        show_source: false

## Tasks, formats, and fills

::: synth_datasets.Task
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.OutputFormat
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.Fill
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.Color
    options:
        show_root_heading: true
        show_source: false

## Class vocabularies

`class_mode` selects which vocabulary the exported class indices follow. COCO categories are 1-based on disk and YOLO classes are 0-based; the helpers below report the 0-based in-memory indices. See [Annotation formats](../datasets/outputs.md) for the on-disk conventions.

::: synth_datasets.ClassMode
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.ClassEntry
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.ClassVocabulary
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.class_vocabulary
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.class_names
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.class_id
    options:
        show_root_heading: true
        show_source: false

## Coordinate helpers

`synth_datasets.families.geometry` holds the pixel-space conversions the writers and annotations rely on. Polygons and oriented-box corners are in pixel-centre space; axis-aligned boxes are in pixel-edge space. The two differ by half a pixel each way and are not interchangeable.

::: synth_datasets.families.geometry.to_pixel_centre
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.geometry.to_pixel_edge
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.geometry.polygon_to_bbox_xyxy
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.geometry.polygon_to_obb
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.geometry.rotate_polygon
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.geometry.bbox_iou
    options:
        show_root_heading: true
        show_source: false

## PyTorch streaming

`SyntheticIterableDataset` is the only torch-dependent name in the namespace. It is resolved lazily on first attribute access, so `import synth_datasets` stays torch-free; using the class requires `pip install "vision-synth[torch]"`. There is no `set_epoch()` method — construct a new dataset with `epoch=...` for a fresh stream. See [distributed ranks and epochs](../datasets/outputs.md#distributed-ranks-and-epochs).

::: synth_datasets.export.datasets.SyntheticIterableDataset
    options:
        show_root_heading: true
        show_source: false
