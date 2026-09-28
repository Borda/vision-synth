---
title: Dataset scenes and outputs API
description: Generated reference for synth_datasets samples, annotations, shape families, backgrounds, baked degradations, and dataset writers.
---

# Dataset scenes and outputs API

This page covers what a generated scene is made of and how it reaches disk: the in-memory sample and annotation objects, the four shape vocabularies, the canvas and degradation knobs that set difficulty, and the writers that serialize COCO and YOLO.

For the generation entry points and configuration object, see [Dataset generation API](datasets-generation.md). Every knob below is pictured in [Customization and extension](../datasets/customization.md), and [Difficulty bands](../datasets/difficulty.md) combines them into three suggested settings.

## Samples and annotations

`SyntheticGenerator.generate` yields `Sample` objects. A sample carries every task representation, so a writer selects the subset its task needs and the generator never needs to know the output format.

::: synth_datasets.Sample
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.Annotation
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.SceneRecord
    options:
        show_root_heading: true
        show_source: false

## Shape families

Four independent vocabularies supply outlines: geometric primitives, traced animal silhouettes, symbols, and stroke letters. Keep one family per keypoint dataset — geometric primitives carry no keypoint schema. See [Shape families](../datasets/shapes.md) for the visual reference.

::: synth_datasets.ShapeFamily
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.family_of
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.shape_outline
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.keypoint_schema_for
    options:
        show_root_heading: true
        show_source: false

### Per-family shape enums

::: synth_datasets.families.primitives.PrimitiveShape
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.animals.AnimalShape
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.symbols.SymbolShape
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.families.letters.LetterShape
    options:
        show_root_heading: true
        show_source: false

### Keypoint schemas

::: synth_datasets.KeypointSchema
    options:
        show_root_heading: true
        show_source: false

## Backgrounds

`background` picks the canvas the objects land on. Each background draws from a side stream of its own rather than from the placement stream, so switching one on at a fixed seed cannot move an object.

::: synth_datasets.Background
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.SolidBackground
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.GradientBackground
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.NoiseBackground
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.ImpulseNoiseBackground
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.TextureBackground
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.ImageBackground
    options:
        show_root_heading: true
        show_source: false

## Baked degradations

`degrade` bakes camera effects into the exported pixels. Generation applies them once; it does not resample them at each training step.

::: synth_datasets.Degradation
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.GaussianBlur
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.GaussianNoise
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.JPEG
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.Contrast
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.ColorCast
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.Vignette
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.Quantize
    options:
        show_root_heading: true
        show_source: false

## Writers

A writer turns samples into files. `register_writer` adds a format under a new name; `get_writer` resolves one. See [Customization and extension](../datasets/customization.md) for a worked custom writer.

::: synth_datasets.DatasetWriter
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.CocoWriter
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.YoloWriter
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.get_writer
    options:
        show_root_heading: true
        show_source: false

::: synth_datasets.register_writer
    options:
        show_root_heading: true
        show_source: false
