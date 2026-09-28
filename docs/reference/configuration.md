---
title: Configuration API
description: Build augmentation-backend-free and declarative backend-specific pipelines.
---

# Configuration API

Use numeric parameters when the pipeline needs no augmentation-backend objects. Use `TransformSpec` when the pipeline belongs in JSON, YAML, Hydra, or experiment configuration.

PyTorch remains a required runtime dependency in both cases.

## Backend-free parameters

<!--phmdoctest-share-names-->

```python
from fused_transforms import Compose

pipe = Compose.from_params(
    rotation=(-20.0, 20.0),
    scale=(0.85, 1.15),
    shear_x=(-5.0, 5.0),
    translate_x=(-8.0, 8.0),
    hflip_p=0.5,
    brightness=0.1,
    contrast=0.1,
)
```

The direct path supports rotation, uniform/per-axis scale, x/y shear, x/y pixel translation, H/V flips, brightness, and contrast. It samples direct geometric parameters independently per item; the `randomness` value is stored but does not change that direct sampling behavior.

`brightness=0.1` and `contrast=0.1` are active features: each describes a multiplicative factor range centered on `1.0`, here `[0.9, 1.1]`. Pass a `(low, high)` tuple instead to set the factor range directly, matching the tuple form of `rotation=` and `scale=`. Both work for any channel count, including grayscale and multi-spectral inputs.

`rotation_p` and `scale_p` gate rotation and the scale family independently, each with its own draw per sample, and the operations always run in the order rotation, scale, shear, translate, whether or not they are gated.

## Declarative specs

<!--phmdoctest-share-names-->

```python
from fused_transforms import Compose, TransformSpec

specs = [
    TransformSpec(
        operation="rotation",
        params={"degrees": (-20.0, 20.0)},
        prob=0.8,
    ),
    TransformSpec(operation="hflip", params={}, prob=0.5),
]

torchvision_pipe = Compose.from_config(specs, backend="torchvision")
native_pipe = Compose.from_config(specs, backend="native")
```

`TransformSpec` freezes its parameter mapping and validates `prob` in `[0, 1]`. Use `to_dict()` and `from_dict()` for JSON/YAML round trips. `from_dict()` restores tuples for known numeric range keys; it does not convert every arbitrary list to a tuple.

Do not put `prob` inside `params`; use the dedicated field.

## Capabilities are backend-specific

```python
from fused_transforms import Compose

if "rotation90" in Compose.supported_ops("albumentations"):
    ...
```

The global operation vocabulary is larger than any one backend's coverage. See the exact [declarative construction matrix](../concepts/capabilities.md#declarative-construction-matrix).

By default, `from_config` validates all specs before constructing transforms and reports every unsupported item together. To explicitly build the supported subset:

```python
pipe = Compose.from_config(
    specs,
    backend="torchvision",
    on_unsupported="warn_skip",
)
```

Skipping changes the requested pipeline, so use it only when a warning and partial pipeline are acceptable.

## Canonical parameters and portability

The resolver translates a small canonical parameter vocabulary, including common rotation, scale, shear, translation, brightness, and contrast names. Unrecognized keys are passed to the backend constructor. A spec that contains backend-only keys is not portable merely because its operation name is canonical.

Treat backend swapping as validated reconstruction, not an assurance of identical random streams, interpolation numerics, defaults, or parameter semantics.

## Default reordering

`from_params` and `from_config` default to `ReorderPolicy.POINTWISE`, unlike direct `Compose(...)`, which defaults to `NONE`.

Use explicit `ReorderPolicy.NONE` when declaration order must be preserved:

```python
from fused_transforms import Compose, ReorderPolicy

pipe = Compose.from_config(
    specs,
    backend="kornia",
    reorder=ReorderPolicy.NONE,
)
```

## `TransformSpec`

::: fused_transforms.TransformSpec
    options:
        show_root_heading: true
        show_source: false
        members:
            - to_dict
            - from_dict

## Configuration policies

::: fused_transforms.ReorderPolicy
    options:
        show_root_heading: true
        show_source: false

::: fused_transforms.RandomnessPolicy
    options:
        show_root_heading: true
        show_source: false
