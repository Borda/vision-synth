---
title: Reproducible augmentation pipelines
description: Control PyTorch, NumPy, Albumentations, DataLoader, reorder, and execution settings for reproducible vision-synth experiments.
---

# Reproducible augmentation pipelines

Reproducibility requires more than one seed. Record the pipeline definition, backend versions, tensor shapes, execution strategy, reorder policy, and every random-number domain that the selected transforms consume.

## Short answer

- Backend-free, Kornia, and TorchVision fused paths primarily consume PyTorch randomness.
- Albumentations fused geometry consumes both global NumPy randomness for package activation gates and each transform's internal random generator for parameters.
- `generator=` hands every draw on the direct-parameter path to a caller-owned `torch.Generator`, so a pipeline no longer shares the global stream with the rest of the process.
- `randomness="backend"` preserves batch-sampling style; it does not promise an identical native random stream or pixel-identical output.
- Batch size, fast-path selection, skipped transforms, and backend version can change random draw consumption.

## Randomness matrix

| Pipeline path                                  | Activation/probability source | Parameter source            | Required control                                                        |
| ---------------------------------------------- | ----------------------------- | --------------------------- | ----------------------------------------------------------------------- |
| `Compose.from_params` without a backend        | PyTorch                       | PyTorch                     | `torch.manual_seed` and fixed batch/shape/order                         |
| `Compose.from_params(..., generator=...)`      | Caller generator              | Caller generator            | Pass one `torch.Generator`; global seeding no longer affects the result |
| Kornia transforms                              | PyTorch                       | Kornia through PyTorch      | `torch.manual_seed`; record Kornia version and `same_on_batch` settings |
| TorchVision transforms                         | PyTorch                       | TorchVision through PyTorch | `torch.manual_seed`; record v1/v2, batch shape, and randomness policy   |
| Albumentations transforms on fused tensor path | Global NumPy                  | Transform-internal RNG      | Seed `numpy.random` and call the transform's supported seed method      |
| Albumentations native NumPy path (HWC in)      | Albumentations/native path    | Transform-internal RNG      | Use Albumentations' supported seeding API and record its version        |

The native NumPy path and the tensor path consume the same draws from `numpy.random`, so a seed is portable across input types: the same pipeline under the same seed samples the same geometry whether it is called with an array or a tensor. Two smaller differences remain and are bounded rather than eliminated. Their composed matrices can differ by up to one float32 epsilon (1.19e-07) on a multi-op chain, because the NumPy path multiplies the float64 matrices Albumentations produces while the adapter path rounds each factor through float32 — that moves a 1024-pixel coordinate by 1e-04 of a pixel. And they *render* differently by up to one intensity level, because the NumPy path warps the caller's `uint8` array directly, as Albumentations does, while the tensor path warps a float32 copy. That one level is the quantisation and holds under OpenCV 5; under OpenCV 4 the integer resampler itself differs and the same comparison measures up to four levels, so an environment pinned below OpenCV 5 — which includes anything still on numpy 1.x, since OpenCV 5 requires numpy 2 — should expect the wider gap. Record the input type alongside the seed when comparing renders; when comparing draws, it does not matter.

## A deterministic backend-free run

Use an explicit order-preserving policy and reset the seed before building or calling each comparison pipeline:

```python
import torch

from fused_transforms import Compose, ReorderPolicy


def run_once(seed: int, images: torch.Tensor) -> torch.Tensor:
    """Build and execute one reproducible backend-free augmentation call."""
    torch.manual_seed(seed)
    pipe = Compose.from_params(
        rotation=(-20.0, 20.0),
        scale=(0.9, 1.1),
        hflip_p=0.5,
        reorder=ReorderPolicy.NONE,
    )
    return pipe(images.clone())


images = torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(7))
first = run_once(1234, images)
second = run_once(1234, images)
torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
```

Keep the batch size and input shape fixed when asserting byte-for-byte reproduction. Some optimized paths consume draws differently when batch size or the active subset changes.

## Caller-owned randomness with `generator=`

Global seeding is process-wide: any other component drawing from the torch stream between two calls changes what the pipeline samples. Passing a `torch.Generator` moves every pipeline-owned draw — geometry, colour factors, and the per-transform probability gates — onto a stream the caller owns:

```python
import torch

from fused_transforms import Compose, ReorderPolicy

images = torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(7))


def run_seeded(seed: int) -> torch.Tensor:
    """Run one pipeline whose every draw comes from a caller-owned generator."""
    pipe = Compose.from_params(
        rotation=(-20.0, 20.0),
        hflip_p=0.5,
        reorder=ReorderPolicy.NONE,
        generator=torch.Generator().manual_seed(seed),
    )
    return pipe(images.clone())


first = run_seeded(1234)
torch.manual_seed(999)
torch.rand(1024)  # unrelated global draws between the two runs
second = run_seeded(1234)
torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
```

Scope and limits:

- Supported on the direct-parameter engine: `Compose.from_params(...)` with `backend=None` and with `backend="native"`.
- Kornia, TorchVision, and Albumentations transforms sample inside their own libraries, and none of those samplers accept a `torch.Generator`. Passing one together with backend transforms raises at construction instead of falling back to the global stream, because a silent fallback would look reproducible without being reproducible. Seed those backends as described in the sections below.
- `generator=None` (the default) keeps the historical global-stream behaviour unchanged, so existing pipelines need no edit.
- A generator on any device seeds a pipeline on any device: the draw happens on the generator's device and is copied to the tensors' device.
- Pickling a pipeline (which is what a DataLoader worker receives) carries the generator's state as of pickling, so every worker would otherwise replay one identical stream. Give each worker its own seed. The state travels by value — the unpickled pipeline holds a *new* generator at the same state, so re-seeding the original object leaves the restored pipeline untouched; seed `pipe.generator` on the pipeline the worker actually holds.

## Seed Albumentations' two RNG domains

For the fused Albumentations tensor path, seed global NumPy for package probability gates and seed each transform's internal generator for parameters:

```python
import numpy as np
import torch
import albumentations as A

from fused_transforms import Compose, ReorderPolicy


def build_albumentations_pipe(seed: int) -> Compose:
    """Build an Albumentations-backed pipeline with both RNG domains seeded."""
    transforms = [
        A.Rotate(limit=(-15, 15), p=0.8),
        A.HorizontalFlip(p=0.5),
    ]
    for index, transform in enumerate(transforms):
        transform.set_random_seed(seed + index)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Compose(transforms, reorder=ReorderPolicy.NONE)


images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(9))
first = build_albumentations_pipe(123)(images.clone())
second = build_albumentations_pipe(123)(images.clone())
torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
```

Calling `np.random.seed` alone does not reset an existing Albumentations 2.x transform's internal parameter stream. Conversely, seeding only the transform does not control the package's NumPy activation gates.

## DataLoader workers

PyTorch assigns a distinct base seed to each worker, but custom NumPy and backend-owned generators must be seeded explicitly. The exact dataset structure is application-specific; this pattern assumes the dataset stores the pipeline as `dataset.augment`:

```python
# phmdoctest:skip
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info


def seed_augmentation_worker(worker_id: int) -> None:
    """Seed Python, NumPy, the pipeline generator, and Albumentations transforms in one worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()
    if worker_info is None:
        return

    pipeline = worker_info.dataset.augment
    # A caller-owned generator is pickled into every worker with the same state,
    # so without this each worker replays one identical parameter stream.
    if pipeline.generator is not None:
        pipeline.generator.manual_seed(worker_seed)
    for index, transform in enumerate(pipeline.original_transforms):
        set_seed = getattr(transform, "set_random_seed", None)
        if callable(set_seed):
            set_seed((worker_seed + index) % (2**32))


generator = torch.Generator().manual_seed(2026)
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_augmentation_worker,
    generator=generator,
)
```

Adapt the `dataset.augment` lookup to your dataset. Constructing a fresh pipeline inside every `__getitem__` call has a different state and cost model; document that choice if you use it.

For `SyntheticIterableDataset`, reproducibility also includes distributed identity. Pass an immutable `epoch`; `rank` and `world_size` come from the initialized `torch.distributed` process group unless passed explicitly. `num_images` is the count produced by each rank, and every stream is seeded from `SeedSequence(seed, spawn_key=(rank, epoch, worker_id))`. Construct a fresh dataset and `DataLoader` per epoch with `persistent_workers=False`, as shown in [the streaming dataset recipe](../datasets/outputs.md#distributed-ranks-and-epochs). The dataset has no `set_epoch()`, so persistent workers cannot receive a new epoch implicitly.

## Reorder and execution settings are part of the experiment

`ReorderPolicy.POINTWISE` can change results because color operations do not always commute with padding, interpolation, and clipping. `from_params` and `from_config` default to `POINTWISE`; set `NONE` explicitly when comparing runs or native references.

Albumentations `execution="cv2"` and `execution="torch"` can use the same sampled matrices but differ in border and subpixel numerics. Record the chosen strategy. Also record `mask_interpolation`, `padding_mode`, `clip_policy`, `antialias`, and `compile`.

### Which engine to pin

- `execution="cv2"` (the default) is the fastest choice on the host and is bit-exact with the native cv2 backend. On a detection-shaped step it measured five to sixteen times faster than `grid_sample` on the same CPU.
- `execution="torch"` is the explicit engine for accelerator execution and differentiable registered warps. On CPU, a batch-one tensor with `requires_grad=True` automatically bypasses the cv2 fast path and uses the differentiable torch route; pin `execution="torch"` when the engine choice itself must stay stable across batch sizes. The cv2 path remains the faster choice for ordinary non-differentiable host work. A CPU-only Albumentations passthrough still transfers an accelerator tensor to the host and back; this setting avoids the fused warp's round trip, not opaque native transforms.
- `execution="auto"` resolves per call: host data to cv2, accelerator data to torch. **Choosing it opts out of bit-reproducibility across environments** — the engine becomes a function of where the tensors live rather than of the recorded configuration, so the same config on two machines can render differently and nothing in the config explains why. It is never the default, and it is the only setting under which `pipeline.resolved_execution` can differ from what you configured. Read that attribute after a call when you need to know which engine drew the pixels; it is `None` until the first call warps something.

A pipeline pickled with `"auto"` re-resolves on unpickling rather than freezing the engine it happened to pick. A `DataLoader` worker that receives it therefore routes for its own data, which is what asking for `"auto"` means; if you need every worker pinned to one engine, pin the engine instead.

This recipe keeps the transfer decision visible without requiring an accelerator. It selects CUDA when available, pins the registered warp to `"torch"`, and leaves a CPU-only color passthrough visible in the plan:

```python
import albumentations as A
import torch

from fused_transforms import Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = Compose(
    [A.Affine(rotate=(15.0, 15.0), p=1.0), A.HueSaturationValue(p=1.0)],
    execution="torch",
)
images = torch.rand(2, 3, 32, 32, device=device)
output = pipe(images)

assert output.device == device
print(pipe.fusion_plan)
```

<details>
<summary>Registered warp plus native CPU passthrough</summary>

```
fused(Affine) → passthrough(HueSaturationValue)
```

</details>

On a CUDA run, the passthrough is the transfer boundary; replace it with a registered torch-native operation or accept and measure the host round trip.

`Compose` is an `nn.Module`, so register forward pre- and post-hooks on the pipeline when instrumentation or result replacement is part of your contract. Public hooks run for exact, fused, and general tensor routes, including tensor keyword and mixed image/target calls. Hooks registered on internal segment modules are a separate implementation detail and are not dispatched by the top-level segmented call.

## Record the effective pipeline

Save enough information to explain a future mismatch:

```python
import json
import torch

from fused_transforms import Compose

torch.manual_seed(7)
images = torch.rand(2, 3, 32, 32)
pipe = Compose.from_params(rotation=(-15.0, 15.0))

output, last_matrix = pipe(images, return_matrix=True)
plan = [descriptor.to_dict() for descriptor in pipe.fusion_plan_descriptors]

run_metadata = {
    "fusion_plan": plan,
    "last_matrix_shape": None if last_matrix is None else list(last_matrix.shape),
    "torch_version": torch.__version__,
}
print(json.dumps({**run_metadata, "torch_version": "<runtime>"}, indent=2))
```

<details>
<summary>Recorded pipeline metadata with the runtime version normalized</summary>

```
{
  "fusion_plan": [
    {
      "kind": "fused",
      "transforms": [
        "_DirectParamTransform"
      ],
      "n_warps_saved": 0,
      "backend": null,
      "barrier": null,
      "split_reason": null,
      "refused": null
    }
  ],
  "last_matrix_shape": [
    2,
    3,
    3
  ],
  "torch_version": "<runtime>"
}
```

</details>

The returned matrix is the actual pixel-centre matrix from only the last supported matrix-producing segment, not a whole-pipeline matrix. Save it when it is useful for audit, but do not use it to reconstruct pipelines that cross barriers or backend boundaries.

For research artifacts, also record:

- `vision-synth`, Python, NumPy, and backend versions;
- device type and model, torch build, and accelerator runtime;
- transform configuration and construction route;
- input batch size, shape, dtype, and value range;
- every seed and worker count, plus `rank`, `world_size`, and immutable dataset `epoch` when streaming distributed synthetic data;
- randomness, reorder, execution, interpolation, padding, mask interpolation and `mask_fill`, color-clipping, antialias, and compile settings;
- `fusion_plan_descriptors` and warnings;
- whether timing includes compilation/warmup and device synchronization.

## What reproducibility does not imply

A repeatable fused run is not necessarily equal to a native backend run. The fused engine can use different coordinate, interpolation, clipping, and execution rules. Reproducibility answers “can I repeat this pipeline?”; numerical parity answers “does it match another implementation?” Treat them as separate validation gates.

See [Known limitations](../known-limitations.md) for backend parity and accelerator constraints.

Exact geometry now uses the segment's activation policy and one discrete parameter draw for both pixels and matrices. Seeded output may differ from older singleton/native exact shortcuts; record the package revision alongside seeds. This does not promise native-backend RNG parity.
