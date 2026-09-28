---
title: Known limitations and safety boundaries
description: Synthetic dataset scope limits plus the verified compatibility, target-safety, numerical-parity, randomness, GPU, and performance limits of vision-synth.
---

# Known limitations and safety boundaries

`vision-synth` ships two capabilities with separate limits. `synth_datasets` draws labelled synthetic scenes; its boundaries are about what synthetic data can tell you and which controls exist. `fused_transforms` is a tensor-first matrix-fusion engine; it has a real, tested advantage — compatible geometric transforms in one segment can share a single interpolation pass — but it is not a behaviorally identical replacement for every Kornia, TorchVision, or Albumentations pipeline.

Dataset limits come first below; everything from [Compatibility at a glance](#compatibility-at-a-glance) onward concerns the augmentation engine.

!!! danger "Auxiliary targets require a supported contract"

    When `data_keys` contains a mask, boxes, or keypoints, an unknown or unclassified spatial transform is refused before any segment executes. Treat any warning of the form `Unknown ... transform ... treating as SPATIAL_KERNEL barrier` as a review point for image-only calls.

    Common TorchVision transforms such as `RandomCrop`, `CenterCrop`, and `Resize` are not registered target-aware operations. A target-aware call refuses them before they can change only the image; the image-only path may still invoke a native passthrough.

    `data_keys` itself is validated too: an entry outside `input`, `mask`, `bbox_xyxy`, `bbox_xywh`, `keypoints`, and `rboxes` raises `ValueError` with a did-you-mean hint (`"boxes"` suggests `"bbox_xyxy"`), because an unrecognized target would skip the geometry and misalign silently.

    Use only explicitly supported geometric transforms in a multi-target pipeline. Otherwise, apply the operation through a native target-aware pipeline or transform every target yourself. See [Auxiliary targets](guides/auxiliary-targets.md).

## Synthetic dataset limits

These apply to `synth_datasets` and are independent of the augmentation engine.

| Limit                                                                          | Consequence                                                                                                             |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Objects are drawn primitives, animal silhouettes, symbols, and letters         | Photorealistic scene generation is out of scope; a synthetic result does not establish accuracy on photographs          |
| There is no `difficulty=` argument and no curriculum scheduler                 | Difficulty is a combination of ordinary `SyntheticConfig` fields; advancement belongs to your trainer                   |
| Geometric primitives carry no keypoint schema                                  | `task="keypoints"` needs shapes from one keypoint-bearing family — animals, symbols, or letters — and one family only   |
| Split membership depends on `num_images` and the split ratios                  | Changing either in a seeded export can move samples between splits; freeze an exported evaluation set for comparisons   |
| A seed reproduces generation only within the same environment                  | Record the full configuration, package and dependency versions, and any source background files alongside the seed      |
| `generate_dataset` accepts `config=` or content keywords, not both             | Passing `config=` together with `task=` or `img_size=` is rejected rather than merged                                   |
| Occluders cover pixels but do not clip geometry                                | Boxes and polygons keep full-object extent; a covered keypoint becomes visibility `1`. These are not visible-only masks |
| Degradations are baked once at generation                                      | Effects do not resample per training step; for per-epoch variation use an augmentation pipeline                         |
| Placement constraints can yield fewer objects than requested                   | `max_objects` and `overlap_iou` are limits, not guarantees                                                              |
| COCO export accumulates per-split annotation metadata                          | Direct generation and YOLO export keep sample storage bounded; size large COCO runs accordingly                         |
| `SyntheticIterableDataset` requires the `torch` extra and has no `set_epoch()` | Build a new dataset and loader with `epoch=...` for a fresh stream; see [annotation formats](datasets/outputs.md)       |
| Difficulty-band statistics are training-free                                   | They rank knob combinations by image statistics; they do not establish convergence or an accuracy ordering              |

Both writers convert point fields back to pixel-edge space at the file boundary, so an exported COCO or YOLO file carries one convention throughout. In memory the conventions differ: polygons and oriented-box corners are pixel-centre, axis-aligned boxes are pixel-edge, and the two are not interchangeable.

## Compatibility at a glance

| Question                                          | Honest answer                                                                                                                                                                                      |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is `Compose` a drop-in native compose class?      | No. It accepts transform objects from supported backends through its own tensor-first contract.                                                                                                    |
| What is the normal image format?                  | A floating PyTorch tensor shaped `(B, C, H, W)`. Fused geometry does not accept PIL images, and an unbatched `(C, H, W)` tensor raises `ValueError` suggesting `x[None]`.                          |
| Is Albumentations NumPy input supported?          | Yes. `image=HWC_array` works on its own, and with `data_keys` declared the same call carries masks, boxes, keypoints and rotated boxes. Albumentations' label processors are still not replicated. |
| Are all upstream transforms fused?                | No. Built-in adapters use finite registries. Unknown transforms become passthrough barriers or are refused.                                                                                        |
| Does fused output equal native output?            | Not universally. Sampling, coordinate conventions, interpolation, padding, clipping, and operation order can differ.                                                                               |
| Does `transform_matrix` cover the whole pipeline? | No. It is the actual matrix from the most recent call's last supported affine/projective, exact D4, or direct deterministic letterbox segment.                                                     |
| Is every GPU or MPS configuration faster?         | No. Speed depends on device, batch, image size, operation mix, warmup, and passthrough transfers. Benchmark your exact pipeline.                                                                   |

## Input and backend limits

The main fused path expects BCHW tensors. Native compose classes accept broader families of inputs and metadata that this package does not reproduce:

- TorchVision pipelines may accept PIL images, unbatched tensors, TVTensors, and nested sample structures. Fused geometric segments require BCHW tensors.
- Albumentations' HWC NumPy keyword path accepts auxiliary targets only when the pipeline declares `data_keys`, and only for chains whose every segment publishes a matrix; a crop, an exact affine or an opaque passthrough sends the call back through tensors. Either way the targets are this package's contract, not Albumentations' label-processor semantics.
- Kornia's `AugmentationSequential` has container behavior and metadata contracts beyond the transform-object compatibility provided here.
- Multi-backend pipelines are supported for BCHW tensors, but a backend change creates a segment boundary. Transforms from different backends do not share one fused matrix.

TorchVision `RandomRotation(expand=True)` is explicitly unsupported. The fused TorchVision geometry uses an `align_corners=True` pixel convention with center `((W - 1) / 2, (H - 1) / 2)`, which differs from native TorchVision geometry. Fill, interpolation, and center behavior must be validated rather than assumed from the original transform object.

## Auxiliary-target limits

### Safe only within the declared contract

| Pipeline element                                          | Mask/box/keypoint behavior                                | Safety decision                                                 |
| --------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| Registered affine/projective transform                    | Routes supported targets with the segment grid or matrix  | Supported, subject to the numerical limits below                |
| Registered exact flip or discrete operation               | Routes supported targets through exact or matrix logic    | Supported for the registered operation                          |
| Registered `RandomResizedCrop`                            | Routes targets to the target spatial size                 | Supported; output size changes                                  |
| Known pointwise or kernel passthrough                     | Leaves coordinate targets unchanged                       | Safe only when the operation truly preserves coordinates        |
| Named coordinate-changing passthrough on the refusal list | Raises rather than silently misaligning targets           | Safe refusal, not target support                                |
| Unknown transform or custom callable                      | Raises before any segment executes with auxiliary targets | Fail-closed; image-only passthrough remains a separate contract |

Masks and coordinates have additional contracts:

- Detection `bbox_xyxy` and `bbox_xywh` use pixel-edge extents: a full `(H, W)` canvas is `[0, W] x [0, H]`. Image sampling, keypoints, and rotated boxes retain pixel-centre coordinates through `[0, W - 1] x [0, H - 1]`; box helpers perform the centre-to-edge matrix conversion at their boundary. Do not apply the box convention to image, keypoint, or rotated-box matrices.
- `synth_datasets` follows the same split: a generated `polygon`, `keypoints` table and `obb_corners` are in pixel-centre coordinates, while `bbox_xyxy` is in pixel-edge ones, so each field can be passed to its matching transform unchanged. Exported COCO and YOLO files convert the point fields back to edge space at the file boundary and carry one convention throughout. See [Tasks and keypoints](datasets/tasks.md#coordinate-conventions).
- Nearest-neighbor mask sampling uses scalar `mask_fill=0` by default, independently of image `fill` and `padding_mode`. Configure a finite dtype-compatible scalar for an ignore value such as `255` (`uint8`) or `-1` (signed integer). Bilinear masks remain floating-only; nearest masks retain their no-autograd contract.
- `padding_mode="reflection"` reflects about the outer pixel *centres* (OpenCV `BORDER_REFLECT_101`), following this package's `align_corners=True` sampling. An implementation sampling with `align_corners=False` reflects about the outer pixel *edges* (`BORDER_REFLECT`) and its mirrored band differs by a pixel of phase. `"zeros"` and `"border"` are convention-free; see "Half-pixel convention" in `guides/configuration.md`.
- A canvas thinner than two pixels on either axis is refused: the `align_corners=True` normalization divides by `L - 1`. A `(H, 1)` or `(1, W)` image raises naming the axis rather than warping through an infinite scale.
- The nearest mask path is deliberately executed without autograd. This removes gradients with respect to both the mask values and sampling grid. Bilinear sampling is available for floating soft masks, mixes values at boundaries, and keeps an autograd path.
- Boxes are dense `(B, N, 4)` tensors and keypoints are dense `(B, N, 2)` tensors. Validation rejects batch mismatches, mask-canvas mismatches, wrong coordinate widths, and extra box columns before execution; empty `N=0` tables remain valid. The package does not clip them to image bounds, filter invisible or zero-area boxes, calculate visibility, carry labels, or manage variable `N`.
- Rotated boxes are returned as axis-aligned bounding boxes, which can be larger than the rotated object.
- Coordinate targets remain PyTorch tensors when image and mask outputs are converted with `output_backend="numpy"`.

The dense auxiliary-target route intentionally does not understand detector metadata or ragged instance axes. `augment_detection_batch` is the explicit TorchVision-style adapter: it accepts one mapping per image with `boxes` and int64 `labels`, packs and unpacks the dense box route, clips to pixel-edge extents, recomputes supplied `area`, and applies one keep mask to supported `iscrowd` and `image_id` fields. Unsupported per-instance fields raise instead of being silently discarded.

## Passthrough is not transparent

An unregistered transform can execute through its native backend on image-only calls, but "passthrough" does not mean identical data or behavior. With auxiliary targets, unknown or unsafe passthroughs are refused before any segment runs:

- It splits fusion segments and can add transfers or native calls.
- A coordinate-changing or unclassified passthrough cannot route auxiliary targets and is refused as described above.
- Albumentations passthrough on the tensor path receives HWC float data derived from the BCHW tensor. Operations designed around uint8 magnitude, including some noise, fog, and compression transforms, may behave differently or become ineffective.
- `substitute_passthrough=True` is explicitly behavior-changing. The current substitution can replace Albumentations Gaussian blur with a Kornia blur that has different kernels, borders, and randomness.

Two common Albumentations transforms produce a passthrough for reasons worth knowing before you plan a chain around them:

- `HueSaturationValue` always does. It is registered `POINTWISE`: reorderable, but non-linear in RGB, so it composes into neither a color matrix nor a per-channel lookup table. There is no fused segment for it and, in the current design, cannot be one. A chain of geometry plus `HueSaturationValue` therefore fuses the geometry and then leaves the tensor for the color step.
- `RandomResizedCrop` and `RandomSizedCrop` do so **only on image-only calls under `execution="cv2"` or `"auto"`**. Under `execution="torch"` the crop routes through the segment instead, because that chain already resamples the image with `grid_sample` and the native crop would buy parity the chain has given up while still paying a device round trip. A historical MPS sweep reported 0.83x at batch 8, 1.17x at 32 and 1.39x at 64; treat those values as dated evidence, not a current performance guarantee. The small-batch regression was attributed to the segment's then-current matrix assembly, so rerun the current path before carrying that explanation forward. `"auto"` stays on the passthrough because it resolves per call and the device is unknown when segments are built. When the call carries a mask, box, or keypoint target, the crop routes through a `CropResizeSegment` and no passthrough appears. Without an auxiliary target the crop runs natively instead, which keeps it bit-exact against Albumentations at the cost of a segment break. This is a parity choice, not a missing registration — but it means the same chain plans differently depending on whether you declared `data_keys`, and the image-only plan is the slower one on an accelerator.

Kornia and TorchVision fuse a crop into the preceding affine as a single segment regardless of auxiliary targets, so a plan comparison across backends is not a like-for-like comparison of the same chain.

## Reordering changes semantics

`ReorderPolicy.POINTWISE` moves color and pointwise operations across geometric operations to create larger fusion groups. Pointwise math does not generally commute with finite-image warps because padding, interpolation, and clipping make order observable.

Use `ReorderPolicy.NONE` when declared order, native comparison, or experiment reproduction matters. `from_params` and `from_config` currently default to `POINTWISE`, so pass `reorder=ReorderPolicy.NONE` explicitly for parity-oriented work. `AGGRESSIVE` currently behaves like `POINTWISE`; it is not a stronger parity guarantee.

## Numerical parity and color limits

Fusing intentionally replaces multiple resampling steps with one. The result can be higher quality than sequential interpolation while still differing from the original backend output.

- TorchVision geometry has a known center/`align_corners` difference. Native pixel parity is not promised.
- Albumentations `execution="torch"` uses `grid_sample`; its border and subpixel weights differ from OpenCV. Use `execution="cv2"` when the OpenCV execution convention matters.
- Affine and projective transforms form separate segment types. Crossing between them can require another resampling pass.
- Default color `clip_policy="final"` clamps after the fused color run, whereas native chains may clamp after every operation.
- `clip_policy="per_op_parity"` improves parity for gamut-escaping chains but is still approximate for some contrast sequences. It can differ at roughly the `1e-2` scale in the known mean-relative contrast case.
- Fused color matrices are defined for three-channel RGB. Other channel counts fall back to sequential native execution.

## Randomness and reproducibility limits

`randomness="backend"` preserves the intended batch sampling style, not a guarantee of an identical random stream or identical native output.

Albumentations-backed fused geometry uses two random-number domains: package activation gates use global NumPy randomness, while transform parameters use the Albumentations transform's internal RNG. Seeding only `torch`, or only `numpy.random`, is insufficient. See [Reproducibility](guides/reproducibility.md).

Fast paths and different batch sizes can consume random draws differently. For strict experiments, record package/backend versions, batch size, execution strategy, reorder policy, seeds, and the machine-readable fusion plan.

## GPU, compilation, and antialiasing limits

`compile=True` is off by default and is a no-op on CPU. Non-CPU compilation is environment-dependent; the test suite does not establish a universal CUDA or MPS speedup, dynamic-shape guarantee, or compiler compatibility matrix. Measure warmup and steady state separately on the deployment host.

`antialias=True` is limited to aggressive crop-resize downscaling. Each sample's axis scales determine whether it is prefiltered and its Gaussian support; safe samples remain unchanged. Enabling the option requires Kornia and raises `ImportError` during pipeline construction when that optional dependency is unavailable. The flag is off by default, so ordinary pipelines do not acquire this dependency or filtering cost. Scale estimation reads bounded device values into Python, which can synchronize accelerator work.

Passthrough operations are particularly important on accelerators. A native CPU-only passthrough can erase the advantage of a fused GPU segment through device transfers. `execution="torch"` keeps the registered Albumentations fused warp on the input device, but it cannot move an opaque CPU-only native transform onto that device. Inspect the plan and benchmark the complete pipeline, not only the fused warp.

That cost was measured in the historical September 5, 2026 NVIDIA L4 sweep rather than asserted. Chains carrying exactly one passthrough ran 2.0x to 3.0x slower on CUDA than the same chains on the CPU engine, while chains with no passthrough tied or won — and the penalty grew with batch size, because `call_nonfused` copies the whole batch to the host and back once per passthrough per call. One passthrough was enough to erase a fused warp's advantage in that run; revalidate on the current revision and deployment hardware.

Fusion's value on an accelerator is backend-dependent, and the dated CPU/MPS profile below is no longer a current attribution. An older MPS profile attributed 40% of a fused call to `.to()`, a further 26% to `torch.tensor`, and 86% cumulatively to matrix composition. That profile predates the current batched matrix-transfer path and must not be used to explain current timings. The historical CUDA sweep still showed backend-specific wins and losses, but current-head CUDA/MPS measurements and runner availability are unverified. Do not assume a CPU speedup transfers to a GPU; measure the complete deployment pipeline. See the dated [Benchmarks](research/benchmarks.md#historical-cuda-batch-sweep-september-5-2026).

## Introspection limits

`fusion_plan` and `fusion_plan_descriptors` describe segmentation. They are the right tools for detecting barriers and backend boundaries.

`transform_matrix` and `return_matrix=True` expose the actual forward pixel-centre matrix for the last supported matrix-producing segment. Fused affine/projective segments, exact D4/flip/quarter-turn segments, and direct deterministic `letterbox` publish this `(B, 3, 3)` coordinate provenance. They do not compose across backend boundaries, passthrough barriers, separate affine/projective segments, or multiple fused segments; the matrix is never a whole-pipeline trace. The property is mutable per-call state and should not be read concurrently from a shared pipeline instance. Before a call, or after a call with no supported geometry, it is `None`.

Native NumPy single-image exact calls retain the native layout, including rectangular outputs when a 90-degree or transpose operation swaps height and width, and publish the same actual-call matrix. Uniform BCHW exact D4/90-degree/transpose batches also support rectangular inputs and swap the common output shape. A batch whose samples would produce heterogeneous spatial shapes is refused; use a uniform exact draw (for example `same_on_batch=True`) or separate those samples.

`n_warps_saved` is a planning heuristic, not a literal count of native interpolations or an observed speedup. In particular, exact flips can contribute to the metric even though native flips are already non-interpolating.

## Test-time inverse limits

`pipe.inverse(prediction, matrix=matrix)` maps a prediction back to the original geometric frame, but only for a pipeline that reduces to one fused affine or projective image segment. It raises for crop-resize or standalone deterministic letterbox (cropped, padded, or resized pixels are not reconstructed), color/LUT/blur or passthrough segments, exact-only images, and multi-segment pipelines. Exact D4/flip/quarter-turn and letterbox matrices remain useful for coordinate recovery through the target helpers; publishing provenance does not widen the image-inverse contract.

The inverse is geometric-only. It cannot recover values discarded by interpolation or padding, and it does not undo color, LUT, or blur operations. Recovered boxes are axis-aligned, so a forward-then-inverse box is exact only for axis-aligned transforms (flip, scale, translation) and inflates under rotation, shear, or a projective warp. Always pass the matrix returned by the same `forward(..., return_matrix=True)` call rather than the mutable `transform_matrix` property.

## How to decide whether the package fits

Use `vision-synth` when all of the following are true:

1. Your main path uses BCHW PyTorch tensors.
2. Your geometric transforms appear in the documented capability surface.
3. You accept fused-engine numerics instead of requiring native pixel identity.
4. Any auxiliary targets stay inside the explicitly supported routing contract.
5. A representative benchmark and task-quality check pass on your hardware.

Use the native backend, or keep a native reference pipeline, when input compatibility, native target processors, exact random streams, native pixel parity, or unsupported spatial transforms are requirements.
