---
title: Fused augmentation for detection and keypoints
description: Transform bounding boxes and keypoints as dense tensors, then apply the clipping, filtering, and keep-mask postprocessing the package deliberately leaves to you.
---

# Detection and keypoints

The package applies matrix math to coordinate tensors. It is not a detection framework's target processor, and the difference matters more here than anywhere else in these docs.

What it does: transforms `(B, N, 4)` boxes and `(B, N, 2)` keypoints through the same fused geometry as the image.

The core tensor route does not clip coordinates to the image, drop boxes that left the frame, shrink boxes to their visible part, remove degenerate areas, propagate class labels alongside a filtered box tensor, or update keypoint visibility flags. When a detector already owns one ragged target mapping per image, use `augment_detection_batch` as the explicit boundary described below.

Skip that postprocessing and you train on boxes that sit outside the image, on one-pixel slivers, and on labels that no longer line up with their coordinates.

<!--phmdoctest-share-names-->

```python
import torch

from fused_transforms import Compose, ReorderPolicy

torch.manual_seed(11)

detect = Compose.from_params(
    rotation=(30.0, 30.0),
    data_keys=["input", "bbox_xyxy"],
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(1, 3, 128, 128)
boxes = torch.tensor(
    [[[10.0, 10.0, 40.0, 40.0], [90.0, 90.0, 126.0, 126.0], [0.0, 0.0, 3.0, 3.0]]],
)

image_out, boxes_out = detect(images, boxes)

print(tuple(boxes_out.shape))
```

<details>
<summary>Box tensor shape after a fixed 30-degree rotation</summary>

```
(1, 3, 4)
```

</details>

Three boxes in, three boxes out — including the one now rotated off the canvas and the three-pixel sliver. The tensor shape is preserved because filtering is your decision, not the package's.

## The postprocessing you own

```python
labels = torch.tensor([[1, 2, 3]])

clipped = boxes_out.clone()
clipped[..., 0::2] = clipped[..., 0::2].clamp(0, 128)
clipped[..., 1::2] = clipped[..., 1::2].clamp(0, 128)

widths = clipped[..., 2] - clipped[..., 0]
heights = clipped[..., 3] - clipped[..., 1]
keep = (widths > 2) & (heights > 2)

kept_boxes = clipped[keep]
kept_labels = labels[keep]

print(int(keep.sum()))
print(tuple(kept_boxes.shape), tuple(kept_labels.shape))
```

<details>
<summary>Surviving box count and the aligned label tensor</summary>

```
2
(2, 4) (2,)
```

</details>

The important line is `labels[keep]`. The same boolean mask has to be applied to every per-instance array you carry — classes, track identifiers, crowd flags, scores — or your labels silently shift by one instance for the rest of training.

Once boxes are filtered per image, `N` differs across the batch, so the result is a ragged list rather than a padded tensor. Re-pad it, or collate it the way your detector expects, before it reaches the model.

## Augment a ragged detector batch

`augment_detection_batch` is the narrow adapter for a TorchVision-style detector target list. It requires a `FusedCompose` whose `data_keys` are exactly `["input", "bbox_xyxy"]`. It validates one mapping per image, packs the per-image `(N, 4)` floating `boxes` into a dense batch, runs the declared geometry, clips boxes to the output pixel-edge canvas `[0, W] x [0, H]`, and restores a new ragged list.

Each mapping requires `boxes` and int64 `labels`. Optional fields are floating same-dtype `area`, integer or boolean `iscrowd`, and scalar or one-element `image_id`. Unsupported fields, missing values, shape/device/dtype mismatches, and non-sequence targets raise at the boundary. The helper applies one validity mask to boxes and supported per-instance fields; supplied `area` is recomputed after clipping, and input mappings and tensors are left untouched. `min_size` and `min_visibility` are explicit thresholds; zero-area boxes are always removed because detector models reject them.

```python
import torch

from fused_transforms import Compose, augment_detection_batch

images = torch.zeros(1, 1, 16, 16)
targets = [
    {
        "boxes": torch.tensor([[0.0, 2.0, 4.0, 6.0], [10.0, 2.0, 14.0, 6.0]]),
        "labels": torch.tensor([1, 2]),
    }
]
pipe = Compose.from_params(
    translate_x=(-4.0, -4.0),
    data_keys=["input", "bbox_xyxy"],
)

_, augmented_targets = augment_detection_batch(pipe, images, targets)
print(augmented_targets[0]["boxes"].tolist())
print(augmented_targets[0]["labels"].tolist())
```

<details>
<summary>Ragged target fields stay aligned after clipping and filtering</summary>

```
[[6.0, 2.0, 10.0, 6.0]]
[2]
```

</details>

The adapter pads only inside the call and returns one target mapping per input image, including empty `(0, 4)` box and `(0,)` label tensors. It returns no keep mask or instance indices, and it rejects any field outside `boxes`, `labels`, `area`, `iscrowd`, and `image_id`. To carry other per-instance metadata — track identifiers, scores — use the dense route from [The postprocessing you own](#the-postprocessing-you-own) instead: run the pipeline with `data_keys=["input", "bbox_xyxy"]`, compute the keep mask yourself (as shown there, or with `instance_keep_mask`), and index every per-instance array with that one mask.

## Rotation inflates axis-aligned boxes

A rotated rectangle is not a rectangle. The package returns the axis-aligned bounding box of the rotated corners, which is strictly larger than the original for any rotation that is not a multiple of 90 degrees. Repeated augmentation therefore grows boxes over epochs if you feed outputs back as inputs.

[Test-time augmentation](test-time-augmentation.md) measures that inflation on a round trip.

## Keypoints

Keypoints transform as `(B, N, 2)` coordinates. Visibility is not part of that tensor, so a keypoint pushed outside the image keeps whatever flag it had. Recompute visibility from the transformed coordinates and the output extent yourself — the standard COCO convention of `2` visible, `1` occluded, `0` unlabelled is not something the package can infer.
