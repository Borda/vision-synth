---
title: Test-time augmentation with an invertible fused segment
description: Augment at inference, map predictions back to the original frame with the paired forward matrix, and measure where the inverse is exact.
---

# Test-time augmentation

Test-time augmentation predicts on several transformed copies of an image and merges the results. The merge only works if every prediction is first mapped back to the original frame, and that is what `inverse` does for a single fused segment.

Pass the matrix returned by the exact same forward call. `transform_matrix` is mutable compatibility state, so reading it afterwards is unsafe under concurrency, and a matrix from a different call produces silently wrong geometry with no error.

<!--phmdoctest-share-names-->

```python
import torch

from fused_transforms import Compose, ReorderPolicy

torch.manual_seed(13)

shift = Compose.from_params(
    translate_x=(6.0, 6.0),
    translate_y=(-4.0, -4.0),
    data_keys=["input", "bbox_xyxy"],
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(1, 3, 128, 128)
boxes = torch.tensor([[[20.0, 20.0, 60.0, 60.0]]])

(augmented, augmented_boxes), matrix = shift(images, boxes, return_matrix=True)
restored, restored_boxes = shift.inverse(augmented, augmented_boxes, matrix=matrix)

print(tuple(restored.shape), bool(torch.allclose(restored_boxes, boxes, atol=1e-3)))
```

<details>
<summary>Round-tripped image shape and exact box recovery under translation</summary>

```
(1, 3, 128, 128) True
```

</details>

Under translation, scaling, and flips the box round trip is exact, because an axis-aligned transform maps a rectangle to a rectangle. The image recovers to sampling precision rather than bit-exactly: the inverse applies a second `grid_sample`, and two interpolations are not the identity.

## Lift strided predictions before inversion

`inverse` interprets its prediction tensor on the augmentation canvas. A dense prediction head that returns a stride-two `16x16` grid for a `32x32` augmented image must be lifted to the `32x32` canvas before calling `inverse`; pass the restored result through the model's output-grid conversion afterward. Calling `inverse` directly on the low-resolution logits silently applies the pixel matrix in the wrong coordinate frame.

This runnable probe uses a nonzero eight-pixel translation, keeps the source and augmentation canvases explicit, and checks the peak location after the round trip:

```python
import torch
from torch.nn import functional as F

from fused_transforms import Compose

source_canvas = (32, 32)
augmentation_canvas = (32, 32)
stride = 2

image = torch.zeros(1, 1, *source_canvas)
image[0, 0, 16:18, 8:10] = 1.0
translate = Compose.from_params(translate_x=(8.0, 8.0))

augmented, matrix = translate(image, return_matrix=True)
assert tuple(augmented.shape[-2:]) == augmentation_canvas
logits = F.max_pool2d(augmented, kernel_size=stride)
reference_logits = F.max_pool2d(image, kernel_size=stride)

wrong_frame = translate.inverse(logits, matrix=matrix)
lifted_logits = F.interpolate(logits, size=augmentation_canvas, mode="nearest")
restored_logits = F.max_pool2d(
    translate.inverse(lifted_logits, matrix=matrix),
    kernel_size=stride,
)

expected_peak = int(reference_logits.flatten().argmax())
wrong_peak = int(wrong_frame.flatten().argmax())
restored_peak = int(restored_logits.flatten().argmax())
print(
    (
        source_canvas,
        augmentation_canvas,
        tuple(logits.shape[-2:]),
        expected_peak,
        wrong_peak,
        restored_peak,
    ),
)
assert wrong_peak != expected_peak
assert restored_peak == expected_peak
torch.testing.assert_close(restored_logits, reference_logits, rtol=1e-4, atol=1e-6)
```

<details>
<summary>Stride-two prediction peak before and after lifting</summary>

```
((32, 32), (32, 32), (16, 16), 132, 128, 132)
```

</details>

The expected peak is flat index `132` on the `16x16` output grid. The direct low-resolution inverse lands at `128`; lifting to the augmentation canvas before inversion restores `132` exactly in this probe. A real head may need bilinear upsampling or an output-stride offset chosen by its model contract, but it still must express predictions in the augmentation frame before this inverse.

## Rotation does not round trip for boxes

```python
torch.manual_seed(19)

rotate = Compose.from_params(
    rotation=(25.0, 25.0),
    data_keys=["input", "bbox_xyxy"],
    reorder=ReorderPolicy.NONE,
)

(rotated, rotated_boxes), rotate_matrix = rotate(images, boxes, return_matrix=True)
_, recovered_boxes = rotate.inverse(rotated, rotated_boxes, matrix=rotate_matrix)


def area(box: torch.Tensor) -> torch.Tensor:
    """Return the axis-aligned area of an xyxy box tensor."""
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


print(bool((area(recovered_boxes) > area(boxes)).all().item()))
print(round(float(area(recovered_boxes).item() / area(boxes).item()), 1))
```

<details>
<summary>Box area grows across a rotate-then-invert round trip</summary>

```
True
3.1
```

</details>

The forward rotation replaces the box with the axis-aligned hull of its rotated corners, and the inverse rotation takes the hull of *that* larger rectangle. Two hull operations at 25 degrees inflate the area by roughly three times. Merging boxes across rotated test-time views therefore needs oriented boxes or mask-based merging, not this round trip.

Keypoints do not have this problem: they are mapped pointwise, so the inverse recovers them to floating-point precision. Masks do not inflate either, but they recover only where the forward view kept the content on the canvas. A rotation, translation, or zoom-in pushes some pixels outside the output grid; the forward warp replaces what it cannot sample with `mask_fill`, and the inverse cannot bring those pixels back. Before merging mask predictions across views, compute each view's valid region — push an all-ones mask through the same forward and inverse matrices — and average only over pixels that every contributing view actually saw.

## One segment only

`inverse` supports a single fused affine or projective segment — a chain already fused into that one segment is fine, but a pipeline that splits into geometry, color, and geometry again is not invertible this way. Check `fusion_plan` before designing around it, and see [Introspection](../guides/introspection.md) for the mechanics.

## Recover coordinates after exact transforms and letterbox

Exact flips/quarter-turns and a direct deterministic `letterbox` also return their actual forward matrix. Use target helpers with its inverse to recover prediction coordinates. These matrices map **pixel centres**; `transform_bbox_xyxy` converts them internally for **pixel-edge** boxes. The image `inverse()` API retains its narrower scope: discarded content and interpolation cannot be undone by coordinate recovery.

```python
import torch

from fused_transforms import Compose, transform_bbox_xyxy

source_size = (5, 7)
detector_size = (10, 12)
preprocess = Compose.from_params(letterbox=detector_size, allow_upscale=False)
detector_input, forward_matrix = preprocess(
    torch.zeros(1, 3, *source_size),
    return_matrix=True,
)
# A detector prediction in its input canvas; padding moved the source box by (2, 2).
prediction = torch.tensor([[[3.0, 3.0, 8.0, 6.0]]])
source_prediction = transform_bbox_xyxy(prediction, torch.linalg.inv(forward_matrix))
print(tuple(detector_input.shape[-2:]), source_prediction.tolist())
```

<details>
<summary>Letterbox prediction recovered onto the source canvas</summary>

```
(10, 12) [[[1.0, 1.0, 6.0, 4.0]]]
```

</details>

For pose predictions, coordinate recovery with `transform_keypoints` is separate from restoring semantic left/right slots: apply the configured involutive pair table when the recorded transform reverses orientation. A matrix from a multi-segment pipeline describes only its last supported geometric segment; it is not a complete preprocessing trace. Random crop-only pipelines still return no matrix under this contract.
