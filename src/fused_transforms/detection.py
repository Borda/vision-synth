"""Pack ragged detector targets through a dense :class:`FusedCompose` box route.

The pipeline core only accepts dense ``(B, N, 4)`` boxes. This module owns the small boundary that pads COCO-style per-
image targets, applies one shared keep mask to their supported metadata, and restores the detector's ragged target list.
It does not define a detector, sampler, or general target container.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

import torch
from torch import Tensor

from fused_transforms.core.pipeline import FusedCompose
from fused_transforms.targets import clip_bbox_xyxy, instance_keep_mask

_ALLOWED_TARGET_FIELDS = frozenset({"boxes", "labels", "area", "iscrowd", "image_id"})
_ISCROWD_DTYPES = frozenset({torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64})


def _threshold(value: float, name: str, upper: float | None = None) -> float:
    """Validate one explicit detector filtering threshold."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if value < 0.0 or (upper is not None and value > upper):
        limit = f"[0, {upper}]" if upper is not None else "[0, inf)"
        raise ValueError(f"{name} must be in {limit}, got {value}")
    return value


def _target_tensor(target: Mapping[str, Tensor], name: str, index: int) -> Tensor:
    """Return one required target tensor with a boundary-specific error."""
    value = target.get(name)
    if value is None:
        raise ValueError(f"targets[{index}] is missing required {name!r}")
    if not isinstance(value, Tensor):
        raise TypeError(f"targets[{index}][{name!r}] must be a tensor")
    return value


def _validate_target(target: Mapping[str, Tensor], index: int, images: Tensor) -> Tensor:
    """Validate one detector target before its fields are packed together."""
    unsupported = sorted(set(target) - _ALLOWED_TARGET_FIELDS)
    if unsupported:
        raise ValueError(f"targets[{index}] has unsupported field(s): {unsupported}")
    boxes = _target_tensor(target, "boxes", index)
    labels = _target_tensor(target, "labels", index)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"targets[{index}]['boxes'] must have shape (N, 4), got {tuple(boxes.shape)}")
    if not boxes.is_floating_point() or boxes.dtype != images.dtype or boxes.device != images.device:
        raise ValueError(f"targets[{index}]['boxes'] must match the image floating dtype and device")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(f"targets[{index}]['labels'] must have shape ({boxes.shape[0]},), got {tuple(labels.shape)}")
    if labels.dtype != torch.int64 or labels.device != images.device:
        raise ValueError(f"targets[{index}]['labels'] must be int64 on the image device")
    for name in ("area", "iscrowd"):
        if name not in target:
            continue
        value = target[name]
        if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != boxes.shape[0]:
            raise ValueError(f"targets[{index}][{name!r}] must have shape ({boxes.shape[0]},)")
        if value.device != images.device:
            raise ValueError(f"targets[{index}][{name!r}] must be on the image device")
    area = target.get("area")
    if area is not None and (not area.is_floating_point() or area.dtype != boxes.dtype):
        raise ValueError(f"targets[{index}]['area'] must match the box floating dtype")
    iscrowd = target.get("iscrowd")
    if iscrowd is not None and iscrowd.dtype not in _ISCROWD_DTYPES:
        raise ValueError(f"targets[{index}]['iscrowd'] must use a supported integer or bool dtype")
    image_id = target.get("image_id")
    if "image_id" in target and (not isinstance(image_id, Tensor) or image_id.numel() != 1):
        raise ValueError(f"targets[{index}]['image_id'] must be a scalar or one-element tensor")
    if image_id is not None and image_id.device != images.device:
        raise ValueError(f"targets[{index}]['image_id'] must be on the image device")
    return boxes


def augment_detection_batch(
    pipeline: FusedCompose,
    images: Tensor,
    targets: Sequence[Mapping[str, Tensor]],
    *,
    min_size: float = 0.0,
    min_visibility: float = 0.0,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Augment a detector batch while keeping supported ragged target fields aligned.

    The pipeline must declare exactly ``["input", "bbox_xyxy"]``. Each target
    requires floating ``boxes`` of shape ``(N, 4)`` and int64 ``labels`` of shape
    ``(N,)`` on the image device; ``area``, ``iscrowd``, and scalar ``image_id``
    are optional. The returned list contains new dictionaries and tensors: boxes
    are clipped to the output pixel-edge canvas, positive-area instances meeting
    the explicit thresholds survive, ``labels``/``iscrowd`` follow the same mask,
    and supplied ``area`` is recomputed from the clipped boxes.

    Args:
        pipeline: A :class:`FusedCompose` declaring ``input`` and ``bbox_xyxy``.
        images: Floating image batch with shape ``(B, C, H, W)``.
        targets: One supported detector target mapping per image.
        min_size: Minimum clipped width and height in pixels.
        min_visibility: Minimum clipped/original box-area ratio in ``[0, 1]``.

    Returns:
        The augmented image batch and one ragged, detector-ready target mapping
        per input image.

    Raises:
        TypeError: If the pipeline, images, targets, or thresholds have an
            incompatible type.
        ValueError: If target schemas, devices, dtypes, fields, or the pipeline
            declaration are incompatible with this boundary.

    """
    if not isinstance(pipeline, FusedCompose):
        raise TypeError("pipeline must be a FusedCompose")
    if pipeline.data_keys != ["input", "bbox_xyxy"]:
        raise ValueError("pipeline.data_keys must be exactly ['input', 'bbox_xyxy']")
    if not isinstance(images, Tensor) or images.ndim != 4 or not images.is_floating_point():
        raise ValueError("images must be a floating BCHW tensor")
    if images.shape[0] == 0:
        raise ValueError("images must contain at least one image")
    if not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of target mappings")
    if len(targets) != images.shape[0]:
        raise ValueError(f"targets must contain one mapping per image, got {len(targets)} for batch {images.shape[0]}")
    min_size = _threshold(min_size, "min_size")
    min_visibility = _threshold(min_visibility, "min_visibility", upper=1.0)

    box_rows: list[Tensor] = []
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise TypeError(f"targets[{index}] must be a mapping")
        box_rows.append(_validate_target(target, index, images))
    # The validity mask carries the original ragged boundary through the dense pipeline.
    max_instances = max(boxes.shape[0] for boxes in box_rows)
    packed_boxes = images.new_zeros((images.shape[0], max_instances, 4))
    valid = torch.zeros((images.shape[0], max_instances), dtype=torch.bool, device=images.device)
    for index, boxes in enumerate(box_rows):
        count = boxes.shape[0]
        packed_boxes[index, :count] = boxes
        valid[index, :count] = True

    output = pipeline(images, packed_boxes)
    if not isinstance(output, tuple) or len(output) != 2 or not all(isinstance(value, Tensor) for value in output):
        raise RuntimeError("pipeline must return image and bbox_xyxy tensors")
    augmented_images, warped_boxes = output
    if augmented_images.ndim != 4 or warped_boxes.shape != packed_boxes.shape:
        raise RuntimeError("pipeline returned incompatible image or bbox_xyxy shapes")
    clipped_boxes = clip_bbox_xyxy(warped_boxes, height=augmented_images.shape[-2], width=augmented_images.shape[-1])
    positive_area = (clipped_boxes[..., 2] > clipped_boxes[..., 0]) & (clipped_boxes[..., 3] > clipped_boxes[..., 1])
    # A detector rejects zero-area boxes even when both caller thresholds intentionally default to zero.
    keep = valid & positive_area & instance_keep_mask(warped_boxes, clipped_boxes, min_size, min_visibility)

    unpacked: list[dict[str, Tensor]] = []
    for index, (target, boxes) in enumerate(zip(targets, box_rows, strict=True)):
        mask = keep[index, : boxes.shape[0]]
        output_target = {"boxes": clipped_boxes[index, : boxes.shape[0]][mask], "labels": target["labels"][mask]}
        if "area" in target:
            kept_boxes = output_target["boxes"]
            output_target["area"] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
        if "iscrowd" in target:
            output_target["iscrowd"] = target["iscrowd"][mask]
        if "image_id" in target:
            output_target["image_id"] = target["image_id"].clone()
        unpacked.append(output_target)
    return augmented_images, unpacked
