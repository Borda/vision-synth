"""Auxiliary target transform helpers for fused augmentation pipelines.

Pure mathematical functions that transform masks, bounding boxes, and keypoints
using precomputed affine matrices or grids from the fused pipeline.

These helpers are called internally by :class:`~fused_transforms.affine.segment.FusedAffineSegment` and
:class:`~fused_transforms.affine.segment.ExactAffineSegment` when
``data_keys`` includes auxiliary targets. They are also exported as public API
for callers that want to apply the same math outside of the pipeline
(e.g. to transform a stored transform matrix after the fact).

All functions are stateless and operate on PyTorch tensors with a leading batch
dimension ``batch_size``. Nearest-neighbour mask sampling is
non-differentiable; bilinear sampling is available for float soft masks.
The other three functions are differentiable.

Examples:
    ```pycon
    >>> import torch
    >>> from fused_transforms.targets import transform_keypoints
    >>> keypoints = torch.tensor([[[10.0, 20.0]]])  # (batch_size=1, num_points=1, 2)
    >>> matrix = torch.eye(3).unsqueeze(0)           # identity (1, 3, 3)
    >>> out = transform_keypoints(keypoints, matrix)
    >>> torch.allclose(out, keypoints)
    True

    ```

"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral, Real

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fused_transforms.types import MaskInterpolationStr

#: Guards the visibility division for a box whose unclipped area underflows to zero.
_AREA_EPS = 1e-12

#: Trailing size of a rotated box: ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Corners per quadrilateral, and coordinates per point.
_QUAD_CORNERS = 4
_POINT_DIM = 2
#: A corner table is at least ``(num_boxes, 4, 2)``.
_RBOX_CORNER_NDIM = 3

#: Optional caller-supplied convention applied to fitted rotated boxes. This package
#: imposes none of its own -- see :func:`transform_rboxes`.
RBoxCanonicalizer = Callable[[Tensor], Tensor]


def _check_rboxes(rboxes: Tensor) -> None:
    """Raise when ``rboxes`` does not carry a trailing ``(cx, cy, w, h, theta)``."""
    if rboxes.shape[-1] != _RBOX_DIM:
        msg = f"rboxes must have a trailing dimension of 5 (cx, cy, w, h, theta), got {rboxes.shape[-1]}."
        raise ValueError(msg)


def _validate_mask_fill(mask: Tensor, fill: object) -> float:
    """Return a finite fill that survives the mask's sampling and dtype round trip."""
    if not isinstance(fill, Real):
        raise TypeError(f"mask fill must be a scalar real value, got {type(fill).__name__}")
    try:
        fill_value = float(fill)
    except OverflowError as error:
        raise ValueError("mask fill must be finite") from error
    if not math.isfinite(fill_value):
        raise ValueError("mask fill must be finite")
    if mask.dtype == torch.bool:
        if fill_value not in (0.0, 1.0):
            raise ValueError("boolean mask fill must be False or True")
        return fill_value
    if mask.is_floating_point():
        if abs(fill_value) > torch.finfo(mask.dtype).max:
            raise ValueError(f"mask fill {fill_value} is not finite in sampling dtype {mask.dtype}")
        return fill_value
    if isinstance(fill, Integral):
        # Preserve a Python integer before converting it to float. Above 2**53,
        # that conversion can otherwise erase a non-representable low bit.
        fill_int = int(fill)
    else:
        if not fill_value.is_integer():
            raise ValueError("integer mask fill must be integral")
        fill_int = int(fill_value)
    dtype_range = torch.iinfo(mask.dtype)
    if not dtype_range.min <= fill_int <= dtype_range.max:
        raise ValueError(f"mask fill {fill_int} is outside the range of {mask.dtype}")
    if int(torch.tensor(fill_int, dtype=torch.float32).item()) != fill_int:
        raise ValueError(f"integer mask fill {fill_int} is not exactly representable by float32")
    return fill_value


def transform_mask(
    mask: Tensor,
    grid: Tensor,
    mode: MaskInterpolationStr = "nearest",
    fill: int | float = 0,  # noqa: PYI041 -- integer values are accepted mask fills.
) -> Tensor:
    """Apply a precomputed affine grid to a segmentation mask.

    The default ``mode='nearest'`` preserves integer class labels without
    fractional mixing and carries no gradient. Use
    ``mode='bilinear'`` for differentiable float soft masks; labels may mix at
    boundaries. Out-of-bounds samples use the scalar ``fill`` value, which
    defaults to 0.

    Args:
        mask: Segmentation mask. Shape ``(batch_size, channels, height, width)``, typically ``channels=1``.
            dtype: Any floating or integer dtype. Integer masks are
            automatically cast to a floating dtype for ``grid_sample`` and cast
            back to the original dtype afterward. Value range: integer class
            indices (e.g. 0, 1, 2, …). Channel convention: channel-first
            (PyTorch).
        grid: Sampling grid from ``torch.nn.functional.affine_grid``.
            Shape ``(batch_size, height, width, 2)``. Any floating dtype (``float16``,
            ``float32``, ``float64``) is accepted; integer masks are
            cast to ``float32`` internally regardless of the grid
            dtype to avoid fp16/bf16 rounding while keeping memory
            usage and bandwidth lower than ``float64``. Note: ``float32``
            exactly represents integer class IDs up to ``2**24 - 1``
            (16777215); larger integer IDs may be rounded.
            Coordinates in normalised ``[-1, 1]`` space with ``align_corners=True``.
        mode: Mask sampling mode, either ``"nearest"`` (default) or
            ``"bilinear"``. Bilinear mode requires a floating-point mask.
        fill: Finite scalar used outside the source canvas. Floating masks require
            a value that remains finite in their sampling dtype. Integer masks
            require an integral in-range value that round-trips through float32;
            boolean masks allow only ``False``/``True`` values.

    Returns:
        Warped mask with the same shape and dtype as ``mask``.

    Examples:
        Identity grid leaves the mask unchanged:

        ```pycon
        >>> import torch
        >>> import torch.nn.functional as F
        >>> mask = torch.zeros(1, 1, 4, 4)
        >>> mask[0, 0, 1, 1] = 1
        >>> eye2 = torch.eye(2, 3).unsqueeze(0)  # identity theta (1, 2, 3)
        >>> grid = F.affine_grid(eye2, [1, 1, 4, 4], align_corners=True)
        >>> out = transform_mask(mask, grid)
        >>> out.shape
        torch.Size([1, 1, 4, 4])
        >>> bool(out[0, 0, 1, 1] == 1)
        True

        ```

    """
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"unsupported mask interpolation mode {mode!r}; expected 'nearest' or 'bilinear'")
    if mode == "bilinear" and not mask.is_floating_point():
        raise TypeError("bilinear mask interpolation requires a floating-point mask")
    fill_value = _validate_mask_fill(mask, fill)

    needs_cast_back = not mask.is_floating_point()
    sample_mask = mask
    sample_grid = grid

    if needs_cast_back:
        # Integer masks must not be sampled through fp16/bf16 in mixed precision,
        # otherwise class IDs can be rounded before being cast back. ``float32``
        # is sufficient to preserve typical class ID ranges while avoiding the
        # memory and bandwidth overhead of ``float64``.
        sample_mask = mask.to(dtype=torch.float32)
        sample_grid = grid.to(dtype=torch.float32)
    elif sample_grid.dtype != sample_mask.dtype:
        # The image warp may use an opt-in low-precision grid while a floating
        # auxiliary mask intentionally stays in its caller-provided dtype.
        sample_grid = grid.to(dtype=sample_mask.dtype)
    if mode == "nearest":
        # Keep the default path exactly as before: nearest sampling is detached.
        with torch.no_grad():
            sampled = F.grid_sample(
                sample_mask,
                sample_grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )
            if fill_value:
                coverage = F.grid_sample(
                    torch.ones_like(sample_mask),
                    sample_grid,
                    mode=mode,
                    padding_mode="zeros",
                    align_corners=True,
                )
    else:
        sampled = F.grid_sample(
            sample_mask,
            sample_grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )
    if fill_value:
        if mode == "bilinear":
            coverage = F.grid_sample(
                torch.ones_like(sample_mask),
                sample_grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )
        # Sampling a zero-padded coverage mask applies fill only outside the
        # source canvas, avoiding lossy subtraction from in-bounds labels.
        sampled = sampled + (1 - coverage) * fill_value
    if needs_cast_back:
        return sampled.to(dtype=mask.dtype)
    return sampled


def _bbox_edge_matrix(mtx_forward: Tensor) -> Tensor:
    """Convert an image-centre homography with ``T(+0.5) @ M @ T(-0.5)`` for AABBs."""
    to_edge = (
        torch.eye(3, device=mtx_forward.device, dtype=mtx_forward.dtype).expand(mtx_forward.shape[0], -1, -1).clone()
    )
    to_edge[:, 0, 2] = 0.5
    to_edge[:, 1, 2] = 0.5
    to_center = to_edge.clone()
    to_center[:, 0, 2] = -0.5
    to_center[:, 1, 2] = -0.5
    return to_edge @ mtx_forward @ to_center


def transform_bbox_xyxy(boxes: Tensor, mtx_forward: Tensor) -> Tensor:
    """Transform ``(batch_size, num_boxes, 4)`` xyxy boxes by a ``(batch_size, 3, 3)`` forward homography.

    Computes all four corners of each box, transforms them through the forward
    matrix using homogeneous multiplication, then returns the axis-aligned
    bounding box (AABB) that tightly wraps the transformed corners.

    The AABB wrapping step means output boxes are always axis-aligned and may be
    larger than the true rotated box. This is the standard trade-off for box
    transforms that must remain in xyxy format.

    Args:
        boxes: Bounding boxes in xyxy format. Shape ``(batch_size, num_boxes, 4)``,
            columns ``[x1, y1, x2, y2]`` as pixel-edge extents. A full image spans
            ``[0, width] x [0, height]``, dtype ``float32``.
        mtx_forward: Forward (not inverse) affine or projective matrix in
            image-centre pixel coordinates. Shape ``(batch_size, 3, 3)``, dtype
            ``float32``. The helper converts it internally to the AABB edge
            coordinate space; callers must not pre-convert it.

    Returns:
        Transformed AABB boxes. Shape ``(batch_size, num_boxes, 4)``, xyxy format.

    Examples:
        Identity matrix leaves boxes unchanged:

        ```pycon
        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 20.0, 50.0, 80.0]]])  # (batch_size=1, num_boxes=1, 4)
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> out = transform_bbox_xyxy(boxes, mtx_identity)
        >>> torch.allclose(out, boxes)
        True

        ```

    """
    box_x1 = boxes[..., 0]  # (batch_size, num_boxes)
    box_y1 = boxes[..., 1]
    box_x2 = boxes[..., 2]
    box_y2 = boxes[..., 3]

    # Build all 4 corners: (B, N, 3, 4) homogeneous [x, y, 1]
    ones = torch.ones_like(box_x1)
    corners_x = torch.stack([box_x1, box_x2, box_x2, box_x1], dim=-1)  # (B, N, 4)
    corners_y = torch.stack([box_y1, box_y1, box_y2, box_y2], dim=-1)
    corners_h = torch.stack(
        [corners_x, corners_y, ones.unsqueeze(-1).expand_as(corners_x)],
        dim=-2,
    )  # (B, N, 3, 4)

    # Image resampling matrices map pixel centres. AABBs use pixel edges, so conjugate
    # once here rather than changing image, keypoint, or rotated-box centre geometry.
    mtx_unsqueezed = _bbox_edge_matrix(mtx_forward).unsqueeze(1)  # (B, 1, 3, 3)
    transformed = mtx_unsqueezed @ corners_h  # (B, N, 3, 4)

    # Perspective division (for affine, homogeneous_w_raw=1 so this is a no-op)
    homogeneous_w_raw = transformed[:, :, 2, :]  # (B, N, 4) — homogeneous w
    # Guard against zero or extremely small |w| to avoid inf/NaN from division.
    # Use finfo.eps (not finfo.tiny): dividing by tiny overflows float32 to inf,
    # so a tiny-based clamp does not actually prevent non-finite outputs.
    eps = torch.finfo(homogeneous_w_raw.dtype).eps
    small_mask = homogeneous_w_raw.abs() < eps
    sign_val = torch.sign(homogeneous_w_raw)
    sign_val = torch.where(sign_val == 0, torch.ones_like(sign_val), sign_val)
    safe_homogeneous_w = torch.where(small_mask, eps * sign_val, homogeneous_w_raw)
    transformed_x = transformed[:, :, 0, :] / safe_homogeneous_w
    transformed_y = transformed[:, :, 1, :] / safe_homogeneous_w

    # amin/amax, not min(dim=).values: Tensor.min(dim=) also computes argmin and dispatches to a
    # parallelised reduction whose thread-pool cost is ~38 us here regardless of tensor size, while
    # torch.amin over the same (batch_size, num_boxes, 4) corner axis costs ~0.9 us. Four calls per
    # step made this the single largest fixed cost of a detection-shaped call.
    #
    # Forward values are identical, NaN included. Backward differs in two ways, neither of which any
    # target modality in this package relies on -- boxes are input data, and the documented
    # differentiable target is the soft mask:
    #   * At ties -- which every axis-aligned box has, since its corner list repeats each coordinate
    #     twice -- amin splits the subgradient evenly instead of routing all of it to the lowest
    #     index: [0.5, 0, 0, 0.5] rather than [1, 0, 0, 0]. The total is 1.0 either way.
    #   * With a NaN corner, min(dim=) routes the gradient to the NaN's index while amin returns NaN
    #     for every corner.
    new_x1 = torch.amin(transformed_x, dim=-1)
    new_y1 = torch.amin(transformed_y, dim=-1)
    new_x2 = torch.amax(transformed_x, dim=-1)
    new_y2 = torch.amax(transformed_y, dim=-1)

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)


def transform_bbox_xywh(boxes: Tensor, mtx_forward: Tensor) -> Tensor:
    """Transform ``(batch_size, num_boxes, 4)`` xywh boxes by a ``(batch_size, 3, 3)`` forward homography.

    Converts boxes from ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``, delegates to
    :func:`transform_bbox_xyxy` (4-corner transform + AABB), then converts back to
    ``[x, y, w, h]``.

    The output ``width`` and ``height`` reflect the AABB after rotation, so they will be
    larger than the input for non-axis-aligned transforms.

    Args:
        boxes: Bounding boxes in xywh format. Shape ``(batch_size, num_boxes, 4)``,
            columns ``[x, y, w, h]`` where ``(x, y)`` is the top-left pixel edge,
            dtype ``float32``.
        mtx_forward: Forward (not inverse) affine or projective matrix in
            image-centre pixel coordinates. Shape ``(batch_size, 3, 3)``, dtype
            ``float32``. :func:`transform_bbox_xyxy` converts it to edge space.

    Returns:
        Transformed boxes in xywh format. Shape ``(batch_size, num_boxes, 4)``.

    Examples:
        Identity matrix leaves boxes unchanged:

        ```pycon
        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 20.0, 40.0, 60.0]]])  # x, y, w, h
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> out = transform_bbox_xywh(boxes, mtx_identity)
        >>> torch.allclose(out, boxes)
        True

        ```

    """
    box_left, box_top, box_width, box_height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = torch.stack([box_left, box_top, box_left + box_width, box_top + box_height], dim=-1)
    xyxy_out = transform_bbox_xyxy(xyxy, mtx_forward)
    box_x1, box_y1, box_x2, box_y2 = (
        xyxy_out[..., 0],
        xyxy_out[..., 1],
        xyxy_out[..., 2],
        xyxy_out[..., 3],
    )
    return torch.stack([box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1], dim=-1)


def transform_keypoints(keypoints: Tensor, mtx_forward: Tensor) -> Tensor:
    """Transform ``(batch_size, num_points, 2)`` keypoints by a ``(batch_size, 3, 3)`` forward homography.

    Converts each keypoint to homogeneous coordinates ``[x, y, 1]``, multiplies by
    the forward matrix, and returns the first two components of the result per point::

        keypoints'[batch_size, num_points] = (mtx_forward[batch_size] @ [x, y, 1]^T)[:2]

    Unlike bounding boxes, keypoints are transformed exactly (no AABB widening).
    The operation is differentiable with respect to both ``keypoints`` and ``mtx_forward``.

    Args:
        keypoints: Keypoints in pixel coordinates. Shape ``(batch_size, num_points, 2)``,
            columns ``[coord_x, coord_y]``, dtype ``float32``.
        mtx_forward: Forward (not inverse) affine or projective matrix in pixel
            coordinates. Shape ``(batch_size, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed keypoints. Shape ``(batch_size, num_points, 2)``.

    Examples:
        Identity matrix leaves keypoints unchanged:

        ```pycon
        >>> import torch
        >>> keypoints = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (batch_size=1, num_points=2, 2)
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> out = transform_keypoints(keypoints, mtx_identity)
        >>> torch.allclose(out, keypoints)
        True

        ```

    """
    batch_size, num_kps, _ = keypoints.shape
    ones = torch.ones(batch_size, num_kps, 1, device=keypoints.device, dtype=keypoints.dtype)
    keypoints_h = torch.cat([keypoints, ones], dim=-1)  # (batch_size, num_points, 3)

    # mtx_forward: (B, 3, 3); keypoints_h: (B, num_points, 3) -> (B, 3, num_points) for matmul
    transformed = mtx_forward @ keypoints_h.transpose(1, 2)  # (batch_size, 3, num_points)
    # Perspective division (for affine, homogeneous_w=1 so this is a no-op).
    # Clamp homogeneous_w away from 0 to avoid Inf/NaN for degenerate homographies.
    homogeneous_w = transformed[:, 2:3, :]  # (batch_size, 1, num_points)
    eps = torch.finfo(keypoints.dtype).eps
    abs_homogeneous_w = homogeneous_w.abs()
    sign_homogeneous_w = torch.sign(homogeneous_w)
    # Ensure we have a non-zero sign so clamped values keep a consistent direction.
    sign_homogeneous_w = torch.where(sign_homogeneous_w == 0, torch.ones_like(sign_homogeneous_w), sign_homogeneous_w)
    safe_homogeneous_w = torch.where(abs_homogeneous_w < eps, sign_homogeneous_w * eps, homogeneous_w)
    return (transformed[:, :2, :] / safe_homogeneous_w).transpose(1, 2)  # (B, N, 2)


def clip_bbox_xyxy(boxes: Tensor, height: float, width: float) -> Tensor:
    """Clamp ``xyxy`` boxes to the ``[0, width] x [0, height]`` canvas extent.

    Clipping is geometry, not policy: it says where the canvas ends, not which instances
    deserve to survive. Pair it with :func:`instance_keep_mask`, which compares the clipped
    extent against the unclipped one to decide survival.

    The canvas is the pixel *extent*, so a box covering the whole image spans ``[0, width]``
    -- one unit wider than the ``width - 1`` distance between the outer pixel centres. Both
    conventions appear in the wild; this one matches the area interpretation the visibility
    ratio needs.

    Args:
        boxes: ``(..., 4)`` boxes as ``(x1, y1, x2, y2)`` in pixels.
        height: Canvas height in pixels.
        width: Canvas width in pixels.

    Returns:
        A new tensor of the same shape with every coordinate clamped to the canvas.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import clip_bbox_xyxy
        >>> boxes = torch.tensor([[[-4.0, -4.0, 6.0, 6.0]]])
        >>> clip_bbox_xyxy(boxes, height=8.0, width=8.0).tolist()
        [[[0.0, 0.0, 6.0, 6.0]]]

        ```

    """
    clipped = boxes.clone()
    clipped[..., 0::2] = boxes[..., 0::2].clamp(0.0, float(width))
    clipped[..., 1::2] = boxes[..., 1::2].clamp(0.0, float(height))
    return clipped


def instance_keep_mask(
    boxes: Tensor,
    clipped_boxes: Tensor,
    min_size: float = 0.0,
    min_visibility: float = 0.0,
) -> Tensor:
    """Return which instances survive a warp, by clipped size and kept-area fraction.

    An instance is kept when its clipped box is at least ``min_size`` on both axes **and**
    retains at least ``min_visibility`` of its unclipped area. Both thresholds come from the
    caller: they are a training recipe's numbers, and this package does not have an opinion
    about them.

    The return value is the **mask**, never a filtered box tensor. Every other modality on
    the same instance axis -- labels, keypoints, rotated boxes, polygon rings, per-instance
    flags -- has to be filtered by this same mask, and the caller is the only one holding
    all of them. A helper that returned filtered boxes would leave a pipeline whose boxes
    and keypoints describe different instances while every shape still lines up and nothing
    raises.

    Not to be confused with ``clip_policy``, which decides when a *colour* chain clamps to
    ``[0, 1]`` during fusion. This is instance survival after a geometric warp; the names
    are close and the concepts share nothing.

    Args:
        boxes: ``(..., 4)`` warped ``xyxy`` boxes **before** clipping to the canvas.
        clipped_boxes: ``(..., 4)`` the same boxes after clipping (see :func:`clip_bbox_xyxy`).
        min_size: Minimum clipped width *and* height, in pixels. ``0.0`` keeps every size.
        min_visibility: Minimum clipped-area / unclipped-area fraction, in ``[0, 1]``.
            ``0.0`` keeps everything the size rule keeps.

    Returns:
        A boolean tensor of shape ``boxes.shape[:-1]`` -- one flag per input instance, in
        input order, always the full length.

    Raises:
        ValueError: If the two box tensors have different shapes, or either lacks a
            trailing dimension of 4.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import clip_bbox_xyxy, instance_keep_mask
        >>> warped = torch.tensor([[[2.0, 2.0, 6.0, 6.0], [-9.0, -9.0, -1.0, -1.0]]])
        >>> clipped = clip_bbox_xyxy(warped, height=8.0, width=8.0)
        >>> instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.25).tolist()
        [[True, False]]

        ```

    """
    if boxes.shape != clipped_boxes.shape:
        msg = (
            f"boxes and clipped_boxes must have the same shape, got "
            f"{tuple(boxes.shape)} and {tuple(clipped_boxes.shape)}."
        )
        raise ValueError(msg)
    if boxes.shape[-1] != 4:
        msg = f"boxes must have a trailing dimension of 4 (x1, y1, x2, y2), got {boxes.shape[-1]}."
        raise ValueError(msg)

    widths = (clipped_boxes[..., 2] - clipped_boxes[..., 0]).clamp(min=0.0)
    heights = (clipped_boxes[..., 3] - clipped_boxes[..., 1]).clamp(min=0.0)
    area_before = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    area_after = widths * heights
    # A degenerate pre-clip box has no area to keep a fraction of; calling its visibility
    # zero drops it under any positive threshold and keeps it under min_visibility=0.0,
    # which is the same treatment a fully clipped-away instance gets.
    visibility = torch.where(
        area_before > 0.0, area_after / area_before.clamp(min=_AREA_EPS), torch.zeros_like(area_before)
    )
    return (widths >= min_size) & (heights >= min_size) & (visibility >= min_visibility)


def rboxes_to_corners(rboxes: Tensor) -> Tensor:
    """Expand ``(cx, cy, w, h, theta)`` rotated boxes into their four corners.

    ``theta`` rotates the box's own ``w`` axis away from ``+x``; the corners run from the
    local ``(-w/2, -h/2)`` corner around the rectangle. The angle is used exactly as given
    -- no canonicalization -- so ``theta`` and ``theta + pi`` produce the same rectangle
    with its corner slots rolled by two.

    Args:
        rboxes: ``(batch_size, num_boxes, 5)`` rotated boxes ``(cx, cy, w, h, theta)``,
            ``theta`` in radians.

    Returns:
        ``(batch_size, num_boxes, 4, 2)`` corner coordinates in pixels.

    Raises:
        ValueError: If the trailing dimension is not 5.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import rboxes_to_corners
        >>> box = torch.tensor([[[5.0, 3.0, 4.0, 2.0, 0.0]]])
        >>> rboxes_to_corners(box)[0, 0].tolist()
        [[3.0, 2.0], [7.0, 2.0], [7.0, 4.0], [3.0, 4.0]]

        ```

    """
    _check_rboxes(rboxes)
    centre = rboxes[..., :2]
    cos, sin = torch.cos(rboxes[..., 4]), torch.sin(rboxes[..., 4])
    half_w, half_h = rboxes[..., 2] / 2, rboxes[..., 3] / 2
    along_w = torch.stack([cos * half_w, sin * half_w], dim=-1)
    along_h = torch.stack([-sin * half_h, cos * half_h], dim=-1)
    return torch.stack(
        [
            centre - along_w - along_h,
            centre + along_w - along_h,
            centre + along_w + along_h,
            centre - along_w + along_h,
        ],
        dim=-2,
    )


def corners_to_rboxes(corners: Tensor) -> Tensor:
    """Fit ``(cx, cy, w, h, theta)`` rotated boxes to quadrilaterals.

    No single vertex decides an output: the centre is the vertex centroid, each extent is
    the mean of its pair of opposite side lengths, and the direction is the mean of that
    pair's two antiparallel edge vectors. On an exact rectangle all three reduce to the
    exact values; on a quad within ``eps`` of a rectangle the extents land within ``eps``.

    The result is **not** canonicalized: no ``w >= h`` swap and no angle range is imposed,
    because which representative of a rectangle is the right one is a downstream convention
    (long-edge, or otherwise) rather than a geometric fact.

    Args:
        corners: ``(batch_size, num_boxes, 4, 2)`` quadrilateral corners in pixels, each
            ring in order around its quad (either winding).

    Returns:
        ``(batch_size, num_boxes, 5)`` rotated boxes, un-canonicalized.

    Raises:
        ValueError: If the trailing dimensions are not ``(4, 2)``.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import corners_to_rboxes
        >>> quad = torch.tensor([[[[3.0, 2.0], [7.0, 2.0], [7.0, 4.0], [3.0, 4.0]]]])
        >>> [round(value, 4) for value in corners_to_rboxes(quad)[0, 0].tolist()]
        [5.0, 3.0, 4.0, 2.0, 0.0]

        ```

    """
    if corners.ndim < _RBOX_CORNER_NDIM or corners.shape[-2:] != (_QUAD_CORNERS, _POINT_DIM):
        msg = f"corners must have trailing dimensions (4, 2), got {tuple(corners.shape)}."
        raise ValueError(msg)
    edges = corners.roll(-1, dims=-2) - corners
    lengths = edges.norm(dim=-1)
    # Opposite sides of a rectangle are the (0, 2) and (1, 3) edge pairs under any cyclic
    # ordering, and are antiparallel -- hence the difference, not the sum, for direction.
    width = (lengths[..., 0] + lengths[..., 2]) / 2
    height = (lengths[..., 1] + lengths[..., 3]) / 2
    direction = (edges[..., 0, :] - edges[..., 2, :]) / 2
    theta = torch.atan2(direction[..., 1], direction[..., 0])
    centre = corners.mean(dim=-2)
    return torch.stack([centre[..., 0], centre[..., 1], width, height, theta], dim=-1)


def transform_rboxes(rboxes: Tensor, mtx_forward: Tensor, canonicalize: RBoxCanonicalizer | None = None) -> Tensor:
    """Transform ``(batch_size, num_boxes, 5)`` rotated boxes by a ``(batch_size, 3, 3)`` forward matrix.

    A general affine does **not** map a rectangle to a rectangle: only a similarity does.
    Shear sends a rectangle to a parallelogram, which no ``(cx, cy, w, h, theta)`` can
    describe. So the box is expanded to the four corners the warp actually moves, those
    corners are mapped, and a box is re-fitted to them (:func:`corners_to_rboxes`).

    Under a similarity -- rotation, uniform scale, translation, mirror, or any composition
    of them -- the fit reproduces the warped rectangle exactly. Under a shear of angle ``s``
    the warped quad is a parallelogram whose sides meet at ``pi / 2 - s``, and the fitted
    corners sit ``(h / 2) * sqrt(tan(s)**2 + (1 - 1 / cos(s))**2)`` from the warped ones,
    where ``h`` is the extent across the shear. To first order in ``s`` that is the
    often-quoted ``h * sin(s / 2)``; the exact value exceeds it from the third order on, so
    the short form is an estimate rather than a bound (measured in
    ``tests/test_unit/test_rboxes.py``). The returned box is that fit, not a lossless
    re-parameterization, and this function does not pretend otherwise.

    Args:
        rboxes: ``(batch_size, num_boxes, 5)`` rotated boxes ``(cx, cy, w, h, theta)``.
        mtx_forward: ``(batch_size, 3, 3)`` forward pixel matrix, the same one the image
            was warped by.
        canonicalize: Optional callable applied to the fitted boxes. This package imposes
            no convention of its own -- the long-edge form (``w >= h`` with a bounded
            ``theta``) is one reading of the literature among several, and belongs to the
            caller that also owns the assigner, the loss and the evaluation kernel.

    Returns:
        ``(batch_size, num_boxes, 5)`` transformed boxes; un-canonicalized unless a
        ``canonicalize`` callable was supplied.

    Raises:
        ValueError: If the trailing dimension is not 5.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import transform_rboxes
        >>> double = torch.eye(3).unsqueeze(0) * 2.0
        >>> double[0, 2, 2] = 1.0
        >>> box = torch.tensor([[[1.0, 2.0, 4.0, 2.0, 0.0]]])
        >>> [round(value, 4) for value in transform_rboxes(box, double)[0, 0].tolist()]
        [2.0, 4.0, 8.0, 4.0, 0.0]

        ```

    """
    corners = rboxes_to_corners(rboxes)
    batch_size, num_boxes = corners.shape[0], corners.shape[1]
    flat = corners.reshape(batch_size, num_boxes * _QUAD_CORNERS, _POINT_DIM)
    warped = transform_keypoints(flat, mtx_forward).reshape(batch_size, num_boxes, _QUAD_CORNERS, _POINT_DIM)
    fitted = corners_to_rboxes(warped)
    return fitted if canonicalize is None else canonicalize(fitted)


def mirror_rboxes(rboxes: Tensor, width: int, canonicalize: RBoxCanonicalizer | None = None) -> Tensor:
    """Mirror rotated boxes about this package's horizontal-flip axis.

    The mirror line sits at ``(width - 1) / 2``, the axis a horizontal image flip uses under
    the ``align_corners=True`` convention this package samples with -- **not** ``width / 2``,
    which an extent-convention implementation would use. Boxes and image therefore stay in
    step; a caller matching a ``width / 2`` implementation is off by half a pixel.

    The centre reflects and the long-edge direction reflects with it: ``(cos t, sin t)``
    maps to ``(-cos t, sin t)``, the direction of ``pi - t``. That is ``-t`` plus a half
    turn, and a rectangle is invariant under a half turn, so either value describes the same
    rectangle -- ``pi - t`` is returned because it is what fitting the mirrored corners
    yields, which keeps this helper and the pipeline's matrix path parameter-identical
    rather than merely geometrically equal. Extents are unchanged, a mirror being an
    isometry.

    Args:
        rboxes: ``(batch_size, num_boxes, 5)`` rotated boxes ``(cx, cy, w, h, theta)``.
        width: Canvas width in pixels.
        canonicalize: Optional callable applied to the mirrored boxes; see
            :func:`transform_rboxes`.

    Returns:
        ``(batch_size, num_boxes, 5)`` mirrored boxes.

    Raises:
        ValueError: If the trailing dimension is not 5.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import mirror_rboxes
        >>> box = torch.tensor([[[3.0, 5.0, 8.0, 4.0, 0.25]]])
        >>> [round(value, 4) for value in mirror_rboxes(box, width=10)[0, 0].tolist()]
        [6.0, 5.0, 8.0, 4.0, 2.8916]

        ```

    """
    _check_rboxes(rboxes)
    mirrored = rboxes.clone()
    mirrored[..., 0] = (width - 1) - rboxes[..., 0]
    mirrored[..., 4] = math.pi - rboxes[..., 4]
    return mirrored if canonicalize is None else canonicalize(mirrored)


def shift_rboxes(
    rboxes: Tensor,
    offset_x: float,
    offset_y: float,
    canonicalize: RBoxCanonicalizer | None = None,
) -> Tensor:
    """Translate rotated boxes by a pixel offset.

    A translation moves the centre and touches nothing else, so unlike
    :func:`transform_rboxes` this needs no corner round trip and introduces no fitting
    residual at all -- worth having as its own path for the placement-only cases.

    Args:
        rboxes: ``(batch_size, num_boxes, 5)`` rotated boxes ``(cx, cy, w, h, theta)``.
        offset_x: Horizontal offset in pixels, added to ``cx``.
        offset_y: Vertical offset in pixels, added to ``cy``.
        canonicalize: Optional callable applied to the shifted boxes; see
            :func:`transform_rboxes`.

    Returns:
        ``(batch_size, num_boxes, 5)`` translated boxes.

    Raises:
        ValueError: If the trailing dimension is not 5.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import shift_rboxes
        >>> box = torch.tensor([[[2.0, 3.0, 6.0, 4.0, 0.5]]])
        >>> [round(value, 4) for value in shift_rboxes(box, 10.0, -1.0)[0, 0].tolist()]
        [12.0, 2.0, 6.0, 4.0, 0.5]

        ```

    """
    _check_rboxes(rboxes)
    shifted = rboxes.clone()
    shifted[..., 0] = rboxes[..., 0] + offset_x
    shifted[..., 1] = rboxes[..., 1] + offset_y
    return shifted if canonicalize is None else canonicalize(shifted)


def rbox_envelopes(rboxes: Tensor) -> Tensor:
    """Return each rotated box's tight axis-aligned ``xyxy`` envelope.

    This is the bridge to the axis-aligned machinery: pair it with
    :func:`clip_bbox_xyxy` and :func:`instance_keep_mask` to decide which rotated instances
    survive a warp. A rotated box clipped to the canvas is in general no longer a rotated
    box -- the clipped shape is a polygon -- so this package deliberately supplies the
    envelope rather than a clip that would have to invent a rectangle.

    Args:
        rboxes: ``(batch_size, num_boxes, 5)`` rotated boxes ``(cx, cy, w, h, theta)``.

    Returns:
        ``(batch_size, num_boxes, 4)`` axis-aligned ``(x1, y1, x2, y2)`` envelopes.

    Raises:
        ValueError: If the trailing dimension is not 5.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import rbox_envelopes
        >>> box = torch.tensor([[[5.0, 3.0, 4.0, 2.0, 0.0]]])
        >>> rbox_envelopes(box)[0, 0].tolist()
        [3.0, 2.0, 7.0, 4.0]

        ```

    """
    corners = rboxes_to_corners(rboxes)
    minimum = corners.amin(dim=-2)
    maximum = corners.amax(dim=-2)
    return torch.cat([minimum, maximum], dim=-1)


def orientation_reversed(mtx_forward: Tensor) -> Tensor:
    """Return which samples' transforms reverse orientation, from the matrix determinant.

    A mirror is not visible as a discrete operation once it has been composed into a larger
    matrix -- "is there a flip in the transform list" is unanswerable after fusion, and
    wrong whenever two mirrors compose back to a rotation. The determinant of the linear
    part answers it exactly: negative means the map turns the plane over.

    Args:
        mtx_forward: ``(batch_size, 3, 3)`` forward pixel matrices.

    Returns:
        ``(batch_size,)`` boolean tensor, ``True`` where the sample's transform mirrors.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import orientation_reversed
        >>> mirror = torch.tensor([[[-1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        >>> orientation_reversed(mirror).tolist()
        [True]
        >>> orientation_reversed(mirror @ mirror).tolist()
        [False]

        ```

    """
    linear = mtx_forward[..., :2, :2]
    determinant = linear[..., 0, 0] * linear[..., 1, 1] - linear[..., 0, 1] * linear[..., 1, 0]
    return determinant < 0


def permute_keypoint_pairs(keypoints: Tensor, flip_index: Tensor, reversed_mask: Tensor) -> Tensor:
    """Reorder the keypoint axis for samples whose transform reversed orientation.

    A mirrored image has its left and right anatomy swapped, so a "left elbow" landmark now
    sits where the right elbow is. The coordinates are already correct after the warp -- it
    is the *identity* of each slot that has to follow, which is what this permutation does.

    Which slots pair with which is dataset schema, not geometry, so ``flip_index`` comes
    from the caller. This package only decides *when* to apply it, and decides that from the
    composed matrix rather than from the presence of a flip transform.

    Args:
        keypoints: ``(batch_size, num_points, 2)`` keypoints, already warped.
        flip_index: ``(num_points,)`` integer permutation: slot ``i`` takes its value from
            slot ``flip_index[i]`` after a mirror.
        reversed_mask: ``(batch_size,)`` boolean, ``True`` for samples to permute.

    Returns:
        ``(batch_size, num_points, 2)`` keypoints with the permutation applied to the
        selected samples and the others untouched.

    Raises:
        ValueError: If ``flip_index`` does not have one entry per keypoint slot.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.targets import permute_keypoint_pairs
        >>> points = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        >>> pairs = torch.tensor([0, 2, 1])
        >>> permute_keypoint_pairs(points, pairs, torch.tensor([True]))[0].tolist()
        [[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]]

        ```

    """
    if flip_index.shape[0] != keypoints.shape[-2]:
        msg = (
            f"keypoint_flip_index has {flip_index.shape[0]} entries but the keypoints carry "
            f"{keypoints.shape[-2]} slots."
        )
        raise ValueError(msg)
    permuted = keypoints.index_select(-2, flip_index.to(device=keypoints.device))
    return torch.where(reversed_mask[:, None, None], permuted, keypoints)
