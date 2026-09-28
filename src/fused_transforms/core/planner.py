"""Plan fusion segments across augmentation backends.

This module groups mixed-backend transforms, builds backend-specific segments, and keeps passthrough dispatch metadata
stable across pickle round-trips.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from fused_transforms._backend import Backend
from fused_transforms.affine.segment import (
    AlbuFusedAffineSegment,
    AlbuProjectiveSegment,
    CropResizeSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    FusedColorSegment,
    FusedGaussianBlurSegment,
    FusedLUTSegment,
    ProjectiveSegment,
    _OpaqueBorderModeTransform,
    build_segments,
    reorder_aggressive,
    reorder_pointwise,
)
from fused_transforms.types import (
    ClipPolicyStr,
    ComposePaddingModeStr,
    ExecutionStr,
    InterpolationStr,
    MaskFillValue,
    MaskInterpolationStr,
    RandomnessPolicy,
    ReorderPolicy,
    TransformAdapter,
)


@dataclass(frozen=True, slots=True)
class _PassthroughSegment:
    """Serializable passthrough segment that carries its adapter explicitly."""

    transform: object
    adapter: TransformAdapter
    split_reason: str | None = None


def _adapter_for_backend(backend: Backend) -> TransformAdapter:
    """Return the adapter instance for a known backend.

    Args:
        backend: A ``Backend`` enum value.

    Returns:
        The corresponding ``TransformAdapter`` instance.

    Raises:
        NotImplementedError: If the backend is not supported.

    """
    if backend == Backend.KORNIA:
        from fused_transforms.adapters.kornia import KorniaAdapter

        return KorniaAdapter()
    if backend == Backend.ALBUMENTATIONS:
        from fused_transforms.adapters.albumentations import AlbumentationsAdapter

        return AlbumentationsAdapter()
    if backend == Backend.TORCHVISION:
        from fused_transforms.adapters.torchvision import TorchVisionAdapter

        return TorchVisionAdapter()
    msg = f"Backend '{backend.value}' not yet supported"
    raise NotImplementedError(msg)


def _build_mixed_segments(
    transforms: list[object],
    per_backends: list[Backend | None],
    reorder: ReorderPolicy,
    interpolation: InterpolationStr | None,
    padding_mode: ComposePaddingModeStr | None,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    *,
    route_coords_via_grid: bool = False,
    route_crop_aux: bool = False,
    execution: ExecutionStr = "cv2",
    compile_warp: bool = False,
    antialias: bool = False,
    clip_policy: ClipPolicyStr = "final",
    mask_interpolation: MaskInterpolationStr = "nearest",
    mask_fill: MaskFillValue = 0,
    fill: tuple[float, ...] | None = None,
    keypoint_flip_index: tuple[int, ...] | None = None,
) -> tuple[TransformAdapter, list[object], dict[int, TransformAdapter]]:
    """Build segments for a mixed-backend pipeline.

    Groups consecutive transforms that share the same backend, then calls
    ``build_segments`` on each group with the appropriate adapter. Backend
    boundaries act as segment breaks — the same as ``SPATIAL_KERNEL`` barriers.

    Transforms with ``backend=None`` (unrecognised) are emitted as
    passthrough objects directly.

    Args:
        transforms: Full list of transform objects.
        per_backends: Per-transform backend from ``detect_backends_per_transform``.
        reorder: Reorder policy (applied within each same-backend group).
        interpolation: Interpolation mode forwarded to segments.
        padding_mode: Padding mode forwarded to segments.
        randomness: Batch randomness policy forwarded to segments.
        route_coords_via_grid: Forwarded to ``build_segments`` — routes all-exact runs
            through the grid path when box/keypoint aux targets are present.
        route_crop_aux: Forwarded to ``build_segments`` — routes ``CROP_RESIZE_FIXED``
            ops through a ``CropResizeSegment`` (instead of an image-only passthrough) on
            the Albumentations numpy path when any auxiliary target is present.
        execution: Forwarded to ``build_segments`` — Albumentations warp strategy
            (``"cv2"`` default, ``"torch"`` opt-in batched grid_sample).
        compile_warp: Forwarded to ``build_segments`` — opt-in ``torch.compile`` of
            the torch warp core (default off, no-op on CPU / older torch).
        antialias: Forwarded to ``build_segments`` — opt-in downscale antialiasing on
            crop-resize segments (default off, no-op above the scale threshold).
        clip_policy: Forwarded to ``build_segments`` — color-segment clamp policy
            (``"final"`` default, ``"per_op_parity"`` opt-in).
        mask_interpolation: Forwarded to ``build_segments`` for auxiliary masks.
        mask_fill: Scalar border value forwarded to routed auxiliary masks.
        fill: Forwarded to ``build_segments`` — constant image border consumed by the
            warp calls of every backend group.
        keypoint_flip_index: Forwarded to ``build_segments`` — keypoint pair permutation
            applied per group wherever that group's composed matrix reverses orientation,
            the same determinant-sign rule as the single-backend path.

    Returns:
        A 3-tuple ``(primary_adapter, segments, transform_adapters)`` where:

        - ``primary_adapter`` (``TransformAdapter``): the adapter for the first
          recognised backend, used as the fallback adapter on
          ``FusedCompose._adapter``.
        - ``segments`` (``list[object]``): flat, ordered list of fused and
          passthrough segments ready for ``_wrap_passthrough_segments``.
        - ``transform_adapters`` (``dict[int, TransformAdapter]``): maps each
          transform's positional index in the original ``transforms`` list
          to the adapter used for that transform, enabling per-transform
          dispatch in the mixed-backend forward pass.

    """
    adapter_cache: dict[Backend, TransformAdapter] = {}

    def _get_adapter(backend: Backend) -> TransformAdapter:
        if backend not in adapter_cache:
            adapter_cache[backend] = _adapter_for_backend(backend)
        return adapter_cache[backend]

    transform_adapters: dict[int, TransformAdapter] = {}
    for idx_tfm, (_tfm, backend_name) in enumerate(zip(transforms, per_backends, strict=True)):
        if backend_name is not None:
            transform_adapters[idx_tfm] = _get_adapter(backend_name)

    groups: list[tuple[Backend | None, list[object]]] = []
    for tfm, backend_name in zip(transforms, per_backends, strict=True):
        if groups and groups[-1][0] == backend_name:
            groups[-1][1].append(tfm)
        else:
            groups.append((backend_name, [tfm]))

    all_segments: list[object] = []
    for backend_name, group_transforms in groups:
        if backend_name is None:
            all_segments.extend(group_transforms)
            continue

        adapter = _get_adapter(backend_name)
        if reorder == ReorderPolicy.POINTWISE:
            group_transforms = reorder_pointwise(group_transforms, adapter)
        elif reorder == ReorderPolicy.AGGRESSIVE:
            group_transforms = reorder_aggressive(group_transforms, adapter)
        group_segments = build_segments(
            transforms=group_transforms,
            adapter=adapter,
            interpolation=interpolation,
            padding_mode=None if padding_mode == "per_transform" else padding_mode,
            randomness=randomness,
            use_numpy=(backend_name == Backend.ALBUMENTATIONS),
            route_coords_via_grid=route_coords_via_grid,
            route_crop_aux=route_crop_aux,
            execution=execution,
            compile_warp=compile_warp,
            per_transform_padding=padding_mode == "per_transform",
            antialias=antialias,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            fill=fill,
            keypoint_flip_index=keypoint_flip_index,
        )
        all_segments.extend(group_segments)

    primary_backend = next((backend_name for backend_name in per_backends if backend_name is not None), None)
    if primary_backend is None:
        raise ValueError(
            "_build_mixed_segments called with no recognised backends. "
            "This is unreachable from FusedCompose.__init__; if you see "
            "this error, a caller is invoking _build_mixed_segments directly "
            "with an all-None per_backends list."
        )
    primary_adapter = _get_adapter(primary_backend)

    return primary_adapter, all_segments, transform_adapters


def _wrap_passthrough_segments(
    segments: list[object],
    default_adapter: TransformAdapter | None,
    transform_adapters: dict[int, TransformAdapter] | None,
    original_transforms: list[object] | None = None,
) -> list[object]:
    """Wrap raw passthrough transforms with their adapter for stable dispatch.

    When ``transform_adapters`` is keyed by positional index (mixed-backend path), ``original_transforms`` is used to
    resolve the index of each raw passthrough transform object.

    """
    wrapped_segments: list[object] = []
    fused_segment_types = (
        FusedAffineSegment,
        AlbuFusedAffineSegment,
        ExactAffineSegment,
        ProjectiveSegment,
        AlbuProjectiveSegment,
        FusedColorSegment,
        FusedLUTSegment,
        FusedGaussianBlurSegment,
        CropResizeSegment,
    )

    _id_to_index: dict[int, int] = {}
    if original_transforms is not None:
        for idx, tfm in enumerate(original_transforms):
            if id(tfm) in _id_to_index:
                warnings.warn(
                    f"The same transform object appears at two positions in the pipeline "
                    f"({type(tfm).__name__!r} at indices {_id_to_index[id(tfm)]} and {idx}). "
                    "Reusing the same object is unsupported; only the last occurrence will "
                    "be used for adapter dispatch.",
                    UserWarning,
                    stacklevel=3,
                )
            _id_to_index[id(tfm)] = idx

    for seg in segments:
        if isinstance(seg, (*fused_segment_types, _PassthroughSegment)):
            wrapped_segments.append(seg)
            continue

        split_reason = None
        transform = seg
        if isinstance(seg, _OpaqueBorderModeTransform):
            transform = seg.transform
            split_reason = seg.split_reason

        seg_adapter = None
        if transform_adapters is not None:
            seg_idx = _id_to_index.get(id(transform))
            if seg_idx is not None:
                seg_adapter = transform_adapters.get(seg_idx)

        if seg_adapter is None and transform_adapters is not None:
            msg = (
                "Passthrough transform encountered in mixed-backend mode but no "
                "corresponding adapter was found in transform_adapters. This is "
                "likely caused by a backend=None or custom transform without a "
                "registered adapter and would otherwise be dispatched through an "
                "incompatible default_adapter. Please ensure all passthrough "
                "transforms have an explicit adapter entry."
            )
            raise RuntimeError(msg)

        if seg_adapter is None:
            seg_adapter = default_adapter
        if seg_adapter is None:
            msg = "Passthrough transform encountered but no adapter found; this is a bug in build_segments"
            raise RuntimeError(msg)
        wrapped_segments.append(
            _PassthroughSegment(transform=transform, adapter=seg_adapter, split_reason=split_reason)
        )

    return wrapped_segments
