"""Compose -- fused augmentation pipeline replacing the backend's Compose/Sequential.

Wraps a list of augmentation transforms, fuses consecutive geometric ops into a
single ``grid_sample`` pass, and provides the same forward-call interface as the
backend.

v0.3 additions: ``data_keys`` parameter routes auxiliary targets (masks,
bounding boxes, keypoints) alongside the image through every segment.
``from_params()`` classmethod constructs a fused pipeline directly from
numeric parameter ranges, without importing any backend.

v0.5 additions: mixed-backend support allows transforms from multiple
frameworks (e.g. TorchVision geometric + Kornia color) in a single
pipeline. ``_PassthroughSegment`` carries non-fused transforms across
pickle round-trips via an index-keyed adapter map that survives
deserialisation (object ids change; positional indices do not).

Examples:
    ```pycon
    >>> import torch
    >>> from fused_transforms.compose import Compose
    >>> pipe = Compose([])
    >>> image = torch.zeros(1, 3, 8, 8)
    >>> pipe(image).shape
    torch.Size([1, 3, 8, 8])

    ```
"""

from __future__ import annotations

import contextlib
import difflib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from fused_transforms._backend import Backend, detect_backends_per_transform, is_builtin_adapter
from fused_transforms._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, backend_available
from fused_transforms._random import GeneratorPicklingMixin, reject_backend_randomness
from fused_transforms.affine.matrix import (
    hflip_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    perspective_grid,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)
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
    _clear_current_call_matrix,
    _current_call_matrix,
    _FusedGeoCropSegment,
    _matrix_public_dtype,
    _validate_execution,
    build_segments,
    reorder_aggressive,
    reorder_pointwise,
)
from fused_transforms.core.config_validation import (
    _COORD_DATA_KEYS,
    _PIPELINE_TORCH_DTYPES,
    _apply_passthrough_substitution,
    _coerce_randomness_policy,
    _has_aux_target,
    _has_coord_aux,
    _validate_clip_policy,
    _validate_fill,
    _validate_keypoint_flip_index,
    _validate_mask_fill,
    _validate_mask_interpolation,
    _validate_pipeline_dtype,
)
from fused_transforms.core.factories import (
    FactoriesMixin,
    _DirectFlipTransform,
    _DirectParamAdapter,
    _DirectParamTransform,
)
from fused_transforms.core.introspection import IntrospectionMixin
from fused_transforms.core.planner import (
    _adapter_for_backend,
    _build_mixed_segments,
    _PassthroughSegment,
    _wrap_passthrough_segments,
)
from fused_transforms.types import (
    BackendConverter,
    ClipPolicyStr,
    ComposePaddingModeStr,
    ExecutionStr,
    FillValue,
    InterpolationStr,
    MaskFillValue,
    MaskInterpolationStr,
    PipelineDtypeStr,
    RandomnessPolicy,
    ReorderPolicy,
    SegmentDescriptor,
    TransformAdapter,
    TransformCategory,
    TransformSpec,
    is_coordinate_changing_passthrough,
)

__doctest_skip__: list[str] = []
if not _KORNIA_AVAILABLE:
    __doctest_skip__ += ["FusedCompose.from_config"]
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["FusedCompose.__call__"]

if TYPE_CHECKING:
    from numpy.typing import NDArray

_KNOWN_DATA_KEYS = {"input", "mask", "bbox_xyxy", "bbox_xywh", "keypoints", "rboxes"}
# Albumentations-style keyword aliases accepted by the dict-output call form
# (``pipe(image=..., mask=..., bboxes=...)``). ``image`` maps to the ``"input"``
# data key; ``bboxes`` maps to whichever box key the pipeline declared. Exact
# data-key names are also accepted as their own aliases.
_KWARG_ALIASES: dict[str, str | tuple[str, ...]] = {
    "image": "input",
    "bboxes": ("bbox_xyxy", "bbox_xywh"),
}

#: Segment tags (see the tag table in ``__init__``) a NumPy-native multi-target call can serve.
#: ``0`` is the fused affine segment, which leaves behind the matrix its auxiliary targets are
#: routed through; ``2``, ``6`` and ``7`` are colour, LUT and blur segments, which cannot move a
#: coordinate at all. Every other tag either changes geometry without publishing a matrix or is
#: opaque, and sends the call back to the tensor path.
_AUX_SAFE_NATIVE_TAGS = frozenset({0, 2, 6, 7})


class _NumpyRoundTrip(NamedTuple):
    """Input layout of a NumPy multi-target call, used to return outputs in the caller's own layout.

    ``image_ndim`` is the rank of the input image array (2 for ``(height, width)``, 3 for ``(height, width, channels)``,
    4 for a batch). ``aux_ndims`` holds the same for each auxiliary target that arrived as an array; auxiliary targets
    passed as tensors are absent from it and are returned as tensors.

    ``image_dtype`` records the image normalization to undo. ``mask_dtype`` preserves label dtype, without intensity
    normalization, and is ``None`` when no mask arrived as an array.

    """

    image_ndim: int
    aux_ndims: dict[str, int]
    image_dtype: np.dtype[Any]
    mask_dtype: np.dtype[Any] | None = None


def _denormalize_numpy_image(array: NDArray[Any], dtype: np.dtype[Any]) -> NDArray[Any]:
    """Undo the converter's intensity normalisation, returning the caller's own dtype.

    ``NumpyToTorchConverter`` scales ``uint8`` by 1/255 and ``uint16`` by 1/65535 on the way in; both are inverted here
    with a round-then-clip so a value that drifted marginally outside ``[0, 1]`` during the warp does not wrap around
    the integer range. Every other dtype was cast without rescaling, so restoring it is a plain cast.

    """
    if array.dtype == dtype:
        return array
    if dtype == np.uint8:
        return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if dtype == np.uint16:
        return (np.clip(array, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
    return array.astype(dtype)


def _tensor_to_numpy_image(tensor: Tensor, input_ndim: int) -> NDArray[Any]:
    """Convert a ``(batch_size, channels, height, width)`` tensor back to the caller's array rank.

    A grayscale ``(height, width)`` input round-trips without a trailing singleton channel axis; every other rank
    follows the standard converter, which squeezes a single-image batch to ``(height, width, channels)``.

    """
    from fused_transforms.dispatch.converters import TorchToNumpyConverter

    array = TorchToNumpyConverter().convert(tensor)
    if input_ndim == 2 and array.ndim == 3 and array.shape[-1] == 1:
        return array[:, :, 0]
    return array


def _tensor_to_numpy_coords(tensor: Tensor, input_ndim: int) -> NDArray[Any]:
    """Convert a ``(batch_size, num_instances, ...)`` coordinate tensor back to the caller's rank.

    Coordinate targets have no channel layout to restore, so this only drops the batch axis that
    :meth:`FusedCompose._normalize_aux_input` added for an unbatched input.

    """
    array = cast("NDArray[Any]", tensor.detach().cpu().numpy())
    if input_ndim == 2 and array.ndim == 3 and array.shape[0] == 1:
        return cast("NDArray[Any]", array[0])
    return array


# Plan-time segment dispatch tags. Computed once per pipeline at construction and
# stored on the instance so ``forward`` loops over integer tags instead of running
# an isinstance chain per segment per call. Kept as plain ints (not bound methods)
# so the dispatch plan survives the default nn.Module pickle round-trip.
_TAG_MATRIX = 0  # records supported geometric segment matrices
_TAG_PLAIN = 1  # no matrix: color/LUT/unmarked crop
_TAG_PASSTHROUGH = 2  # _PassthroughSegment: adapter.call_nonfused
_TAG_LEGACY = 3  # pre-wrapper pickles / unknown segment — resolve adapter by index

# These imports were historically available directly from this module. Retain the
# compatibility surface while their implementations now reside in focused modules.
_HISTORICAL_IMPORTS = (
    contextlib,
    math,
    _COORD_DATA_KEYS,
    _DirectFlipTransform,
    _DirectParamTransform,
    hflip_matrix,
    matmul3x3,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
    _FusedGeoCropSegment,
    TransformCategory,
    TransformSpec,
)
# Coordinate aux targets that ExactAffineSegment cannot route through a non-flip exact
# op: their per-sample rotation is not recoverable without re-sampling. When present, an
# all-exact geometric run is routed through the (still-lossless for D4) grid path.


def _reject_generator_with_backend_transforms(
    generator: torch.Generator | None,
    transforms: list[object],
) -> None:
    """Reject a caller-owned generator on a pipeline whose parameters a backend samples.

    Every backend adapter delegates parameter sampling to its own library, and none of
    those samplers accept a ``torch.Generator``. Accepting the generator and drawing from
    the global stream anyway would report reproducibility the pipeline does not have, so
    the combination is refused at construction instead of at the first activated draw.

    Args:
        generator: The caller-owned generator, or ``None``.
        transforms: The transform list the pipeline was constructed with.

    Raises:
        ValueError: If ``generator`` is not ``None`` and ``transforms`` is non-empty.

    """
    if generator is None or not transforms:
        return
    names = sorted({type(tfm).__name__ for tfm in transforms})
    reject_backend_randomness(generator, f"backend transform(s) {', '.join(names)}")


class FusedCompose(FactoriesMixin, IntrospectionMixin, GeneratorPicklingMixin, nn.Module):
    """Fused augmentation pipeline that replaces the backend's native Compose.

    Segments the transform list into fused geometric segments and passthrough transforms, then executes them
    sequentially. Consecutive geometric ops are grouped and executed as either:

    - A :class:`~fused_transforms.affine.segment.FusedAffineSegment` - when the run
      contains at least one ``GEOMETRIC_INTERP`` op. Matrices are composed and
      a single ``grid_sample`` call is used, eliminating redundant interpolation
      passes.
    - An :class:`~fused_transforms.affine.segment.ExactAffineSegment` - when the run
      contains *only* ``GEOMETRIC_EXACT`` ops (HFlip, VFlip). Transforms are
      applied via ``tensor.flip`` with zero interpolation error.

    ``SPATIAL_KERNEL`` and nonlinear ``POINTWISE`` transforms are passed through
    to the backend adapter unchanged. Supported ``POINTWISE_LINEAR`` color
    transforms, including standard Normalize, are folded into a color segment.

    ``ReorderPolicy.POINTWISE`` is fully implemented: before segmentation,
    ``POINTWISE`` and ``POINTWISE_LINEAR`` ops are bubbled past geometric ops
    within each ``SPATIAL_KERNEL``-bounded stretch, maximising the geometric
    run length available for fusion.

    Args:
        transforms: List of augmentation transform objects.
        reorder: Reorder policy applied before segmentation.
            ``NONE`` (default) preserves the original order. ``POINTWISE`` reorders pointwise ops after geometric
            chains. ``AGGRESSIVE`` currently aliases ``POINTWISE`` and is kept for API compatibility with future
            stronger reorder semantics.
        interpolation: Interpolation mode override for fused segments
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode override for fused segments
            (``"zeros"``, ``"border"``, ``"reflection"``), or the opt-in
            ``"per_transform"`` policy. The latter honors compatible transform
            modes and keeps opaque modes as native boundaries. Defaults to
            ``"zeros"`` when ``None``.
        data_keys: List of key names describing positional arguments to
            :meth:`forward`. The first key should be ``"input"`` (the image).
            Auxiliary keys (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"rboxes"``,
            ``"keypoints"``) are routed through segments and transformed alongside the image. Unknown keys raise
            ``ValueError`` naming the supported keys and the closest match. ``None`` preserves backward-compatible
            single-tensor input/output.
            Albumentations fused segments route auxiliary targets through the composed pixel matrix, matching the
            Kornia/TorchVision coordinate convention, so multi-target ``data_keys`` are supported for every backend.
        output_backend: Target output format. ``"numpy"`` (or its alias
            ``"numpy_hwc"``) converts the primary image output to a NumPy
            ``ndarray`` with channel-last layout: batched inputs of shape
            ``(batch_size, channels, height, width)`` become ``(batch_size, height, width, channels)``,
            while unbatched inputs of shape ``(channels, height, width)`` become ``(height, width, channels)`` (i.e.
            the batch dimension is implicit/squeezed). **Variable-batch trap**: a batch of size one,
            ``(1, channels, height, width)``, is *also* squeezed and returns ``(height, width, channels)`` --
            the leading batch axis is dropped rather than kept as ``(1, height, width, channels)``. Callers
            that loop over a variable batch size must special-case ``batch_size == 1`` (or re-insert the axis
            with ``np.expand_dims(out, 0)``) to avoid a rank mismatch. ``"torch"`` or ``None`` keeps the native
            ``torch.Tensor`` output. For multi-target ``data_keys`` the conversion is applied per target: image
            and mask outputs are converted, while coordinate targets (boxes, keypoints) stay tensors.
        randomness: Batch randomness policy. ``"backend"`` preserves native
            backend semantics, including TorchVision v2 batch-shared sampling.
            ``"per_sample"`` asks adapters with canonical samplers to draw
            independent parameters per batch item.
        execution: Warp strategy for the Albumentations fused segments.
            ``"cv2"`` (default) warps each sample with OpenCV -- bit-exact with the
            native cv2 backend and fastest on CPU at small batch sizes.
            ``"torch"`` composes the same per-sample matrices (identical sampling
            and RNG) but applies one batched ``grid_sample`` for the whole batch.
            Throughput depends on the batch, device, and workload; its
            border/bilinear numerics differ slightly from cv2. Only affects
            Albumentations pipelines; the Kornia/TorchVision backends already run
            the torch engine.
        compile: Opt-in ``torch.compile`` of the warp core (matrix normalize ->
            ``affine_grid`` -> ``grid_sample``) of the torch-backed fused segments.
            Off by default. When ``True``, the compiled region is used only on
            non-CPU tensors and only when the installed torch is new enough
            (``>= 2.2``); otherwise the eager path runs and the flag is a no-op, so
            the output is unchanged. Probability masking stays outside the compiled
            region, so the graph has no data-dependent breaks. The first non-CPU call
            pays a one-time compilation cost; ``dynamic=True`` avoids per-shape
            recompiles.
        antialias: Opt-in antialiasing for aggressive downscales. Off by default.
            When ``True``, a crop-resize (or fused geo+crop) segment whose worst-axis
            scale drops below ``0.5`` prefilters the input with a Gaussian
            (mipmap-rule sigma) before the single warp, so the downscale no longer
            aliases. Each active image is filtered separately, preventing a batch
            neighbour from changing its filter support. A no-op when the scale is
            safe or the flag is off, keeping the default output bit-identical;
            ``antialias=True`` requires the optional kornia dependency at
            construction.
        substitute_passthrough: When ``True`` (default ``False``), non-fusible
            passthrough ops that have a registered torch-native equivalent in an
            already-installed backend are replaced with that equivalent, so the
            pipeline can stay on-device instead of paying a device-to-host round-trip.
            The initial registry maps Albumentations ``GaussianBlur`` to Kornia
            ``RandomGaussianBlur``. This is **behaviour-changing**: the substitute uses
            a different kernel implementation, border handling, and per-call random
            stream, so outputs and RNG differ from the original op — every substitution
            emits a :class:`UserWarning`. Substitution happens only when the target
            backend is importable; otherwise the original op is kept silently and runs
            on the normal passthrough path. The default (``False``) leaves outputs
            unchanged.
        clip_policy: Clamp policy for fused color segments. ``"final"`` (default)
            applies one clamp after the composed color matmul — most precise.
            ``"per_op_parity"`` splits the fused color run at any op where a
            composed intermediate would leave ``[0, 1]`` and clamps in between,
            approximating the native per-op clamped chain on gamut-escaping
            pipelines. Known gap: when a gamut-escaping op immediately precedes a
            mean-relative contrast op, the contrast midpoint is taken from the
            pre-clamp mean, so output can differ from native by ~1e-2.
        mask_interpolation: Auxiliary mask sampling mode. ``"nearest"`` (default)
            preserves hard labels; ``"bilinear"`` provides differentiable soft-mask
            sampling and requires floating-point mask input.
        mask_fill: Scalar auxiliary-mask border value, independent of image fill. Defaults
            to zero; use a dtype-compatible ignore label such as ``255`` or ``-1``.
            Bilinear soft masks interpolate toward this value outside the source canvas.
        pipeline_dtype: Optional ``"bfloat16"`` or ``"float16"`` GPU execution
            dtype for fused affine/projective/crop warps and fused color/LUT applies.
            Matrix composition and inversion remain float32 or float64, and the
            returned image keeps its input dtype. CPU ignores this option and uses
            the existing float32/float64 path. MPS supports both requested dtypes
            for ``affine_grid`` and ``grid_sample`` on PyTorch 2.10.
        keypoint_flip_index: Keypoint pair permutation applied to the ``"keypoints"``
            target whenever the composed transform reverses orientation: slot ``i`` takes
            its value from slot ``flip_index[i]``. A mirrored image has its left and right
            anatomy swapped, so the *identity* of each slot has to follow the coordinates.
            Which slots pair with which is dataset schema and stays with the caller; this
            package only decides when to apply it, and decides that from the sign of the
            composed matrix's determinant rather than from the presence of a flip transform
            — after fusion the mirror is no longer a discrete op, and two mirrors compose
            back to a rotation that must not swap. ``None`` (default) leaves the keypoint
            axis in input order.
        fill: Constant value written outside the source canvas of every resampling
            warp, in the image's own value range (``114`` for a uint8 grey, ``114 / 255``
            for the same grey as float). A scalar fills every channel; a sequence fills
            one channel each. ``None`` (default) keeps the plain zero border, so existing
            pipelines are unchanged. The fill applies to the **image** only;
            ``mask_fill`` independently controls auxiliary-mask borders. Requires ``padding_mode="zeros"``: the
            other modes have no constant to replace, and the combination raises.
        generator: Caller-owned ``torch.Generator`` driving every pipeline-owned
            draw. ``None`` (default) keeps the global torch stream, leaving existing
            pipelines bit-for-bit unchanged. Backend transform objects sample through
            their own libraries (Kornia's samplers, Albumentations' ``py_random`` and
            ``np.random``, TorchVision's parameter hooks), none of which accept a
            ``torch.Generator``; passing one together with backend transforms raises
            rather than silently falling back to the global stream. Use
            :meth:`from_params` (or ``backend="native"``) for a fully caller-seeded
            pipeline.
        **backend_kwargs: Reserved for backend-specific options (currently unused).

    """

    def __init__(
        self,
        transforms: list[object],
        reorder: ReorderPolicy = ReorderPolicy.NONE,
        interpolation: InterpolationStr | None = None,
        padding_mode: ComposePaddingModeStr | None = None,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy | Literal["backend", "per_sample"] = RandomnessPolicy.BACKEND,
        execution: ExecutionStr = "cv2",
        compile: bool = False,  # noqa: A002 — public flag name mirrors torch.compile; shadowing builtin is intentional
        antialias: bool = False,
        substitute_passthrough: bool = False,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        pipeline_dtype: PipelineDtypeStr | None = None,
        generator: torch.Generator | None = None,
        fill: FillValue | None = None,
        keypoint_flip_index: Sequence[int] | None = None,
        mask_fill: MaskFillValue = 0,
        **backend_kwargs: object,
    ) -> None:
        """Initialize ``FusedCompose``."""
        super().__init__()
        _reject_generator_with_backend_transforms(generator, transforms)
        fill = _validate_fill(fill, padding_mode)
        mask_fill = _validate_mask_fill(mask_fill)
        flip_index = _validate_keypoint_flip_index(keypoint_flip_index)
        randomness_policy = _coerce_randomness_policy(randomness)
        execution = _validate_execution(execution)
        clip_policy = _validate_clip_policy(clip_policy)
        mask_interpolation = _validate_mask_interpolation(mask_interpolation)
        pipeline_dtype = _validate_pipeline_dtype(pipeline_dtype)
        if antialias and not backend_available("kornia"):
            # Installed is not enough: a Kornia that fails to import would only fail later, on the first downscale.
            raise ImportError("antialias=True requires the optional kornia dependency, installed and importable")

        if reorder not in (ReorderPolicy.NONE, ReorderPolicy.POINTWISE, ReorderPolicy.AGGRESSIVE):
            msg = f"ReorderPolicy.{reorder.name} not yet supported"
            raise NotImplementedError(msg)

        # Opt-in substitution: replace registered non-fusible passthrough ops with an
        # already-installed backend's torch-native equivalent so the pipeline can stay
        # on-device. Applied before backend detection so the substitute (a torch op)
        # flows through the normal segmentation path. Behaviour-changing → each swap
        # warns; default off leaves the transform list untouched.
        if substitute_passthrough and transforms:
            transforms = _apply_passthrough_substitution(transforms)

        adapter: TransformAdapter | None
        segments: list[object]
        tfm_adapters: dict[int, TransformAdapter] | None = None

        if not transforms:
            adapter = None
            segments = []
        else:
            per_backends = detect_backends_per_transform(transforms)
            unique_backends = {backend_name for backend_name in per_backends if backend_name is not None}

            if len(unique_backends) <= 1:
                # Single-backend fast path (backward compatible)
                if not unique_backends:
                    raise ValueError(
                        "No recognised backend transforms found in pipeline. "
                        "Ensure transforms are from a supported backend: "
                        "Kornia, TorchVision, or Albumentations."
                    )
                backend = unique_backends.pop()
                adapter = _adapter_for_backend(backend)
                # Unrecognised transforms fall back to this backend's adapter for
                # passthrough dispatch. Kornia/TorchVision adapters call transforms
                # tensor-style, which is compatible with custom torch callables, but
                # the Albumentations adapter uses the dict convention
                # ``transform(image=<ndarray>)``, which silently misbehaves or breaks
                # on foreign objects — reject that combination up front.
                if backend == Backend.ALBUMENTATIONS and any(bknd is None for bknd in per_backends):
                    unknown_names = [
                        type(tfm).__name__ for tfm, bknd in zip(transforms, per_backends, strict=False) if bknd is None
                    ]
                    raise ValueError(
                        f"Unrecognised transform(s) {unknown_names} cannot be dispatched through the "
                        "Albumentations adapter (dict calling convention transform(image=...)). "
                        "Remove them or use a registered Albumentations transform."
                    )
                if reorder == ReorderPolicy.POINTWISE:
                    transforms = reorder_pointwise(transforms, adapter)
                elif reorder == ReorderPolicy.AGGRESSIVE:
                    transforms = reorder_aggressive(transforms, adapter)
                segments = build_segments(
                    transforms=transforms,
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=None if padding_mode == "per_transform" else padding_mode,
                    randomness=randomness_policy,
                    use_numpy=(backend == Backend.ALBUMENTATIONS),
                    route_coords_via_grid=_has_coord_aux(data_keys),
                    route_crop_aux=_has_aux_target(data_keys),
                    execution=execution,
                    compile_warp=compile,
                    per_transform_padding=padding_mode == "per_transform",
                    antialias=antialias,
                    clip_policy=clip_policy,
                    mask_interpolation=mask_interpolation,
                    mask_fill=mask_fill,
                    generator=generator,
                    fill=fill,
                    keypoint_flip_index=flip_index,
                )
            else:
                # Mixed-backend path: group by backend, build segments per group
                adapter, segments, tfm_adapters = _build_mixed_segments(
                    transforms=transforms,
                    per_backends=per_backends,
                    reorder=reorder,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    randomness=randomness_policy,
                    route_coords_via_grid=_has_coord_aux(data_keys),
                    route_crop_aux=_has_aux_target(data_keys),
                    execution=execution,
                    compile_warp=compile,
                    antialias=antialias,
                    clip_policy=clip_policy,
                    mask_interpolation=mask_interpolation,
                    mask_fill=mask_fill,
                    fill=fill,
                    keypoint_flip_index=flip_index,
                )

        self._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
            transform_adapters=tfm_adapters,
            output_backend=output_backend,
            randomness=randomness_policy,
            execution=execution,
            compile_warp=compile,
            antialias=antialias,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            pipeline_dtype=pipeline_dtype,
            generator=generator,
            fill=fill,
            keypoint_flip_index=flip_index,
        )

    def _setup_instance(
        self,
        transforms: list[object],
        reorder: ReorderPolicy,
        interpolation: InterpolationStr | None,
        padding_mode: ComposePaddingModeStr | None,
        data_keys: list[str] | None,
        adapter: TransformAdapter | None,
        segments: list[object],
        transform_adapters: dict[int, TransformAdapter] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        execution: ExecutionStr = "cv2",
        compile_warp: bool = False,
        antialias: bool = False,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        pipeline_dtype: PipelineDtypeStr | None = None,
        generator: torch.Generator | None = None,
        fill: tuple[float, ...] | None = None,
        keypoint_flip_index: tuple[int, ...] | None = None,
        mask_fill: MaskFillValue = 0,
    ) -> None:
        """Assign all instance attributes.

        Called by both ``__init__`` and ``from_params``.

        """
        self.original_transforms: list[object] = list(transforms)
        self.reorder: ReorderPolicy = reorder
        self.interpolation: InterpolationStr | None = interpolation
        self.padding_mode: ComposePaddingModeStr | None = padding_mode
        self.data_keys: list[str] | None = data_keys
        self.randomness: RandomnessPolicy = randomness
        self.execution: ExecutionStr = execution
        self.compile_warp: bool = compile_warp
        self.antialias: bool = antialias
        self.clip_policy: ClipPolicyStr = clip_policy
        self.mask_interpolation: MaskInterpolationStr = _validate_mask_interpolation(mask_interpolation)
        self.mask_fill: MaskFillValue = _validate_mask_fill(mask_fill)
        self.pipeline_dtype: PipelineDtypeStr | None = _validate_pipeline_dtype(pipeline_dtype)
        #: Caller-owned generator, or ``None`` when draws come from the global torch stream.
        #: The segments hold the same object; unpickling resumes its state as of pickling,
        #: so DataLoader workers must be seeded per worker rather than sharing one stream.
        self.generator: torch.Generator | None = generator
        #: Validated constant image border, or ``None`` for the plain zero border.
        self.fill: tuple[float, ...] | None = fill
        #: Validated keypoint pair permutation applied on an orientation-reversing transform.
        self.keypoint_flip_index: tuple[int, ...] | None = keypoint_flip_index
        self._adapter: TransformAdapter | None = adapter
        # Heterogeneous by design: fused/color/exact/crop segments, passthrough wrappers, or raw
        # legacy transforms — dispatched at runtime via _seg_dispatch_tags, so elements stay Any.
        self._segments: list[Any] = _wrap_passthrough_segments(
            segments=segments,
            default_adapter=adapter,
            transform_adapters=transform_adapters,
            original_transforms=transforms,
        )
        self._last_transform_matrix: Tensor | None = None
        self._last_matrix_segment: object | None = None  # deferred resolution
        self._fusion_plan_cache: tuple[bool, str] | None = None
        self._fusion_plan_descriptors_cache: list[SegmentDescriptor] | None = None
        self._transform_adapters: dict[int, TransformAdapter] = transform_adapters or {}
        self._output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = output_backend

        # Derive every runtime dispatch attribute from the core state assigned above.
        # __setstate__ calls the same builder so a pickle produced before any derived
        # attribute existed gets a complete, self-consistent dispatch set instead of
        # raising AttributeError on the first forward after unpickling.
        self._build_derived_state()

        if data_keys is not None:
            # Enforce documented contract: first key must map to the image ("input").
            if not data_keys:
                raise ValueError(
                    "data_keys cannot be an empty list. Omit data_keys entirely for single-tensor "
                    "mode, or include 'input' as the first entry for multi-target mode."
                )
            if data_keys[0] != "input":
                raise ValueError(
                    "data_keys[0] must be 'input' (the image tensor), got "
                    f"{data_keys[0]!r}. This prevents silent misrouting of positional arguments "
                    "in multi-target mode."
                )
            unknown = [key for key in data_keys if key not in _KNOWN_DATA_KEYS]
            if unknown:
                raise ValueError(self._unknown_data_keys_message(unknown))

        # Albumentations fused segments route aux targets through the composed pixel
        # matrix (same coordinate convention as the torch path), so multi-target
        # data_keys are supported for both the cv2 and torch execution strategies.

    def _build_derived_state(self) -> None:
        """Derive every runtime dispatch attribute from the assigned/restored core state.

        Called by both :meth:`_setup_instance` (construction) and :meth:`__setstate__` (unpickling). The derived
        attributes are pure functions of the core state (``_segments``, ``_adapter``, ``data_keys``, ``randomness``,
        ``_output_backend``), so recomputing them on unpickle is always safe and repopulates any attribute absent from
        an older pickle. This keeps :meth:`forward` free of per-call fallbacks and prevents ``AttributeError`` on cross-
        version pickles.

        """
        adapter = self._adapter
        data_keys = self.data_keys

        # Device tracker: a zero-element buffer whose only purpose is to follow the
        # module across ``.to(device)`` / ``.cuda()`` / ``.mps()`` so ``fusion_plan``
        # can report whether the configured pipeline device is non-CPU (a CPU
        # passthrough on a GPU pipeline is a poison pill worth surfacing). Geometric
        # and passthrough segments hold no buffers of their own, so nothing else
        # tracks the module's device.
        if not hasattr(self, "_device_tracker"):
            self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

        # Register nn.Module transforms (backend ops, learnable or stateful custom ops) as submodules so
        # parameters(), state_dict(), .to() and .eval()/.train() reach them. Dispatch still runs through
        # ``_segments`` in plan order; this list only exposes the modules. A pickle predating it gets it here.
        if "_module_transforms" not in self._modules:
            self._module_transforms = nn.ModuleList([
                tfm for tfm in getattr(self, "original_transforms", []) if isinstance(tfm, nn.Module)
            ])

        # Plan-time dispatch: precompute one integer tag per segment so forward()
        # loops without an isinstance chain. data_keys is constructor state, so the
        # single-tensor vs multi-target split is resolved once here as well.
        self._seg_dispatch_tags: list[int] = self._build_dispatch_tags()
        self._multi_target: bool = data_keys is not None
        self._aux_keys: list[str] = list(data_keys[1:]) if data_keys is not None else []

        # Single-transform FusedAffineSegment fast path: bypass the segment's
        # nn.Module.__call__ for single-transform GEOMETRIC_INTERP pipelines (a-group
        # rotate/scale/shear).  Saves ~10-15 us vs going through the segment
        # Module machinery.  Guards:
        # - Single segment, single transform (prevents cv2-path regression from
        #   iter 1 where multi-transform segments were incorrectly bypassed)
        # - Kornia or TorchVision adapter only (not Albu — uses _forward_albu_native)
        # - For TorchVision: v2 transforms only (v1 has incompatible interpolation)
        self._single_fused_fast_seg: FusedAffineSegment | None = None
        if (
            len(self._segments) == 1
            and isinstance(self._segments[0], FusedAffineSegment)
            and len(self._segments[0].transforms) == 1
            and getattr(self._segments[0], "_fast_path", None) in ("kornia", "torchvision")
            and adapter is not None
            and not isinstance(adapter, _DirectParamAdapter)
        ):
            _seg0 = self._segments[0]
            _tfm0 = _seg0.transforms[0]
            _bypass_ok = True
            if _seg0._fast_path == "torchvision":
                try:
                    from fused_transforms.adapters.torchvision import is_torchvision_v2_transform

                    _bypass_ok = is_torchvision_v2_transform(_tfm0)
                except ImportError:
                    _bypass_ok = False
            if _bypass_ok:
                self._single_fused_fast_seg = _seg0
        # Single-transform Albumentations direct fast path: for pipelines with exactly
        # one AlbuFusedAffineSegment containing one transform, bypass
        # _forward_albu_native entirely.  Saves ~5-10 us/call: 4 lazy imports,
        # function call overhead, forward_numpy dispatch, and segment isinstance loop.
        # Guards: single segment + single transform + AlbumentationsAdapter only.
        # Not used when aux_targets are passed (kwargs size > 1).
        self._single_albu_direct_seg: AlbuFusedAffineSegment | None = None
        if (
            len(self._segments) == 1
            and isinstance(self._segments[0], AlbuFusedAffineSegment)
            and len(self._segments[0].transforms) == 1
        ):
            try:
                from fused_transforms.adapters.albumentations import (
                    AlbumentationsAdapter as AlbuAdapterCls,
                )

                if isinstance(adapter, AlbuAdapterCls):
                    self._single_albu_direct_seg = self._segments[0]
            except ImportError:
                pass

        # Pre-computed Albumentations dict-input routing flag: avoids the
        # per-call lazy import + isinstance check in __call__ for Albu pipelines.
        _is_albu: bool = False
        if adapter is not None and not isinstance(adapter, _DirectParamAdapter):
            # Non-importing check: a Kornia/TorchVision pipeline must not import Albumentations to learn it is not one.
            _is_albu = is_builtin_adapter(adapter, Backend.ALBUMENTATIONS)
        self._is_albu_native: bool = _is_albu or (adapter is None and not self._segments)

        # Pre-classify segments for the _forward_albu_native hot path: replace
        # per-call isinstance checks with an integer-tagged dispatch list.
        # Tags: 0=AlbuFusedAffineSegment, 1=ExactAffine, 2=FusedColor,
        # 3=CropResize, 4=Passthrough(Albu), 5=Passthrough(non-Albu), 6=FusedLUT,
        # 7=Gaussian blur, -1=unsupported.
        # Built only for Albu pipelines where _forward_albu_native is used.
        self._albu_seg_tags: list[int] | None = None
        if self._is_albu_native and self._segments:
            tags: list[int] = []
            for seg in self._segments:
                if isinstance(seg, AlbuFusedAffineSegment):
                    tags.append(0)
                elif isinstance(seg, ExactAffineSegment):
                    tags.append(1)
                elif isinstance(seg, FusedColorSegment):
                    tags.append(2)
                elif isinstance(seg, FusedLUTSegment):
                    tags.append(6)
                elif isinstance(seg, FusedGaussianBlurSegment) and not seg.geometric_transforms:
                    tags.append(7)
                elif isinstance(seg, CropResizeSegment):
                    tags.append(3)
                elif isinstance(seg, _PassthroughSegment):
                    tags.append(4 if is_builtin_adapter(seg.adapter, Backend.ALBUMENTATIONS) else 5)
                else:
                    tags.append(-1)
            self._albu_seg_tags = tags

        # Multi-target calls can stay in NumPy/cv2 space only when every segment either carries a
        # composed matrix to route the targets through (tag 0) or cannot move them at all (colour,
        # LUT, blur). A crop, an exact affine or an opaque passthrough changes the geometry without
        # exposing a matrix, so those pipelines keep taking the tensor path.
        #
        # ``execution="torch"`` also opts out, even though the NumPy path would be far faster on CPU:
        # execution is a documented reproducibility axis fixed at construction, and a caller who asked
        # for grid_sample must not be handed a cv2 render because the input happened to be an array.
        # ``execution="auto"`` is eligible: a NumPy array is host data, which is exactly the case its
        # routing rule sends to cv2.
        self._native_multi_ok: bool = bool(
            self._is_albu_native
            and self._multi_target
            and self._aux_keys
            and getattr(self, "_output_backend", None) is None
            and self._albu_seg_tags is not None
            and all(tag in _AUX_SAFE_NATIVE_TAGS for tag in self._albu_seg_tags)
            and all(getattr(seg, "execution", "cv2") in ("cv2", "auto") for seg in self._segments)
        )

        # Resolve the output-backend converter. On a legacy pickle that predates the
        # stored backend flag, keep whatever converter was restored (it cannot be
        # re-derived without the flag) rather than silently dropping it.
        if hasattr(self, "_output_backend"):
            self._output_converter = self._resolve_output_converter(self._output_backend)
        elif not hasattr(self, "_output_converter"):
            self._output_converter = None

    @staticmethod
    def _unknown_data_keys_message(unknown: list[str]) -> str:
        """Build the error for unsupported ``data_keys`` entries, with a did-you-mean hint per key."""
        # Spellings other libraries use for the supported targets, which difflib alone would not map.
        aliases = {
            "image": "input",
            "images": "input",
            "boxes": "bbox_xyxy",
            "bboxes": "bbox_xyxy",
            "bbox": "bbox_xyxy",
            "masks": "mask",
            "segmentation": "mask",
            "keypoint": "keypoints",
            "kpts": "keypoints",
            "rbox": "rboxes",
            "obb": "rboxes",
        }
        known = sorted(_KNOWN_DATA_KEYS)
        parts = []
        for key in unknown:
            guess = aliases.get(key.lower()) or next(iter(difflib.get_close_matches(key, known, n=1)), None)
            parts.append(f"{key!r} (did you mean {guess!r}?)" if guess else repr(key))
        return (
            f"Unknown data_keys {', '.join(parts)}. Supported keys: {known}. An unsupported target would skip the "
            "geometric transform and silently misalign with the augmented image."
        )

    def _check_batched(self, image: object) -> None:
        """Refuse a non-``(B, C, H, W)`` tensor before a fused segment fails on it with an unpacking error."""
        if not isinstance(image, Tensor) or image.ndim == 4:
            return
        if not any(tag in (_TAG_MATRIX, _TAG_PLAIN) for tag in self._dispatch_tags()):
            return  # passthrough-only pipelines hand the tensor to the backend unchanged
        hint = " For a single (C, H, W) image, add the batch axis with x[None] (or x.unsqueeze(0))." * (image.ndim == 3)
        raise ValueError(f"Expected a batched (B, C, H, W) image tensor, got shape {tuple(image.shape)}.{hint}")

    def _inverse_padding_mode(self) -> str:
        """Return the concrete ``grid_sample`` padding mode :meth:`inverse` samples with."""
        if self.padding_mode != "per_transform":
            return self.padding_mode or "zeros"
        # The per-transform policy resolved a concrete mode on the (single, invertible) geometric segment.
        return getattr(self._segments[0], "padding_mode", None) or "zeros"

    def extra_repr(self) -> str:
        """Show the fusion plan and the legacy saved-warp count in ``repr()``."""
        return f"plan={self.fusion_plan!r}, n_warps_saved={self.n_warps_saved}"

    def _resolve_output_converter(
        self,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None,
    ) -> BackendConverter | None:
        """Build the output-backend converter for the configured ``output_backend``.

        Args:
            output_backend: The requested output format, or ``None`` for the native
                ``torch.Tensor`` output.

        Returns:
            A :class:`~fused_transforms.types.BackendConverter` for NumPy output, or
            ``None`` when the output stays as a ``torch.Tensor``.

        Raises:
            ValueError: If *output_backend* is not a supported value.
            RuntimeError: If the resolved converter's ``target_backend`` disagrees with
                the requested *output_backend*.

        """
        if output_backend is None or output_backend == "torch":
            return None
        if output_backend in ("numpy", "numpy_hwc"):
            from fused_transforms.dispatch.converters import TorchToNumpyConverter

            converter: BackendConverter = TorchToNumpyConverter()
        else:
            msg = f"Unknown output_backend {output_backend!r}; supported: 'numpy', 'numpy_hwc', 'torch', None"
            raise ValueError(msg)
        if converter.target_backend not in (
            output_backend,
            "numpy",  # "numpy_hwc" alias maps to TorchToNumpyConverter whose target_backend is "numpy"
        ):
            msg = (
                "Configured output converter target_backend does not match output_backend. "
                f"Requested {output_backend!r}, converter advertises {converter.target_backend!r}."
            )
            raise RuntimeError(msg)
        return converter

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state, rebuilding every derived attribute absent from older pickles.

        All runtime dispatch attributes (``_seg_dispatch_tags``, ``_multi_target``,
        ``_aux_keys``, the single-segment fast paths, ``_is_albu_native``,
        ``_albu_seg_tags``, ``_output_converter``) are derived in
        :meth:`_build_derived_state` and read directly by :meth:`forward`,
        :meth:`__call__`, and :meth:`_build_aux_targets` with no per-call fallback.
        Pickles produced before any of these attributes existed would raise
        :class:`AttributeError` on the first call after unpickling, so this hook
        first restores the backward-compatible defaults of the config/core fields the
        builder reads, then calls the same builder :meth:`_setup_instance` uses. Within
        a single version the builder recomputes identical values, so this is a no-op.

        Args:
            state: The pickled instance ``__dict__`` restored by :mod:`pickle`.

        Examples:
            ```pycon
            >>> import pickle
            >>> from fused_transforms import Compose  # doctest: +SKIP
            >>> pipe = Compose([...])  # doctest: +SKIP
            >>> reloaded = pickle.loads(pickle.dumps(pipe))  # doctest: +SKIP

            ```

        """
        super().__setstate__(state)
        # Config/core fields default to their backward-compatible values on pickles
        # that predate them, so the shared builder below has everything it reads.
        if not hasattr(self, "execution"):
            # Pickles predating the execution flag default to the cv2 strategy.
            self.execution = "cv2"
        if not hasattr(self, "clip_policy"):
            self.clip_policy = "final"
        if not hasattr(self, "mask_interpolation"):
            # Pickles predating configurable mask sampling retain nearest behavior.
            self.mask_interpolation = "nearest"
        if not hasattr(self, "mask_fill"):
            self.mask_fill = 0
        mask_segments = list(self._segments)
        for segment in self._segments:
            if isinstance(segment, FusedGaussianBlurSegment):
                mask_segments.extend((segment._geometric, segment._prefix, segment._suffix))
        for segment in mask_segments:
            if isinstance(
                segment,
                (
                    FusedAffineSegment,
                    ProjectiveSegment,
                    AlbuFusedAffineSegment,
                    AlbuProjectiveSegment,
                    CropResizeSegment,
                ),
            ) and not hasattr(segment, "mask_fill"):
                segment.mask_fill = self.mask_fill
            if isinstance(segment, FusedAffineSegment) and (
                segment.__dict__.pop("_render_overridden_derived", False) or not hasattr(segment, "_render_overridden")
            ):
                # Pickles predating the flag: the Compose-level overrides decide, exactly as build_segments does,
                # replacing the guess the segment's own __setstate__ made from its arguments.
                segment._render_overridden = (
                    getattr(self, "fill", None) is not None
                    or self.interpolation is not None
                    or self.padding_mode not in (None, "per_transform")
                )
            if isinstance(segment, AlbuProjectiveSegment) and not hasattr(segment, "_tfm_tags"):
                segment._tfm_tags = AlbuFusedAffineSegment._classify_transforms(segment.transforms, segment.adapter)
        if not hasattr(self, "pipeline_dtype"):
            # Pickles predating optional low-precision execution retain default precision.
            self.pipeline_dtype = None
        if not hasattr(self, "randomness"):
            self.randomness = RandomnessPolicy.BACKEND
        if not hasattr(self, "generator"):
            # Pickles predating caller-owned randomness drew from the global stream.
            self.generator = None
        if not hasattr(self, "fill"):
            # Pickles predating the constant border padded with zeros.
            self.fill = None
        if not hasattr(self, "keypoint_flip_index"):
            # Pickles predating keypoint pair swapping left the keypoint axis in input order.
            self.keypoint_flip_index = None
        if not hasattr(self, "data_keys"):
            self.data_keys = None
        if not hasattr(self, "_transform_adapters"):
            self._transform_adapters = {}
        # Transient per-call state carries no meaning across a pickle round-trip.
        self._last_transform_matrix = None
        if not hasattr(self, "_last_matrix_segment"):
            self._last_matrix_segment = None
        # Invalidate cached plans: cheaply recomputed and may depend on device-tracking
        # state that a legacy pickle lacks.
        self._fusion_plan_cache = None
        self._fusion_plan_descriptors_cache = None
        # Rebuild every derived dispatch attribute so a pre-attr pickle cannot raise
        # AttributeError on the first forward.
        self._build_derived_state()
        # Pipeline and segments each restored an independent copy of the generator
        # (GeneratorPicklingMixin snapshots by value; identity does not survive pickle).
        # Independent copies advance separate streams while looking correct, so every
        # segment is rebound to the single pipeline-owned instance.
        if self.generator is not None:
            for seg in self._segments:
                if hasattr(seg, "generator"):
                    seg.generator = self.generator

    def __call__(self, *args: object, return_matrix: bool = False, **kwargs: object) -> object:
        """Route to the Albumentations native dict path or the standard tensor path.

        When called with ``image=<numpy.ndarray>`` and the pipeline adapter is an
        :class:`~fused_transforms.adapters.albumentations.AlbumentationsAdapter`,
        the call is dispatched to :meth:`_forward_albu_native`, which keeps the native
        dtype and returns a ``{"image": ndarray}`` dict matching the
        :class:`albumentations.Compose` calling convention.

        A multi-target pipeline (``data_keys`` with auxiliary entries) also accepts the image
        positionally alongside auxiliary keywords — ``pipe(image, bboxes=...)`` — and returns the same
        dict as the all-keyword form. Only that one mixed shape is accepted: a single positional
        argument, with no conflicting ``image=`` keyword.

        On the multi-target path the image may be a channel-last NumPy array
        (``(height, width)``, ``(height, width, channels)`` or a batch) instead of a tensor. Auxiliary
        targets then have to be arrays too, and results come back as arrays in the caller's own
        layout. NumPy image input is normalised the way ``output_backend`` normalises it — ``uint8``
        is scaled internally to float32 in ``[0, 1]`` — and the inferred NumPy round-trip
        restores its original dtype. Mask labels are never intensity-normalized.

        All other invocations fall through to the standard
        :meth:`~torch.nn.Module.__call__` → :meth:`forward` path.

        Args:
            *args: Positional arguments forwarded to :meth:`forward`.
            **kwargs: Keyword arguments.  When ``"image"`` is present,
                ``isinstance(kwargs["image"], np.ndarray)`` is checked to
                decide routing.
            return_matrix: When ``True``, return the output together with the
                matrix produced by the last supported geometric segment.

        Returns:
            ``dict`` with ``"image"`` key (HWC NumPy) for the Albu native path;
            a ``dict`` keyed by the caller's keyword names for a multi-target
            keyword call (``image=..., mask=...``); or whatever :meth:`forward`
            returns for the positional tensor path.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import albumentations as A
            >>> from fused_transforms import Compose
            >>> pipe = Compose([A.HorizontalFlip(p=0.0)])
            >>> img = np.zeros((8, 8, 3), dtype=np.uint8)
            >>> out = pipe(image=img)
            >>> isinstance(out, dict) and out["image"].shape == (8, 8, 3)
            True

            ```

        """
        # Dict-out: an Albumentations-style keyword call (image=..., mask=..., etc.)
        # on a multi-target pipeline routes the aux targets through the fused
        # segments and returns a dict keyed by the caller's keyword names. Only
        # the keyword entry adds dict output; the positional data_keys API is
        # unchanged and still returns a tuple.
        if kwargs and "image" in kwargs and self._multi_target and self._aux_keys:
            if args:
                raise TypeError("Pass the image either positionally or by keyword, not both")
            if isinstance(kwargs["image"], Tensor):
                return super().__call__(return_matrix=return_matrix, **kwargs)
            return self._forward_kwargs_dict(kwargs, return_matrix=return_matrix)

        if (
            self._single_albu_direct_seg is not None
            and len(kwargs) == 1
            and "image" in kwargs
            and isinstance(kwargs["image"], np.ndarray)
        ):
            _clear_current_call_matrix()
            img = self._single_albu_direct_seg.forward_numpy(kwargs["image"])
            self._last_transform_matrix = self._single_albu_direct_seg.last_matrix
            result: object = {"image": img}
            if return_matrix:
                return result, _current_call_matrix()
            return result
        if self._is_albu_native and "image" in kwargs and isinstance(kwargs["image"], np.ndarray):
            if len(kwargs) > 1:
                extra_keys = sorted(key for key in kwargs if key != "image")
                raise NotImplementedError(
                    f"The Albumentations native dict path only transforms 'image'; auxiliary "
                    f"keys {extra_keys} would be silently dropped. Auxiliary targets are not "
                    "supported on this path — pass tensors with data_keys instead."
                )
            _clear_current_call_matrix()
            result = self._forward_albu_native(kwargs["image"])
            if return_matrix:
                return result, _current_call_matrix()
            return result

        # Mixed call: the image positionally, auxiliary targets by keyword -- pipe(img, bboxes=...).
        # Accepted on multi-target pipelines only, and only in this one shape: exactly one positional
        # argument, no "image" keyword to conflict with it. Every other mix still raises below,
        # because with two or more positional arguments there is no unambiguous mapping onto
        # data_keys once some of those keys are also being passed by name.
        if len(args) == 1 and kwargs and self._multi_target and self._aux_keys and "image" not in kwargs:
            if isinstance(args[0], Tensor):
                return super().__call__(*args, return_matrix=return_matrix, **kwargs)
            return self._forward_kwargs_dict({"image": args[0], **kwargs}, return_matrix=return_matrix)

        # Only the declared multi-target API consumes data keywords. Reject other keyword calls
        # consistently before an adapter can reinterpret or silently discard them.
        if kwargs:
            misrouted = sorted(kwargs)
            raise TypeError(
                f"Unexpected keyword argument(s) {misrouted} for a tensor pipeline. Pass the image "
                "positionally -- pipe(image_tensor), not pipe(image=image_tensor). The image=... keyword "
                "call is only supported for Albumentations native (NumPy image) pipelines, or for "
                "multi-target pipelines built with data_keys (image=..., mask=..., ...). A multi-target "
                "pipeline also accepts the image positionally alongside auxiliary keywords -- "
                "pipe(image, bboxes=...) -- but only with a single positional argument."
            )
        return super().__call__(*args, return_matrix=return_matrix, **kwargs)

    def _forward_numpy_exact(self, segment: ExactAffineSegment, image: NDArray[Any]) -> NDArray[Any]:
        """Apply exact pixels and record their matrix without normalizing NumPy intensities."""
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        output = cast(Tensor, segment.forward(tensor.permute(2, 0, 1).unsqueeze(0)))
        self._last_transform_matrix = _current_call_matrix()
        return _tensor_to_numpy_image(output, image.ndim)

    def _forward_albu_native(self, img_hwc: NDArray[Any]) -> dict[str, NDArray[Any]]:
        """Execute the pipeline in Albumentations native dict-input mode.

        Resampling stays in NumPy space. Exact operations use a dtype-preserving
        tensor view so the applied pixels and recorded matrix share one parameter draw.
        :class:`~fused_transforms.affine.segment.AlbuFusedAffineSegment`
        segments are dispatched via their :meth:`forward_numpy` method;
        :class:`~fused_transforms.compose._PassthroughSegment` segments
        are dispatched via
        :meth:`~fused_transforms.adapters.albumentations.AlbumentationsAdapter.call_nonfused_numpy`.

        Args:
            img_hwc: ``(H, W, C)`` or ``(H, W)`` NumPy array (uint8 or float32).

        Returns:
            ``{"image": result_hwc}`` matching the :class:`albumentations.Compose`
            output convention.

        Raises:
            RuntimeError: If the pipeline contains a segment type that is not
                supported in the Albumentations native I/O path.

        """
        # Use pre-classified segment tags when available (built at __init__)
        # to avoid per-call isinstance dispatch overhead.
        _tags = self._albu_seg_tags
        if _tags is not None:
            from fused_transforms.adapters.albumentations import AlbumentationsAdapter

            for idx_segment, seg in enumerate(self._segments):
                tag = _tags[idx_segment]
                if tag == 0:  # AlbuFusedAffineSegment
                    img_hwc = seg.forward_numpy(img_hwc)
                    self._last_transform_matrix = seg.last_matrix
                elif tag == 1:  # ExactAffineSegment
                    img_hwc = self._forward_numpy_exact(seg, img_hwc)
                elif tag == 2:  # FusedColorSegment
                    for tfm in seg._transforms:
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 3:  # CropResizeSegment
                    for tfm in seg.transforms:
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag in (6, 7):  # Folded LUT or Gaussian blur segment
                    img_hwc = seg.forward_numpy(img_hwc)
                elif tag == 4:  # Passthrough(Albu adapter)
                    img_hwc = AlbumentationsAdapter.call_nonfused_numpy(seg.transform, img_hwc)
                elif tag == 5:  # Passthrough(non-Albu adapter)
                    msg = (
                        f"Passthrough segment adapter {type(seg.adapter).__name__!r} does not "
                        "support the Albumentations native I/O path. Use tensor input instead."
                    )
                    raise RuntimeError(msg)
                else:  # unsupported
                    msg = (
                        f"Segment type {type(seg).__name__!r} is not supported in the "
                        "Albumentations native I/O path. Use tensor input instead."
                    )
                    raise RuntimeError(msg)
        else:
            # Fallback: isinstance dispatch (used when _albu_seg_tags was not built).
            from fused_transforms.adapters.albumentations import AlbumentationsAdapter

            for seg in self._segments:
                if isinstance(seg, AlbuFusedAffineSegment):
                    img_hwc = seg.forward_numpy(img_hwc)
                    self._last_transform_matrix = seg.last_matrix
                    continue
                if isinstance(seg, ExactAffineSegment):
                    img_hwc = self._forward_numpy_exact(seg, img_hwc)
                    continue
                if isinstance(seg, FusedColorSegment):
                    for tfm in seg._transforms:
                        img_hwc = tfm(image=img_hwc)["image"]  # type: ignore[operator]
                    continue
                if isinstance(seg, CropResizeSegment):
                    for tfm in seg.transforms:
                        img_hwc = tfm(image=img_hwc)["image"]  # type: ignore[operator]
                    continue
                if isinstance(seg, FusedLUTSegment):
                    img_hwc = seg.forward_numpy(img_hwc)
                    continue
                if isinstance(seg, FusedGaussianBlurSegment) and not seg.geometric_transforms:
                    img_hwc = seg.forward_numpy(img_hwc)
                    continue
                if isinstance(seg, _PassthroughSegment):
                    if isinstance(seg.adapter, AlbumentationsAdapter):
                        img_hwc = AlbumentationsAdapter.call_nonfused_numpy(seg.transform, img_hwc)
                    else:
                        msg = (
                            f"Passthrough segment adapter {type(seg.adapter).__name__!r} does not "
                            "support the Albumentations native I/O path. Use tensor input instead."
                        )
                        raise RuntimeError(msg)
                    continue
                msg = (
                    f"Segment type {type(seg).__name__!r} is not supported in the "
                    "Albumentations native I/O path. Use tensor input instead."
                )
                raise RuntimeError(msg)

        return {"image": img_hwc}

    def _resolve_kwarg_to_data_key(self, name: str) -> str:
        """Map an Albumentations-style keyword name to a declared data key.

        Args:
            name: The keyword used in the call (e.g. ``"image"``, ``"bboxes"``,
                or an exact data-key name like ``"bbox_xyxy"``).

        Returns:
            The matching entry from the pipeline's ``data_keys``.

        Raises:
            ValueError: If the keyword does not correspond to any declared data key.

        """
        data_keys = cast("list[str]", self.data_keys)
        alias = _KWARG_ALIASES.get(name, name)
        candidates: tuple[str, ...] = alias if isinstance(alias, tuple) else (alias,)
        for candidate in candidates:
            if candidate in data_keys:
                return candidate
        msg = (
            f"Keyword {name!r} does not match any entry in data_keys={data_keys}. "
            "Pass a keyword whose (aliased) name is one of the declared data keys."
        )
        raise ValueError(msg)

    def _forward_kwargs_dict(self, kwargs: dict[str, object], *, return_matrix: bool = False) -> object:
        """Run the pipeline for an Albumentations-style keyword call and return a dict.

        Each keyword is mapped to a declared data key, the targets are ordered to
        match ``data_keys``, routed through :meth:`forward`, and returned in a dict
        keyed by the caller's original keyword names. The positional ``data_keys``
        tuple API is unaffected — this dict form is added only for keyword calls.

        Args:
            kwargs: The keyword arguments from the call; must contain ``"image"``
                and one entry per auxiliary ``data_keys`` target.
            return_matrix: When ``True``, return the output dict and its last
                fused geometric matrix.

        Returns:
            A dict mapping each input keyword name to its transformed output.

        Raises:
            ValueError: If a keyword does not match a declared data key, or if the
                keyword set does not cover exactly the pipeline's ``data_keys``.

        """
        data_keys = cast("list[str]", self.data_keys)
        kwarg_to_key = {name: self._resolve_kwarg_to_data_key(name) for name in kwargs}
        # Reject aliasing collisions (e.g. both ``bboxes`` and ``bbox_xyxy``) before
        # inverting the map — otherwise one input would be silently dropped.
        key_to_kwarg: dict[str, str] = {}
        for name, key in kwarg_to_key.items():
            if key in key_to_kwarg:
                msg = (
                    f"Keywords {key_to_kwarg[key]!r} and {name!r} both resolve to data key {key!r}. "
                    "Pass exactly one keyword per declared data key."
                )
                raise ValueError(msg)
            key_to_kwarg[key] = name
        if sorted(key_to_kwarg) != sorted(data_keys):
            msg = (
                f"Keyword call resolved to data keys {sorted(key_to_kwarg)}, but the pipeline "
                f"declares data_keys={data_keys}. Pass exactly one keyword per declared data key."
            )
            raise ValueError(msg)

        ordered_args = tuple(cast("torch.Tensor", kwargs[key_to_kwarg[key]]) for key in data_keys)
        matrix = None
        if (
            self._native_multi_ok
            and all(isinstance(value, np.ndarray) for value in ordered_args)
            and ordered_args[0].ndim in (2, 3)
        ):
            self._last_transform_matrix = None
            _clear_current_call_matrix()
            result: object = self._forward_albu_native_multi(ordered_args)
            matrix = _current_call_matrix()
        else:
            result = self._forward_multi(ordered_args, return_matrix=return_matrix)
            if return_matrix:
                result, matrix = cast("tuple[object, Tensor | None]", result)
        outputs = result if isinstance(result, tuple) else (result,)
        output = {key_to_kwarg[key]: outputs[idx] for idx, key in enumerate(data_keys)}
        return (output, matrix) if return_matrix else output

    def _convert_primary_output(self, image: torch.Tensor) -> torch.Tensor | NDArray[Any]:
        """Convert the primary image output to the requested backend."""
        if self._output_converter is None:
            return image

        image_for_conversion = image.unsqueeze(0) if image.ndim == 3 else image
        return self._output_converter.convert(image_for_conversion)  # type: ignore[no-any-return]

    def forward(
        self,
        *args: torch.Tensor,
        return_matrix: bool = False,
        **kwargs: object,
    ) -> object:
        """Apply the augmentation pipeline to an image batch and optional auxiliary targets.

        **Single-tensor mode** (``data_keys=None``, default): accepts one image
        tensor and returns one tensor or a NumPy ``ndarray`` when
        ``output_backend="numpy"``. This is the backward-compatible path.

        **Multi-target mode** (``data_keys`` is set): accepts positional
        arguments in the same order as ``data_keys``. The first key must map
        to the image (``"input"``); subsequent keys are auxiliary targets
        (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``).
        Returns a tuple of tensors in ``data_keys`` order, or a bare tensor
        when ``data_keys`` contains exactly one entry.

        Args:
            *args: Positional tensors matching the ``data_keys`` list.
                ``args[0]`` is always the image ``(B, C, H, W)`` float32
                channel-first. Auxiliary args follow in ``data_keys[1:]``
                order: ``"mask"`` as ``(B, C, H, W)`` float/int;
                ``"bbox_xyxy"`` / ``"bbox_xywh"`` as ``(B, N, 4)`` float32;
                ``"keypoints"`` as ``(B, N, 2)`` float32; ``"rboxes"`` as
                ``(B, N, 5)`` float32 ``(cx, cy, w, h, theta)``, theta in radians,
                returned un-canonicalized (this package imposes no long-edge or
                angle-range convention -- see :func:`~fused_transforms.targets.transform_rboxes`).
            return_matrix: When ``True``, return the output and its last supported
                geometric pixel-centre matrix.
            **kwargs: Declared multi-target keyword inputs. Tensor keyword calls return a
                dict and retain the same public module hooks as positional tensor calls.

        Returns:
            Single ``Tensor`` or NumPy ``ndarray`` when ``data_keys`` is
            ``None`` or has one entry. NumPy output is returned only when
            ``output_backend="numpy"``. ``tuple[Tensor, ...]`` in
            ``data_keys`` order otherwise.

            When ``return_matrix=True``, returns ``(output, matrix)`` where
            ``matrix`` is the full-precision pixel-centre matrix from the last supported
            geometric segment, or ``None`` when no such segment ran.

        Raises:
            TypeError: If the number of positional arguments does not match
                the number of ``data_keys`` entries (when ``data_keys`` is set),
                or if more than one argument is passed when ``data_keys`` is
                ``None``.

        Note:
            Only known coordinate-preserving passthrough transforms may run with
            auxiliary targets. Unclassified or coordinate-changing passthrough is
            refused before any segment executes, since it cannot route the targets.

            ``output_backend`` conversion is applied per target in multi-target
            mode: image and mask outputs are converted to the requested backend,
            while coordinate targets (boxes, keypoints) remain tensors because the
            channel-last image layout does not apply to them.

        """
        if kwargs:
            if not self._multi_target or not self._aux_keys:
                raise TypeError("Keyword target inputs require a multi-target data_keys pipeline")
            if args:
                if len(args) != 1 or "image" in kwargs:
                    raise TypeError("Pass one positional image or image=, with auxiliary targets by keyword")
                kwargs = {"image": args[0], **kwargs}
            return self._forward_kwargs_dict(kwargs, return_matrix=return_matrix)
        # data_keys is constructor state: dispatch to the pre-resolved single-tensor
        # or multi-target path instead of re-parsing it on every call.
        if not self._multi_target:
            if len(args) != 1:
                msg = f"Expected 1 argument (data_keys is None), got {len(args)}"
                raise TypeError(msg)
            return self._forward_single(args[0], return_matrix=return_matrix)
        return self._forward_multi(args, return_matrix=return_matrix)

    def _build_dispatch_tags(self) -> list[int]:
        """Assign one integer dispatch tag to every segment (see the ``_TAG_*`` constants)."""
        tags: list[int] = []
        for seg in self._segments:
            if isinstance(
                seg,
                (
                    FusedAffineSegment,
                    AlbuFusedAffineSegment,
                    FusedGaussianBlurSegment,
                    ProjectiveSegment,
                    AlbuProjectiveSegment,
                    ExactAffineSegment,
                ),
            ) or (
                isinstance(seg, CropResizeSegment) and getattr(seg.transform, "_coordinate_matrix_recoverable", False)
            ):
                tags.append(_TAG_MATRIX)
            elif isinstance(seg, (FusedColorSegment, FusedLUTSegment, CropResizeSegment)):
                tags.append(_TAG_PLAIN)
            elif isinstance(seg, _PassthroughSegment):
                tags.append(_TAG_PASSTHROUGH)
            else:
                tags.append(_TAG_LEGACY)
        return tags

    def _dispatch_tags(self) -> list[int]:
        """Return the per-segment dispatch tags, rebuilding them for pre-tag pickles."""
        tags = getattr(self, "_seg_dispatch_tags", None)
        if tags is None:
            tags = self._build_dispatch_tags()
            self._seg_dispatch_tags = tags
        return tags

    def _low_precision_segment_dtype(self, segment: object, image: Tensor) -> torch.dtype | None:
        """Return the requested temporary dtype for one supported GPU segment."""
        if self.pipeline_dtype is None or image.device.type == "cpu" or not image.is_floating_point():
            return None
        supported = (FusedAffineSegment, ProjectiveSegment, CropResizeSegment, FusedColorSegment, FusedLUTSegment)
        if isinstance(segment, supported):
            return _PIPELINE_TORCH_DTYPES[self.pipeline_dtype]
        if isinstance(segment, (AlbuFusedAffineSegment, AlbuProjectiveSegment)) and segment.execution == "torch":
            return _PIPELINE_TORCH_DTYPES[self.pipeline_dtype]
        return None

    def _forward_fused_segment(
        self,
        segment: nn.Module,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run one fused segment, narrowing only its GPU image operation when requested."""
        requested_dtype = self._low_precision_segment_dtype(segment, image)
        if requested_dtype is None:
            return cast("Tensor | tuple[Tensor, dict[str, Tensor]]", segment.forward(image, aux_targets))
        result = segment.forward(image.to(dtype=requested_dtype), aux_targets)
        if aux_targets is None:
            return cast(Tensor, result).to(dtype=image.dtype)
        output, routed_targets = cast("tuple[Tensor, dict[str, Tensor]]", result)
        return output.to(dtype=image.dtype), routed_targets

    def _forward_single(self, image: torch.Tensor, *, return_matrix: bool = False) -> object:
        """Run the pipeline in single-tensor mode (``data_keys is None``)."""
        self._check_batched(image)
        # Reset per-call state so transform_matrix returns None when only
        # unsupported segments run (no stale matrix from a previous call).
        self._last_transform_matrix = None
        _clear_current_call_matrix()
        call_matrix: Tensor | None = None

        # Whole-pipeline single-segment fast paths: bypass the segment's
        # nn.Module.__call__ and probability machinery entirely.
        _fast_seg = self._single_fused_fast_seg
        if _fast_seg is not None and self._low_precision_segment_dtype(_fast_seg, image) is None:
            image = cast(Tensor, _fast_seg.forward(image, None))
            call_matrix = _current_call_matrix()
            self._last_transform_matrix = call_matrix
            result = self._convert_primary_output(image)
            return (result, call_matrix) if return_matrix else result

        # Per-segment: call ``forward`` directly (skips nn.Module.__call__ per
        # segment — this generalises the single-op bypass to every segment in a
        # multi-segment pipeline).
        for seg, tag in zip(self._segments, self._dispatch_tags(), strict=True):
            if tag == _TAG_MATRIX:
                image = cast(Tensor, self._forward_fused_segment(seg, image, None))
                call_matrix = _current_call_matrix()
                self._last_transform_matrix = call_matrix
            elif tag == _TAG_PLAIN:
                image = cast(Tensor, self._forward_fused_segment(seg, image, None))
            elif tag == _TAG_PASSTHROUGH:
                image = seg.adapter.call_nonfused(seg.transform, image)
            else:
                image = self._legacy_passthrough(seg, image)
        result = self._convert_primary_output(image)
        return (result, call_matrix) if return_matrix else result

    def _normalize_multi_inputs(
        self,
        args: tuple[object, ...],
    ) -> tuple[tuple[Any, ...], _NumpyRoundTrip | None]:
        """Convert NumPy inputs on the multi-target path to the tensor layout the segments expect.

        The multi-target path operates on ``(batch_size, channels, height, width)`` image tensors and
        ``(batch_size, num_instances, ...)`` coordinate tensors. A caller passing channel-last NumPy
        arrays (the Albumentations convention) is normalised here so both call styles reach the same
        code, and the returned round-trip record lets :meth:`_restore_numpy_outputs` hand results back
        in the caller's own layout.

        Args:
            args: Positional targets in ``data_keys`` order. Mixed tensor/array input is allowed for
                the auxiliary targets; only the image decides whether this is a NumPy call.

        Returns:
            The normalised arguments, and a :class:`_NumpyRoundTrip` recording the input layout — or
            ``None`` when the image was already a tensor, in which case nothing was converted.

        Raises:
            TypeError: If an auxiliary target is a NumPy array but the image is a tensor. Mixing the
                two layouts in one call has no unambiguous batch-dimension interpretation.

        """
        if not isinstance(args[0], np.ndarray):
            if any(isinstance(value, np.ndarray) for value in args[1:]):
                msg = (
                    "Auxiliary targets were passed as NumPy arrays alongside a tensor image. Pass "
                    "every target in the same layout: all tensors, or all NumPy arrays."
                )
                raise TypeError(msg)
            return args, None

        from fused_transforms.dispatch.converters import NumpyToTorchConverter

        data_keys = cast("list[str]", self.data_keys)
        aux_ndims: dict[str, int] = {}
        normalized: list[Any] = [NumpyToTorchConverter().convert(args[0])]
        mask_dtype: np.dtype[Any] | None = None
        for key, value in zip(data_keys[1:], args[1:], strict=True):
            if key == "mask" and isinstance(value, np.ndarray):
                mask_dtype = value.dtype
            normalized.append(self._normalize_aux_input(key, value, aux_ndims))
        return tuple(normalized), _NumpyRoundTrip(
            image_ndim=args[0].ndim,
            aux_ndims=aux_ndims,
            image_dtype=args[0].dtype,
            mask_dtype=mask_dtype,
        )

    @staticmethod
    def _normalize_aux_input(key: str, value: object, aux_ndims: dict[str, int]) -> Any:  # noqa: ANN401
        """Normalise one auxiliary target and record its input rank in *aux_ndims*.

        Masks share the image's channel-last layout but retain label values and dtype. Coordinate targets are already
        numeric ``(num_instances, ...)`` tables, so they only gain the leading batch axis and the float32 dtype the
        target transforms operate on.

        """
        if not isinstance(value, np.ndarray):
            return value
        aux_ndims[key] = value.ndim
        if key == "mask":
            # Labels are not intensities: normalizing uint8 would alter ignore IDs and
            # allow integer masks to bypass the bilinear floating-mask requirement.
            array = np.ascontiguousarray(value)
            if array.dtype == np.uint16:
                # Torch versions before 2.3 cannot ingest uint16; int32 preserves each label.
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(-1)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 4:
                raise ValueError(f"mask must have HW, HWC or BHWC layout, got {tuple(tensor.shape)}")
            return tensor.permute(0, 3, 1, 2)
        tensor = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32)
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

    def _restore_numpy_outputs(self, result: object, round_trip: _NumpyRoundTrip) -> object:
        """Return multi-target outputs in the NumPy layout the caller used for its inputs.

        Applied only when no explicit ``output_backend`` converter is configured — an explicit ``output_backend`` is the
        caller stating what it wants back, and it wins over the inferred round-trip.

        The image and array mask retain their input dtype. Image intensity normalization is undone; mask label values
        were never normalized. Native and tensor-backed routes share this contract. Coordinate targets stay ``float32``
        — a warp puts boxes and keypoints at fractional positions, so restoring an integer input dtype would quantise
        them.

        """
        if self._output_converter is not None:
            return result
        data_keys = cast("list[str]", self.data_keys)
        outputs = result if isinstance(result, tuple) else (result,)
        restored: list[object] = []
        for idx, key in enumerate(data_keys):
            value = outputs[idx]
            if idx == 0:
                image_out = _tensor_to_numpy_image(cast("Tensor", value), round_trip.image_ndim)
                restored.append(_denormalize_numpy_image(image_out, round_trip.image_dtype))
            elif key not in round_trip.aux_ndims:
                restored.append(value)
            elif key == "mask":
                mask_out = _tensor_to_numpy_image(cast("Tensor", value), round_trip.aux_ndims[key])
                if round_trip.mask_dtype is not None:
                    mask_out = mask_out.astype(round_trip.mask_dtype, copy=False)
                restored.append(mask_out)
            else:
                restored.append(_tensor_to_numpy_coords(cast("Tensor", value), round_trip.aux_ndims[key]))
        return tuple(restored) if isinstance(result, tuple) else restored[0]

    def _forward_albu_native_multi(self, ordered_args: tuple[object, ...]) -> tuple[object, ...]:
        """Run a multi-target call without leaving NumPy/cv2 space.

        The tensor path normalises the image to ``float32`` ``(batch_size, channels, height, width)``, warps it, and
        converts it back — three full-image layout copies and a four-times-wider warp that Albumentations never pays.
        Nothing about auxiliary targets requires any of that: coordinate tables are small enough that routing them as
        tensors is free, and the fused matrix the NumPy image path already composes is the same matrix the tensor path
        routes them through. So the image stays in the caller's own dtype and layout, and only the targets round-trip.

        Eligibility is decided once at construction (``_native_multi_ok``); a pipeline holding a crop, an exact affine
        or an opaque passthrough is not eligible and never reaches here.

        Args:
            ordered_args: The call's targets in ``data_keys`` order, image first, every one a NumPy array.

        Returns:
            The transformed targets in ``data_keys`` order. The image keeps its input dtype and layout; coordinate
            targets come back as ``float32`` arrays, matching the tensor path.

        """
        data_keys = cast("list[str]", self.data_keys)
        img_hwc = cast("NDArray[Any]", ordered_args[0])
        aux_ndims: dict[str, int] = {}
        mask_dtype: np.dtype[Any] | None = None
        aux_targets: dict[str, Any] = {}
        for key, value in zip(data_keys[1:], ordered_args[1:], strict=True):
            if key == "mask" and isinstance(value, np.ndarray):
                mask_dtype = value.dtype
            aux_targets[key] = self._normalize_aux_input(key, value, aux_ndims)

        # Validate only shape metadata here: native images must not pay for a full tensor conversion.
        height, width = img_hwc.shape[:2]
        channels = img_hwc.shape[2] if img_hwc.ndim == 3 else 1
        self._validate_aux_targets((1, channels, height, width), aux_targets)
        tags = cast("list[int]", self._albu_seg_tags)
        for idx_segment, seg in enumerate(self._segments):
            if tags[idx_segment] == 2:  # FusedColorSegment — cannot move a coordinate
                for tfm in seg._transforms:
                    img_hwc = tfm(image=img_hwc)["image"]
                continue
            img_hwc = seg.forward_numpy(img_hwc)
            if tags[idx_segment] == 0:  # AlbuFusedAffineSegment — carries the matrix to route through
                self._last_transform_matrix = seg.last_matrix
                seg.route_numpy_aux(aux_targets)

        outputs: list[object] = [img_hwc]
        for key in data_keys[1:]:
            value = aux_targets[key]
            if key not in aux_ndims:
                outputs.append(value)
            elif key == "mask":
                mask_out = _tensor_to_numpy_image(cast("Tensor", value), aux_ndims[key])
                outputs.append(mask_out.astype(mask_dtype, copy=False) if mask_dtype is not None else mask_out)
            else:
                outputs.append(_tensor_to_numpy_coords(cast("Tensor", value), aux_ndims[key]))
        return tuple(outputs)

    def _forward_multi(self, args: tuple[torch.Tensor, ...], *, return_matrix: bool = False) -> object:
        """Run the pipeline in multi-target mode (``data_keys`` is set)."""
        self._last_transform_matrix = None
        _clear_current_call_matrix()
        call_matrix: Tensor | None = None
        data_keys = cast("list[str]", self.data_keys)
        if len(args) != len(data_keys):
            msg = f"Expected {len(data_keys)} arguments for data_keys={self.data_keys}, got {len(args)}"
            raise TypeError(msg)
        args, numpy_round_trip = self._normalize_multi_inputs(args)
        self._check_batched(args[0])
        image = args[0]
        aux_targets = self._build_aux_targets(args)
        # Refuse the entire unsafe chain before a preceding transform can mutate inputs or consume RNG.
        if aux_targets:
            for seg, tag in zip(self._segments, self._dispatch_tags(), strict=True):
                if tag == _TAG_PASSTHROUGH:
                    self._check_passthrough_aux_policy(seg.transform, seg.adapter)
                elif tag == _TAG_LEGACY:
                    self._check_passthrough_aux_policy(seg, self._adapter)

        for seg, tag in zip(self._segments, self._dispatch_tags(), strict=True):
            if tag == _TAG_MATRIX:
                image, aux_targets = self._forward_fused_segment(seg, image, aux_targets)
                call_matrix = _current_call_matrix()
                self._last_transform_matrix = call_matrix
            elif tag == _TAG_PLAIN:
                image, aux_targets = self._forward_fused_segment(seg, image, aux_targets)
            elif tag == _TAG_PASSTHROUGH:
                image = seg.adapter.call_nonfused(seg.transform, image)
            else:
                image = self._legacy_passthrough(seg, image)
        assembled = self._assemble_multi_output(image, aux_targets, args)
        result: object = assembled
        if numpy_round_trip is not None:
            result = self._restore_numpy_outputs(assembled, numpy_round_trip)
        return (result, call_matrix) if return_matrix else result

    @staticmethod
    def _check_passthrough_aux_policy(transform: object, adapter: TransformAdapter | None) -> None:
        """Refuse passthrough unless its adapter proves that coordinates stay unchanged.

        Known distortions retain their specific error. Unknown barriers are unsafe with targets even
        if they happen to preserve an image's shape; only classified pointwise operations and Gaussian
        filtering may leave auxiliary targets untouched. Image-only pipelines keep native dispatch.

        Args:
            transform: The operation that would receive only the image.
            adapter: Its resolved backend classifier, or ``None`` for an unresolved legacy operation.

        Raises:
            ValueError: If coordinate preservation is unknown or the operation needs target routing.

        """
        if is_coordinate_changing_passthrough(transform):
            raise ValueError(
                f"Coordinate-changing passthrough transform {type(transform).__name__!r} moves "
                "image content but does NOT route auxiliary targets. Use a supported target-aware "
                "transform or apply the operation and targets together outside this pipeline."
            )
        safe_categories = {
            TransformCategory.POINTWISE,
            TransformCategory.POINTWISE_LINEAR,
            TransformCategory.POINTWISE_LUT,
            TransformCategory.SPATIAL_LINEAR,
        }
        if adapter is None or adapter.category(transform) not in safe_categories:
            raise ValueError(
                f"Unclassified or unsafe passthrough transform {type(transform).__name__!r} cannot "
                "preserve auxiliary targets reliably. Use a supported coordinate-preserving or "
                "target-aware transform, or handle the image and targets together outside Compose."
            )

    def _validate_aux_targets(self, image_shape: tuple[int, ...], aux_targets: dict[str, Tensor]) -> None:
        """Validate normalized target schemas before any transform executes.

        Coordinate rows carry geometry only; labels and ragged validity remain separate. Empty instance axes are valid,
        while batch broadcasting and extra columns would silently corrupt annotations. Unknown data keys keep their
        documented passthrough contract.

        """
        if len(image_shape) != 4:
            raise ValueError(f"input must have BCHW shape, got {image_shape}")
        widths = {"bbox_xyxy": 4, "bbox_xywh": 4, "keypoints": 2, "rboxes": 5}
        for key, target in aux_targets.items():
            if key not in widths and key != "mask":
                continue
            if not isinstance(target, Tensor):
                raise TypeError(f"{key} must be a tensor after NumPy input normalization")
            expected_ndim = 4 if key == "mask" else 3
            if target.ndim != expected_ndim:
                raise ValueError(f"{key} must have {expected_ndim} dimensions, got shape {tuple(target.shape)}")
            if target.shape[0] != image_shape[0]:
                raise ValueError(f"{key} batch size {target.shape[0]} does not match input batch size {image_shape[0]}")
            if key == "mask":
                if target.shape[-2:] != image_shape[-2:]:
                    raise ValueError(
                        f"mask spatial dimensions {tuple(target.shape[-2:])} must match input {image_shape[-2:]}"
                    )
                continue
            if target.shape[-1] != widths[key]:
                raise ValueError(f"{key} trailing dimension must be {widths[key]}, got {target.shape[-1]}")
            if (
                key == "keypoints"
                and self.keypoint_flip_index is not None
                and target.shape[1] != len(self.keypoint_flip_index)
            ):
                raise ValueError("keypoints slots must match the length of keypoint_flip_index")

    def _build_aux_targets(self, args: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Build the auxiliary-target dict from positional args, rejecting duplicate keys."""
        aux_keys = self._aux_keys
        # Forbid duplicate auxiliary keys to avoid silent overwrites in aux_targets.
        if len(aux_keys) != len(set(aux_keys)):
            msg = (
                "Duplicate entries detected in auxiliary data_keys "
                f"(data_keys[1:]): {aux_keys}. Auxiliary keys must be unique."
            )
            raise ValueError(msg)
        if not isinstance(args[0], Tensor):
            raise TypeError("input must be a tensor after NumPy input normalization")
        targets = dict(zip(aux_keys, args[1:], strict=True))
        self._validate_aux_targets(tuple(args[0].shape), targets)
        return targets

    def _assemble_multi_output(
        self,
        image: torch.Tensor,
        aux_targets: dict[str, torch.Tensor],
        args: tuple[torch.Tensor, ...],
    ) -> torch.Tensor | NDArray[Any] | tuple[torch.Tensor, ...]:
        """Assemble the multi-target return value in ``data_keys`` order.

        When ``output_backend`` is set, the conversion is applied per target: the image and every mask are converted
        (both are ``(B, C, H, W)`` tensors); coordinate targets (boxes, keypoints) are left as tensors because the NumPy
        image converter's channel-last layout does not apply to them.

        """
        data_keys = cast("list[str]", self.data_keys)
        if len(data_keys) == 1:
            return self._convert_primary_output(image)
        result_list: list[torch.Tensor | NDArray[Any]] = []
        for idx, key in enumerate(data_keys):
            value = image if idx == 0 else aux_targets.get(key, args[idx])
            result_list.append(self._convert_target_output(key, value))
        return cast("tuple[torch.Tensor, ...]", tuple(result_list))

    def _convert_target_output(self, key: str, value: torch.Tensor) -> torch.Tensor | NDArray[Any]:
        """Apply the output-backend conversion to a single multi-target output.

        Args:
            key: The target's ``data_keys`` name (``"input"``, ``"mask"``,
                ``"bbox_xyxy"``, ``"bbox_xywh"``, or ``"keypoints"``).
            value: The transformed target tensor.

        Returns:
            The converted value for image/mask targets (a NumPy array when
            ``output_backend="numpy"``), or the unmodified tensor for coordinate
            targets and when no converter is configured.

        """
        if self._output_converter is None or key not in ("input", "mask"):
            return value
        return self._convert_primary_output(value)

    def _legacy_passthrough(self, seg: object, image: torch.Tensor) -> torch.Tensor:
        """Dispatch a legacy (pre-wrapper) passthrough segment by resolving its adapter by index."""
        pt_adapter = None
        for idx, orig_tfm in enumerate(self.original_transforms):
            if orig_tfm is seg:
                pt_adapter = self._transform_adapters.get(idx, self._adapter)
                break
        if pt_adapter is None:
            msg = f"Unknown segment type {type(seg).__name__!r} — update FusedCompose.forward dispatch"
            raise RuntimeError(msg)
        return pt_adapter.call_nonfused(seg, image)

    def inverse(self, image: Tensor, *auxiliary_targets: Tensor, matrix: Tensor | None = None) -> object:
        """Map a paired augmented tensor back to its original geometric frame.

        Pass the matrix returned by the exact paired ``forward(...,
        return_matrix=True)`` call. This avoids reading :attr:`transform_matrix`,
        which is mutable compatibility state and therefore unsafe to pair with an
        output from another concurrent call. The method supports one fused affine
        or projective segment, including a chain already fused into that segment.
        It applies one ``grid_sample`` to the image and routes declared masks,
        boxes, and keypoints through the inverse pixel matrix.

        Keypoints and masks recover to sampling precision. Bounding boxes are
        axis-aligned (AABB), so a forward-then-inverse box is exact only for
        axis-aligned transforms (flip, scale, translation) and inflates under a
        rotation, shear, or projective warp. The matrix is not validated against
        the paired image: a matrix from a different call yields silently wrong
        geometry, so always pass the matrix returned by the same forward call.

        Args:
            image: Augmented ``(B, C, H, W)`` floating-point image tensor.
            *auxiliary_targets: Augmented targets in ``data_keys[1:]`` order.
            matrix: Forward pixel matrix returned by the paired forward call.

        Returns:
            The de-augmented image, or a tuple in ``data_keys`` order when
            auxiliary targets are supplied.

        Raises:
            TypeError: If the image or matrix has an unsupported tensor shape or dtype.
            ValueError: If the pipeline has a non-invertible segment, no paired
                matrix, or targets that do not match ``data_keys``.

        Examples:
            ```pycon
            >>> import torch
            >>> from fused_transforms import Compose
            >>> pipe = Compose.from_params(translate_x=(2.0, 2.0))
            >>> image = torch.rand(1, 3, 8, 8)
            >>> augmented, matrix = pipe(image, return_matrix=True)
            >>> recovered = pipe.inverse(augmented, matrix=matrix)
            >>> recovered.shape
            torch.Size([1, 3, 8, 8])

            ```

        """
        unsupported_reason = self._inverse_unsupported_reason()
        if unsupported_reason is not None:
            raise ValueError(unsupported_reason)
        if matrix is None:
            raise ValueError(
                "Cannot inverse without a paired forward matrix. Pass matrix= returned by the same "
                "forward(..., return_matrix=True) call; transform_matrix is mutable shared state."
            )
        if image.ndim != 4 or not image.is_floating_point():
            raise TypeError("inverse image must be a floating-point BCHW tensor")
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
            raise TypeError("inverse matrix must have shape (batch_size, 3, 3)")
        if matrix.shape[0] != image.shape[0]:
            raise ValueError(
                f"inverse matrix batch size {matrix.shape[0]} does not match image batch size {image.shape[0]}"
            )
        if self.data_keys is None and auxiliary_targets:
            raise ValueError("inverse auxiliary targets require matching data_keys on the pipeline")
        if self.data_keys is not None and len(auxiliary_targets) != len(self._aux_keys):
            raise ValueError(
                f"inverse expected {len(self._aux_keys)} auxiliary targets for data_keys={self.data_keys}, "
                f"got {len(auxiliary_targets)}"
            )

        height, width = image.shape[-2:]
        # Normalize and invert the matrix in full precision even when the augmented
        # image is low precision (mirrors the forward warp), then cast the sampling
        # grid to the image dtype only at the grid_sample boundary. For float32/float64
        # images this is a no-op, so the default path is unchanged.
        matrix_dtype = _matrix_public_dtype(image.dtype)
        forward_matrix = matrix.to(device=image.device, dtype=matrix_dtype)
        sampling_matrix = normalize_matrix(forward_matrix, height, width)
        if isinstance(self._segments[0], (ProjectiveSegment, AlbuProjectiveSegment)):
            grid = perspective_grid(sampling_matrix, height, width)
        else:
            grid = F.affine_grid(
                sampling_matrix[:, :2, :],
                [image.shape[0], image.shape[1], height, width],
                align_corners=True,
            )
        recovered = F.grid_sample(
            image,
            grid.to(dtype=image.dtype),
            mode=self.interpolation or "bilinear",
            padding_mode=self._inverse_padding_mode(),
            align_corners=True,
        )
        if self.data_keys is None:
            return self._convert_primary_output(recovered)

        args = (image, *auxiliary_targets)
        aux_targets = self._build_aux_targets(args)
        inverse_matrix = inv3x3(forward_matrix)
        # Forward routing uses the forward pixel matrix; de-augmentation uses its
        # inverse while reusing the matching output-to-input sampling grid.
        FusedAffineSegment._route_grid_aux(
            aux_targets,
            grid,
            inverse_matrix,
            self.mask_interpolation,
            keypoint_flip_index=self.keypoint_flip_index,
            mask_fill=self.mask_fill,
        )
        return self._assemble_multi_output(recovered, aux_targets, args)


# ---------------------------------------------------------------------------
# Mixed-backend helpers
# ---------------------------------------------------------------------------


# Short alias for convenience; FusedCompose is the canonical name
Compose = FusedCompose
AugmentationSequential = FusedCompose
