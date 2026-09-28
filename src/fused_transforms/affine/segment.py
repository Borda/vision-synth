"""Fused affine segment — vectorised matrix composition and single grid_sample pass.

``FusedAffineSegment`` accumulates per-sample affine matrices for an entire chain
of geometric transforms, inverts the composed matrix once, and executes a single
``grid_sample`` call. No intermediate image warps are performed.

Examples:
    ```pycon
    >>> import torch
    >>> import kornia.augmentation as K
    >>> from fused_transforms.affine.segment import FusedAffineSegment
    >>> from fused_transforms.adapters.kornia import KorniaAdapter
    >>> t = K.RandomHorizontalFlip(p=1.0)
    >>> seg = FusedAffineSegment([t], KorniaAdapter())
    >>> out = seg(torch.zeros(1, 3, 8, 8))
    >>> out.shape
    torch.Size([1, 3, 8, 8])

    ```

"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import Tensor, nn

from fused_transforms._backend import Backend, is_builtin_adapter
from fused_transforms._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, backend_available, import_backend
from fused_transforms._random import GeneratorPicklingMixin, reject_backend_randomness
from fused_transforms._random import rand as _rand
from fused_transforms.affine.matrix import (
    _singularity_threshold,
    apply_d4_image,
    classify_d4_batch,
    estimate_scale,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    normalize_matrix_io,
    perspective_grid,
)
from fused_transforms.types import (
    ClipPolicyStr,
    ExecutionStr,
    InterpolationStr,
    MaskFillValue,
    MaskInterpolationStr,
    PaddingModeStr,
    RandomnessPolicy,
    TransformAdapter,
    TransformCategory,
)

# OpenCV flag values for the cv2 warp paths used by FusedAffineSegment (B=1 CPU fast path) and
# AlbuFusedAffineSegment (Albumentations cv2 backend). They are fixed constants of OpenCV's C API, spelled
# out here because reading them from the module would import OpenCV whenever this package is imported;
# ``test_cv2_flag_literals_match_opencv`` pins them to the installed module.
_CV2_INTERP: dict[str, int] = {
    "bilinear": 1,  # cv2.INTER_LINEAR
    "nearest": 0,  # cv2.INTER_NEAREST
    "bicubic": 2,  # cv2.INTER_CUBIC
}
_CV2_BORDER: dict[str, int] = {
    "zeros": 0,  # cv2.BORDER_CONSTANT
    "border": 1,  # cv2.BORDER_REPLICATE
    # torch grid_sample(padding_mode="reflection", align_corners=True) reflects
    # about the edge sample without duplicating it — cv2.BORDER_REFLECT_101,
    # not cv2.BORDER_REFLECT (which duplicates the edge pixel).
    "reflection": 4,  # cv2.BORDER_REFLECT_101
}
_CV2_WARP_INVERSE_MAP: int = 16  # cv2.WARP_INVERSE_MAP


def _cv2_module() -> ModuleType | None:
    """Return OpenCV, imported on first use, or ``None`` when it is not installed or fails to import."""
    return import_backend("cv2")


#: Base of the two planes a pixel index rides an exact D4 draw in, ``index // base`` and ``index % base``. Each plane
#: stays below ``2**24``, which float32 — the widest float MPS has — represents exactly, so the index is rebuilt as an
#: integer on every device; a canvas past ``base**2`` pixels is refused rather than routed inexactly.
_INDEX_PLANE_BASE = 2**24


def _index_planes(image: Tensor) -> Tensor:
    """Return ``(B, 2, H, W)`` planes encoding each pixel's integer index exactly in the image's stacking dtype.

    Raises:
        ValueError: If the canvas has more pixels than two planes can index exactly.

    """
    batch, _, height, width = image.shape
    base = _INDEX_PLANE_BASE
    if height * width > base * base:
        raise ValueError(
            f"a {height}x{width} canvas is too large to route a mask through an exact transform: its pixel indices "
            f"exceed the {base * base} two index planes can hold exactly"
        )
    index = torch.arange(height * width, device=image.device, dtype=torch.int64).reshape(1, 1, height, width)
    planes = torch.cat([index // base, index % base], dim=1).to(torch.promote_types(image.dtype, torch.float32))
    return planes.expand(batch, 2, height, width)


def _require_cv2() -> ModuleType:
    """Return OpenCV for a cv2 warp path, raising an actionable error when it is unavailable."""
    module = import_backend("cv2")
    if module is None:
        raise ImportError("This cv2 warp path requires opencv-python; install it or use execution='torch'.")
    return module


__doctest_skip__: list[str] = []
if not _KORNIA_AVAILABLE:
    __doctest_skip__ += [".", "ExactAffineSegment", "_FusedGeoCropSegment"]
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["AlbuFusedAffineSegment"]

# Dtype used to compose and invert the (B, 3, 3) affine/projective chain on the
# torch path. float64 keeps matrix accumulation independent of chain length; the
# result is cast back to the image dtype at the grid_sample boundary. The cv2 and
# NumPy paths already accumulate in float64.
_COMPOSE_DTYPE: torch.dtype = torch.float64

# A sampled Gaussian kernel grows quadratically with its radius. Capping the
# support keeps adversarial sigma ranges from allocating an unbounded kernel.
_MAX_SAMPLED_GAUSSIAN_RADIUS = 31


def _matrix_public_dtype(image_dtype: torch.dtype) -> torch.dtype:
    """Return the full-precision dtype exposed for a composed image matrix."""
    if image_dtype == torch.float64:
        return torch.float64
    return torch.float32


def _matrix_geometry_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype auxiliary-target geometry is computed in.

    Coordinate routing is geometry, not image storage. A composed matrix that inherited a low-precision or integer image
    dtype would truncate the affine coefficients that place the image's own boxes and keypoints -- an integer matrix
    rounds every translation and scale to a whole pixel, and silently, because the shapes still line up. Only
    ``float64`` keeps the wider dtype; everything else routes through ``float32``.

    """
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


_CURRENT_CALL_MATRIX: ContextVar[Tensor | None] = ContextVar("fuse_current_call_matrix", default=None)


@dataclass(frozen=True, slots=True)
class _OpaqueBorderModeTransform:
    """Carry a geometric transform that must remain a native border boundary."""

    transform: object
    split_reason: str = "opaque_border_mode"


def _clear_current_call_matrix() -> None:
    """Clear the per-context matrix produced by the next segment call."""
    _CURRENT_CALL_MATRIX.set(None)


def _current_call_matrix() -> Tensor | None:
    """Return the matrix produced by the most recent segment in this context."""
    return _CURRENT_CALL_MATRIX.get()


def _set_current_call_matrix(matrix: Tensor) -> None:
    """Publish a segment's local matrix without using shared pipeline state."""
    _CURRENT_CALL_MATRIX.set(matrix)


def _matrix_compose_dtype(image_dtype: torch.dtype, device: torch.device, num_transforms: int) -> torch.dtype:
    """Pick the dtype used to accumulate and invert a matrix chain.

    A single transform has no chain to accumulate, so float32/float64 images keep
    their existing dtype (and native-warp compatibility). Low-precision image
    operations always use float32 matrix math. Longer float32 chains use float64
    to remove chain-length-dependent drift, except on MPS, which has no float64
    support and therefore composes in float32.

    Args:
        image_dtype: Dtype of the image tensor being warped.
        device: Device the image lives on.
        num_transforms: Number of transforms fused in the segment.

    Returns:
        The dtype to use for matrix composition and inversion.

    """
    if image_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if num_transforms <= 1 or device.type == "mps":
        return image_dtype
    return _COMPOSE_DTYPE


def _scatter_active_matrices(
    mtx: Tensor,
    active: Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Place per-sample matrices into a ``(batch_size, 3, 3)`` identity-filled batch.

    Kornia stores sampled parameters for the ACTIVE subset only (the samples whose
    per-sample probability draw passed), so a reconstructed matrix can have shape
    ``(n_active, 3, 3)`` rather than ``(batch_size, 3, 3)``. This scatters those
    active matrices back into their batch positions, leaving identity on the samples
    the probability mask skipped.

    Args:
        mtx: Reconstructed matrices, shape ``(batch_size, 3, 3)``, ``(n_active, 3, 3)``, or ``(1, 3, 3)``.
        active: Boolean ``(batch_size,)`` mask of applied samples, or ``None`` when the transform always applies.
        batch_size: Target batch size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A ``(batch_size, 3, 3)`` matrix batch.

    """
    if active is None:
        if mtx.shape[0] == 1 and batch_size > 1:
            return mtx.expand(batch_size, -1, -1)
        return mtx
    full = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    if mtx.shape[0] == batch_size:
        # Full-batch matrices: keep active rows, identity on the rest.
        return torch.where(active[:, None, None], mtx, full)
    if mtx.shape[0] == 1:
        return torch.where(active[:, None, None], mtx.expand(batch_size, -1, -1), full)
    n_active = int(active.sum().item())
    if mtx.shape[0] == n_active:
        full[active] = mtx
        return full
    msg = (
        f"Cannot align a reconstructed matrix batch of shape {tuple(mtx.shape)} with a batch of "
        f"{batch_size} ({n_active} active). Expected the batch size, the active count, or 1."
    )
    raise RuntimeError(msg)


def _shares_randomness_across_batch(
    adapter: TransformAdapter,
    transform: object,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
) -> bool:
    """Return whether a transform should draw one random decision for the batch."""
    if randomness is RandomnessPolicy.PER_SAMPLE:
        return False
    same_on_batch = getattr(adapter, "same_on_batch", None)
    if callable(same_on_batch):
        return bool(same_on_batch(transform))
    return bool(getattr(transform, "same_on_batch", False))


def _samples_on_the_matrix_device(transform: object) -> bool:
    """Return whether this transform's matrix build may draw RNG on the sampling device.

    Only Kornia's ``RandomRotation90`` does. Its ``build_matrix`` falls back to
    ``torch.randint(..., device=params["_batch_size"].device)`` when the sampled parameters carry
    no ``k90``, so moving its sampling to the host would move that draw to a different RNG stream
    and change the quarter-turns a seeded pipeline produces. Every other adapter path derives its
    parameters from host RNG, where the construction device of the resulting tensor is immaterial.

    Detected by name so an absent or older Kornia does not make this an import-time dependency; the
    check is a per-call guard on a small transform list, and a false positive only forgoes an
    optimisation.

    Args:
        transform: A transform instance from a fused segment's chain.

    Returns:
        ``True`` when parameter sampling must stay on the image's device.

    """
    return type(transform).__name__ == "RandomRotation90"


def _sample_transform_params(
    adapter: TransformAdapter,
    transform: object,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Sample params, preferring adapter-provided per-sample sampling when requested.

    Args:
        adapter: The adapter owning parameter sampling for ``transform``.
        transform: The transform to sample parameters for.
        input_shape: ``(batch_size, channels, height, width)`` shape tuple.
        device: Target device for the returned tensors.
        randomness: Batch randomness policy.
        generator: Caller-owned generator, or ``None`` for the global stream.

    Returns:
        Dict of canonical parameter tensors.

    """
    if generator is not None and not getattr(adapter, "supports_generator", False):
        # Backstop: construction already rejects a generator on backend pipelines.
        # Degrading to the global stream here would look reproducible and not be.
        reject_backend_randomness(generator, f"{type(adapter).__name__} sampling for {type(transform).__name__}")
    if randomness is RandomnessPolicy.PER_SAMPLE:
        sample_params_per_sample = getattr(adapter, "sample_params_per_sample", None)
        if callable(sample_params_per_sample):
            return cast(dict[str, Tensor], sample_params_per_sample(transform, input_shape, device))
    if generator is not None:
        # Only an adapter advertising supports_generator reaches this line (the guard above),
        # and its sample_params takes the extra keyword the protocol does not declare.
        seeded_sampler = cast(Callable[..., dict[str, Tensor]], adapter.sample_params)
        return seeded_sampler(transform, input_shape, device, generator=generator)
    return adapter.sample_params(transform, input_shape, device)


def _transform_prob(transform: object, default: float = 1.0) -> float:
    """Return a transform's application probability, preferring ``prob`` then ``p``."""
    prob = getattr(transform, "prob", None)
    if prob is not None:
        return float(prob)
    return float(getattr(transform, "p", default))


def _validate_execution(execution: str) -> ExecutionStr:
    """Validate and return an Albumentations execution-strategy value.

    Args:
        execution: The requested strategy; must be ``"cv2"``, ``"torch"`` or ``"auto"``.

    Returns:
        The validated strategy string.

    Raises:
        ValueError: If ``execution`` is not one of the three accepted values.

    Examples:
        ```pycon
        >>> _validate_execution("cv2")
        'cv2'
        >>> _validate_execution("auto")
        'auto'

        ```

    """
    if execution not in ("cv2", "torch", "auto"):
        msg = f"execution must be 'cv2', 'torch' or 'auto', got {execution!r}."
        raise ValueError(msg)
    return cast(ExecutionStr, execution)


def _resolve_execution(execution: ExecutionStr, device: torch.device) -> ExecutionStr:
    """Resolve ``"auto"`` to a concrete engine for one call; pass the other values through.

    The rule is fixed and documented rather than adaptive or measured at runtime: host data goes to
    OpenCV, accelerator data goes to ``grid_sample``. That matches where each engine can actually run --
    ``cv2.warpAffine`` requires a host array, and moving a CUDA batch to the host to warp it would cost
    more than the warp -- and it is simple enough to state in one sentence, which is the property a
    routing rule needs if a caller is to predict it.

    Args:
        execution: The configured strategy.
        device: Device the image being warped lives on.

    Returns:
        ``"cv2"`` or ``"torch"``; never ``"auto"``.

    Examples:
        ```pycon
        >>> import torch
        >>> _resolve_execution("auto", torch.device("cpu"))
        'cv2'
        >>> _resolve_execution("torch", torch.device("cpu"))
        'torch'

        ```

    """
    if execution != "auto":
        return execution
    return "cv2" if device.type == "cpu" else "torch"


#: cv2 reads ``borderValue`` as a ``Scalar``, which carries exactly four components.
_CV2_SCALAR_SLOTS = 4


def _fill_tensor(
    fill: tuple[float, ...] | None, num_channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor | None:
    """Shape a validated fill into a ``(1, channels, 1, 1)`` tensor for the warp helpers.

    Args:
        fill: Validated fill tuple (length 1 for a scalar), or ``None``.
        num_channels: Channel count of the image being warped.
        device: Device of the image being warped.
        dtype: Dtype of the image being warped.

    Returns:
        A broadcastable ``(1, channels, 1, 1)`` tensor, or ``None`` when there is no fill.

    Raises:
        ValueError: If a per-channel fill's length matches neither 1 nor ``num_channels``.

    """
    if fill is None:
        return None
    if len(fill) not in (1, num_channels):
        msg = f"fill has {len(fill)} value(s) but the image has {num_channels} channel(s)."
        raise ValueError(msg)
    values = fill * num_channels if len(fill) == 1 else fill
    return torch.tensor(values, device=device, dtype=dtype).reshape(1, num_channels, 1, 1)


def _cv2_border_value(fill: tuple[float, ...] | None, num_channels: int) -> tuple[float, ...]:
    """Return the cv2 ``borderValue`` scalar tuple for a validated fill.

    cv2 reads ``borderValue`` as a 4-component ``Scalar``, so a bare number fills only
    channel 0 and leaves the rest black — the tuple is always built out to the image's
    channel count rather than passed through as a scalar.

    Args:
        fill: Validated fill tuple (length 1 for a scalar), or ``None``.
        num_channels: Channel count of the image being warped.

    Returns:
        A ``borderValue`` tuple of length ``num_channels``; all zeros when there is no fill.

    Raises:
        ValueError: If a per-channel fill's length matches neither 1 nor ``num_channels``,
            or if the image has more than the four channels a cv2 ``Scalar`` can carry.

    """
    if fill is None:
        return (0.0,) * min(num_channels, _CV2_SCALAR_SLOTS)
    if len(fill) not in (1, num_channels):
        msg = f"fill has {len(fill)} value(s) but the image has {num_channels} channel(s)."
        raise ValueError(msg)
    if num_channels > _CV2_SCALAR_SLOTS:
        msg = (
            f"fill is not expressible on the cv2 warp path for a {num_channels}-channel image: "
            f"cv2 borderValue carries at most {_CV2_SCALAR_SLOTS} components. "
            "Use execution='torch' for this image."
        )
        raise ValueError(msg)
    return fill * num_channels if len(fill) == 1 else fill


def _grid_sample_affine_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
    fill: Tensor | None = None,
    *,
    compiling: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Warp ``image`` by an affine matrix batch with one ``affine_grid`` + ``grid_sample`` pass.

    Inverts the composed forward matrix, normalizes it to the ``[-1, 1]`` grid
    convention, builds an affine grid, and resamples the whole batch at once. This
    is the single batched affine executor shared by the torch affine segment and
    the Albumentations torch execution strategy.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        acc: ``(batch_size, 3, 3)`` composed forward matrix. Any floating dtype;
            the inversion runs in this dtype, so callers pass float64 on CPU/CUDA
            and float32 on MPS (which has no float64).
        interpolation: ``grid_sample`` interpolation mode.
        padding_mode: ``grid_sample`` padding mode.
        fill: Optional ``(1, channels, 1, 1)`` constant written outside the source
            canvas. ``grid_sample`` has no constant padding mode, so the fill is
            subtracted before sampling and added back after: an out-of-canvas sample
            reads the zero padding and comes back as exactly the fill, while a boundary
            sample blends the interior with the fill the way a constant border does.
        compiling: Forwarded to :func:`~fused_transforms.affine.matrix.inv3x3`
            to select the compile-safe branch explicitly; ``None`` falls back to
            ambient ``torch.compile`` detection for ordinary eager calls.

    Returns:
        A ``(warped_image, grid)`` tuple; the grid is reused to warp mask aux targets.

    """
    batch_size, num_channels, height, width = image.shape
    dtype = image.dtype
    mtx_inv = inv3x3(acc, compiling=compiling)
    mtx_norm = normalize_matrix(mtx_inv, height, width).to(dtype=dtype)

    grid = F.affine_grid(mtx_norm[:, :2, :], [batch_size, num_channels, height, width], align_corners=True)
    sampled = image if fill is None else image - fill
    warped = F.grid_sample(sampled, grid, mode=interpolation, padding_mode=padding_mode, align_corners=True)
    return (warped if fill is None else warped + fill), grid


def _grid_sample_perspective_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
    fill: Tensor | None = None,
    *,
    compiling: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Warp ``image`` by a homography batch with one ``perspective_grid`` + ``grid_sample`` pass.

    Like :func:`_grid_sample_affine_batched` but builds a perspective grid (with
    the perspective division ``F.affine_grid`` cannot express), so it handles the
    full ``3x3`` homography. Shared by the torch projective segment and the
    Albumentations projective torch execution strategy.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        acc: ``(batch_size, 3, 3)`` composed forward homography (float64 on
            CPU/CUDA, float32 on MPS).
        interpolation: ``grid_sample`` interpolation mode.
        padding_mode: ``grid_sample`` padding mode.
        fill: Optional ``(1, channels, 1, 1)`` constant written outside the source
            canvas, applied by the same subtract/add construction as
            :func:`_grid_sample_affine_batched`.
        compiling: Forwarded to :func:`~fused_transforms.affine.matrix.inv3x3`;
            see :func:`_grid_sample_affine_batched`.

    Returns:
        A ``(warped_image, grid)`` tuple.

    """
    _, _, height, width = image.shape
    dtype = image.dtype
    mtx_inv = inv3x3(acc, compiling=compiling)
    mtx_norm = normalize_matrix(mtx_inv, height, width).to(dtype=dtype)

    grid = perspective_grid(mtx_norm, height, width)
    sampled = image if fill is None else image - fill
    warped = F.grid_sample(sampled, grid, mode=interpolation, padding_mode=padding_mode, align_corners=True)
    return (warped if fill is None else warped + fill), grid


def _compiling_grid_sample_affine_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
    fill: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """``_grid_sample_affine_batched`` with the compile-safe inversion branch pinned on.

    The only entry point ``torch.compile`` wraps. Binding ``compiling=True`` here is a plain Python constant fixed at
    trace time, not an ambient runtime probe, so branch selection cannot depend on how a given torch/dynamo version
    reports its own tracing state from a resumed frame (observed as a spurious graph break with ambient detection on
    torch 2.2).

    """
    return _grid_sample_affine_batched(image, acc, interpolation, padding_mode, fill, compiling=True)


def _compiling_grid_sample_perspective_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
    fill: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """``_grid_sample_perspective_batched`` with the compile-safe inversion branch pinned on.

    See :func:`_compiling_grid_sample_affine_batched`.

    """
    return _grid_sample_perspective_batched(image, acc, interpolation, padding_mode, fill, compiling=True)


def _torch_supports_compile() -> bool:
    """Return whether the installed torch is new enough for a reliable ``torch.compile``.

    The compiled warp region is gated on torch >= 2.2 at runtime rather than
    raising the package floor: older torch keeps the eager path unchanged and the
    ``compile=True`` flag becomes a documented no-op. The version is parsed from
    ``torch.__version__`` (major/minor only; release-candidate and ``+cpu`` build
    suffixes are ignored).

    Returns:
        ``True`` when ``torch.__version__`` is ``2.2`` or newer, ``False`` otherwise.

    Examples:
        ```pycon
        >>> from fused_transforms.affine.segment import _torch_supports_compile
        >>> isinstance(_torch_supports_compile(), bool)
        True

        ```

    """
    parts = torch.__version__.split("+", 1)[0].split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return False
    return (major, minor) >= (2, 2)


# Signature shared by both warp cores: (image, matrix, interpolation, padding) ->
# (warped_image, sampling_grid). Used to type the compiled-warp cache/selectors.
WarpFn = Callable[[Tensor, Tensor, InterpolationStr, PaddingModeStr, Tensor | None], tuple[Tensor, Tensor]]

# Module-level compiled warp cores, built lazily on first use so importing the
# module never triggers a compile. ``dynamic=True`` keeps a single guarded graph
# across varying (batch, height, width) instead of recompiling per shape. Keyed
# once at module scope so every segment shares the same compiled function object
# (and therefore the same inductor cache), rather than compiling per instance.
_COMPILED_WARP_CACHE: dict[str, WarpFn] = {}


ColorFn = Callable[[Tensor, Tensor], Tensor]


def _apply_color_matrix(image: Tensor, acc: Tensor) -> Tensor:
    """Apply a homogeneous color matrix to every RGB pixel in an image batch.

    Args:
        image: ``(batch_size, 3, height, width)`` image tensor.
        acc: ``(batch_size, 4, 4)`` homogeneous color matrices.

    Returns:
        The color-transformed image with the same shape and dtype as ``image``.

    """
    batch_size, channels, height, width = image.shape
    pixels = image.reshape(batch_size, channels, height * width)
    linear = acc[:, :channels, :channels]
    bias = acc[:, :channels, channels : channels + 1]
    transformed = torch.baddbmm(bias, linear, pixels)
    return transformed.reshape(batch_size, channels, height, width)


# Color parameter sampling and probability masking stay eager. This cache holds only
# the dense post-warp matrix application, whose inputs are already ordinary tensors.
_COMPILED_COLOR_CACHE: dict[str, ColorFn] = {}


def _compiled_color_fn() -> ColorFn:
    """Return the dynamic-shape compiled color-matrix application core.

    Returns:
        The cached ``torch.compile`` wrapper for :func:`_apply_color_matrix`.

    """
    cached = _COMPILED_COLOR_CACHE.get("matrix")
    if cached is None:
        cached = torch.compile(_apply_color_matrix, dynamic=True)
        _COMPILED_COLOR_CACHE["matrix"] = cached
    return cached


def _compiled_warp_fn(kind: str) -> WarpFn:
    """Return the ``torch.compile``-wrapped warp core for ``kind`` (``"affine"`` or ``"perspective"``).

    The compiled function wraps the same ``inv3x3 -> normalize -> grid -> grid_sample``
    core used on the eager path; only the matrix-inversion / grid-generation math is
    inside the graph. Probability masking and active-subset selection stay in the
    callers, so the compiled region has no data-dependent control flow and no graph
    breaks. Compilation happens once per kind and is memoized.

    Args:
        kind: ``"affine"`` for :func:`_grid_sample_affine_batched` or
            ``"perspective"`` for :func:`_grid_sample_perspective_batched`.

    Returns:
        The compiled callable for the requested warp core.

    """
    cached = _COMPILED_WARP_CACHE.get(kind)
    if cached is not None:
        return cached
    base = _compiling_grid_sample_affine_batched if kind == "affine" else _compiling_grid_sample_perspective_batched
    compiled = torch.compile(base, dynamic=True)
    _COMPILED_WARP_CACHE[kind] = compiled
    return compiled


# Below this per-axis scale (output/input), a plain grid_sample downscale drops
# high-frequency detail between samples and aliases; the antialias path kicks in
# only when a downscale is this aggressive. At/above it the warp is bit-identical
# to the un-antialiased path (Nyquist headroom), so the opt-in flag is a no-op.
_ANTIALIAS_SCALE_THRESHOLD: float = 0.5
# Off-diagonal magnitude under which the composed 2x2 counts as axis-aligned
# (pure scale + translation, no rotation/shear) — the case an ``F.interpolate``
# antialias tail handles exactly, matching ``torchvision.Resize(antialias=True)``.
_AXIS_ALIGNED_EPS: float = 1e-6


def _mipmap_sigma(scale: float) -> float:
    """Return the Gaussian pre-blur sigma for a downscale factor ``scale`` (< 1).

    Uses the standard mipmap pre-filter rule ``sigma = 0.5 * sqrt((1/s)^2 - 1)``:
    the blur that band-limits the input to the output Nyquist frequency before a
    ``1/s``-fold decimation, so the single downscaling warp no longer aliases.

    Args:
        scale: The per-axis output/input scale factor, expected ``0 < scale < 1``.

    Returns:
        The Gaussian standard deviation in input pixels (``0.0`` when ``scale >= 1``).

    Examples:
        ```pycon
        >>> from fused_transforms.affine.segment import _mipmap_sigma
        >>> round(_mipmap_sigma(0.25), 4)
        1.9365

        ```

    """
    if scale >= 1.0:
        return 0.0
    return 0.5 * math.sqrt((1.0 / scale) ** 2 - 1.0)


def _antialias_axis_scales(mtx: Tensor) -> tuple[Tensor, Tensor]:
    """Return one ``(width-axis, height-axis)`` scale pair for every image.

    Unlike :func:`~fused_transforms.affine.matrix.estimate_scale` (whose two
    singular values are sorted by *magnitude*, not by image axis), this maps each
    scale to the axis the anisotropic Gaussian must blur:

    - **Axis-aligned** linear part (off-diagonal magnitude below
      :data:`_AXIS_ALIGNED_EPS`, i.e. pure scale + translation): the width- and
      height-axis scales are read straight off the diagonal ``mtx[:, 0, 0]`` /
      ``mtx[:, 1, 1]``, so a height-dominant shrink blurs the height axis and a
      width-dominant shrink blurs the width axis.
    - **Rotated / sheared** linear part: width and height no longer align with the
      matrix axes, so the smallest singular value (worst-axis scale) is applied
      isotropically — a conservative band-limit that never under-blurs.

    The returned tensors retain one scale pair per batch row. The prefilter then
    processes only the rows that need filtering, so a neighbouring image cannot
    widen this image's Gaussian support or blur an otherwise safe sample.

    Args:
        mtx: ``(batch_size, 3, 3)`` forward pixel matrix of the downscaling warp.

    Returns:
        A ``(scale_x, scale_y)`` tuple of ``(batch_size,)`` tensors.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.affine.segment import _antialias_axis_scales
        >>> mtx = torch.eye(3).unsqueeze(0)
        >>> mtx[:, 0, 0], mtx[:, 1, 1] = 0.9, 0.2  # shrink height much harder than width
        >>> sx, sy = _antialias_axis_scales(mtx)
        >>> round(float(sx[0]), 3), round(float(sy[0]), 3)
        (0.9, 0.2)

        ```

    """
    linear = mtx[:, :2, :2].to(dtype=torch.float32)
    off_diagonal = torch.maximum(linear[:, 0, 1].abs(), linear[:, 1, 0].abs())
    axis_aligned = off_diagonal <= _AXIS_ALIGNED_EPS
    singular_min = torch.linalg.svdvals(linear)[:, 1]
    scale_x = torch.where(axis_aligned, linear[:, 0, 0].abs(), singular_min)
    scale_y = torch.where(axis_aligned, linear[:, 1, 1].abs(), singular_min)
    return scale_x, scale_y


def _maybe_antialias_prefilter(image: Tensor, mtx: Tensor, enabled: bool) -> Tensor:
    """Gaussian-prefilter ``image`` before a downscaling warp when antialiasing is on.

    Computes each sample's per-axis scale once from the forward matrix ``mtx``.
    Every aggressive sample is then filtered independently, so a smaller or more
    anisotropic neighbour cannot determine its Gaussian support. When antialiasing
    is enabled and the worst axis downscales below
    :data:`_ANTIALIAS_SCALE_THRESHOLD`, band-limits the input with a per-axis
    Gaussian (mipmap sigma rule) so the single ``grid_sample`` no longer aliases.
    The blur runs in the image dtype via the installed kornia backend. Construction
    rejects ``antialias=True`` without that optional dependency. Returns ``image``
    untouched when disabled or every scale is safe, so the default path stays
    bit-identical.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        mtx: ``(batch_size, 3, 3)`` forward pixel matrix of the downscaling warp.
        enabled: Whether the ``antialias`` flag is set for this segment.

    Returns:
        The prefiltered image, or the original image when no filtering applies.

    """
    if not enabled:
        return image
    scale_x, scale_y = _antialias_axis_scales(mtx)
    active = torch.minimum(scale_x, scale_y) < _ANTIALIAS_SCALE_THRESHOLD
    if not bool(active.any().item()):
        return image
    sigma_x = 0.5 * torch.sqrt((scale_x.reciprocal().square() - 1.0).clamp_min(0.0))
    sigma_y = 0.5 * torch.sqrt((scale_y.reciprocal().square() - 1.0).clamp_min(0.0))
    result = image.clone()
    for index in active.nonzero(as_tuple=True)[0].tolist():
        blurred = _kornia_gaussian_blur(image[index : index + 1], sigma_x[index], sigma_y[index])
        if blurred is None:
            raise RuntimeError("antialias=True requires the optional kornia dependency")
        result[index : index + 1] = blurred
    return result


def _kornia_gaussian_blur(image: Tensor, sigma_x: float | Tensor, sigma_y: float | Tensor) -> Tensor | None:
    """Anisotropic Gaussian pre-blur via kornia, or ``None`` when kornia is absent.

    Blurs ``image`` with per-axis sigmas using the installed kornia
    ``gaussian_blur2d`` (no custom kernel). The kernel size follows the ``3-sigma``
    rule, rounded up to the next odd integer per axis. Returns ``None`` when kornia
    is not importable so the caller can fall back to the un-filtered warp.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        sigma_x: Gaussian standard deviation along the width axis, in pixels.
        sigma_y: Gaussian standard deviation along the height axis, in pixels.

    Returns:
        The blurred tensor, or ``None`` when kornia is unavailable or both sigmas are ~0.

    """
    batch_size = image.shape[0]
    sig_x = torch.as_tensor(sigma_x, device=image.device, dtype=image.dtype).reshape(-1)
    sig_y = torch.as_tensor(sigma_y, device=image.device, dtype=image.dtype).reshape(-1)
    if sig_x.numel() == 1:
        sig_x = sig_x.expand(batch_size)
    if sig_y.numel() == 1:
        sig_y = sig_y.expand(batch_size)
    if sig_x.numel() != batch_size or sig_y.numel() != batch_size:
        msg = "Gaussian blur sigmas must be scalar or have one value per image"
        raise ValueError(msg)
    if bool(((sig_x <= 0.0) & (sig_y <= 0.0)).all()):
        return image
    if not backend_available("kornia"):
        return None
    from kornia.filters import gaussian_blur2d

    ksize_x = 2 * math.ceil(3.0 * float(sig_x.max().item())) + 1
    ksize_y = 2 * math.ceil(3.0 * float(sig_y.max().item())) + 1
    # kornia expects positive sigmas on both axes; clamp the near-zero axis to a
    # tiny value so a single-axis downscale still runs through one call.
    sigma = torch.stack([sig_y.clamp_min(1e-6), sig_x.clamp_min(1e-6)], dim=1)
    return gaussian_blur2d(image, kernel_size=(ksize_y, ksize_x), sigma=sigma, border_type="reflect")


class ExactAffineSegment(GeneratorPicklingMixin, nn.Module):
    """Lossless segment for GEOMETRIC_EXACT-only chains.

    Used when a run of consecutive geometric transforms consists entirely of ``GEOMETRIC_EXACT`` operations, such as
    flips and other discrete, lossless image-space transforms supported by the active adapter (for example 90-degree
    rotations or transpose-like ops). Applies each transform via :meth:`TransformAdapter.exact_apply` instead of
    ``grid_sample``, introducing zero interpolation error.

    Per-sample probability masking is implemented by sampling a boolean mask of shape ``(B,)`` from each transform's
    application probability and applying the exact transform only to active samples. The fused engine prefers a ``prob``
    attribute when present and falls back to backend ``p`` for native transform objects.

    Auxiliary-target routing: masks route for every exact op, and coordinates
    route through the matrix built from the same sampled parameters as pixels.
    Flips retain their pixel-edge AABB and keypoint-pair rules. A geometric run
    that combines a non-flip exact op with box/keypoint targets may also be
    built as a :class:`FusedAffineSegment` through ``route_coords_via_grid``.

    Args:
        transforms: List of ``GEOMETRIC_EXACT`` transform objects.
        adapter: A ``TransformAdapter`` providing ``exact_apply`` for image
            updates and, when auxiliary targets are used, ``exact_flip_dims`` for flip-compatible target routing.
        randomness: Batch randomness policy. ``BACKEND`` preserves native
            backend semantics; ``PER_SAMPLE`` draws probability masks per item.
        generator: Caller-owned generator driving the per-transform probability
            gates, or ``None`` for the global torch stream.

    Examples:
        ```pycon
        >>> import torch
        >>> import kornia.augmentation as K
        >>> from fused_transforms.affine.segment import ExactAffineSegment
        >>> from fused_transforms.adapters.kornia import KorniaAdapter
        >>> t = K.RandomHorizontalFlip(p=1.0)
        >>> seg = ExactAffineSegment([t], KorniaAdapter())
        >>> out = seg(torch.zeros(1, 3, 8, 8))
        >>> out.shape
        torch.Size([1, 3, 8, 8])

        ```

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        generator: torch.Generator | None = None,
        keypoint_flip_index: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize ``ExactAffineSegment``."""
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.randomness = randomness
        self.generator = generator
        self.keypoint_flip_index = keypoint_flip_index
        self._last_matrix: Tensor | None = None

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the actual forward matrix from the most recent exact call."""
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply exact transforms losslessly with per-sample masking.

        For each transform, draws a per-sample boolean mask from the transform's
        ``prob`` probability, applies :meth:`TransformAdapter.exact_apply` only to
        active samples, and scatters the transformed subset back into the batch.
        Auxiliary targets use the same matrix and sampled parameters as exact
        pixels; flips additionally retain their dedicated edge-coordinate rules.

        Args:
            image: Input image batch. Shape: ``(batch_size, channels, height, width)``, dtype: float32.
                Value range and channel convention follow the calling pipeline.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        self._last_matrix = None
        batch_size = image.shape[0]
        device = image.device
        matrix_dtype = _matrix_geometry_dtype(image.dtype)
        acc = torch.eye(3, device=device, dtype=matrix_dtype).expand(batch_size, -1, -1).clone()

        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm, self.randomness)
            if not same_on_batch:
                # Independent Bernoulli draw per sample.
                active = _rand(batch_size, device=device, generator=self.generator) < prob
            else:
                # Single Bernoulli draw shared across the entire batch.
                active_scalar = _rand((), device=device, generator=self.generator) < prob
                active = active_scalar.repeat(batch_size)

            # Skip this transform entirely if no samples are active.
            if not bool(active.any().item()):
                continue

            height, width = image.shape[-2:]
            active_idx = active.nonzero(as_tuple=True)[0]
            active_shape = (len(active_idx), *image.shape[1:])
            params = _sample_transform_params(
                self.adapter,
                tfm,
                active_shape,
                device,
                self.randomness,
                generator=self.generator,
            )
            # ExactAffineSegment owns the shape-changing discrete path. The adapter
            # remains conservative for generic mixed affine fusion, while this
            # segment verifies active/inactive compatibility before scattering.
            params["_exact_allow_shape_change"] = torch.tensor(True, device=device)
            active_matrix = self.adapter.build_matrix(tfm, params, height, width).to(device=device, dtype=matrix_dtype)
            matrix = _scatter_active_matrices(
                active_matrix,
                active,
                batch_size,
                device,
                matrix_dtype,
            )
            acc = matmul3x3(matrix, acc)

            # A flip exposes its axes via exact_flip_dims. Non-flip D4 operations
            # instead route coordinates through the matrix built from these same
            # sampled parameters, while mask stacking keeps their pixels aligned.
            try:
                flip_dims: list[int] | None = self.adapter.exact_flip_dims(tfm)
            except (TypeError, NotImplementedError):
                flip_dims = None

            image = self._apply_exact_with_mask(tfm, image, active, aux_targets, flip_dims, params)
            if aux_targets:
                self._route_exact_coord_aux(tfm, flip_dims, active, aux_targets, height, width, matrix)

        self._last_matrix = acc.to(dtype=matrix_dtype).detach().clone()
        _set_current_call_matrix(self._last_matrix)

        if not _has_aux:
            return image
        return image, aux_targets

    def _apply_exact_with_mask(
        self,
        tfm: object,
        image: Tensor,
        active: Tensor,
        aux_targets: dict[str, Tensor],
        flip_dims: list[int] | None,
        params: dict[str, Tensor],
    ) -> Tensor:
        """Apply an exact op to the image, stacking the mask for non-flip ops.

        For a non-flip D4 op (rot90/transpose) with a mask, the mask is concatenated
        onto the image channels for a single :meth:`TransformAdapter.exact_apply` call
        so both share the identical per-sample random draw (e.g. the same ``rot90``
        count ``k``) — a separate call would re-sample and misalign them. Flip ops are
        deterministic per axis and route the mask via ``exact_flip_dims`` afterwards,
        so no stacking is needed. The active subset is scattered back so inactive
        samples are untouched.

        Args:
            tfm: The exact transform to apply.
            image: ``(B, C, H, W)`` input batch.
            active: ``(B,)`` bool mask of samples this transform applies to.
            aux_targets: Aux dict; its ``"mask"`` entry is updated in place for
                non-flip ops.
            flip_dims: Result of ``exact_flip_dims`` (``None`` for non-flip ops).
            params: Canonical parameters already sampled for the active subset.

        Returns:
            The transformed ``(B, C, H, W)`` image.

        """
        mask = aux_targets.get("mask")
        # Only non-flip ops need the mask stacked to share sampling; flips route it
        # afterwards via exact_flip_dims, so the image-only path is kept unchanged.
        stack_mask = mask is not None and flip_dims is None
        num_channels = image.shape[1]
        stack = image
        if mask is not None and stack_mask:
            # Route the mask by where its pixels go, not by its values: two planes encoding each pixel's integer
            # index ride the image through the same single draw, and the mask is gathered from them below. A D4 op
            # only moves pixels, so this is exact for any label in any dtype on any device; stacking the labels
            # themselves rounded those above 2**53, and one float32 index plane rounded indices above 2**24.
            planes = _index_planes(image)
            stack = torch.cat([image.to(planes.dtype), planes], dim=1)

        active_idx = active.nonzero(as_tuple=True)[0]
        if image.shape[0] == 1 or bool(active.all().item()):
            stack_out = (
                self.adapter.exact_apply(tfm, stack)
                if flip_dims is not None
                else self.adapter.exact_apply(tfm, stack, params=params)
            )
        else:
            transformed = (
                self.adapter.exact_apply(tfm, stack[active_idx])
                if flip_dims is not None
                else self.adapter.exact_apply(tfm, stack[active_idx], params=params)
            )
            if transformed.shape[-2:] != stack.shape[-2:]:
                raise ValueError(
                    "ExactAffineSegment cannot mix active and inactive samples for an exact transform "
                    "that changes canvas dimensions. Use same_on_batch=True or a square input."
                )
            stack_out = stack.clone()
            stack_out[active_idx] = transformed

        if not stack_mask or mask is None:
            return stack_out
        planes_out = stack_out[:, num_channels:].round().long()
        index_out = planes_out[:, :1] * _INDEX_PLANE_BASE + planes_out[:, 1:]
        batch, mask_channels = mask.shape[:2]
        gathered = mask.reshape(batch, mask_channels, -1).gather(
            2, index_out.reshape(batch, 1, -1).expand(-1, mask_channels, -1)
        )
        aux_targets["mask"] = gathered.reshape(batch, mask_channels, *index_out.shape[-2:])
        return stack_out[:, :num_channels].to(image.dtype)

    def _route_exact_coord_aux(
        self,
        tfm: object,
        flip_dims: list[int] | None,
        active: Tensor,
        aux_targets: dict[str, Tensor],
        height: int,
        width: int,
        matrix: Tensor,
    ) -> None:
        """Route exact mask and coordinate targets with the matrix used for pixels.

        Flips retain direct pixel-edge AABB and keypoint-pair routing. Non-flip D4
        operations use the same sampled forward matrix as the exact image call.

        Args:
            tfm: The exact transform.
            flip_dims: ``exact_flip_dims`` result, or ``None`` for non-flip ops.
            active: ``(B,)`` bool mask of active samples.
            aux_targets: Aux dict, updated in place.
            height: Image height in pixels.
            width: Image width in pixels.
            matrix: ``(B, 3, 3)`` forward centre-coordinate matrix built from the
                canonical parameters passed to ``exact_apply``.

        """
        if flip_dims is None:
            from fused_transforms.targets import (
                transform_bbox_xywh,
                transform_bbox_xyxy,
                transform_rboxes,
            )

            for key in list(aux_targets.keys()):
                value = aux_targets[key]
                if key == "bbox_xyxy":
                    aux_targets[key] = transform_bbox_xyxy(value, matrix)
                elif key == "bbox_xywh":
                    aux_targets[key] = transform_bbox_xywh(value, matrix)
                elif key == "keypoints":
                    aux_targets[key] = _route_keypoints(value, matrix, self.keypoint_flip_index)
                elif key == "rboxes":
                    aux_targets[key] = transform_rboxes(value, matrix)
            return
        is_hflip = 3 in flip_dims
        is_vflip = 2 in flip_dims
        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                flipped_val = val.flip(dims=flip_dims)
                aux_targets[key] = torch.where(active[:, None, None, None], flipped_val, val)
                continue
            if key == "bbox_xyxy":
                aux_targets[key] = _flip_bbox_xyxy(val, active, is_hflip, is_vflip, height, width)
                continue
            if key == "bbox_xywh":
                xyxy = _xywh_to_xyxy(val)
                xyxy = _flip_bbox_xyxy(xyxy, active, is_hflip, is_vflip, height, width)
                aux_targets[key] = _xyxy_to_xywh(xyxy)
                continue
            if key == "keypoints":
                flipped_points = _flip_keypoints(val, active, is_hflip, is_vflip, height, width)
                # One mirror turns the plane over; two perpendicular mirrors are a half turn,
                # which does not — so the pair swap fires on exactly one of the two flips.
                if self.keypoint_flip_index is not None and (is_hflip != is_vflip):
                    from fused_transforms.targets import permute_keypoint_pairs

                    flipped_points = permute_keypoint_pairs(
                        flipped_points, _flip_index_tensor(self.keypoint_flip_index, val.device), active
                    )
                aux_targets[key] = flipped_points
                continue
            if key == "rboxes":
                aux_targets[key] = _flip_rboxes(val, active, is_hflip, is_vflip, height, width)


class _BaseAffineSegment(GeneratorPicklingMixin, nn.Module):
    """Shared matrix-composition engine for the torch-backed fused segments.

    Holds the single copy of the per-sample matrix accumulation loop and the
    auxiliary-target routing that :class:`FusedAffineSegment` (affine) and
    :class:`ProjectiveSegment` (homography) both use. Subclasses supply the warp
    itself via :meth:`_apply_grid` -- affine grids for the affine segment, a
    perspective grid for the projective one. The composition, float64
    chain-length handling, probability masking, and pixel-matrix contract are
    identical across both, so they live here once.

    Args:
        transforms: List of geometric transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to canonical
            parameters and matrices.
        interpolation: Optional interpolation mode override (``"bilinear"``,
            ``"nearest"``, ``"bicubic"``). Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override (``"zeros"``, ``"border"``,
            ``"reflection"``). Defaults to ``"zeros"`` when ``None``.
        randomness: Batch randomness policy for the fused run.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.
        mask_fill: Scalar value written outside a routed mask's source canvas.
        generator: Caller-owned generator driving parameter sampling and the
            per-transform probability gates, or ``None`` for the global torch stream.
        fill: Validated constant written outside the source canvas, in the image's own
            value range, or ``None`` for plain zero padding. Applies to the image only;
            auxiliary masks keep their zero padding.

    """

    _eye3: Tensor

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_warp: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
        generator: torch.Generator | None = None,
        fill: tuple[float, ...] | None = None,
        keypoint_flip_index: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize the shared matrix-composition state."""
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.randomness = randomness
        self.mask_interpolation = mask_interpolation
        self.mask_fill = mask_fill
        self.generator = generator
        self.fill = fill
        self.keypoint_flip_index = keypoint_flip_index
        self._last_matrix: Tensor | None = None
        # Opt-in torch.compile of the warp core. Enabled only when the flag is set
        # AND the installed torch is new enough; otherwise the eager path runs
        # unchanged. The compiled function is selected per forward call by device
        # (CPU stays eager — documented no-op), so store just the intent here.
        self._compile_warp: bool = compile_warp and _torch_supports_compile()
        self.register_buffer("_eye3", torch.eye(3, dtype=torch.float32))

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The detached, cloned composed matrix, or ``None`` before the first call to :meth:`forward`.

        """
        return self._last_matrix

    def _compose(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Accumulate the per-transform ``(B, 3, 3)`` matrix chain into one matrix.

        Draws each transform's per-sample activation, samples its parameters,
        builds its matrix, masks skipped samples back to identity, and folds the
        chain into a single composed forward matrix. Composition runs in the
        chain-length-independent dtype from :func:`_matrix_compose_dtype` (float64
        for float32 chains, float32 for low-precision image operations and MPS).
        The public matrix retains the default image dtype, except low-precision
        image operations expose their full-precision float32 matrix.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.

        Returns:
            A ``(acc, acc_img)`` tuple: ``acc`` is the composed forward matrix in
            the compose dtype (for inversion and exactness checks), ``acc_img`` is
            the same matrix in the public matrix dtype (for ``_last_matrix`` and
            auxiliary-target routing).

        """
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (batch_size, num_channels, height, width)

        # Compose and invert the multi-transform chain in float64: the (B, 3, 3)
        # matmul cost is negligible next to the H x W warp, and float64 removes the
        # chain-length-dependent drift that float32 accumulation introduces. The
        # default public matrix dtype remains unchanged. Low-precision image
        # operations expose float32, and cast only after normalization at the
        # sampling-grid boundary.
        compose_dtype = _matrix_compose_dtype(dtype, device, len(self.transforms))
        eye = self._eye3.to(device=device, dtype=compose_dtype)
        eye_batch = eye[None].expand(batch_size, -1, -1)
        acc = eye_batch.clone()
        sample_device = self._matrix_sample_device(device)

        # Sample parameters and build each matrix on ``sample_device``, then move the whole
        # stack across in one transfer. Adapters construct their parameter tensors with an
        # explicit ``device=``, so sampling on an accelerator issues a separate host-to-device
        # copy per parameter -- roughly ten per call for a five-transform chain. Profiled on
        # MPS that dominated the call: 40% in ``Tensor.to`` and 26% in ``torch.tensor``,
        # against about 1% in ``affine_grid`` and a ``grid_sample`` that did not reach the top
        # of the profile. The 3x3 matmuls are free either way; only their placement mattered.
        #
        # Every RNG draw stays exactly where it was. The activation draws below still run on
        # the image's device against the same generator, and parameter sampling reads host
        # RNG on every adapter path -- see ``_matrix_sample_device`` for the one exception,
        # which opts back out rather than moving a draw.
        activations: list[Tensor] = []
        sampled: list[Tensor] = []

        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm, self.randomness)
            if same_on_batch:
                active_scalar = _rand((), device=device, generator=self.generator) < prob
                active = active_scalar.repeat(batch_size)
            else:
                active = _rand(batch_size, device=device, generator=self.generator) < prob

            params = _sample_transform_params(
                self.adapter, tfm, input_shape, sample_device, self.randomness, generator=self.generator
            )
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)

            # Expand to batch if adapter returned (1, 3, 3)
            if mtx_i.shape[0] == 1 and batch_size > 1:
                mtx_i = mtx_i.expand(batch_size, -1, -1)

            activations.append(active)
            sampled.append(mtx_i)

        # One transfer for the whole chain, in the compose dtype; the adapter builds float32.
        # An empty chain has nothing to stack -- a segment can legitimately hold no matrix-building
        # transforms (a blur-only run, or a passthrough probed with aux targets) and must compose to
        # the identity rather than raise out of ``torch.stack``.
        if not sampled:
            return acc, acc.to(dtype=_matrix_public_dtype(dtype))
        stacked = torch.stack(sampled).to(device=device, dtype=compose_dtype)

        for index, active in enumerate(activations):
            mtx_i = torch.where(active[:, None, None], stacked[index], eye_batch)
            acc = matmul3x3(mtx_i, acc)

        return acc, acc.to(dtype=_matrix_public_dtype(dtype))

    def _matrix_sample_device(self, device: torch.device) -> torch.device:
        """Return the device to sample parameters and build per-transform matrices on.

        The host, whenever that cannot move an RNG draw. Building on the host lets ``_compose``
        collapse one host-to-device copy per parameter into a single copy for the whole chain,
        which is the bulk of a fused call on an accelerator.

        Kornia's ``RandomRotation90`` is the one opt-out. Its ``build_matrix`` falls back to
        ``torch.randint(..., device=params["_batch_size"].device)`` when ``k90`` is absent from
        the sampled parameters, so the sampling device decides which RNG stream that draw
        consumes. Every other adapter path derives its parameters from host RNG and is
        unaffected by where the resulting tensor is constructed. Rather than reason about when
        the fallback fires, a chain containing that transform keeps sampling on the image's
        device and forgoes the optimisation.

        Args:
            device: The device the image being composed lives on.

        Returns:
            ``device`` itself when it is already the host, or when the chain contains a
            transform whose sampling device would change an RNG draw; the host otherwise.

        """
        if device.type == "cpu":
            return device
        if any(_samples_on_the_matrix_device(tfm) for tfm in self.transforms):
            return device
        return torch.device("cpu")

    def _apply_grid(self, image: Tensor, acc: Tensor) -> tuple[Tensor, Tensor]:
        """Warp ``image`` by the composed matrix and return the warped image and grid.

        Subclass hook: :class:`FusedAffineSegment` builds an ``F.affine_grid`` from
        the inverse of the composed matrix, :class:`ProjectiveSegment` builds a
        perspective grid. Both invert the composed forward matrix, normalize it to
        the ``[-1, 1]`` grid convention, and run one ``F.grid_sample``.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            acc: ``(batch_size, 3, 3)`` composed forward matrix in the compose dtype.

        Returns:
            A ``(warped_image, grid)`` tuple; the grid is reused to warp mask aux targets.

        """
        raise NotImplementedError

    def _select_warp_fn(self, kind: str, device: torch.device) -> WarpFn:
        """Pick the eager or compiled warp core for this call.

        The compiled core is used only when ``compile=True`` was requested (and
        torch is new enough) AND the tensor is not on CPU. CPU stays eager: the
        inductor CPU backend gives no speedup for this grid_sample workload, so
        compiling it there would only add warm-up cost — a documented no-op. The
        masking / active-subset selection lives entirely in the callers, so the
        selected core sees only dense tensors and never breaks its graph.

        Args:
            kind: ``"affine"`` or ``"perspective"``.
            device: Device of the image being warped.

        Returns:
            The compiled warp core when eligible, else the eager one.

        """
        eager: WarpFn = _grid_sample_affine_batched if kind == "affine" else _grid_sample_perspective_batched
        if self._compile_warp and device.type != "cpu":
            return _compiled_warp_fn(kind)
        return eager

    @staticmethod
    def _route_grid_aux(
        aux_targets: dict[str, Tensor],
        grid: Tensor,
        acc_img: Tensor,
        mask_interpolation: MaskInterpolationStr = "nearest",
        keypoint_flip_index: tuple[int, ...] | None = None,
        mask_fill: MaskFillValue = 0,
    ) -> None:
        """Route auxiliary targets through the warp grid and composed pixel matrix.

        The mask is resampled with the same ``grid`` used for the image; boxes and
        keypoints go through the composed forward pixel matrix ``acc_img``. Mutates
        ``aux_targets`` in place. Shared by the affine and projective forward paths.

        Args:
            aux_targets: Auxiliary targets to transform (``"mask"``, ``"bbox_xyxy"``,
                ``"bbox_xywh"``, ``"keypoints"``).
            grid: The sampling grid produced by :meth:`_apply_grid`.
            acc_img: ``(batch_size, 3, 3)`` composed forward pixel matrix; recast to the geometry dtype here.
            mask_interpolation: Mask sampling mode for the ``"mask"`` target.
            keypoint_flip_index: Optional caller-supplied keypoint pair permutation, applied
                where the composed matrix reverses orientation.
            mask_fill: Scalar border value for the ``"mask"`` target.

        """
        from fused_transforms.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_mask,
            transform_rboxes,
        )

        acc_img = acc_img.to(dtype=_matrix_geometry_dtype(acc_img.dtype))

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                aux_targets[key] = transform_mask(val, grid, mode=mask_interpolation, fill=mask_fill)
                continue
            if key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, acc_img)
                continue
            if key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, acc_img)
                continue
            if key == "keypoints":
                aux_targets[key] = _route_keypoints(val, acc_img, keypoint_flip_index)
                continue
            if key == "rboxes":
                aux_targets[key] = transform_rboxes(val, acc_img)


class FusedAffineSegment(_BaseAffineSegment):
    """Fused affine segment that composes geometric transforms into one grid_sample call.

    Accumulates per-sample ``(B, 3, 3)`` forward affine matrices for every transform in the segment, inverts the
    composed matrix once, and applies a single ``grid_sample`` warp. All operations are vectorised over the batch
    dimension -- no Python loop per sample.

    Args:
        transforms: List of geometric transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to canonical parameters and matrices.
        interpolation: Optional interpolation mode override (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults
            to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to
            ``"zeros"`` when ``None``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.
        render_overridden: Whether the caller overrode how warps render (``fill``, ``interpolation`` or
            ``padding_mode``). When ``True``, a one-transform segment skips its native fast path, which would render
            with the transform's own settings, and warps through the matrix path that honours the override.
            ``None`` (default) derives it from this segment's own ``fill``/``interpolation``/``padding_mode``
            arguments; :func:`build_segments` passes it explicitly because under ``padding_mode="per_transform"``
            the segment's ``padding_mode`` is the transform's own mode, not an override.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_warp: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
        generator: torch.Generator | None = None,
        fill: tuple[float, ...] | None = None,
        keypoint_flip_index: tuple[int, ...] | None = None,
        render_overridden: bool | None = None,
    ) -> None:
        """Initialize ``FusedAffineSegment``."""
        super().__init__(
            transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            compile_warp=compile_warp,
            generator=generator,
            fill=fill,
            keypoint_flip_index=keypoint_flip_index,
        )
        # Whether the caller overrode how warps render (fill / interpolation / border). Resolved by the caller
        # because only it knows: under ``padding_mode="per_transform"`` the segment's own ``padding_mode`` is the
        # transform's mode, not an override. Standalone construction treats every explicit argument as one.
        self._render_overridden: bool = (
            render_overridden
            if render_overridden is not None
            else fill is not None or interpolation is not None or padding_mode is not None
        )
        # Pre-compute fast-path selector once at construction to avoid repeated
        # isinstance checks on every forward call.
        # Non-importing checks: a Kornia segment must not import TorchVision (or vice versa) to classify its adapter.
        self._fast_path: str | None = None
        if is_builtin_adapter(adapter, Backend.KORNIA):
            self._fast_path = "kornia"
        elif is_builtin_adapter(adapter, Backend.TORCHVISION):
            self._fast_path = "torchvision"
        # Single-transform fast paths still reconstruct _last_matrix because
        # Compose.transform_matrix is a public API used for coordinate warping.
        self._skip_matrix_recon: bool = False
        # cv2 warp fast path: for B=1 CPU multi-transform segments, cv2.warpAffine is
        # ~2x faster than PyTorch's affine_grid + grid_sample because it avoids
        # the grid construction overhead entirely.
        self._cv2_warp: bool = len(transforms) > 1 and _cv2_module() is not None
        # Pre-compute cv2 flags once (used by the cv2 fast path every call).
        _interp_str = self.interpolation or "bilinear"
        _pad_str = self.padding_mode or "zeros"
        self._cv2_interp_flag: int = _CV2_INTERP.get(_interp_str, _CV2_INTERP.get("bilinear", 1))
        self._cv2_border_flag: int = _CV2_BORDER.get(_pad_str, _CV2_BORDER.get("zeros", 0))
        # Numpy-native matrix builder for the cv2 warp path: builds a (3,3)
        # float64 matrix directly in numpy, avoiding the torch tensor
        # allocations in build_matrix that are immediately converted back to
        # numpy.  Resolved once at construction from self._fast_path.
        self._np_matrix_builder = None
        # Fused sample+build builder: calls generate_parameters directly and
        # skips the canonical param dict entirely, saving ~3-6 torch tensor
        # allocations per transform per forward call.  Only available for the
        # Kornia cv2 path.  Returns None for unsupported types (caller falls
        # back to the two-step path).
        self._np_fused_builder = None
        if self._cv2_warp and self._fast_path == "kornia":
            try:
                from fused_transforms.adapters.kornia import (
                    build_matrix_numpy_b1_kornia,
                    sample_and_build_matrix_numpy_b1_kornia,
                )

                self._np_matrix_builder = build_matrix_numpy_b1_kornia
                self._np_fused_builder = sample_and_build_matrix_numpy_b1_kornia
            except ImportError:
                pass
        elif self._cv2_warp and self._fast_path == "torchvision":
            try:
                from fused_transforms.adapters.torchvision import (
                    build_matrix_numpy_b1_tv,
                    sample_and_build_matrix_numpy_b1_tv,
                )

                self._np_matrix_builder = build_matrix_numpy_b1_tv
                self._np_fused_builder = sample_and_build_matrix_numpy_b1_tv  # type: ignore[assignment]
            except ImportError:
                pass
        # Pre-allocated (1, 3, 3) float32 buffer for cv2 path _last_matrix writes.
        # Avoids per-call torch.as_tensor + unsqueeze + clone (~3-5 us).
        self._cv2_last_mat_buf: Tensor = torch.empty((1, 3, 3), dtype=torch.float32)
        # Pre-cached B=1 CPU float32 identity for single-transform fast-path _last_matrix
        # writes.  Avoids per-call torch.eye + unsqueeze + expand + clone (~4-6 us)
        # when device and dtype match.
        self._eye_1x3x3_f32: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled segment, defaulting ``_render_overridden`` for one written before the flag existed.

        Such a segment gets ``False``: its fast path ran unconditionally then, so it keeps rendering exactly as it did.
        Deriving the flag from its own ``fill``/``interpolation``/``padding_mode`` would not — under
        ``padding_mode="per_transform"`` the segment stores the transform's own mode, which is no override, and the
        guess moved such a pickle onto the matrix path. ``_render_overridden_derived`` marks the default, so an
        enclosing ``FusedCompose`` replaces it with the Compose-level answer that only it knows.

        Args:
            state: The pickled ``__dict__``.

        """
        super().__setstate__(state)
        if "_render_overridden" not in state:
            # The fast path was unconditional before the flag, so a restored pickle keeps rendering as it did.
            self._render_overridden = False
            self._render_overridden_derived = True

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused affine transform chain via a single grid_sample call.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (batch_size, num_channels, height, width)

        # ------------------------------------------------------------------ #
        # Single-operation fast path: skip matrix pipeline + grid_sample entirely.   #
        # Call the native adapter transform and reconstruct _last_matrix from  #
        # the sampled params.  Only safe when aux_targets is None (no grid    #
        # needed for coord transforms).                                        #
        # ------------------------------------------------------------------ #
        if (
            len(self.transforms) == 1
            and not _has_aux
            and self._fast_path is not None
            and self.randomness is RandomnessPolicy.BACKEND
            and image.dtype not in (torch.float16, torch.bfloat16)
            # The native call renders with the transform's own fill, border and interpolation, so a
            # caller-level override sends the op down the matrix path, which honours it.
            and not self._render_overridden
        ):
            _tfm = self.transforms[0]

            if self._fast_path == "kornia":
                from fused_transforms.adapters.kornia import KorniaAdapter

                # After call_nonfused, Kornia stores sampled params in tfm._params.
                # convert_native_params reads those to build a consistent matrix.
                image = KorniaAdapter.call_nonfused(_tfm, image)

                # Early escape: if all samples were skipped (batch_prob all False),
                # OR if this transform type has an expensive build_matrix with no
                # test requiring its _last_matrix value — use identity and return.
                _bp_raw = getattr(_tfm, "_params", {}).get("batch_prob")
                _all_skipped = _bp_raw is not None and not _bp_raw.to(device=device).bool().any()
                if _all_skipped or self._skip_matrix_recon:
                    _mtx_eye = self._eye_1x3x3_f32
                    _last_matrix = (
                        _mtx_eye
                        if (batch_size == 1 and dtype == _mtx_eye.dtype and device == _mtx_eye.device)
                        else _mtx_eye.expand(batch_size, -1, -1).detach().clone().to(device=device, dtype=dtype)
                    )
                    self._last_matrix = _last_matrix
                    _set_current_call_matrix(_last_matrix.detach().clone())
                    return image

                _native_p = KorniaAdapter.convert_native_params(_tfm, device)
                if _native_p:
                    _mtx = self.adapter.build_matrix(_tfm, _native_p, height, width)
                    if _mtx.shape[0] == 0:
                        # All samples skipped (prob=0.0) — identity for the whole batch.
                        _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        _mtx = _mtx.to(device=device, dtype=dtype)
                        # Kornia stores params for the ACTIVE subset only, so at
                        # batch>1 with prob<1 the matrix has shape (n_active, 3, 3).
                        # Scatter into a full-batch identity keyed by batch_prob so
                        # skipped samples stay identity and shapes never mismatch.
                        _active = None
                        if _bp_raw is not None:
                            _active = _bp_raw.to(device=device).bool()
                            if _active.shape[0] == 1 and batch_size > 1:
                                _active = _active.expand(batch_size)
                        _mtx = _scatter_active_matrices(_mtx, _active, batch_size, device, dtype)
                else:
                    _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                _last_matrix = _mtx.detach().clone()
                self._last_matrix = _last_matrix
                _set_current_call_matrix(_last_matrix)
                return image

            if self._fast_path == "torchvision":
                from fused_transforms.adapters.torchvision import (
                    TorchVisionAdapter,
                    is_torchvision_v2_transform,
                )

                if is_torchvision_v2_transform(_tfm):
                    if self._skip_matrix_recon:
                        image = TorchVisionAdapter.call_nonfused(_tfm, image)
                        _mtx_eye = self._eye_1x3x3_f32
                        if batch_size == 1 and dtype == _mtx_eye.dtype and device == _mtx_eye.device:
                            _last_matrix = _mtx_eye
                        else:
                            _last_matrix = (
                                _mtx_eye.expand(batch_size, -1, -1).detach().clone().to(device=device, dtype=dtype)
                            )
                        self._last_matrix = _last_matrix
                        _set_current_call_matrix(_last_matrix.detach().clone())
                        return image
                    # TV v2 GEOMETRIC_INTERP transforms (RandomRotation, RandomAffine)
                    # always apply and make exactly one RNG call (get_params) - same as
                    # our sample_params.  Save/restore RNG state so both draws use the
                    # same seed.  Restricted to v2: v1 transforms use a different
                    # pixel-center convention that does not match our grid_sample output,
                    # breaking parity tests.
                    _rng = torch.get_rng_state()
                    image = TorchVisionAdapter.call_nonfused(_tfm, image)
                    torch.set_rng_state(_rng)
                    _params = self.adapter.sample_params(_tfm, input_shape, device)
                    if _params:
                        _mtx = self.adapter.build_matrix(_tfm, _params, height, width)
                        if _mtx.shape[0] == 1 and batch_size > 1:
                            _mtx = _mtx.expand(batch_size, -1, -1)
                    else:
                        _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                    _last_matrix = _mtx.to(device=device, dtype=dtype).detach().clone()
                    self._last_matrix = _last_matrix
                    _set_current_call_matrix(_last_matrix)
                    return image

        # ------------------------------------------------------------------ #
        # B=1 CPU cv2 fast path: cv2.warpAffine is ~2x faster than PyTorch's  #
        # affine_grid + grid_sample for single-image CPU tensors because it    #
        # avoids the H*W grid construction overhead entirely.  Compose the     #
        # matrix using the same sample_params / build_matrix loop but replace    #
        # the PyTorch warp backend with cv2.  Only activates when:             #
        # - B=1, CPU, no CUDA, no aux_targets, cv2 available, >1 transform    #
        # ------------------------------------------------------------------ #
        # Gate on CPU tensors only: the cv2 warp round-trips through NumPy
        # (image[0]...numpy()), which raises on any non-CPU device (CUDA and MPS).
        # Gradient-bearing inputs stay on the differentiable torch path; NumPy cannot preserve autograd.
        if (
            self._cv2_warp
            and batch_size == 1
            and not _has_aux
            and image.device.type == "cpu"
            and not image.requires_grad
        ):
            acc_np = np.eye(3, dtype=np.float64)
            # Select the numpy-native matrix builder when available (avoids
            # creating intermediate torch tensors that are immediately converted
            # back to numpy).
            _np_builder = self._np_matrix_builder
            # Fused sample+build: calls generate_parameters directly and builds
            # the matrix in one numpy-native call, avoiding ~15-25us of
            # adapter.sample_params overhead per active transform. Falls back to
            # two-step path when the transform type is not handled (returns None).
            _np_fused = self._np_fused_builder
            for tfm in self.transforms:
                prob = _transform_prob(tfm)
                # Draw the activation gate from the torch RNG, unconditionally, so
                # it responds to torch.manual_seed (np.random was uncontrollable
                # from the torch seed). NOTE: full cross-backend param-draw parity
                # does NOT hold for prob<1 chains — the torch path below samples
                # params for inactive transforms (vectorized, masked via
                # torch.where) while this path skips them, so RNG stream positions
                # diverge after the first inactive transform.
                active = bool((_rand((), device=torch.device("cpu"), generator=self.generator) < prob).item())
                if not active:
                    continue
                if _np_fused is not None:
                    mtx_np = _np_fused(tfm, input_shape, height, width)
                    if mtx_np is not None:
                        acc_np = mtx_np @ acc_np
                        continue
                params = _sample_transform_params(
                    self.adapter, tfm, input_shape, device, self.randomness, generator=self.generator
                )
                if _np_builder is not None:
                    mtx_np = _np_builder(tfm, params, height, width)
                    acc_np = mtx_np @ acc_np
                else:
                    mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                    acc_np = mtx_i[0].double().cpu().numpy() @ acc_np

            np.copyto(self._cv2_last_mat_buf[0].numpy(), acc_np, casting="unsafe")
            # Clone: _cv2_last_mat_buf is a reused buffer overwritten in place on the next
            # call, so a bare reference would let call N+1 mutate call N's returned matrix.
            # Honors the last_matrix "detached, cloned" contract.
            self._last_matrix = self._cv2_last_mat_buf.clone()
            _set_current_call_matrix(torch.as_tensor(acc_np, dtype=dtype).unsqueeze(0).detach().clone())

            # Symbolic-exactness fast path on the cv2 B=1 branch: a chain that composes
            # to a D4 element is applied losslessly via flip/rot90, skipping the cv2 warp
            # entirely (zero interpolation). No aux here (gated above).
            d4_op = classify_d4_batch(self._cv2_last_mat_buf, height, width)
            if d4_op is not None:
                return apply_d4_image(image, d4_op).to(device=device, dtype=dtype)

            m_inv_np = _inv3x3_affine_np(acc_np)
            img_np = image[0].detach().permute(1, 2, 0).contiguous().numpy()  # detach: no grad through cv2 segments
            if num_channels == 1:
                warped = _warp(
                    img_np[:, :, 0], m_inv_np, width, height, self._cv2_interp_flag, self._cv2_border_flag, self.fill
                )
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, m_inv_np, width, height, self._cv2_interp_flag, self._cv2_border_flag, self.fill)
            image = torch.from_numpy(warped).permute(2, 0, 1).unsqueeze(0)
            return image.to(device=device, dtype=dtype)

        # Compose the multi-transform affine chain into one matrix (shared engine).
        acc, acc_img = self._compose(image)
        # One shared detached clone for both the last_matrix property and the per-call
        # context matrix (return_matrix path); compose.py only reads them, never mutates.
        matrix_copy = acc_img.detach().clone()
        self._last_matrix = matrix_copy
        _set_current_call_matrix(matrix_copy)

        # Symbolic-exactness fast path: when the whole batch composes to one D4-group
        # element (flip / 90-degree rotation, zero net translation beyond the
        # border-preserving form), apply it losslessly via flip/rot90 instead of
        # grid_sample -- zero interpolation error. Non-D4 chains fall through unchanged.
        # CPU-only: classify_d4_batch reads the matrix to host (``.item()``), which
        # drains the accelerator stream every warp. Off-CPU the lossless shortcut is
        # skipped -- a grid_sample of an exact D4 map at align_corners=True is
        # near-bit-identical, so only a micro-optimisation (not correctness) is lost.
        d4_op = classify_d4_batch(acc, height, width) if acc.device.type == "cpu" else None
        if d4_op is not None:
            image = apply_d4_image(image, d4_op)
            if aux_targets:
                self._route_d4_aux(aux_targets, d4_op, acc_img, self.keypoint_flip_index)
            if not _has_aux:
                return image
            return image, aux_targets

        image, grid = self._apply_grid(image, acc)

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            self._route_grid_aux(
                aux_targets,
                grid,
                acc_img,
                self.mask_interpolation,
                self.keypoint_flip_index,
                self.mask_fill,
            )

        if not _has_aux:
            return image
        return image, aux_targets

    def _apply_grid(self, image: Tensor, acc: Tensor) -> tuple[Tensor, Tensor]:
        """Warp ``image`` with a single ``F.affine_grid`` + ``grid_sample`` pass.

        Inverts the composed forward matrix, normalizes it to the ``[-1, 1]`` grid
        convention, builds an affine grid, and resamples. The grid is returned so
        the caller can reuse it to warp mask auxiliary targets.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            acc: ``(batch_size, 3, 3)`` composed forward matrix in the compose dtype.

        Returns:
            A ``(warped_image, grid)`` tuple.

        """
        warp_fn = self._select_warp_fn("affine", image.device)
        return warp_fn(
            image,
            acc,
            self.interpolation or "bilinear",
            self.padding_mode or "zeros",
            _fill_tensor(self.fill, image.shape[1], image.device, image.dtype),
        )

    @staticmethod
    def _route_d4_aux(
        aux_targets: dict[str, Tensor],
        d4_op: str,
        acc_img: Tensor,
        keypoint_flip_index: tuple[int, ...] | None = None,
    ) -> None:
        """Route aux targets through an exact D4 op with zero interpolation.

        The mask is transformed by the same lossless ``flip``/``rot90`` op applied to
        the image (no nearest-resample); boxes and keypoints go through the exact
        composed forward pixel matrix ``acc_img`` (integer-valued for a D4 chain), the
        same convention as the interpolating path. Mutates ``aux_targets`` in place.

        Args:
            aux_targets: Auxiliary targets to transform (``"mask"``, ``"bbox_xyxy"``,
                ``"bbox_xywh"``, ``"keypoints"``).
            d4_op: The D4 op name from :func:`classify_d4_batch`.
            acc_img: ``(B, 3, 3)`` composed forward pixel matrix; recast to the geometry dtype here.
            keypoint_flip_index: Optional caller-supplied keypoint pair permutation, applied
                where the composed matrix reverses orientation.

        """
        from fused_transforms.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_rboxes,
        )

        acc_img = acc_img.to(dtype=_matrix_geometry_dtype(acc_img.dtype))

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                aux_targets[key] = apply_d4_image(val, d4_op)
                continue
            if key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, acc_img)
                continue
            if key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, acc_img)
                continue
            if key == "keypoints":
                aux_targets[key] = _route_keypoints(val, acc_img, keypoint_flip_index)
                continue
            if key == "rboxes":
                aux_targets[key] = transform_rboxes(val, acc_img)


class _FusedGeoCropSegment(FusedAffineSegment):
    """Fuse a preceding geometric run and a ``CROP_RESIZE_FIXED`` op into one warp.

    A ``RandomResizedCrop`` immediately after a fusible geometric run
    (``GEOMETRIC_INTERP``/``GEOMETRIC_EXACT``) is normally a hard segment
    boundary — the geo run does one ``grid_sample`` at input size, then
    :class:`CropResizeSegment` does a second one at target size. This segment
    composes both into a single matrix ``M_crop @ M_geo`` and applies exactly one
    ``grid_sample`` at the crop's ``(target_h, target_w)`` output size, saving an
    interpolation pass and improving precision (one resample instead of two).

    The crop reads the geo chain's output, so the forward composite is
    ``M_crop @ M_geo`` (geo applied first, crop after). Like
    :class:`CropResizeSegment`, per-sample probability is *not* applied to the
    crop (shape-changing ops must produce a uniform output size); the geo run's
    per-sample ``prob`` gates are honoured exactly as in :class:`FusedAffineSegment`.

    ``transforms`` is ``[*geo_transforms, crop_transform]`` so the inherited
    fusion-plan machinery (``n_warps_saved``, ``fusion_plan``, ``transform_matrix``)
    counts the crop as one more fused op and exposes the full geo∘crop matrix.

    Args:
        geo_transforms: The preceding fusible geometric transforms, in order.
        crop_transform: A single ``CROP_RESIZE_FIXED`` transform.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_matrix``.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.
        randomness: Batch randomness policy for the fused geometric run.

    Examples:
        ```pycon
        >>> import torch
        >>> import kornia.augmentation as K
        >>> from fused_transforms.affine.segment import _FusedGeoCropSegment
        >>> from fused_transforms.adapters.kornia import KorniaAdapter
        >>> geo = K.RandomHorizontalFlip(p=1.0)
        >>> crop = K.RandomResizedCrop((8, 8), scale=(0.5, 0.5), ratio=(1.0, 1.0))
        >>> seg = _FusedGeoCropSegment([geo], crop, KorniaAdapter())
        >>> seg(torch.zeros(1, 3, 16, 16)).shape
        torch.Size([1, 3, 8, 8])

        ```

    """

    def __init__(
        self,
        geo_transforms: list[object],
        crop_transform: object,
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_warp: bool = False,
        antialias: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
        generator: torch.Generator | None = None,
        fill: tuple[float, ...] | None = None,
        keypoint_flip_index: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize ``_FusedGeoCropSegment``."""
        # nn.Module state only; skip FusedAffineSegment.__init__'s cv2/numpy
        # fast-path wiring (it keys on len(transforms) and would try to resolve a
        # numpy builder for the crop transform — this segment overrides forward).
        nn.Module.__init__(self)
        self.register_buffer("_eye3", torch.eye(3, dtype=torch.float32))
        self.geo_transforms = geo_transforms
        self.crop_transform = crop_transform
        # transforms holds the full fused run so inherited fusion-plan machinery
        # (n_warps_saved = n-1, fusion_plan naming) counts the crop as fused.
        self.transforms: list[object] = [*geo_transforms, crop_transform]
        self.generator = generator
        self.fill = fill
        self.keypoint_flip_index = keypoint_flip_index
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.randomness = randomness
        self.mask_interpolation = mask_interpolation
        self.mask_fill = mask_fill
        self._last_matrix: Tensor | None = None
        # `compile_warp` is accepted for build_segments API symmetry but deliberately not
        # stored: forward() builds its own affine_grid/grid_sample and never routes through
        # _select_warp_fn (the sole consumer of _compile_warp), so a store would be dead.
        # Crop-resize segments do not honor torch.compile of the warp core.
        self._antialias: bool = antialias

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused geo∘crop chain via one ``grid_sample`` at the target size.

        Args:
            image: ``(batch_size, channels, height_in, width_in)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). Masks are warped with the output grid; boxes and
                keypoints via the composed forward matrix.

        Returns:
            ``(batch_size, channels, height_out, width_out)`` tensor when
            ``aux_targets`` is ``None``; ``(tensor, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (batch_size, num_channels, height, width)

        # Accumulate the geometric run exactly as FusedAffineSegment does, but the
        # chain always has >1 fused op (geo + crop), so compose in float64 where
        # available. The crop matrix has no per-sample prob gate.
        compose_dtype = _matrix_compose_dtype(dtype, device, len(self.transforms))
        eye = self._eye3.to(device=device, dtype=compose_dtype)
        eye_batch = eye[None].expand(batch_size, -1, -1)
        acc = eye_batch.clone()

        for tfm in self.geo_transforms:
            prob = _transform_prob(tfm)
            if _shares_randomness_across_batch(self.adapter, tfm, self.randomness):
                active = (_rand((), device=device, generator=self.generator) < prob).repeat(batch_size)
            else:
                active = _rand(batch_size, device=device, generator=self.generator) < prob
            params = _sample_transform_params(
                self.adapter, tfm, input_shape, device, self.randomness, generator=self.generator
            )
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)
            if mtx_i.shape[0] == 1 and batch_size > 1:
                mtx_i = mtx_i.expand(batch_size, -1, -1)
            mtx_i = mtx_i.to(device=device, dtype=compose_dtype)
            mtx_i = torch.where(active[:, None, None], mtx_i, eye_batch)
            acc = matmul3x3(mtx_i, acc)

        target_h, target_w, mtx_crop = self._build_crop_matrix(input_shape, device, compose_dtype)
        acc_full = matmul3x3(mtx_crop, acc)  # crop reads geo's output

        self._last_matrix = acc_full.to(dtype=_matrix_public_dtype(dtype)).detach().clone()
        _set_current_call_matrix(self._last_matrix)

        # The prefilter computes per-sample scales once and only touches aggressive
        # rows; the default path remains the original one-warp execution.
        antialias_mtx = acc_full.to(dtype=dtype)
        if self._antialias:
            image = _maybe_antialias_prefilter(image, antialias_mtx, enabled=True)

        mtx_inv = inv3x3(acc_full)
        mtx_norm = normalize_matrix_io(mtx_inv, height, width, target_h, target_w).to(dtype=dtype)
        grid = F.affine_grid(
            mtx_norm[:, :2, :],
            [batch_size, num_channels, target_h, target_w],
            align_corners=True,
        )
        fill_value = _fill_tensor(self.fill, num_channels, device, dtype)
        out = F.grid_sample(
            image if fill_value is None else image - fill_value,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )
        if fill_value is not None:
            out = out + fill_value

        if aux_targets:
            self._warp_aux(
                aux_targets,
                grid,
                acc_full.to(dtype=_matrix_public_dtype(dtype)),
                self.mask_interpolation,
                self.keypoint_flip_index,
                self.mask_fill,
            )

        if not _has_aux:
            return out
        if aux_targets is None:
            raise RuntimeError("internal error: aux_targets is None in return branch")
        return out, aux_targets

    def _build_crop_matrix(
        self,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
        compose_dtype: torch.dtype,
    ) -> tuple[int, int, Tensor]:
        """Sample the crop and return ``(target_h, target_w, (B, 3, 3) crop matrix)``."""
        batch_size, _, height, width = input_shape
        params = _sample_transform_params(
            self.adapter, self.crop_transform, input_shape, device, self.randomness, generator=self.generator
        )
        if not (
            torch.all(params["target_h"] == params["target_h"][0])
            and torch.all(params["target_w"] == params["target_w"][0])
        ):
            raise ValueError(
                "_FusedGeoCropSegment requires a uniform target size across the batch "
                f"(got target_h={params['target_h'].tolist()}, target_w={params['target_w'].tolist()})"
            )
        target_h = int(params["target_h"][0].item())
        target_w = int(params["target_w"][0].item())
        mtx_crop = self.adapter.build_matrix(self.crop_transform, params, height, width)
        if mtx_crop.shape[0] == 1 and batch_size > 1:
            mtx_crop = mtx_crop.expand(batch_size, -1, -1)
        return target_h, target_w, mtx_crop.to(device=device, dtype=compose_dtype)

    @staticmethod
    def _warp_aux(
        aux_targets: dict[str, Tensor],
        grid: Tensor,
        mtx: Tensor,
        mask_interpolation: MaskInterpolationStr = "nearest",
        keypoint_flip_index: tuple[int, ...] | None = None,
        mask_fill: MaskFillValue = 0,
    ) -> None:
        """Warp auxiliary targets in place: mask via the output grid, coords via ``mtx``."""
        from fused_transforms.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_mask,
            transform_rboxes,
        )

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                aux_targets[key] = transform_mask(val, grid, mode=mask_interpolation, fill=mask_fill)
            elif key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, mtx)
            elif key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, mtx)
            elif key == "keypoints":
                aux_targets[key] = _route_keypoints(val, mtx, keypoint_flip_index)
            elif key == "rboxes":
                aux_targets[key] = transform_rboxes(val, mtx)


class FusedGaussianBlurSegment(nn.Module):
    """Fold Gaussian blurs and safely move them after one affine warp.

    The segment represents an ordered ``geometric -> blur -> geometric`` stretch.
    It samples the blur once per input item, adds variances across consecutive
    blurs, and only moves the result after the affine warp when the sampled
    matrix has no downscaling direction. Axis-aligned matrices retain the
    native primitive; rotated or sheared matrices use a sampled covariance kernel.

    Args:
        transforms: Original transforms in pipeline order.
        blur_transforms: Consecutive Gaussian blur transforms in the stretch.
        prefix_geometric_transforms: Geometric transforms before the blur run.
        suffix_geometric_transforms: Geometric transforms after the blur run.
        adapter: Backend adapter for sampling affine parameters.
        interpolation: Optional geometric interpolation mode.
        padding_mode: Optional geometric padding mode.
        randomness: Batch randomness policy for sampling.
        mask_interpolation: Sampling mode for routed masks.

    """

    def __init__(
        self,
        transforms: list[object],
        blur_transforms: list[object],
        prefix_geometric_transforms: list[object],
        suffix_geometric_transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
    ) -> None:
        """Initialize a folded Gaussian blur and affine segment."""
        super().__init__()
        self.transforms = transforms
        self.blur_transforms = blur_transforms
        self.prefix_geometric_transforms = prefix_geometric_transforms
        self.suffix_geometric_transforms = suffix_geometric_transforms
        self.geometric_transforms = [*prefix_geometric_transforms, *suffix_geometric_transforms]
        self.adapter = adapter
        self.randomness = randomness
        self._geometric = FusedAffineSegment(
            self.geometric_transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
        )
        self._prefix = FusedAffineSegment(
            prefix_geometric_transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
        )
        self._suffix = FusedAffineSegment(
            suffix_geometric_transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
        )
        self._last_matrix: Tensor | None = None

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the composed affine matrix from the most recent forward call."""
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply folded blur transforms and their adjoining geometric run.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            aux_targets: Optional mask, box, or keypoint targets for the affine warp.

        Returns:
            The transformed image, with routed auxiliary targets when supplied.

        """
        if not self.geometric_transforms:
            sigma = _sample_folded_gaussian_sigma(self.blur_transforms, image, self.randomness)
            blurred = _apply_folded_gaussian(image, sigma)
            if aux_targets is None:
                return blurred
            return blurred, aux_targets

        prefix, _prefix_img = self._prefix._compose(image)
        sigma = _sample_folded_gaussian_sigma(self.blur_transforms, image, self.randomness)
        suffix, _suffix_img = self._suffix._compose(image)
        acc = matmul3x3(suffix, prefix)
        acc_img = acc.to(dtype=_matrix_public_dtype(image.dtype))
        self._last_matrix = acc_img.detach().clone()
        _set_current_call_matrix(self._last_matrix)
        commutable = _is_commutable_gaussian_matrix(suffix)
        if commutable:
            warped, grid = self._geometric._apply_grid(image, acc)
            if aux_targets:
                self._geometric._route_grid_aux(
                    aux_targets,
                    grid,
                    acc_img,
                    self._geometric.mask_interpolation,
                    self._geometric.keypoint_flip_index,
                    self._geometric.mask_fill,
                )
            if _is_axis_aligned_gaussian_matrix(suffix):
                sigma = _transform_gaussian_sigma(sigma, suffix)
                result = _apply_folded_gaussian(warped, sigma)
            else:
                covariance = _transform_gaussian_covariance(sigma, suffix)
                result = _apply_gaussian_covariance(warped, covariance)
        else:
            # Safety fallback for a non-commutable sampled matrix. A configured affine
            # can still sample a downscaling matrix through shear, so apply the true image order
            # suffix(blur(prefix(x))) with separate warps rather than assuming the blur
            # commutes with the prefix. Auxiliary targets transform by the full affine
            # (the blur is a spatial no-op for them), so they still route through ``acc``.
            working = image
            if self.prefix_geometric_transforms:
                working, _ = self._prefix._apply_grid(working, prefix)
            working = _apply_folded_gaussian(working, sigma)
            if self.suffix_geometric_transforms:
                working, _ = self._suffix._apply_grid(working, suffix)
            result = working
            if aux_targets:
                _, aux_grid = self._geometric._apply_grid(image, acc)
                self._geometric._route_grid_aux(
                    aux_targets,
                    aux_grid,
                    acc_img,
                    self._geometric.mask_interpolation,
                    self._geometric.keypoint_flip_index,
                    self._geometric.mask_fill,
                )
        if aux_targets is None:
            return result
        return result, aux_targets

    def forward_numpy(self, image_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply a folded Albumentations Gaussian blur using OpenCV.

        Args:
            image_hwc: Single HWC NumPy image in an Albumentations-native call.

        Returns:
            The folded Gaussian-blur result in the original NumPy layout.

        Raises:
            RuntimeError: If a geometric run or OpenCV is unavailable on this path.

        """
        if self.geometric_transforms or _cv2_module() is None:
            msg = "NumPy Gaussian blur fusion requires OpenCV and no adjoining geometric transforms"
            raise RuntimeError(msg)
        variance = 0.0
        for transform in self.blur_transforms:
            random_source = getattr(transform, "py_random", None)
            sigma_range = getattr(transform, "sigma_limit", None)
            if random_source is None or not isinstance(sigma_range, tuple):
                msg = f"Unsupported NumPy Gaussian blur transform {type(transform).__name__!r}"
                raise TypeError(msg)
            if random_source.random() < _transform_prob(transform):
                sigma = random_source.uniform(*sigma_range)
                variance += sigma * sigma
        if variance == 0.0:
            return image_hwc
        sigma = math.sqrt(variance)
        blurred: NDArray[Any] = _require_cv2().GaussianBlur(
            image_hwc, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=_CV2_BORDER["reflection"]
        )
        return blurred


def _sample_folded_gaussian_sigma(transforms: list[object], image: Tensor, randomness: RandomnessPolicy) -> Tensor:
    """Sample and variance-add scalar Gaussian blur sigmas for a tensor batch."""
    batch_size = image.shape[0]
    variance = torch.zeros((batch_size, 2), device=image.device, dtype=image.dtype)
    for transform in transforms:
        sigma, active = _sample_gaussian_sigma(transform, image, randomness)
        variance += (sigma * active[:, None]).square()
    return variance.sqrt()


def _sample_gaussian_sigma(transform: object, image: Tensor, randomness: RandomnessPolicy) -> tuple[Tensor, Tensor]:
    """Sample one isotropic Gaussian sigma and activation flag per batch item."""
    batch_size, channels, height, width = image.shape
    name = type(transform).__name__
    if name == "RandomGaussianBlur":
        params = transform.generate_parameters(torch.Size((batch_size, channels, height, width)))  # type: ignore[attr-defined]
        raw_sigma = params["sigma"].to(device=image.device, dtype=image.dtype)
    elif name == "GaussianBlur":
        sigma_range = getattr(transform, "sigma", None) or getattr(transform, "sigma_limit", None)
        if not isinstance(sigma_range, tuple):
            msg = "GaussianBlur must expose a two-value sigma range"
            raise TypeError(msg)
        raw_sigma = torch.empty(batch_size, device=image.device, dtype=image.dtype).uniform_(*sigma_range)
    else:
        msg = f"Unsupported spatial-linear transform {name!r}"
        raise TypeError(msg)
    if raw_sigma.ndim == 0:
        raw_sigma = raw_sigma.expand(batch_size)
    if raw_sigma.shape[0] == 1 and batch_size > 1:
        raw_sigma = raw_sigma.expand(batch_size)
    sigma = raw_sigma[:, None].expand(-1, 2) if raw_sigma.ndim == 1 else raw_sigma[:, :2]
    prob = _transform_prob(transform)
    if name == "GaussianBlur" and not hasattr(transform, "p"):
        active = torch.ones(batch_size, device=image.device, dtype=torch.bool)
    elif randomness is not RandomnessPolicy.PER_SAMPLE and bool(getattr(transform, "same_on_batch", False)):
        active = (torch.rand((), device=image.device) < prob).expand(batch_size)
    else:
        active = torch.rand(batch_size, device=image.device) < prob
    return sigma, active


def _apply_folded_gaussian(image: Tensor, sigma: Tensor) -> Tensor:
    """Apply a batched axis-aligned Gaussian primitive when any sigma is positive."""
    blurred = _kornia_gaussian_blur(image, sigma[:, 0], sigma[:, 1])
    return image if blurred is None else blurred


def _is_axis_aligned_gaussian_matrix(matrix: Tensor) -> bool:
    """Return whether an affine matrix preserves diagonal Gaussian covariance."""
    off_diagonal = matrix[:, :2, :2][:, (0, 1), (1, 0)].abs().max()
    return bool(off_diagonal < _AXIS_ALIGNED_EPS)


def _is_commutable_gaussian_matrix(matrix: Tensor) -> bool:
    """Return whether an affine matrix can move a Gaussian blur without aliasing."""
    # A blur only aliases when the warp downscales (smallest singular value < 1). The
    # tolerance keeps a pure rotation (true minimum singular value 1.0, which float32
    # can round to just below) on the commuting path instead of the correct-but-slower
    # fallback; a scale within 1e-6 of unity does not meaningfully alias.
    return bool(estimate_scale(matrix)[0] >= 1.0 - 1e-6)


def _transform_gaussian_sigma(sigma: Tensor, matrix: Tensor) -> Tensor:
    """Transform diagonal blur covariance through an axis-aligned affine matrix."""
    linear = matrix[:, :2, :2].to(dtype=sigma.dtype)
    return torch.stack([linear[:, 0, 0].abs() * sigma[:, 0], linear[:, 1, 1].abs() * sigma[:, 1]], dim=1)


def _transform_gaussian_covariance(sigma: Tensor, matrix: Tensor) -> Tensor:
    """Transform diagonal Gaussian covariance through a full affine matrix."""
    linear = matrix[:, :2, :2].to(dtype=sigma.dtype)
    covariance = torch.diag_embed(sigma.square())
    return linear @ covariance @ linear.transpose(-1, -2)


def _apply_gaussian_covariance(image: Tensor, covariance: Tensor) -> Tensor:
    """Apply a per-sample Gaussian convolution for transformed full covariances."""
    if bool(covariance[:, 0, 1].abs().max() < _AXIS_ALIGNED_EPS):
        sigma = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0.0).sqrt()
        return _apply_folded_gaussian(image, sigma)
    return _sampled_gaussian_convolution(image, covariance)


def _sampled_gaussian_convolution(image: Tensor, covariance: Tensor) -> Tensor:
    """Convolve each image with a sampled, normalized full-covariance Gaussian."""
    work_dtype = torch.float32 if image.dtype in (torch.float16, torch.bfloat16) else image.dtype
    covariance = covariance.to(dtype=work_dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    safe_eigenvalues = eigenvalues.clamp_min(torch.finfo(work_dtype).eps)
    safe_covariance = eigenvectors @ torch.diag_embed(safe_eigenvalues) @ eigenvectors.transpose(-1, -2)
    radius = min(
        math.ceil(3.0 * float(safe_eigenvalues[:, 1].sqrt().max().item())),
        _MAX_SAMPLED_GAUSSIAN_RADIUS,
    )
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=work_dtype)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    positions = torch.stack([grid_x, grid_y], dim=-1)
    inverse = torch.linalg.inv(safe_covariance)
    quadratic = torch.einsum("hwi,bij,hwj->bhw", positions, inverse, positions)
    kernel = torch.exp(-0.5 * quadratic)
    kernel = kernel / kernel.sum(dim=(-1, -2), keepdim=True)
    batch_size, channels, height, width = image.shape
    weights = kernel[:, None].expand(-1, channels, -1, -1).reshape(batch_size * channels, 1, *kernel.shape[-2:])
    padded = F.pad(image.reshape(1, batch_size * channels, height, width), (radius,) * 4, mode="reflect")
    return F.conv2d(padded, weights.to(dtype=image.dtype), groups=batch_size * channels).reshape_as(image)


# ---------------------------------------------------------------------------
# AlbuFusedAffineSegment — cv2 backend for Albumentations
# ---------------------------------------------------------------------------

ImageArray = NDArray[np.integer[Any] | np.floating[Any]]
MatrixArray = NDArray[np.floating[Any]]


def _warp(
    img: ImageArray,
    matrix_dst2src_3x3: MatrixArray,
    width: int,
    height: int,
    interp_flag: int,
    border_flag: int,
    fill: tuple[float, ...] | None = None,
) -> ImageArray:
    """Apply cv2.warpAffine with the dst->src 3x3 pixel-space matrix.

    ``matrix_dst2src_3x3`` maps destination pixel coordinates to source pixel
    coordinates.  ``cv2.WARP_INVERSE_MAP`` (16) is OR-ed into *interp_flag* so
    the matrix is used directly without re-inversion.  cv2 handles all channels
    in a single call, avoiding the per-channel loop previously needed for scipy.

    Args:
        img: HxW or HxWxC float32 numpy array.
        matrix_dst2src_3x3: 3x3 matrix mapping destination pixels to source pixels.
        width: Output width in pixels.
        height: Output height in pixels.
        interp_flag: cv2 interpolation constant (e.g. ``1`` for ``INTER_LINEAR``).
        border_flag: cv2 border mode constant (e.g. ``0`` for ``BORDER_CONSTANT``).
        fill: Optional validated constant border value in the image's own value range;
            ``None`` means an all-zero border.

    Returns:
        Warped image array with the same dtype and channel count as ``img``.

    """
    import cv2

    m_2x3 = matrix_dst2src_3x3[:2, :].astype(np.float64)
    num_channels = 1 if img.ndim == 2 else img.shape[2]
    warp_affine = cast(Any, cv2.warpAffine)
    return cast(
        MatrixArray,
        warp_affine(
            img,
            m_2x3,
            (width, height),
            flags=interp_flag | _CV2_WARP_INVERSE_MAP,
            borderMode=border_flag,
            borderValue=_cv2_border_value(fill, num_channels),
        ),
    )


def _inv3x3_affine_np(mtx: MatrixArray) -> MatrixArray:
    """Closed-form inverse of a 3x3 affine matrix (bottom row = [0, 0, 1]).

    Uses Cramer's rule for the upper-left 2x2 sub-matrix, avoiding LAPACK
    dispatch overhead (~15-20us) of ``np.linalg.inv`` for a single 3x3 matrix.

    Args:
        mtx: A (3, 3) float64 ndarray representing a forward affine transform.
           The bottom row must be ``[0, 0, 1]`` (standard affine convention).

    Returns:
        The (3, 3) inverse affine matrix as a float64 ndarray.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> mtx = np.eye(3, dtype=np.float64)
        >>> _inv3x3_affine_np(mtx)
        array([[ 1., -0.,  0.],
               [-0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        ```

    """
    m00, m01, trans_x = mtx[0, 0], mtx[0, 1], mtx[0, 2]
    m10, m11, trans_y = mtx[1, 0], mtx[1, 1], mtx[1, 2]
    det = m00 * m11 - m01 * m10
    # Match the torch path (matrix.inv3x3), which raises for near-singular
    # matrices at float32 eps — without this guard the division silently
    # produces inf/NaN that propagates into cv2.warpAffine.
    threshold = _singularity_threshold(torch.float32)
    if abs(det) < threshold:
        msg = f"Singular affine matrix cannot be inverted (|det|={abs(det):.3e} < {threshold:.3e})."
        raise ValueError(msg)
    inv_det = 1.0 / det
    return np.array(
        [
            [m11 * inv_det, -m01 * inv_det, (m01 * trans_y - m11 * trans_x) * inv_det],
            [-m10 * inv_det, m00 * inv_det, (m10 * trans_x - m00 * trans_y) * inv_det],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class AlbuFusedAffineSegment(nn.Module):
    """Fused affine segment for the Albumentations cv2 backend.

    Loops over B samples, composes per-sample ``(3, 3)`` forward affine matrices, and applies a single
    ``cv2.warpAffine`` call per sample.

    The input and output are ``(B, C, H, W)`` float32 ``torch.Tensor`` objects. Conversion to/from ``(H, W, C)``
    NumPy arrays happens inside ``forward()``.

    No ``normalize_matrix`` step is needed — ``cv2.warpAffine`` operates in pixel coordinates natively. The
    accumulated forward (src->dst) matrix is inverted once per sample and passed to :func:`_warp` via
    ``cv2.WARP_INVERSE_MAP``.

    Args:
        transforms: List of Albumentations transform objects to fuse.
        adapter: An ``AlbumentationsAdapter`` providing ``sample_params``,
            ``build_matrix``, and category lookup.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults to ``"bilinear"``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to ``"zeros"``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import torch
        >>> from fused_transforms.affine.segment import AlbuFusedAffineSegment
        >>> from fused_transforms.adapters.albumentations import AlbumentationsAdapter
        >>> seg = AlbuFusedAffineSegment([], AlbumentationsAdapter())
        >>> out = seg(torch.zeros(1, 3, 8, 8))
        >>> out.shape
        torch.Size([1, 3, 8, 8])

        ```

    """

    # Pre-classified dispatch tags for forward_numpy fast path.
    _TAG_INTERP: int = 0
    _TAG_HFLIP: int = 1
    _TAG_VFLIP: int = 2
    _TAG_ADAPTER: int = 3  # fallback: use adapter round-trip
    _TAG_FAST_ROTATE: int = 4  # A.Rotate fast path (direct numpy, bypasses albu gpdd)

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        execution: ExecutionStr = "cv2",
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
        fill: tuple[float, ...] | None = None,
        keypoint_flip_index: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize ``AlbuFusedAffineSegment``."""
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation or "bilinear"
        self.padding_mode = padding_mode or "zeros"
        self.execution: ExecutionStr = _validate_execution(execution)
        #: Engine the most recent call actually used. Equal to ``execution`` unless that is
        #: ``"auto"``, in which case it records what the routing rule chose. ``None`` before the
        #: first call, and before any call that warps nothing.
        self._last_execution: ExecutionStr | None = None
        self.mask_interpolation = mask_interpolation
        self.mask_fill = mask_fill
        self.fill = fill
        self.keypoint_flip_index = keypoint_flip_index
        self._last_matrix: Tensor | None = None
        # Pre-compute cv2 flags once instead of dict-lookups per call.
        self._interp_flag: int = _CV2_INTERP.get(self.interpolation, _CV2_INTERP.get("bilinear", 1))
        self._border_flag: int = _CV2_BORDER.get(self.padding_mode, _CV2_BORDER.get("zeros", 0))
        # Pre-classify transforms to avoid per-call _is_albu_instance dispatch.
        self._tfm_tags: list[int] = self._classify_transforms(transforms, adapter)
        # Pre-allocated identity (1,3,3) — reused for zero/single-transform early returns.
        self._identity_1x3x3: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        # Pre-allocated (1,3,3) buffer for forward_numpy last_matrix writes.
        # Avoids per-call tensor allocation from torch.from_numpy(...).unsqueeze(0).
        self._last_matrix_buffer: Tensor = torch.empty((1, 3, 3), dtype=torch.float32)
        self._last_matrix_np_buffer: NDArray[np.float32] = np.empty((3, 3), dtype=np.float32)
        self._last_matrix_np_tensor: Tensor = torch.from_numpy(self._last_matrix_np_buffer)

    @staticmethod
    def _classify_transforms(transforms: list[object], adapter: TransformAdapter) -> list[int]:
        """Classify each transform for dispatch in ``forward_numpy``.

        Returns a list of integer tags (one per transform) enabling O(1) dispatch in the hot loop instead of O(n)
        ``isinstance`` chains.

        """
        tags: list[int] = []
        try:
            from fused_transforms.adapters.albumentations import (
                _HFLIP_TYPES,
                _INTERP_TYPES,
                _VFLIP_TYPES,
                _is_albu_instance,
            )
        except ImportError:
            return [AlbuFusedAffineSegment._TAG_ADAPTER] * len(transforms)

        try:
            from albumentations import Rotate as _AlbuRotate

            _rotate_type: type | None = _AlbuRotate
        except ImportError:
            _rotate_type = None

        for tfm in transforms:
            if _rotate_type is not None and isinstance(tfm, _rotate_type) and not getattr(tfm, "crop_border", True):
                tags.append(AlbuFusedAffineSegment._TAG_FAST_ROTATE)
            elif _is_albu_instance(tfm, _INTERP_TYPES):
                tags.append(AlbuFusedAffineSegment._TAG_INTERP)
            elif _is_albu_instance(tfm, _HFLIP_TYPES):
                tags.append(AlbuFusedAffineSegment._TAG_HFLIP)
            elif _is_albu_instance(tfm, _VFLIP_TYPES):
                tags.append(AlbuFusedAffineSegment._TAG_VFLIP)
            else:
                tags.append(AlbuFusedAffineSegment._TAG_ADAPTER)
        return tags

    @staticmethod
    def _sample_matrix_numpy(
        adapter: TransformAdapter,
        transform: object,
        tag: int,
        channels: int,
        height: int,
        width: int,
        *,
        tensor_roundtrip: bool,
    ) -> MatrixArray:
        """Sample one Albumentations matrix through the native preparation path.

        ``forward_numpy`` keeps raw float64 matrices. Tensor callers retain the
        historical adapter conversion through float32 before float64 composition,
        so this optimization removes wrapper work without changing their sampled
        geometry or established warp numerics.

        Args:
            adapter: Backend bridge used only for transforms without a native
                Albumentations preparation path.
            transform: Active Albumentations transform.
            tag: Pre-classified native preparation tag.
            channels: Input channel count for the fallback adapter route.
            height: Input canvas height.
            width: Input canvas width.
            tensor_roundtrip: Whether to reproduce the tensor adapter's float32
                matrix conversion before returning the float64 accumulator value.

        Returns:
            One ``(3, 3)`` float64 forward matrix.

        """
        from fused_transforms.adapters.albumentations import (
            _sample_matrices,
            hflip_matrix_np,
            vflip_matrix_np,
        )

        if tag == AlbuFusedAffineSegment._TAG_FAST_ROTATE:
            angle = transform.py_random.uniform(*transform.limit)  # type: ignore[attr-defined]
            radians = math.radians(angle)
            cos_angle, sin_angle = math.cos(radians), math.sin(radians)
            center_x = width / 2.0 - 0.5
            center_y = height / 2.0 - 0.5
            matrix = np.array(
                [
                    [cos_angle, sin_angle, center_x * (1.0 - cos_angle) - center_y * sin_angle],
                    [-sin_angle, cos_angle, center_y * (1.0 - cos_angle) + center_x * sin_angle],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        elif tag == AlbuFusedAffineSegment._TAG_INTERP:
            matrix = _sample_matrices(transform, 1, height, width)[0]
        elif tag == AlbuFusedAffineSegment._TAG_HFLIP:
            matrix = hflip_matrix_np(width=width)
        elif tag == AlbuFusedAffineSegment._TAG_VFLIP:
            matrix = vflip_matrix_np(height=height)
        else:
            params = adapter.sample_params(transform, (1, channels, height, width), torch.device("cpu"))
            matrix = adapter.build_matrix(transform, params, height, width)[0].double().cpu().numpy()

        if tensor_roundtrip:
            return np.asarray(matrix, dtype=np.float32).astype(np.float64)
        return np.asarray(matrix, dtype=np.float64)

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The composed forward matrix (detached clone), or ``None`` before the first call to :meth:`forward`.

        """
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply fused affine chain via per-sample cv2.warpAffine.

        Args:
            image: ``(B, C, H, W)`` float32 input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image. Masks are resampled through a grid built from the same
                composed matrix used for the image; bounding boxes and keypoints
                are transformed through the composed forward pixel matrix. The
                coordinate convention (center/``align_corners=True``) matches the
                torch affine path exactly.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size = image.shape[0]

        # Compose the per-sample forward matrices with Albumentations' own
        # per-sample sampling (numpy RNG stream). This is identical for both
        # execution strategies — only the warp backend below differs — so the
        # sampled geometry is byte-for-byte the same whether the batch is warped
        # by cv2 (default) or a batched grid_sample.
        accs, any_active = self._compose_matrices(image)
        composed_batch = self._stack_matrices(accs)
        # The public matrix keeps the image's own precision (float32/float64); only a
        # low-precision image promotes it to float32, matching the torch affine path.
        public_dtype = _matrix_public_dtype(image.dtype)
        self._last_matrix = composed_batch.to(dtype=public_dtype).clone().detach()
        _set_current_call_matrix(composed_batch.to(device=image.device, dtype=public_dtype).detach().clone())

        if batch_size == 0 or len(self.transforms) == 0:
            return (image, aux_targets) if _has_aux else image

        self._last_execution = _resolve_execution(self.execution, image.device)
        if self._last_execution == "torch":
            image = self._warp_torch(image, composed_batch)
        else:
            image = self._warp_cv2(image, accs, any_active)

        if aux_targets:
            self._route_aux(aux_targets, composed_batch, image)
        return (image, aux_targets) if _has_aux else image

    def _route_aux(self, aux_targets: dict[str, Tensor], composed_batch: Tensor, image: Tensor) -> None:
        """Route auxiliary targets through the composed forward pixel matrix.

        Masks are resampled with a grid built from the same composed matrix used
        for the image warp; boxes and keypoints go through the composed forward
        pixel matrix. This reuses the shared :mod:`~fused_transforms.targets`
        builders so the numpy/cv2 path matches the torch affine path convention
        (center/``align_corners=True``). Mutates ``aux_targets`` in place.

        Args:
            aux_targets: Auxiliary targets to transform (``"mask"``, ``"bbox_xyxy"``,
                ``"bbox_xywh"``, ``"keypoints"``).
            composed_batch: ``(B, 3, 3)`` composed forward matrix (CPU float64).
            image: The warped image tensor, used to resolve the mask device/dtype.

        """
        acc_img = composed_batch.to(device=image.device, dtype=_matrix_geometry_dtype(image.dtype))
        grid: Tensor | None = None
        if "mask" in aux_targets:
            mask = aux_targets["mask"]
            acc_mask = composed_batch.to(device=mask.device, dtype=torch.float32)
            mtx_inv = inv3x3(acc_mask)
            mtx_norm = normalize_matrix(mtx_inv, mask.shape[-2], mask.shape[-1]).to(dtype=torch.float32)
            grid = F.affine_grid(
                mtx_norm[:, :2, :],
                [mask.shape[0], mask.shape[1], mask.shape[-2], mask.shape[-1]],
                align_corners=True,
            )
        self._route_grid_aux(
            aux_targets,
            grid,
            acc_img,
            self.mask_interpolation,
            self.keypoint_flip_index,
            self.mask_fill,
        )

    @staticmethod
    def _route_grid_aux(
        aux_targets: dict[str, Tensor],
        grid: Tensor | None,
        acc_img: Tensor,
        mask_interpolation: MaskInterpolationStr = "nearest",
        keypoint_flip_index: tuple[int, ...] | None = None,
        mask_fill: MaskFillValue = 0,
    ) -> None:
        """Route auxiliary targets through the warp grid and composed pixel matrix.

        Mask entries use ``grid`` (nearest-neighbour resample); boxes and keypoints
        use the composed forward pixel matrix ``acc_img``. Mutates ``aux_targets``
        in place.

        Args:
            aux_targets: Auxiliary targets to transform.
            grid: Sampling grid for the mask, or ``None`` when no mask is present.
            acc_img: ``(B, 3, 3)`` composed forward pixel matrix; recast to the geometry dtype here.
            mask_interpolation: Mask sampling mode for the ``"mask"`` target.
            mask_fill: Scalar border value for the ``"mask"`` target.
            keypoint_flip_index: Optional caller-supplied keypoint pair permutation, applied
                where the composed matrix reverses orientation.

        """
        from fused_transforms.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_mask,
            transform_rboxes,
        )

        acc_img = acc_img.to(dtype=_matrix_geometry_dtype(acc_img.dtype))

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask" and grid is not None:
                aux_targets[key] = transform_mask(val, grid, mode=mask_interpolation, fill=mask_fill)
            elif key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, acc_img)
            elif key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, acc_img)
            elif key == "keypoints":
                aux_targets[key] = _route_keypoints(val, acc_img, keypoint_flip_index)
            elif key == "rboxes":
                aux_targets[key] = transform_rboxes(val, acc_img)

    def _compose_matrices(self, image: Tensor) -> tuple[list[MatrixArray], list[bool]]:
        """Compose the per-sample forward affine matrices via Albumentations sampling.

        Runs the exact per-sample activation draws and ``sample_params`` calls that
        the cv2 path has always used, so the numpy RNG stream is unchanged. The
        result feeds either the cv2 or the batched-torch warp.

        Args:
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            A ``(accs, any_active)`` pair: ``accs`` is a per-sample list of
            ``(3, 3)`` float64 forward matrices; ``any_active[b]`` is ``True`` when
            at least one transform applied to sample ``b`` (identity otherwise).

        """
        batch_size, num_channels, height, width = image.shape

        # Pre-draw per-transform active masks before the sample loop so that
        # same_on_batch=True collapses to a single Bernoulli draw shared across all samples.
        active_masks: list[Any] = []
        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = bool(getattr(tfm, "same_on_batch", False))
            if same_on_batch:
                draw = bool(np.random.rand() < prob)
                active_masks.append(np.full(batch_size, draw))
            else:
                active_masks.append(np.random.rand(batch_size) < prob)

        # same_on_batch transforms share ONE param draw across the whole batch —
        # matching the shared activation draw above. Sample once here so every
        # sample gets identical geometry (previously only the activation was
        # shared while params were re-drawn per sample).
        shared_mtx: dict[int, MatrixArray] = {}
        for t_idx, tfm in enumerate(self.transforms):
            if bool(getattr(tfm, "same_on_batch", False)) and bool(np.any(active_masks[t_idx])):
                shared_mtx[t_idx] = self._sample_matrix_numpy(
                    self.adapter,
                    tfm,
                    self._tfm_tags[t_idx],
                    num_channels,
                    height,
                    width,
                    tensor_roundtrip=True,
                )

        accs: list[MatrixArray] = []
        any_active: list[bool] = []
        for b_idx in range(batch_size):
            acc: MatrixArray = np.eye(3, dtype=np.float64)
            active = False
            for t_idx, tfm in enumerate(self.transforms):
                # Skip BEFORE sampling (matching forward_numpy): inactive transforms
                # must not consume RNG draws, otherwise entry points diverge under a
                # fixed seed for any prob < 1.0 chain.
                if not active_masks[t_idx][b_idx]:
                    continue
                if t_idx in shared_mtx:
                    active = True
                    acc = shared_mtx[t_idx] @ acc
                    continue
                mtx_i = self._sample_matrix_numpy(
                    self.adapter,
                    tfm,
                    self._tfm_tags[t_idx],
                    num_channels,
                    height,
                    width,
                    tensor_roundtrip=True,
                )
                active = True
                acc = mtx_i @ acc
            accs.append(acc)
            any_active.append(active)
        return accs, any_active

    @staticmethod
    def _stack_matrices(accs: list[MatrixArray]) -> Tensor:
        """Stack per-sample ``(3, 3)`` numpy matrices into a CPU ``(B, 3, 3)`` float64 tensor.

        The batch is built on CPU because the source matrices are numpy (CPU) and
        MPS has no float64 support; callers move and cast it as needed (the torch
        warp path casts to a device-safe dtype before touching the accelerator).

        Args:
            accs: Per-sample forward matrices from :meth:`_compose_matrices`.

        Returns:
            A CPU ``(len(accs), 3, 3)`` float64 tensor, or a ``(0, 3, 3)`` tensor
            when ``accs`` is empty.

        """
        if not accs:
            return torch.empty((0, 3, 3), dtype=torch.float64)
        return torch.as_tensor(np.stack(accs), dtype=torch.float64)

    def _warp_cv2(self, image: Tensor, accs: list[MatrixArray], any_active: list[bool]) -> Tensor:
        """Warp each sample with one ``cv2.warpAffine`` (default CPU strategy).

        Args:
            image: ``(B, C, H, W)`` input tensor.
            accs: Per-sample forward matrices from :meth:`_compose_matrices`.
            any_active: Per-sample activity flags; inactive samples pass through untouched.

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        interp_flag = self._interp_flag
        border_flag = self._border_flag

        output_np: list[ImageArray] = []
        for b_idx in range(batch_size):
            img_np = image[b_idx].detach().permute(1, 2, 0).cpu().numpy()  # detach: no grad through cv2 segments
            if not any_active[b_idx]:
                output_np.append(img_np)
                continue
            # acc is the composed forward (src->dst) matrix; invert to get dst->src for _warp
            m_dst2src = np.linalg.inv(accs[b_idx])
            if num_channels == 1:
                img_np = img_np[:, :, 0]
                warped = _warp(img_np, m_dst2src, width, height, interp_flag, border_flag, self.fill)
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, m_dst2src, width, height, interp_flag, border_flag, self.fill)
            output_np.append(warped)

        return torch.stack([torch.as_tensor(np.ascontiguousarray(img)).permute(2, 0, 1) for img in output_np]).to(
            device=device, dtype=dtype
        )

    def _warp_torch(self, image: Tensor, composed_batch: Tensor) -> Tensor:
        """Warp the whole batch with one ``grid_sample`` (opt-in torch strategy).

        Applies a single batched ``affine_grid`` + ``grid_sample`` using the
        matrices already composed by :meth:`_compose_matrices`, so the sampled
        geometry matches the cv2 path exactly; only the resampling backend differs.
        Inactive samples keep identity and pass through the near-identity warp.

        Args:
            image: ``(B, C, H, W)`` input tensor (any device — this path is the GPU/MPS warp).
            composed_batch: ``(B, 3, 3)`` float64 forward matrices (built on CPU).

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        # MPS has no float64; invert/normalize in float32 there (mirrors the torch twins).
        acc_dtype = _matrix_compose_dtype(image.dtype, image.device, len(self.transforms))
        acc = composed_batch.to(device=image.device, dtype=acc_dtype)
        warped, _ = _grid_sample_affine_batched(
            image,
            acc,
            self.interpolation,
            self.padding_mode,
            _fill_tensor(self.fill, image.shape[1], image.device, image.dtype),
        )
        return warped

    def forward_numpy(self, img_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply fused affine chain to a single HWC NumPy image (no tensor conversion).

        Reuses the same matrix composition logic as :meth:`forward` but operates
        entirely in NumPy/cv2 space, eliminating the BCHW tensor round-trip for
        the Albumentations native dict-input calling convention.

        Args:
            img_hwc: ``(height, width, channels)`` or ``(height, width)`` NumPy array (uint8 or float32).
                cv2 requires a C-contiguous array; a copy is made automatically
                if the input is not contiguous.

        Returns:
            Warped array with the same dtype and shape as ``img_hwc``.

        Note:
            ``_last_matrix`` is set to shape ``(1, 3, 3)`` after this call,
            matching the B=1 single-image case.  ``aux_targets`` are not
            supported; a ``RuntimeError`` is raised if aux routing is attempted
            via this path.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from fused_transforms.affine.segment import AlbuFusedAffineSegment
            >>> from fused_transforms.adapters.albumentations import AlbumentationsAdapter
            >>> seg = AlbuFusedAffineSegment([], AlbumentationsAdapter())
            >>> img = np.zeros((8, 8, 3), dtype=np.uint8)
            >>> out = seg.forward_numpy(img)
            >>> out.shape
            (8, 8, 3)

            ```

        """
        if not img_hwc.flags["C_CONTIGUOUS"]:
            img_hwc = np.ascontiguousarray(img_hwc)
        height, width = img_hwc.shape[:2]
        n_ch = img_hwc.shape[2] if img_hwc.ndim == 3 else 1
        original_2d = img_hwc.ndim == 2

        if len(self.transforms) == 0:
            # Clone the shared identity buffer so callers get an independent matrix
            # (honors the last_matrix "detached clone" contract; prevents a caller's
            # in-place edit from corrupting the reused constant).
            self._last_matrix = self._identity_1x3x3.detach().clone()
            _set_current_call_matrix(self._identity_1x3x3.detach().clone())
            return img_hwc

        # Draw per-transform active masks for bsz=1, one draw per transform whatever its probability.
        # Skipping the draw for prob 0.0 or 1.0 would give the same activation for ~1 us less work, and
        # would leave the global NumPy stream in a different place than the same chain reached through
        # forward(): a caller who seeded once and switched between array and tensor input would then get
        # different sampled geometry from an identical pipeline, with nothing in the configuration
        # explaining it. Entry-point agreement is worth more than the draw.
        active_masks: list[bool] = []
        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            active_masks.append(bool(np.random.rand() < prob))

        acc: MatrixArray = np.eye(3, dtype=np.float64)
        any_active = False

        # The shared native preparer also serves tensor callers, whose explicit
        # float32 roundtrip preserves their existing matrix numerics.
        _tags = self._tfm_tags
        for idx_tfm, tfm in enumerate(self.transforms):
            if not active_masks[idx_tfm]:
                # Skip expensive sample_params + build_matrix for inactive transforms.
                continue
            matrix = self._sample_matrix_numpy(
                self.adapter,
                tfm,
                _tags[idx_tfm],
                n_ch,
                height,
                width,
                tensor_roundtrip=False,
            )
            any_active = True
            acc = matrix @ acc

        np.copyto(self._last_matrix_np_buffer, acc, casting="unsafe")
        self._last_matrix_buffer[0].copy_(self._last_matrix_np_tensor)
        # Clone: _last_matrix_buffer is reused and overwritten in place on the next call,
        # so a bare reference would let call N+1 mutate call N's returned matrix.
        self._last_matrix = self._last_matrix_buffer.clone()
        _set_current_call_matrix(torch.from_numpy(np.asarray(acc, dtype=np.float32)).unsqueeze(0).clone())
        # This path is cv2 by construction -- it never builds a tensor to hand grid_sample -- so an
        # "auto" pipeline reached through NumPy reports the engine that actually drew the pixels.
        self._last_execution = "cv2"

        if not any_active:
            return img_hwc

        tol = 1e-6
        is_bottom_row = (
            abs(float(acc[2, 0])) < tol and abs(float(acc[2, 1])) < tol and abs(float(acc[2, 2]) - 1.0) < tol
        )
        is_no_shear = abs(float(acc[0, 1])) < tol and abs(float(acc[1, 0])) < tol
        if is_bottom_row and is_no_shear:
            is_hflip = (
                abs(float(acc[0, 0]) + 1.0) < tol
                and abs(float(acc[1, 1]) - 1.0) < tol
                and abs(float(acc[0, 2]) - float(width - 1)) < tol
                and abs(float(acc[1, 2])) < tol
            )
            if is_hflip:
                return np.ascontiguousarray(np.flip(img_hwc, axis=1))

            is_vflip = (
                abs(float(acc[0, 0]) - 1.0) < tol
                and abs(float(acc[1, 1]) + 1.0) < tol
                and abs(float(acc[0, 2])) < tol
                and abs(float(acc[1, 2]) - float(height - 1)) < tol
            )
            if is_vflip:
                return np.ascontiguousarray(np.flip(img_hwc, axis=0))

            is_hvflip = (
                abs(float(acc[0, 0]) + 1.0) < tol
                and abs(float(acc[1, 1]) + 1.0) < tol
                and abs(float(acc[0, 2]) - float(width - 1)) < tol
                and abs(float(acc[1, 2]) - float(height - 1)) < tol
            )
            if is_hvflip:
                return np.ascontiguousarray(np.flip(img_hwc, axis=(0, 1)))

        m_dst2src: MatrixArray = _inv3x3_affine_np(acc)

        if original_2d:
            return _warp(img_hwc, m_dst2src, width, height, self._interp_flag, self._border_flag, self.fill)
        if n_ch == 1:
            warped = _warp(img_hwc[:, :, 0], m_dst2src, width, height, self._interp_flag, self._border_flag, self.fill)
            return warped[:, :, np.newaxis]
        return _warp(img_hwc, m_dst2src, width, height, self._interp_flag, self._border_flag, self.fill)

    def route_numpy_aux(self, aux_targets: dict[str, Tensor]) -> None:
        """Route auxiliary targets through the matrix left behind by :meth:`forward_numpy`.

        The NumPy-native path warps the image in cv2 space and never builds the BCHW tensor that
        :meth:`forward` routes auxiliary targets from. Coordinate targets are small enough that
        routing them as tensors costs nothing measurable, so this reuses the same
        :meth:`_route_grid_aux` the tensor path uses rather than restating the AABB and keypoint
        conventions in NumPy, where they would drift.

        Call it immediately after :meth:`forward_numpy` on the same segment -- it reads that call's
        composed matrix. Mutates ``aux_targets`` in place; a segment that composed nothing leaves
        every target untouched.

        Args:
            aux_targets: Auxiliary targets as tensors, keyed as in :meth:`_route_grid_aux`.

        """
        composed = self._last_matrix
        if composed is None or not aux_targets:
            return

        grid: Tensor | None = None
        if "mask" in aux_targets:
            mask = aux_targets["mask"]
            acc_mask = composed.to(device=mask.device, dtype=torch.float32)
            mtx_inv = inv3x3(acc_mask)
            mtx_norm = normalize_matrix(mtx_inv, mask.shape[-2], mask.shape[-1]).to(dtype=torch.float32)
            grid = F.affine_grid(
                mtx_norm[:, :2, :],
                [mask.shape[0], mask.shape[1], mask.shape[-2], mask.shape[-1]],
                align_corners=True,
            )
        self._route_grid_aux(
            aux_targets,
            grid,
            composed,
            self.mask_interpolation,
            self.keypoint_flip_index,
            self.mask_fill,
        )


# ---------------------------------------------------------------------------
# ProjectiveSegment — PyTorch backend for perspective transforms
# ---------------------------------------------------------------------------


class ProjectiveSegment(_BaseAffineSegment):
    """Fused projective segment that composes homography matrices into one grid_sample call.

    Identical to :class:`FusedAffineSegment` in accumulation and auxiliary-target
    handling -- both share the :class:`_BaseAffineSegment` composition engine --
    but overrides :meth:`_apply_grid` to use
    :func:`~fused_transforms.affine.matrix.perspective_grid` instead of
    ``F.affine_grid`` so the full ``3x3`` homography (including perspective
    division) is applied correctly.

    Args:
        transforms: List of projective transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to canonical parameters and matrices.
        interpolation: Optional interpolation mode override (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults
            to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to
            ``"zeros"`` when ``None``.

    """

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused projective transform chain via a single grid_sample call.

        Args:
            image: ``(B, C, H, W)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        acc, acc_img = self._compose(image)
        # One shared detached clone for both the last_matrix property and the per-call
        # context matrix (return_matrix path); compose.py only reads them, never mutates.
        matrix_copy = acc_img.detach().clone()
        self._last_matrix = matrix_copy
        _set_current_call_matrix(matrix_copy)

        image, grid = self._apply_grid(image, acc)

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            self._route_grid_aux(
                aux_targets,
                grid,
                acc_img,
                self.mask_interpolation,
                self.keypoint_flip_index,
                self.mask_fill,
            )

        if not _has_aux:
            return image
        return image, aux_targets

    def _apply_grid(self, image: Tensor, acc: Tensor) -> tuple[Tensor, Tensor]:
        """Warp ``image`` with a single ``perspective_grid`` + ``grid_sample`` pass.

        Inverts the composed forward homography, normalizes it to the ``[-1, 1]``
        grid convention, builds a perspective grid (with the perspective division
        ``F.affine_grid`` cannot express), and resamples. The grid is returned so
        the caller can reuse it to warp mask auxiliary targets.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            acc: ``(batch_size, 3, 3)`` composed forward homography in the compose dtype.

        Returns:
            A ``(warped_image, grid)`` tuple.

        """
        warp_fn = self._select_warp_fn("perspective", image.device)
        return warp_fn(
            image,
            acc,
            self.interpolation or "bilinear",
            self.padding_mode or "zeros",
            _fill_tensor(self.fill, image.shape[1], image.device, image.dtype),
        )


# ---------------------------------------------------------------------------
# AlbuProjectiveSegment — cv2 backend for Albumentations perspective transforms
# ---------------------------------------------------------------------------


class AlbuProjectiveSegment(nn.Module):
    """Fused projective segment for NumPy/cv2 backends (Albumentations).

    Loops over B samples, composes per-sample ``(3, 3)`` forward homography
    matrices, and applies a single ``cv2.warpPerspective`` per sample.

    The input and output are ``(B, C, H, W)`` float32 ``torch.Tensor`` objects.
    Conversion to/from ``(H, W, C)`` NumPy arrays happens inside ``forward()``.

    Args:
        transforms: List of Albumentations perspective transform objects to fuse.
        adapter: An ``AlbumentationsAdapter`` providing ``sample_params``,
            ``build_matrix``, and category lookup.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults to ``"bilinear"``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to ``"zeros"``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        execution: ExecutionStr = "cv2",
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
        fill: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize ``AlbuProjectiveSegment``."""
        super().__init__()
        self.execution: ExecutionStr = _validate_execution(execution)
        #: Engine the most recent call actually used. Equal to ``execution`` unless that is
        #: ``"auto"``, in which case it records what the routing rule chose. ``None`` before the
        #: first call, and before any call that warps nothing.
        self._last_execution: ExecutionStr | None = None
        # cv2 is only required for the default cv2 warp strategy; the torch
        # strategy warps with grid_sample and needs no OpenCV.
        if self.execution in ("cv2", "auto") and _cv2_module() is None:
            raise ImportError(
                "AlbuProjectiveSegment requires opencv-python because it uses cv2.warpPerspective under the hood."
            )
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation or "bilinear"
        self.padding_mode = padding_mode or "zeros"
        self.mask_interpolation = mask_interpolation
        self.mask_fill = mask_fill
        self._last_matrix: Tensor | None = None
        self._interp_flag: int = _CV2_INTERP.get(self.interpolation, 1)
        self._border_flag: int = _CV2_BORDER.get(self.padding_mode, 0)
        self.fill = fill
        self._tfm_tags = AlbuFusedAffineSegment._classify_transforms(transforms, adapter)

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The composed forward matrix (detached clone), or ``None`` before the first call to :meth:`forward`.

        """
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply fused projective chain via per-sample cv2.warpPerspective.

        Args:
            image: ``(B, C, H, W)`` float32 input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image. Masks are resampled through a perspective grid built
                from the same composed homography used for the image; bounding
                boxes and keypoints are transformed through the composed forward
                homography (with perspective division). The coordinate convention
                matches the torch projective path.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size = image.shape[0]

        # Compose per-sample forward homographies with Albumentations' own
        # per-sample sampling (numpy RNG stream) — identical for both execution
        # strategies; only the warp backend differs.
        accs, any_active = self._compose_matrices(image)
        composed_batch = self._stack_matrices(accs)
        # The public matrix keeps the image's own precision (float32/float64); only a
        # low-precision image promotes it to float32, matching the torch affine path.
        public_dtype = _matrix_public_dtype(image.dtype)
        self._last_matrix = composed_batch.to(dtype=public_dtype).clone().detach()
        _set_current_call_matrix(composed_batch.to(device=image.device, dtype=public_dtype).detach().clone())

        if batch_size == 0 or len(self.transforms) == 0:
            return (image, aux_targets) if _has_aux else image

        self._last_execution = _resolve_execution(self.execution, image.device)
        if self._last_execution == "torch":
            image = self._warp_torch(image, composed_batch)
        else:
            image = self._warp_cv2(image, accs, any_active)

        if aux_targets:
            self._route_aux(aux_targets, composed_batch, image)
        return (image, aux_targets) if _has_aux else image

    def _route_aux(self, aux_targets: dict[str, Tensor], composed_batch: Tensor, image: Tensor) -> None:
        """Route auxiliary targets through the composed forward homography.

        Masks are resampled with a perspective grid built from the same composed
        homography used for the image warp; boxes and keypoints go through the
        composed forward homography. Reuses the shared
        :mod:`~fused_transforms.targets` builders so the numpy/cv2 path matches
        the torch projective path convention. Mutates ``aux_targets`` in place.

        Args:
            aux_targets: Auxiliary targets to transform.
            composed_batch: ``(B, 3, 3)`` composed forward homography (CPU float64).
            image: The warped image tensor, used to resolve the box/keypoint dtype.

        """
        acc_img = composed_batch.to(device=image.device, dtype=_matrix_geometry_dtype(image.dtype))
        grid: Tensor | None = None
        if "mask" in aux_targets:
            mask = aux_targets["mask"]
            acc_mask = composed_batch.to(device=mask.device, dtype=torch.float32)
            mtx_inv = inv3x3(acc_mask)
            mtx_norm = normalize_matrix(mtx_inv, mask.shape[-2], mask.shape[-1]).to(dtype=torch.float32)
            grid = perspective_grid(mtx_norm, mask.shape[-2], mask.shape[-1])
        AlbuFusedAffineSegment._route_grid_aux(
            aux_targets,
            grid,
            acc_img,
            self.mask_interpolation,
            mask_fill=self.mask_fill,
        )

    def _compose_matrices(self, image: Tensor) -> tuple[list[MatrixArray], list[bool]]:
        """Compose per-sample forward homographies via Albumentations sampling.

        Runs the exact per-sample activation draws and ``sample_params`` calls the
        cv2 path has always used, so the numpy RNG stream is unchanged.

        Args:
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            A ``(accs, any_active)`` pair: ``accs`` is a per-sample list of
            ``(3, 3)`` float64 forward homographies; ``any_active[b]`` is ``True``
            when at least one transform applied to sample ``b``.

        """
        batch_size, num_channels, height, width = image.shape

        active_masks: list[Any] = []
        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = bool(getattr(tfm, "same_on_batch", False))
            if same_on_batch:
                draw = bool(np.random.rand() < prob)
                active_masks.append(np.full(batch_size, draw))
            else:
                active_masks.append(np.random.rand(batch_size) < prob)

        # same_on_batch transforms share ONE param draw across the whole batch,
        # matching the shared activation draw above (mirrors AlbuFusedAffineSegment).
        shared_mtx: dict[int, MatrixArray] = {}
        for t_idx, tfm in enumerate(self.transforms):
            if bool(getattr(tfm, "same_on_batch", False)) and bool(np.any(active_masks[t_idx])):
                shared_mtx[t_idx] = AlbuFusedAffineSegment._sample_matrix_numpy(
                    self.adapter,
                    tfm,
                    self._tfm_tags[t_idx],
                    num_channels,
                    height,
                    width,
                    tensor_roundtrip=True,
                )

        accs: list[MatrixArray] = []
        any_active: list[bool] = []
        for b_idx in range(batch_size):
            acc: MatrixArray = np.eye(3, dtype=np.float64)
            active = False
            for t_idx, tfm in enumerate(self.transforms):
                # Skip BEFORE sampling (matching forward_numpy): inactive transforms
                # must not consume RNG draws, otherwise entry points diverge under a
                # fixed seed for any prob < 1.0 chain.
                if not active_masks[t_idx][b_idx]:
                    continue
                if t_idx in shared_mtx:
                    active = True
                    acc = shared_mtx[t_idx] @ acc
                    continue
                mtx_i = AlbuFusedAffineSegment._sample_matrix_numpy(
                    self.adapter,
                    tfm,
                    self._tfm_tags[t_idx],
                    num_channels,
                    height,
                    width,
                    tensor_roundtrip=True,
                )
                active = True
                acc = mtx_i @ acc
            accs.append(acc)
            any_active.append(active)
        return accs, any_active

    @staticmethod
    def _stack_matrices(accs: list[MatrixArray]) -> Tensor:
        """Stack per-sample ``(3, 3)`` numpy homographies into a CPU ``(B, 3, 3)`` float64 tensor.

        Built on CPU (numpy source, and MPS has no float64); the torch warp path
        casts to a device-safe dtype and moves to the accelerator.

        Args:
            accs: Per-sample forward homographies from :meth:`_compose_matrices`.

        Returns:
            A CPU ``(len(accs), 3, 3)`` float64 tensor, or a ``(0, 3, 3)`` tensor
            when ``accs`` is empty.

        """
        if not accs:
            return torch.empty((0, 3, 3), dtype=torch.float64)
        return torch.as_tensor(np.stack(accs), dtype=torch.float64)

    def _warp_cv2(self, image: Tensor, accs: list[MatrixArray], any_active: list[bool]) -> Tensor:
        """Warp each sample with one ``cv2.warpPerspective`` (default CPU strategy).

        Args:
            image: ``(B, C, H, W)`` input tensor.
            accs: Per-sample forward homographies from :meth:`_compose_matrices`.
            any_active: Per-sample activity flags; inactive samples pass through untouched.

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        cv2_interp = self._interp_flag
        cv2_border = self._border_flag

        output_np: list[ImageArray] = []
        for b_idx in range(batch_size):
            img_np = image[b_idx].detach().permute(1, 2, 0).cpu().numpy()  # detach: no grad through cv2 segments
            if not any_active[b_idx]:
                output_np.append(img_np)
                continue
            # acc is the composed forward (src->dst) matrix; invert to get dst->src
            mtx_inv = np.linalg.inv(accs[b_idx])
            warped: ImageArray = _require_cv2().warpPerspective(
                img_np,
                mtx_inv,  # dst->src inverse map
                (width, height),  # dsize = (W, H)
                flags=cv2_interp | _CV2_WARP_INVERSE_MAP,
                borderMode=cv2_border,
                borderValue=_cv2_border_value(self.fill, num_channels),
            )
            if warped.ndim == 2:
                warped = warped[..., None]
            output_np.append(warped)

        return torch.stack([torch.as_tensor(np.ascontiguousarray(img)).permute(2, 0, 1) for img in output_np]).to(
            device=device, dtype=dtype
        )

    def _warp_torch(self, image: Tensor, composed_batch: Tensor) -> Tensor:
        """Warp the whole batch with one perspective ``grid_sample`` (opt-in torch strategy).

        Uses the homographies already composed by :meth:`_compose_matrices`, so the
        sampled geometry matches the cv2 path exactly; only the resampling backend
        differs. Inactive samples keep identity and pass through the near-identity warp.

        Args:
            image: ``(B, C, H, W)`` input tensor (any device — this path is the GPU/MPS warp).
            composed_batch: ``(B, 3, 3)`` float64 forward homographies on the image device.

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        acc_dtype = _matrix_compose_dtype(image.dtype, image.device, len(self.transforms))
        acc = composed_batch.to(device=image.device, dtype=acc_dtype)
        warped, _ = _grid_sample_perspective_batched(
            image,
            acc,
            self.interpolation,
            self.padding_mode,
            _fill_tensor(self.fill, image.shape[1], image.device, image.dtype),
        )
        return warped


class FusedColorSegment(GeneratorPicklingMixin, nn.Module):
    """Fused colour-space segment that composes POINTWISE_LINEAR transforms into one matrix multiply.

    Accumulates per-sample ``(B, 4, 4)`` homogeneous colour-space affine matrices for every transform in the
    segment, multiplies all pixels by the composed matrix, and clamps the result to ``[0, 1]``.  All operations are
    vectorised over the batch dimension.

    Colour transforms do **not** affect spatial layout, so auxiliary targets (masks, bounding boxes, keypoints) are
    returned unchanged.

    Args:
        transforms: List of ``POINTWISE_LINEAR`` transform objects to fuse.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_color_matrix`` for each transform.
        clip_output: When ``True`` (default), the fused output is clamped to ``[0, 1]`` after the matrix multiply,
            matching the typical behaviour of individual colour transforms.  Set to ``False`` only when the pipeline
            intentionally produces values outside this range (e.g. transforms configured with ``clip_output=False``
            in the underlying library).

    """

    # Buffer — declared here so mypy resolves self._eye4 as Tensor, not Tensor | Module.
    _eye4: Tensor

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        clip_output: bool = True,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        *,
        compile_color: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize ``FusedColorSegment``.

        Args:
            transforms: ``POINTWISE_LINEAR`` transforms to fuse.
            adapter: Adapter providing ``sample_params`` and ``build_color_matrix``.
            clip_output: Whether to clamp the final output to ``[0, 1]``.
            randomness: Batch randomness policy.
            clip_policy: ``"final"`` (default) fuses the whole chain into one matmul and clamps once;
                ``"per_op_parity"`` clamps at each op whose intermediate could leave ``[0, 1]``,
                matching a native per-op clamped chain.
            compile_color: Compile only the pure color matrix application on non-CPU devices.
            generator: Caller-owned generator driving factor sampling and the per-transform
                probability gates, or ``None`` for the global torch stream.

        """
        super().__init__()
        self._transforms = transforms
        self._adapter = adapter
        self.clip_output = clip_output
        self.randomness = randomness
        self.generator = generator
        if clip_policy not in ("final", "per_op_parity"):
            msg = "unknown clip policy {!r}; expected 'final' or 'per_op_parity'"
            raise ValueError(msg.format(clip_policy))
        self.clip_policy: ClipPolicyStr = clip_policy
        self._compile_color: bool = compile_color and _torch_supports_compile()
        # Register identity matrix as a buffer so device moves (.to(), .cuda())
        # propagate automatically — avoids re-allocating every forward pass.
        self.register_buffer("_eye4", torch.eye(4, dtype=torch.float32))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state; back-compat: add missing fields from older pickles."""
        super().__setstate__(state)
        if "_eye4" not in self._buffers:
            self.register_buffer("_eye4", torch.eye(4, dtype=torch.float32))
        # clip_output added in v0.7; default True preserves pre-existing behaviour.
        if not hasattr(self, "clip_output"):
            self.clip_output = True
        if not hasattr(self, "randomness"):
            self.randomness = RandomnessPolicy.BACKEND
        # clip_policy added later; default "final" preserves the single-matmul behaviour.
        if not hasattr(self, "clip_policy"):
            self.clip_policy = "final"
        if not hasattr(self, "_compile_color"):
            self._compile_color = False

    @property
    def transforms(self) -> list[object]:
        """Return the list of transforms in this segment."""
        return list(self._transforms)

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Any] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Apply the fused colour matrix to the image batch.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor with values in ``[0, 1]``.
            aux_targets: Optional dict of auxiliary targets. Colour transforms
                do not affect spatial layout, so these are returned unchanged.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        batch_size, channels, height, width = image.shape

        # The 4x4 color matrix is defined for 3-channel RGB images only. For non-RGB inputs, an adapter whose
        # colour ops treat every channel alike applies them per channel; others fall back to passthrough.
        if channels != 3 and getattr(self._adapter, "channel_uniform_color", False):
            return self._forward_channel_uniform(image, aux_targets)
        if channels != 3:
            for tfm in self._transforms:
                image = self._adapter.call_nonfused(tfm, image)
            if aux_targets is None:
                return image
            return image, aux_targets

        device = image.device
        dtype = image.dtype
        image_in = image

        # Cast the registered buffer to the current device/dtype (no-op for float32 CPU)
        eye = self._eye4.to(device=device, dtype=dtype)

        input_shape = (batch_size, channels, height, width)

        # Per-channel mean of the image reaching the current transform, (batch_size, channels).
        # Mid-chain contrast ops need the luminance of THEIR input, not the raw segment input;
        # because every fused op is linear (c' = M c + b), the mean transforms exactly as
        # mean(M c + b) = M mean(c) + b, so we carry mean_ch forward through the same matrices.
        mean_ch = image.reshape(batch_size, channels, height * width).mean(dim=2)  # (batch_size, channels)
        parity_image = image
        parity_sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
        parity_sub_started = False

        matrices: list[Tensor] = []
        for tfm in self._transforms:
            if (
                self.clip_policy == "per_op_parity"
                and self._is_normalize(tfm)
                and parity_sub_started
                and self._range_escapes_gamut(parity_sub)
            ):
                parity_image = self._matmul_image(parity_image, parity_sub).clamp(0.0, 1.0)
                mean_ch = parity_image.reshape(batch_size, channels, height * width).mean(dim=2)
                parity_sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
                parity_sub_started = False

            prob = _transform_prob(tfm)
            same_on_batch = _shares_randomness_across_batch(self._adapter, tfm, self.randomness)
            if same_on_batch:
                active_scalar = _rand((), device=device, generator=self.generator) < prob
                active = active_scalar.expand(batch_size)
            else:
                active = _rand(batch_size, device=device, generator=self.generator) < prob

            params = _sample_transform_params(
                self._adapter, tfm, input_shape, device, self.randomness, generator=self.generator
            )
            # Contrast-like ops take their midpoint from the per-image luminance of their input;
            # pass it so the fused matrix reproduces the native mean-relative contrast exactly.
            # Only thread `mean` when a mean-relative op actually needs it, so adapters whose
            # build_color_matrix predates the parameter (or custom ones) keep the two-arg call.
            luma_mean = self._prefix_luma_mean(tfm, mean_ch)
            try:
                if luma_mean is None:
                    mat = self._adapter.build_color_matrix(tfm, params)  # (batch_size, 4, 4)
                else:
                    mat = self._adapter.build_color_matrix(tfm, params, mean=luma_mean)  # (batch_size, 4, 4)
            except NotImplementedError:
                # If a transform that passed the build-time probe raises NotImplementedError
                # at forward time (e.g. probe used empty params), abort fusion entirely and
                # restart from the original image so no partial fused state is applied.
                image_fallback = image_in
                for tfm_nonfused in self._transforms:
                    image_fallback = self._adapter.call_nonfused(tfm_nonfused, image_fallback)
                if aux_targets is None:
                    return image_fallback
                return image_fallback, aux_targets

            # Expand to batch if adapter returned (1, 4, 4)
            if mat.shape[0] == 1 and batch_size > 1:
                mat = mat.expand(batch_size, -1, -1)

            mat = mat.to(device=device, dtype=dtype)
            mat = torch.where(active[:, None, None], mat, eye.unsqueeze(0).expand(batch_size, -1, -1))
            matrices.append(mat)

            if self.clip_policy == "per_op_parity" and not self._is_normalize(tfm):
                candidate = torch.bmm(mat, parity_sub) if parity_sub_started else mat
                if self._range_escapes_gamut(candidate):
                    parity_image = self._matmul_image(parity_image, candidate).clamp(0.0, 1.0)
                    mean_ch = parity_image.reshape(batch_size, channels, height * width).mean(dim=2)
                    parity_sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
                    parity_sub_started = False
                    continue
                parity_sub = candidate
                parity_sub_started = True
            mean_ch = self._advance_channel_mean(mean_ch, mat)

        image_out = self._apply_color_matrices(image, matrices, eye)

        if self.clip_output:
            image_out = image_out.clamp(0.0, 1.0)
        if aux_targets is None:
            return image_out
        return image_out, aux_targets

    def _forward_channel_uniform(
        self,
        image: Tensor,
        aux_targets: dict[str, Any] | None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Apply channel-uniform colour ops to an image of any channel count.

        Used for non-RGB input when the adapter declares ``channel_uniform_color``: each op's RGB matrix scales
        every channel by one factor about one midpoint, so it reduces to a per-image ``gain * x + bias``. Gates and
        parameters are drawn in the same order as the RGB path, so one seed gives the same factors for any channel
        count.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor with values in ``[0, 1]``.
            aux_targets: Optional auxiliary targets, returned unchanged.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``; ``(image, aux_targets)`` tuple otherwise.

        """
        batch_size, channels, height, width = image.shape
        input_shape = (batch_size, channels, height, width)
        device = image.device
        out = image
        for tfm in self._transforms:
            prob = _transform_prob(tfm)
            if _shares_randomness_across_batch(self._adapter, tfm, self.randomness):
                active = (_rand((), device=device, generator=self.generator) < prob).expand(batch_size)
            else:
                active = _rand(batch_size, device=device, generator=self.generator) < prob
            params = _sample_transform_params(
                self._adapter,
                tfm,
                input_shape,
                device,
                self.randomness,
                generator=self.generator,
            )
            mat = self._adapter.build_color_matrix(tfm, params).to(device=device, dtype=image.dtype)
            if mat.shape[0] == 1 and batch_size > 1:
                mat = mat.expand(batch_size, -1, -1)
            gain = torch.where(active, mat[:, 0, 0], torch.ones_like(mat[:, 0, 0]))
            bias = torch.where(active, mat[:, 0, 3], torch.zeros_like(mat[:, 0, 3]))
            out = out * gain.view(-1, 1, 1, 1) + bias.view(-1, 1, 1, 1)
            if self.clip_policy == "per_op_parity":
                out = out.clamp(0.0, 1.0)
        if self.clip_output:
            out = out.clamp(0.0, 1.0)
        if aux_targets is None:
            return out
        return out, aux_targets

    def _apply_color_matrices(self, image: Tensor, matrices: list[Tensor], eye: Tensor) -> Tensor:
        """Apply the per-op color matrices, splitting for clamp parity when requested.

        Under ``clip_policy="final"`` the matrices compose into ONE ``(B, 4, 4)`` matmul and the whole
        chain is applied in a single pass (the more precise, bit-compatible default). Under
        ``"per_op_parity"`` the chain is split at every op whose intermediate provably escapes
        ``[0, 1]``: the fused sub-chain before it is applied and clamped, matching a native per-op
        clamped chain, while safe adjacent ops stay fused.

        Args:
            image: ``(B, 3, H, W)`` input image.
            matrices: Per-op probability-masked ``(B, 4, 4)`` color matrices, in application order.
            eye: A ``(4, 4)`` identity in the image device/dtype.

        Returns:
            The transformed ``(B, 3, H, W)`` image (final clamp applied by the caller).

        """
        batch_size = image.shape[0]
        if not matrices:
            return image
        has_normalize = any(self._is_normalize(transform) for transform in self._transforms)
        if self.clip_policy == "final" and not has_normalize:
            acc = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
            for mat in matrices:
                acc = torch.bmm(mat, acc)
            return self._apply_color_matrix(image, acc)

        # Apply one fused sub-chain at a time. A Normalize boundary first clamps an escaping
        # prefix because native color ops clamp before Normalize, while Normalize itself does not.
        out = image
        sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
        sub_started = False
        for transform, mat in zip(self._transforms, matrices, strict=True):
            if self._is_normalize(transform) and sub_started and self._range_escapes_gamut(sub):
                out = self._apply_color_matrix(out, sub).clamp(0.0, 1.0)
                sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
                sub_started = False

            candidate = torch.bmm(mat, sub) if sub_started else mat
            if (
                self.clip_policy == "per_op_parity"
                and not self._is_normalize(transform)
                and self._range_escapes_gamut(candidate)
            ):
                out = self._apply_color_matrix(out, candidate).clamp(0.0, 1.0)
                sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
                sub_started = False
                continue
            sub = candidate
            sub_started = True
        return self._apply_color_matrix(out, sub)

    def _apply_color_matrix(self, image: Tensor, acc: Tensor) -> Tensor:
        """Apply a color matrix through the eager or compiled dense tensor core."""
        if self._compile_color and image.device.type != "cpu":
            return _compiled_color_fn()(image, acc)
        return _apply_color_matrix(image, acc)

    def _is_normalize(self, transform: object) -> bool:
        """Return whether the adapter identifies *transform* as Normalize."""
        checker = getattr(self._adapter, "is_normalize", None)
        return bool(checker(transform)) if callable(checker) else type(transform).__name__ == "Normalize"

    @staticmethod
    def _matmul_image(image: Tensor, acc: Tensor) -> Tensor:
        """Apply a ``(B, 4, 4)`` homogeneous color matrix to a ``(B, 3, H, W)`` image.

        The homogeneous bottom row is inert for the pixel outputs, so the affine
        part ``c' = A c + b`` is applied directly with a single ``baddbmm`` instead
        of building the ``(B, 4, H*W)`` augmented pixel block and running a full
        ``4x4`` matmul: no ``ones`` allocation, no ``cat``, ~25% fewer FLOPs. The
        accumulation order differs from the old homogeneous form, so results may
        drift by ~1e-7 (well within color-parity tolerances).

        """
        return _apply_color_matrix(image, acc)

    @staticmethod
    def _range_escapes_gamut(acc: Tensor) -> bool:
        """Return whether a composed color matrix can map a ``[0, 1]`` pixel outside ``[0, 1]``.

        For a per-channel affine ``c' = A c + b`` the reachable output range over inputs in
        ``[0, 1]^3`` is bounded by summing the positive parts (max) and negative parts (min) of each
        row plus the bias. If any channel's bound crosses the gamut for any sample, the fused chain
        must be clamped here to match a native per-op chain.

        Args:
            acc: ``(B, 4, 4)`` composed homogeneous color matrix.

        Returns:
            ``True`` if any sample/channel can leave ``[0, 1]``.

        """
        lin = acc[:, :3, :3]  # (B, 3, 3)
        bias = acc[:, :3, 3]  # (B, 3)
        hi = lin.clamp(min=0.0).sum(dim=2) + bias  # inputs at 1 for positive weights, 0 for negative
        lo = lin.clamp(max=0.0).sum(dim=2) + bias  # inputs at 1 for negative weights, 0 for positive
        eps = 1e-6
        return bool((hi > 1.0 + eps).any().item() or (lo < -eps).any().item())

    def _prefix_luma_mean(self, transform: object, mean_ch: Tensor) -> Tensor | None:
        """Weighted luminance of the transform's input, or ``None`` when it needs no mean.

        Only contrast-like ops with a mean-relative midpoint (``ColorJitter`` contrast) report
        luminance weights via :meth:`TransformAdapter.color_luma_weights`. For those, the per-image
        luminance is the same weighted sum of channel means the native op computes.

        Args:
            transform: The color transform about to be applied.
            mean_ch: Per-channel mean of the image reaching this transform, ``(batch_size, channels)``.

        Returns:
            A ``(batch_size,)`` luminance tensor for mean-relative ops, or ``None`` otherwise.

        """
        # color_luma_weights is an optional adapter capability (Protocol default returns None).
        # Adapters predating it — or lightweight custom ones — simply have no mean-relative op.
        get_weights = getattr(self._adapter, "color_luma_weights", None)
        weights = get_weights(transform) if callable(get_weights) else None
        if weights is None:
            return None
        w = torch.tensor(weights, device=mean_ch.device, dtype=mean_ch.dtype)  # (3,)
        return (mean_ch * w).sum(dim=1)  # (batch_size,)

    @staticmethod
    def _advance_channel_mean(mean_ch: Tensor, mat: Tensor) -> Tensor:
        """Apply a color matrix to the running per-channel mean.

        Because every fused color op is affine (``c' = M c + b``), the mean of the output channels is
        ``M mean(c) + b``. Carrying the mean this way lets a later contrast op read the luminance of the
        image it would actually see, matching a native per-op chain.

        Args:
            mean_ch: Per-channel mean before the op, ``(batch_size, channels)``.
            mat: The op's ``(batch_size, 4, 4)`` homogeneous color matrix (already probability-masked).

        Returns:
            Per-channel mean after the op, ``(batch_size, channels)``.

        """
        channels = mean_ch.shape[1]
        mean_hom = torch.cat([mean_ch, torch.ones_like(mean_ch[:, :1])], dim=1)  # (batch_size, 4)
        advanced = torch.bmm(mat, mean_hom.unsqueeze(2)).squeeze(2)  # (batch_size, 4)
        return advanced[:, :channels]


def _try_build_color_matrix(adapter: TransformAdapter, transform: object) -> bool:
    """Probe whether *adapter* supports ``build_color_matrix`` for *transform*.

    Calls the method with an empty param dict and classifies the outcome:

    - No exception → ``True`` (method succeeds with any params)
    - ``NotImplementedError`` / ``AttributeError`` → ``False`` (explicitly unsupported)
    - ``KeyError`` / ``IndexError`` → ``True`` (method exists, needs real params)
    - Any other exception (``RuntimeError``, etc.) → ``False`` (unexpected error;
      treat as unsupported to avoid silently mis-fusing a broken adapter)

    """
    try:
        adapter.build_color_matrix(transform, {})
        return True
    except (NotImplementedError, AttributeError):
        return False
    except (KeyError, IndexError):
        # Method exists but needs real params to succeed (missing param key).
        return True
    except Exception:
        # Unexpected error (e.g. RuntimeError from GPU OOM, device mismatch).
        # Treat as "not supported" to avoid silently mis-fusing a broken adapter.
        return False


def _flush_color(
    transforms: list[object],
    adapter: TransformAdapter,
    segments: list[object],
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    clip_policy: ClipPolicyStr = "final",
    compile_color: bool = False,
    generator: torch.Generator | None = None,
) -> None:
    """Flush a run of ``POINTWISE_LINEAR`` transforms into segments.

    If the adapter supports ``build_color_matrix`` for **every** transform in the run, they are folded into a single
    :class:`FusedColorSegment`. Otherwise the transforms fall back to passthrough (appended as-is). This helper
    intentionally mutates ``transforms`` in-place (clears it).

    Args:
        transforms: The pending run of ``POINTWISE_LINEAR`` transforms (cleared in place).
        adapter: Adapter used to probe and build color matrices.
        segments: Output segment list, appended in place.
        randomness: Batch randomness policy passed to the color segment.
        clip_policy: Clamp policy forwarded to :class:`FusedColorSegment`.
        compile_color: Whether non-CPU color matrix applications may use ``torch.compile``.
        generator: Caller-owned generator forwarded to :class:`FusedColorSegment`.

    """
    if not transforms:
        return
    # Probe each transform in the run; any failure means full passthrough.
    for tfm in transforms:
        if not _try_build_color_matrix(adapter, tfm):
            segments.extend(transforms)
            # Intentionally clears the caller-owned run buffer.
            transforms.clear()
            return
    color_transforms = list(transforms)
    clip_output = not any(_is_normalize_transform(adapter, tfm) for tfm in color_transforms)
    segments.append(
        FusedColorSegment(
            color_transforms,
            adapter,
            clip_output=clip_output,
            randomness=randomness,
            clip_policy=clip_policy,
            compile_color=compile_color,
            generator=generator,
        )
    )
    # Intentionally clears the caller-owned run buffer.
    transforms.clear()


def _is_normalize_transform(adapter: TransformAdapter, transform: object) -> bool:
    """Return whether an adapter marks a transform as pointwise Normalize."""
    checker = getattr(adapter, "is_normalize", None)
    return bool(checker(transform)) if callable(checker) else type(transform).__name__ == "Normalize"


#: Default number of grid points on the float interpolation-LUT path (uniform over ``[0, 1]``).
_DEFAULT_LUT_LEVELS = 1024


LutFn = Callable[..., Tensor]


def _apply_uint8_lut(image: Tensor, table: Tensor) -> Tensor:
    """Map integer pixels through a normalized 256-entry lookup table."""
    batch_size, channels, height, width = image.shape
    table_i = (table * 255).round().clamp(0, 255).to(torch.long)
    index = image.long().clamp(0, 255).reshape(batch_size, channels, height * width)
    mapped = torch.gather(table_i, 2, index).reshape(batch_size, channels, height, width)
    return mapped.to(image.dtype)


def _apply_runtime_float_lut(image: Tensor, table: Tensor) -> Tensor:
    """Map floating pixels through a byte-domain lookup table."""
    batch_size, channels, height, width = image.shape
    index = (image.clamp(0.0, 1.0) * 255).floor().long()
    index = index.reshape(batch_size, channels, height * width)
    mapped = torch.gather(table.to(dtype=image.dtype), 2, index).reshape(batch_size, channels, height, width)
    return mapped.to(image.dtype)


def _apply_interp_lut(image: Tensor, composed: Tensor, num_levels: int) -> Tensor:
    """Map floating pixels by interpolating a composed lookup table."""
    batch_size, channels, height, width = image.shape
    clamped = image.clamp(0.0, 1.0).reshape(batch_size, channels, height * width)
    pos = clamped * (num_levels - 1)
    lo = pos.floor().clamp(0.0, num_levels - 2)
    frac = pos - lo
    lo_idx = lo.to(torch.long)
    table = composed.to(dtype=image.dtype)
    lo_val = torch.gather(table, 2, lo_idx)
    hi_val = torch.gather(table, 2, lo_idx + 1)
    out = lo_val + (hi_val - lo_val) * frac
    jumps = (table[..., 1:] - table[..., :-1]).abs()
    neighbours = torch.maximum(F.pad(jumps[..., :-1], (1, 0)), F.pad(jumps[..., 1:], (0, 1)))
    discontinuity = (jumps > 8 * neighbours) & (jumps > 2 / (num_levels - 1))
    step = torch.gather(discontinuity, 2, lo_idx)
    out = torch.where(step & (frac >= 0.5), hi_val, out)
    return out.reshape(batch_size, channels, height, width).to(image.dtype)


# Runtime Equalize table construction is deliberately excluded. These functions
# only gather from an already-built table, so no image-dependent Python control flow
# enters the compiled region.
_COMPILED_LUT_CACHE: dict[str, LutFn] = {}


def _lut_fn(kind: str) -> LutFn:
    """Return the eager lookup-table application core for ``kind``.

    Args:
        kind: ``"uint8"``, ``"runtime_float"``, or ``"interp"`` application mode.

    Returns:
        The eager function for the requested lookup application.

    Raises:
        ValueError: If ``kind`` is not a known lookup application mode.

    """
    functions: dict[str, LutFn] = {
        "uint8": _apply_uint8_lut,
        "runtime_float": _apply_runtime_float_lut,
        "interp": _apply_interp_lut,
    }
    try:
        return functions[kind]
    except KeyError as exc:
        msg = f"unknown lookup application mode {kind!r}"
        raise ValueError(msg) from exc


def _compiled_lut_fn(kind: str) -> LutFn:
    """Return a dynamic-shape compiled lookup-table application core.

    Args:
        kind: ``"uint8"``, ``"runtime_float"``, or ``"interp"`` application mode.

    Returns:
        The cached compiled function for the requested lookup application.

    Raises:
        ValueError: If ``kind`` is not a known lookup application mode.

    """
    cached = _COMPILED_LUT_CACHE.get(kind)
    if cached is not None:
        return cached
    compiled = torch.compile(_lut_fn(kind), dynamic=True)
    _COMPILED_LUT_CACHE[kind] = compiled
    return compiled


class FusedLUTSegment(nn.Module):
    """Fused lookup-table segment that composes ``POINTWISE_LUT`` transforms into one lookup.

    Gamma, solarize, and posterize are static per-channel *non-linear scalar* maps. A contiguous
    run threads their domain grid through ``adapter.build_lut`` and applies each resulting static
    table once. Equalize is also accepted when its adapter supports a scalar, per-channel runtime
    histogram table: static tables flush before equalize, equalize builds one table per image, then
    later static maps begin a new table. Per-op probability is honoured with a per-sample active
    mask, exactly like :class:`FusedColorSegment`.

    Lookup maps leave spatial layout untouched, so auxiliary targets (masks, boxes, keypoints) pass
    through unchanged.

    Two execution paths, chosen by the image's dtype:

    - **Floating image — K-entry interpolation LUT (approximate).** The composed map is sampled on a
      uniform ``num_levels``-point grid over ``[0, 1]`` (default 1024) and applied by ``gather`` +
      interpolation. Smooth intervals use linear interpolation; a large isolated table jump uses a
      nearest-side step rule, avoiding a synthetic ramp at solarize and posterize boundaries. This
      remains an interpolation-tolerance path rather than an exact float implementation.
    - **Integer image — 256-entry table.** The map is enumerated over the 256 byte values, snapping to
      bytes between ops, so the composed table is exact *with respect to this segment's own composed
      byte maps* and carries no interpolation error. Because each op's byte map is built from this
      segment's evaluation rather than the backend's native integer kernel, a uint8 *tensor* result may
      differ from a backend's own native uint8 op by a few levels at posterize/solarize step boundaries.

    The Albumentations native NumPy (uint8) path is served by :meth:`forward_numpy`, which builds each
    op's 256-entry table from the transform's own native application, so it is bit-exact against the
    native sequential chain.

    Args:
        transforms: List of ``POINTWISE_LUT`` transform objects to fuse.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_lut`` for each transform.
        num_levels: Grid resolution of the float interpolation-LUT path. Default 1024.
        randomness: Batch randomness policy (mirrors :class:`FusedColorSegment`).

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms.affine.segment import FusedLUTSegment  # doctest: +SKIP
        >>> seg = FusedLUTSegment([gamma_tfm, solarize_tfm], adapter)  # doctest: +SKIP
        >>> out = seg(torch.rand(2, 3, 8, 8))  # doctest: +SKIP

        ```

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        num_levels: int = _DEFAULT_LUT_LEVELS,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_lut: bool = False,
    ) -> None:
        """Initialize ``FusedLUTSegment``.

        Args:
            transforms: ``POINTWISE_LUT`` transforms to fuse.
            adapter: Adapter providing ``sample_params`` and ``build_lut``.
            num_levels: Grid resolution of the float interpolation-LUT path (>= 2).
            randomness: Batch randomness policy.
            compile_lut: Compile only the pure LUT gather/interpolation on non-CPU devices.

        """
        super().__init__()
        if num_levels < 2:
            msg = f"num_levels must be >= 2, got {num_levels}"
            raise ValueError(msg)
        self._transforms = transforms
        self._adapter = adapter
        self.num_levels = num_levels
        self.randomness = randomness
        self._compile_lut: bool = compile_lut and _torch_supports_compile()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state; back-compat defaults for fields absent from older pickles."""
        super().__setstate__(state)  # type: ignore[no-untyped-call]
        if not hasattr(self, "num_levels"):
            self.num_levels = _DEFAULT_LUT_LEVELS
        if not hasattr(self, "randomness"):
            self.randomness = RandomnessPolicy.BACKEND
        if not hasattr(self, "_compile_lut"):
            self._compile_lut = False

    @property
    def transforms(self) -> list[object]:
        """Return the list of transforms in this segment."""
        return list(self._transforms)

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Any] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Apply the fused lookup table to the image batch.

        Args:
            image: ``(batch_size, channels, height, width)`` input tensor. Floating tensors take the
                interpolation-LUT path (values assumed in ``[0, 1]``); integer tensors take the exact
                256-entry path (values assumed in ``[0, 255]``).
            aux_targets: Optional auxiliary targets, returned unchanged (lookup maps are spatial no-ops).

        Returns:
            Bare mapped ``image`` when ``aux_targets`` is ``None``; ``(image, aux_targets)`` otherwise.

        """
        out = self._forward_float(image) if image.is_floating_point() else self._forward_uint8(image)
        if aux_targets is None:
            return out
        return out, aux_targets

    def _sample_lut_params(self, image: Tensor, transform: object) -> tuple[dict[str, Tensor], Tensor]:
        """Sample one lookup transform's parameters and its per-sample application mask."""
        batch_size, channels, height, width = image.shape
        device = image.device
        prob = _transform_prob(transform)
        if _shares_randomness_across_batch(self._adapter, transform, self.randomness):
            active = (torch.rand((), device=device) < prob).expand(batch_size)
        else:
            active = torch.rand(batch_size, device=device) < prob
        params = _sample_transform_params(
            self._adapter,
            transform,
            (batch_size, channels, height, width),
            device,
            self.randomness,
        )
        return params, active

    def _is_runtime_lut(self, transform: object) -> bool:
        """Return whether ``transform`` needs an image-dependent lookup table."""
        checker = getattr(self._adapter, "is_runtime_lut", None)
        return bool(checker(transform)) if callable(checker) else False

    def _apply_lut(
        self,
        kind: str,
        image: Tensor,
        table: Tensor,
        num_levels: int | None = None,
    ) -> Tensor:
        """Apply an eager or compiled LUT core after table construction completes."""
        fn = _compiled_lut_fn(kind) if self._compile_lut and image.device.type != "cpu" else _lut_fn(kind)
        if num_levels is not None:
            return fn(image, table, num_levels)
        return fn(image, table)

    def _runtime_table(self, transform: object, params: dict[str, Tensor], image: Tensor) -> Tensor | None:
        """Build an image-dependent normalized byte table through the active adapter."""
        builder = getattr(self._adapter, "build_runtime_lut", None)
        if not callable(builder):
            return None
        try:
            table = cast(Tensor, builder(transform, params, image))
        except NotImplementedError:
            return None
        return table.to(device=image.device, dtype=torch.float32)

    def _compose_grid(self, image: Tensor, grid: Tensor, *, snap_levels: int | None) -> Tensor | None:
        """Thread ``grid`` through every op to build the composed per-channel table.

        Args:
            image: The input image (for batch size, device, and param sampling shape).
            grid: ``(batch_size, channels, num_points)`` starting intensities (a uniform domain grid).
            snap_levels: When set, snap the running values to this many byte levels after each op
                (the exact integer path); ``None`` keeps full-precision values (the float path).

        Returns:
            The composed ``(batch_size, channels, num_points)`` table, or ``None`` if any op is not
            lookup-fusible at forward time (the caller then falls back to sequential application).

        """
        composed = grid
        for tfm in self._transforms:
            if self._is_runtime_lut(tfm):
                return None
            params, active = self._sample_lut_params(image, tfm)
            try:
                mapped = self._adapter.build_lut(tfm, params, composed)
            except NotImplementedError:
                return None
            mapped = mapped.to(device=composed.device, dtype=composed.dtype)
            composed = torch.where(active[:, None, None], mapped, composed)
            if snap_levels is not None:
                composed = (composed * (snap_levels - 1)).round().clamp(0.0, snap_levels - 1) / (snap_levels - 1)
        return composed

    def _forward_float(self, image: Tensor) -> Tensor:
        """Apply the interpolation-LUT path to a floating image (values in ``[0, 1]``)."""
        batch_size, channels, _height, _width = image.shape
        num_levels = self.num_levels
        grid = (
            torch
            .linspace(0.0, 1.0, num_levels, device=image.device, dtype=torch.float32)
            .view(1, 1, num_levels)
            .expand(batch_size, channels, num_levels)
            .contiguous()
        )
        out = image
        composed = grid
        for tfm in self._transforms:
            params, active = self._sample_lut_params(image, tfm)
            if self._is_runtime_lut(tfm):
                out = self._apply_lut("interp", out, composed, num_levels)
                composed = grid
                table = self._runtime_table(tfm, params, out)
                if table is None:
                    return self._fallback_nonfused(image)
                identity = torch.linspace(0.0, 1.0, 256, device=image.device).view(1, 1, 256)
                table = torch.where(active[:, None, None], table, identity)
                mapped = self._apply_lut("runtime_float", out, table)
                preserve = params.get("_runtime_preserve_float")
                if preserve is not None:
                    mapped = torch.where(preserve[:, :, None, None], out, mapped)
                out = torch.where(active[:, None, None, None], mapped, out)
                continue
            try:
                mapped = self._adapter.build_lut(tfm, params, composed).to(dtype=composed.dtype)
            except NotImplementedError:
                return self._fallback_nonfused(image)
            composed = torch.where(active[:, None, None], mapped, composed)
        return self._apply_lut("interp", out, composed, num_levels)

    def _forward_uint8(self, image: Tensor) -> Tensor:
        """Apply the exact 256-entry path to an integer image (values in ``[0, 255]``)."""
        batch_size, channels, _height, _width = image.shape
        levels = 256
        grid = (
            (torch.arange(levels, device=image.device, dtype=torch.float32) / (levels - 1))
            .view(1, 1, levels)
            .expand(batch_size, channels, levels)
            .contiguous()
        )
        out = image
        composed = grid
        for tfm in self._transforms:
            params, active = self._sample_lut_params(image, tfm)
            if self._is_runtime_lut(tfm):
                out = self._apply_lut("uint8", out, composed)
                composed = grid
                table = self._runtime_table(tfm, params, out)
                if table is None:
                    return self._fallback_nonfused(image)
                identity = torch.linspace(0.0, 1.0, levels, device=image.device).view(1, 1, levels)
                table = torch.where(active[:, None, None], table, identity)
                out = self._apply_lut("uint8", out, table)
                continue
            try:
                mapped = self._adapter.build_lut(tfm, params, composed).to(dtype=composed.dtype)
            except NotImplementedError:
                return self._fallback_nonfused(image)
            composed = torch.where(active[:, None, None], mapped, composed)
            composed = (composed * (levels - 1)).round().clamp(0.0, levels - 1) / (levels - 1)
        return self._apply_lut("uint8", out, composed)

    @staticmethod
    def _apply_uint8(image: Tensor, table: Tensor) -> Tensor:
        """Map an integer image through a normalized 256-entry table exactly."""
        return _apply_uint8_lut(image, table)

    @staticmethod
    def _apply_runtime_float(image: Tensor, table: Tensor) -> Tensor:
        """Apply a byte-domain runtime table with the backend's quantized input domain."""
        return _apply_runtime_float_lut(image, table)

    @staticmethod
    def _apply_interp(image: Tensor, composed: Tensor, num_levels: int) -> Tensor:
        """Map floats by linear interpolation, preserving detected lookup-table jumps sharply."""
        return _apply_interp_lut(image, composed, num_levels)

    def _fallback_nonfused(self, image: Tensor) -> Tensor:
        """Apply every transform sequentially via the adapter (fusion aborted mid-run)."""
        out = image
        for tfm in self._transforms:
            out = self._adapter.call_nonfused(tfm, out)
        return out

    def forward_numpy(self, img_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply the fused lookup table to a uint8 HWC NumPy image (Albumentations native path).

        Builds each op's 256-entry byte map by running the transform on the ``0..255`` byte ramp
        (pointwise ops map the ramp exactly as they would any image), composes the maps by integer
        indexing, and applies the single composed table. Bit-exact against applying the transforms
        sequentially, and per-op probability is honoured natively by each transform call.

        Args:
            img_hwc: ``(H, W, C)`` or ``(H, W)`` uint8 image.

        Returns:
            The mapped image, same shape and dtype as ``img_hwc``.

        """
        arr = img_hwc
        squeeze = arr.ndim == 2
        if squeeze:
            arr = arr[:, :, None]
        num_channels = arr.shape[2]
        ramp = np.arange(256, dtype=np.uint8)
        probe = np.tile(ramp[None, :, None], (1, 1, num_channels))  # (1, 256, C)
        composed = np.tile(ramp[:, None], (1, num_channels)).astype(np.intp)  # (256, C) identity
        for tfm in self._transforms:
            if self._is_runtime_lut(tfm):
                for channel in range(num_channels):
                    arr[..., channel] = composed[arr[..., channel].astype(np.intp), channel]
                arr = tfm(image=arr)["image"]  # type: ignore[operator]
                composed = np.tile(ramp[:, None], (1, num_channels)).astype(np.intp)
                continue
            # Each op maps the byte ramp to its own (256, C) table; compose by integer indexing.
            mapped_probe = tfm(image=probe)["image"]  # type: ignore[operator]
            lut_i = np.asarray(mapped_probe)[0].astype(np.intp)
            composed = np.take_along_axis(lut_i, composed, axis=0)  # composed[i,c] = lut_i[composed[i,c], c]
        out = np.empty_like(arr)
        for channel in range(num_channels):
            out[..., channel] = composed[arr[..., channel].astype(np.intp), channel]
        return out[:, :, 0] if squeeze else out


def _try_build_lut(adapter: TransformAdapter, transform: object) -> bool:
    """Probe whether *adapter* supports ``build_lut`` for *transform*.

    Mirrors :func:`_try_build_color_matrix`: calls the method with an empty param dict and a tiny value grid,
    classifying the outcome. A supported op reads a missing param key and raises ``KeyError`` (treated as supported); an
    unsupported op raises ``NotImplementedError`` / ``AttributeError`` (treated as unsupported). The probe never draws
    backend randomness because the missing param short-circuits before any op is applied.

    """
    runtime_checker = getattr(adapter, "is_runtime_lut", None)
    if callable(runtime_checker) and runtime_checker(transform):
        return callable(getattr(adapter, "build_runtime_lut", None))
    try:
        adapter.build_lut(transform, {}, torch.zeros(1, 1, 2))
        return True
    except (NotImplementedError, AttributeError):
        return False
    except (KeyError, IndexError):
        # Method exists but needs real params to succeed (missing param key).
        return True
    except Exception:
        # Unexpected error: treat as unsupported to avoid silently mis-fusing a broken adapter.
        return False


def _flush_lut(
    transforms: list[object],
    adapter: TransformAdapter,
    segments: list[object],
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    num_levels: int = _DEFAULT_LUT_LEVELS,
    compile_lut: bool = False,
) -> None:
    """Flush a run of ``POINTWISE_LUT`` transforms into segments.

    If the adapter supports ``build_lut`` for **every** transform in the run, they are folded into a
    single :class:`FusedLUTSegment`. Otherwise the transforms fall back to passthrough (appended
    as-is). Mirrors :func:`_flush_color`; mutates ``transforms`` in place (clears it).

    Args:
        transforms: The pending run of ``POINTWISE_LUT`` transforms (cleared in place).
        adapter: Adapter used to probe and build lookup tables.
        segments: Output segment list, appended in place.
        randomness: Batch randomness policy passed to the lookup segment.
        num_levels: Float interpolation-LUT grid resolution forwarded to :class:`FusedLUTSegment`.
        compile_lut: Whether non-CPU lookup applications may use ``torch.compile``.

    """
    if not transforms:
        return
    for tfm in transforms:
        if not _try_build_lut(adapter, tfm):
            segments.extend(transforms)
            # Intentionally clears the caller-owned run buffer.
            transforms.clear()
            return
    lut_transforms = list(transforms)
    segments.append(
        FusedLUTSegment(
            lut_transforms,
            adapter,
            num_levels=num_levels,
            randomness=randomness,
            compile_lut=compile_lut,
        )
    )
    # Intentionally clears the caller-owned run buffer.
    transforms.clear()


class CropResizeSegment(nn.Module):
    """Segment for a single ``CROP_RESIZE_FIXED`` transform.

    Samples the random crop region, builds the forward affine matrix, normalizes it
    via :func:`~fused_transforms.affine.matrix.normalize_matrix_io` (which accounts
    for different input and output spatial dimensions), and applies exactly one
    ``grid_sample`` call at the target ``(H_out, W_out)`` dimensions.

    Unlike :class:`FusedAffineSegment`, the output shape is ``(batch_size, channels, height_out, width_out)``
    which generally differs from the input shape ``(batch_size, channels, height_in, width_in)``.

    .. note::
        Per-sample probability ``prob`` is **not** applied: shape-changing transforms must produce a consistent
        output size for all batch elements, so the crop is always applied.  Use ``prob=1.0`` (the standard default)
        when constructing ``RandomResizedCrop`` transforms.

    .. note::
        Auxiliary targets (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``) are
        warped through the crop affine matrix at the target output size. Masks use nearest-neighbour
        sampling to preserve integer class labels; boxes and keypoints are transformed via the forward
        affine matrix.

    Args:
        transform: A single ``CROP_RESIZE_FIXED`` transform object.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_matrix`` for the transform.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.
        fill: Validated constant written outside the source canvas, in the image's own
            value range, or ``None`` for plain zero padding. Image only; the routed mask
            keeps its zero padding.

    """

    def __init__(
        self,
        transform: object,
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        antialias: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
        mask_fill: MaskFillValue = 0,
        fill: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize ``CropResizeSegment``."""
        super().__init__()
        self.transform = transform
        self.transforms: list[object] = [transform]
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.randomness = randomness
        # Opt-in antialiasing: prefilter/interpolate only when a downscale is
        # aggressive enough to alias (see _ANTIALIAS_SCALE_THRESHOLD). Off by
        # default → output bit-identical to the plain single grid_sample warp.
        self._antialias: bool = antialias
        self.mask_interpolation = mask_interpolation
        self.mask_fill = mask_fill
        self.fill = fill
        self._last_matrix: Tensor | None = None

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the deterministic letterbox matrix from the most recent call, if available."""
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the crop-resize via a single ``grid_sample`` call at target output size.

        Args:
            image: ``(batch_size, channels, height_in, width_in)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside the image.
                Supported keys: ``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``.
                Masks are warped with nearest-neighbour sampling to the target size;
                boxes and keypoints are transformed via the forward affine matrix.

        Returns:
            ``(batch_size, channels, height_out, width_out)`` tensor when ``aux_targets`` is ``None``;
            ``(tensor, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        self._last_matrix = None
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype

        params = _sample_transform_params(
            adapter=self.adapter,
            transform=self.transform,
            input_shape=(batch_size, num_channels, height, width),
            device=device,
            randomness=self.randomness,
        )
        if not (
            torch.all(params["target_h"] == params["target_h"][0])
            and torch.all(params["target_w"] == params["target_w"][0])
        ):
            raise ValueError(
                "CropResizeSegment requires a uniform target size across the batch "
                f"(got target_h={params['target_h'].tolist()}, target_w={params['target_w'].tolist()})"
            )
        target_h = int(params["target_h"][0].item())
        target_w = int(params["target_w"][0].item())

        mtx = self.adapter.build_matrix(self.transform, params, height, width)
        if mtx.shape[0] == 1 and batch_size > 1:
            mtx = mtx.expand(batch_size, -1, -1)
        mtx = mtx.to(device=device, dtype=_matrix_geometry_dtype(dtype))

        if getattr(self.transform, "_coordinate_matrix_recoverable", False):
            self._last_matrix = mtx.to(dtype=_matrix_public_dtype(dtype)).detach().clone()
            _set_current_call_matrix(self._last_matrix)

        # The helper leaves safe rows untouched and computes their scales only once.
        if self._antialias:
            image = _maybe_antialias_prefilter(image, mtx, enabled=True)

        mtx_inv = inv3x3(mtx)
        mtx_norm = normalize_matrix_io(mtx_inv, height, width, target_h, target_w)

        grid = F.affine_grid(
            mtx_norm[:, :2, :],
            [batch_size, num_channels, target_h, target_w],
            align_corners=True,
        )
        fill_value = _fill_tensor(self.fill, num_channels, device, dtype)
        out = F.grid_sample(
            image if fill_value is None else image - fill_value,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )
        if fill_value is not None:
            out = out + fill_value

        if aux_targets:
            from fused_transforms.targets import (
                transform_bbox_xywh,
                transform_bbox_xyxy,
                transform_keypoints,
                transform_mask,
                transform_rboxes,
            )

            for key in list(aux_targets.keys()):
                val = aux_targets[key]
                if key == "mask":
                    aux_targets[key] = transform_mask(val, grid, mode=self.mask_interpolation, fill=self.mask_fill)
                    continue
                if key == "bbox_xyxy":
                    aux_targets[key] = transform_bbox_xyxy(val, mtx)
                    continue
                if key == "bbox_xywh":
                    aux_targets[key] = transform_bbox_xywh(val, mtx)
                    continue
                if key == "keypoints":
                    aux_targets[key] = transform_keypoints(val, mtx)
                    continue
                if key == "rboxes":
                    aux_targets[key] = transform_rboxes(val, mtx)

        if not _has_aux:
            return out
        if aux_targets is None:
            raise RuntimeError("internal error: aux_targets is None in return branch")
        return out, aux_targets


# ---------------------------------------------------------------------------
# Reorder helpers
# ---------------------------------------------------------------------------


def reorder_pointwise(
    transforms: list[object],
    adapter: TransformAdapter,
) -> list[object]:
    """Reorder transforms so POINTWISE ops are pushed after geometric chains.

    Walks the transform list left to right.  Within each stretch between
    ``SPATIAL_KERNEL`` barriers, geometric ops (``GEOMETRIC_INTERP`` and
    ``GEOMETRIC_EXACT``) are kept in order, while ``POINTWISE`` ops are
    deferred and flushed after the geometric group.  ``POINTWISE`` ops
    never move across a ``SPATIAL_KERNEL`` barrier.

    Args:
        transforms: List of transform objects to reorder.
        adapter: A ``TransformAdapter`` used for category lookup on each transform.

    Returns:
        New list containing the same transforms, possibly reordered so that
        ``POINTWISE`` ops sit after geometric runs within each
        barrier-bounded stretch.

    Examples:
        Given a pipeline ``[Rotate, Brightness, HFlip]`` where ``Brightness``
        is ``POINTWISE`` and ``Rotate`` / ``HFlip`` are geometric, the
        ``Brightness`` is pushed after the geometric group:

        Input order:  ``[Rotate, Brightness, HFlip]``
        Output order: ``[Rotate, HFlip, Brightness]``

        Using stub objects (the KorniaAdapter registry does not include any
        POINTWISE transforms in v0.2):

    ```pycon
    >>> from fused_transforms.affine.segment import reorder_pointwise
    >>> from fused_transforms.types import TransformCategory
    >>> class _StubAdapter:
    ...     def category(self, transform):
    ...         return transform._cat
    ...
    >>> class _TransformStub:
    ...     def __init__(self, cat):
    ...         self._cat = cat
    ...
    >>> adapter = _StubAdapter()
    >>> geo = _TransformStub(TransformCategory.GEOMETRIC_INTERP)
    >>> pw  = _TransformStub(TransformCategory.POINTWISE)
    >>> result = reorder_pointwise([geo, pw, geo], adapter)
    >>> [transform._cat.name for transform in result]
    ['GEOMETRIC_INTERP', 'GEOMETRIC_INTERP', 'POINTWISE']

    ```

    """
    geometric = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT, TransformCategory.PROJECTIVE}

    result: list[object] = []
    geo_buf: list[object] = []
    pw_buf: list[object] = []

    def _flush() -> None:
        result.extend(geo_buf)
        result.extend(pw_buf)
        geo_buf.clear()
        pw_buf.clear()

    _reorderable = {
        TransformCategory.POINTWISE,
        TransformCategory.POINTWISE_LINEAR,
        TransformCategory.POINTWISE_LUT,
    }

    for tfm in transforms:
        cat = adapter.category(tfm)
        if cat in _reorderable:
            pw_buf.append(tfm)
            continue
        if cat in geometric:
            geo_buf.append(tfm)
            continue

        # SPATIAL_KERNEL barrier: flush current stretch, then emit the barrier
        _flush()
        result.append(tfm)

    _flush()
    return result


def reorder_aggressive(
    transforms: list[object],
    adapter: TransformAdapter,
) -> list[object]:
    """Reorder transforms aggressively -- bubble-sort POINTWISE ops after geometric chains.

    Applies the POINTWISE reorder algorithm iteratively until the list stabilizes
    (convergence guarantee). ``SPATIAL_KERNEL`` barriers are never crossed.

    For typical pipelines the result is identical to a single :func:`reorder_pointwise`
    pass; the multi-pass variant provides a convergence guarantee for pathological
    orderings.

    Args:
        transforms: List of transform objects to reorder.
        adapter: TransformAdapter for category lookup.

    Returns:
        Reordered list with all POINTWISE ops placed after geometric runs within
        each SPATIAL_KERNEL-bounded stretch.

    """
    current = transforms
    for _ in range(len(transforms)):  # max n iterations
        reordered = reorder_pointwise(current, adapter)
        if len(reordered) == len(current) and all(
            item_a is item_b for item_a, item_b in zip(reordered, current, strict=True)
        ):
            break
        current = reordered
    return current


def _can_commute_gaussian_blur(geometric: list[object], adapter: TransformAdapter) -> bool:
    """Return whether configured geometric transforms cannot downscale a Gaussian blur."""
    for transform in geometric:
        category = adapter.category(transform)
        if category is TransformCategory.GEOMETRIC_EXACT:
            return False
        if category is not TransformCategory.GEOMETRIC_INTERP:
            return False
        generator = getattr(transform, "_param_generator", transform)
        scale = getattr(generator, "scale", getattr(transform, "scale", None))
        if isinstance(scale, Sequence) and min(float(value) for value in scale) < 1.0:
            return False
    return True


def _transform_border_mode(adapter: TransformAdapter, transform: object) -> PaddingModeStr | None:
    """Return a transform's compatible border mode, treating unknown modes as opaque."""
    accessor = getattr(adapter, "border_mode", None)
    if accessor is None:
        return "zeros"
    mode = accessor(transform)
    if mode in ("zeros", "border", "reflection"):
        return cast("PaddingModeStr", mode)
    return None


def _partition_border_modes(
    transforms: list[object],
    adapter: TransformAdapter,
) -> list[tuple[list[object], PaddingModeStr | None, str | None]]:
    """Partition a geometric run into compatible constant-border sub-runs."""
    partitions: list[tuple[list[object], PaddingModeStr | None, str | None]] = []
    current: list[object] = []
    current_mode: PaddingModeStr | None = None
    current_split_reason: str | None = None

    for transform in transforms:
        mode = _transform_border_mode(adapter, transform)
        if mode is None:
            if current:
                partitions.append((current, current_mode, current_split_reason))
                current = []
            partitions.append(([transform], None, "opaque_border_mode"))
            current_mode = None
            current_split_reason = None
            continue
        if not current:
            current = [transform]
            current_mode = mode
            current_split_reason = None
            continue
        if mode == current_mode:
            current.append(transform)
            continue
        partitions.append((current, current_mode, current_split_reason))
        current = [transform]
        current_mode = mode
        current_split_reason = "border_mode_change"

    if current:
        partitions.append((current, current_mode, current_split_reason))
    return partitions


def build_segments(
    transforms: list[object],
    adapter: TransformAdapter,
    interpolation: InterpolationStr | None = None,
    padding_mode: PaddingModeStr | None = None,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    *,
    use_numpy: bool = False,
    route_coords_via_grid: bool = False,
    route_crop_aux: bool = False,
    execution: ExecutionStr = "cv2",
    compile_warp: bool = False,
    per_transform_padding: bool = False,
    antialias: bool = False,
    clip_policy: ClipPolicyStr = "final",
    mask_interpolation: MaskInterpolationStr = "nearest",
    mask_fill: MaskFillValue = 0,
    generator: torch.Generator | None = None,
    fill: tuple[float, ...] | None = None,
    keypoint_flip_index: tuple[int, ...] | None = None,
) -> list[object]:
    """Split a transform list into fused segments and passthrough transforms.

    Walks the transforms left to right and groups consecutive geometric transforms (``GEOMETRIC_INTERP`` or
    ``GEOMETRIC_EXACT``) into a single segment.  Any ``SPATIAL_KERNEL``, ``POINTWISE``, or ``POINTWISE_LINEAR``
    transform breaks the current geometric group and is returned as-is.

    After grouping, each accumulated geometric run is classified:

    - **EXACT-only** - if the run contains *only* ``GEOMETRIC_EXACT`` ops
      (e.g. flips, 90-degree rotations, transpose-like discrete ops), it
      becomes an :class:`ExactAffineSegment` that uses adapter-provided exact
      image operations with zero interpolation error.
    - **Mixed / INTERP** - if any op in the run is ``GEOMETRIC_INTERP``, the
      whole run becomes a :class:`FusedAffineSegment` that composes matrices
      and applies one ``grid_sample`` call.

    When ``ReorderPolicy.POINTWISE`` is active in
    :class:`~fused_transforms.compose.FusedCompose`, ``reorder_pointwise``
    is called first to bubble pointwise ops out of geometric chains, and
    ``build_segments`` then classifies the reordered list.

    Args:
        transforms: List of transform objects (already reordered if a reorder policy applies).
        adapter: A ``TransformAdapter`` for category lookup and matrix building.
        interpolation: Interpolation mode override forwarded to each
            :class:`FusedAffineSegment` (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
        padding_mode: Padding mode override forwarded to each
            :class:`FusedAffineSegment` (``"zeros"``, ``"border"``, ``"reflection"``).
        use_numpy: When ``True``, produce :class:`AlbuFusedAffineSegment` instances
            (Albumentations/scipy backend) instead of the PyTorch
            :class:`FusedAffineSegment`. Used for the Albumentations backend.
        randomness: Batch randomness policy for fused PyTorch segments.
        route_coords_via_grid: When ``True`` (set by the caller when the pipeline
            carries box/keypoint auxiliary targets), route an all-exact geometric run
            through :class:`FusedAffineSegment` instead of :class:`ExactAffineSegment`.
            An all-exact run always composes to a D4 element, so the fused segment's
            exact-dispatch still applies it losslessly while routing boxes/keypoints
            through the composed pixel matrix — avoiding the ``ExactAffineSegment``
            box/keypoint limitation without an interpolation penalty. Torch path only.
        route_crop_aux: When ``True`` (set by the caller when the pipeline carries any
            auxiliary target), emit a :class:`CropResizeSegment` for a ``CROP_RESIZE_FIXED``
            op on the ``use_numpy`` (Albumentations) path instead of an image-only
            passthrough, so masks/boxes/keypoints route to the crop's output size. Without
            it the numpy-path crop resizes the image only, silently desyncing aux targets.
            No effect on the PyTorch backends (they always build ``CropResizeSegment``).
        execution: Execution strategy forwarded to the Albumentations fused segments
            (``use_numpy=True`` path). ``"cv2"`` (default) warps each sample with
            OpenCV; ``"torch"`` opts into one batched ``grid_sample`` for the whole
            batch. Ignored for the PyTorch backends.
        compile_warp: When ``True`` (and torch is new enough), the torch warp core
            (matrix normalize -> ``affine_grid`` -> ``grid_sample``) of each fused
            geometric or projective segment runs under ``torch.compile``. Off by
            default and a no-op on CPU or older torch — the eager output is unchanged.
            Crop-resize segments (:class:`CropResizeSegment`, :class:`_FusedGeoCropSegment`)
            build their warp inline and do not honor this flag.
        per_transform_padding: When ``True``, partition geometric runs by each
            transform's compatible border mode instead of applying ``padding_mode``
            to the entire run. Opaque modes remain native passthrough boundaries.
        antialias: When ``True``, crop-resize segments prefilter the input before an
            aggressive downscale so the single warp does not alias. Off by default and
            a no-op unless the scale drops past the threshold — the output is unchanged.
            Requires the optional kornia dependency at construction.
        clip_policy: Clamp policy forwarded to each :class:`FusedColorSegment`.
            ``"final"`` (default) fuses the color chain into one matmul and clamps once;
            ``"per_op_parity"`` clamps at each op that could leave ``[0, 1]``.
        mask_interpolation: Sampling mode for routed masks. ``"nearest"`` keeps
            hard integer labels; ``"bilinear"`` supports float soft masks.
        mask_fill: Scalar border value for routed masks, independent of image ``fill``.
        fill: Validated constant border value written outside the source canvas of every
            resampling segment (image only; routed masks keep zero padding), or ``None``
            for a zero border.
        keypoint_flip_index: Validated keypoint pair permutation applied wherever a segment's
            composed transform reverses orientation, or ``None`` to leave the keypoint axis
            in input order.
        generator: Caller-owned ``torch.Generator`` forwarded to every torch-side segment
            that owns a draw (probability gates and direct parameter sampling), or ``None``
            for the global torch stream. Only the direct-parameter adapter can honour it;
            backend adapters are rejected at pipeline construction.

    Returns:
        Flat list where each element is a :class:`FusedAffineSegment`
        (mixed/INTERP geometric run), an :class:`ExactAffineSegment`
        (EXACT-only geometric run; auxiliary targets remain flip-only),
        a :class:`CropResizeSegment` (``CROP_RESIZE_FIXED`` op on the
        PyTorch path), a :class:`FusedColorSegment` (``POINTWISE_LINEAR``
        run where the adapter supports ``build_color_matrix``), or the
        original transform object (passthrough for ``SPATIAL_KERNEL``,
        ``POINTWISE``, ``CROP_RESIZE_FIXED`` on the numpy path, and
        unsupported ``POINTWISE_LINEAR`` transforms).

    """
    if antialias and not backend_available("kornia"):
        raise ImportError("antialias=True requires the optional kornia dependency, installed and importable")

    fusible = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT}
    projective_cat = TransformCategory.PROJECTIVE
    pointwise_linear_cat = TransformCategory.POINTWISE_LINEAR
    pointwise_lut_cat = TransformCategory.POINTWISE_LUT
    crop_resize_cat = TransformCategory.CROP_RESIZE_FIXED
    spatial_linear_cat = TransformCategory.SPATIAL_LINEAR

    segments: list[object] = []
    current_geo: list[object] = []
    current_proj: list[object] = []
    current_color: list[object] = []
    current_lut: list[object] = []

    def _append_segment(segment: object, split_reason: str | None) -> None:
        """Append a segment and retain its construction-time split reason."""
        if split_reason is not None:
            cast("Any", segment)._fusion_split_reason = split_reason
        segments.append(segment)

    def _append_geo_segment(
        geo_transforms: list[object],
        geo_padding_mode: PaddingModeStr | None,
        split_reason: str | None,
    ) -> None:
        """Build one geometric segment using a single compatible border mode."""
        has_interp = any(
            adapter.category(transform) == TransformCategory.GEOMETRIC_INTERP for transform in geo_transforms
        )
        if use_numpy:
            if has_interp:
                _append_segment(
                    AlbuFusedAffineSegment(
                        transforms=geo_transforms,
                        adapter=adapter,
                        interpolation=interpolation,
                        padding_mode=geo_padding_mode,
                        execution=execution,
                        mask_interpolation=mask_interpolation,
                        mask_fill=mask_fill,
                        fill=fill,
                        keypoint_flip_index=keypoint_flip_index,
                    ),
                    split_reason,
                )
            else:
                _append_segment(
                    ExactAffineSegment(
                        geo_transforms,
                        adapter,
                        randomness=randomness,
                        generator=generator,
                        keypoint_flip_index=keypoint_flip_index,
                    ),
                    split_reason,
                )
            return

        if has_interp or route_coords_via_grid:
            _append_segment(
                FusedAffineSegment(
                    transforms=geo_transforms,
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=geo_padding_mode,
                    # geo_padding_mode may be the transform's own mode (per-transform policy); the caller's
                    # override is build_segments' padding_mode, which that policy passes as None.
                    render_overridden=fill is not None or interpolation is not None or padding_mode is not None,
                    randomness=randomness,
                    compile_warp=compile_warp,
                    mask_interpolation=mask_interpolation,
                    mask_fill=mask_fill,
                    generator=generator,
                    fill=fill,
                    keypoint_flip_index=keypoint_flip_index,
                ),
                split_reason,
            )
            return

        _append_segment(
            ExactAffineSegment(
                geo_transforms,
                adapter,
                randomness=randomness,
                generator=generator,
                keypoint_flip_index=keypoint_flip_index,
            ),
            split_reason,
        )

    def _flush_geo() -> None:
        if not current_geo:
            return

        if per_transform_padding:
            for group, group_mode, split_reason in _partition_border_modes(current_geo, adapter):
                if group_mode is None:
                    transform = group[0]
                    warnings.warn(
                        f"{type(transform).__name__} has an opaque border mode; keeping it as a native boundary.",
                        UserWarning,
                        stacklevel=3,
                    )
                    segments.append(_OpaqueBorderModeTransform(transform, split_reason or "opaque_border_mode"))
                else:
                    _append_geo_segment(group, group_mode, split_reason)
        else:
            _append_geo_segment(list(current_geo), padding_mode, None)
        current_geo.clear()

    def _append_projective_segment(
        projective_transforms: list[object],
        projective_padding_mode: PaddingModeStr | None,
        split_reason: str | None,
    ) -> None:
        """Build one projective segment using a single compatible border mode."""
        if use_numpy:
            _append_segment(
                AlbuProjectiveSegment(
                    transforms=projective_transforms,
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=projective_padding_mode,
                    execution=execution,
                    mask_interpolation=mask_interpolation,
                    mask_fill=mask_fill,
                    fill=fill,
                ),
                split_reason,
            )
            return
        _append_segment(
            ProjectiveSegment(
                transforms=projective_transforms,
                adapter=adapter,
                interpolation=interpolation,
                padding_mode=projective_padding_mode,
                randomness=randomness,
                compile_warp=compile_warp,
                mask_interpolation=mask_interpolation,
                mask_fill=mask_fill,
                fill=fill,
            ),
            split_reason,
        )

    def _flush_proj() -> None:
        if not current_proj:
            return
        if per_transform_padding:
            for group, group_mode, split_reason in _partition_border_modes(current_proj, adapter):
                if group_mode is None:
                    transform = group[0]
                    warnings.warn(
                        f"{type(transform).__name__} has an opaque border mode; keeping it as a native boundary.",
                        UserWarning,
                        stacklevel=3,
                    )
                    segments.append(_OpaqueBorderModeTransform(transform, split_reason or "opaque_border_mode"))
                else:
                    _append_projective_segment(group, group_mode, split_reason)
        else:
            _append_projective_segment(list(current_proj), padding_mode, None)
        current_proj.clear()

    consumed_linear_indices: set[int] = set()
    for index, transform in enumerate(transforms):
        if index in consumed_linear_indices:
            continue
        category = adapter.category(transform)
        if category in fusible:
            _flush_proj()  # flush any pending projective
            _flush_color(current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator)
            _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
            current_geo.append(transform)
            continue
        if category == projective_cat:
            _flush_geo()  # flush any pending affine
            _flush_color(current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator)
            _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
            current_proj.append(transform)
            continue
        if category == pointwise_linear_cat:
            _flush_geo()
            _flush_proj()
            _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
            current_color.append(transform)
            continue
        if category == pointwise_lut_cat:
            # Non-linear per-channel scalar map (gamma/solarize/posterize): starts/extends a lookup
            # run. Colour (matrix) and lookup (table) runs are mutually exclusive, so flush colour.
            _flush_geo()
            _flush_proj()
            _flush_color(current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator)
            current_lut.append(transform)
            continue
        if category == crop_resize_cat:
            # CROP_RESIZE_FIXED. On the torch path, fuse into the immediately
            # preceding fusible geometric run: compose M_crop @ M_geo and
            # apply ONE grid_sample at the crop's target size instead of two.
            # No pending geo run → emit a standalone CropResizeSegment. Projective
            # and color runs still flush (crop only fuses with affine geo runs).
            # The numpy/albumentations path emits the transform as a passthrough,
            # except when the pipeline carries auxiliary targets: a passthrough crop
            # resizes the image only and silently desyncs masks/boxes/keypoints, so a
            # CropResizeSegment (which the Albumentations adapter fully supports for
            # CROP_RESIZE_FIXED) is emitted instead to route aux to the output size.
            #
            # The image-only passthrough exists to keep the crop bit-exact: it runs
            # Albumentations' own crop rather than our resampler. That trade only pays while
            # the rest of the chain is bit-exact too. Under execution="torch" the image is
            # already warped by grid_sample, so a native crop buys parity the chain has
            # given up -- and still costs a device-to-host round trip for the whole batch
            # per call, measured at 2.95x against the cv2 engine on an L4 at batch 64. So
            # "torch" routes the crop through the segment as well.
            #
            # "auto" is deliberately left on the passthrough: it resolves per call, the
            # device is unknown when segments are built, and its documented common case is
            # host data on cv2, where the native crop is the right choice.
            #
            # This is a win at training batch sizes and a loss at small ones. Measured on
            # MPS against the passthrough it runs 0.83x at batch 8, 1.17x at 32 and 1.39x
            # at 64, so the crossover sits somewhere below 32. The remaining batch-8
            # regression is the segment paying its own per-call matrix assembly: batching
            # that chain's transfers (see _matrix_sample_device) moved batch 8 from 0.48x to
            # 0.83x, which narrowed the regression without closing it.
            _flush_proj()
            _flush_color(current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator)
            _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
            keep_crop_native = use_numpy and not route_crop_aux and execution != "torch"
            if per_transform_padding:
                _flush_geo()
                crop_padding_mode = _transform_border_mode(adapter, transform)
                if crop_padding_mode is None:
                    warnings.warn(
                        f"{type(transform).__name__} has an opaque border mode; keeping it as a native boundary.",
                        UserWarning,
                        stacklevel=3,
                    )
                    segments.append(_OpaqueBorderModeTransform(transform))
                elif keep_crop_native:
                    segments.append(transform)
                else:
                    segments.append(
                        CropResizeSegment(
                            transform=transform,
                            adapter=adapter,
                            interpolation=interpolation,
                            padding_mode=crop_padding_mode,
                            randomness=randomness,
                            antialias=antialias,
                            mask_interpolation=mask_interpolation,
                            mask_fill=mask_fill,
                            fill=fill,
                        )
                    )
                continue
            if use_numpy:
                _flush_geo()
                if keep_crop_native:
                    segments.append(transform)
                else:
                    segments.append(
                        CropResizeSegment(
                            transform=transform,
                            adapter=adapter,
                            interpolation=interpolation,
                            padding_mode=padding_mode,
                            randomness=randomness,
                            antialias=antialias,
                            mask_interpolation=mask_interpolation,
                            mask_fill=mask_fill,
                            fill=fill,
                        )
                    )
                continue
            if current_geo:
                segments.append(
                    _FusedGeoCropSegment(
                        geo_transforms=list(current_geo),
                        crop_transform=transform,
                        adapter=adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                        randomness=randomness,
                        compile_warp=compile_warp,
                        antialias=antialias,
                        mask_interpolation=mask_interpolation,
                        mask_fill=mask_fill,
                        generator=generator,
                        fill=fill,
                        keypoint_flip_index=keypoint_flip_index,
                    )
                )
                current_geo.clear()
            else:
                segments.append(
                    CropResizeSegment(
                        transform=transform,
                        adapter=adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                        randomness=randomness,
                        antialias=antialias,
                        mask_interpolation=mask_interpolation,
                        mask_fill=mask_fill,
                        fill=fill,
                    )
                )
            continue
        if category == spatial_linear_cat:
            blur_end = index + 1
            while blur_end < len(transforms) and adapter.category(transforms[blur_end]) == spatial_linear_cat:
                blur_end += 1
            affine_end = blur_end
            while affine_end < len(transforms) and adapter.category(transforms[affine_end]) in fusible:
                affine_end += 1
            following_geo = transforms[blur_end:affine_end]
            if (
                following_geo
                and not use_numpy
                and not per_transform_padding
                and _can_commute_gaussian_blur(following_geo, adapter)
            ):
                _flush_proj()
                _flush_color(
                    current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator
                )
                _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
                blur_transforms = transforms[index:blur_end]
                segments.append(
                    FusedGaussianBlurSegment(
                        [*current_geo, *blur_transforms, *following_geo],
                        blur_transforms,
                        list(current_geo),
                        following_geo,
                        adapter,
                        interpolation,
                        padding_mode,
                        randomness,
                        mask_interpolation=mask_interpolation,
                        mask_fill=mask_fill,
                    )
                )
                current_geo.clear()
                consumed_linear_indices.update(range(index + 1, affine_end))
                continue
            if blur_end > index + 1:
                _flush_geo()
                _flush_proj()
                _flush_color(
                    current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator
                )
                _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
                blur_transforms = transforms[index:blur_end]
                segments.append(
                    FusedGaussianBlurSegment(
                        blur_transforms,
                        blur_transforms,
                        [],
                        [],
                        adapter,
                        interpolation,
                        padding_mode,
                        randomness,
                        mask_interpolation=mask_interpolation,
                        mask_fill=mask_fill,
                    )
                )
                consumed_linear_indices.update(range(index + 1, blur_end))
                continue
        # SPATIAL_KERNEL / POINTWISE barrier (blur/noise, saturation/hue, equalize): flush all
        _flush_geo()
        _flush_proj()
        _flush_color(current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator)
        _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
        segments.append(transform)

    _flush_geo()
    _flush_proj()
    _flush_color(current_color, adapter, segments, randomness, clip_policy, compile_warp, generator=generator)
    _flush_lut(current_lut, adapter, segments, randomness, compile_lut=compile_warp)
    return segments


# ---------------------------------------------------------------------------
# Private helpers for ExactAffineSegment auxiliary-target flipping
# ---------------------------------------------------------------------------


def _flip_bbox_xyxy(
    boxes: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip bounding boxes (batch_size, num_boxes, 4) xyxy format using direct coordinate arithmetic.

    AABBs use pixel-edge extents, so HFlip is ``coord_x' = width - coord_x``
    and VFlip is ``coord_y' = height - coord_y``; each swaps its extent ends.

    """
    box_x1, box_y1, box_x2, box_y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    if is_hflip:
        new_x1 = width - box_x2
        new_x2 = width - box_x1
        box_x1, box_x2 = new_x1, new_x2

    if is_vflip:
        new_y1 = height - box_y2
        new_y2 = height - box_y1
        box_y1, box_y2 = new_y1, new_y2

    flipped = torch.stack([box_x1, box_y1, box_x2, box_y2], dim=-1)

    # active shape: (batch_size,) -> (batch_size, 1, 1) for broadcasting with (batch_size, num_boxes, 4)
    mask = active[:, None, None]
    return torch.where(mask, flipped, boxes)


def _flip_rboxes(
    rboxes: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Mirror ``(batch_size, num_boxes, 5)`` rotated boxes on the exact-flip path.

    Centres reflect about the same axes the image flip uses (``(width - 1) / 2``,
    ``(height - 1) / 2``) and the extents are unchanged, a mirror being an isometry. The
    angle follows the direction vector: a horizontal mirror sends ``(cos t, sin t)`` to
    ``(-cos t, sin t)``, the direction of ``pi - t``; a vertical one sends it to
    ``(cos t, -sin t)``, the direction of ``-t``. Both flips together give ``t - pi``, a
    half turn, under which a rectangle is invariant. These are the same values fitting the
    mirrored corners yields, so this path and the fused-matrix path agree parameter for
    parameter rather than only geometrically.

    """
    flipped = rboxes.clone()
    if is_hflip:
        flipped[..., 0] = (width - 1) - rboxes[..., 0]
        flipped[..., 4] = math.pi - flipped[..., 4]
    if is_vflip:
        flipped[..., 1] = (height - 1) - rboxes[..., 1]
        flipped[..., 4] = -flipped[..., 4]
    return torch.where(active[:, None, None], flipped, rboxes)


def _route_keypoints(keypoints: Tensor, mtx: Tensor, flip_index: tuple[int, ...] | None) -> Tensor:
    """Transport keypoints by ``mtx`` and swap the caller's pairs where the map mirrors.

    The swap is decided by the sign of the composed matrix's determinant, never by whether a
    flip transform appears in the pipeline: after fusion the mirror is part of a larger
    matrix and is no longer visible as a discrete op, and two mirrors compose back to a
    rotation that must *not* swap.

    """
    from fused_transforms.targets import orientation_reversed, permute_keypoint_pairs, transform_keypoints

    warped = transform_keypoints(keypoints, mtx)
    if flip_index is None:
        return warped
    reversed_mask = orientation_reversed(mtx)
    if reversed_mask.shape[0] != warped.shape[0]:
        reversed_mask = reversed_mask.expand(warped.shape[0])
    return permute_keypoint_pairs(warped, _flip_index_tensor(flip_index, keypoints.device), reversed_mask)


def _flip_index_tensor(flip_index: tuple[int, ...], device: torch.device) -> Tensor:
    """Return the caller's keypoint pair permutation as an index tensor on ``device``."""
    return torch.tensor(flip_index, device=device, dtype=torch.int64)


def _flip_keypoints(
    keypoints: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip keypoints (batch_size, num_points, 2) using direct coordinate arithmetic.

    HFlip: ``coord_x' = width - 1 - coord_x``.
    VFlip: ``coord_y' = height - 1 - coord_y``.

    """
    flipped = keypoints.clone()
    if is_hflip:
        flipped[..., 0] = (width - 1) - keypoints[..., 0]
    if is_vflip:
        flipped[..., 1] = (height - 1) - keypoints[..., 1]

    mask = active[:, None, None]
    return torch.where(mask, flipped, keypoints)


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (batch_size, num_boxes, 4) boxes from xywh to xyxy format."""
    box_left, box_top, box_width, box_height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([box_left, box_top, box_left + box_width, box_top + box_height], dim=-1)


def _xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert (batch_size, num_boxes, 4) boxes from xyxy to xywh format."""
    box_x1, box_y1, box_x2, box_y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1], dim=-1)
