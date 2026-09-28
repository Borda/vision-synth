"""Type definitions for the vision-synth library."""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

#: Whether an adapter samples one parameter set per image (``"per_sample"``) or one per batch (``"per_batch"``).
SamplingSemantics = Literal["per_sample", "per_batch"]


class TransformCategory(Enum):
    """Category of an augmentation transform for fusion classification.

    Attributes:
        GEOMETRIC_INTERP: Fusible geometric op requiring interpolation (rotate, scale, shear).
        GEOMETRIC_EXACT: Fusible only when INTERP is present; lossless alone (flip, 90-deg rot).
        POINTWISE: Reorderable per-pixel op; not fusible (color jitter, normalize).
        SPATIAL_KERNEL: Barrier; not fusible and not reorderable (blur, noise, erase).
        PROJECTIVE: Fusible projective (perspective) op requiring full 3x3 homography.
        POINTWISE_LINEAR: Reorderable per-pixel *linear* op; self-fusible as 4x4 color-space
            affine matrix (brightness, contrast, channel mix).  Consecutive runs are fused into
            a single ``FusedColorSegment`` via ``build_color_matrix``; adapters that do not
            support ``build_color_matrix`` for a given transform fall back to passthrough.
        POINTWISE_LUT: Reorderable per-pixel *non-linear scalar* op whose per-channel intensity
            map is a pure lookup (gamma, solarize, posterize).  Consecutive runs are composed into
            a single per-channel lookup table via ``build_lut`` and applied once by
            :class:`~fused_transforms.affine.segment.FusedLUTSegment`; adapters that do not
            support ``build_lut`` for a given transform fall back to passthrough.  Distinct from
            ``POINTWISE_LINEAR`` (which composes into a colour *matrix*) and from ``POINTWISE``
            (cross-channel ops such as saturation/hue that never fuse into a lookup table).
        CROP_RESIZE_FIXED: Fixed-output-size crop followed by resize.  Each op produces a
            ``CropResizeSegment`` with a single ``grid_sample`` call at the target ``(H, W)``
            dimensions.  The output shape differs from the input shape.
        SPATIAL_LINEAR: A Gaussian blur that can fold with adjacent Gaussian blurs and,
            after a safe axis-aligned upscale, move to the end of an affine chain.

    """

    GEOMETRIC_INTERP = "geometric_interp"
    GEOMETRIC_EXACT = "geometric_exact"
    POINTWISE = "pointwise"
    SPATIAL_KERNEL = "spatial_kernel"
    PROJECTIVE = "projective"
    POINTWISE_LINEAR = "pointwise_linear"
    POINTWISE_LUT = "pointwise_lut"
    CROP_RESIZE_FIXED = "crop_resize_fixed"
    SPATIAL_LINEAR = "spatial_linear"


#: Class names of passthrough transforms that apply a per-pixel *coordinate*
#: displacement (a spatially-varying warp), as opposed to a kernel/pointwise op
#: that leaves geometry unchanged. These ops move image content to new positions,
#: so auxiliary targets (masks, boxes, keypoints) that skip them silently desync
#: from the image. Compared by class name (not ``isinstance``) to stay backend- and
#: import-agnostic: the same conceptual op appears under different classes across
#: Albumentations, Kornia, and TorchVision, and the sets overlap by name.
_COORDINATE_CHANGING_PASSTHROUGH_NAMES: frozenset[str] = frozenset({
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
    "PiecewiseAffine",
    "ThinPlateSpline",
    "GridDropout",  # zeroes a coordinate-dependent grid of cells (mask must follow)
    "CoarseDropout",  # zeroes coordinate-dependent boxes; native albu zeroes the mask too
    "XYMasking",  # zeroes coordinate-dependent stripes; native albu masks them as well
    "MaskDropout",  # drops regions derived from the mask itself — mask must follow
    "RandomElasticTransform",  # kornia
    "RandomThinPlateSpline",  # kornia
    "RandomErasing",  # kornia/torchvision: zeroes a coordinate-dependent box
})


def is_coordinate_changing_passthrough(transform: object) -> bool:
    """Return whether a passthrough transform changes pixel coordinates.

    Distinguishes geometric *distortion* passthrough ops (elastic, grid, optical
    distortion, thin-plate-spline, piecewise-affine) — which move image content and
    therefore desync auxiliary targets that skip them — from kernel/pointwise
    passthrough ops (blur, noise, gamma) which leave geometry intact and let
    auxiliary targets legitimately pass unchanged. Classification is by transform
    class name (see :data:`_COORDINATE_CHANGING_PASSTHROUGH_NAMES`) so it works for
    every backend without importing optional dependencies.

    Args:
        transform: The passthrough transform object being classified.

    Returns:
        ``True`` for a coordinate-changing (geometric distortion) passthrough op,
        ``False`` for a kernel/pointwise passthrough op.

    Examples:
        ```pycon
        >>> class ElasticTransform:
        ...     pass
        >>> class GaussianBlur:
        ...     pass
        >>> is_coordinate_changing_passthrough(ElasticTransform())
        True
        >>> is_coordinate_changing_passthrough(GaussianBlur())
        False

        ```

    """
    return type(transform).__name__ in _COORDINATE_CHANGING_PASSTHROUGH_NAMES


class ReorderPolicy(Enum):
    """Controls whether transforms are reordered before segmentation.

    Attributes:
        NONE: No reordering; fuse only consecutive geometric ops as-is (v0.1 default).
        POINTWISE: Move POINTWISE ops out of geometric chains (v0.2).
        AGGRESSIVE: Alias of POINTWISE today; reserved for stronger reorder semantics later.

    """

    NONE = "none"
    POINTWISE = "pointwise"
    AGGRESSIVE = "aggressive"


class RandomnessPolicy(Enum):
    """Controls how batch randomness is sampled by fused segments.

    Attributes:
        BACKEND: Preserve each backend's native batch-randomness semantics.
        PER_SAMPLE: Prefer one independent probability/parameter draw per batch
            item when the adapter exposes a canonical per-sample sampler.

    """

    BACKEND = "backend"
    PER_SAMPLE = "per_sample"


class InterpolationMode(IntEnum):
    """Interpolation modes ordered by quality (higher = finer).

    Examples:
        ```pycon
        >>> InterpolationMode.BICUBIC > InterpolationMode.BILINEAR
        True

        ```

    """

    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2


class PaddingMode(IntEnum):
    """Padding modes ordered by quality (higher = fewer artifacts).

    Examples:
        ```pycon
        >>> PaddingMode.REFLECTION > PaddingMode.ZEROS
        True

        ```

    """

    ZEROS = 0
    BORDER = 1
    REFLECTION = 2


#: String literal type for the ``interpolation`` parameter accepted by pipeline
#: constructors and segment classes. Maps to ``torch.nn.functional.grid_sample``
#: ``mode`` values; ordered by quality (bicubic > bilinear > nearest).
InterpolationStr = Literal["bilinear", "nearest", "bicubic"]

#: String literal type for auxiliary mask sampling. ``"nearest"`` preserves
#: hard labels; ``"bilinear"`` keeps float masks differentiable.
MaskInterpolationStr = Literal["nearest", "bilinear"]

#: String literal type for the ``padding_mode`` parameter accepted by pipeline
#: constructors and segment classes. Maps to ``torch.nn.functional.grid_sample``
#: ``padding_mode`` values; ordered by quality (reflection > border > zeros).
PaddingModeStr = Literal["zeros", "border", "reflection"]

#: Padding mode accepted by pipeline constructors. ``"per_transform"`` is an
#: opt-in segmentation policy rather than a ``grid_sample`` padding value.
ComposePaddingModeStr = PaddingModeStr | Literal["per_transform"]

#: Constant value written into the out-of-canvas region of a warped **image**, in
#: the image's own value range (``114`` for a uint8 grey, ``114 / 255`` for the same
#: grey as float). A scalar fills every channel; a sequence fills one channel each.
#: ``None`` keeps the plain zero padding. Auxiliary masks are never filled -- an
#: out-of-canvas mask region means "no instance", not "the fill colour".
FillValue = float | Sequence[float]

#: Constant scalar written into the out-of-canvas region of a warped **mask**.
#: This is independent of :data:`FillValue`: masks carry one label per pixel, not
#: image channels. Boolean values are accepted at runtime as the integer values 0 and 1.
MaskFillValue = int | float

#: String literal type for the ``kind`` field of :class:`SegmentDescriptor`.
SegmentKind = Literal["fused", "exact", "projective", "passthrough", "color", "lut", "crop_resize", "gaussian_blur"]

#: String literal type for the ``execution`` strategy of the Albumentations fused
#: segments. ``"cv2"`` (default) applies one ``cv2.warp*`` per sample -- bit-exact
#: with the native cv2 backend and the fastest choice on CPU at small batch sizes.
#: ``"torch"`` composes the same per-sample matrices but applies one batched
#: ``grid_sample`` for the whole batch, giving batch-size-independent throughput and
#: a native GPU/MPS warp; its border/bilinear numerics differ slightly from cv2.
#: ``"auto"`` resolves per call by a fixed, documented rule -- host input to cv2,
#: accelerator input to torch -- and is never the default. Choosing it opts out of
#: bit-reproducibility across environments, because the engine is then a function of
#: where the data happens to live rather than of the recorded configuration.
ExecutionStr = Literal["cv2", "torch", "auto"]

#: String literal type for the ``clip_policy`` of the fused color segment.
#: ``"final"`` (default) applies the whole color chain as one 4x4 matmul and clamps
#: the result once -- the more precise option, but out-of-gamut intermediates are not
#: clamped between ops, so it can diverge from a native per-op chain. ``"per_op_parity"``
#: splits the color chain so each op that could push a pixel outside ``[0, 1]`` is clamped
#: on its own, matching a native sequential (clamp-after-each-op) chain at the cost of
#: extra passes only where clipping is actually possible.
ClipPolicyStr = Literal["final", "per_op_parity"]

#: Optional low-precision dtype for fused GPU image operations. Matrix composition
#: remains in float32 or float64 regardless of this image-operation preference.
PipelineDtypeStr = Literal["bfloat16", "float16"]


@runtime_checkable
class BackendConverter(Protocol):
    """Protocol for output format converters used with ``output_backend=``.

    Implementations convert the pipeline's native ``torch.Tensor`` output to a target backend format (e.g. NumPy
    ``ndarray``).

    """

    def convert(self, tensor: Any) -> Any:  # noqa: ANN401
        """Convert tensor to target backend format.

        Args:
            tensor: The input tensor to convert.

        Returns:
            Converted object in the target backend format.

        """
        ...

    @property
    def target_backend(self) -> str:
        """Target backend identifier (e.g. ``'numpy'``, ``'torch'``)."""
        ...


@runtime_checkable
class TransformAdapter(Protocol):
    """Adapter between a backend transform and the fused affine engine.

    Implementations bridge framework-specific transforms (Kornia, Albumentations, TorchVision) to the canonical
    parameter representation used by FusedAffineSegment.

    Optional descriptive members (read via ``fused_transforms._backend.adapter_capabilities`` and direct
    ``getattr``, so absence is tolerated for backwards compatibility):

    - ``capabilities: frozenset[str]`` — canonical op names (see
      :data:`~fused_transforms.dispatch.resolver.SUPPORTED_OPS`) the adapter can build.
    - ``sampling_semantics: SamplingSemantics`` — whether the adapter draws one parameter set per sample or one per
      batch.

    These are intentionally **not** declared as Protocol members: doing so would make ``@runtime_checkable``
    ``isinstance`` require them, breaking adapters that predate the attributes. Instead they are read defensively via
    ``fused_transforms._backend.adapter_capabilities`` and ``getattr``, so absence is tolerated. ``isinstance``
    therefore continues to check only the methods below.

    """

    def category(self, transform: object) -> TransformCategory:
        """Return the TransformCategory of the given transform.

        Args:
            transform: The backend transform object.

        Returns:
            The category classification for the transform.

        """
        ...

    def sample_params(
        self,
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, Tensor]:
        """Sample random parameters for a batch of images.

        Args:
            transform: The backend transform object.
            input_shape: (batch_size, channels, height, width) tuple.
            device: Target device for parameter tensors.

        Returns:
            Dict mapping canonical parameter names to (batch_size,) tensors.

        """
        ...

    def build_matrix(
        self,
        transform: object,
        params: dict[str, Tensor],
        height: int,
        width: int,
    ) -> Tensor:
        """Build a (batch_size, 3, 3) pixel-space forward affine matrix from sampled params.

        Args:
            transform: The backend transform object.
            params: Canonical-unit parameter dict from sample_params().
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Tensor of shape (batch_size, 3, 3).

        """
        ...

    def exact_flip_dims(self, transform: object) -> list[int]:
        """Return the tensor dimensions to flip for a GEOMETRIC_EXACT transform.

        Args:
            transform: The backend transform object (must be GEOMETRIC_EXACT category).

        Returns:
            List of dimension indices passed to ``tensor.flip(dims=...)``,
            e.g. ``[3]`` for a horizontal flip, ``[2]`` for a vertical flip.

        Raises:
            NotImplementedError: If the adapter does not support ExactAffineSegment.

        """
        raise NotImplementedError("Adapter does not implement exact_flip_dims; required for ExactAffineSegment support")

    def exact_apply(
        self,
        transform: object,
        image: Tensor,
        *,
        params: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly to an image batch.

        Implementers **must** provide this method for adapters that are used with :class:`ExactAffineSegment`.
        A typical implementation flips the image along the dims returned by :meth:`exact_flip_dims`.
        Adapters that support non-flip discrete ops (e.g. 90-degree rotations, transposes) can instead dispatch via
        ``torch.rot90``, ``.permute``, etc.

        Args:
            transform: The backend transform object (GEOMETRIC_EXACT category).
            image: ``(batch_size, channels, height, width)`` input tensor.
            params: Canonical parameters sampled for this exact application. Stochastic
                discrete adapters must consume these rather than draw a second value.

        Returns:
            Transformed ``(batch_size, channels, height, width)`` tensor.

        """
        del params
        return image.flip(dims=self.exact_flip_dims(transform))

    def call_nonfused(
        self,
        transform: object,
        image: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Apply a non-fusible transform directly via its native backend.

        Args:
            transform: The backend transform object.
            image: Input image tensor.
            **kwargs: Additional keyword arguments forwarded to the transform.

        Returns:
            Transformed image tensor.

        """
        ...

    def build_color_matrix(
        self,
        transform: object,
        params: dict[str, Tensor],
        mean: Tensor | None = None,
    ) -> Tensor:
        """Build a (batch_size, 4, 4) homogeneous color-space affine matrix from sampled params.

        Adapters that support ``POINTWISE_LINEAR`` fusion must override this method to return a ``(batch_size, 4, 4)``
        matrix encoding the per-channel linear colour transform (3x3 colour matrix + 3-element bias in homogeneous
        form).  The default implementation raises ``NotImplementedError`` so that adapters without colour-fusion
        support fall back to passthrough segmentation automatically.

        Accuracy caveats vs native backends (inherent to single-matrix fusion):

        - No intermediate clamping: native backends clamp to ``[0, 1]`` after EACH op; a fused matrix
          cannot represent clamping between ops, so out-of-gamut intermediates diverge from native.
          :class:`~fused_transforms.affine.segment.FusedColorSegment` clamps only the FINAL result
          (``clip_output=True`` default). See ``clip_policy`` on
          :class:`~fused_transforms.compose.FusedCompose` for a per-op-parity split option.
        - Contrast midpoint: TorchVision/Kornia ``ColorJitter`` contrast is relative to the per-image
          mean luminance (``c' = cf*c + (1-cf)*luma_mean``). When ``mean`` is provided (the fused segment
          supplies the per-image luminance of the transform's input), that mean is used, matching native.
          When ``mean`` is ``None`` (a bare, out-of-context call), a fixed midpoint of ``0.5`` is used so
          the returned matrix is well-defined without an image.

        Args:
            transform: The backend transform object (``POINTWISE_LINEAR`` category).
            params: Canonical parameter dict from :meth:`sample_params`.
            mean: Optional per-image luminance of the image reaching this transform, shape
                ``(batch_size,)``. Used only by contrast-like ops with a mean-relative midpoint
                (``ColorJitter`` contrast). ``None`` falls back to the fixed ``0.5`` midpoint.

        Returns:
            Tensor of shape ``(batch_size, 4, 4)``.

        Raises:
            NotImplementedError: If the adapter does not support colour-space matrix fusion for this transform.

        """
        raise NotImplementedError(
            "Adapter does not implement build_color_matrix; required for FusedColorSegment support"
        )

    def build_lut(
        self,
        transform: object,
        params: dict[str, Tensor],
        values: Tensor,
    ) -> Tensor:
        """Apply a ``POINTWISE_LUT`` transform's per-channel intensity map to a grid of values.

        Adapters that support static ``POINTWISE_LUT`` fusion (gamma, solarize, posterize) must override
        this to evaluate the transform's pointwise, per-channel scalar map on ``values`` using its
        own backend function (never a re-derived formula), returning the mapped intensities. The
        default implementation raises ``NotImplementedError`` so adapters without lookup-fusion
        support fall back to passthrough segmentation automatically.

        :class:`~fused_transforms.affine.segment.FusedLUTSegment` composes a contiguous run of
        such ops by *threading a domain grid through each op in order*: ``values`` starts as a
        uniform grid over ``[0, 1]`` and each op maps it in place, so after the run ``values``
        holds the composed function sampled on that grid — one lookup table applied once. Because
        each op is evaluated exactly on the running values (not re-interpolated between ops), the
        composition itself introduces no error.

        Accuracy caveats vs native backends:

        - **uint8 / 256-entry path (exact).** When the segment enumerates the map over all 256
          byte values, the composed table is bit-exact against a native sequential chain: each op
          is a deterministic ``0..255 -> 0..255`` map and integer composition loses nothing.
        - **float / K-entry interp path (approximate).** For a floating image the composed table
          is sampled on a uniform ``K``-point grid (default ``K = 1024``) and applied by
          ``gather`` + linear interpolation. This is exact-ish only for *smooth* maps: a steep
          region (e.g. ``gamma < 1`` near black) can differ from native by ~2 uint8 levels, and a
          *discontinuous* map (solarize threshold, posterize step) uses a detected sharp step instead
          of a linear ramp, but its inferred breakpoint can still differ by part of a grid cell. Do
          not claim the float path beats native precision; gate float-path parity at a documented
          interpolation tolerance, never as ``>=`` native.

        Args:
            transform: The backend transform object (``POINTWISE_LUT`` category).
            params: Canonical parameter dict from :meth:`sample_params`.
            values: ``(batch_size, channels, num_points)`` float tensor of input intensities in
                ``[0, 1]`` to map. The per-sample leading dim lets per-sample parameters (a random
                gamma per image) produce a per-sample lookup table.

        Returns:
            Tensor of shape ``(batch_size, channels, num_points)``: ``values`` after the
            transform's per-channel intensity map, clamped to ``[0, 1]`` as the native op does.

        Raises:
            NotImplementedError: If the adapter does not support lookup-table fusion for this transform.

        """
        raise NotImplementedError("Adapter does not implement build_lut; required for FusedLUTSegment support")


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Declarative specification for a single augmentation transform.

    A backend-agnostic, JSON-serialisable description of one augmentation operation. Used by
    :meth:`FusedCompose.from_config <fused_transforms.compose.FusedCompose.from_config>` and
    :meth:`FusedCompose.from_params <fused_transforms.compose.FusedCompose.from_params>` to build pipelines
    from configuration data rather than live transform objects.

    Args:
        operation: Canonical operation name (e.g. ``"rotation"``, ``"hflip"``).
        params: Operation-specific parameters (e.g. ``{"degrees": (-30, 30)}``).
            Range values (``degrees``, ``scale``, etc.) must be 2-tuples, not
            lists. For JSON/YAML-deserialized configs use :meth:`from_dict`,
            which restores tuple semantics from any sequence type (``list``,
            OmegaConf ``ListConfig``, etc.) automatically.
        prob: Per-sample application probability. Default ``1.0``.

    Examples:
        ```pycon
        >>> spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        >>> spec.operation
        'rotation'
        >>> spec.prob
        0.8

        ```

    """

    operation: str
    params: Mapping[str, object]
    prob: float = 1.0

    def __post_init__(self) -> None:
        """Freeze mapping-like params and validate probability bounds."""
        if not (0.0 <= self.prob <= 1.0):
            msg = f"TransformSpec.prob must be in [0.0, 1.0], got {self.prob!r}"
            raise ValueError(msg)
        object.__setattr__(self, "params", _freeze_param_mapping(self.params))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict representation.

        Returns:
            Dict with keys ``"operation"``, ``"params"``, and ``"prob"``.

        Examples:
            ```pycon
            >>> spec = TransformSpec(operation="hflip", params={}, prob=0.5)
            >>> spec.to_dict()
            {'operation': 'hflip', 'params': {}, 'prob': 0.5}

            ```

        """
        params = _to_json_compatible(self.params)
        return {"operation": self.operation, "params": params, "prob": self.prob}

    @classmethod
    def from_dict(cls, data_dict: dict[str, object]) -> "TransformSpec":
        """Construct a ``TransformSpec`` from a dict (e.g. parsed JSON).

        Args:
            data_dict: Dict with at least an ``"operation"`` key. ``"params"`` defaults to
                ``{}`` and ``"prob"`` defaults to ``1.0`` when absent.

        Returns:
            A new ``TransformSpec`` instance.

        Note:
            Tuple restoration applies to any sequence type (``list``, OmegaConf
            ``ListConfig``, etc.) but only for canonical range-parameter keys
            (``'degrees'``, ``'factor'``, ``'scale'``, ``'pixels'``, etc. — the
            full set is :data:`_RANGE_PARAM_KEYS`). Backend-specific keys not in
            that set (e.g. Albumentations ``'limit'``) are preserved as lists.
            OmegaConf ``DictConfig`` objects can be passed directly without calling
            ``OmegaConf.to_container()`` first. This is documented behaviour, not a bug.

        Examples:
            ```pycon
            >>> import json
            >>> spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
            >>> restored = TransformSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
            >>> restored == spec  # list → tuple restored
            True

            ```

        """
        if "params" in data_dict:
            raw_params = data_dict["params"]
            if raw_params is None:
                raw_params = {}
            elif not isinstance(raw_params, Mapping):
                raise TypeError(
                    f"TransformSpec.from_dict expected 'params' to be a mapping, got {type(raw_params).__name__!r}."
                )
        else:
            raw_params = {}
        params = _normalize_loaded_params(raw_params)
        raw_p = data_dict.get("prob", 1.0)
        prob = float(raw_p)  # type: ignore[arg-type]
        return cls(operation=str(data_dict["operation"]), params=params, prob=prob)


_RANGE_PARAM_KEYS: frozenset[str] = frozenset({
    "degrees",
    "factor",
    "pixels",
    "rotation",
    "scale",
    "scale_x",
    "scale_y",
    "shear",
    "shear_x",
    "shear_y",
    "times",
    "translate",
    "translate_x",
    "translate_y",
})


def _freeze_param_mapping(params: Mapping[str, object]) -> MappingProxyType[str, object]:
    """Copy params into an immutable top-level mapping."""
    return MappingProxyType({key: _freeze_param_value(value) for key, value in params.items()})


def _freeze_param_value(value: object) -> object:
    """Recursively freeze nested mappings while preserving list semantics."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_param_value(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return [_freeze_param_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_param_value(item) for item in value)
    return copy.deepcopy(value)


def _to_json_compatible(value: object) -> Any:  # noqa: ANN401
    """Convert params tree into JSON-friendly builtin containers."""
    if isinstance(value, Mapping):
        return {key: _to_json_compatible(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    return value


def _normalize_loaded_params(params: Mapping[str, object]) -> dict[str, object]:
    """Restore range tuples from any sequence type while preserving non-range sequences as lists."""
    return {key: _normalize_loaded_value(key, value) for key, value in params.items()}


def _normalize_loaded_value(key: str | None, value: object) -> object:
    """Normalize one loaded param tree."""
    if isinstance(value, Mapping):
        return {nested_key: _normalize_loaded_value(nested_key, nested) for nested_key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, tuple)):
        normalized_items = [_normalize_loaded_value(None, item) for item in value]
        if (
            key in _RANGE_PARAM_KEYS
            and len(normalized_items) == 2
            and all(isinstance(item, (int, float)) for item in normalized_items)
        ):
            return tuple(normalized_items)
        return normalized_items
    return value


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    """Structured description of one segment in a fused augmentation pipeline.

    Returned by :attr:`FusedCompose.fusion_plan_descriptors
    <fused_transforms.compose.FusedCompose.fusion_plan_descriptors>`. Each instance describes exactly one segment —
    a fused geometric group, a lossless exact segment, a projective segment, or a passthrough barrier — and is frozen
    and JSON-serialisable via :meth:`to_dict`.

    Args:
        kind: Segment type. One of ``"fused"``, ``"exact"``, ``"projective"``,
            ``"color"``, ``"crop_resize"``, or ``"passthrough"``.
        transforms: Class names of the transforms in this segment, in
            execution order.
        n_warps_saved: Number of ``grid_sample`` interpolation passes
            eliminated by fusing this segment. Zero for passthrough and
            single-transform segments.
        backend: Adapter class name used for this segment
            (for example ``"KorniaAdapter"``, ``"AlbumentationsAdapter"``,
            ``"TorchVisionAdapter"``), or ``None`` for backend-free pipelines
            created via
            :meth:`FusedCompose.from_params <fused_transforms.compose.FusedCompose.from_params>`.
        barrier: Machine-readable reason this segment ends a fusion run and forces
            a boundary, or ``None`` when the segment is fused/does not act as a
            barrier. ``"spatial_kernel"`` for blur/noise passthrough, ``"pointwise"``
            for non-linear pixel passthrough, ``"coordinate_change"`` for geometric
            distortion passthrough (elastic/grid/optical), ``"crop_resize"`` for a
            :class:`CropResizeSegment`. This is the structured counterpart to the
            human-readable :attr:`fusion_plan` string.
        split_reason: Machine-readable reason a run that could otherwise fuse was
            split at this segment, or ``None``. Currently ``"backend_boundary"`` when
            a mixed-backend group break created the split, else ``None``.
        refused: Machine-readable reason an op stayed on the passthrough path instead
            of being fused, or ``None`` for fused segments. ``"not_fusible"`` for ops
            with no fusion representation (default for passthrough), or
            ``"substitution_unavailable"`` when substitution was requested but the
            target backend was not importable.

    Examples:
        ```pycon
        >>> d = SegmentDescriptor(
        ...     kind="fused",
        ...     transforms=("RandomRotation", "RandomHorizontalFlip"),
        ...     n_warps_saved=1,
        ...     backend="KorniaAdapter",
        ... )
        >>> d.kind
        'fused'
        >>> d.n_warps_saved
        1
        >>> d.barrier is None
        True
        >>> d.to_dict()  # doctest: +SKIP
        {'kind': 'fused', 'transforms': ['RandomRotation', ...], ...}

        ```

    """

    kind: SegmentKind
    transforms: tuple[str, ...]
    n_warps_saved: int
    backend: str | None = None
    barrier: str | None = None
    split_reason: str | None = None
    refused: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict representation of this descriptor.

        Returns:
            Dict with keys ``"kind"``, ``"transforms"``, ``"n_warps_saved"``,
            ``"backend"``, ``"barrier"``, ``"split_reason"``, and ``"refused"``.
            The ``"transforms"`` value is a ``list`` of strings (not a ``tuple``)
            for JSON compatibility.

        Examples:
            ```pycon
            >>> d = SegmentDescriptor(
            ...     kind="passthrough",
            ...     transforms=("RandomGaussianBlur",),
            ...     n_warps_saved=0,
            ...     backend="kornia",
            ...     barrier="spatial_kernel",
            ...     refused="not_fusible",
            ... )
            >>> d.to_dict()["barrier"]
            'spatial_kernel'
            >>> d.to_dict()["refused"]
            'not_fusible'

            ```

        """
        return {
            "kind": self.kind,
            "transforms": list(self.transforms),
            "n_warps_saved": self.n_warps_saved,
            "backend": self.backend,
            "barrier": self.barrier,
            "split_reason": self.split_reason,
            "refused": self.refused,
        }
