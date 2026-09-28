"""Backend resolver for canonical operation names.

Maps ``(operation: str, backend: str) -> type`` so that :meth:`FusedCompose.from_config
<fused_transforms.compose.FusedCompose.from_config>` can construct backend-specific transforms from a declarative
:class:`TransformSpec`.

Each backend adapter exposes wrapper classes (e.g. Kornia's ``_RandomRotation``, TorchVision's ``RandomRotation``).
This module builds a reverse lookup from canonical operation names (``"rotation"``, ``"hflip"``, etc.) to those wrapper
classes, importing each backend lazily to avoid hard dependencies.

Examples:
    ```pycon
    >>> from fused_transforms.dispatch.resolver import SUPPORTED_OPS
    >>> "rotation" in SUPPORTED_OPS
    True

    ```

"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.util import find_spec
from typing import Literal

if find_spec("torchvision") is None:
    __doctest_skip__ = ["resolve_op"]

SUPPORTED_OPS: frozenset[str] = frozenset({
    "rotation",
    "affine",
    "shear",
    "translate",
    "hflip",
    "vflip",
    "scale",
    "perspective",
    "rotation90",
    "normalize",
    "brightness",
    "contrast",
})

# Per-backend op coverage matrix.
# op               | kornia | torchvision | albumentations
# ------------|--------|-------------|----------------
# rotation    |   v    |      v      |       v
# affine      |   v    |      v      |       v
# shear       |   v    |      x      |       x
# translate   |   v    |      x      |       x
# hflip       |   v    |      v      |       v
# vflip       |   v    |      v      |       v
# scale       |   v    |      v      |       v
# perspective |   v    |      v      |       v
# rotation90  |   v    |      x      |       v
# normalize   |   v    |      v      |       v
# brightness  |   v    |      x      |       x
# contrast    |   v    |      x      |       x

SUPPORTED_BACKENDS: frozenset[str] = frozenset({
    "kornia",
    "torchvision",
    "albumentations",
    "native",
})

_NATIVE_OPS: frozenset[str] = frozenset({
    "rotation",
    "scale",
    "shear",
    "translate",
    "hflip",
    "vflip",
    "brightness",
    "contrast",
})


@dataclass(frozen=True, slots=True)
class _NativeOpSpec:
    """Describe a zero-dependency operation for the native builder."""

    operation: str


#: String literal type for canonical operation names accepted by :func:`translate_params`
#: and :func:`resolve_op`.
OpStr = Literal[
    "rotation",
    "affine",
    "shear",
    "translate",
    "hflip",
    "vflip",
    "scale",
    "perspective",
    "rotation90",
    "normalize",
    "brightness",
    "contrast",
]

#: String literal type for backend names accepted by :func:`translate_params`
#: and :func:`resolve_op`.
BackendStr = Literal["kornia", "torchvision", "albumentations", "native"]


@cache
def _kornia_registry() -> dict[str, type]:
    """Build op -> class map for the Kornia backend (lazy import)."""
    from kornia.augmentation import Normalize as _Normalize
    from kornia.augmentation import RandomAffine as _RandomAffine
    from kornia.augmentation import RandomBrightness as _RandomBrightness
    from kornia.augmentation import RandomContrast as _RandomContrast
    from kornia.augmentation import RandomHorizontalFlip as _RandomHorizontalFlip
    from kornia.augmentation import RandomPerspective as _RandomPerspective
    from kornia.augmentation import RandomRotation as _RandomRotation
    from kornia.augmentation import RandomRotation90 as _RandomRotation90
    from kornia.augmentation import RandomShear as _RandomShear
    from kornia.augmentation import RandomTranslate as _RandomTranslate
    from kornia.augmentation import RandomVerticalFlip as _RandomVerticalFlip

    return {
        "rotation": _RandomRotation,
        "affine": _RandomAffine,
        "shear": _RandomShear,
        "translate": _RandomTranslate,
        "hflip": _RandomHorizontalFlip,
        "vflip": _RandomVerticalFlip,
        "scale": _RandomAffine,  # scale is a subset of affine
        "perspective": _RandomPerspective,
        "rotation90": _RandomRotation90,
        "normalize": _Normalize,
        "brightness": _RandomBrightness,
        "contrast": _RandomContrast,
    }


@cache
def _torchvision_registry() -> dict[str, type]:
    """Build op -> class map for the TorchVision backend (lazy import)."""
    # Prefer v2 when available, fall back to v1
    try:
        from torchvision.transforms.v2 import (
            Normalize,
            RandomAffine,
            RandomHorizontalFlip,
            RandomPerspective,
            RandomRotation,
            RandomVerticalFlip,
        )
    except ImportError:
        from torchvision.transforms import (
            Normalize,
            RandomAffine,
            RandomHorizontalFlip,
            RandomPerspective,
            RandomRotation,
            RandomVerticalFlip,
        )

    return {
        "rotation": RandomRotation,
        "affine": RandomAffine,
        "hflip": RandomHorizontalFlip,
        "vflip": RandomVerticalFlip,
        "scale": RandomAffine,  # scale is a subset of affine
        "perspective": RandomPerspective,
        "normalize": Normalize,
    }


@cache
def _albumentations_registry() -> dict[str, type]:
    """Build op -> class map for the Albumentations backend (lazy import)."""
    from albumentations import Affine as _Affine
    from albumentations import HorizontalFlip as _HorizontalFlip
    from albumentations import Normalize as _Normalize
    from albumentations import Perspective as _Perspective
    from albumentations import RandomRotate90 as _RandomRotate90
    from albumentations import Rotate as _Rotate
    from albumentations import VerticalFlip as _VerticalFlip

    return {
        "rotation": _Rotate,
        "affine": _Affine,
        "hflip": _HorizontalFlip,
        "vflip": _VerticalFlip,
        "scale": _Affine,  # scale is a subset of affine
        "perspective": _Perspective,
        "rotation90": _RandomRotate90,
        "normalize": _Normalize,
    }


_BACKEND_REGISTRY_BUILDERS: dict[str, Callable[[], dict[str, type]]] = {
    "kornia": _kornia_registry,
    "torchvision": _torchvision_registry,
    "albumentations": _albumentations_registry,
}


def _registry_for(backend: str) -> dict[str, type]:
    """Return the op -> class registry for *backend*, tolerating backends without one.

    A backend with no registered builder (e.g. a future ``"native"`` backend added before it grows a
    registry) yields an empty dict rather than a ``KeyError``. A backend whose optional dependency is not
    installed also yields an empty dict, so capability queries never require the backend to be importable.

    Args:
        backend: Backend name.

    Returns:
        The backend's op -> class registry, or an empty dict when the backend has no builder or is not installed.

    """
    builder = _BACKEND_REGISTRY_BUILDERS.get(backend)
    if builder is None:
        return {}
    try:
        return builder()
    except (ImportError, ModuleNotFoundError):
        return {}


def capability_matrix() -> dict[str, frozenset[str]]:
    """Return the backend -> supported canonical-op-names map for all supported backends.

    The op vocabulary is anchored on each backend's resolver registry keys (a subset of
    :data:`SUPPORTED_OPS`). Backends whose optional dependency is not installed report an empty frozenset rather than
    raising, so this is a pure, side-effect-free capability query.

    Returns:
        Mapping from each backend in :data:`SUPPORTED_BACKENDS` to the frozenset of canonical op names it can build.

    Examples:
        ```pycon
        >>> matrix = capability_matrix()
        >>> set(matrix) == set(SUPPORTED_BACKENDS)
        True
        >>> "rotation90" in matrix["torchvision"]
        False

        ```

    """
    return {
        backend: (_NATIVE_OPS if backend == "native" else frozenset(_registry_for(backend)))
        for backend in SUPPORTED_BACKENDS
    }


def translate_params(op_name: OpStr, backend: BackendStr, params: dict[str, object]) -> dict[str, object]:
    """Translate canonical ``TransformSpec.params`` into backend ctor kwargs.

    Args:
        op_name: Canonical operation name.
        backend: Target backend name.
        params: Canonical transform params.

    Returns:
        Backend-specific constructor kwargs.

    Raises:
        ValueError: If ``operation`` or ``backend`` is unknown.

    Notes:
        Canonical-to-backend translations applied automatically:

        - rotation / affine + albumentations:
          'degrees' -> 'limit' (rotation) or 'rotate' (affine).
        - scale + kornia / torchvision:
          'factor' -> 'scale' and degrees=0.0 injected (required by Affine ctor).
        - scale + albumentations:
          'factor' -> 'scale' only; degrees is NOT injected.
        - shear + kornia:
          'degrees' -> 'shear'.

    """
    if backend not in SUPPORTED_BACKENDS:
        msg = f"unknown backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
        raise ValueError(msg)
    if op_name not in SUPPORTED_OPS:
        msg = f"unknown op {op_name!r}; supported: {sorted(SUPPORTED_OPS)}"
        raise ValueError(msg)

    kwargs = dict(params)

    def _move_param(source_key: str, target_key: str) -> None:
        if source_key in kwargs:
            value = kwargs.pop(source_key)
            if target_key in kwargs and kwargs[target_key] != value:
                warnings.warn(
                    f"Both {source_key!r} and {target_key!r} were provided for op {op_name!r}; "
                    f"keeping {target_key!r}={kwargs[target_key]!r} and discarding "
                    f"{source_key!r}={value!r}.",
                    UserWarning,
                    stacklevel=3,
                )
            kwargs.setdefault(target_key, value)

    if op_name in {"rotation", "affine"}:
        if backend == "albumentations":
            _move_param("rotation", "rotate" if op_name == "affine" else "limit")
            _move_param("degrees", "rotate" if op_name == "affine" else "limit")
        else:
            _move_param("rotation", "degrees")

    if op_name == "scale":
        _move_param("factor", "scale")
        if backend in {"torchvision", "kornia"}:
            kwargs.setdefault("degrees", 0.0)

    if op_name == "shear" and backend == "kornia":
        _move_param("degrees", "shear")

    if op_name == "translate" and backend == "kornia":
        if "pixels" in kwargs:
            pixels = kwargs.pop("pixels")
            kwargs.setdefault("translate_x", pixels)
            kwargs.setdefault("translate_y", pixels)
        if "translate" in kwargs:
            translate = kwargs.pop("translate")
            kwargs.setdefault("translate_x", translate)
            kwargs.setdefault("translate_y", translate)

    if op_name == "rotation90" and backend == "kornia":
        # Kornia requires explicit bounds; this default matches full quarter-turn sampling.
        kwargs.setdefault("times", (0, 3))

    if op_name in {"brightness", "contrast"} and backend == "kornia":
        _move_param("factor", op_name)

    return kwargs


def resolve_op(operation: OpStr, backend: BackendStr) -> type | _NativeOpSpec:
    """Resolve a canonical operation name to its backend transform class.

    Args:
        operation: Canonical operation name (e.g. ``"rotation"``, ``"hflip"``).
            Must be one of :data:`SUPPORTED_OPS`.
        backend: Backend name (e.g. ``"kornia"``, ``"torchvision"``,
            ``"albumentations"``). Must be one of :data:`SUPPORTED_BACKENDS`.

    Returns:
        The backend-specific transform class for the given operation.

    Raises:
        ValueError: If ``op_name`` is not in :data:`SUPPORTED_OPS` or ``backend``
            is not in :data:`SUPPORTED_BACKENDS`, or the backend does not
            support the requested operation.

    Examples:
        ```pycon
        >>> resolve_op("hflip", "torchvision").__name__
        'RandomHorizontalFlip'

        ```

    """
    if backend not in SUPPORTED_BACKENDS:
        msg = f"unknown backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
        raise ValueError(msg)
    if operation not in SUPPORTED_OPS:
        msg = f"unknown op {operation!r}; supported: {sorted(SUPPORTED_OPS)}"
        raise ValueError(msg)

    if backend == "native":
        if operation not in _NATIVE_OPS:
            msg = (
                f"backend {backend!r} does not support op {operation!r}; "
                f"supported ops for {backend}: {sorted(_NATIVE_OPS)}"
            )
            raise ValueError(msg)
        return _NativeOpSpec(operation)

    builder = _BACKEND_REGISTRY_BUILDERS[backend]
    try:
        registry = builder()
    except (ImportError, ModuleNotFoundError) as exc:
        msg = (
            f"backend {backend!r} is not available because its optional dependency is "
            f"not installed. Install it with e.g. `pip install vision-synth[{backend}]`."
        )
        raise ValueError(msg) from exc
    if operation not in registry:
        msg = f"backend {backend!r} does not support op {operation!r}; supported ops for {backend}: {sorted(registry)}"
        raise ValueError(msg)
    return registry[operation]
