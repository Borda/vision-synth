"""Backend detection for augmentation transform pipelines.

Inspects transform module paths to determine which backend framework (Kornia, Albumentations, TorchVision) is in use.

Detection is driven by a pluggable adapter registry. The three built-in backends register at import time by module
prefix only: detection matches a transform's module path, so it never needs the backend imported, and each built-in
adapter module (with the backend library it imports) loads on first access to its registry entry's ``adapter``.
Third-party adapters may register through :func:`register_adapter` or the ``fused_transforms.adapters`` entry-point
group; those are loaded **lazily** on the first detection miss (never at package import) so that ``import
fused_transforms`` neither executes third-party code nor pays their import cost.

Examples:
    ```pycon
    >>> from fused_transforms._backend import detect_backend
    >>> detect_backend([])
    <Backend.UNKNOWN: 'unknown'>

    ```

"""

from __future__ import annotations

import sys
import warnings
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from fused_transforms.types import TransformAdapter


class Backend(Enum):
    """Supported augmentation backend frameworks."""

    KORNIA = "kornia"
    ALBUMENTATIONS = "albumentations"
    TORCHVISION = "torchvision"
    UNKNOWN = "unknown"


#: Name of the entry-point group third-party packages use to register adapters.
ADAPTERS_ENTRY_POINT_GROUP = "fused_transforms.adapters"


class _Entry:
    """A registered adapter: its backend tag, module prefixes, and an adapter instance resolved on first access.

    Detection reads only ``backend`` and ``prefixes``. The adapter comes either ready-made (:func:`register_adapter`)
    or from a zero-argument ``loader`` (the built-ins), which runs once, the first time ``adapter`` or
    ``capabilities`` is read, so registering a built-in never imports its backend library.

    Attributes:
        backend: The ``Backend`` this adapter detects (``Backend.UNKNOWN`` for third-party adapters that map to no
            built-in enum member).
        prefixes: Module-path prefixes (e.g. ``"kornia."``) that identify transforms handled by this adapter.

    Examples:
        ```pycon
        >>> entry = _Entry(Backend.UNKNOWN, ("dummypkg.",), loader=lambda: object())
        >>> entry.prefixes
        ('dummypkg.',)

        ```

    """

    __slots__ = ("_adapter", "_loader", "backend", "prefixes")

    def __init__(
        self,
        backend: Backend,
        prefixes: tuple[str, ...],
        adapter: TransformAdapter | None = None,
        *,
        loader: Callable[[], TransformAdapter] | None = None,
    ) -> None:
        """Initialize the entry from exactly one of ``adapter`` and ``loader``."""
        if (adapter is None) == (loader is None):
            raise ValueError("_Entry needs exactly one of adapter= or loader=")
        self.backend = backend
        self.prefixes = prefixes
        self._adapter = adapter
        self._loader = loader

    @property
    def adapter(self) -> TransformAdapter:
        """Return the adapter instance, running the loader on first access."""
        if self._adapter is None:
            # Benign race: two threads may both run the loader; adapters are stateless, the last one wins.
            self._adapter = cast("Callable[[], TransformAdapter]", self._loader)()
        return self._adapter

    @property
    def capabilities(self) -> frozenset[str]:
        """Return the canonical op names the adapter can build (loads a lazily registered adapter)."""
        return adapter_capabilities(self.adapter)


#: name -> _Entry. Built-ins self-register at import; third-party adapters load lazily (see ``_load_entrypoints``).
_ADAPTER_REGISTRY: dict[str, _Entry] = {}

#: Guard so the entry-point group is scanned at most once (idempotent; a failed scan is not retried repeatedly).
_ENTRYPOINTS_LOADED = False


def adapter_capabilities(adapter: object) -> frozenset[str]:
    """Return an adapter's declared ``capabilities``, defaulting to empty.

    ``capabilities`` is an optional member of the :class:`~fused_transforms.types.TransformAdapter` protocol; adapters
    predating it need not define it. This getattr helper keeps the registry backwards-compatible.

    Args:
        adapter: An adapter instance (or class).

    Returns:
        The adapter's ``capabilities`` as a ``frozenset[str]``, or an empty frozenset when the member is absent.

    Examples:
        ```pycon
        >>> adapter_capabilities(object())
        frozenset()

        ```

    """
    return frozenset(getattr(adapter, "capabilities", frozenset()))


def register_adapter(
    name: str,
    adapter: TransformAdapter,
    module_prefixes: str | tuple[str, ...] | list[str],
    *,
    backend: Backend = Backend.UNKNOWN,
) -> None:
    """Register a backend adapter for detection and fusion.

    .. warning::

        **Experimental.** ``register_adapter`` and the ``fused_transforms.adapters`` entry-point group are a
        provisional third-party extension API. The signature may change until an external adapter validates it.

    Args:
        name: Unique registry key for the adapter (re-registering the same name overwrites the prior entry).
        adapter: An instance implementing the :class:`~fused_transforms.types.TransformAdapter` protocol.
        module_prefixes: One or more module-path prefixes (e.g. ``"mypkg.transforms."``) whose transforms this
            adapter handles. A trailing dot is recommended to avoid spurious prefix collisions.
        backend: The :class:`Backend` enum member this adapter maps to. Defaults to ``Backend.UNKNOWN`` for
            third-party backends without a built-in enum member.

    Examples:
        ```pycon
        >>> class _Dummy:
        ...     capabilities = frozenset({"rotation"})
        >>> register_adapter("dummy", _Dummy(), "dummypkg.")
        >>> "dummy" in _ADAPTER_REGISTRY
        True

        ```

    """
    prefixes = (module_prefixes,) if isinstance(module_prefixes, str) else tuple(module_prefixes)
    _ADAPTER_REGISTRY[name] = _Entry(backend, prefixes, adapter)


def _load_entrypoints() -> None:
    """Lazily load third-party adapters from the entry-point group (idempotent, failure-isolated).

    Called on the first detection miss, never at package import (avoids executing third-party code and paying its import
    cost on ``import fused_transforms``). Each entry point is loaded in isolation; a failing ``load()`` or
    ``register()`` is warned and skipped so one broken plugin cannot break detection for the rest.

    """
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True

    from importlib import metadata

    try:
        entry_points = metadata.entry_points(group=ADAPTERS_ENTRY_POINT_GROUP)
    except Exception as exc:
        warnings.warn(f"Failed to query adapter entry points: {exc!r}", UserWarning, stacklevel=2)
        return

    for ep in entry_points:
        _load_one_entrypoint(ep)


def _load_one_entrypoint(ep: object) -> None:
    """Load and invoke a single adapter entry point in isolation.

    Failure isolation is per entry point: a broken ``load()`` or ``register()`` is warned and skipped so one bad
    plugin cannot break detection for the rest (kept a separate function to isolate the broad ``except`` from the
    discovery loop).

    Args:
        ep: An ``importlib.metadata.EntryPoint`` whose ``load()`` returns a zero-arg registration callable.

    """
    try:
        register = ep.load()  # type: ignore[attr-defined]
        register()
    except Exception as exc:
        name = getattr(ep, "name", ep)
        warnings.warn(f"Failed to load adapter entry point {name!r}: {exc!r}; skipping.", UserWarning, stacklevel=2)


def _classify_transform(transform: object) -> Backend | None:
    """Resolve a single transform's backend from its module, falling back to its MRO.

    Shared by :func:`detect_backend` and :func:`detect_backends_per_transform` so both agree on
    subclassed transforms: a direct module-prefix match is tried first, then (on a miss) the
    transform's method resolution order is walked for an ancestor whose module matches a known
    backend prefix. This handles subclasses defined outside the backend package (e.g. a user-defined
    ``class MyRot(torchvision.transforms.RandomRotation)`` in ``__main__``).

    Args:
        transform: A transform object.

    Returns:
        The matched ``Backend``, or ``None`` when neither the module nor any MRO ancestor matches.

    """
    module = type(transform).__module__ or ""
    backend = _match_backend(module)
    if backend is None:
        backend = _match_backend_from_mro(type(transform))
    return backend


def _warn_unrecognized(transform: object) -> None:
    """Emit the shared ``UserWarning`` for a transform that matched no known backend prefix."""
    warnings.warn(
        f"Unrecognized transform {type(transform).__name__!r}; treating as SPATIAL_KERNEL barrier.",
        UserWarning,
        stacklevel=3,
    )


def detect_backend(transforms: list[object]) -> Backend:
    """Detect the backend from a list of transforms by inspecting module paths.

    A direct module-prefix match is tried first; on a miss the transform's MRO is walked for a
    matching ancestor (via :func:`_classify_transform`), so a subclass of a backend transform
    resolves to that backend rather than being treated as unrecognized. This mirrors
    :func:`detect_backends_per_transform`, keeping the two in agreement on subclassed transforms.

    Args:
        transforms: List of transform objects.

    Returns:
        A ``Backend`` enum member.

    Raises:
        ValueError: If transforms come from more than one backend.

    Examples:
        ```pycon
        >>> detect_backend([])
        <Backend.UNKNOWN: 'unknown'>

        ```

    """
    backends: set[Backend] = set()

    for transform in transforms:
        backend = _classify_transform(transform)
        if backend is None:
            _warn_unrecognized(transform)
        else:
            backends.add(backend)

    if len(backends) > 1:
        msg = (
            "Mixed backends are not supported by detect_backend(); all transforms must "
            "use the same backend. For mixed-backend pipelines, use "
            "detect_backends_per_transform()."
        )
        raise ValueError(msg)

    if len(backends) == 1:
        return backends.pop()
    return Backend.UNKNOWN


def detect_backends_per_transform(transforms: list[object]) -> list[Backend | None]:
    """Return a per-transform backend list without raising on mixed backends.

    Each entry is the ``Backend`` for the corresponding transform, or ``None``
    if the transform's module could not be matched to any known backend prefix.
    Unrecognised transforms emit a ``UserWarning``.

    When a direct module-prefix match fails, the function falls back to
    checking the transform's MRO (method resolution order) for any ancestor
    class whose ``__module__`` matches a known backend prefix. This handles
    subclasses defined outside the backend package (e.g. a user-defined
    ``class MyRot(torchvision.transforms.RandomRotation)`` in ``__main__``).

    Note:
        This is a semi-public API accessible via ``fused_transforms._backend``.
        It is not part of the stable public surface and may change without notice.

    Args:
        transforms: List of transform objects.

    Returns:
        List of ``Backend | None``, same length as *transforms*.

    Examples:
        ```pycon
        >>> detect_backends_per_transform([])
        []

        ```

    """
    result: list[Backend | None] = []
    for transform in transforms:
        backend = _classify_transform(transform)
        if backend is None:
            _warn_unrecognized(transform)
        result.append(backend)
    return result


def _lookup_prefix(module: str) -> Backend | None:
    """Return the backend whose registered prefix best (longest) matches *module*, or ``None``.

    Args:
        module: A transform type's ``__module__`` string.

    Returns:
        The ``Backend`` of the longest matching registered prefix, or ``None`` when nothing matches.

    """
    best_len = -1
    best: Backend | None = None
    for entry in _ADAPTER_REGISTRY.values():
        for prefix in entry.prefixes:
            if module.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                best = entry.backend
    return best


def _match_backend(module: str) -> Backend | None:
    """Match a module path to a registered backend prefix.

    Consults the adapter registry (longest-prefix match wins). On a miss, third-party adapters are loaded lazily from
    the entry-point group and the lookup is retried once.

    Args:
        module: The ``__module__`` attribute of a transform type.

    Returns:
        ``Backend`` enum member, or ``None`` if no prefix matches.

    """
    backend = _lookup_prefix(module)
    if backend is not None:
        return backend
    if not _ENTRYPOINTS_LOADED:
        _load_entrypoints()
        return _lookup_prefix(module)
    return None


def _match_backend_from_mro(cls: type) -> Backend | None:
    """Walk the MRO looking for an ancestor whose module matches a known backend.

    Skips ``object`` and the class itself (already checked by the caller via its direct ``__module__``).

    Args:
        cls: The type of the transform.

    Returns:
        ``Backend`` enum member from the first matching ancestor, or ``None``.

    """
    for ancestor in cls.__mro__[1:]:
        if ancestor is object:
            continue
        module = ancestor.__module__ or ""
        backend = _match_backend(module)
        if backend is not None:
            return backend
    return None


def _builtin_adapter_loader(module_name: str, class_name: str) -> Callable[[], TransformAdapter]:
    """Return a loader that imports a built-in adapter module on first use and instantiates its adapter class.

    Args:
        module_name: Dotted adapter module, e.g. ``"fused_transforms.adapters.kornia"``.
        class_name: Adapter class defined there, e.g. ``"KorniaAdapter"``.

    Returns:
        A zero-argument callable producing the adapter instance.

    """

    def load() -> TransformAdapter:
        return cast("TransformAdapter", getattr(import_module(module_name), class_name)())

    return load


#: Built-in backend -> adapter class name defined in ``fused_transforms.adapters.<backend value>``.
_BUILTIN_ADAPTER_CLASSES: dict[Backend, str] = {
    Backend.KORNIA: "KorniaAdapter",
    Backend.ALBUMENTATIONS: "AlbumentationsAdapter",
    Backend.TORCHVISION: "TorchVisionAdapter",
}


def is_builtin_adapter(adapter: object, backend: Backend) -> bool:
    """Return whether ``adapter`` is (a subclass of) the built-in adapter for ``backend``, without importing it.

    An instance of an adapter class implies the module defining that class is already imported, so this consults
    ``sys.modules`` rather than importing: asking whether a Kornia pipeline's adapter is the Albumentations one
    must not import Albumentations.

    Args:
        adapter: Any object, typically a pipeline's or segment's adapter.
        backend: The built-in backend to test against.

    Returns:
        ``True`` when ``adapter`` is an instance of that backend's built-in adapter class.

    Examples:
        ```pycon
        >>> is_builtin_adapter(object(), Backend.KORNIA)
        False

        ```

    """
    class_name = _BUILTIN_ADAPTER_CLASSES.get(backend)
    module = sys.modules.get(f"fused_transforms.adapters.{backend.value}")
    adapter_cls = getattr(module, class_name, None) if module is not None and class_name is not None else None
    return isinstance(adapter_cls, type) and isinstance(adapter, adapter_cls)


def _register_builtins() -> None:
    """Register the three built-in adapters (kornia, torchvision, albumentations) at import, by module prefix only.

    Kept as a function (invoked at module bottom) so import order is explicit and the registry is populated exactly
    once. Nothing here imports an adapter module or a backend library: each entry holds a loader that does so the first
    time its ``adapter`` (or ``capabilities``, read from the adapter class) is needed.

    """
    for backend, class_name in _BUILTIN_ADAPTER_CLASSES.items():
        name = backend.value
        loader = _builtin_adapter_loader(f"fused_transforms.adapters.{name}", class_name)
        _ADAPTER_REGISTRY[name] = _Entry(backend, (f"{name}.",), loader=loader)


_register_builtins()
