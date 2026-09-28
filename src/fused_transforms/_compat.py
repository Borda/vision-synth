"""Optional-dependency availability, answered lazily.

``import fused_transforms`` must stay cheap for a user of one backend, and it runs again in every DataLoader worker
started with ``spawn`` (the macOS and Windows default). So availability is split in two:

- :func:`backend_installed` asks the import system's spec finder whether a backend is present. It never imports the
  backend, so the module-level flags below (``_KORNIA_AVAILABLE`` and friends) cost a directory lookup each.
- :func:`import_backend` performs the real import on first use and caches the module. A backend that is installed but
  fails with :class:`ImportError` (a broken binary wheel, a missing shared library) emits a :class:`UserWarning`
  carrying the original error text and is then treated as unavailable, instead of silently looking uninstalled.
  Any other exception raised while importing propagates unchanged.

The flags therefore mean "installed". Code about to *use* a backend asks :func:`backend_available`, which imports it.

"""

from __future__ import annotations

import warnings
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

#: Real-import results by module name: the module, or ``None`` when it is absent or failed to import.
_LOADED: dict[str, ModuleType | None] = {}


def backend_installed(name: str) -> bool:
    """Return whether an optional backend is installed, without importing it.

    Args:
        name: Top-level module name, e.g. ``"kornia"``.

    Returns:
        ``True`` when the import system can locate the module.

    Examples:
        ```pycon
        >>> backend_installed("definitely_not_an_installed_backend")
        False

        ```

    """
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        # ValueError: a module in sys.modules without __spec__; ImportError: a broken parent package.
        return False


def import_backend(name: str) -> ModuleType | None:
    """Import an optional backend on first use and cache the outcome.

    Args:
        name: Top-level module name, e.g. ``"albumentations"``.

    Returns:
        The imported module, or ``None`` when the backend is not installed or is installed but fails to import.
        The latter case warns once, with the original :class:`ImportError` text.

    Examples:
        ```pycon
        >>> import_backend("definitely_not_an_installed_backend") is None
        True

        ```

    """
    if name in _LOADED:
        return _LOADED[name]
    module: ModuleType | None = None
    if backend_installed(name):
        try:
            module = import_module(name)
        except ImportError as err:
            warnings.warn(
                f"Optional backend {name!r} is installed but failed to import, so it is treated as unavailable. "
                f"Fix or reinstall it to use it. Original error: {err}",
                UserWarning,
                stacklevel=2,
            )
    _LOADED[name] = module
    return module


def backend_available(name: str) -> bool:
    """Return whether an optional backend is installed *and* importable, importing it if needed.

    Args:
        name: Top-level module name, e.g. ``"cv2"``.

    Returns:
        ``True`` when :func:`import_backend` returns a module.

    Examples:
        ```pycon
        >>> backend_available("definitely_not_an_installed_backend")
        False

        ```

    """
    return import_backend(name) is not None


# "Installed" flags: spec lookups only, safe to evaluate at import time.
_KORNIA_AVAILABLE: bool = backend_installed("kornia")
_TORCHVISION_AVAILABLE: bool = backend_installed("torchvision")
_ALBUMENTATIONS_AVAILABLE: bool = backend_installed("albumentations")
_CV2_AVAILABLE: bool = backend_installed("cv2")


def __getattr__(name: str) -> bool:
    """Resolve ``_TORCHVISION_V2_AVAILABLE`` lazily: locating a submodule imports its parent package."""
    if name == "_TORCHVISION_V2_AVAILABLE":
        return _TORCHVISION_AVAILABLE and backend_installed("torchvision.transforms.v2")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
