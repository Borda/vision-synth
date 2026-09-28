"""Backend adapters for vision-synth.

Each adapter implements the ``TransformAdapter`` protocol to bridge
framework-specific transforms to the fused affine engine.

The adapter classes are exported lazily: each adapter module imports its backend library, so importing this package
(which any ``fused_transforms.adapters.<name>`` import does first) must not pull in every backend. Accessing
``KorniaAdapter`` imports Kornia and nothing else.

Examples:
    ```pycon
    >>> from fused_transforms.adapters import KorniaAdapter
    >>> adapter = KorniaAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...KorniaAdapter...>

    ```

"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from fused_transforms._backend import register_adapter

if TYPE_CHECKING:
    from fused_transforms.adapters.albumentations import AlbumentationsAdapter
    from fused_transforms.adapters.kornia import KorniaAdapter
    from fused_transforms.adapters.torchvision import TorchVisionAdapter

__all__ = ["AlbumentationsAdapter", "KorniaAdapter", "TorchVisionAdapter", "register_adapter"]

#: Lazily exported adapter class name -> submodule that defines it.
_LAZY_ADAPTERS: dict[str, str] = {
    "AlbumentationsAdapter": "albumentations",
    "KorniaAdapter": "kornia",
    "TorchVisionAdapter": "torchvision",
}


def __getattr__(name: str) -> type:
    """Import an adapter class's submodule on first access (PEP 562)."""
    submodule = _LAZY_ADAPTERS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    adapter_cls: type = getattr(import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = adapter_cls  # later lookups skip __getattr__
    return adapter_cls


def __dir__() -> list[str]:
    """List the lazy exports alongside the module's own names."""
    return sorted({*globals(), *__all__})
