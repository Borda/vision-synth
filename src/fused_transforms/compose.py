"""Compatibility import surface for fused augmentation pipelines.

The implementation lives in :mod:`fused_transforms.core.pipeline`. This module forwards both public and historical
private attributes so existing imports and pickle payloads that reference ``fused_transforms.compose`` remain valid.

"""

from __future__ import annotations

from dataclasses import dataclass

from fused_transforms.affine.segment import _OpaqueBorderModeTransform  # noqa: F401
from fused_transforms.core import pipeline as _pipeline

__all__ = [name for name in dir(_pipeline) if not name.startswith("_")] + ["dataclass"]


def __getattr__(name: str) -> object:
    """Forward compatibility lookups to the implementation module."""
    return getattr(_pipeline, name)


def __dir__() -> list[str]:
    """Return the compatibility surface together with module metadata."""
    return sorted(set(globals()) | set(dir(_pipeline)))
