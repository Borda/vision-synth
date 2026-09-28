"""Backend dispatch: canonical-op resolution, output-format conversion, opt-in passthrough substitution.

Every submodule here is imported lazily by its caller, never at package import time, so importing
:mod:`fused_transforms.dispatch` on its own never pulls in an optional backend. This package intentionally re-exports
nothing, so importing it does not defeat that laziness.

"""

from __future__ import annotations
