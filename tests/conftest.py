"""Shared pytest fixtures for vision-synth test suite.

``torch`` is an optional dependency (the ``torch`` extra) since ``synth_datasets`` needs none of it. This module
therefore imports it defensively so collection still works in a torch-free environment; fixtures that hand out tensors
use :func:`pytest.importorskip` to skip cleanly if a test happens to request one there.

"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset all random seeds (torch, numpy, stdlib random, CUDA) before each test."""
    if torch is not None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(autouse=True)
def disable_interactive_prompts() -> None:
    """Keep test runs non-interactive across dependency versions."""
    old_breakpoint = os.environ.get("PYTHONBREAKPOINT")
    os.environ["PYTHONBREAKPOINT"] = "0"

    restore_kornia = None
    with contextlib.suppress(ImportError):
        from kornia.config import InstallationMode, kornia_config

        restore_kornia = kornia_config.lazyloader.installation_mode
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE

    try:
        yield
    finally:
        if old_breakpoint is None:
            os.environ.pop("PYTHONBREAKPOINT", None)
        else:
            os.environ["PYTHONBREAKPOINT"] = old_breakpoint

        if restore_kornia is not None:
            from kornia.config import kornia_config

            kornia_config.lazyloader.installation_mode = restore_kornia


@pytest.fixture
def image8x8_batch1() -> torch.Tensor:
    """Return a (1, 3, 8, 8) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(1, 3, 8, 8)


@pytest.fixture
def image8x8_batch2() -> torch.Tensor:
    """Return a (2, 3, 8, 8) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(2, 3, 8, 8)


@pytest.fixture
def image16x16_batch1() -> torch.Tensor:
    """Return a (1, 3, 16, 16) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(1, 3, 16, 16)


@pytest.fixture
def image16x16_batch2() -> torch.Tensor:
    """Return a (2, 3, 16, 16) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(2, 3, 16, 16)


@pytest.fixture
def image32x32_batch2() -> torch.Tensor:
    """Return a (2, 3, 32, 32) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def image32x32_batch4() -> torch.Tensor:
    """Return a (4, 3, 32, 32) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(4, 3, 32, 32)


@pytest.fixture
def image64x64_batch2() -> torch.Tensor:
    """Return a (2, 3, 64, 64) float32 image tensor."""
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    return torch.rand(2, 3, 64, 64)
