"""Tests for lazy optional-backend loading.

``import fused_transforms`` must not import any optional backend (Kornia, Albumentations, OpenCV, TorchVision): the
built-in adapters register by module prefix and load on first use, availability flags come from the import system's spec
finder, and OpenCV loads only when a cv2 warp runs. A backend that is installed but fails to import must say so with the
original error instead of silently looking absent.

"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

from fused_transforms import _compat
from fused_transforms.affine import segment

_OPTIONAL_BACKENDS = ("kornia", "albumentations", "cv2", "torchvision")


def _modules_loaded_after(code: str) -> set[str]:
    """Run ``code`` in a fresh interpreter and return which optional backends ended up in ``sys.modules``."""
    probe = f"{code}\nimport sys\nprint(','.join(m for m in {_OPTIONAL_BACKENDS!r} if m in sys.modules))"
    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter and a literal program
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    return {name for name in result.stdout.strip().split(",") if name}


def test_package_import_loads_no_optional_backend() -> None:
    """A fresh ``import fused_transforms`` leaves every optional backend unimported.

    Each DataLoader worker started with ``spawn`` (the macOS and Windows default) pays this import again, so a user of
    one backend must not pay for all of them.

    """
    assert _modules_loaded_after("import fused_transforms") == set()


def test_kornia_pipeline_loads_only_kornia() -> None:
    """Building and running a one-op Kornia pipeline imports Kornia but no other optional backend.

    Adapter classification uses a non-importing check, so asking whether a Kornia adapter is the Albumentations or
    TorchVision one must not import those libraries.

    """
    pytest.importorskip("kornia")
    code = (
        "import torch, kornia.augmentation as K\n"
        "from fused_transforms import Compose\n"
        "Compose([K.RandomRotation(10.0)])(torch.rand(2, 3, 8, 8))"
    )
    assert _modules_loaded_after(code) == {"kornia"}


def test_broken_backend_warns_with_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed backend whose import fails warns once, carrying the original error text."""
    monkeypatch.setattr(_compat, "_LOADED", {})
    monkeypatch.setattr(_compat, "backend_installed", lambda name: True)

    def _broken(name: str) -> object:
        raise ImportError(f"dlopen({name}/_spropack.so): symbol not found")

    monkeypatch.setattr(_compat, "import_module", _broken)
    with pytest.warns(UserWarning, match=r"(?s)'albumentations' is installed but failed to import.*_spropack"):
        assert _compat.import_backend("albumentations") is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _compat.import_backend("albumentations") is None  # cached: no second warning
    assert _compat.backend_available("albumentations") is False


def test_missing_backend_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend that is simply not installed returns ``None`` without a warning."""
    monkeypatch.setattr(_compat, "_LOADED", {})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _compat.import_backend("definitely_not_an_installed_backend") is None


def test_flags_answer_from_spec_without_importing() -> None:
    """The legacy boolean flags mirror whether each backend is installed."""
    assert _compat._KORNIA_AVAILABLE is _compat.backend_installed("kornia")
    assert _compat._ALBUMENTATIONS_AVAILABLE is _compat.backend_installed("albumentations")
    assert _compat._CV2_AVAILABLE is _compat.backend_installed("cv2")


def test_cv2_flag_literals_match_opencv() -> None:
    """The OpenCV flag values spelled out in ``segment`` (to avoid importing cv2) equal the module's constants."""
    cv2 = pytest.importorskip("cv2")
    assert segment._CV2_INTERP == {
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "bicubic": cv2.INTER_CUBIC,
    }
    assert segment._CV2_BORDER == {
        "zeros": cv2.BORDER_CONSTANT,
        "border": cv2.BORDER_REPLICATE,
        "reflection": cv2.BORDER_REFLECT_101,
    }
    assert cv2.WARP_INVERSE_MAP == segment._CV2_WARP_INVERSE_MAP
