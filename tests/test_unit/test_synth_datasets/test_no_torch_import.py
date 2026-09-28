"""Structural guarantee: `import synth_datasets` and dataset generation never pull in torch or fused_transforms."""

from __future__ import annotations

import subprocess
import sys

_SCRIPT = """
import sys
import tempfile
from pathlib import Path

import numpy as np
import synth_datasets
from PIL import Image
from synth_datasets.families.animals import AnimalShape
from synth_datasets.content.backgrounds import ImageBackground
from synth_datasets.content.degradations import JPEG, GaussianBlur


def _forbidden_loaded():
    return [name for name in ("torch", "fused_transforms") if name in sys.modules]


assert not _forbidden_loaded(), f"importing synth_datasets pulled in {_forbidden_loaded()}"

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # Detection + YOLO writer.
    synth_datasets.generate_dataset(tmp_path / "yolo_detect", num_images=2, fmt="yolo", task="detection", seed=0)
    assert not _forbidden_loaded(), f"yolo/detection pulled in {_forbidden_loaded()}"

    # Segmentation + COCO writer.
    synth_datasets.generate_dataset(tmp_path / "coco_seg", num_images=2, fmt="coco", task="segmentation", seed=0)
    assert not _forbidden_loaded(), f"coco/segmentation pulled in {_forbidden_loaded()}"

    # Oriented bounding boxes.
    synth_datasets.generate_dataset(tmp_path / "coco_obb", num_images=2, fmt="coco", task="obb", seed=0)
    assert not _forbidden_loaded(), f"coco/obb pulled in {_forbidden_loaded()}"

    # Keypoints, which need a single keypoint-bearing shape family.
    synth_datasets.generate_dataset(
        tmp_path / "coco_kpts", num_images=2, fmt="coco", task="keypoints", shapes=(AnimalShape.DUCK,), seed=0
    )
    assert not _forbidden_loaded(), f"coco/keypoints pulled in {_forbidden_loaded()}"

    # ImageBackground, the one background that reads files the package does not ship.
    pictures = tmp_path / "pictures"
    pictures.mkdir()
    Image.fromarray(np.full((40, 40, 3), 128, np.uint8)).save(pictures / "a.png")
    synth_datasets.generate_dataset(
        tmp_path / "yolo_imgbg",
        num_images=1,
        fmt="yolo",
        task="detection",
        background=ImageBackground(pictures),
        img_size=32,
        seed=0,
    )
    assert not _forbidden_loaded(), f"ImageBackground path pulled in {_forbidden_loaded()}"

    # Degradations whose `apply` methods defer `from PIL import ...` to call time instead of module import
    # time — the exact style a stray torch import could hide behind, since sys.modules only shows what
    # actually ran.
    synth_datasets.generate_dataset(
        tmp_path / "yolo_degrade",
        num_images=1,
        fmt="yolo",
        task="detection",
        degrade=(GaussianBlur(), JPEG()),
        img_size=32,
        seed=0,
    )
    assert not _forbidden_loaded(), f"degradations path pulled in {_forbidden_loaded()}"

print("OK")
"""


def test_importing_synth_datasets_does_not_import_torch_or_fused_transforms() -> None:
    """`import synth_datasets` and dataset generation never touch `sys.modules['torch']` or `['fused_transforms']`.

    Runs in a fresh subprocess so an already-torch-loaded test session cannot mask a regression, and drives every
    format/task combination plus the two code paths with their own deferred, call-time imports (`ImageBackground` and
    the `GaussianBlur`/`JPEG` degradations, both of which defer `from PIL import ...` to inside a method rather than the
    module top) — a torch import added in that same deferred style would be invisible to a probe that only imports the
    package and writes one plain YOLO/detection dataset, which is all the previous version of this test exercised.

    """
    result = subprocess.run(  # noqa: S603 - fixed script, no untrusted input
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
