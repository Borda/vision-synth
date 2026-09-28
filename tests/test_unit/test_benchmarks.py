"""Regression tests for benchmark measurement contracts."""

from __future__ import annotations

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fused_transforms.affine.segment import ExactAffineSegment

_ROOT = Path(__file__).parents[2]


class TimelineAction(Enum):
    """Profiler action names at the experiment script's external boundary."""

    PREEXISTING = 1
    CREATE = 2
    INCREMENT_VERSION = 3
    DESTROY = 4


def _load_experiment(name: str):
    """Load an experiment script without adding it to the package API."""
    path = _ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_memory():
    """Load the memory benchmark once for timeline-accounting checks."""
    pytest.importorskip(
        "resource", reason="bench_memory requires the POSIX resource module", exc_type=ModuleNotFoundError
    )
    pytest.importorskip(
        "albumentations", reason="bench_memory imports the Albumentations benchmark path", exc_type=ModuleNotFoundError
    )
    pytest.importorskip(
        "kornia.augmentation", reason="bench_memory imports the Kornia benchmark path", exc_type=ModuleNotFoundError
    )
    pytest.importorskip(
        "torchvision.transforms.v2",
        reason="bench_memory imports the TorchVision benchmark path",
        exc_type=ModuleNotFoundError,
    )
    pytest.importorskip("rich", reason="bench_memory imports Rich table output", exc_type=ModuleNotFoundError)
    return _load_experiment("bench_memory")


@pytest.fixture(scope="module")
def bench_gpu_batch():
    """Load the throughput benchmark for error-reporting checks."""
    pytest.importorskip("albumentations", exc_type=ModuleNotFoundError)
    pytest.importorskip("kornia.augmentation", exc_type=ModuleNotFoundError)
    pytest.importorskip("torchvision.transforms.v2", exc_type=ModuleNotFoundError)
    pytest.importorskip("rich", exc_type=ModuleNotFoundError)
    return _load_experiment("bench_gpu_batch")


@pytest.fixture(scope="module")
def bench_rfdetr_shape():
    """Load the detection-shape benchmark once for endpoint checks."""
    pytest.importorskip(
        "albumentations",
        reason="bench_rfdetr_shape benchmarks Albumentations detection pipelines",
        exc_type=ModuleNotFoundError,
    )
    return _load_experiment("bench_rfdetr_shape")


def _timeline_profile(*events: tuple[TimelineAction, int]) -> SimpleNamespace:
    """Build the profiler fragment consumed by the timeline accounting helper."""
    timeline = tuple((index, action, None, nbytes) for index, (action, nbytes) in enumerate(events))
    return SimpleNamespace(_memory_profile=lambda: SimpleNamespace(timeline=timeline))


@pytest.mark.parametrize(
    ("events", "expected_peak"),
    [
        pytest.param(
            (
                (TimelineAction.CREATE, 4),
                (TimelineAction.DESTROY, 4),
                (TimelineAction.CREATE, 4),
                (TimelineAction.DESTROY, 4),
            ),
            4,
            id="sequential",
        ),
        pytest.param(
            (
                (TimelineAction.CREATE, 4),
                (TimelineAction.CREATE, 4),
                (TimelineAction.DESTROY, 4),
                (TimelineAction.DESTROY, 4),
            ),
            8,
            id="overlapping",
        ),
    ],
)
def test_timeline_stats_uses_profiler_actions(bench_memory, events, expected_peak):
    """Only CREATE events allocate and DESTROY events release live bytes."""
    stats = bench_memory._timeline_stats(_timeline_profile(*events))

    assert stats.live_peak_bytes == expected_peak
    assert stats.incremental_peak_bytes == expected_peak
    assert stats.allocation_count == 2


def test_timeline_stats_ignores_version_increments(bench_memory):
    """Version changes do not allocate a second physical tensor buffer."""
    stats = bench_memory._timeline_stats(
        _timeline_profile(
            (TimelineAction.CREATE, 4),
            (TimelineAction.INCREMENT_VERSION, 4),
            (TimelineAction.DESTROY, 4),
        )
    )

    assert stats.live_peak_bytes == 4
    assert stats.allocation_count == 1


def test_timeline_stats_tracks_preexisting_baseline_separately(bench_memory):
    """Profiler preexisting memory stays out of the measured allocation count."""
    stats = bench_memory._timeline_stats(
        _timeline_profile((TimelineAction.PREEXISTING, 8), (TimelineAction.CREATE, 4), (TimelineAction.DESTROY, 4))
    )

    assert stats.preexisting_bytes == 8
    assert stats.live_peak_bytes == 12
    assert stats.incremental_peak_bytes == 4
    assert stats.allocation_count == 1


def test_timeline_stats_reports_an_unavailable_profiler(bench_memory):
    """A missing timeline is an error, never a believable zero-memory sample."""
    unavailable = SimpleNamespace(_memory_profile=lambda: (_ for _ in ()).throw(ValueError("stack data missing")))

    with pytest.raises(RuntimeError, match="Profiler memory timeline unavailable"):
        bench_memory._timeline_stats(unavailable)


def test_timeline_stats_accepts_the_installed_profiler_events(bench_memory):
    """The action-name boundary accepts the installed torch profiler event type."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        value = torch.zeros(1024)
        del value

    stats = bench_memory._timeline_stats(prof)

    assert stats.live_peak_bytes == 4096
    assert stats.incremental_peak_bytes == 4096
    assert stats.allocation_count == 1


def test_gpu_benchmark_distinguishes_runtime_failure_from_unsupported_case(bench_gpu_batch):
    """A broken supported path must not be reported as an expected skip."""
    device = bench_gpu_batch.Device("cpu", torch.device("cpu"), lambda: None)
    case = bench_gpu_batch.CaseKey(SimpleNamespace(label="broken", nb_geom=1), "kornia", device, 1)
    config = SimpleNamespace(warmup=0, measure=1)

    def failing_call():
        raise RuntimeError("kernel failed")

    def unsupported_call():
        raise bench_gpu_batch.SkipCase("unsupported")

    failed = bench_gpu_batch._measure_mode(failing_call, case, "fused", config)
    skipped = bench_gpu_batch._measure_mode(unsupported_call, case, "native", config)

    assert failed["status"] == "error"
    assert failed["error"] == "RuntimeError: kernel failed"
    assert skipped["status"] == "skipped"
    assert skipped["skip_reason"] == "unsupported"


def test_memory_benchmark_distinguishes_runtime_failure_from_unsupported_case(bench_memory):
    """Profiler-path failures must be visible separately from device skips."""
    device = bench_memory.Device("cpu", torch.device("cpu"), lambda: None, counter="torch-profiler timeline")
    case = bench_memory.CaseKey(SimpleNamespace(label="broken", nb_geom=1), "kornia", device, 1)

    def failing_call():
        raise RuntimeError("profiler path failed")

    def unsupported_call():
        raise bench_memory.SkipCase("unsupported")

    failed = bench_memory._measure_mode(failing_call, case, "fused", warmup=0)
    skipped = bench_memory._measure_mode(unsupported_call, case, "native", warmup=0)

    assert failed["status"] == "error"
    assert failed["error"] == "RuntimeError: profiler path failed"
    assert skipped["status"] == "skipped"
    assert skipped["skip_reason"] == "unsupported"


def test_gpu_benchmark_has_runnable_exact_flip_probe(bench_gpu_batch):
    """The device sweep must expose the exact path where sync cost may occur."""
    sequences, _ = bench_gpu_batch._load_sequences()
    exact = next(seq for seq in sequences if seq.label == "e02_exact_flips")
    device = bench_gpu_batch.Device("cpu", torch.device("cpu"), lambda: None)
    native, fused = bench_gpu_batch._tensor_thunks(exact, "kornia", device, 1)
    pipeline = bench_gpu_batch._build_fused(exact.kornia, exact.reorder)

    assert exact.nb_geom == 3
    assert isinstance(pipeline._segments[0], ExactAffineSegment)
    assert native().shape == (1, 3, 256, 256)
    assert fused().shape == (1, 3, 256, 256)


def _assert_model_ready_endpoint(result: tuple[torch.Tensor, torch.Tensor, torch.Tensor], resolution: int) -> None:
    """Assert the benchmark's common image/box/label training boundary."""
    image, boxes, labels = result
    assert image.shape == (1, 3, resolution, resolution)
    assert image.dtype is torch.float32
    assert image.device.type == "cpu"
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert boxes.ndim == 2
    assert boxes.shape[1] == 4
    assert boxes.dtype is torch.float32
    assert boxes.device == image.device
    assert labels.ndim == 1
    assert labels.shape[0] == boxes.shape[0]
    assert labels.dtype is torch.int64
    assert labels.device == image.device


def test_detection_endpoint_preserves_empty_survival_shape(bench_rfdetr_shape):
    """A fully cropped sample remains a valid empty ``(0, 4)`` detection batch."""
    result = bench_rfdetr_shape._model_ready_endpoint(
        np.zeros((32, 32, 3), dtype=np.uint8), np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    )

    _assert_model_ready_endpoint(result, resolution=32)


@pytest.mark.parametrize(
    "builder",
    [
        lambda module: module._albu_step(32, "two_op"),
        lambda module: module._fuse_step(32, "cv2", "two_op"),
        lambda module: module._fuse_step(32, "torch", "two_op"),
        lambda module: module._fuse_tensor_step(32, "cv2", "two_op"),
    ],
    ids=["albumentations", "fuse-cv2", "fuse-torch", "fuse-cv2-tensor-input"],
)
def test_detection_benchmark_returns_one_model_ready_endpoint(bench_rfdetr_shape, builder):
    """All timed variants finish with the same device, tensor layout, and target metadata."""
    _assert_model_ready_endpoint(builder(bench_rfdetr_shape)(), resolution=32)


@pytest.mark.parametrize(
    "builder",
    [
        lambda module: module._albu_step(32, "four_op"),
        lambda module: module._fuse_step(32, "cv2", "four_op"),
        lambda module: module._fuse_step(32, "torch", "four_op"),
        lambda module: module._fuse_tensor_step(32, "cv2", "four_op"),
    ],
    ids=["albumentations", "fuse-cv2", "fuse-torch", "fuse-cv2-tensor-input"],
)
def test_detection_benchmark_rebuilds_reproduce_each_variant(bench_rfdetr_shape, builder):
    """Resetting the benchmark seed reproduces each variant's multi-call endpoint sequence."""
    bench_rfdetr_shape._seed_everything()
    first_step = builder(bench_rfdetr_shape)
    first = [first_step() for _ in range(3)]
    bench_rfdetr_shape._seed_everything()
    second_step = builder(bench_rfdetr_shape)
    second = [second_step() for _ in range(3)]

    for first_result, second_result in zip(first, second, strict=True):
        for first_value, second_value in zip(first_result, second_result, strict=True):
            assert torch.equal(first_value, second_value)
