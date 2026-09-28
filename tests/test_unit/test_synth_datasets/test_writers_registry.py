"""``generate_dataset(fmt=...)`` resolves every key registered with ``register_writer``.

It used to coerce ``fmt`` through ``OutputFormat(fmt)`` before looking the writer up, so a custom format registered
exactly as ``register_writer``'s docstring shows was rejected with ``'<key>' is not a valid OutputFormat``.

"""

from __future__ import annotations

import pytest

from synth_datasets import OutputFormat, YoloWriter, generate_dataset, register_writer
from synth_datasets.export import writers


@pytest.fixture
def isolated_registry(monkeypatch):
    """Give each test its own copy of the writer registry so a registration never leaks."""
    monkeypatch.setattr(writers, "_WRITERS", dict(writers._WRITERS))


def test_generate_dataset_resolves_a_registered_writer_key(tmp_path, isolated_registry):
    written: list[list[str]] = []

    class RecordingWriter(YoloWriter):
        def write(self, splits, output_dir):
            written.append(list(splits))
            super().write(splits, output_dir)

    register_writer("recording-yolo", RecordingWriter)

    counts = generate_dataset(tmp_path, num_images=3, fmt="recording-yolo", img_size=32, seed=0)

    assert written == [list(counts)]
    assert (tmp_path / "data.yaml").is_file()


@pytest.mark.parametrize("fmt", [OutputFormat.COCO, "coco", OutputFormat.YOLO, "yolo"])
def test_builtin_formats_still_resolve_by_member_and_value(tmp_path, fmt):
    assert generate_dataset(tmp_path, num_images=1, fmt=fmt, img_size=32, seed=0) == {"train": 1}


def test_unknown_format_lists_known_ones_and_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="known formats"):
        generate_dataset(tmp_path / "ds", num_images=1, fmt="no-such-format", img_size=32, seed=0)
    assert not (tmp_path / "ds").exists()
