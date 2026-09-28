"""Streaming tests: writers consume single-pass generators; facade stays lazy."""

from __future__ import annotations

import json

from synth_datasets import generate_dataset
from synth_datasets.core.config import SyntheticConfig, Task, class_vocabulary
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.export.writers import CocoWriter, YoloWriter


def test_coco_writer_consumes_a_generator(tmp_path):
    """CocoWriter writes correctly from a single-pass generator (not a list)."""
    gen = SyntheticGenerator(SyntheticConfig(img_size=48, min_objects=1, max_objects=3))
    vocabulary = class_vocabulary(SyntheticConfig().class_mode, SyntheticConfig().shapes)
    CocoWriter(Task.DETECTION, vocabulary).write({"train": gen.generate(4, seed=0)}, tmp_path)
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    assert len(doc["images"]) == 4


def test_yolo_writer_consumes_a_generator(tmp_path):
    """YoloWriter writes correctly from a single-pass generator (not a list)."""
    gen = SyntheticGenerator(SyntheticConfig(img_size=48, min_objects=1, max_objects=3))
    vocabulary = class_vocabulary(SyntheticConfig().class_mode, SyntheticConfig().shapes)
    YoloWriter(Task.DETECTION, vocabulary).write({"train": gen.generate(4, seed=0)}, tmp_path)
    assert len(list((tmp_path / "images" / "train").glob("*.jpg"))) == 4


def test_facade_streams_via_lazy_source(tmp_path, monkeypatch):
    """The facade drives a single lazy generate() stream, handed to the writer un-materialized."""
    calls = {"n": 0}
    real_generate = SyntheticGenerator.generate

    def counting_generate(self, num_images, seed=None):
        calls["n"] += 1
        stream = real_generate(self, num_images, seed=seed)
        assert hasattr(stream, "__next__")  # lazy generator, not a pre-built list
        return stream

    monkeypatch.setattr(SyntheticGenerator, "generate", counting_generate)
    counts = generate_dataset(tmp_path, num_images=6, fmt="yolo", img_size=32, seed=0)
    assert calls["n"] == 1
    assert sum(counts.values()) == 6
