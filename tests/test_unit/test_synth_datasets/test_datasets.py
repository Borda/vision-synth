"""Tests for the torch IterableDataset wrapper (synth_datasets.export.datasets)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402 - after the importorskip guard above, by design
import torch.multiprocessing as torch_mp  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from synth_datasets import SyntheticIterableDataset  # noqa: E402
from synth_datasets.core.generator import SyntheticGenerator  # noqa: E402
from synth_datasets.core.sample import Sample  # noqa: E402


def test_len_reports_budget():
    ds = SyntheticIterableDataset(num_images=7, img_size=32, seed=0)
    assert len(ds) == 7


def test_iter_yields_samples():
    ds = SyntheticIterableDataset(num_images=4, img_size=32, seed=0)
    samples = list(ds)
    assert len(samples) == 4
    assert all(isinstance(s, Sample) for s in samples)


def test_iter_is_reproducible():
    ds = SyntheticIterableDataset(num_images=3, img_size=32, seed=5)
    first = [ann.bbox_xyxy for s in ds for ann in s.annotations]
    second = [ann.bbox_xyxy for s in ds for ann in s.annotations]
    assert first == second


def test_dataloader_batches_cover_all_samples():
    ds = SyntheticIterableDataset(num_images=5, img_size=32, seed=0)
    loader = DataLoader(ds, batch_size=2, collate_fn=list)
    batches = list(loader)
    assert sum(len(b) for b in batches) == 5


def test_multiworker_shards_cover_all_samples():
    ds = SyntheticIterableDataset(num_images=6, img_size=32, seed=0)
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=list)
    first = [sample.image.tobytes() for batch in loader for sample in batch]
    second = [sample.image.tobytes() for batch in loader for sample in batch]
    assert len(first) == len(second) == 6
    assert first == second


def _images(dataset: SyntheticIterableDataset) -> list[bytes]:
    """Return a compact, value-sensitive identity for one generated stream."""
    return [sample.image.tobytes() for sample in dataset]


def _worker_info(worker_id: int, num_workers: int) -> SimpleNamespace:
    """Build the worker-info fields the dataset consumes without spawning a worker."""
    return SimpleNamespace(id=worker_id, num_workers=num_workers)


def _loader_digests(dataset: SyntheticIterableDataset) -> list[str]:
    """Return ordered image digests from a two-worker loader for distributed assertions."""
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=list)
    digests: list[str] = []
    for batch in loader:
        for sample in batch:
            digest = hashlib.sha256(sample.image.tobytes())
            digest.update(repr(sample.annotations).encode())
            digests.append(digest.hexdigest())
    return digests


def _gloo_rank_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Generate and persist one rank's two-epoch Gloo acceptance evidence."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        epochs: dict[str, dict[str, list[str] | bool]] = {}
        for epoch in range(2):
            args = {"num_images": 5, "img_size": 32, "seed": 7, "rank": rank, "world_size": world_size, "epoch": epoch}
            first = _loader_digests(SyntheticIterableDataset(**args))
            resumed = _loader_digests(SyntheticIterableDataset(**args))
            epochs[str(epoch)] = {"digests": first, "resumed": resumed, "reproducible": first == resumed}
        dist.barrier()
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(epochs), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"num_images": -1}, ValueError, "num_images must be >= 0"),
        ({"num_images": 1.5}, TypeError, "num_images must be an integer"),
        ({"rank": -1}, ValueError, "rank must be >= 0"),
        ({"rank": 1, "world_size": 1}, ValueError, "rank must be < world_size"),
        ({"world_size": 0}, ValueError, "world_size must be >= 1"),
        ({"epoch": -1}, ValueError, "epoch must be >= 0"),
        ({"epoch": 1.5}, TypeError, "epoch must be an integer"),
    ],
)
def test_rank_epoch_inputs_are_validated(kwargs, error, match):
    with pytest.raises(error, match=match):
        SyntheticIterableDataset(img_size=32, seed=0, **{"num_images": 1, **kwargs})


def test_rank_epoch_inputs_are_read_only():
    ds = SyntheticIterableDataset(num_images=1, img_size=32, seed=0, rank=0, world_size=2, epoch=3)

    with pytest.raises(AttributeError):
        ds.rank = 1
    with pytest.raises(AttributeError):
        ds.world_size = 1
    with pytest.raises(AttributeError):
        ds.epoch = 4


def test_rank_zero_epoch_zero_worker_seeds_preserve_legacy_streams(monkeypatch):
    monkeypatch.setattr(
        "synth_datasets.export.datasets.get_worker_info",
        lambda: _worker_info(worker_id=1, num_workers=2),
    )
    ds = SyntheticIterableDataset(num_images=3, img_size=32, seed=7, rank=0, world_size=2, epoch=0)

    assert _images(ds) == _images(SyntheticGenerator(ds.config).generate(1, seed=8))


def test_rank_and_epoch_make_seeded_streams_distinct_and_repeatable():
    common = {"num_images": 2, "img_size": 32, "seed": 7, "world_size": 2}

    rank_zero = _images(SyntheticIterableDataset(rank=0, epoch=1, **common))
    rank_one = _images(SyntheticIterableDataset(rank=1, epoch=1, **common))
    next_epoch = _images(SyntheticIterableDataset(rank=0, epoch=2, **common))

    assert rank_zero == _images(SyntheticIterableDataset(rank=0, epoch=1, **common))
    assert rank_zero != rank_one
    assert rank_zero != next_epoch


def test_two_rank_two_worker_shards_keep_each_rank_count(monkeypatch):
    counts: list[int] = []
    seeds: list[object] = []
    for rank in range(2):
        ds = SyntheticIterableDataset(num_images=5, img_size=32, seed=7, rank=rank, world_size=2, epoch=1)
        for worker_id in range(2):
            monkeypatch.setattr(
                "synth_datasets.export.datasets.get_worker_info",
                lambda worker_id=worker_id: _worker_info(worker_id=worker_id, num_workers=2),
            )
            count, seed = ds._worker_shard()
            counts.append(count)
            seeds.append(seed)

    assert counts == [3, 2, 3, 2]
    assert sum(counts[:2]) == sum(counts[2:]) == 5
    assert len({repr(seed) for seed in seeds}) == 4


def test_two_rank_two_worker_loaders_have_equal_disjoint_streams():
    rank_streams: list[list[bytes]] = []
    for rank in range(2):
        ds = SyntheticIterableDataset(num_images=5, img_size=32, seed=7, rank=rank, world_size=2, epoch=1)
        loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=list)
        rank_streams.append([sample.image.tobytes() for batch in loader for sample in batch])

    assert [len(stream) for stream in rank_streams] == [5, 5]
    assert set(rank_streams[0]).isdisjoint(rank_streams[1])


def test_two_process_gloo_two_worker_epoch_streams(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch was built without CPU Gloo support")

    init_file = tmp_path / "gloo-init"
    torch_mp.spawn(_gloo_rank_worker, args=(2, str(init_file), str(tmp_path)), nprocs=2, join=True)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]

    streams: list[list[str]] = []
    for epoch in range(2):
        epoch_streams = [record[str(epoch)]["digests"] for record in records]
        assert [len(stream) for stream in epoch_streams] == [5, 5]
        assert all(record[str(epoch)]["reproducible"] for record in records)
        assert set(epoch_streams[0]).isdisjoint(epoch_streams[1])
        streams.extend(epoch_streams)
    assert set(streams[0]).isdisjoint(streams[2])
    assert set(streams[1]).isdisjoint(streams[3])


@pytest.mark.parametrize(
    ("num_images", "expected_counts"),
    [(0, [0, 0, 0]), (2, [1, 1, 0]), (5, [2, 2, 1])],
)
def test_worker_shards_cover_zero_small_and_remainder_budgets(monkeypatch, num_images, expected_counts):
    ds = SyntheticIterableDataset(num_images=num_images, img_size=32, seed=7, rank=1, world_size=2, epoch=1)
    actual_counts: list[int] = []
    for worker_id in range(3):
        monkeypatch.setattr(
            "synth_datasets.export.datasets.get_worker_info",
            lambda worker_id=worker_id: _worker_info(worker_id=worker_id, num_workers=3),
        )
        count, _ = ds._worker_shard()
        actual_counts.append(count)

    assert actual_counts == expected_counts
    assert sum(actual_counts) == num_images


def _loader_for_epoch(epoch: int, *, persistent_workers: bool) -> DataLoader:
    """Build the documented fresh-loader recipe for one immutable dataset epoch."""
    if persistent_workers:
        raise ValueError("the epoch-rebuild recipe requires persistent_workers=False")
    dataset = SyntheticIterableDataset(num_images=2, img_size=32, seed=7, rank=0, world_size=2, epoch=epoch)
    return DataLoader(dataset, batch_size=1, collate_fn=list, persistent_workers=False)


def test_epoch_boundary_resume_rebuilds_dataset_and_rejects_persistent_workers():
    with pytest.raises(ValueError, match="persistent_workers=False"):
        _loader_for_epoch(1, persistent_workers=True)

    first = _loader_for_epoch(3, persistent_workers=False)
    resumed = _loader_for_epoch(3, persistent_workers=False)
    next_epoch = _loader_for_epoch(4, persistent_workers=False)

    assert first.dataset is not resumed.dataset
    assert _images(first.dataset) == _images(resumed.dataset)
    assert _images(first.dataset) != _images(next_epoch.dataset)


def test_forwarded_config_kwargs_normalize_a_raw_fill():
    """`**config_kwargs` reaches `SyntheticConfig`, so a raw `(r, g, b)` fill normalizes on the streaming path too.

    This wrapper forwards its keywords verbatim, which is the reason `SyntheticConfig` coerces at construction rather
    than trusting its callers — `class_mode="shape"` arrives here as a bare string for exactly the same reason. A fill
    is the third spelling to go through that boundary, and the only one whose failure would surface far away: an un-
    normalized tuple has no `.rgb`, so it would raise inside the generator's draw call rather than at construction.

    """
    ds = SyntheticIterableDataset(num_images=1, img_size=32, seed=0, colors=((255, 215, 0),))

    assert [fill.label for fill in ds.config.colors] == ["ffd700"]
    assert len(list(ds)) == 1
