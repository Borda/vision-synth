"""PyTorch ``IterableDataset`` streaming synthetic samples for training feeds.

Wraps :meth:`~synth_datasets.core.generator.SyntheticGenerator.generate`
so samples can be pulled straight into a ``DataLoader`` with no disk round-trip.
The dataset is worker-shard aware: under multi-worker loading each worker produces
a disjoint, deterministically-seeded slice of the requested count.

Because object annotations are ragged (a variable number per image), pass a custom
``collate_fn`` to ``DataLoader`` (e.g. ``collate_fn=list``) — the default collate
cannot stack variable-length targets.

Examples:
    ```pycon
    >>> from torch.utils.data import DataLoader
    >>> from synth_datasets.export.datasets import SyntheticIterableDataset
    >>> ds = SyntheticIterableDataset(num_images=4, img_size=32, seed=0)
    >>> loader = DataLoader(ds, batch_size=2, collate_fn=list)
    >>> batches = list(loader)
    >>> sum(len(b) for b in batches)
    4

    ```

"""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from synth_datasets.core.config import SyntheticConfig
from synth_datasets.core.generator import SyntheticGenerator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from synth_datasets.core.sample import Sample


_StreamSeed = int | np.random.SeedSequence | None


def _non_negative_int(value: object, name: str, minimum: int = 0) -> int:
    """Return an integer input after rejecting booleans and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    integer = index(value)
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {integer}")
    return integer


class SyntheticIterableDataset(IterableDataset["Sample"]):
    """Stream synthetic :class:`Sample` objects into a PyTorch training loop.

    Args:
        num_images: Number of samples yielded by each rank for this epoch.
        config: Full :class:`SyntheticConfig`; when given, ``config_kwargs`` are ignored.
        seed: Base seed for reproducible streams; ``None`` uses fresh entropy.
        rank: Distributed-process rank supplied by the training process.
        world_size: Number of distributed processes supplied by the training process.
        epoch: Immutable epoch identity for this dataset instance.
        **config_kwargs: Extra :class:`SyntheticConfig` fields (e.g. ``img_size``,
            ``class_mode``) used only when ``config`` is not supplied.

    Examples:
        ```pycon
        >>> from synth_datasets.export.datasets import SyntheticIterableDataset
        >>> ds = SyntheticIterableDataset(num_images=3, img_size=32, seed=1)
        >>> samples = list(ds)
        >>> len(samples)
        3

        ```

    """

    def __init__(
        self,
        num_images: int,
        config: SyntheticConfig | None = None,
        seed: int | None = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
        **config_kwargs: Any,  # noqa: ANN401 - forwarded verbatim to SyntheticConfig
    ) -> None:
        """Store immutable stream identity and build the underlying generator."""
        super().__init__()
        self.num_images = _non_negative_int(num_images, "num_images")
        self._world_size = _non_negative_int(world_size, "world_size", minimum=1)
        self._rank = _non_negative_int(rank, "rank")
        if self._rank >= self._world_size:
            raise ValueError(f"rank must be < world_size, got rank={self._rank}, world_size={self._world_size}")
        self._epoch = _non_negative_int(epoch, "epoch")
        self.config = config or SyntheticConfig(**config_kwargs)
        self.seed = seed
        self._generator = SyntheticGenerator(self.config)

    @property
    def rank(self) -> int:
        """Return the immutable distributed rank for this stream."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Return the immutable distributed topology size for this stream."""
        return self._world_size

    @property
    def epoch(self) -> int:
        """Return the immutable epoch identity for this stream."""
        return self._epoch

    def __len__(self) -> int:
        """Return the per-epoch sample count."""
        return self.num_images

    def _worker_shard(self) -> tuple[int, _StreamSeed]:
        """Return this worker's count and seed namespace within the rank's budget."""
        info = get_worker_info()
        num_workers = 1 if info is None else info.num_workers
        worker_id = 0 if info is None else info.id
        per, remainder = divmod(self.num_images, num_workers)
        count = per + (1 if worker_id < remainder else 0)
        if self.seed is None:
            return count, None
        if self.rank == 0 and self.epoch == 0:
            return count, self.seed + worker_id
        seed = np.random.SeedSequence(self.seed, spawn_key=(self.rank, self.epoch, worker_id))
        return count, seed

    def __iter__(self) -> Iterator[Sample]:
        """Yield this worker's deterministically-seeded slice of samples."""
        count, seed = self._worker_shard()
        return self._generator.generate(count, seed=seed)
