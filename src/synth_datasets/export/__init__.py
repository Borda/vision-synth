"""On-disk dataset output: format writers and the torch-dependent iterable dataset.

:mod:`~synth_datasets.export.datasets` is the only torch-dependent module in this package; nothing here re-exports it,
so importing :mod:`synth_datasets.export` on its own stays torch-free.

"""

from __future__ import annotations
