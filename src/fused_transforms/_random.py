"""Random-draw helpers that honour a caller-supplied :class:`torch.Generator`.

Every pipeline-owned draw (per-transform probability gates, direct parameter
sampling) routes through :func:`rand` or :func:`uniform` so a caller can own the
random stream instead of sharing the global one.

Two invariants hold across this module:

- ``generator=None`` draws from the global torch stream with the plain torch
  call — same function, same device, no extra hop — so seeding with
  ``torch.manual_seed`` controls it bit for bit on every device.
- A generator whose device differs from the target device draws on the
  generator's device and copies the result across, so one CPU generator can seed
  a CUDA or MPS pipeline. ``torch`` itself rejects that combination.

"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["GeneratorPicklingMixin", "rand", "reject_backend_randomness", "uniform"]


def _draw_device(generator: torch.Generator, device: torch.device) -> torch.device:
    """Return the device to draw on for ``generator`` targeting ``device``.

    Args:
        generator: The caller-supplied generator.
        device: The device the drawn tensor is needed on.

    Returns:
        ``device`` when the generator can drive it directly, else the generator's own device.

    """
    return device if generator.device.type == device.type else generator.device


def rand(
    shape: tuple[int, ...] | int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw uniform ``[0, 1)`` samples, optionally from a caller-owned generator.

    Args:
        shape: Output shape; ``()`` for a scalar draw.
        device: Device the returned tensor must live on.
        generator: Caller-owned generator, or ``None`` for the global stream.

    Returns:
        A float tensor of ``shape`` on ``device``.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms._random import rand
        >>> gen = torch.Generator().manual_seed(0)
        >>> first = rand(3, device=torch.device("cpu"), generator=gen)
        >>> gen.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> bool(torch.equal(first, rand(3, device=torch.device("cpu"), generator=gen)))
        True

        ```

    """
    if generator is None:
        return torch.rand(shape, device=device)
    draw_device = _draw_device(generator, device)
    drawn = torch.rand(shape, device=draw_device, generator=generator)
    return drawn if draw_device == device else drawn.to(device)


def uniform(
    shape: tuple[int, ...] | int,
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw uniform ``[low, high)`` samples, optionally from a caller-owned generator.

    Args:
        shape: Output shape; ``()`` for a scalar draw.
        low: Inclusive lower bound.
        high: Exclusive upper bound.
        device: Device the returned tensor must live on.
        dtype: Output dtype, or ``None`` for the torch default.
        generator: Caller-owned generator, or ``None`` for the global stream.

    Returns:
        A tensor of ``shape`` on ``device`` filled with uniform samples.

    Examples:
        ```pycon
        >>> import torch
        >>> from fused_transforms._random import uniform
        >>> gen = torch.Generator().manual_seed(0)
        >>> drawn = uniform(4, -1.0, 1.0, device=torch.device("cpu"), generator=gen)
        >>> bool((drawn >= -1.0).all() and (drawn < 1.0).all())
        True

        ```

    """
    if generator is None:
        return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)
    draw_device = _draw_device(generator, device)
    drawn = torch.empty(shape, device=draw_device, dtype=dtype).uniform_(low, high, generator=generator)
    return drawn if draw_device == device else drawn.to(device)


def reject_backend_randomness(generator: torch.Generator | None, source: str) -> None:
    """Raise when a caller-owned generator would reach a backend-owned draw.

    Backend transforms sample through their own libraries (Kornia's samplers,
    Albumentations' ``py_random`` and ``np.random``, TorchVision's ``make_params``),
    none of which accept a ``torch.Generator``. Falling back to the global stream
    there would look reproducible without being reproducible, so the pipeline
    refuses the combination instead.

    Args:
        generator: The caller-owned generator, or ``None``.
        source: Human-readable description of the draw that cannot be seeded.

    Raises:
        ValueError: If ``generator`` is not ``None``.

    """
    if generator is None:
        return
    msg = (
        f"generator= cannot seed {source}: that draw belongs to the backend library and does not accept a "
        "torch.Generator. Use FusedCompose.from_params(...) (or backend='native') for a fully caller-seeded "
        "pipeline, or drop generator= and seed the backend's own global stream."
    )
    raise ValueError(msg)


class GeneratorPicklingMixin:
    """Pickle a caller-owned ``generator`` attribute by value instead of by reference.

    ``torch._C.Generator`` is not picklable on the oldest supported torch (2.2 raises
    ``TypeError: cannot pickle 'torch._C.Generator' object``), so an object holding one
    cannot cross a ``DataLoader`` worker boundary — the very place a seeded pipeline has
    to survive. The mixin swaps the live generator for a ``(device, state)`` snapshot on
    the way out and rebuilds an equivalent generator on the way in, so the restored
    stream resumes where the pickled one stood on every supported version.

    Object *identity* is not preserved: a pipeline and its segments share one generator
    before the round trip and each restore their own copy. Rebinding the shared instance
    is the owner's job — :class:`~fused_transforms.core.pipeline.FusedCompose` does it in
    its ``__setstate__`` — because independent copies would advance separate streams
    while looking correct.

    Examples:
        ```pycon
        >>> import pickle
        >>> import torch
        >>> from fused_transforms import FusedCompose
        >>> pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), generator=torch.Generator().manual_seed(0))
        >>> restored = pickle.loads(pickle.dumps(pipe))  # noqa: S301
        >>> restored.generator is None
        False
        >>> bool(torch.equal(restored(torch.zeros(1, 3, 8, 8)), pipe(torch.zeros(1, 3, 8, 8))))
        True

        ```

    """

    generator: torch.Generator | None

    def __getstate__(self) -> dict[str, Any]:
        """Return the instance state with the generator replaced by a snapshot.

        Returns:
            The pickled state; ``generator`` is ``None`` and, when one was set, the
            extra key ``_generator_snapshot`` carries its device and byte state.

        """
        getter = getattr(super(), "__getstate__", None)
        state = dict(getter()) if getter is not None else dict(self.__dict__)
        generator = state.get("generator")
        if generator is not None:
            state["generator"] = None
            state["_generator_snapshot"] = (str(generator.device), generator.get_state())
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the instance, rebuilding the generator from its snapshot.

        Args:
            state: The pickled instance state produced by :meth:`__getstate__`.

        """
        snapshot = state.pop("_generator_snapshot", None)
        setter = getattr(super(), "__setstate__", None)
        if setter is not None:
            setter(state)
        else:
            self.__dict__.update(state)
        if snapshot is not None:
            device, generator_state = snapshot
            # A CUDA-seeded pipeline unpickled on a CPU-only box raises here rather than
            # silently drawing from a different device's stream.
            generator = torch.Generator(device=device)
            generator.set_state(generator_state)
            self.generator = generator
