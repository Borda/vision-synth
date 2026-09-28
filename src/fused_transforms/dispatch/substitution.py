"""Opt-in substitution of non-fusible passthrough ops with torch-native equivalents.

A non-fusible operation (blur, noise) on a GPU pipeline forces a device-to-host
round-trip -- the "GPU poison pill". When the user opts in via
``substitute_passthrough=True`` and an already-installed backend exposes a
torch-native equivalent, this module swaps the original passthrough transform for
that equivalent so the whole pipeline stays on-device.

Substitution is **behaviour-changing**: the torch-native op uses a different kernel
implementation, border handling, and random-parameter stream than the original, so
outputs and per-call RNG differ. It is therefore opt-in (default off) and every
substitution emits a :class:`UserWarning` documenting the change. Substitution
happens only when the target backend is importable; otherwise the original
passthrough is kept silently (the caller falls back to the normal passthrough path).

Registry entries are added only where the parameter mapping is faithful (validated
against the installed source of both the origin and target op). The registry ships
with one entry -- Albumentations ``GaussianBlur`` -> Kornia ``RandomGaussianBlur`` --
and is extensible.

Examples:
    ```pycon
    >>> from fused_transforms.dispatch.substitution import substitution_target_name
    >>> substitution_target_name("GaussianBlur")
    'RandomGaussianBlur'
    >>> substitution_target_name("MotionBlur") is None
    True

    ```

"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable


def _build_kornia_gaussian_blur(transform: object) -> object | None:
    """Build a Kornia ``RandomGaussianBlur`` equivalent to an Albumentations ``GaussianBlur``.

    Maps the Albumentations sigma range directly to Kornia's ``sigma`` range and
    derives a fixed kernel size from the range's upper bound using Albumentations'
    own ``blur_limit=0`` formula ``int(sigma * 3.5) * 2 + 1`` (odd, matches its
    largest kernel). The two implementations are not bit-identical -- Kornia uses a
    ``reflect`` border and its own separable kernel -- but the parameter mapping is
    faithful.

    Args:
        transform: An Albumentations ``GaussianBlur`` instance.

    Returns:
        A configured Kornia ``RandomGaussianBlur`` (probability 1.0, since the fused
        pipeline already applied the transform's own probability upstream), or
        ``None`` if Kornia is not importable.

    """
    try:
        import kornia.augmentation as kornia_aug
    except ImportError:
        return None

    sigma_limit = getattr(transform, "sigma_limit", (0.5, 3.0))
    sigma_lo, sigma_hi = _as_range(sigma_limit, default=(0.5, 3.0))
    # Albumentations' PIL-exact kernel-from-sigma rule at blur_limit=0.
    kernel_size = int(sigma_hi * 3.5) * 2 + 1
    kernel_size = max(kernel_size, 3)
    return cast(
        "object",
        kornia_aug.RandomGaussianBlur(
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma_lo, sigma_hi),
            p=1.0,
        ),
    )


def _as_range(value: object, default: tuple[float, float]) -> tuple[float, float]:
    """Coerce a scalar or 2-sequence into a ``(low, high)`` float tuple."""
    if isinstance(value, (int, float)):
        return (0.0, float(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return default


#: Maps an origin passthrough class name to ``(target_class_name, builder)``. The
#: builder returns a configured torch-native equivalent, or ``None`` when the target
#: backend is not importable. Extend only where the parameter mapping is faithful and
#: validated against the installed source of both ops.
_PASSTHROUGH_SUBSTITUTIONS: dict[str, tuple[str, Callable[[object], object | None]]] = {
    "GaussianBlur": ("RandomGaussianBlur", _build_kornia_gaussian_blur),
}


def substitution_target_name(origin_class_name: str) -> str | None:
    """Return the target op class name registered for a passthrough op, or ``None``.

    Args:
        origin_class_name: The passthrough transform's class name (e.g. ``"GaussianBlur"``).

    Returns:
        The registered torch-native target class name, or ``None`` if no substitution
        is registered for that op.

    Examples:
        ```pycon
        >>> substitution_target_name("GaussianBlur")
        'RandomGaussianBlur'
        >>> substitution_target_name("Unknown") is None
        True

        ```

    """
    entry = _PASSTHROUGH_SUBSTITUTIONS.get(origin_class_name)
    return entry[0] if entry is not None else None


def try_substitute_passthrough(transform: object) -> object | None:
    """Return a torch-native equivalent for a passthrough transform, or ``None``.

    Looks up the transform's class name in :data:`_PASSTHROUGH_SUBSTITUTIONS`. When a
    substitution is registered *and* its target backend is importable, builds the
    equivalent and emits a :class:`UserWarning` documenting the behaviour/numerics/RNG
    change. Returns ``None`` when no substitution is registered or the target backend
    is not importable -- the caller then keeps the original passthrough silently.

    Args:
        transform: The passthrough transform to substitute.

    Returns:
        A configured torch-native transform that can be applied on-device via the
        Kornia adapter's ``call_nonfused``, or ``None`` to keep the original.

    Examples:
        ```pycon
        >>> import albumentations as A  # doctest: +SKIP
        >>> sub = try_substitute_passthrough(A.GaussianBlur(p=1.0))  # doctest: +SKIP
        >>> type(sub).__name__  # doctest: +SKIP
        'RandomGaussianBlur'

        ```

    """
    entry = _PASSTHROUGH_SUBSTITUTIONS.get(type(transform).__name__)
    if entry is None:
        return None
    target_name, builder = entry
    substitute = builder(transform)
    if substitute is None:
        # Target backend not importable — fall back to normal passthrough silently.
        return None
    warnings.warn(
        f"substitute_passthrough=True replaced {type(transform).__name__!r} with the "
        f"torch-native {target_name!r}. This keeps the pipeline on-device but changes "
        "output numerics and the per-call random stream (different kernel, border "
        "handling, and RNG). Disable substitution to preserve the original op's exact "
        "behaviour.",
        UserWarning,
        stacklevel=2,
    )
    return substitute
