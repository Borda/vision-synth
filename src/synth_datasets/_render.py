"""Helpers shared by the canvas fillers and the pointwise effects.

Both :mod:`~synth_datasets.content.backgrounds` and :mod:`~synth_datasets.content.degradations` hand ``uint8`` images
around and both are handed a side stream that is ``None`` whenever the object declared it draws nothing. The two rules
those facts imply — round and clip exactly once, and fail by name rather than by an attribute error on ``None`` — live
here so the two modules cannot drift apart on either.

Numpy only, and nothing public: neither name is exported from the package surface.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def to_uint8(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Round a float image to the writable, C-contiguous ``uint8`` array every renderer returns.

    Args:
        values: Float pixel values in any range, of any shape.

    Returns:
        The same values clipped to ``[0, 255]`` and rounded once. One rounding rather than several is
        what keeps two implementations of the same ramp agreeing on the last bit.

    """
    return np.ascontiguousarray(np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8))


def require_stream(rng: np.random.Generator | None, owner: str) -> np.random.Generator:
    """Return the side stream a drawing object was promised, or name whoever broke the contract.

    Args:
        rng: The stream handed to a ``render`` or ``apply`` call.
        owner: The class name, so the message names the object that needs a stream.

    Returns:
        ``rng`` unchanged.

    Raises:
        TypeError: If ``rng`` is ``None``. Only an object whose ``consumes_randomness`` is ``False``
            is handed ``None``, so reaching here means the declaration and the behaviour disagree —
            a caller error worth naming rather than one to surface as an attribute error.

    """
    if rng is None:
        raise TypeError(f"{owner} draws randomness but got no side stream; its consumes_randomness says it needs one")
    return rng
