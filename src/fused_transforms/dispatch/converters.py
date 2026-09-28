"""Concrete BackendConverter implementations for cross-backend output.

Provides converters between PyTorch tensors and NumPy arrays, preserving the ``(batch_size, channels, height, width)``
pipeline invariant on the torch side and converting to/from ``(batch_size, height, width, channels)`` HWC layout on the
NumPy side.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NumpyToTorchConverter:
    """Convert NumPy HWC/BHWC arrays to ``(batch_size, channels, height, width)`` torch tensors.

    All pipeline outputs are ``(batch_size, channels, height, width)`` ``torch.Tensor`` regardless of backend. ``uint8``
    inputs are normalised to ``float32`` in ``[0, 1]`` (÷255); ``uint16`` likewise (÷65535); ``float32`` inputs are
    passed through unchanged. Any other dtype (``float64``, ``float16``, signed integers) is cast to ``float32`` WITHOUT
    rescaling — values keep their original range.

    """

    @property
    def target_backend(self) -> str:
        """Target backend identifier."""
        return "torch"

    def convert(self, array: NDArray[Any]) -> torch.Tensor:
        """Convert a NumPy array (HW/HWC/BHWC) to a float32 torch tensor (BCHW).

        Args:
            array: NumPy array in channel-last layout:
                ``(height, width)``, ``(height, width, channels)``, or ``(batch_size, height, width, channels)``.

        Returns:
            ``torch.float32`` tensor of shape ``(1, 1, height, width)`` for 2-D input,
            ``(1, channels, height, width)`` for 3-D input, or ``(batch_size, channels, height, width)`` for 4-D input.

        Raises:
            ValueError: If ``array`` is not 2-D/3-D/4-D, or if a 3-D input is
                ambiguous and does not look like ``(H, W, C)``.

        """
        import numpy as np_mod

        if array.dtype == np_mod.uint16:
            # Normalise in numpy space: torch < 2.3 has no uint16 dtype and torch.from_numpy
            # rejects np.uint16 arrays. Same semantics as uint8: unsigned ints -> [0, 1].
            array = array.astype(np_mod.float32) / 65535.0

        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            # (H, W) -> (1, H, W, 1)
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        elif tensor.ndim == 3:
            # Treat 3-D arrays as single-image HWC by default so arbitrary
            # channel counts round-trip cleanly. Batched grayscale must be
            # passed explicitly as (batch_size, height, width, 1) to avoid ambiguity.
            if tensor.shape[-1] == 0:
                msg = (
                    f"Ambiguous 3-D array shape {tuple(tensor.shape)}; expected channel-last (height, width, channels) "
                    "with a non-empty channel axis. For batched grayscale, pass (batch_size, height, width, 1)."
                )
                raise ValueError(msg)
            # (H, W, C) -> (1, H, W, C)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            msg = f"Expected 2-D/3-D/4-D numpy array, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)

        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        elif tensor.dtype != torch.float32:
            # Keep the pipeline invariant: image tensors are float32. NOTE: other
            # integer dtypes (int16/int32/...) are cast WITHOUT rescaling — values
            # keep their original range; only uint8/uint16 are normalised.
            tensor = tensor.to(torch.float32)

        # (batch_size, height, width, channels) -> (batch_size, channels, height, width)
        return tensor.permute(0, 3, 1, 2)


class TorchToNumpyConverter:
    """Convert ``(batch_size, channels, height, width)`` torch tensors to NumPy HWC/BHWC arrays.

    Single-image batches ``(1, channels, height, width)`` are squeezed to ``(height, width, channels)`` for convenience.
    Multi-image batches produce ``(batch_size, height, width, channels)``.

    """

    @property
    def target_backend(self) -> str:
        """Target backend identifier."""
        return "numpy"

    def convert(self, tensor: torch.Tensor) -> NDArray[Any]:
        """Convert a torch tensor (BCHW) to a NumPy array (HWC or BHWC).

        Args:
            tensor: ``torch.Tensor`` of shape ``(batch_size, channels, height, width)``.

        Returns:
            NumPy ``ndarray`` of shape ``(height, width, channels)`` when ``batch_size == 1``, or
            ``(batch_size, height, width, channels)`` otherwise. Dtype is preserved (typically ``float32``).

        Raises:
            ValueError: If ``tensor`` is not a 4-D ``(batch_size, channels, height, width)`` tensor.

        """
        import numpy as np_mod

        if tensor.ndim != 4:
            msg = f"Expected 4-D tensor (batch_size, channels, height, width), got shape {tuple(tensor.shape)}"
            raise ValueError(msg)

        ndarray_out = tensor.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
        if ndarray_out.shape[0] == 1:
            # (1, height, width, channels) -> (height, width, channels)
            ndarray_out = np_mod.squeeze(ndarray_out, axis=0)
        return ndarray_out
