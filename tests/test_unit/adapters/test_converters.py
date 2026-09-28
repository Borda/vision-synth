"""Tests for NumpyToTorchConverter and TorchToNumpyConverter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fused_transforms import BackendConverter
from fused_transforms.dispatch.converters import NumpyToTorchConverter, TorchToNumpyConverter


class TestNumpyToTorchConverter:
    """Verify NumpyToTorchConverter layout and dtype conversion."""

    def test_hwc_to_1chw(self) -> None:
        """3-D HWC float32 input becomes (1, 3, H, W) torch.Tensor with float32 dtype."""
        ndarray_in = np.random.rand(16, 24, 3).astype(np.float32)
        converter = NumpyToTorchConverter()
        result = converter.convert(ndarray_in)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 16, 24)
        assert result.dtype == torch.float32

    def test_bhwc_to_bchw(self) -> None:
        """4-D BHWC input is permuted to BCHW layout."""
        ndarray_in = np.random.rand(4, 16, 24, 3).astype(np.float32)
        converter = NumpyToTorchConverter()
        result = converter.convert(ndarray_in)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3, 16, 24)

    def test_uint8_normalised(self) -> None:
        """Uint8 input is normalised to float32 in [0, 1] (255 -> 1.0)

        Albumentations and many user pipelines operate on uint8 HWC arrays; the converter must rescale to the float32
        [0, 1] range expected by torch transforms.

        """
        ndarray_in = np.full((8, 8, 3), 255, dtype=np.uint8)
        converter = NumpyToTorchConverter()
        result = converter.convert(ndarray_in)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.ones(1, 3, 8, 8))

    def test_hwc_with_non_rgb_channel_count_round_trips(self) -> None:
        """Channel-last input with C != 3 (e.g. C=5) is converted correctly.

        Validates that the converter treats the trailing axis as channels regardless of size, supporting multi-spectral
        or mask-channel inputs rather than hard-coding RGB.

        """
        ndarray_in = np.random.rand(8, 8, 5).astype(np.float32)
        converter = NumpyToTorchConverter()
        result = converter.convert(ndarray_in)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 5, 8, 8)

    def test_zero_channel_axis_raises(self) -> None:
        """An empty channel axis (C=0) raises ValueError with an actionable message."""
        ndarray_in = np.empty((8, 8, 0), dtype=np.float32)
        converter = NumpyToTorchConverter()
        with pytest.raises(ValueError, match="non-empty channel axis"):
            converter.convert(ndarray_in)

    def test_isinstance_backend_converter(self) -> None:
        """NumpyToTorchConverter satisfies the BackendConverter protocol."""
        assert isinstance(NumpyToTorchConverter(), BackendConverter)

    def test_target_backend(self) -> None:
        """NumpyToTorchConverter declares 'torch' as its target backend."""
        assert NumpyToTorchConverter().target_backend == "torch"


class TestTorchToNumpyConverter:
    """Verify TorchToNumpyConverter layout conversion."""

    def test_1chw_to_hwc(self) -> None:
        """Single-batch BCHW tensor (B=1) is squeezed and permuted to HWC ndarray."""
        tensor = torch.rand(1, 3, 16, 24)
        converter = TorchToNumpyConverter()
        result = converter.convert(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 24, 3)
        assert result.dtype == np.float32

    def test_bchw_to_bhwc(self) -> None:
        """Multi-batch BCHW tensor is permuted to BHWC ndarray without squeezing."""
        tensor = torch.rand(4, 3, 16, 24)
        converter = TorchToNumpyConverter()
        result = converter.convert(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 16, 24, 3)

    def test_multichannel_round_trip(self) -> None:
        """Round trip torch -> numpy -> torch is lossless for non-RGB channel counts.

        Guards against permutation or dtype drift when feeding multi-channel masks or feature maps through a fused
        pipeline that converts between backends.

        """
        tensor = torch.rand(1, 5, 8, 8)
        numpy_result = TorchToNumpyConverter().convert(tensor)
        torch_result = NumpyToTorchConverter().convert(numpy_result)
        torch.testing.assert_close(torch_result, tensor)

    def test_isinstance_backend_converter(self) -> None:
        """TorchToNumpyConverter satisfies the BackendConverter protocol."""
        assert isinstance(TorchToNumpyConverter(), BackendConverter)

    def test_target_backend(self) -> None:
        """TorchToNumpyConverter declares 'numpy' as its target backend."""
        assert TorchToNumpyConverter().target_backend == "numpy"

    def test_3d_chw_input_raises_valueerror(self) -> None:
        """3-D tensor (num_channels, height, width) raises ValueError.

        Converter expects 4-D (batch_size, num_channels, height, width).

        """
        converter = TorchToNumpyConverter()
        tensor_3d = torch.rand(3, 16, 16)
        with pytest.raises(ValueError, match="Expected 4-D tensor"):
            converter.convert(tensor_3d)


class TestNumpyToTorchConverterEdgeCases:
    """Edge-case dimensionality checks for NumpyToTorchConverter."""

    def test_1d_input_raises_valueerror(self) -> None:
        """1-D array raises ValueError — only 2-D/3-D/4-D are accepted."""
        ndarray_in = np.zeros(8, dtype=np.float32)
        converter = NumpyToTorchConverter()
        with pytest.raises(ValueError, match="Expected 2-D/3-D/4-D"):
            converter.convert(ndarray_in)

    def test_5d_input_raises_valueerror(self) -> None:
        """5-D array raises ValueError — only 2-D/3-D/4-D are accepted."""
        ndarray_in = np.zeros((2, 2, 8, 8, 3), dtype=np.float32)
        converter = NumpyToTorchConverter()
        with pytest.raises(ValueError, match="Expected 2-D/3-D/4-D"):
            converter.convert(ndarray_in)


class TestNumpyToTorchConverterDtypesAndGrayscale:
    """2-D grayscale layout path and non-{uint8, float32} dtype casts."""

    def test_hw_grayscale_to_1_1_h_w(self) -> None:
        """2-D (H, W) input is expanded to a (1, 1, H, W) tensor."""
        ndarray_in = np.random.rand(8, 10).astype(np.float32)
        tensor_out = NumpyToTorchConverter().convert(ndarray_in)
        assert tensor_out.shape == (1, 1, 8, 10)
        assert torch.allclose(tensor_out[0, 0], torch.from_numpy(ndarray_in))

    def test_hw_grayscale_uint8_normalised(self) -> None:
        """2-D uint8 input is expanded to (1, 1, H, W) and normalised to float32 [0, 1]."""
        ndarray_in = np.full((4, 4), 255, dtype=np.uint8)
        tensor_out = NumpyToTorchConverter().convert(ndarray_in)
        assert tensor_out.shape == (1, 1, 4, 4)
        assert tensor_out.dtype == torch.float32
        assert torch.allclose(tensor_out, torch.ones(1, 1, 4, 4))

    def test_float64_cast_to_float32(self) -> None:
        """Float64 input is cast to float32 without rescaling."""
        ndarray_in = np.random.rand(6, 6, 3)  # float64 in [0, 1]
        tensor_out = NumpyToTorchConverter().convert(ndarray_in)
        assert tensor_out.dtype == torch.float32
        assert tensor_out.shape == (1, 3, 6, 6)
        expected = torch.from_numpy(ndarray_in.astype(np.float32)).permute(2, 0, 1)
        assert torch.allclose(tensor_out[0], expected)

    def test_float16_cast_to_float32(self) -> None:
        """Float16 input is cast to float32 without rescaling."""
        ndarray_in = np.random.rand(4, 4, 3).astype(np.float16)
        tensor_out = NumpyToTorchConverter().convert(ndarray_in)
        assert tensor_out.dtype == torch.float32
        assert torch.allclose(tensor_out[0], torch.from_numpy(ndarray_in).permute(2, 0, 1).to(torch.float32))

    def test_uint16_normalised(self) -> None:
        """Uint16 input is normalised to float32 [0, 1] by dividing by 65535."""
        ndarray_in = np.full((4, 4, 3), 65535, dtype=np.uint16)
        tensor_out = NumpyToTorchConverter().convert(ndarray_in)
        assert tensor_out.dtype == torch.float32
        assert torch.allclose(tensor_out, torch.ones(1, 3, 4, 4))

    def test_int32_cast_without_rescale(self) -> None:
        """Int32 input is cast to float32 with values preserved (no 255 normalisation)."""
        ndarray_in = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
        tensor_out = NumpyToTorchConverter().convert(ndarray_in)
        assert tensor_out.dtype == torch.float32
        assert float(tensor_out.max()) == 11.0
