"""Unit tests for data_keys routing in FusedCompose.

Tests cover the v0.3 data_keys contract:
- data_keys=None -> forward(img) returns single tensor (backward compat)
- data_keys=["input"] single key -> returns single tensor (unwrapped)
- data_keys=["input", "mask"] -> forward(img, mask) returns (img_out, mask_out) tuple
- Unknown key in data_keys emits UserWarning and passes value through
- Wrong number of args vs data_keys raises ValueError or TypeError

"""

from __future__ import annotations

import pytest
import torch

from fused_transforms.compose import Compose


class TestDataKeysNone:
    """data_keys=None preserves backward compatibility (v0.1/v0.2 behavior)."""

    def test_returns_single_tensor(self):
        """Forward(img) with data_keys=None returns a single Tensor, not a tuple."""
        pipe = Compose([], data_keys=None)
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"

    def test_shape_preserved(self):
        """Output shape matches input shape for data_keys=None."""
        pipe = Compose([], data_keys=None)
        x = torch.rand(1, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape

    def test_values_unchanged_empty_pipeline(self):
        """Empty pipeline with data_keys=None returns input unchanged."""
        pipe = Compose([], data_keys=None)
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        torch.testing.assert_close(out, x)


class TestDataKeysSingleKey:
    """data_keys=["input"] with single key returns unwrapped tensor."""

    def test_single_key_returns_tensor_not_tuple(self):
        """Single data_key returns an unwrapped Tensor, not a 1-tuple.

        Wrapping a single output in a 1-tuple would break the common `out = pipe(img)` call site that expects a tensor
        directly; the unwrap rule keeps single-key pipelines drop-in compatible with tensor-returning APIs.

        """
        pipe = Compose([], data_keys=["input"])
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
        assert not isinstance(out, tuple), "Single key should not return a tuple"

    def test_single_key_shape_preserved(self):
        """Single data_key preserves input shape."""
        pipe = Compose([], data_keys=["input"])
        x = torch.rand(1, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape


class TestDataKeysMultipleKeys:
    """data_keys=["input", "mask"] returns tuple in data_keys order."""

    def test_two_keys_returns_tuple(self):
        """Forward(img, mask) with two data_keys returns a 2-tuple."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 3, (2, 1, 8, 8)).float()
        out = pipe(img, mask)
        assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
        assert len(out) == 2, f"Expected 2-tuple, got length {len(out)}"

    def test_two_keys_shapes_preserved(self):
        """Both image and mask preserve their shapes."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 3, (2, 1, 8, 8)).float()
        out_img, out_mask = pipe(img, mask)
        assert out_img.shape == img.shape, f"Image shape mismatch: {out_img.shape} vs {img.shape}"
        assert out_mask.shape == mask.shape, f"Mask shape mismatch: {out_mask.shape} vs {mask.shape}"

    def test_values_unchanged_empty_pipeline(self):
        """Empty pipeline with data_keys returns inputs unchanged."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 3, (2, 1, 8, 8)).float()
        out_img, out_mask = pipe(img, mask)
        torch.testing.assert_close(out_img, img)
        torch.testing.assert_close(out_mask, mask)

    def test_tuple_order_matches_data_keys(self):
        """Return tuple order matches data_keys declaration order, not internal storage order.

        Uses sentinel values (all-ones image vs all-zeros mask) so a swapped return order would be detected via the mean
        comparison rather than silently passing on identical-shaped tensors.

        """
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.ones(1, 3, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        out_img, out_mask = pipe(img, mask)
        # img was all ones, mask was all zeros - order matters
        assert out_img.mean() > 0.5, "First element should be the image (ones)"
        assert out_mask.mean() < 0.5, "Second element should be the mask (zeros)"


class TestDataKeysUnknownKey:
    """Unknown keys in data_keys fail closed at construction."""

    def test_unknown_key_raises(self):
        """An unknown data_key raises ``ValueError`` naming the key, even for an empty pipeline.

        Passing an unsupported target through untransformed would silently misalign it with the augmented image, which
        is the failure the documented fail-closed target contract exists to prevent.

        """
        with pytest.raises(ValueError, match="custom_field"):
            Compose([], data_keys=["input", "custom_field"])


class TestDataKeysArgCountMismatch:
    """Wrong number of positional args vs data_keys raises an error."""

    @pytest.mark.parametrize(
        "data_keys, args",
        [
            pytest.param(["input", "mask"], (torch.zeros(1, 3, 4, 4),), id="too_few"),
            pytest.param(["input"], (torch.zeros(1, 3, 4, 4), torch.zeros(1, 1, 4, 4)), id="too_many"),
            pytest.param(["input", "mask"], (), id="zero"),
        ],
    )
    def test_arg_count_mismatch(self, data_keys, args):
        """Arg count != len(data_keys) raises ValueError or TypeError."""
        pipe = Compose([], data_keys=data_keys)
        with pytest.raises((ValueError, TypeError)):
            pipe(*args)


class TestDataKeysEmptyList:
    """data_keys=[] edge case: zero-length key list."""

    def test_empty_data_keys(self):
        """Constructing with data_keys=[] raises ValueError with guidance."""
        with pytest.raises(ValueError, match="data_keys cannot be an empty list"):
            Compose([], data_keys=[])


class TestDataKeysFirstKeyValidation:
    """data_keys[0] != 'input' raises ValueError at construction time."""

    @pytest.mark.parametrize(
        "data_keys",
        [
            pytest.param(["mask", "input"], id="mask_first"),
            pytest.param(["image"], id="image_only"),
        ],
    )
    def test_first_key_must_be_input(self, data_keys):
        """Any non-'input' string at data_keys[0] raises ValueError."""
        with pytest.raises(ValueError, match="data_keys\\[0\\] must be 'input'"):
            Compose([], data_keys=data_keys)


class TestDataKeysDuplicateAuxKeys:
    """Duplicate auxiliary keys (data_keys[1:]) raise ValueError on forward()."""

    def test_duplicate_auxiliary_keys_raises_on_forward(self):
        """data_keys=['input','mask','mask'] raises ValueError when forward() is called."""
        pipe = Compose([], data_keys=["input", "mask", "mask"])
        img = torch.rand(1, 3, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        with pytest.raises(ValueError, match="Duplicate entries detected in auxiliary data_keys"):
            pipe(img, mask, mask)


class TestDataKeysDuplicates:
    """Duplicate entries in data_keys."""

    def test_duplicate_data_keys(self):
        """data_keys=["input","input"] constructs and returns a 2-tuple.

        The enumerate-based return loop correctly preserves positional order:
        out[0] = args[0] (img1), out[1] = args[1] (img2). Duplicate keys are
        accepted at construction but the second occurrence maps to the second
        positional arg, not a copy of the first.

        """
        pipe = Compose([], data_keys=["input", "input"])
        img1 = torch.ones(1, 3, 4, 4)
        img2 = torch.zeros(1, 3, 4, 4)
        out = pipe(img1, img2)
        assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
        assert len(out) == 2, f"Expected 2-tuple, got {len(out)}"
        # enumerate fix: out[0]=img1, out[1]=img2 (positional, not key-matched)
        out_a, out_b = out
        torch.testing.assert_close(out_a, img1)
        torch.testing.assert_close(out_b, img2)
