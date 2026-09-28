"""Regression tests for ``FusedCompose`` contracts at the module boundary.

* Unknown ``data_keys`` fail closed with the known keys and a did-you-mean hint.
* ``torch.nn.Module`` transforms are registered submodules (parameters, ``state_dict``, ``.to``, ``.eval``).
* ``inverse()`` works for a ``padding_mode="per_transform"`` pipeline.
* An unbatched ``(C, H, W)`` tensor is refused with an actionable message.
* ``repr()`` shows the fusion plan.

"""

from __future__ import annotations

import pickle
import warnings

import pytest
import torch
from torch import nn

from fused_transforms import Compose

T = pytest.importorskip("torchvision.transforms.v2")


class _Gain(nn.Module):
    """A learnable passthrough transform: multiplies the image by one parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((1,), 2.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.weight


def _pipeline_with_gain() -> tuple[Compose, _Gain]:
    gain = _Gain()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # unrecognised transform -> passthrough barrier
        pipe = Compose([T.RandomRotation((0, 0)), gain])
    return pipe, gain


class TestUnknownDataKeys:
    """Unknown ``data_keys`` raise instead of passing the target through untransformed."""

    @pytest.mark.parametrize(("key", "hint"), [("boxes", "bbox_xyxy"), ("masks", "mask"), ("keypoint", "keypoints")])
    def test_unknown_key_raises_with_hint(self, key: str, hint: str) -> None:
        """A likely typo names the key it probably meant."""
        with pytest.raises(ValueError, match=rf"(?s){key!r}.*did you mean {hint!r}"):
            Compose([T.RandomRotation(10)], data_keys=["input", key])

    def test_unknown_key_lists_known_keys(self) -> None:
        """A key with no close match still lists every supported key."""
        with pytest.raises(ValueError, match="bbox_xyxy") as excinfo:
            Compose([T.RandomRotation(10)], data_keys=["input", "custom_field"])
        assert "did you mean" not in str(excinfo.value)
        assert "keypoints" in str(excinfo.value)


class TestModuleTransformsRegistered:
    """``nn.Module`` transforms follow the pipeline's module lifecycle."""

    def test_parameters_and_state_dict_reach_custom_module(self) -> None:
        pipe, gain = _pipeline_with_gain()
        assert any(param is gain.weight for param in pipe.parameters())
        assert any(key.endswith("weight") for key in pipe.state_dict())

    def test_to_and_eval_reach_custom_module(self) -> None:
        pipe, gain = _pipeline_with_gain()
        pipe.eval()
        assert not gain.training
        pipe.train()
        assert gain.training
        pipe.to(torch.float64)
        assert gain.weight.dtype == torch.float64

    def test_dispatch_order_unchanged(self) -> None:
        """Registering the module does not change what runs or in which order."""
        pipe, _ = _pipeline_with_gain()
        image = torch.rand(2, 3, 8, 8)
        torch.testing.assert_close(pipe(image), image * 2.0)

    def test_pickle_round_trip_keeps_registration(self) -> None:
        pipe, _ = _pipeline_with_gain()
        restored = pickle.loads(pickle.dumps(pipe))  # noqa: S301 - own object round-trip
        restored_gain = restored.original_transforms[1]
        assert any(param is restored_gain.weight for param in restored.parameters())
        image = torch.rand(2, 3, 8, 8)
        torch.testing.assert_close(restored(image), pipe(image))


def test_inverse_supports_per_transform_padding() -> None:
    """``inverse()`` resolves the per-transform policy to a concrete ``grid_sample`` mode."""
    pipe = Compose([T.RandomRotation(30), T.RandomAffine(0, translate=(0.1, 0.1))], padding_mode="per_transform")
    image = torch.rand(2, 3, 16, 16)
    out, matrix = pipe(image, return_matrix=True)
    recovered = pipe.inverse(out, matrix=matrix)
    assert recovered.shape == image.shape
    assert torch.isfinite(recovered).all()


class TestUnbatchedInput:
    """A ``(C, H, W)`` tensor gets an actionable error instead of an unpacking failure."""

    def test_single_tensor_mode(self) -> None:
        with pytest.raises(ValueError, match=r"\(C, H, W\).*x\[None\]"):
            Compose([T.RandomRotation(10)])(torch.rand(3, 16, 16))

    def test_multi_target_mode(self) -> None:
        pipe = Compose([T.RandomRotation(10)], data_keys=["input", "mask"])
        with pytest.raises(ValueError, match=r"x\[None\]"):
            pipe(torch.rand(3, 16, 16), torch.zeros(1, 16, 16))


class TestRepr:
    """``repr()`` surfaces the fusion plan."""

    def test_repr_contains_plan(self) -> None:
        pipe = Compose([T.RandomRotation(10), T.RandomAffine(0, translate=(0.1, 0.1))])
        text = repr(pipe)
        assert pipe.fusion_plan in text
        assert f"n_warps_saved={pipe.n_warps_saved}" in text

    def test_repr_of_empty_pipeline(self) -> None:
        assert "plan='empty'" in repr(Compose([]))
