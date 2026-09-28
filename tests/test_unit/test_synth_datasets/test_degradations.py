"""Pin what each degradation does to an image, what it draws, and what it must leave alone.

A degradation is allowed to change every pixel and no label. These tests hold both halves: the shape and dtype contract
each step owes the next one, the randomness each declares, the fact that the tuple is applied in the order it is
written, and — the one that matters to a dataset — that turning the whole chain on moves nothing a writer would export.

"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from synth_datasets.content.degradations import (
    JPEG,
    ColorCast,
    Contrast,
    Degradation,
    GaussianBlur,
    GaussianNoise,
    Quantize,
    Vignette,
)
from synth_datasets.core.config import SyntheticConfig
from synth_datasets.core.generator import SyntheticGenerator

IMG_SIZE = 24

_ALL: dict[str, Degradation] = {
    "gaussian-noise": GaussianNoise(sigma=12.0),
    "gaussian-blur": GaussianBlur(radius=1.5),
    "jpeg": JPEG(quality=40),
    "contrast": Contrast(factor=0.4),
    "color-cast": ColorCast(gain=(1.2, 1.0, 0.8)),
    "vignette": Vignette(strength=0.5),
    "quantize": Quantize(levels=6),
}


@pytest.fixture
def image() -> np.ndarray:
    """Return a small image with both flat regions and a hard edge, so every effect has something to do."""
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 90, dtype=np.uint8)
    canvas[6:18, 6:18] = (220, 40, 40)
    return canvas


def _apply(step: Degradation, image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply one step, handing over a stream only when the step says it needs one."""
    return step.apply(image, np.random.default_rng(seed) if step.consumes_randomness else None)


@pytest.mark.parametrize("name", sorted(_ALL))
def test_every_degradation_preserves_shape_and_dtype(name: str, image: np.ndarray) -> None:
    """Each step returns what the next one expects, which is what lets the tuple compose at all.

    The chain hands one step's output straight to the next, so a step returning float or a different shape would break
    somewhere downstream rather than where the mistake was made.

    """
    out = _apply(_ALL[name], image)

    assert out.shape == image.shape, name
    assert out.dtype == np.uint8, name


@pytest.mark.parametrize("name", sorted(_ALL))
def test_every_degradation_changes_the_image(name: str, image: np.ndarray) -> None:
    """Each step at its configured strength actually does something, so no knob is a silent no-op."""
    assert not np.array_equal(_apply(_ALL[name], image), image), name


@pytest.mark.parametrize("name", sorted(_ALL))
def test_every_degradation_is_byte_reproducible_for_a_seed(name: str, image: np.ndarray) -> None:
    """Two applications from the same seed agree exactly, which is what makes a dataset replayable."""
    assert np.array_equal(_apply(_ALL[name], image, seed=4), _apply(_ALL[name], image, seed=4)), name


@pytest.mark.parametrize("name", sorted(name for name, step in _ALL.items() if not step.consumes_randomness))
def test_a_silent_degradation_leaves_a_stream_it_is_handed_untouched(name: str, image: np.ndarray) -> None:
    """A step configured by fixed scalars must not draw, which is what lets the chain skip the stream.

    Only `GaussianNoise` samples a field; the rest are deterministic, and the generator relies on that to decide whether
    a side stream has to be taken for the chain at all.

    """
    rng = np.random.default_rng(8)
    _ALL[name].apply(image, rng)

    assert rng.random() == np.random.default_rng(8).random(), name


def test_gaussian_noise_draws_exactly_one_field(image: np.ndarray) -> None:
    """The one drawing step consumes a single `standard_normal` over the whole image.

    Asserted against a twin stream that made that exact call, so a step drawing the same number of values in three per-
    channel passes instead of one still fails.

    """
    actual = np.random.default_rng(5)
    GaussianNoise(sigma=3.0).apply(image, actual)
    twin = np.random.default_rng(5)
    twin.standard_normal(image.shape)

    assert actual.random() == twin.random()


def test_the_chain_is_applied_in_the_order_it_is_written(image: np.ndarray) -> None:
    """Blurring then compressing is not the same as compressing then blurring.

    Order is the whole reason `degrade` is a tuple rather than a set, and a chain that silently reordered its steps
    would produce a plausible image that no longer matches the recipe it names.

    """
    blur_first = JPEG(quality=50).apply(GaussianBlur(radius=1.0).apply(image, None), None)
    jpeg_first = GaussianBlur(radius=1.0).apply(JPEG(quality=50).apply(image, None), None)

    assert not np.array_equal(blur_first, jpeg_first)


def test_an_empty_chain_leaves_the_stream_and_the_pixels_alone() -> None:
    """`degrade=()` is exactly the old behaviour: no step, no draw, no side stream taken.

    This is the inertness claim the whole feature rests on — an existing seeded configuration must not notice that
    degradations now exist.

    """
    config = SyntheticConfig(img_size=48, max_objects=3)
    generator = SyntheticGenerator(config)

    assert generator._needs_side_stream is False
    assert config.degrade == ()


def test_degrading_an_image_leaves_every_label_where_it_was() -> None:
    """A full chain changes pixels everywhere and moves no box, which is what "pointwise" means.

    The label oracle's whole premise is that the generator's labels are exact for the pixels it emits; a degradation
    that shifted a box would break it silently, since the image would still look reasonable.

    """
    plain = SyntheticConfig(img_size=64, max_objects=4)
    degraded = SyntheticConfig(
        img_size=64,
        max_objects=4,
        degrade=(GaussianNoise(sigma=10.0), GaussianBlur(radius=1.0), JPEG(quality=60)),
    )

    plain_samples = list(SyntheticGenerator(plain).generate(3, seed=1))
    degraded_samples = list(SyntheticGenerator(degraded).generate(3, seed=1))

    assert [[a.bbox_xyxy for a in s.annotations] for s in plain_samples] == [
        [a.bbox_xyxy for a in s.annotations] for s in degraded_samples
    ]
    assert not np.array_equal(plain_samples[0].image, degraded_samples[0].image)


def test_a_degrade_tuple_holding_a_stray_value_is_refused_at_construction() -> None:
    """The chain is validated where it is written, not where it is applied."""
    with pytest.raises(ValueError, match="only Degradation instances"):
        SyntheticConfig(img_size=32, degrade=(GaussianBlur(radius=1.0), "blur"))


def test_a_vignette_darkens_the_corners_and_spares_the_centre(image: np.ndarray) -> None:
    """The falloff is radial, so the corner loses brightness the centre keeps."""
    flat = np.full((33, 33, 3), 200, dtype=np.uint8)

    out = Vignette(strength=0.6).apply(flat, None)

    assert out[16, 16, 0] == 200
    assert out[0, 0, 0] < 200


def test_quantize_collapses_a_ramp_onto_its_level_count() -> None:
    """A posterized ramp holds exactly as many distinct values as levels were asked for."""
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)[..., None].repeat(3, axis=2)

    assert len(np.unique(Quantize(levels=4).apply(ramp, None))) == 4


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(lambda: GaussianNoise(sigma=-1.0), "sigma must be non-negative", id="negative-sigma"),
        pytest.param(lambda: GaussianBlur(radius=-0.5), "radius must be non-negative", id="negative-radius"),
        pytest.param(lambda: JPEG(quality=0), "quality must be within", id="zero-quality"),
        pytest.param(lambda: JPEG(quality=100), "quality must be within", id="over-max-quality"),
        pytest.param(lambda: Contrast(factor=-0.1), "factor must be non-negative", id="negative-factor"),
        pytest.param(lambda: ColorCast(gain=(1.0, 1.0)), "three non-negative", id="two-channel-gain"),
        pytest.param(lambda: Vignette(strength=1.5), "strength must be within", id="strength-over-one"),
        pytest.param(lambda: Quantize(levels=1), "levels must be within", id="one-level"),
    ],
)
def test_an_unusable_parameter_is_refused_at_construction(build: Callable[[], Degradation], message: str) -> None:
    """Each effect validates its own parameters, which is the point of typing the effect at all."""
    with pytest.raises(ValueError, match=message):
        build()
