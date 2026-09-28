"""Pin what each background renders, what it consumes, and what it must never disturb.

Three separate contracts live here. A background must render a valid canvas; it must consume exactly the randomness its
type documents, no more and no less; and switching one on must leave every object placement where it was, because it
draws from a side stream the placement loop never sees. The last is the one that makes the whole feature safe to add to
an existing seeded configuration, so it is checked against real generated samples rather than against the renderer
alone.

"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from synth_datasets.content.backgrounds import (
    Background,
    GradientBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    SolidBackground,
    TextureBackground,
)
from synth_datasets.content.degradations import Contrast
from synth_datasets.core.config import Color, Fill, SyntheticConfig
from synth_datasets.core.generator import SyntheticGenerator

IMG_SIZE = 32

#: Each background paired with the exact sequence of generator calls its documentation claims it
#: makes. The check is not "how many draws" by inspection but "does a twin stream that made these
#: calls end up in the same place", which no amount of internal restructuring can fake.
_DRAW_SCRIPTS: dict[str, tuple[Background, Callable[[np.random.Generator], None]]] = {
    "gradient-sampled-angle": (GradientBackground(), lambda rng: rng.uniform(0.0, 2.0 * np.pi)),
    "noise": (NoiseBackground(), lambda rng: rng.standard_normal((IMG_SIZE, IMG_SIZE, 3))),
    "impulse": (
        ImpulseNoiseBackground(),
        lambda rng: (rng.random((IMG_SIZE, IMG_SIZE)), rng.random((IMG_SIZE, IMG_SIZE)), None)[-1],
    ),
    "texture-two-octaves": (
        TextureBackground(octaves=2, frequency=4.0),
        lambda rng: (rng.uniform(-1.0, 1.0, size=(5, 5)), rng.uniform(-1.0, 1.0, size=(9, 9)), None)[-1],
    ),
}

#: The backgrounds that draw nothing at all, so the generator never even takes a side stream for them.
_SILENT = {
    "solid": SolidBackground(),
    "gradient-fixed-angle": GradientBackground(direction=0.4),
    "gradient-radial": GradientBackground(radial=True),
}

_ALL = {
    **_SILENT,
    "gradient-sampled-angle": GradientBackground(),
    "noise": NoiseBackground(sigma=12.0),
    "impulse": ImpulseNoiseBackground(amount=0.1),
    "texture": TextureBackground(octaves=2, frequency=4.0),
    "texture-quantized": TextureBackground(octaves=2, frequency=4.0, quantize=4),
}


def _render(background: Background, seed: int = 0) -> np.ndarray:
    """Render one canvas, handing over a stream only when the background says it needs one."""
    rng = np.random.default_rng(seed) if background.consumes_randomness else None
    return background.render(rng, IMG_SIZE)


@pytest.mark.parametrize("name", sorted(_ALL))
def test_every_background_renders_a_valid_canvas(name: str) -> None:
    """Each mode returns the writable, C-contiguous ``uint8`` canvas the base class promises.

    The generator hands the result straight to `Image.fromarray` and then draws polygons into it, so a read-only or
    oddly-strided array would fail at rasterization rather than here.

    """
    canvas = _render(_ALL[name])

    assert canvas.shape == (IMG_SIZE, IMG_SIZE, 3), name
    assert canvas.dtype == np.uint8, name
    assert canvas.flags["C_CONTIGUOUS"], name
    assert canvas.flags["WRITEABLE"], name


@pytest.mark.parametrize("name", sorted(_ALL))
def test_every_background_is_byte_reproducible_for_a_seed(name: str) -> None:
    """Two renders from the same seed agree exactly, which is what makes a dataset replayable."""
    first, second = _render(_ALL[name], seed=3), _render(_ALL[name], seed=3)

    assert np.array_equal(first, second), name


@pytest.mark.parametrize("name", sorted(_SILENT))
def test_a_silent_background_leaves_a_stream_it_is_handed_untouched(name: str) -> None:
    """A mode that declares it draws nothing must not draw even when a stream is available.

    The generator skips the side stream entirely for these, so a mode that quietly drew would be handed `None` and
    crash; this checks the declaration against the behaviour directly instead.

    """
    background = _SILENT[name]
    rng = np.random.default_rng(9)
    background.render(rng, IMG_SIZE)

    assert background.consumes_randomness is False, name
    assert rng.random() == np.random.default_rng(9).random(), name


@pytest.mark.parametrize("name", sorted(_DRAW_SCRIPTS))
def test_draw_counts_match_the_documented_script(name: str) -> None:
    """Each drawing mode consumes exactly the calls its documentation lists, in that order.

    Asserted by running a twin generator through the documented calls and comparing where both streams end up, so a
    renderer that drew the right *number* of values in a different *shape* — a per-pixel field drawn as three passes
    instead of one, say — still fails.

    """
    background, script = _DRAW_SCRIPTS[name]
    actual = np.random.default_rng(5)
    background.render(actual, IMG_SIZE)
    twin = np.random.default_rng(5)
    script(twin)

    assert actual.random() == twin.random(), name


@pytest.mark.parametrize("size", [1, 2, 33])
def test_a_solid_canvas_is_writable_at_every_size(size: int) -> None:
    """The writability contract holds at `img_size=1`, where a broadcast view is already contiguous.

    At one pixel every axis is length one, so `ascontiguousarray` hands the read-only broadcast view straight back and
    the copy that was meant to make it writable never happens. The renderer draws polygons into this array, so the
    contract is not decorative.

    """
    canvas = SolidBackground((10, 20, 30)).render(None, size)

    assert canvas.flags["WRITEABLE"], size
    assert canvas.shape == (size, size, 3)


def test_a_quantized_texture_shows_its_documented_level_count_at_one_octave() -> None:
    """At a single octave the field spans the whole range, so `quantize` means exactly what it says.

    Above one octave it does not: the octaves are summed and divided by their weight total, so the
    field spans less than `[-1, 1]` and some levels hold nothing — `quantize=8` at `octaves=3`
    measures 7. The docstrings state that; this pins the one case where the count is exact.

    """
    field = TextureBackground(octaves=1, frequency=8.0, quantize=8)._field(np.random.default_rng(0), 128)

    assert len(np.unique(field)) == 8


def test_a_bare_triple_still_names_a_flat_canvas() -> None:
    """The historical `background=(r, g, b)` spelling normalizes to a solid background.

    Every configuration written before backgrounds were types passes a triple, so this is the path that decides whether
    those configurations still render what they always rendered.

    """
    config = SyntheticConfig(img_size=IMG_SIZE, background=(10, 20, 30))

    assert isinstance(config.background, SolidBackground)
    assert config.background.color == Fill.parse((10, 20, 30))


def test_a_named_color_is_accepted_as_a_canvas_fill() -> None:
    """A `Color` member is a fill like any other, so it names a canvas as well as an object."""
    config = SyntheticConfig(img_size=IMG_SIZE, background=Color.BLUE)

    assert config.background.render(None, 2)[0, 0].tolist() == [0, 0, 255]


def test_an_unusable_fill_is_refused_at_construction() -> None:
    """A canvas fill is validated like an object fill, which it never used to be."""
    with pytest.raises(ValueError, match="Color member or an"):
        SyntheticConfig(img_size=IMG_SIZE, background=(10, 20))


def test_switching_the_background_leaves_every_placement_where_it_was() -> None:
    """A noisy canvas and a flat one at the same seed annotate identically, object for object.

    This is the guarantee the side stream exists to provide: the background draws from a child of the
    caller's generator, so the placement draws that follow are the same bits either way. Checked
    across a stream rather than on one image, because the failure it guards against — knob draws
    shifting the *next* image — cannot appear in a single sample.

    """
    flat = SyntheticConfig(img_size=64, max_objects=4)
    noisy = SyntheticConfig(img_size=64, max_objects=4, background=NoiseBackground(sigma=24.0))

    flat_samples = list(SyntheticGenerator(flat).generate(4, seed=1))
    noisy_samples = list(SyntheticGenerator(noisy).generate(4, seed=1))

    assert [[a.bbox_xyxy for a in s.annotations] for s in flat_samples] == [
        [a.bbox_xyxy for a in s.annotations] for s in noisy_samples
    ]


def test_two_images_in_a_stream_get_independent_background_randomness() -> None:
    """Successive samples take their own child stream, so a stream is not one canvas repeated."""
    config = SyntheticConfig(img_size=48, min_objects=1, max_objects=1, background=NoiseBackground(sigma=20.0))

    first, second = list(SyntheticGenerator(config).generate(2, seed=2))

    assert not np.array_equal(first.image, second.image)


def test_impulse_noise_reaches_both_endpoints() -> None:
    """Salt and pepper are replacements, so both extremes appear whatever the base colour is.

    Adding a fixed offset to the base would leave no pure black or white behind, and the endpoints are precisely what
    makes impulse noise survive a blur that erases a Gaussian field.

    """
    canvas = ImpulseNoiseBackground(base=(120, 130, 140), amount=0.4).render(np.random.default_rng(0), 64)

    assert (canvas == 255).any()
    assert (canvas == 0).any()


def test_a_linear_ramp_spans_both_stops() -> None:
    """A fixed-direction ramp reaches each stop at its own end of the canvas."""
    canvas = GradientBackground(stops=((0, 0, 0), (255, 255, 255)), direction=0.0).render(None, 16)

    assert canvas[0, 0, 0] == 0
    assert canvas[0, -1, 0] == 255


def test_a_radial_ramp_is_darkest_at_the_centre() -> None:
    """A radial ramp puts its first stop at the centre and its second at the corners."""
    canvas = GradientBackground(stops=((0, 0, 0), (255, 255, 255)), radial=True).render(None, 17)

    assert canvas[8, 8, 0] < canvas[0, 0, 0]


def _detail(canvas: np.ndarray) -> float:
    """Return mean absolute horizontal neighbour difference — a cheap spatial-frequency proxy."""
    grey = canvas.astype(np.float64).mean(axis=2)
    return float(np.abs(np.diff(grey, axis=1)).mean())


def test_raising_the_texture_frequency_raises_the_measured_detail() -> None:
    """Finer lattices produce more high-frequency structure, monotonically over a seed sweep.

    `frequency` is the knob a difficulty band is written against, so it has to mean something measurable rather than
    merely change the picture; a texture whose detail did not rise with it would make every band table below it
    meaningless.

    """
    measured = [
        float(
            np.mean([
                _detail(TextureBackground(octaves=1, frequency=freq).render(np.random.default_rng(seed), 128))
                for seed in range(4)
            ])
        )
        for freq in (2.0, 4.0, 8.0, 16.0)
    ]

    assert measured == sorted(measured), measured


def test_quantizing_a_texture_collapses_it_onto_few_levels() -> None:
    """A posterized texture draws from far fewer distinct values than a continuous one."""
    continuous = TextureBackground(octaves=1, frequency=4.0).render(np.random.default_rng(0), 64)
    banded = TextureBackground(octaves=1, frequency=4.0, quantize=4).render(np.random.default_rng(0), 64)

    assert len(np.unique(banded)) < len(np.unique(continuous))


@pytest.mark.parametrize(
    ("background", "message"),
    [
        pytest.param(lambda: NoiseBackground(sigma=-1.0), "sigma must be non-negative", id="negative-sigma"),
        pytest.param(lambda: ImpulseNoiseBackground(amount=1.5), "amount must be within", id="amount-out-of-range"),
        pytest.param(lambda: TextureBackground(octaves=0), "octaves must be at least 1", id="no-octaves"),
        pytest.param(lambda: TextureBackground(frequency=0.0), "frequency must be positive", id="zero-frequency"),
        pytest.param(lambda: TextureBackground(quantize=1), "quantize must be at least 2", id="one-level"),
        pytest.param(lambda: GradientBackground(stops=((0, 0, 0),)), "exactly two fills", id="one-stop"),
    ],
)
def test_an_unusable_parameter_is_refused_at_construction(background: Callable[[], Background], message: str) -> None:
    """Each mode validates its own parameters, which is the point of typing the mode at all."""
    with pytest.raises(ValueError, match=message):
        background()


def test_one_knob_does_not_move_another_knob() -> None:
    """Each non-placement consumer owns its own side stream, so knobs cannot shift each other.

    `ImpulseNoiseBackground(amount=0.0)` paints exactly what `SolidBackground` paints while consuming two draws rather
    than none, which isolates the coupling from every pixel difference. Under one shared side child the background's
    draws shifted what the distractors drew next — 1341 pixels differed on this configuration — so two byte-identical
    canvases carried different clutter, and a proxy statistic could not be attributed to the knob that was changed.

    """
    flat = SolidBackground((128, 128, 128))
    silent = ImpulseNoiseBackground(base=(128, 128, 128), amount=0.0)
    assert np.array_equal(flat.render(None, 32), silent.render(np.random.default_rng(0), 32))

    def scene(background: Background) -> np.ndarray:
        config = SyntheticConfig(img_size=96, min_objects=1, max_objects=1, distractors=5, background=background)
        return next(iter(SyntheticGenerator(config).generate(1, seed=0))).image

    assert np.array_equal(scene(flat), scene(silent))


def test_an_undegraded_sample_does_not_copy_the_canvas() -> None:
    """A run with no degradation chain hands back Pillow's own buffer, exactly as it always did.

    Copying unconditionally would cost a full canvas per sample — 1.2 MB at `img_size=640` — for every caller including
    the ones that asked for nothing, and would flip the image's writeable flag as a side effect no release note
    mentioned.

    """
    plain = next(iter(SyntheticGenerator(SyntheticConfig(img_size=64)).generate(1, seed=0)))
    degraded_config = SyntheticConfig(img_size=64, degrade=(Contrast(factor=0.8),))
    degraded = next(iter(SyntheticGenerator(degraded_config).generate(1, seed=0)))

    assert plain.image.flags["WRITEABLE"] is False
    assert degraded.image.flags["WRITEABLE"] is True
