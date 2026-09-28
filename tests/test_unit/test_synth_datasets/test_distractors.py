"""Pin what unlabelled clutter adds to an image and what it must never add to the labels.

Distractors exist to make a detector classify rather than blob-find: they are drawn by the same process as the labelled
objects, from shapes and colours no class owns, and nothing about them reaches an annotation. The tests below hold that
line from both sides — the pixels change, the label list does not — and check the two pool rules that decide what
clutter is allowed to look like.

"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from synth_datasets.content.degradations import Contrast
from synth_datasets.core.config import DISTRACTOR_PALETTE, Color, Fill, SyntheticConfig, Task
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.families import ALL_SHAPES
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.primitives import PrimitiveShape

#: How far apart two fills must stay after the harshest documented contrast knock-down. Chosen as a
#: distance a viewer and a first-layer filter both still resolve, well above the JPEG noise floor.
_MIN_SEPARATION = 20.0


def _annotations(config: SyntheticConfig, seed: int = 0, count: int = 3) -> list[list[tuple[float, ...]]]:
    """Return the box list of every sample in a short stream, which is what clutter must not change."""
    return [[a.bbox_xyxy for a in s.annotations] for s in SyntheticGenerator(config).generate(count, seed=seed)]


def test_clutter_never_reaches_the_annotations() -> None:
    """Six distractors and none produce the same annotation list at the same seed.

    Stated as "the same list" rather than "exactly `num_objects` annotations" on purpose: `sample` accepts fewer than
    the sampled count whenever the placement budget runs out and `min_objects` was already met, so a count assertion
    would fail on behaviour that predates this feature entirely.

    """
    plain = SyntheticConfig(img_size=64, max_objects=4)
    cluttered = SyntheticConfig(img_size=64, max_objects=4, distractors=6)

    assert _annotations(plain) == _annotations(cluttered)


def test_clutter_changes_the_pixels_it_does_not_change_the_labels() -> None:
    """The point of clutter is that it is visible; a no-op would be a silent knob."""
    plain = SyntheticConfig(img_size=64, max_objects=2)
    cluttered = SyntheticConfig(img_size=64, max_objects=2, distractors=6)

    plain_image = next(iter(SyntheticGenerator(plain).generate(1, seed=0))).image
    cluttered_image = next(iter(SyntheticGenerator(cluttered).generate(1, seed=0))).image

    assert not np.array_equal(plain_image, cluttered_image)


def test_distractors_work_on_a_stock_configuration() -> None:
    """`distractors=3` succeeds on the default config, which the naive complement rule would refuse.

    `colors` defaults to all three named `Color` members, so "the complement of colors" against that
    same vocabulary is empty. Taking it against a packaged palette instead is what makes a refusal
    mean something: it fires only when a user really has claimed the whole palette.

    """
    config = SyntheticConfig(img_size=64, distractors=3)

    assert config.resolved_distractor_colors == DISTRACTOR_PALETTE
    assert len(config.resolved_distractor_shapes) == len(ALL_SHAPES) - len(config.shapes)


def test_a_palette_entry_claimed_under_another_name_is_withheld() -> None:
    """A user fill sharing a palette RGB removes that entry from the clutter pool.

    `Fill` compares on its name as well as its triple, so subtracting whole fills would leave the palette entry
    available and paint unlabelled pixels in a colour a labelled class owns — invisible to the vocabulary and
    indistinguishable on screen, which under `ClassMode.COLOR` is fatal.

    """
    slate = DISTRACTOR_PALETTE[0].rgb
    config = SyntheticConfig(img_size=64, colors=(Fill(rgb=slate, name="my-grey"),), distractors=2)

    assert all(fill.rgb != slate for fill in config.resolved_distractor_colors)


def test_an_exhausted_colour_pool_is_refused_with_the_empty_set_named() -> None:
    """Claiming every palette triple and asking for clutter fails at construction, saying which pool went empty."""
    every_fill = tuple(Fill(rgb=fill.rgb) for fill in DISTRACTOR_PALETTE)

    with pytest.raises(ValueError, match="distractor_colors resolved to an empty pool"):
        SyntheticConfig(img_size=64, colors=every_fill, distractors=1)


def test_an_exhausted_shape_pool_is_refused_with_the_empty_set_named() -> None:
    """A run drawing every shape there is has no complement left to build clutter from."""
    with pytest.raises(ValueError, match="distractor_shapes resolved to an empty pool"):
        SyntheticConfig(img_size=64, shapes=ALL_SHAPES, distractors=1)


def test_an_exhausted_pool_is_tolerated_while_no_clutter_is_asked_for() -> None:
    """The pools resolve on every config, but only a nonzero `distractors` makes an empty one an error."""
    config = SyntheticConfig(img_size=64, shapes=ALL_SHAPES)

    assert config.resolved_distractor_shapes == ()


def test_explicit_pools_override_the_complement() -> None:
    """A caller who names the pools gets exactly those, normalized like every other fill."""
    config = SyntheticConfig(
        img_size=64,
        distractors=2,
        distractor_shapes=(PrimitiveShape.CIRCLE,),
        distractor_colors=((10, 20, 30),),
    )

    assert config.resolved_distractor_shapes == (PrimitiveShape.CIRCLE,)
    assert config.resolved_distractor_colors == (Fill(rgb=(10, 20, 30)),)


def test_clutter_carries_no_landmarks_under_the_keypoints_task() -> None:
    """A distractor is never annotated, so it never computes a landmark table whatever the task says.

    The keypoints task is the one where a clutter item quietly acquiring landmarks would be structurally valid — a table
    of the right width, attached to an object no class names — and so the one worth checking directly.

    """
    config = SyntheticConfig(
        img_size=96, task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK, AnimalShape.CAMEL), distractors=4
    )

    samples = list(SyntheticGenerator(config).generate(2, seed=0))

    assert all(a.keypoints is not None for s in samples for a in s.annotations)
    assert _annotations(config) == _annotations(
        SyntheticConfig(img_size=96, task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK, AnimalShape.CAMEL)), count=3
    )


def test_clutter_is_byte_reproducible_for_a_seed() -> None:
    """Two runs of one seed place the same clutter, so a cluttered dataset replays like any other."""
    config = SyntheticConfig(img_size=64, max_objects=2, distractors=5)

    first = next(iter(SyntheticGenerator(config).generate(1, seed=7))).image
    second = next(iter(SyntheticGenerator(config).generate(1, seed=7))).image

    assert np.array_equal(first, second)


def test_more_clutter_covers_more_canvas() -> None:
    """Raising `distractors` raises the painted area, so the knob means something measurable."""
    coverage = []
    for count in (0, 4, 12):
        config = SyntheticConfig(img_size=96, min_objects=1, max_objects=1, distractors=count)
        image = next(iter(SyntheticGenerator(config).generate(1, seed=3))).image
        grey = np.asarray(config.background.color.rgb, dtype=np.int64)
        coverage.append(int(np.any(image.astype(np.int64) != grey, axis=2).sum()))

    assert coverage == sorted(coverage)
    assert coverage[0] < coverage[-1]


def test_the_palette_stays_separable_from_the_class_vocabulary_under_harsh_contrast() -> None:
    """Every clutter fill stays far from every named `Color` after the strongest documented contrast knock-down.

    `Contrast(0.3)` is the low end of the documented range and pulls every fill toward one grey, which is exactly when a
    clutter colour could collapse onto a class colour. The palette is only useful if it survives that, so the separation
    is measured rather than asserted by eye.

    Only palette-against-class distances are measured, not palette-against-palette ones. Two clutter fills resembling
    each other costs a dataset nothing, since neither is labelled and neither names a class; a clutter fill resembling a
    *class* fill is what would paint unlabelled pixels in a colour a detector has been asked to find.

    """
    palette = np.asarray([fill.rgb for fill in DISTRACTOR_PALETTE])
    classes = np.asarray([Fill.parse(color).rgb for color in Color])
    image = np.repeat(np.concatenate([palette, classes])[None, :, :], 8, axis=0).astype(np.uint8)

    flattened = Contrast(factor=0.3).apply(image, None)[0].astype(np.float64)
    clutter, claimed = flattened[: len(palette)], flattened[len(palette) :]
    separation = np.linalg.norm(clutter[:, None, :] - claimed[None, :, :], axis=2)

    assert float(separation.min()) >= _MIN_SEPARATION, separation.min()


def test_replacing_shapes_rederives_the_clutter_pool() -> None:
    """`dataclasses.replace` on `shapes` must not leave clutter wearing a silhouette a class now owns.

    Writing the resolved complement back over the field made the *default* survive a change to what
    it was derived from: the second construction saw a concrete tuple instead of `None` and kept it.
    Clutter was then drawn as unlabelled duplicates of a real class, with nothing raised anywhere —
    and `replace` on a config is an idiom already used inside this repository.

    """
    base = SyntheticConfig(img_size=32, distractors=2)

    moved = replace(base, shapes=(AnimalShape.DUCK,))

    assert set(moved.shapes) & set(moved.resolved_distractor_shapes) == set()


def test_replacing_colors_rederives_the_clutter_pool() -> None:
    """The same defect on the colour axis: a newly claimed RGB must leave the clutter palette."""
    slate = DISTRACTOR_PALETTE[0].rgb
    base = SyntheticConfig(img_size=32, distractors=2)

    moved = replace(base, colors=(Fill(rgb=slate, name="mine"),))

    assert all(fill.rgb != slate for fill in moved.resolved_distractor_colors)


def test_replacing_into_an_empty_pool_still_refuses() -> None:
    """Deriving on read keeps the refusal honest too, rather than only the pool."""
    base = SyntheticConfig(img_size=32, distractors=2)

    with pytest.raises(ValueError, match="distractor_shapes resolved to an empty pool"):
        replace(base, shapes=ALL_SHAPES)
