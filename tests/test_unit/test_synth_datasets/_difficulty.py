"""Training-free statistics that rank how hard a configuration is, and the three bands they rank.

The original plan for this work named a held-out mAP gate in a downstream repository. That gate is
not here: there is no ``scripts/`` directory, ``ultralytics`` is neither a dependency nor an extra,
and a repo-wide search finds the downstream project only in prose. Rather than reimplement a training
loop to measure a data generator, the ladder is measured with deterministic, numpy-only statistics
that run in the unit suite. What they rank is the pipeline's sensitivity to each nuisance, which is
what a regression gate needs; what they cannot claim is an architecture ranking on real photographs,
and the documentation page built from them says so.

Five statistics per configuration, each cheap and each pointing at a different way a scene gets hard:

* mean and 10th-percentile object area, for the small-object regime
* the fraction of objects below COCO's small-object area, rescaled to this canvas
* boundary contrast, which is what a first-layer edge filter actually sees
* background signal-to-noise against the object fill, which is what survives that filter
* clutter coverage, the share of the canvas unlabelled distractors paint

Every number comes from a fixed seed, so two runs of this module agree exactly and a moved number is
a finding rather than sampling noise.

"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

from synth_datasets.content.backgrounds import NoiseBackground, TextureBackground
from synth_datasets.content.degradations import JPEG, GaussianBlur
from synth_datasets.core.config import SyntheticConfig
from synth_datasets.core.generator import SyntheticGenerator
from synth_datasets.core.sample import Sample
from synth_datasets.families.letters import LetterShape

#: Canvas the bands are measured on. Large enough that the hard band's 0.03 size ratio is a ~8-pixel
#: glyph — the small-object regime the generator could not produce before this work — and small
#: enough that the whole sweep stays inside a unit-test budget.
IMG_SIZE = 256

#: Images per band. Enough that one unlucky placement cannot swing a mean, few enough to stay cheap.
SAMPLES_PER_BAND = 8

#: Area, in pixels, below which an object counts as small. COCO calls an object small below 32x32 on
#: its own roughly 640-wide images; that threshold rescaled to this canvas is
#: ``1024 * (256 / 640) ** 2``. Scaling it rather than picking a round number is what keeps the
#: fraction comparable to the term everyone already uses.
SMALL_AREA = 1024.0 * (IMG_SIZE / 640.0) ** 2


@dataclass(frozen=True)
class ProxyStats:
    """The five statistics one configuration scores, all in pixel or ratio units.

    Args:
        mean_area: Mean labelled-object area in pixels.
        p10_area: Tenth-percentile labelled-object area in pixels.
        small_fraction: Share of labelled objects below :data:`SMALL_AREA`.
        boundary_contrast: Mean absolute grey-level step across the object outline.
        background_snr: Mean object-to-background grey separation divided by the background's own
            spread — how far a fill stands out from what surrounds it, in units of the noise.
        clutter_coverage: Share of canvas pixels painted by unlabelled distractors.

    """

    mean_area: float
    p10_area: float
    small_fraction: float
    boundary_contrast: float
    background_snr: float
    clutter_coverage: float


def _polygon_area(polygon: list[float]) -> float:
    """Return the shoelace area of a flat ``[x1, y1, x2, y2, ...]`` outline."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _object_mask(sample: Sample, img_size: int) -> np.ndarray:
    """Return a boolean raster of every labelled object in one sample.

    Rasterized from the annotations rather than recovered from the pixels: a fill that happened to
    match the canvas would vanish from a colour-based recovery, and that is exactly the hard case
    these statistics exist to score.

    """
    stencil = Image.new("1", (img_size, img_size), 0)
    painter = ImageDraw.Draw(stencil)
    for annotation in sample.annotations:
        points = np.asarray(annotation.polygon, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] >= 3:
            painter.polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(stencil, dtype=bool)


def _ring(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel band just outside a mask, which is what an edge filter straddles."""
    grown = mask.copy()
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        grown |= np.roll(mask, shift, axis=axis)
    return grown & ~mask


def _grey(sample: Sample) -> np.ndarray:
    """Return the sample as a float grey image, which every contrast statistic below reads."""
    return sample.image.astype(np.float64).mean(axis=2)


def _clutter_coverage(sample: Sample, config: SyntheticConfig) -> float:
    """Return the share of canvas painted by unlabelled clutter.

    Measured by exact colour match against the resolved distractor palette, minus whatever a labelled object covers.
    Exact only while the pixels are undegraded, which is why the caller measures it on an undegraded twin: a blurred or
    compressed fill no longer matches the palette and would silently under-count.

    """
    if not config.distractors:
        return 0.0
    painted = np.zeros(sample.image.shape[:2], dtype=bool)
    for fill in config.resolved_distractor_colors:
        painted |= np.all(sample.image == np.asarray(fill.rgb, dtype=np.uint8), axis=2)
    return float((painted & ~_object_mask(sample, config.img_size)).mean())


def measure(config: SyntheticConfig, seed: int = 0, count: int = SAMPLES_PER_BAND) -> ProxyStats:
    """Return the five proxy statistics for one configuration over a fixed seeded stream.

    Args:
        config: The configuration to score.
        seed: Stream seed, so the numbers are reproducible rather than indicative.
        count: Images to average over.

    Returns:
        The statistics. ``clutter_coverage`` is read from an undegraded twin of ``config`` at the same
        seed, which carries identical clutter: degradations are applied after every shape is drawn and
        draw from the side stream last, so removing them moves no placement.

    """
    areas: list[float] = []
    contrasts: list[float] = []
    ratios: list[float] = []
    for sample in SyntheticGenerator(config).generate(count, seed=seed):
        areas.extend(_polygon_area(annotation.polygon) for annotation in sample.annotations)
        mask = _object_mask(sample, config.img_size)
        grey = _grey(sample)
        ring = _ring(mask)
        if mask.any() and ring.any():
            inside, outside = grey[mask], grey[ring]
            contrasts.append(float(abs(inside.mean() - outside.mean())))
            spread = float(grey[~mask].std()) if (~mask).any() else 0.0
            ratios.append(float(abs(inside.mean() - grey[~mask].mean()) / max(spread, 1.0)))
    undegraded = replace(config, degrade=()) if config.degrade else config
    coverages = [
        _clutter_coverage(sample, undegraded) for sample in SyntheticGenerator(undegraded).generate(count, seed=seed)
    ]
    area = np.asarray(areas, dtype=np.float64)
    return ProxyStats(
        mean_area=float(area.mean()),
        p10_area=float(np.percentile(area, 10)),
        small_fraction=float((area < SMALL_AREA).mean()),
        boundary_contrast=float(np.mean(contrasts)),
        background_snr=float(np.mean(ratios)),
        clutter_coverage=float(np.mean(coverages)),
    )


#: The three documented bands, as whole configurations rather than as single knobs. Difficulty is a
#: property of the combination — a large solid primitive on a noisy canvas is still separable on
#: colour alone, while a 8-pixel glyph beside clutter drawn by the same process is not — so a band is
#: named by what it sets, never by one number.
BANDS: dict[str, SyntheticConfig] = {
    "easy": SyntheticConfig(
        img_size=IMG_SIZE,
        background=NoiseBackground(sigma=12.0),
        min_size_ratio=0.10,
        max_size_ratio=0.30,
        min_objects=3,
        max_objects=6,
    ),
    "moderate": SyntheticConfig(
        img_size=IMG_SIZE,
        background=TextureBackground(frequency=8.0),
        min_size_ratio=0.08,
        max_size_ratio=0.25,
        min_objects=3,
        max_objects=6,
        distractors=3,
        degrade=(GaussianBlur(radius=0.5), JPEG(quality=75)),
    ),
    "hard": SyntheticConfig(
        img_size=IMG_SIZE,
        background=TextureBackground(frequency=8.0),
        shapes=tuple(LetterShape),
        min_size_ratio=0.03,
        max_size_ratio=0.25,
        min_objects=4,
        max_objects=8,
        distractors=6,
        degrade=(GaussianBlur(radius=0.5), JPEG(quality=75)),
    ),
}


def measure_bands() -> dict[str, ProxyStats]:
    """Return the statistics for every documented band, in band order."""
    return {name: measure(config) for name, config in BANDS.items()}


def render_table() -> str:
    """Return the band table as Markdown, which is what the documentation page is written from.

    Kept beside the measurement rather than typed into the page by hand, so a number in the docs can
    always be regenerated from the code that produced it::

        python -c "from tests.test_unit.test_data._difficulty import render_table; print(render_table())"

    """
    header = "| band | mean area px | p10 area px | small fraction | boundary contrast | background SNR | clutter |"
    rows = [header, "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, stats in measure_bands().items():
        rows.append(
            f"| {name} | {stats.mean_area:.0f} | {stats.p10_area:.0f} | {stats.small_fraction:.3f} "
            f"| {stats.boundary_contrast:.1f} | {stats.background_snr:.2f} | {stats.clutter_coverage:.3f} |"
        )
    return "\n".join(rows)
