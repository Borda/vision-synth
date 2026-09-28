"""Render one before/after picture per scene knob: backgrounds, degradations, clutter, bands.

``render_shape_reference.py`` pictures the *vocabulary* — one upright shape per file — and
``animate_synthetic_dataset.py`` pictures the *tasks* — the same stream under four overlays. Neither
pictures the knobs that decide how hard a sample is to read, which is what this script adds.

Each knob setting gets its own file, and each file is one pair: the bare canvas on the left, the same
canvas with its objects and their exported ``bbox_xyxy`` on the right. The left half is what makes a
canvas knob legible at all — a texture or a photographic crop stops being readable once shapes cover
it — and the boxes on the right say which of the ink is labelled, which is the whole question once
clutter is on. Polygon and keypoint overlays stay out: those belong to the task previews, and three
overlays at this size hide the pixels they sit on.

One knob per file rather than a contact sheet of them all, so a page can put each picture beside the
paragraph that explains it, and so a knob whose rendering changes produces a one-file diff instead of
a rewritten grid. Each panel is captioned with what it is; the *setting* stays out of the pixels and
is named by the page, so re-wording a caption does not mean re-rendering to rename a knob.

Every file is rendered from the same seed, so the three shapes sit in the same three places in all of
them. That is not decoration: every background and clutter knob draws from a side stream rather than
from the placement stream (see
:attr:`~synth_datasets.content.backgrounds.Background.consumes_randomness`), so switching one on
cannot move an object, and a reader flipping between two of these files sees exactly that.

Rendering uses Pillow and numpy (both base dependencies); the CLI below uses ``fire``, which ships
in the ``cli`` extra:

    pip install "vision-synth[cli]"

Write every picture (under ``docs/assets/datasets/scene/<group>/``, named for its setting):
    python examples/render_scene_gallery.py

Write one group:
    python examples/render_scene_gallery.py --groups backgrounds

"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synth_datasets import (
    JPEG,
    ColorCast,
    Contrast,
    GaussianBlur,
    GaussianNoise,
    GradientBackground,
    ImageBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    Quantize,
    SolidBackground,
    SyntheticConfig,
    SyntheticGenerator,
    TextureBackground,
    Vignette,
)
from synth_datasets.core.generator import _SIDE_STREAM_ROLES
from synth_datasets.families.geometry import PIXEL_CENTRE_OFFSET
from synth_datasets.families.letters import LetterShape

if TYPE_CHECKING:
    from collections.abc import Callable

    from synth_datasets.content.backgrounds import Background
    from synth_datasets.content.degradations import Degradation

#: Side length of one panel. A file holds two of them side by side, so the written picture is roughly
#: 520 pixels wide -- wide enough that the caption under each panel reads at the size a docs column
#: renders it, which 192 was not, and that an eight-pixel glyph in the hard band is still a glyph.
_PANEL = 256
#: White gutter between the two panels, so a dark canvas does not read as one wide image.
_DIVIDER = 6
_PAPER = (255, 255, 255)
_INK = (32, 32, 32)
#: Strip reserved under the panels for their captions, and the type size drawn into it. The size is
#: bounded by the longest caption: it has to stay inside one panel's width at this panel size.
_CAPTION_HEIGHT = 28
_CAPTION_SIZE = 16
#: One caption per panel, in panel order. The right one names the overlay when a picture draws
#: something other than the plain box -- see :data:`_OVERLAYS`.
_CAPTIONS = ("empty scene with background", "scene with annotated objects")
#: Matches ``animate_synthetic_dataset.py``'s ``_OVERLAY_RGB``, so a box means the same thing in a clip
#: and in a still. Yellow rather than the blue these previews used to draw: the object fills are
#: ``Color.RED``, ``Color.GREEN`` and ``Color.BLUE``, so a blue box lands on a blue object regularly,
#: and it is also the closest of the candidates to the canvases. Measured as CIEDE2000 against every
#: pixel of four samples from each documented configuration: blue sits 6.1 from the nearest scene
#: colour and within 25 of a quarter of the hard band's pixels, while yellow sits 15.4 away and within
#: 25 of 0.3%. Magenta, cyan and amber all score between the two.
_BBOX_RGB = (255, 255, 0)
_BBOX_WIDTH = 2
#: One seed for every picture, so the clutter group's "none" is the same scene as the degradation
#: group's "none" and a reader can compare across groups, not only within one.
_SEED = 3
#: The overlays each band is drawn under, as ``(slug, caption)``. One scene per band, annotated three
#: ways, rather than three scenes annotated one way: the bands differ in what they do to a *detector*,
#: and holding the objects still is what lets a reader see that the box, the oriented box and the
#: outline are three readings of the same ink. ``keypoints`` is absent because only the hard band's
#: vocabulary carries landmarks, and a tab present for one band only invites the wrong comparison.
_OVERLAYS: tuple[tuple[str, str], ...] = (
    ("detection", "scene with detection boxes"),
    ("obb", "scene with oriented boxes"),
    ("segmentation", "scene with segmentation outlines"),
)

#: The backgrounds group, in the order the prose introduces them, as ``(slug, background)``.
#: ``ImageBackground`` is absent: it reads files the package does not ship, so it is built separately
#: on a directory synthesized for the run.
_BACKGROUNDS: tuple[tuple[str, Background], ...] = (
    ("solid", SolidBackground()),
    ("gradient", GradientBackground()),
    ("gradient-radial", GradientBackground(radial=True)),
    ("noise", NoiseBackground(sigma=24.0)),
    ("impulse", ImpulseNoiseBackground()),
    ("texture", TextureBackground()),
)

#: The degradations group. Each runs alone, at a strength chosen to be visible at this panel size
#: rather than at its default, so a picture shows what the knob does rather than what a conservative
#: default looks like. The page carries the exact call that produced each one.
_DEGRADATIONS: tuple[tuple[str, Degradation | None], ...] = (
    ("none", None),
    ("gaussian-noise", GaussianNoise(sigma=16.0)),
    ("gaussian-blur", GaussianBlur(radius=1.5)),
    ("jpeg", JPEG(quality=15)),
    ("contrast", Contrast(factor=0.45)),
    ("color-cast", ColorCast(gain=(1.3, 1.0, 0.7))),
    ("vignette", Vignette(strength=0.7)),
    ("quantize", Quantize(levels=4)),
)

#: The clutter group: the two knobs that add unlabelled ink, alone and together. Both draw shapes from
#: the same process as the targets, which is what these pictures are for -- an unlabelled square is
#: not distinguishable from a labelled one by any "is this a shape" test, only by the box.
_CLUTTER: tuple[tuple[str, dict[str, int]], ...] = (
    ("none", {}),
    ("distractors", {"distractors": 6}),
    ("occluders", {"occluders": 3}),
    ("both", {"distractors": 6, "occluders": 3}),
)

#: The three bands from ``docs/datasets/difficulty.md``, kept as the keyword arguments that page's
#: table lists so the picture and the prose cannot drift apart. Changing a band there means changing
#: it here and re-running this script.
_BANDS: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "easy",
        {
            "background": NoiseBackground(sigma=12.0),
            "min_size_ratio": 0.10,
            "max_size_ratio": 0.30,
        },
    ),
    (
        "moderate",
        {
            "background": TextureBackground(frequency=8.0),
            "min_size_ratio": 0.08,
            "max_size_ratio": 0.25,
            "distractors": 3,
            "degrade": (GaussianBlur(radius=0.5), JPEG(quality=75)),
        },
    ),
    (
        "hard",
        {
            "background": TextureBackground(frequency=8.0),
            "shapes": tuple(LetterShape),
            "min_size_ratio": 0.03,
            "max_size_ratio": 0.25,
            "min_objects": 4,
            "max_objects": 8,
            "distractors": 6,
            "degrade": (GaussianBlur(radius=0.5), JPEG(quality=75)),
        },
    ),
)


def _bare_canvas(config: SyntheticConfig, seed: int) -> Image.Image:
    """Return the canvas the sample was drawn on, before a single object reached it.

    Reproduced rather than intercepted: :meth:`SyntheticGenerator.sample` takes its four side streams
    as the first thing it does, so spawning the same four children off a generator at :data:`_SEED`
    hands the background the stream state it was rendered with, and the left panel is the right
    panel's canvas pixel for pixel. A configuration whose knobs draw nothing skips the spawn
    entirely, which changes nothing here: the only background that skips it is one that ignores the
    stream anyway.

    The degradation chain runs over the bare canvas too, since a degradation is a property of the
    whole image rather than of the objects in it -- that is what makes a pair show what an effect
    costs on flat pixels next to what it costs on an edge.

    """
    background = config.resolved_background
    canvas_stream, _clutter, _occluders, degrade_stream = np.random.default_rng(seed).spawn(_SIDE_STREAM_ROLES)
    pixels = background.render(canvas_stream if background.consumes_randomness else None, config.img_size)
    image = np.array(pixels, dtype=np.uint8)
    for step in config.degrade:
        image = step.apply(image, degrade_stream if step.consumes_randomness else None)
    return Image.fromarray(image)


def _annotated_scene(config: SyntheticConfig, seed: int, overlay: str = "detection") -> Image.Image:
    """Return the first sample of a stream at ``seed``, with one overlay drawn over every object.

    Args:
        config: The configuration to sample from.
        seed: Stream seed.
        overlay: Which of the three exported geometries to draw -- ``"detection"`` for the axis-aligned box,
            ``"obb"`` for the oriented one, ``"segmentation"`` for the polygon. The scene itself is identical
            whichever is asked for: an annotation carries all three off one placement, and the generator draws
            no extra randomness for a task, so switching the overlay never moves an object.

    """
    sample = next(iter(SyntheticGenerator(config).generate(1, seed=seed)))
    scene = Image.fromarray(sample.image)
    draw = ImageDraw.Draw(scene)
    for annotation in sample.annotations:
        if overlay == "detection":
            draw.rectangle(annotation.bbox_xyxy, outline=_BBOX_RGB, width=_BBOX_WIDTH)
            continue
        flat = annotation.polygon if overlay == "segmentation" else annotation.obb_corners
        # Point fields are in pixel-centre space while the canvas is in edge space, so the half-pixel
        # goes back on before drawing -- without it an outline sits visibly off its own ink.
        points = [
            (flat[index] + PIXEL_CENTRE_OFFSET, flat[index + 1] + PIXEL_CENTRE_OFFSET)
            for index in range(0, len(flat), 2)
        ]
        draw.line([*points, points[0]], fill=_BBOX_RGB, width=_BBOX_WIDTH, joint="curve")
    return scene


def _panels(overrides: dict[str, object], *, seed: int = _SEED, overlay: str = "detection") -> list[Image.Image]:
    """Return the panels of one picture: the bare canvas, then that same canvas with its objects.

    Every knob not named in ``overrides`` keeps its package default, so one picture differs from the
    next by exactly what its page says it does.

    Args:
        overrides: ``SyntheticConfig`` fields for this picture. A mapping rather than ``**kwargs`` so a group's
            own field table can be forwarded whole without a keyword of this function's ever colliding with a
            config field name.
        seed: Stream seed. One shared seed across every group, so a picture is comparable with any other.
        overlay: Which exported geometry to draw over the objects -- see :func:`_annotated_scene`.

    """
    defaults: dict[str, object] = {"img_size": _PANEL, "max_objects": 4}
    config = SyntheticConfig(**{**defaults, **overrides})
    return [_bare_canvas(config, seed), _annotated_scene(config, seed, overlay)]


def _caption_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the caption face: the scalable default where Pillow offers one, the bitmap default otherwise.

    ``ImageFont.load_default`` grew its ``size`` argument in Pillow 10.1, and the package floor is ``pillow>=10``. The
    bitmap fallback is legible but tight -- its inter-word space is a single pixel, which reads as a missing space at
    this panel width -- so it is a fallback rather than the choice.

    """
    try:
        return ImageFont.load_default(size=_CAPTION_SIZE)
    except TypeError:  # pragma: no cover - only on pillow 10.0
        return ImageFont.load_default()


def _compose(panels: list[Image.Image], captions: tuple[str, ...] = _CAPTIONS) -> Image.Image:
    """Lay the panels out side by side and caption each one underneath.

    The captions are drawn here rather than inside :func:`_panels` because a group may scale its panels first: text
    rendered before a resize would be resampled along with the pixels, and a bitmap font does not survive that.

    A caption per panel rather than one per picture, because the two halves are not the same kind of thing -- one is the
    canvas the generator painted, the other is that canvas plus objects plus what the dataset exports about them, and a
    reader meeting the pair on its own has nothing else to tell them apart.

    """
    font = _caption_font()
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + _DIVIDER * (len(panels) - 1)
    picture = Image.new("RGB", (width, height + _CAPTION_HEIGHT), _PAPER)
    draw = ImageDraw.Draw(picture)
    offset = 0
    for panel, caption in zip(panels, captions, strict=True):
        picture.paste(panel, (offset, 0))
        text_width = draw.textlength(caption, font=font)
        draw.text((offset + (panel.width - text_width) / 2.0, height + 5), caption, fill=_INK, font=font)
        offset += panel.width + _DIVIDER
    return picture


def _render(
    overrides: dict[str, object], *, seed: int = _SEED, overlay: str = "detection", caption: str | None = None
) -> Image.Image:
    """Return one finished, captioned picture for ``overrides``, ``caption`` naming the right panel."""
    captions = _CAPTIONS if caption is None else (_CAPTIONS[0], caption)
    return _compose(_panels(overrides, seed=seed, overlay=overlay), captions)


def _photographic(folder: Path) -> list[tuple[str, Image.Image]]:
    """Return the two ``ImageBackground`` pictures, cropped from a directory built in ``folder``.

    The pictures are synthesized rather than shipped: the point is that the canvas came from a file,
    which a generated file makes as well as a photograph would, and a binary added to the repository
    only to be cropped into two previews would be carried forever.

    """
    grid = np.linspace(0.0, 2.0 * np.pi, 240, dtype=np.float64)
    x, y = np.meshgrid(grid, grid)
    for index, (level, period) in enumerate(((70.0, 3.0), (95.0, 5.0), (120.0, 8.0))):
        # Banded, mid-grey and low-saturation on purpose: a crop has to read as a backdrop that a
        # coloured shape still stands out against, which a bright or saturated one does not.
        texture = level + 26.0 * np.sin(period * x) * np.cos(period * y) + 14.0 * np.sin(2.0 * period * y)
        rgb = np.stack([texture + 10.0, texture, texture - 8.0], axis=-1)
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(folder / f"picture_{index}.png")
    return [
        ("image", _render({"background": ImageBackground(folder)})),
        ("image-grayscale", _render({"background": ImageBackground(folder, grayscale=True)})),
    ]


def _backgrounds() -> list[tuple[str, Image.Image]]:
    """Every canvas mode under one placement stream: six procedural, then two read from files."""
    procedural = [(slug, _render({"background": background})) for slug, background in _BACKGROUNDS]
    # The pictures the last two crop from exist only for the length of this call.
    with tempfile.TemporaryDirectory() as folder:
        return procedural + _photographic(Path(folder))


def _degradations() -> list[tuple[str, Image.Image]]:
    """One scene, first clean and then under each degradation on its own."""
    return [(slug, _render({"degrade": () if step is None else (step,)})) for slug, step in _DEGRADATIONS]


def _clutter() -> list[tuple[str, Image.Image]]:
    """The two unlabelled-ink knobs, alone and together.

    Paired like every other group even though neither knob touches the canvas, so the left panel is the
    same flat fill four times over: a reader moving between groups should not have to work out why one
    of them shows a single panel. What separates these pictures is on the right anyway -- only the
    labelled objects wear a box, so the unboxed ink is the distractors and the occluders.

    """
    return [(slug, _render(dict(knobs))) for slug, knobs in _CLUTTER]


def _bands() -> list[tuple[str, Image.Image]]:
    """Each documented band under each overlay, at the canvas size its table was measured on.

    Rendered at 256 pixels and scaled to the panel size afterwards, because the bands are size *ratios*: 0.03 of a
    256-pixel canvas is the eight-pixel glyph the hard band is named for, and rendering the band at a smaller canvas
    would quietly make that glyph smaller still. Nearest-neighbour on the way down, so a thin stroke stays a thin stroke
    rather than being averaged into the canvas. When the render size already matches the panel the resize is a no-op and
    is skipped, which keeps those pixels exact.

    One scene per band under each of :data:`_OVERLAYS`, rather than one overlay over three scenes: what a band costs is
    paid by a detector, and the three geometries over the same objects show what is actually being asked of one -- a box
    around an eight-pixel glyph, the same glyph's rotation, the same glyph's outline.

    """
    pictures = []
    for slug, knobs in _BANDS:
        for overlay, caption in _OVERLAYS:
            panels = _panels({"img_size": 256, **knobs}, overlay=overlay)
            scaled = [
                panel if panel.width == _PANEL else panel.resize((_PANEL, _PANEL), Image.Resampling.NEAREST)
                for panel in panels
            ]
            pictures.append((f"{slug}-{overlay}", _compose(scaled, (_CAPTIONS[0], caption))))
    return pictures


#: One builder per group, so ``--groups`` validates against the same mapping that renders.
_GROUPS: dict[str, Callable[[], list[tuple[str, Image.Image]]]] = {
    "backgrounds": _backgrounds,
    "degradations": _degradations,
    "clutter": _clutter,
    "difficulty": _bands,
}

#: Output folder per group. A group is a directory rather than a file-name prefix, so regenerating one
#: replaces a subtree and a reader browsing the assets sees the same grouping the pages use.
_FOLDERS = {"backgrounds": "backgrounds", "degradations": "degradations", "clutter": "clutter", "difficulty": "bands"}


def main(output_dir: str = "docs/assets/datasets/scene", groups: str = "all", image_format: str = "webp") -> None:
    """Write one picture per knob setting.

    Args:
        output_dir: Root for the pictures, each written as ``<group folder>/<slug>.<format>`` beneath it.
        groups: One of ``backgrounds``, ``degradations``, ``clutter``, ``difficulty``, or ``all``.
        image_format: Still-image container, ``webp`` or ``png``.

    Examples:
        >>> callable(main)
        True

    """
    assert image_format in ("webp", "png"), f"--image_format must be 'webp' or 'png', got {image_format!r}"
    assert groups in (*_GROUPS, "all"), f"--groups must be one of {(*_GROUPS, 'all')}, got {groups!r}"
    wanted = tuple(_GROUPS) if groups == "all" else (groups,)

    root = Path(output_dir)
    for group in wanted:
        out_dir = root / _FOLDERS[group]
        out_dir.mkdir(parents=True, exist_ok=True)
        for slug, picture in _GROUPS[group]():
            path = out_dir / f"{slug}.{image_format}"
            # Lossless, and measured rather than assumed. Lossy WebP at `quality=100` still changes
            # 90.8% of the noise preview's pixels (max delta 168) and 10.4% of the JPEG preview's
            # (max 106), which is precisely the per-pixel behaviour these files document -- a JPEG
            # preview carrying a second encoder's artefacts is a misleading picture. In the other
            # direction there is nothing left to win: `method=6` and `quality=100` move the lossless
            # size by at most 0.1% (sometimes the wrong way) and PNG runs 12-26% larger. The 93 KB
            # `backgrounds/noise` file is full-frame Gaussian grain and is incompressible by
            # construction, not by oversight.
            picture.save(path, lossless=True) if image_format == "webp" else picture.save(path)
            print(f"wrote {path}  ({picture.width}x{picture.height})")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
