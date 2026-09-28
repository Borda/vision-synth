"""Animate synthetic-dataset previews, one looping clip per annotation task.

The static companion (three PNGs under ``docs/assets/datasets/tasks``) showed a single
frame per task. This script renders a short looping animation instead: it draws a
fixed, seeded stream of synthetic images and, for every task, cycles through them
showing each image first bare and then with its yellow annotation overlay drawn
back on — axis-aligned boxes for detection, filled-shape polygons for
segmentation, oriented boxes for OBB, and landmark dots plus a skeleton for keypoints.

Because all tasks share the same seeded sample stream, the clips line up
image-for-image: the same shapes appear in each, differing only in the label type
overlaid, so the annotation representations can be compared directly.

Frames are written as an animated WebP (smaller than GIF, matching the existing
WebP assets); ``--image_format gif`` is available as a fallback.

``--shapes`` picks which vocabulary is drawn: ``geometric`` (square, rectangle,
triangle, circle) writes ``geometry-<task>.webp``, ``animals`` (the twelve
side-profile silhouettes) writes ``animals-<task>.webp``, ``symbols`` (the
seven analytic symbols) writes ``symbols-<task>.webp``, and ``letters`` (the
twenty-six capital-letter stroke figures) writes ``letters-<task>.webp`` so all
four sets can live side by side in the docs. Each vocabulary sets its own scene
(``SCENES``) and clip length (``CLIP_IMAGES``), trading object size for object
count as the vocabulary grows: a preview is only a field guide if every member
of the family actually turns up in it, and the generator draws each object's
shape uniformly, so twenty-six letters need many times the drawn objects that
four geometric shapes do. Coverage is then confirmed rather than hoped for —
``_covering_stream`` walks the seed forward until the stream contains every
shape, and prints which seed it settled on.

A letter (see :mod:`~synth_datasets.families.letters`) is a single outline polygon exactly like a
geometric, animal, or symbol shape, so its ``segmentation``/``obb`` overlay draws the same one
closed loop through ``ann.polygon``/``ann.obb_corners`` every other family uses. Its keypoints
overlay differs per letter, though — unlike the animal/symbol families' one shared topology, a
letter's skeleton edges come from
:meth:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for` per
annotation instead of one schema-wide tuple.

Render every task (detection, segmentation, obb; keypoints when using animals, symbols, or letters):
    python examples/animate_synthetic_dataset.py

Render the animal-shape previews:
    python examples/animate_synthetic_dataset.py --shapes animals

Render the symbol-shape previews:
    python examples/animate_synthetic_dataset.py --shapes symbols

Render the letter-shape previews:
    python examples/animate_synthetic_dataset.py --shapes letters

Render one task:
    python examples/animate_synthetic_dataset.py --task obb --img_size 320 --num_images 6

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from PIL import Image, ImageDraw

from synth_datasets import SyntheticConfig, SyntheticGenerator
from synth_datasets.core.config import DEFAULT_SHAPES, Task, keypoint_schema_for
from synth_datasets.core.sample import Annotation, Sample
from synth_datasets.families.animals import AnimalShape
from synth_datasets.families.geometry import PIXEL_CENTRE_OFFSET
from synth_datasets.families.letters import LetterShape
from synth_datasets.families.symbols import SymbolShape

if TYPE_CHECKING:
    from synth_datasets.core.keypoints import KeypointSchema

TASKS = ("detection", "segmentation", "obb", "keypoints")


class _Scene(TypedDict):
    """Typed scene-size settings shared by the two preview vocabularies."""

    min_objects: int
    max_objects: int
    min_size_ratio: float
    max_size_ratio: float


#: Drawable vocabulary per ``--shapes`` choice; ``animals``/``symbols``/``letters`` ask for their
#: whole roster, so a new animal, symbol, or letter is picked up without editing this script.
SHAPE_SETS = {
    "geometric": DEFAULT_SHAPES,
    "animals": tuple(AnimalShape),
    "symbols": tuple(SymbolShape),
    "letters": tuple(LetterShape),
}
#: Scene knobs per vocabulary. ``geometric`` keeps the original scene and seed. The other
#: three size their objects against how many members they have to get through: the generator picks
#: each object's shape uniformly, so a preview only shows a whole family if the stream draws enough
#: objects to reach every member (see :func:`_covering_stream`). Counts therefore rise, and sizes
#: fall, with vocabulary size — twenty-six letters need far more draws than seven symbols — but a
#: floor stays under the size, because a letter's strokes are a sixth of its extent and thin out
#: into nothing well before a silhouette would.
SCENES: dict[str, _Scene] = {
    "geometric": {"min_objects": 5, "max_objects": 7, "min_size_ratio": 0.1, "max_size_ratio": 0.3},
    "animals": {"min_objects": 5, "max_objects": 8, "min_size_ratio": 0.16, "max_size_ratio": 0.34},
    "symbols": {"min_objects": 4, "max_objects": 6, "min_size_ratio": 0.18, "max_size_ratio": 0.36},
    "letters": {"min_objects": 8, "max_objects": 12, "min_size_ratio": 0.14, "max_size_ratio": 0.26},
}
#: Default clip length (distinct images cycled) per vocabulary, overridable with ``--num_images``.
#: A longer clip is the other half of the coverage budget :data:`SCENES` opens: the twenty-six
#: letters need roughly a hundred drawn objects between them, which is a dozen images even at the
#: raised object count, while four geometric shapes are covered many times over in six.
CLIP_IMAGES = {"geometric": 6, "animals": 8, "symbols": 8, "letters": 12}
#: Filename prefix per vocabulary — every family now gets an explicit prefix (``geometric`` used to
#: write unprefixed ``detection.webp``/``segmentation.webp``/``obb.webp``; renamed to ``geometry-``
#: so all four vocabularies read consistently in a file listing).
PREFIXES = {"geometric": "geometry-", "animals": "animals-", "symbols": "symbols-", "letters": "letters-"}
_OVERLAY_RGB = (255, 255, 0)  # yellow annotation overlay, matching the static previews
_KEYPOINT_COLORS = {1: (255, 165, 0), 2: (255, 255, 0)}  # visible points are bright yellow
_SUPERSAMPLE = 2  # render overlays at 2x then downscale so diagonal edges read smooth
_BARE_MS = 500  # hold the un-annotated image briefly before the label appears
_LABELLED_MS = 1300  # hold the annotated image long enough to read the overlay


def _pairs(flat: list[float], scale: float) -> list[tuple[float, float]]:
    """Turn a flat ``[x1, y1, x2, y2, ...]`` list into scaled ``(x, y)`` point tuples.

    Point fields are in pixel-centre coordinates while the canvas being drawn on is in edge ones, so the offset goes
    back on before scaling -- at ``_SUPERSAMPLE`` magnification half a pixel is visible as an overlay sitting off its
    own ink.

    """
    return [
        ((flat[i] + PIXEL_CENTRE_OFFSET) * scale, (flat[i + 1] + PIXEL_CENTRE_OFFSET) * scale)
        for i in range(0, len(flat), 2)
    ]


def _draw_annotation(
    draw: ImageDraw.ImageDraw,
    ann: Annotation,
    task: str,
    scale: float,
    width: int,
    schema: KeypointSchema | None = None,
) -> None:
    """Draw one annotation's overlay for the given task onto ``draw``.

    ``schema`` is the active run's keypoint schema — see
    :func:`~synth_datasets.core.config.keypoint_schema_for` — and is only read for the
    ``"keypoints"`` task, resolved to ``ann``'s own edges via
    :meth:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for` (identical to
    ``schema.skeleton`` for every family but letters, whose topology genuinely differs per member);
    every other task ignores it.

    """
    if task == "detection":
        x1, y1, x2, y2 = (v * scale for v in ann.bbox_xyxy)
        draw.rectangle((x1, y1, x2, y2), outline=_OVERLAY_RGB, width=width)
        return
    if task == "keypoints":
        if ann.keypoints is None or schema is None:
            return
        visible = {
            index: ((x + PIXEL_CENTRE_OFFSET) * scale, (y + PIXEL_CENTRE_OFFSET) * scale)
            for index, (x, y, visibility) in enumerate(ann.keypoints)
            if visibility > 0
        }
        for first, second in schema.skeleton_for(ann.class_name):
            if first in visible and second in visible:
                draw.line((visible[first], visible[second]), fill=_OVERLAY_RGB, width=max(1, width // 2))
        # Landmarks pack closely together on the same silhouette; a bit more radius than
        # width // 2 keeps each dot legible instead of thinning into the skeleton lines.
        radius = max(3, width)
        for _index, (x, y, visibility) in enumerate(ann.keypoints):
            if visibility <= 0:
                continue
            px, py = (x + PIXEL_CENTRE_OFFSET) * scale, (y + PIXEL_CENTRE_OFFSET) * scale
            color = _KEYPOINT_COLORS.get(visibility, _OVERLAY_RGB)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        return
    points = _pairs(ann.polygon if task == "segmentation" else ann.obb_corners, scale)
    draw.line([*points, points[0]], fill=_OVERLAY_RGB, width=width, joint="curve")


def _render_frame(
    sample: Sample, task: str, out_size: int, annotate: bool, schema: KeypointSchema | None = None
) -> Image.Image:
    """Render one sample as a PIL frame, optionally with the task overlay drawn on."""
    canvas = Image.fromarray(sample.image).resize((out_size * _SUPERSAMPLE,) * 2, Image.Resampling.NEAREST)
    if annotate:
        scale = out_size * _SUPERSAMPLE / sample.width
        draw = ImageDraw.Draw(canvas)
        for ann in sample.annotations:
            _draw_annotation(draw, ann, task, scale, width=_SUPERSAMPLE * 2, schema=schema)
    return canvas.resize((out_size, out_size), Image.Resampling.LANCZOS)


def _build_frames(
    samples: list[Sample], task: str, out_size: int, schema: KeypointSchema | None = None
) -> tuple[list[Image.Image], list[int]]:
    """Build the ordered (image, duration) frame lists cycling bare then annotated per sample."""
    images: list[Image.Image] = []
    durations: list[int] = []
    for sample in samples:
        images.append(_render_frame(sample, task, out_size, annotate=False))
        durations.append(_BARE_MS)
        images.append(_render_frame(sample, task, out_size, annotate=True, schema=schema))
        durations.append(_LABELLED_MS)
    return images, durations


def _save_animation(images: list[Image.Image], durations: list[int], output_path: Path, image_format: str) -> None:
    """Write frames as an animated WebP or GIF loop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {
        "save_all": True,
        "append_images": images[1:],
        "duration": durations,
        "loop": 0,
    }
    if image_format == "webp":
        save_kwargs["quality"] = 90
        save_kwargs["method"] = 6
    else:
        save_kwargs["disposal"] = 2
        save_kwargs["optimize"] = True
    images[0].save(output_path, format=image_format.upper(), **save_kwargs)


#: How many seeds past the requested one :func:`_covering_stream` may try before giving up. Two or
#: three is the usual answer at the counts :data:`SCENES` asks for, so a cap this high is only ever
#: reached by a genuinely unreachable request — a vocabulary too large for the clip to hold.
_SEED_ATTEMPTS = 16


def _covering_stream(config: SyntheticConfig, num_images: int, seed: int) -> tuple[int, list[Sample]]:
    """Return the first seed at or after ``seed`` whose stream draws every class, and its samples.

    Which shapes a seeded stream happens to contain is a fact about that seed rather than a
    probability, because the preview is one fixed stream and not a sampling run — so it can simply be
    checked. That matters most for the twenty-six letters: the generator picks each object's shape
    uniformly, which by the coupon-collector bound needs roughly 145 drawn objects before every
    letter is *likely* present, several times what a legible scene holds. Walking the seed forward to
    one that does cover the vocabulary gets there without inflating the object count past legibility,
    and keeps the guarantee under the bare regeneration command in ``AGENTS.md`` — a seed hand-picked
    on the command line would be lost the next time somebody regenerates the assets.

    Args:
        config: The generator settings for the run; its ``class_names`` are what must all appear.
        num_images: Number of distinct images in the clip.
        seed: First seed to try; the search walks upward from here.

    Returns:
        The seed that covers the vocabulary and the samples it produced.

    Raises:
        AssertionError: If no seed within :data:`_SEED_ATTEMPTS` covers it, naming what was missing —
            the scene is then too sparse for the vocabulary, so raise the object count or the clip
            length rather than the seed budget.

    """
    generator = SyntheticGenerator(config)
    expected = set(generator.class_names)
    missing: set[str] = set()
    for candidate in range(seed, seed + _SEED_ATTEMPTS):
        samples = list(generator.generate(num_images, seed=candidate))
        missing = expected - {ann.class_name for sample in samples for ann in sample.annotations}
        if not missing:
            return candidate, samples
    raise AssertionError(
        f"no seed in {seed}..{seed + _SEED_ATTEMPTS - 1} drew all {len(expected)} classes over "
        f"{num_images} images (last run missed {sorted(missing)}); raise the scene's object count "
        f"or --num_images"
    )


def render(
    samples: list[Sample],
    task: str,
    output_dir: Path,
    out_size: int,
    image_format: str,
    prefix: str = "",
    schema: KeypointSchema | None = None,
) -> Path:
    """Render one task's animation from a shared sample stream and return its path."""
    images, durations = _build_frames(samples, task, out_size, schema=schema)
    output_path = output_dir / f"{prefix}{task}.{image_format}"
    _save_animation(images, durations, output_path, image_format)
    return output_path


def main(
    output_dir: str = "docs/assets/datasets/tasks",
    task: str = "all",
    img_size: int = 320,
    num_images: int | None = None,
    seed: int = 0,
    image_format: str = "webp",
    shapes: str = "geometric",
) -> None:
    """Generate looping preview animations for the synthetic-dataset annotation tasks.

    Args:
        output_dir: Directory for the generated animations.
        task: One of ``detection``, ``segmentation``, ``obb``, ``keypoints``, or ``all``.
        img_size: Output canvas side length in pixels.
        num_images: Number of distinct images cycled in each clip; the vocabulary's own
            :data:`CLIP_IMAGES` default when omitted. Lowering it can leave a large vocabulary
            uncoverable — see :func:`_covering_stream`, which then raises rather than quietly
            shipping a preview missing shapes.
        seed: First seed for the shared sample stream; the run uses the first seed at or after it
            that draws every shape in the vocabulary (see :func:`_covering_stream`), and prints
            which one that was. Output stays byte-identical for a given ``seed``.
        image_format: Animation container, ``webp`` or ``gif``.
        shapes: Drawable vocabulary, ``geometric``, ``animals``, ``symbols``, or ``letters``; the
            latter three write ``<shapes>-<task>`` files so all four preview sets can coexist.

    Examples:
        >>> callable(main)
        True

    """
    assert image_format in ("webp", "gif"), f"--image_format must be 'webp' or 'gif', got {image_format!r}"
    assert shapes in SHAPE_SETS, f"--shapes must be one of {tuple(SHAPE_SETS)}, got {shapes!r}"
    tasks: tuple[str, ...] = TASKS if task == "all" else (task,)
    assert all(t in TASKS for t in tasks), f"--task must be one of {TASKS} or 'all', got {task!r}"
    schema = keypoint_schema_for(SHAPE_SETS[shapes])
    if schema is None and "keypoints" in tasks:
        if task == "all":
            tasks = TASKS[:-1]
        else:
            raise AssertionError("--task keypoints requires --shapes animals, --shapes symbols, or --shapes letters")

    # One shared, seeded stream so every task clip shows the same shapes, only the overlay differs.
    config_task = Task.KEYPOINTS if "keypoints" in tasks else Task.DETECTION
    config = SyntheticConfig(
        img_size=img_size, rotate=True, task=config_task, shapes=SHAPE_SETS[shapes], **SCENES[shapes]
    )
    images = CLIP_IMAGES[shapes] if num_images is None else num_images
    used_seed, samples = _covering_stream(config, images, seed)
    print(
        f"seed {used_seed}: {sum(len(s.annotations) for s in samples)} objects over {images} images, all shapes shown"
    )

    out_dir = Path(output_dir)
    for a_task in tasks:
        print(f"wrote {render(samples, a_task, out_dir, img_size, image_format, PREFIXES[shapes], schema)}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
