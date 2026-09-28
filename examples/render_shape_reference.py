"""Render one static, monochrome reference image per shape, with its detection box and keypoints.

Unlike ``animate_synthetic_dataset.py``'s colored, randomly placed, animated previews, these are
plain single-shape images: upright, at each shape's own authored orientation (no rotation, no
skew) — a field-guide-style lookup of what each shape and its keypoint schema actually look like,
not a sample of what the generator draws in practice. Every symbol and animal is authored
mirror-symmetric about its own vertical axis (see :mod:`~synth_datasets.families.symbols`), so
rendering at that authored angle is what keeps a reference recognizable: an ``arrow`` pointing up,
a ``house`` with its roof up, a ``kite`` on its long axis.

The box drawn in yellow is the shape's plain axis-aligned **detection** box (``bbox``) at this
reference orientation. Since :func:`~synth_datasets.families.geometry.polygon_to_obb` derives the
oriented box in the shape's own upright frame, this same box *is* what the **OBB** task exports at
this unrotated pose — the generator's rotated samples carry it turned rigidly with the shape (see
the animated OBB preview and the "Tasks" section). The three
keypoint-bearing families (animals, symbols, letters) get their landmark dots and skeleton drawn in
the same orange ``animate_synthetic_dataset.py`` uses for a visible-but-occluded keypoint. The
geometric family has no keypoint schema, so its images carry only the outline and the box. There
is no filename caption baked into the image — one shape per file already names it.

Each shape is scaled independently — not to one shared size — to the largest ``size`` that fits
its own outline inside the canvas margin, so a thin shape like the geometric triangle or a tall
one like the giraffe each fill as much of their own frame as their own proportions allow, rather
than leaving the fixed-size margin of a shared scale empty.

Files are named ``<prefix><shape>.png`` with the same prefix convention
``animate_synthetic_dataset.py`` uses for its animated previews (``geometry-``, ``animals-``,
``symbols-``, ``letters-``), so e.g. ``symbols-arrow.png``, ``animals-duck.png``, and
``letters-x.png``.

A letter (see :mod:`~synth_datasets.families.letters`) is a single outline polygon exactly like a
geometric, animal, or symbol shape, so it reuses ``shape_outline`` the same way every other family
does. Its skeleton differs per member (that is what makes it that letter), though, unlike the
animal/symbol families' one shared topology, so its edges come from
:meth:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for` instead of a fixed tuple.

Render every shape in every family:
    python examples/render_shape_reference.py

Render one family:
    python examples/render_shape_reference.py --families symbols

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from synth_datasets.families import shape_outline
from synth_datasets.families.animals import ANIMAL_KEYPOINT_SCHEMA, AnimalShape, animal_keypoints
from synth_datasets.families.geometry import polygon_to_bbox_xyxy
from synth_datasets.families.letters import LETTER_KEYPOINT_SCHEMA, LetterShape, letter_keypoints
from synth_datasets.families.primitives import PrimitiveShape
from synth_datasets.families.symbols import SYMBOL_KEYPOINT_SCHEMA, SymbolShape, symbol_keypoints

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from synth_datasets.core.keypoints import KeypointSchema

_CANVAS = 200
#: Pixels reserved on every side of the canvas so a fitted shape's stroke width and keypoint dot
#: radius never clip at the edge.
_MARGIN = 12
_DOT_RADIUS = 3
_INK = (0, 0, 0)
_FILL = (222, 222, 222)
_PAPER = (255, 255, 255)
#: Drawn for the plain axis-aligned detection box (not the rotated OBB — see the module docstring).
#: Yellow, matching ``render_scene_gallery.py`` and ``animate_synthetic_dataset.py``, so a box means the
#: same thing in a field-guide still, a scene preview and a clip. Nothing in the drawing vocabulary is
#: yellow: the nearest fill anywhere is the ``sand`` clutter at 25.9 CIEDE2000, and the object fills the
#: generator draws from are 43.9 (green) or further.
_BBOX_RGB = (255, 255, 0)
#: A dark keyline drawn immediately outside the box. Yellow on this white page is a 1.07:1 contrast
#: ratio -- invisible as a one-pixel line -- so the keyline carries the contrast (21:1) and the yellow
#: carries the meaning. The scene previews need no such thing: they draw over a mid-grey canvas.
_BBOX_KEYLINE = (60, 60, 60)
#: Matches ``animate_synthetic_dataset.py``'s ``_KEYPOINT_COLORS[1]`` — drawn for landmark dots and
#: the skeleton; more visible than yellow at this dot/line size.
_KEYPOINT_RGB = (255, 165, 0)
_SHAPE_CENTER = (_CANVAS / 2.0, _CANVAS / 2.0)

#: A shape family's file prefix, roster, keypoint placer (``None`` for the schema-less geometric
#: family), and keypoint schema (``None`` to match) — everything :func:`_render_shape` needs, keyed
#: by ``--families`` choice.
_FAMILIES: dict[
    str,
    tuple[str, tuple[str, ...], Callable[..., NDArray[np.float64]] | None, KeypointSchema | None],
] = {
    "geometric": ("geometry-", tuple(s.value for s in PrimitiveShape), None, None),
    "symbols": ("symbols-", tuple(s.value for s in SymbolShape), symbol_keypoints, SYMBOL_KEYPOINT_SCHEMA),
    "animals": ("animals-", tuple(s.value for s in AnimalShape), animal_keypoints, ANIMAL_KEYPOINT_SCHEMA),
    "letters": ("letters-", tuple(s.value for s in LetterShape), letter_keypoints, LETTER_KEYPOINT_SCHEMA),
}

#: Which enum member class a given family's shape values belong to, for building the keypoint
#: function's argument. ``None`` for the schema-less geometric family.
_SHAPE_CLASSES: dict[str, type[AnimalShape | SymbolShape | LetterShape] | None] = {
    "geometric": None,
    "symbols": SymbolShape,
    "animals": AnimalShape,
    "letters": LetterShape,
}


def _fit_size(name: str) -> float:
    """Return the largest ``size`` keeping ``name``'s drawn outline within the canvas margin.

    Computed from a unit-size (``size=1.0``) render at the shape's authored (unrotated)
    orientation: the drawn detection box always equals the outline's own bounding box at this
    orientation (see :func:`_draw_shape`), so fitting the outline alone is sufficient.

    """
    poly = shape_outline(name, (0.0, 0.0), 1.0)
    half_extent = float(np.abs(poly).max())
    half_available = (_CANVAS - 2 * _MARGIN) / 2.0
    return half_available / half_extent


def _draw_shape(draw: ImageDraw.ImageDraw, name: str, size: float) -> None:
    """Draw one shape's outline (filled light gray, black edge) and its yellow detection box.

    The box is the plain axis-aligned bounding box over the outline at this reference orientation — which, at
    an unrotated pose, is exactly the upright-frame oriented box
    :func:`~synth_datasets.families.geometry.polygon_to_obb` derives for the OBB task (see the module
    docstring).

    """
    poly = shape_outline(name, _SHAPE_CENTER, size)
    draw.polygon([(float(x), float(y)) for x, y in poly], outline=_INK, fill=_FILL, width=2)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    draw.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1), outline=_BBOX_KEYLINE, width=1)
    draw.rectangle((x1, y1, x2, y2), outline=_BBOX_RGB, width=1)


def _draw_keypoints(
    draw: ImageDraw.ImageDraw,
    keypoints_fn: Callable[..., NDArray[np.float64]],
    shape: AnimalShape | SymbolShape | LetterShape,
    skeleton: tuple[tuple[int, int], ...],
    size: float,
) -> None:
    """Draw one shape's skeleton edges then its landmark dots, skipping absent (NaN) rows."""
    points = keypoints_fn(shape, _SHAPE_CENTER, size)
    visible = {i: (float(x), float(y)) for i, (x, y) in enumerate(points) if x == x}  # x == x excludes NaN
    for first, second in skeleton:
        if first in visible and second in visible:
            draw.line([visible[first], visible[second]], fill=_KEYPOINT_RGB, width=1)
    for x, y in visible.values():
        draw.ellipse((x - _DOT_RADIUS, y - _DOT_RADIUS, x + _DOT_RADIUS, y + _DOT_RADIUS), fill=_KEYPOINT_RGB)


def _render_shape(
    name: str,
    shape_cls: type[AnimalShape | SymbolShape | LetterShape] | None,
    keypoints_fn: Callable[..., NDArray[np.float64]] | None,
    schema: KeypointSchema | None,
    output_path: Path,
) -> Path:
    """Render one shape's reference image and write it to ``output_path``."""
    size = _fit_size(name)
    canvas = Image.new("RGB", (_CANVAS, _CANVAS), _PAPER)
    draw = ImageDraw.Draw(canvas)
    _draw_shape(draw, name, size)
    if keypoints_fn is not None and shape_cls is not None and schema is not None:
        _draw_keypoints(draw, keypoints_fn, shape_cls(name), schema.skeleton_for(name), size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main(output_dir: str = "docs/assets/shape-references", families: str = "all") -> None:
    """Render one monochrome, canonical-orientation reference image per shape.

    Args:
        output_dir: Directory the ``<prefix><shape>.png`` images are written into.
        families: ``all``, or a comma-separated subset of ``geometric``, ``symbols``, ``animals``,
            ``letters``.

    Examples:
        >>> callable(main)
        True

    """
    chosen = tuple(_FAMILIES) if families == "all" else tuple(f.strip() for f in families.split(","))
    assert all(f in _FAMILIES for f in chosen), f"--families must be a subset of {tuple(_FAMILIES)}, got {families!r}"
    out = Path(output_dir)
    for family in chosen:
        prefix, names, keypoints_fn, schema = _FAMILIES[family]
        for name in names:
            path = _render_shape(name, _SHAPE_CLASSES[family], keypoints_fn, schema, out / f"{prefix}{name}.png")
            print(f"wrote {path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
