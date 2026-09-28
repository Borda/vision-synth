"""Seeded synthetic-image generator producing format-agnostic samples.

:class:`SyntheticGenerator` draws colored shapes on a canvas that
:mod:`~synth_datasets.content.backgrounds` fills — flat grey by default — rejecting
placements that fall off the canvas or overlap existing objects, and returns a
:class:`~synth_datasets.core.sample.Sample` carrying every annotation
representation (polygon, axis-aligned box, oriented box). All randomness flows
through a caller-supplied :class:`numpy.random.Generator`, so a fixed seed yields
byte-identical output.

Coordinate conventions follow the transforms each field travels through, so a generated label can be
handed straight to a pipeline. Outlines and landmarks are emitted in **pixel-centre** space, which is
what :func:`~fused_transforms.targets.transform_keypoints` and the image resampler assume;
``bbox_xyxy`` is emitted in **pixel-edge** space, which is what
:func:`~fused_transforms.targets.transform_bbox_xyxy` assumes. The rasterizer itself draws the
edge-space outline, since Pillow fills pixel ``floor(x)`` for a vertex at ``x``. Mixing the two up
costs a full pixel under any reflection or quarter turn, which is why each field declares its space
rather than sharing one; see :data:`~synth_datasets.families.geometry.PIXEL_CENTRE_OFFSET`.

Under :attr:`~synth_datasets.core.config.Task.KEYPOINTS` each annotation also carries its
family's landmarks plus the schema naming them, derived from the placement that was already
sampled — no extra random draw — so a seed produces the same scene whatever the configured task.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from synth_datasets.core.config import SyntheticConfig
    >>> from synth_datasets.core.generator import SyntheticGenerator
    >>> gen = SyntheticGenerator(SyntheticConfig(img_size=64, min_objects=2, max_objects=2))
    >>> sample = gen.sample(np.random.default_rng(0))
    >>> sample.image.shape
    (64, 64, 3)
    >>> len(sample.annotations)
    2

    ```

"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from synth_datasets.core.config import Fill, Task, as_canvas_size, class_vocabulary
from synth_datasets.core.sample import _EMPTY_SCENE, Annotation, Sample, SceneRecord
from synth_datasets.families import keypoint_schema_for, place_keypoints, shape_outline
from synth_datasets.families.geometry import bbox_iou, polygon_to_bbox_xyxy, to_pixel_centre
from synth_datasets.families.primitives import PrimitiveShape

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from synth_datasets.core.config import SyntheticConfig
    from synth_datasets.families import Shape

_BBox = tuple[float, float, float, float]
#: One sampled placement's full result: shape, color, the polygon bbox/OBB are measured from, and
#: the landmark table (``None`` off the keypoints task or for a shape with no schema).
_Placement = tuple["Shape", Fill, "NDArray[np.float64]", float, "NDArray[np.float64] | None"]

#: COCO visibility flags emitted for a landmark: ``2`` is "labeled and visible", ``1`` is "labeled
#: but not visible", ``0`` is "not labeled". A landmark is hidden by the canvas frame or by an absent
#: optional point (a NaN row from :mod:`~synth_datasets.families.animals`), and occluded when
#: ``occluders`` put an unlabelled shape over it — the placement loop rejects overlapping *labelled*
#: objects, so one labelled shape never covers another's landmark.
#: One side stream per consumer, in this fixed order: background, distractors, occluders,
#: degradations. Four children rather than one shared child, so a knob's draw count cannot shift a
#: different knob's — two backgrounds painting byte-identical canvases while consuming different
#: numbers of draws would otherwise place different clutter.
_SIDE_STREAM_ROLES = 4

_KEYPOINT_VISIBLE = 2
_KEYPOINT_OCCLUDED = 1
_KEYPOINT_HIDDEN = 0


def _visible_keypoints(
    points: NDArray[np.float64], img_size: int | tuple[int, int], occluder_mask: NDArray[np.bool_] | None = None
) -> tuple[tuple[float, float, int], ...]:
    """Tag each landmark with its COCO visibility flag, zeroing the ones off the canvas.

    Args:
        points: ``(num_keypoints, 2)`` landmark coordinates in image pixels.
        img_size: Canvas side length, or ``(width, height)``, in pixels.
        occluder_mask: ``(height, width)`` raster of what an occluder covers, or ``None``. A
            landmark inside the canvas but under the mask is demoted from :data:`_KEYPOINT_VISIBLE`
            to :data:`_KEYPOINT_OCCLUDED`, keeping its coordinates — COCO's "labeled but not
            visible". An already-hidden point is never promoted: it is off the canvas, so the mask
            has nothing to say about it.

    Returns:
        One ``(x, y, visibility)`` triple per landmark, in input order: the coordinates converted to
        pixel-centre space with :data:`_KEYPOINT_VISIBLE` while the point lies inside
        ``[0, width)`` by ``[0, height)``, and ``(0.0, 0.0)`` with :data:`_KEYPOINT_HIDDEN` otherwise.
        Zeroing a clipped point (rather than keeping its off-canvas coordinates) is COCO's
        "not labeled" convention, and the zeroed placeholder is *not* shifted -- it is a flag value,
        not a position. An absent hind limb lands here as ``(nan, nan)``; both comparisons in
        ``0.0 <= nan < width`` are false, so it falls to the hidden branch with no special-casing.

        Visibility is decided on the *incoming* edge-space coordinates, so which landmarks the frame
        hides does not depend on the convention the surviving ones are reported in.

    """
    width, height = as_canvas_size(img_size)
    triples: list[tuple[float, float, int]] = []
    for x, y in points:
        inside = 0.0 <= x < width and 0.0 <= y < height
        if not inside:
            triples.append((0.0, 0.0, _KEYPOINT_HIDDEN))
            continue
        # Edge-space coordinates index the stencil directly. Which pixels an outline covers is the
        # rasterizer's answer, not a geometric one -- Pillow fills *through* a max vertex, so a
        # polygon spanning 2..6 paints columns 2 through 6 inclusive. The canvas and the stencil take
        # the same outline through the same rasterizer, so a landmark flagged occluded is always one
        # that occluder-coloured pixels actually cover.
        covered = occluder_mask is not None and bool(occluder_mask[int(y), int(x)])
        centre = to_pixel_centre(np.array([x, y], dtype=np.float64))
        triples.append((float(centre[0]), float(centre[1]), _KEYPOINT_OCCLUDED if covered else _KEYPOINT_VISIBLE))
    return tuple(triples)


def _scene(occluder_mask: NDArray[np.bool_] | None, background_source: str | None) -> SceneRecord:
    """Return the side-car for one sample, reusing the shared empty record when there is nothing to say.

    Args:
        occluder_mask: What an occluder covered, or ``None``.
        background_source: What a photographic background was cropped from, or ``None``.

    Returns:
        A populated :class:`SceneRecord`, or the shared empty one, which allocates nothing and is
        safe to share precisely because both its fields are immutable scalars.

    """
    if occluder_mask is None and background_source is None:
        return _EMPTY_SCENE
    return SceneRecord(occluder_mask=occluder_mask, background_source=background_source)


def _boundary_overlap(bbox: _BBox, img_size: int | tuple[int, int]) -> float:
    """Return the fraction of ``bbox`` area lying outside the canvas.

    Args:
        bbox: Candidate box ``(x_min, y_min, x_max, y_max)``.
        img_size: Canvas side length, or ``(width, height)``, in pixels.

    Returns:
        ``0.0`` when fully inside, up to ``1.0`` when fully outside.

    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 1.0
    width, height = as_canvas_size(img_size)
    inside_w = max(0.0, min(width, x2) - max(0.0, x1))
    inside_h = max(0.0, min(height, y2) - max(0.0, y1))
    return 1.0 - (inside_w * inside_h) / area


class SyntheticGenerator:
    """Draw colored shapes into reproducible :class:`Sample` objects.

    Args:
        config: Generation knobs; see :class:`~synth_datasets.core.config.SyntheticConfig`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from synth_datasets.core.config import SyntheticConfig
        >>> from synth_datasets.core.generator import SyntheticGenerator
        >>> gen = SyntheticGenerator(SyntheticConfig(img_size=32))
        >>> a = gen.sample(np.random.default_rng(1))
        >>> b = gen.sample(np.random.default_rng(1))
        >>> bool(np.array_equal(a.image, b.image))
        True

        ```

    """

    def __init__(self, config: SyntheticConfig) -> None:
        """Store config and precompute the class vocabulary and keypoint schema this run uses.

        The vocabulary is narrowed to ``config.shapes``, the same list every writer declares as its
        ``categories``/``names`` block, so an annotation's ``class_id`` always resolves against the vocabulary
        written beside it (see :func:`~synth_datasets.core.config.class_vocabulary`). The schema is resolved
        once here and stamped onto every landmark-bearing annotation, so a table always travels with the
        family that produced it.

        """
        self.config = config
        self.vocabulary = class_vocabulary(config.class_mode, config.shapes, config.colors)
        self.class_names = self.vocabulary.names
        self.keypoint_schema = keypoint_schema_for(config.shapes)
        self._needs_side_stream = (
            config.resolved_background.consumes_randomness
            or bool(config.distractors)
            or bool(config.occluders)
            or any(step.consumes_randomness for step in config.degrade)
        )

    def _side_streams(self, rng: np.random.Generator) -> tuple[np.random.Generator | None, ...]:
        """Return one child stream per non-placement consumer, or four ``None`` when nothing draws.

        Args:
            rng: The caller's generator, reserved for object placement and never drawn from here.

        Returns:
            Four streams in the order :data:`_SIDE_STREAM_ROLES` counts — background, distractors,
            occluders, degradations — or four ``None`` when this configuration draws nothing at all.

        :meth:`numpy.random.Generator.spawn` advances the parent's seed-sequence child counter, not
        its bit stream, so the placements that follow are bit-identical whether or not children were
        taken — which is what lets a knob be switched on without moving an object at a fixed seed.
        Children are nonetheless taken only when something will draw, so a configuration using no
        such knob leaves even the child counter where it was.

        One child **per consumer** rather than one shared child, because a shared child couples every
        knob to every other: the background draws first, so changing it shifts what the distractors
        then draw. Two backgrounds painting byte-identical canvases while consuming different numbers
        of draws demonstrably placed different clutter before this split, which would also have made
        a per-knob proxy measurement unattributable.

        Each :meth:`sample` call takes its own children, so two images in one stream get independent
        non-placement randomness.

        Deferring the draws to the end of :meth:`sample` instead would be correct within one image
        and wrong across a stream: :meth:`generate` reuses one generator for every sample, so draws
        made after image *n*'s placements would shift image *n+1*'s.

        """
        if not self._needs_side_stream:
            return (None,) * _SIDE_STREAM_ROLES
        return tuple(rng.spawn(_SIDE_STREAM_ROLES))

    def _attempt_placement(
        self,
        rng: np.random.Generator,
        kept: list[_BBox],
        shapes: tuple[Shape, ...] | None = None,
        colors: tuple[Fill, ...] | None = None,
        *,
        with_landmarks: bool = True,
        respect_boundary: bool = True,
    ) -> _Placement | None:
        """Draw one candidate shape; return it if in-bounds and non-overlapping, else ``None``.

        Exactly one candidate is sampled per call (fixed RNG draw order: shape, color, size, centre x, centre y, angle
        when rotation applies, then an asymmetry skew when ``asymmetry_jitter`` is set), so the caller controls the
        retry budget and the RNG consumption stays deterministic for a given seed. That order is the same whichever
        pools are passed, so the unlabelled callers below cannot perturb it.

        Args:
            rng: The stream to draw this candidate from. Labelled objects pass the caller's placement stream;
                unlabelled clutter passes the side stream, so clutter never moves an object.
            kept: Boxes the candidate must not overlap beyond ``cfg.overlap_iou``. Pass an empty list for clutter,
                which is meant to be overlapped and has no IoU constraint of its own.
            shapes: Shape pool to draw from; ``cfg.shapes`` when ``None``. Drawing from the run's own pool rather than
                the full :class:`Shape` vocabulary is why widening either enum never changes an existing seeded
                configuration.
            colors: Fill pool to draw from; ``cfg.colors`` when ``None``.
            with_landmarks: Compute the landmark table when the run's task asks for one. ``False`` for clutter, which
                is never annotated and so never carries landmarks whatever ``cfg.task`` says.
            respect_boundary: Reject a candidate falling further outside the canvas than ``cfg.boundary_tolerance``.
                ``False`` for an occluder, which is realistic precisely when it runs off the frame.

        Returns:
            The sampled placement, or ``None`` when it was rejected.

        Landmarks (the last tuple element — ``None`` off the keypoints task or for a shape with no schema) are a pure
        function of the placement that was just sampled, so computing them consumes no further randomness and leaves
        every seeded stream unchanged.

        """
        cfg = self.config
        shapes = cfg.shapes if shapes is None else shapes
        colors = cfg.colors if colors is None else colors
        shape = shapes[int(rng.integers(len(shapes)))]
        color = colors[int(rng.integers(len(colors)))]
        width, height = cfg.canvas_size
        # Draw order is fixed -- size, then x, then y -- so a square canvas replays its stream exactly.
        size_px = float(rng.uniform(cfg.min_size_ratio, cfg.max_size_ratio)) * min(width, height)
        center = (float(rng.uniform(0, width)), float(rng.uniform(0, height)))
        angle = float(rng.uniform(0, 2 * np.pi)) if cfg.rotate and shape is not PrimitiveShape.CIRCLE else 0.0
        skew = (
            float(rng.uniform(-cfg.asymmetry_jitter, cfg.asymmetry_jitter))
            if cfg.asymmetry_jitter and shape is not PrimitiveShape.CIRCLE
            else 0.0
        )
        poly = shape_outline(shape.value, center, size_px, angle, skew)
        bbox = polygon_to_bbox_xyxy(poly)
        if respect_boundary and _boundary_overlap(bbox, (width, height)) > cfg.boundary_tolerance:
            return None
        if any(bbox_iou(bbox, other) > cfg.overlap_iou for other in kept):
            return None
        wants_landmarks = with_landmarks and cfg.task is Task.KEYPOINTS
        keypoints = place_keypoints(shape, center, size_px, angle, skew) if wants_landmarks else None
        return shape, color, poly, angle, keypoints

    def _draw_clutter(
        self,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        count: int,
        *,
        respect_boundary: bool = True,
        stencil: ImageDraw.ImageDraw | None = None,
    ) -> list[_BBox]:
        """Draw ``count`` unlabelled shapes and return the boxes they covered.

        Args:
            draw: The canvas to paint onto.
            rng: The side stream — never the placement stream, so clutter cannot move an object.
            count: How many items to attempt.
            respect_boundary: Keep items inside ``cfg.boundary_tolerance``. ``True`` for distractors,
                which are background clutter; ``False`` for an occluder, which is realistic precisely
                when it runs off the frame.
            stencil: A parallel one-bit canvas to paint the same outlines onto, or ``None``. Only the
                occluders use it, to record what they cover.

        Returns:
            One box per item that was actually placed, in draw order.

        Each item gets its own ``max_placement_attempts`` budget and is silently skipped when it runs
        out. Nothing raises: clutter is unlabelled, so a missing piece costs a dataset nothing, where
        a missing labelled object would cost it a class balance. No IoU constraint applies either —
        being overlapped is the entire point — and no landmark table is ever computed, whatever
        ``cfg.task`` says.

        """
        cfg = self.config
        covered: list[_BBox] = []
        for _ in range(count):
            for _attempt in range(cfg.max_placement_attempts):
                placed = self._attempt_placement(
                    rng,
                    [],
                    cfg.resolved_distractor_shapes,
                    cfg.resolved_distractor_colors,
                    with_landmarks=False,
                    respect_boundary=respect_boundary,
                )
                if placed is None:
                    continue
                _shape, color, poly, _angle, _points = placed
                outline = [(float(x), float(y)) for x, y in poly]
                draw.polygon(outline, fill=color.rgb)
                if stencil is not None:
                    stencil.polygon(outline, fill=1)
                covered.append(polygon_to_bbox_xyxy(poly))
                break
        return covered

    def sample(self, rng: np.random.Generator) -> Sample:
        """Generate one image and its annotations.

        Args:
            rng: Random generator driving object count, shapes, colors, and placement. The
                background, the clutter, the occluders and the degradations each draw from a child
                of it rather than from it directly (see :meth:`_side_streams`), so none of them moves
                an object and none of them moves another.

        Returns:
            A :class:`Sample` with an RGB ``uint8`` image and one annotation per drawn shape.

        Raises:
            RuntimeError: If fewer than ``min_objects`` shapes could be placed within the overall
                attempt budget (``num_objects * max_placement_attempts``); relax ``overlap_iou`` or
                ``boundary_tolerance``, lower ``min_objects``, or raise ``max_placement_attempts``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from synth_datasets.core.config import SyntheticConfig
            >>> from synth_datasets.core.generator import SyntheticGenerator
            >>> gen = SyntheticGenerator(SyntheticConfig(img_size=48, min_objects=1, max_objects=3))
            >>> s = gen.sample(np.random.default_rng(7))
            >>> 1 <= len(s.annotations) <= 3
            True

            ```

        """
        cfg = self.config
        canvas_stream, clutter_stream, occluder_stream, degrade_stream = self._side_streams(rng)
        background = cfg.resolved_background
        width, height = cfg.canvas_size
        # A square canvas is passed as a plain int, so a Background subclass written for the int
        # signature keeps working whichever way the config spells a square; only a rectangle is a pair.
        pixels, background_source = background.render_with_source(
            canvas_stream if background.consumes_randomness else None, width if width == height else (width, height)
        )
        canvas = Image.fromarray(pixels)
        draw = ImageDraw.Draw(canvas)
        if cfg.distractors and clutter_stream is not None:
            self._draw_clutter(draw, clutter_stream, cfg.distractors)
        num_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))

        placements: list[_Placement] = []
        kept: list[_BBox] = []
        # Retry failed placements against a shared budget so we reach num_objects when feasible
        # instead of silently dropping objects; the budget bounds RNG draws deterministically.
        budget = num_objects * cfg.max_placement_attempts
        for _ in range(budget):
            if len(placements) >= num_objects:
                break
            placed = self._attempt_placement(rng, kept)
            if placed is None:
                continue
            # The rasterizer and the AABB both read the outline in edge space; only the point
            # fields are converted, because only they travel through a pixel-centre matrix.
            draw.polygon([(float(x), float(y)) for x, y in placed[2]], fill=placed[1].rgb)
            kept.append(polygon_to_bbox_xyxy(placed[2]))
            placements.append(placed)
        if len(placements) < cfg.min_objects:
            raise RuntimeError(
                f"could not place the required min_objects={cfg.min_objects} shapes within the "
                f"placement budget of {budget} attempts (placed {len(placements)}); relax overlap_iou/"
                f"boundary_tolerance, lower min_objects, or raise max_placement_attempts"
            )
        # Occluders go on last of the canvas-side steps, so they cover the labelled shapes rather
        # than sit behind them -- which is also why annotations are built only after this point.
        occluder_mask = (
            self._draw_occluders(draw, occluder_stream) if cfg.occluders and occluder_stream is not None else None
        )
        annotations = [
            self._annotate(placed, bbox, occluder_mask) for placed, bbox in zip(placements, kept, strict=True)
        ]
        # ``np.asarray`` when nothing will touch the pixels, exactly as before degradations existed:
        # it may hand back a read-only view onto Pillow's own buffer, which costs no copy. A chain
        # needs a buffer it owns, so that case — and only that case — pays for ``np.array``.
        image = np.array(canvas, dtype=np.uint8) if cfg.degrade else np.asarray(canvas)
        for step in cfg.degrade:
            image = step.apply(image, degrade_stream if step.consumes_randomness else None)
        return Sample(
            image=image,
            annotations=annotations,
            width=width,
            height=height,
            scene=_scene(occluder_mask, background_source),
        )

    def _annotate(self, placed: _Placement, bbox: _BBox, occluder_mask: NDArray[np.bool_] | None) -> Annotation:
        """Turn one accepted placement into its annotation.

        Args:
            placed: The tuple :meth:`_attempt_placement` returned.
            bbox: The axis-aligned box already measured from its outline.
            occluder_mask: What an occluder covers, or ``None`` when none were drawn.

        Returns:
            The annotation, with landmark visibility resolved against the mask.

        Built after the occluders rather than inside the placement loop, because a landmark's
        visibility depends on what was drawn over it and the occluders are drawn last. The polygon
        and the box are unaffected either way: a COCO polygon describes the object, not its visible
        part, so the mask a writer rasterizes from it is **amodal**. A consumer wanting a modal mask
        subtracts ``sample.scene.occluder_mask`` from it.

        """
        shape, color, poly, angle, points = placed
        class_id = self.vocabulary.id_of(shape, color)
        return Annotation(
            class_id=class_id,
            class_name=self.class_names[class_id],
            polygon=[float(v) for v in to_pixel_centre(poly).reshape(-1)],
            bbox_xyxy=bbox,
            angle=angle,
            keypoints=None if points is None else _visible_keypoints(points, self.config.canvas_size, occluder_mask),
            keypoint_schema=None if points is None else self.keypoint_schema,
        )

    def _draw_occluders(self, draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> NDArray[np.bool_]:
        """Draw the occluders over everything already on the canvas and return what they cover.

        Args:
            draw: The canvas, already carrying the background, the clutter and the labelled shapes.
            rng: The side stream, so an occluder never moves the object it covers.

        Returns:
            A ``(height, width)`` read-only boolean raster, ``True`` where an occluder covers
            the pixel.

        The mask is rasterized from the same outlines into a parallel one-bit image rather than
        recovered from the finished pixels, which would be unable to tell an occluder from an object
        that happened to share its colour. It is marked read-only before it leaves: a frozen
        :class:`~synth_datasets.core.sample.SceneRecord` stops the field being rebound and does
        nothing to stop a caller mutating the buffer, and a mutated mask would disagree with the
        visibility flags already computed from it.

        """
        cfg = self.config
        stencil = Image.new("1", cfg.canvas_size, 0)  # Pillow takes (width, height)
        self._draw_clutter(draw, rng, cfg.occluders, respect_boundary=False, stencil=ImageDraw.Draw(stencil))
        mask = np.asarray(stencil, dtype=bool).copy()
        mask.flags.writeable = False
        return mask

    def generate(self, num_images: int, seed: int | np.random.SeedSequence | None = None) -> Iterator[Sample]:
        """Lazily yield ``num_images`` samples from a fresh seeded generator.

        This is the streaming primitive: only one :class:`Sample` is materialized at
        a time, so both in-memory training feeds and writing very large datasets stay
        within a bounded memory footprint. All samples draw from a single
        :class:`numpy.random.Generator`, so a fixed ``seed`` yields a reproducible stream.

        Args:
            num_images: Number of samples to yield.
            seed: Integer or independent stream seed for the internal generator; ``None`` uses fresh entropy.

        Yields:
            One :class:`Sample` per iteration.

        Examples:
            ```pycon
            >>> from synth_datasets.core.config import SyntheticConfig
            >>> from synth_datasets.core.generator import SyntheticGenerator
            >>> gen = SyntheticGenerator(SyntheticConfig(img_size=32))
            >>> samples = list(gen.generate(3, seed=0))
            >>> len(samples)
            3

            ```

        """
        rng = np.random.default_rng(seed)
        for _ in range(num_images):
            yield self.sample(rng)
