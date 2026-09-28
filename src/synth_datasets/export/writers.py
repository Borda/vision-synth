"""Serialize generated samples to COCO or YOLO dataset layouts.

Both writers consume the same format-agnostic
:class:`~synth_datasets.core.sample.Sample` objects and select which
annotation fields to emit based on the requested
:class:`~synth_datasets.core.config.Task`.

COCO layout (Roboflow-style)::

    <out>/<split>/img_000000.jpg
    <out>/<split>/_annotations.coco.json

YOLO layout (Ultralytics-style)::

    <out>/images/<split>/img_000000.jpg
    <out>/labels/<split>/img_000000.txt
    <out>/data.yaml

COCO has no native oriented-box field, so for :attr:`Task.OBB` the four corners are
stored as a 4-point ``segmentation`` polygon alongside the axis-aligned ``bbox``.

For :attr:`Task.KEYPOINTS` both writers emit the landmark block in addition to the box: COCO gains
a ``segmentation`` polygon (as for :attr:`Task.SEGMENTATION`) plus per-category ``keypoints``/``skeleton``
and per-annotation ``keypoints``/``num_keypoints``, and YOLO appends ``x y v`` triples to the detection
row and declares ``kpt_shape`` plus a horizontal-flip mapping ``flip_idx`` in ``data.yaml``. An
annotation that carries no landmarks — one
generated for a different task and then handed to a keypoint writer — is written as an all-zero,
visibility-``0`` ("not labeled") table rather than a short record, so every row and record still
matches the schema the task declares.

"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from synth_datasets.core.config import ClassVocabulary, OutputFormat, Task
from synth_datasets.families.geometry import PIXEL_CENTRE_OFFSET

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from synth_datasets.core.config import ClassEntry
    from synth_datasets.core.keypoints import KeypointSchema
    from synth_datasets.core.sample import Annotation, Sample

_IMAGE_STEM = "img_{index:06d}"


def _covers(entry: ClassEntry, schema: KeypointSchema) -> bool:
    """Return whether ``schema`` can describe the landmarks of the class ``entry`` names.

    This used to be two functions that rebuilt structure out of a class *name*: one recreated the full
    color-by-shape cross product to test membership, the other recovered the shape half with
    ``name.partition("_")``. Both were correct only while no shape value and no color value contained an
    underscore. :class:`~synth_datasets.core.config.ClassEntry` carries the shape itself, so the test is now
    what it always meant: does this class name a shape of the run's own keypoint family?

    A ``ClassMode.COLOR`` entry names no shape at all (``entry.shape is None``) yet is still drawn as whichever family
    the run was restricted to, so it is always covered.

    """
    return entry.shape is None or str(entry.shape.value) in schema.shape_values


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive ``[lo, hi]`` range."""
    return max(lo, min(hi, value))


def _clamp_flat(flat: list[float], img_w: float, img_h: float) -> list[float]:
    """Clamp a flat ``[x1, y1, ...]`` coordinate list to the image extent."""
    return [_clamp(v, 0.0, img_w) if i % 2 == 0 else _clamp(v, 0.0, img_h) for i, v in enumerate(flat)]


def _edge_flat(flat: list[float]) -> list[float]:
    """Convert a flat pixel-centre coordinate list to the edge space an exported file uses.

    An :class:`~synth_datasets.core.sample.Annotation` carries outlines, oriented-box corners
    and landmarks in pixel-centre space, because that is the space the point transforms in
    :mod:`~fused_transforms.targets` move them through. ``bbox_xyxy`` stays in edge space for the
    same reason -- that is the space its own transform assumes. A COCO or YOLO file has no room for
    two conventions: its ``segmentation`` ring, its landmarks and its ``bbox`` are read as one
    coordinate system, and that system is edge space, so the point fields are converted back here,
    at the file boundary, and nowhere else.

    Args:
        flat: ``[x1, y1, x2, y2, ...]`` coordinates in pixel-centre space.

    Returns:
        The same list shifted into edge space.

    """
    return [v + PIXEL_CENTRE_OFFSET for v in flat]


def _keypoint_triples(
    ann: Annotation, img_w: float, img_h: float, schema: KeypointSchema
) -> list[tuple[float, float, int]]:
    """Return one ``(x, y, visibility)`` landmark triple per keypoint, clamped to the image.

    Args:
        ann: The annotation to read landmarks from.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        schema: The active run's keypoint schema — its ``names`` order and count are what an
            annotation without landmarks falls back to.

    Returns:
        One triple per name in ``schema.names``, in that order. A visible point is converted from
        the annotation's pixel-centre space to the file's edge space (see :func:`_edge_flat`) and
        clamped to the image extent like every other coordinate field; an invisible one keeps the
        zeroed placeholder coordinates, unshifted, rather than being clamped into a spurious corner
        position — ``(0.0, 0.0)`` there is a flag value, not a location. An annotation without
        landmarks yields the all-zero, "not labeled" table — see the module docstring.

    """
    if ann.keypoints is None:
        return [(0.0, 0.0, 0)] * len(schema.names)
    return [
        (_clamp(x + PIXEL_CENTRE_OFFSET, 0.0, img_w), _clamp(y + PIXEL_CENTRE_OFFSET, 0.0, img_h), visibility)
        if visibility > 0
        else (0.0, 0.0, visibility)
        for x, y, visibility in ann.keypoints
    ]


def _save_image(image: NDArray[Any], path: Path) -> None:
    """Write an RGB ``uint8`` array to ``path`` as JPEG."""
    Image.fromarray(image).save(path, quality=95)


class DatasetWriter(ABC):
    """Base class for dataset serializers.

    Args:
        task: The annotation task determining which fields are emitted.
        vocabulary: The classes to declare, in id order — each entry keeps the shape and color it
            was derived from, which is how a writer tells which categories its keypoint schema
            covers without parsing their names.
        keypoint_schema: The keypoint family a :attr:`~synth_datasets.core.config.Task.KEYPOINTS`
            run draws from; required for that task and ignored for every other. It used to default
            to the animal schema for backward compatibility, which meant a directly-constructed
            symbol or letter pose writer silently emitted a 16-landmark animal header over 7- or
            15-landmark rows. There is no safe default, so there is none.

    """

    def __init__(self, task: Task, vocabulary: ClassVocabulary, keypoint_schema: KeypointSchema | None = None) -> None:
        """Store the task and vocabulary, rejecting a keypoints task with no schema to write.

        Raises:
            ValueError: If ``task`` is :attr:`~synth_datasets.core.config.Task.KEYPOINTS` and no
                ``keypoint_schema`` was given.

        """
        if task is Task.KEYPOINTS and keypoint_schema is None:
            raise ValueError(
                "Task.KEYPOINTS needs a keypoint_schema naming the family being written; pass the "
                "one keypoint_schema_for(config.shapes) returns"
            )
        self.task = task
        self.vocabulary = vocabulary
        self.class_names = vocabulary.names
        self.keypoint_schema = keypoint_schema

    @property
    def schema(self) -> KeypointSchema:
        """Return the keypoint schema, which the constructor guarantees for a keypoints task.

        Every landmark-writing path reaches the schema through here rather than through the
        optional attribute, so the "a keypoints writer always has one" invariant is stated once and
        checked, instead of being asserted implicitly at four call sites.

        Raises:
            ValueError: If no schema was supplied — only reachable by mutating the attribute after
                construction, since the constructor rejects a keypoints task without one.

        """
        if self.keypoint_schema is None:
            raise ValueError("this writer has no keypoint schema; it was not built for Task.KEYPOINTS")
        return self.keypoint_schema

    @abstractmethod
    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write all splits under ``output_dir``.

        **Consume each split exactly once, in the order given.** The splits
        :func:`~synth_datasets.generate_dataset` passes are lazy views over a *single*
        shared sample stream, so iterating them out of order, twice, or partially does not merely
        repeat work — it silently redistributes samples between splits or empties them. An
        implementation that needs a split more than once must materialize it itself, accepting the
        memory that costs.

        Args:
            splits: Mapping of split name to its samples, in the order they must be consumed.
            output_dir: Destination root directory (created if absent).

        """


class CocoWriter(DatasetWriter):
    """Write a COCO-format dataset, one JSON per split.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, Task, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.export.writers import CocoWriter
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,))
        >>> CocoWriter(Task.DETECTION, vocab).task.value
        'detection'

        ```

    """

    def _annotation_dict(self, ann: Annotation, ann_id: int, image_id: int, img_w: int, img_h: int) -> dict[str, Any]:
        """Build one COCO annotation record, clamping geometry to the image extent."""
        x1 = _clamp(ann.bbox_xyxy[0], 0, img_w)
        y1 = _clamp(ann.bbox_xyxy[1], 0, img_h)
        x2 = _clamp(ann.bbox_xyxy[2], 0, img_w)
        y2 = _clamp(ann.bbox_xyxy[3], 0, img_h)
        width, height = x2 - x1, y2 - y1
        record = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann.class_id + 1,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        if self.task is Task.SEGMENTATION:
            record["segmentation"] = [_clamp_flat(_edge_flat(ann.polygon), img_w, img_h)]
        elif self.task is Task.OBB:
            record["segmentation"] = [_clamp_flat(_edge_flat(ann.obb_corners), img_w, img_h)]
        elif self.task is Task.KEYPOINTS:
            record["segmentation"] = [_clamp_flat(_edge_flat(ann.polygon), img_w, img_h)]
            triples = _keypoint_triples(ann, img_w, img_h, self.schema)
            record["keypoints"] = [value for triple in triples for value in triple]
            record["num_keypoints"] = sum(1 for *_, visibility in triples if visibility > 0)
        return record

    def _categories(self) -> list[dict[str, Any]]:
        """Build the category records, adding the keypoint schema to every category it covers.

        Under ``ClassMode.SHAPE`` or ``ClassMode.SHAPE_COLOR`` naming the vocabulary can span shapes
        outside the run's own keypoint family — every primitive-shape category always, plus every
        category of another keypoint-bearing family when one is active. Decorating those with a
        schema they can never produce a matching annotation for would misdescribe the dataset to any
        COCO consumer, so only the categories :func:`_covers` accepts are decorated.

        A category's ``skeleton`` prefers
        :meth:`~synth_datasets.core.keypoints.KeypointSchema.skeleton_for` — the letter family's
        per-letter stroke edges — falling back to the family-wide ``skeleton`` for a bare-color
        category or a family whose members all share one topology (animals, symbols).

        """
        categories: list[dict[str, Any]] = [
            {"id": entry.index + 1, "name": entry.name, "supercategory": "none"} for entry in self.vocabulary.entries
        ]
        if self.task is not Task.KEYPOINTS or self.keypoint_schema is None:
            return categories
        schema = self.keypoint_schema
        for entry, category in zip(self.vocabulary.entries, categories, strict=True):
            if not _covers(entry, schema):
                continue
            category["keypoints"] = list(schema.names)
            skeleton = schema.skeleton if entry.shape is None else schema.skeleton_for(str(entry.shape.value))
            # COCO skeleton edges are 1-based indices into the category's own keypoint list.
            category["skeleton"] = [[i + 1, j + 1] for i, j in skeleton]
        return categories

    def _coco_doc(self, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap image and annotation records into a COCO document with categories."""
        categories = self._categories()
        return {
            "info": {"description": "vision-synth synthetic dataset"},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Stream each split to ``<output_dir>/<split>/`` in a single pass over its samples.

        Image pixels are written as they are produced and never held in memory. The COCO schema,
        however, emits one JSON document per split, so lightweight per-image and per-annotation
        metadata records (no pixels) accumulate for the duration of the split and are serialized
        once the split is exhausted: memory is O(n) in the split's image and annotation counts, not
        constant. For a constant-memory path use the YOLO writer (one label file per image) or the
        in-memory :class:`~synth_datasets.export.datasets.SyntheticIterableDataset`.

        """
        output_dir = Path(output_dir)
        for split, samples in splits.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            ann_id = 1
            for image_id, sample in enumerate(samples):
                stem = _IMAGE_STEM.format(index=image_id)
                _save_image(sample.image, split_dir / f"{stem}.jpg")
                images.append({
                    "id": image_id,
                    "file_name": f"{stem}.jpg",
                    "width": sample.width,
                    "height": sample.height,
                })
                for ann in sample.annotations:
                    annotations.append(self._annotation_dict(ann, ann_id, image_id, sample.width, sample.height))
                    ann_id += 1
            doc = self._coco_doc(images, annotations)
            (split_dir / "_annotations.coco.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


class YoloWriter(DatasetWriter):
    """Write a YOLO-format dataset with normalized labels and a ``data.yaml``.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, Task, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.export.writers import YoloWriter
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,))
        >>> YoloWriter(Task.OBB, vocab).task.value
        'obb'

        ```

    """

    @staticmethod
    def _box_coords(ann: Annotation, width: int, height: int) -> list[float]:
        """Return the normalized ``cx cy w h`` of the box clipped to the image extent."""
        # Clamp corners to the image extent before deriving cx/cy/w/h so an edge-crossing box's
        # label matches its clipped visible box (consistent with CocoWriter._annotation_dict).
        x1 = _clamp(ann.bbox_xyxy[0], 0.0, width)
        y1 = _clamp(ann.bbox_xyxy[1], 0.0, height)
        x2 = _clamp(ann.bbox_xyxy[2], 0.0, width)
        y2 = _clamp(ann.bbox_xyxy[3], 0.0, height)
        return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]

    def _keypoint_tokens(self, ann: Annotation, width: int, height: int) -> list[str]:
        """Return the trailing ``x y v`` tokens of a pose row, three per landmark.

        Coordinates are normalized and clamped like every other coordinate; the visibility flag is an index into COCO's
        scale, so it is written as a plain integer and never normalized.

        """
        tokens: list[str] = []
        for x, y, visibility in _keypoint_triples(ann, float(width), float(height), self.schema):
            tokens += [f"{_clamp(x / width, 0.0, 1.0):.6f}", f"{_clamp(y / height, 0.0, 1.0):.6f}", str(visibility)]
        return tokens

    def _label_row(self, ann: Annotation, width: int, height: int) -> str:
        """Format one YOLO label row for the writer's task, clamping coordinates to ``[0, 1]``."""
        if self.task in (Task.DETECTION, Task.KEYPOINTS):
            # A pose row is a detection row plus the landmark block (Ultralytics' order).
            coords = self._box_coords(ann, width, height)
        else:
            flat = _edge_flat(ann.polygon if self.task is Task.SEGMENTATION else ann.obb_corners)
            coords = [v / width if i % 2 == 0 else v / height for i, v in enumerate(flat)]
        tokens = [str(ann.class_id), *(f"{_clamp(c, 0.0, 1.0):.6f}" for c in coords)]
        if self.task is Task.KEYPOINTS:
            tokens += self._keypoint_tokens(ann, width, height)
        return " ".join(tokens)

    def _write_split(self, split: str, samples: Iterable[Sample], output_dir: Path) -> None:
        """Write images and label files for one split."""
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            stem = _IMAGE_STEM.format(index=index)
            _save_image(sample.image, img_dir / f"{stem}.jpg")
            rows = [self._label_row(ann, sample.width, sample.height) for ann in sample.annotations]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def _data_yaml(self, splits: dict[str, Iterable[Sample]]) -> str:
        """Build the ``data.yaml`` contents referencing present splits."""
        if not splits:  # defensive backstop; write() is the primary guard
            raise ValueError("YoloWriter requires at least one split, got an empty mapping")
        lines = ["path: .", f"train: images/{'train' if 'train' in splits else next(iter(splits))}"]
        if "val" in splits:
            lines.append("val: images/val")
        if "test" in splits:
            lines.append("test: images/test")
        lines.append(f"nc: {len(self.class_names)}")
        if self.task is Task.KEYPOINTS:
            # Ultralytics carries one dataset-wide (num_keypoints, dims) shape, hence one shared
            # landmark schema for every class; dims is 3 because each point ships its visibility.
            lines.append(f"kpt_shape: [{self.schema.kpt_shape}, 3]")
            # ``flip_idx`` names the landmark each one becomes under a horizontal flip — see
            # KeypointSchema.flip_idx for what makes each family's mapping (identity for animals,
            # a genuine left/right swap for symbols) correct.
            lines.append(f"flip_idx: {list(self.schema.flip_idx)}")
        lines.append("names:")
        lines.extend(f"  {i}: {name}" for i, name in enumerate(self.class_names))
        return "\n".join(lines) + "\n"

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write images, labels, and ``data.yaml`` under ``output_dir``.

        Raises:
            ValueError: If ``splits`` is empty (checked before any output directory is created).

        """
        if not splits:
            raise ValueError("YoloWriter requires at least one split, got an empty mapping")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, samples in splits.items():
            self._write_split(split, samples, output_dir)
        (output_dir / "data.yaml").write_text(self._data_yaml(splits), encoding="utf-8")


#: Writer class per output format. A dispatch table rather than an if/else so a third party can add
#: a format (Pascal VOC, CVAT, a house schema) without forking this module — see
#: :func:`register_writer`. The two built-ins register themselves below.
_WRITERS: dict[str, type[DatasetWriter]] = {}


def register_writer(fmt: OutputFormat | str, writer: type[DatasetWriter]) -> None:
    """Register the writer class serving one output format.

    Args:
        fmt: The format key. An :class:`~synth_datasets.core.config.OutputFormat` member for the
            built-ins, or any string for a custom format — :func:`get_writer` accepts both, so a
            caller can pass ``fmt="voc"`` straight to
            :func:`~synth_datasets.generate_dataset` once registered.
        writer: A concrete :class:`DatasetWriter` subclass.

    Raises:
        TypeError: If ``writer`` is not a :class:`DatasetWriter` subclass.

    Examples:
        ```pycon
        >>> from synth_datasets.export.writers import YoloWriter, register_writer
        >>> class UltralyticsWriter(YoloWriter):
        ...     pass
        >>> register_writer("ultralytics", UltralyticsWriter)

        ```

    """
    if not (isinstance(writer, type) and issubclass(writer, DatasetWriter)):
        raise TypeError(f"writer must be a DatasetWriter subclass, got {writer!r}")
    _WRITERS[fmt.value if isinstance(fmt, OutputFormat) else str(fmt)] = writer


register_writer(OutputFormat.COCO, CocoWriter)
register_writer(OutputFormat.YOLO, YoloWriter)


def get_writer(
    fmt: OutputFormat | str, task: Task, vocabulary: ClassVocabulary, keypoint_schema: KeypointSchema | None = None
) -> DatasetWriter:
    """Return the writer registered for an output format.

    Args:
        fmt: Target format — an :class:`~synth_datasets.core.config.OutputFormat` member, its
            string value, or any key passed to :func:`register_writer`.
        task: Annotation task to emit.
        vocabulary: The classes to declare, in id order; see :class:`DatasetWriter`.
        keypoint_schema: The keypoint family a :attr:`~synth_datasets.core.config.Task.KEYPOINTS`
            run draws from; see :class:`DatasetWriter`. Required for that task, ignored otherwise.

    Returns:
        A concrete :class:`DatasetWriter`.

    Raises:
        ValueError: If no writer is registered for ``fmt``.

    Examples:
        ```pycon
        >>> from synth_datasets.core.config import ClassMode, OutputFormat, Task, class_vocabulary
        >>> from synth_datasets.families.primitives import PrimitiveShape
        >>> from synth_datasets.export.writers import get_writer
        >>> vocab = class_vocabulary(ClassMode.SHAPE, (PrimitiveShape.SQUARE,))
        >>> type(get_writer(OutputFormat.YOLO, Task.DETECTION, vocab)).__name__
        'YoloWriter'

        ```

    """
    key = fmt.value if isinstance(fmt, OutputFormat) else str(fmt)
    writer = _WRITERS.get(key)
    if writer is None:
        raise ValueError(f"no writer registered for format {key!r}; known formats: {sorted(_WRITERS)}")
    return writer(task, vocabulary, keypoint_schema)
