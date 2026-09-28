"""Fuse augmentation transforms into a single interpolation pass.

``import fused_transforms`` is the single public entry point; the whole public
API lives in this package.

Examples:
    ```pycon
    >>> from fused_transforms import Compose
    >>> pipe = Compose([])
    >>> pipe.__class__.__name__
    'FusedCompose'

    ```

"""

from __future__ import annotations

import importlib.util
import os

from fused_transforms.__about__ import *  # noqa: F403

# The augmentation stack is torch-dependent end to end (nn.Module subclasses, Tensor-typed public
# API); importing any of it eagerly imports torch. Dataset generation alone does not need this --
# use ``import synth_datasets`` instead, which never runs this package's __init__. The try/except
# below only turns a confusing ModuleNotFoundError raised deep inside e.g. converters.py into an
# actionable one; it does not make this package importable without torch (see plan/CLAUDE notes:
# that would require lazy __getattr__ loading here, deliberately out of scope for now).
try:
    from fused_transforms.affine.matrix import (
        LetterboxGeometry,
        letterbox_geometry,
        letterbox_matrix,
    )
    from fused_transforms.affine.segment import (
        CropResizeSegment,
        ExactAffineSegment,
        FusedAffineSegment,
        FusedColorSegment,
        FusedLUTSegment,
        ProjectiveSegment,
        build_segments,
    )

    # Import from the implementation module (not the ``compose`` compatibility
    # shim, whose runtime ``__getattr__`` forwarding is invisible to static doc
    # tooling such as griffe/mkdocstrings). ``compose`` stays a valid import and
    # pickle path for historical payloads.
    from fused_transforms.core.pipeline import (
        AugmentationSequential,
        Compose,
        FusedCompose,
    )
    from fused_transforms.detection import augment_detection_batch
    from fused_transforms.dispatch.converters import NumpyToTorchConverter, TorchToNumpyConverter
    from fused_transforms.targets import (
        clip_bbox_xyxy,
        corners_to_rboxes,
        instance_keep_mask,
        mirror_rboxes,
        orientation_reversed,
        permute_keypoint_pairs,
        rbox_envelopes,
        rboxes_to_corners,
        shift_rboxes,
        transform_bbox_xywh,
        transform_bbox_xyxy,
        transform_keypoints,
        transform_mask,
        transform_rboxes,
    )
    from fused_transforms.types import (
        BackendConverter,
        ClipPolicyStr,
        InterpolationMode,
        PaddingMode,
        RandomnessPolicy,
        ReorderPolicy,
        SegmentDescriptor,
        TransformAdapter,
        TransformCategory,
        TransformSpec,
    )
except ModuleNotFoundError as exc:
    # Only a missing top-level ``torch`` means the extra was never installed. A missing torch
    # *submodule* (too old or half-installed torch) must surface verbatim, and so must the
    # ``exc.name == "torch"`` case where torch does import -- there the real fault lies elsewhere.
    if exc.name != "torch" or importlib.util.find_spec("torch") is not None:
        raise
    raise ModuleNotFoundError(
        "fused_transforms requires the 'torch' extra for its augmentation API: "
        'install with `pip install "vision-synth[torch]"`. Dataset generation alone does '
        "not need torch -- use `import synth_datasets` instead.",
        name="torch",
    ) from exc

__all__ = [
    "AugmentationSequential",
    "BackendConverter",
    "ClipPolicyStr",
    "Compose",
    "CropResizeSegment",
    "ExactAffineSegment",
    "FusedAffineSegment",
    "FusedColorSegment",
    "FusedCompose",
    "FusedLUTSegment",
    "InterpolationMode",
    "LetterboxGeometry",
    "NumpyToTorchConverter",
    "PaddingMode",
    "ProjectiveSegment",
    "RandomnessPolicy",
    "ReorderPolicy",
    "SegmentDescriptor",
    "TorchToNumpyConverter",
    "TransformAdapter",
    "TransformCategory",
    "TransformSpec",
    "augment_detection_batch",
    "build_segments",
    "clip_bbox_xyxy",
    "corners_to_rboxes",
    "instance_keep_mask",
    "letterbox_geometry",
    "letterbox_matrix",
    "mirror_rboxes",
    "orientation_reversed",
    "permute_keypoint_pairs",
    "rbox_envelopes",
    "rboxes_to_corners",
    "shift_rboxes",
    "transform_bbox_xywh",
    "transform_bbox_xyxy",
    "transform_keypoints",
    "transform_mask",
    "transform_rboxes",
]


def __dir__() -> list[str]:
    """List the module's attributes."""
    return sorted(set(globals()) | set(__all__))


_PATH_PACKAGE = os.path.realpath(os.path.dirname(__file__))
_PATH_PROJECT = os.path.dirname(_PATH_PACKAGE)
