"""Inspection properties and inverse-support classification for fused pipelines.

The mixin reads only state owned by ``FusedCompose`` and lower-level segment and type modules. It imports the direct-
parameter adapter from the factory module, never from :mod:`pipeline`, so the package dependency direction remains
acyclic.

"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from fused_transforms.affine.segment import (
    AlbuFusedAffineSegment,
    AlbuProjectiveSegment,
    CropResizeSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    FusedColorSegment,
    FusedGaussianBlurSegment,
    FusedLUTSegment,
    ProjectiveSegment,
    _FusedGeoCropSegment,
)
from fused_transforms.core.factories import _DirectParamAdapter
from fused_transforms.core.planner import _PassthroughSegment
from fused_transforms.types import SegmentDescriptor, TransformAdapter, is_coordinate_changing_passthrough


class IntrospectionMixin:
    """Expose fusion and inverse metadata for a configured pipeline."""

    _adapter: TransformAdapter | None
    _fusion_plan_cache: tuple[bool, str] | None
    _fusion_plan_descriptors_cache: list[SegmentDescriptor] | None
    _last_transform_matrix: torch.Tensor | None
    _segments: list[object]
    _transform_adapters: dict[int, TransformAdapter]
    original_transforms: list[object]

    def _inverse_unsupported_reason(self) -> str | None:
        """Return the named reason why this pipeline cannot be geometrically inverted."""
        for segment in self._segments:
            if isinstance(segment, (CropResizeSegment, _FusedGeoCropSegment)):
                return "Cannot inverse a crop-resize pipeline: crop-resize discards pixels outside the crop."
            if isinstance(segment, (FusedColorSegment, FusedLUTSegment, FusedGaussianBlurSegment)):
                return "Cannot inverse a pipeline containing a non-geometric color, LUT, or blur segment."
            if isinstance(segment, _PassthroughSegment):
                return "Cannot inverse a pipeline containing a passthrough segment without a recorded matrix."
            if isinstance(segment, ExactAffineSegment):
                return "Cannot inverse exact images through this API; recover coordinates with the recorded matrix."
        if len(self._segments) != 1:
            return "Cannot inverse a multi-segment pipeline: return_matrix records only the last segment matrix."
        if not isinstance(
            self._segments[0],
            (FusedAffineSegment, AlbuFusedAffineSegment, ProjectiveSegment, AlbuProjectiveSegment),
        ):
            return "Cannot inverse a pipeline with no recorded fused geometric matrix."
        return None

    @property
    def transform_matrix(self) -> torch.Tensor | None:
        """Return the last supported geometric segment's forward pixel-centre matrix.

        Affine, projective, exact D4, and direct deterministic letterbox segments
        publish their actual-call ``(batch_size, 3, 3)`` matrix. A matrix maps input
        pixel centres to output pixel centres; bbox helpers convert it to pixel-edge
        geometry. Recover prediction coordinates with its inverse and the appropriate
        target helper, keeping the input and output canvases explicit.

        Multiple segment matrices are not composed into a whole-pipeline transform.
        Colour, passthrough and unmarked random-crop operations do not contribute;
        a matrix alone therefore does not establish complete pipeline invertibility.
        The narrower :meth:`inverse` image contract remains separately checked.

        Returns:
            The last recorded supported matrix, or ``None`` before a call or when
            the latest call produced no supported geometry.

        Note:
            Per-instance mutable state is racy across threads sharing a pipeline.
            Prefer ``return_matrix=True`` for the matrix paired with a specific call.

        """
        return self._last_transform_matrix

    @property
    def resolved_execution(self) -> str | None:
        """Return the warp engine the most recent call actually used.

        Equal to the configured ``execution`` unless that is ``"auto"``, whose routing rule picks
        ``"cv2"`` for host data and ``"torch"`` for accelerator data. ``"auto"`` is the only setting
        under which this can differ from the recorded configuration, which is why it is readable at
        all: a run that cannot say which engine drew its pixels cannot explain its own numerics.

        Returns:
            ``"cv2"``, ``"torch"``, or ``None`` when no fused geometric segment has warped anything yet
            -- before the first call, or after one whose segments were all passthrough.

        Note:
            Per-instance mutable state written on every ``forward``, with the same threading caveat as
            :attr:`transform_matrix`.

        """
        for segment in reversed(self._segments):
            resolved = getattr(segment, "_last_execution", None)
            if resolved is not None:
                return str(resolved)
        return None

    @property
    def n_warps_saved(self) -> int:
        """Return the legacy heuristic for operations combined or routed exactly.

        This compatibility counter includes exact transforms and fused colour,
        lookup, and blur work. It is not a literal count of removed interpolation
        passes and does not predict latency or memory savings. Inspect the actual
        segment plan and profile the intended workload for those decisions.

        Returns:
            Historical aggregate count across planned segments, independent of
            which random operations activate on a particular call.

        """
        total = 0
        for seg in self._segments:
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment, ProjectiveSegment, AlbuProjectiveSegment)):
                # n transforms fused → 1 warp, saving n-1 passes.
                num_transforms = len(seg.transforms)
                if num_transforms > 1:
                    total += num_transforms - 1
                continue

            if isinstance(seg, FusedGaussianBlurSegment):
                total += len(seg.transforms) - (2 if seg.geometric_transforms else 1)
                continue

            if isinstance(seg, ExactAffineSegment):
                # Preserve the historical exact-operation count; exact native
                # operations need not have performed interpolation originally.
                total += len(seg.transforms)
                continue

            if isinstance(seg, FusedColorSegment):
                # n color ops fused → 1 matrix multiply, saving n-1 passes.
                num_transforms = len(seg.transforms)
                if num_transforms > 1:
                    total += num_transforms - 1
                continue

            if isinstance(seg, FusedLUTSegment):
                # n lookup ops composed → 1 table lookup, saving n-1 passes.
                num_transforms = len(seg.transforms)
                if num_transforms > 1:
                    total += num_transforms - 1
        return total

    def _plan_device(self) -> torch.device:
        """Return the pipeline's configured device (follows ``.to(device)``)."""
        tracker = getattr(self, "_device_tracker", None)
        if tracker is not None:
            return cast(Tensor, tracker).device
        return torch.device("cpu")

    @property
    def fusion_plan(self) -> str:
        """Return a human-readable summary of what got fused and what didn't.

        When the pipeline is configured on a non-CPU device (via ``.to(device)``),
        each passthrough segment is annotated with a trailing ``" [CPU passthrough]"``
        marker, because a non-fusible op forces a device-to-host round-trip on a
        GPU/MPS pipeline (the "GPU poison pill"). On a CPU pipeline the marker is
        omitted and the string is unchanged.

        Albumentations fused/projective segments running the ``"cv2"`` execution
        strategy warp each sample with OpenCV, which copies to host and back
        (``.cpu()`` -> ``.to(device)``) on every call. On a non-CPU pipeline they
        therefore carry the same marker, since that hidden round-trip is the biggest
        poison pill of all; switching such a pipeline to ``execution="torch"`` keeps
        the warp on-device and drops the marker.

        Returns:
            Arrow-separated description of segments, e.g.
            ``"fused(RandomRotation, RandomHorizontalFlip) -> passthrough(GaussianBlur)"``.
            On a non-CPU pipeline the passthrough entry reads
            ``"passthrough(GaussianBlur) [CPU passthrough]"``, and an Albumentations
            cv2 fused entry reads ``"fused(Affine) [CPU passthrough]"``. Returns
            ``"empty"`` for an empty pipeline.

        """
        non_cpu = self._plan_device().type != "cpu"
        cached = self._fusion_plan_cache
        if cached is not None and cached[0] == non_cpu:
            return cached[1]
        marker = " [CPU passthrough]" if non_cpu else ""
        # Albu cv2 warps host round-trip per call; on a non-CPU pipeline they carry the
        # same poison-pill marker as a passthrough. torch execution stays on-device.
        albu_cv2_marker = marker if getattr(self, "execution", "cv2") == "cv2" else ""
        parts: list[str] = []
        for seg in self._segments:
            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                names = [type(transform).__name__ for transform in seg.transforms]
                seg_marker = albu_cv2_marker if isinstance(seg, AlbuProjectiveSegment) else ""
                parts.append(f"projective({', '.join(names)}){seg_marker}")
                continue

            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = [type(transform).__name__ for transform in seg.transforms]
                seg_marker = albu_cv2_marker if isinstance(seg, AlbuFusedAffineSegment) else ""
                parts.append(f"fused({', '.join(names)}){seg_marker}")
                continue

            if isinstance(seg, FusedGaussianBlurSegment):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"gaussian_blur({', '.join(names)})")
                continue

            if isinstance(seg, ExactAffineSegment):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"exact({', '.join(names)})")
                continue

            if isinstance(seg, FusedColorSegment):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"color({', '.join(names)})")
                continue

            if isinstance(seg, FusedLUTSegment):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"lut({', '.join(names)})")
                continue

            if isinstance(seg, CropResizeSegment):
                parts.append(f"crop_resize({type(seg.transform).__name__})")
                continue

            if isinstance(seg, _PassthroughSegment):
                parts.append(f"passthrough({type(seg.transform).__name__}){marker}")
                continue

            parts.append(f"passthrough({type(seg).__name__}){marker}")
        plan = " → ".join(parts) if parts else "empty"
        self._fusion_plan_cache = (non_cpu, plan)
        return plan

    @property
    def fusion_plan_descriptors(self) -> list[SegmentDescriptor]:
        """Return a structured, machine-readable description of the fusion plan.

        Each element corresponds to one segment in the pipeline, in execution
        order. This is the structured counterpart to the human-readable
        :attr:`fusion_plan` string. Available immediately after construction —
        does not require a :meth:`forward` call.

        Returns:
            List of :class:`~fused_transforms.types.SegmentDescriptor`
            instances, one per segment. Empty list for an empty pipeline.
            Each descriptor's ``backend`` field is the adapter class name
            (e.g. ``"KorniaAdapter"``) for fused, exact, and projective
            segments, and ``None`` for passthrough segments and backend-free
            pipelines created via :meth:`from_params`.

        Examples:
            ```pycon
            >>> import torch
            >>> from fused_transforms.compose import FusedCompose
            >>> pipe = FusedCompose([])
            >>> pipe.fusion_plan_descriptors
            []

            ```

        Note:
            The ``backend`` field on passthrough segments is always ``None``,
            regardless of the pipeline's backend. Only fused, exact, and
            projective segments carry the adapter class name.

        """
        cached = self._fusion_plan_descriptors_cache
        if cached is not None:
            return cached

        def _resolve_backend(seg: object) -> str | None:
            # from_params() uses _DirectParamAdapter; expose backend-free descriptors.
            if isinstance(self._adapter, _DirectParamAdapter):
                return None

            # Mixed-backend mode: resolve by the first transform in this segment.
            if self._transform_adapters and hasattr(seg, "transforms"):
                seg_transforms = getattr(seg, "transforms", None)
                if isinstance(seg_transforms, list) and seg_transforms:
                    first = seg_transforms[0]
                    for idx, orig in enumerate(self.original_transforms):
                        if orig is first:
                            adapter = self._transform_adapters.get(idx)
                            if adapter is not None:
                                return type(adapter).__name__
                            break

            return type(self._adapter).__name__ if self._adapter else None

        descriptors: list[SegmentDescriptor] = []
        # Resolved backend of the previous fused-family/crop segment, so a fused run
        # split by a backend change (e.g. kornia.RandomAffine -> A.Affine) is flagged
        # even when no passthrough sits between the two segments. Reset to None across a
        # passthrough, whose own descriptor already carries the boundary reason.
        prev_backend: str | None = None
        for seg in self._segments:
            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                backend = _resolve_backend(seg)
                descriptors.append(
                    SegmentDescriptor(
                        kind="projective",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=backend,
                        split_reason=getattr(seg, "_fusion_split_reason", None)
                        or self._backend_boundary_reason(backend, prev_backend),
                    )
                )
                prev_backend = backend
                continue
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                backend = _resolve_backend(seg)
                descriptors.append(
                    SegmentDescriptor(
                        kind="fused",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=backend,
                        split_reason=getattr(seg, "_fusion_split_reason", None)
                        or self._backend_boundary_reason(backend, prev_backend),
                    )
                )
                prev_backend = backend
                continue
            if isinstance(seg, FusedGaussianBlurSegment):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                backend = _resolve_backend(seg)
                descriptors.append(
                    SegmentDescriptor(
                        kind="gaussian_blur",
                        transforms=names,
                        n_warps_saved=len(names) - (2 if seg.geometric_transforms else 1),
                        backend=backend,
                    )
                )
                prev_backend = backend
                continue
            if isinstance(seg, ExactAffineSegment):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names)  # Each flip saves 1 warp vs grid_sample
                backend = _resolve_backend(seg)
                descriptors.append(
                    SegmentDescriptor(
                        kind="exact",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=backend,
                        split_reason=getattr(seg, "_fusion_split_reason", None)
                        or self._backend_boundary_reason(backend, prev_backend),
                    )
                )
                prev_backend = backend
                continue
            if isinstance(seg, FusedColorSegment):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                backend = _resolve_backend(seg)
                descriptors.append(
                    SegmentDescriptor(
                        kind="color",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=backend,
                        split_reason=self._backend_boundary_reason(backend, prev_backend),
                    )
                )
                prev_backend = backend
                continue
            if isinstance(seg, FusedLUTSegment):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                descriptors.append(
                    SegmentDescriptor(
                        kind="lut",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, CropResizeSegment):
                backend = _resolve_backend(seg)
                descriptors.append(
                    SegmentDescriptor(
                        kind="crop_resize",
                        transforms=(type(seg.transform).__name__,),
                        n_warps_saved=0,
                        backend=backend,
                        # A crop+resize acts as a segment boundary (it changes output
                        # dimensions), so it is a barrier for adjacent geometric runs.
                        barrier="crop_resize",
                        split_reason=self._passthrough_split_reason(seg),
                    )
                )
                prev_backend = backend
                continue
            if isinstance(seg, _PassthroughSegment):
                descriptors.append(
                    SegmentDescriptor(
                        kind="passthrough",
                        transforms=(type(seg.transform).__name__,),
                        n_warps_saved=0,
                        barrier=self._passthrough_barrier_reason(seg.transform),
                        split_reason=seg.split_reason or self._passthrough_split_reason(seg),
                        refused="not_fusible",
                    )
                )
                prev_backend = None
                continue
            # Legacy passthrough
            descriptors.append(
                SegmentDescriptor(
                    kind="passthrough",
                    transforms=(type(seg).__name__,),
                    n_warps_saved=0,
                    barrier=self._passthrough_barrier_reason(seg),
                    split_reason=self._passthrough_split_reason(seg),
                    refused="not_fusible",
                )
            )
            prev_backend = None
        self._fusion_plan_descriptors_cache = descriptors
        return descriptors

    def _backend_boundary_reason(self, backend: str | None, prev_backend: str | None) -> str | None:
        """Return ``"backend_boundary"`` when a fused run is split by a backend change.

        In a mixed-backend pipeline, two adjacent fused geometric segments from
        different backends (e.g. ``kornia.RandomAffine`` then ``A.Affine``) cannot be
        composed into one warp; the break is a genuine split, not a passthrough
        barrier. This reports it on the *second* segment of such a pair.

        Args:
            backend: The resolved adapter class name of the current segment.
            prev_backend: The resolved adapter class name of the previous
                fused-family/crop segment, or ``None`` after a passthrough or at the
                start of the pipeline.

        Returns:
            ``"backend_boundary"`` when the pipeline is mixed-backend and both
            backends are known and differ, else ``None``.

        """
        if not self._transform_adapters:
            return None
        if backend is None or prev_backend is None:
            return None
        return "backend_boundary" if backend != prev_backend else None

    def _passthrough_barrier_reason(self, transform: object) -> str:
        """Classify why a passthrough transform forms a fusion barrier.

        Args:
            transform: The passthrough transform (the wrapped transform for a
                :class:`_PassthroughSegment`, or the raw legacy segment object).

        Returns:
            ``"coordinate_change"`` for a geometric distortion op (elastic/grid/optical
            distortion, thin-plate-spline, piecewise-affine) whose auxiliary targets
            would desync, else ``"spatial_kernel"`` for a kernel/pointwise op.

        """
        if is_coordinate_changing_passthrough(transform):
            return "coordinate_change"
        return "spatial_kernel"

    def _passthrough_split_reason(self, seg: object) -> str | None:
        """Return why a run was split at ``seg``, or ``None`` when not applicable.

        A passthrough that sits between two same-backend fused runs is a natural
        barrier (not a *split* of an otherwise-fusible run). A passthrough at a
        backend boundary in a mixed-backend pipeline additionally marks where the
        backend changed. Only the mixed-backend case is reported here, as the single-
        backend barrier reason is already carried by ``barrier``.

        Args:
            seg: The segment being described.

        Returns:
            ``"backend_boundary"`` when the pipeline is mixed-backend and this segment's
            transform sits at a backend change, else ``None``.

        """
        if not self._transform_adapters:
            return None
        transform = seg.transform if isinstance(seg, _PassthroughSegment) else seg
        seg_idx = next(
            (idx for idx, orig in enumerate(self.original_transforms) if orig is transform),
            None,
        )
        if seg_idx is None:
            return None
        this_adapter = self._transform_adapters.get(seg_idx)
        neighbour_adapters = {
            self._transform_adapters.get(idx)
            for idx in (seg_idx - 1, seg_idx + 1)
            if 0 <= idx < len(self.original_transforms)
        }
        neighbour_adapters.discard(None)
        neighbour_adapters.discard(this_adapter)
        return "backend_boundary" if neighbour_adapters else None
