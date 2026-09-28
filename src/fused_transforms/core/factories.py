"""Factory methods and backend-free direct-parameter construction support.

The mixin deliberately depends only on lower-level configuration, planning, matrix, and type modules. ``FusedCompose``
remains defined in :mod:`pipeline` so its public import and pickle identity stay stable.

"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

from fused_transforms._random import uniform as _uniform
from fused_transforms.affine.matrix import (
    hflip_matrix,
    letterbox_matrix,
    matmul3x3,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)
from fused_transforms.affine.segment import build_segments
from fused_transforms.core.config_validation import (
    _coerce_randomness_policy,
    _has_coord_aux,
    _validate_fill,
    _validate_keypoint_flip_index,
    _validate_mask_fill,
)
from fused_transforms.types import (
    ClipPolicyStr,
    ComposePaddingModeStr,
    FillValue,
    InterpolationStr,
    MaskFillValue,
    MaskInterpolationStr,
    PipelineDtypeStr,
    RandomnessPolicy,
    ReorderPolicy,
    TransformCategory,
    TransformSpec,
)

if TYPE_CHECKING:
    from fused_transforms.dispatch.resolver import BackendStr, OpStr

# The from_config doctest builds a Kornia-backed pipeline, so it is skipped when
# Kornia is not installed (pytest-doctestplus reads this module-level mapping of
# doctest name patterns to their required modules).
__doctest_requires__ = {"FactoriesMixin.from_config": ["kornia"]}


class FactoriesMixin:
    """Construct pipelines from declarative specs or direct parameter ranges."""

    @classmethod
    def from_config(
        cls,
        specs: list[TransformSpec],
        backend: BackendStr,
        interpolation: InterpolationStr = "bilinear",
        padding_mode: ComposePaddingModeStr = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy | Literal["backend", "per_sample"] = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        on_unsupported: Literal["raise", "warn_skip"] = "raise",
        mask_interpolation: MaskInterpolationStr = "nearest",
        pipeline_dtype: PipelineDtypeStr | None = None,
        generator: torch.Generator | None = None,
        fill: FillValue | None = None,
        keypoint_flip_index: Sequence[int] | None = None,
        mask_fill: MaskFillValue = 0,
    ) -> object:
        """Create a FusedCompose pipeline from a list of TransformSpec objects.

        Resolves each spec's operation to the corresponding backend transform
        class, instantiates it with the spec's params and per-sample probability,
        then builds the pipeline via ``cls(transforms, ...)``.

        All specs are validated against the backend's capability matrix **before** any transform is constructed, so an
        unsupported op is reported together with every other offender in a single aggregated error rather than failing
        on the first one.

        Args:
            specs: List of :class:`TransformSpec` objects describing the
                pipeline.
            backend: Backend name (``"kornia"``, ``"torchvision"``,
                ``"albumentations"``, or ``"native"``). The native backend
                is fully batched and has no optional dependencies.
            interpolation: Interpolation mode for ``grid_sample`` warp.
            padding_mode: Padding mode for out-of-bounds samples.
            reorder: Reorder policy applied before segmentation.
            data_keys: Key list for auxiliary target routing.
            output_backend: Target output format (``"numpy"``,
                ``"numpy_hwc"``, ``"torch"``, or ``None``). Forwarded to
                :meth:`__init__`.
            randomness: Batch randomness policy forwarded to :meth:`__init__`.
            clip_policy: Clamp policy for fused color segments.
            mask_interpolation: Auxiliary mask sampling mode forwarded to
                :meth:`__init__`.
            mask_fill: Scalar border value for routed auxiliary masks, forwarded to
                :meth:`__init__`.
            pipeline_dtype: Optional fused GPU image-operation dtype forwarded to
                :meth:`__init__`.
            generator: Caller-owned ``torch.Generator`` driving every sampled
                parameter. Supported for ``backend="native"`` only — the other
                backends sample through their own libraries, which cannot be
                seeded from a ``torch.Generator`` (see :meth:`from_params`).
            fill: Constant out-of-canvas image border in the image's own value range,
                forwarded to :meth:`__init__`; requires ``padding_mode="zeros"``.
            keypoint_flip_index: Keypoint pair permutation forwarded to :meth:`__init__`,
                applied whenever the composed transform reverses orientation.
            on_unsupported: Policy for specs whose op the backend cannot build.
                ``"raise"`` (default) aggregates all offenders into one
                ``ValueError``; ``"warn_skip"`` drops each unsupported spec with
                a ``UserWarning`` and builds a pipeline from the rest.

        Returns:
            A configured ``FusedCompose`` instance.

        Raises:
            ValueError: If ``backend`` is not in
                :data:`~fused_transforms.dispatch.resolver.SUPPORTED_BACKENDS`; or, when
                ``on_unsupported="raise"``, if any spec's operation is unknown or unsupported by the chosen backend
                (all offenders reported together). This validation applies even when ``specs`` is empty.

        Examples:
            ```pycon
            >>> import torch
            >>> from fused_transforms.compose import FusedCompose
            >>> from fused_transforms.types import TransformSpec
            >>> spec = TransformSpec(operation="hflip", params={}, prob=0.5)
            >>> pipe = FusedCompose.from_config([spec], backend="kornia")
            >>> pipe(torch.zeros(1, 3, 8, 8)).shape
            torch.Size([1, 3, 8, 8])

            ```

        """
        from fused_transforms.dispatch.resolver import SUPPORTED_BACKENDS

        if backend not in SUPPORTED_BACKENDS:
            msg = f"unknown backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
            raise ValueError(msg)
        mask_fill = _validate_mask_fill(mask_fill)

        kept_specs = cls._validate_specs(specs, backend, on_unsupported)
        if backend == "native":
            return cls._from_param_specs(
                specs=kept_specs,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
                randomness=_coerce_randomness_policy(randomness),
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
                mask_fill=mask_fill,
                pipeline_dtype=pipeline_dtype,
                route_coords_via_grid=_has_coord_aux(data_keys),
                native=True,
                generator=generator,
                fill=fill,
                keypoint_flip_index=keypoint_flip_index,
            )
        transforms = [cls._build_transform(spec, backend) for spec in kept_specs]

        constructor = cast(Callable[..., object], cls)
        return constructor(
            transforms,
            generator=generator,
            fill=fill,
            keypoint_flip_index=keypoint_flip_index,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            reorder=reorder,
            output_backend=output_backend,
            randomness=randomness,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            pipeline_dtype=pipeline_dtype,
        )

    @staticmethod
    def _validate_specs(
        specs: list[TransformSpec],
        backend: BackendStr,
        on_unsupported: Literal["raise", "warn_skip"],
    ) -> list[TransformSpec]:
        """Validate every spec via ``resolve_op`` before any construction, aggregating all failures.

        Rejects ``prob`` inside ``params`` (always an error), then probes each op with ``resolve_op`` (the same gate
        construction uses). Under ``"raise"`` every offending spec's message is collected and reported in one
        aggregated ``ValueError``; under ``"warn_skip"`` the offenders are dropped with a ``UserWarning`` and the
        surviving specs are returned.

        Args:
            specs: The specs to validate.
            backend: Target backend name.
            on_unsupported: ``"raise"`` or ``"warn_skip"``.

        Returns:
            The specs to build (all of *specs* under ``"raise"``; only supported ones under ``"warn_skip"``).

        Raises:
            ValueError: On a ``prob``-in-``params`` spec, or (under ``"raise"``) when any op is unsupported.

        """
        from fused_transforms.dispatch.resolver import SUPPORTED_OPS, resolve_op

        offenders: list[str] = []
        kept: list[TransformSpec] = []
        for spec in specs:
            if "prob" in spec.params:
                msg = (
                    f"TransformSpec.params must not include 'prob'; "
                    f"use TransformSpec.prob for operation {spec.operation!r} instead."
                )
                raise ValueError(msg)
            if spec.operation not in SUPPORTED_OPS:
                offenders.append(f"unknown operation {spec.operation!r}; supported: {sorted(SUPPORTED_OPS)}")
                continue
            try:
                resolve_op(cast("OpStr", spec.operation), backend)
            except ValueError as exc:
                offenders.append(str(exc))
            else:
                kept.append(spec)

        if not offenders:
            return kept
        if on_unsupported == "warn_skip":
            warnings.warn(
                "from_config skipping unsupported specs:\n  - " + "\n  - ".join(offenders),
                UserWarning,
                stacklevel=3,
            )
            return kept
        raise ValueError("from_config could not resolve all specs:\n  - " + "\n  - ".join(offenders))

    @staticmethod
    def _build_transform(spec: TransformSpec, backend: BackendStr) -> object:
        """Instantiate one backend transform from a validated spec.

        Applies the per-transform probability via the backend ``p=`` kwarg when accepted, falling back to a ``prob``
        attribute (or a warning) for backends that reject it.

        Args:
            spec: A spec already validated as supported by *backend*.
            backend: Target backend name.

        Returns:
            The constructed backend transform object.

        """
        from fused_transforms.dispatch.resolver import resolve_op, translate_params

        op_name = cast("OpStr", spec.operation)
        tfm_cls = cast(type, resolve_op(op_name, backend))
        kwargs = translate_params(op_name, backend, dict(spec.params))
        # Most backends accept p= for per-transform probability.
        # Some (e.g. TorchVision rotation) don't; pass it and let
        # the backend ignore it via **kwargs or TypeError fallback.
        try:
            return tfm_cls(**kwargs, p=spec.prob)
        except TypeError as exc:
            # Re-raise unless this is specifically a rejected p keyword.
            # Constructor errors about other kwargs/types must not be masked.
            exc_msg = str(exc).lower()
            is_p_keyword_rejection = "'p'" in exc_msg and (
                "unexpected keyword" in exc_msg or "keyword argument" in exc_msg or "does not accept" in exc_msg
            )
            if not is_p_keyword_rejection:
                raise
        # Backend class does not accept p= (e.g. TorchVision RandomRotation)
        tfm = tfm_cls(**kwargs)
        # Attach prob as attribute so the fused engine can read it
        p_set = False
        with contextlib.suppress(AttributeError, TypeError):
            tfm.prob = spec.prob
            p_set = True
        if not p_set and spec.prob != 1.0:
            warnings.warn(
                f"TransformSpec.prob={spec.prob!r} for operation {spec.operation!r} could not be applied to "
                f"{tfm_cls.__name__!r} (backend does not accept p= and attribute is read-only). "
                "The transform will always be applied (prob=1.0 effective).",
                UserWarning,
                stacklevel=3,
            )
        return tfm

    @classmethod
    def supported_ops(cls, backend: BackendStr) -> frozenset[str]:
        """Return the canonical op names *backend* can build.

        Args:
            backend: Backend name (must be in :data:`~fused_transforms.dispatch.resolver.SUPPORTED_BACKENDS`).

        Returns:
            Frozenset of canonical op names the backend supports (empty if the backend's optional dependency is not
            installed).

        Raises:
            KeyError: If *backend* is not in :data:`~fused_transforms.dispatch.resolver.SUPPORTED_BACKENDS`.

        Examples:
            ```pycon
            >>> from fused_transforms.compose import FusedCompose
            >>> "hflip" in FusedCompose.supported_ops("kornia")  # doctest: +SKIP
            True

            ```

        """
        from fused_transforms.dispatch.resolver import capability_matrix

        return capability_matrix()[backend]

    @classmethod
    def capability_matrix(cls) -> dict[str, frozenset[str]]:
        """Return the backend -> supported-op-names map across all supported backends.

        Returns:
            Mapping from each supported backend to the frozenset of canonical op names it can build.

        Examples:
            ```pycon
            >>> from fused_transforms.compose import FusedCompose
            >>> sorted(FusedCompose.capability_matrix())
            ['albumentations', 'kornia', 'native', 'torchvision']

            ```

        """
        from fused_transforms.dispatch.resolver import capability_matrix

        return capability_matrix()

    @classmethod
    def from_params(
        cls,
        rotation: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        scale_x: tuple[float, float] | None = None,
        scale_y: tuple[float, float] | None = None,
        shear_x: tuple[float, float] | None = None,
        shear_y: tuple[float, float] | None = None,
        translate_x: tuple[float, float] | None = None,
        translate_y: tuple[float, float] | None = None,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        brightness: float | tuple[float, float] | None = None,
        contrast: float | tuple[float, float] | None = None,
        interpolation: InterpolationStr = "bilinear",
        padding_mode: ComposePaddingModeStr = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy | Literal["backend", "per_sample"] = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        pipeline_dtype: PipelineDtypeStr | None = None,
        generator: torch.Generator | None = None,
        fill: FillValue | None = None,
        keypoint_flip_index: Sequence[int] | None = None,
        letterbox: tuple[int, int] | int | None = None,
        allow_upscale: bool = True,
        *,
        rotation_p: float = 1.0,
        scale_p: float = 1.0,
        specs: list[TransformSpec] | None = None,
        backend: BackendStr | None = None,
        route_coords_via_grid: bool = False,
        mask_fill: MaskFillValue = 0,
    ) -> object:
        """Create a ``FusedCompose`` pipeline directly from parameter ranges.

        This factory bypasses backend transform objects entirely and samples
        parameters directly using ``_matrix.py`` primitives. Useful for
        backend-agnostic pipelines or when no backend is installed.

        When ``backend`` is provided, the factory delegates to
        :meth:`from_config` semantics instead, using real backend transform
        objects. This allows ``from_params`` to serve as a single entry
        point that works both with and without a backend.

        All geometric parameters are sampled independently per batch item on
        every :meth:`forward` call (i.e. ``same_on_batch=False`` semantics).
        If all geometric params are ``None`` and both flip probabilities are
        0.0, the returned pipeline is an identity passthrough.

        Args:
            rotation: ``(min_deg, max_deg)`` rotation range, or ``None`` to
                disable rotation.
            scale: ``(min_factor, max_factor)`` uniform scale range applied to
                both axes equally, or ``None``. Overridden per-axis by
                ``scale_x``/``scale_y`` when those are also set.
            scale_x: ``(min_factor, max_factor)`` x-axis-only scale range, or
                ``None``. Per-axis only in backend-free mode -- see note below.
            scale_y: ``(min_factor, max_factor)`` y-axis-only scale range, or
                ``None``. Per-axis only in backend-free mode -- see note below.
            shear_x: ``(min_deg, max_deg)`` x-shear range, or ``None``.
                Per-axis only in backend-free mode -- see note below.
            shear_y: ``(min_deg, max_deg)`` y-shear range, or ``None``.
                Per-axis only in backend-free mode -- see note below.
            translate_x: ``(min_px, max_px)`` x-translation range in pixels,
                or ``None``. Per-axis only in backend-free mode -- see note below.
            translate_y: ``(min_px, max_px)`` y-translation range in pixels,
                or ``None``. Per-axis only in backend-free mode -- see note below.
            hflip_p: Probability of horizontal flip per sample. Default 0.0.
            vflip_p: Probability of vertical flip per sample. Default 0.0.
            rotation_p: Probability of applying ``rotation`` per sample, keyword-only.
                Default ``1.0``, which applies it to every sample. Ignored when
                ``rotation`` is ``None``.
            scale_p: Probability of applying the scale family (``scale``,
                ``scale_x``, ``scale_y``) per sample, keyword-only. The three share
                one probability because they describe one scaling of the same
                sample; use ``specs=`` or :meth:`from_config` for finer control.
                Default ``1.0``. Ignored when no scale range is set.
            brightness: Maximum brightness deviation, or a ``(low, high)`` factor range. A value
                of ``0.1`` samples factors in ``[0.9, 1.1]``.
            contrast: Maximum contrast deviation, or a ``(low, high)`` factor range. A value of
                ``0.1`` samples factors in ``[0.9, 1.1]`` around midpoint 0.5.
            interpolation: Interpolation mode for the ``grid_sample`` warp.
                One of ``"bilinear"`` (default), ``"nearest"``, ``"bicubic"``.
            padding_mode: Padding for out-of-bounds samples.
                One of ``"zeros"`` (default), ``"border"``, ``"reflection"``.
            reorder: Reorder policy applied before segmentation.
                Defaults to ``ReorderPolicy.POINTWISE``.
            data_keys: Key list for auxiliary target routing, forwarded to
                :meth:`__init__`. ``None`` preserves single-tensor I/O.
            output_backend: Target output format (``"numpy"``,
                ``"numpy_hwc"``, ``"torch"``, or ``None``). Forwarded to
                :meth:`__init__`.
            randomness: Batch randomness policy forwarded to :meth:`__init__`
                or :meth:`from_config`. NOTE: in backend-free mode
                (``backend=None``) this value is stored but has no effect on
                parameter sampling — the direct-param path always draws
                independent per-sample parameters. It only changes behaviour
                when ``backend=`` is set (delegation to :meth:`from_config`).
            clip_policy: Clamp policy forwarded to fused color segments.
            mask_interpolation: Auxiliary mask sampling mode forwarded to
                :meth:`__init__` or :meth:`from_config`.
            mask_fill: Scalar border value for routed auxiliary masks, forwarded to
                :meth:`__init__` or :meth:`from_config`.
            pipeline_dtype: Optional fused GPU image-operation dtype forwarded to
                :meth:`__init__` or :meth:`from_config`.
            generator: Caller-owned ``torch.Generator`` driving every sampled
                parameter and every per-transform probability gate. ``None``
                (default) keeps the global torch stream, so existing pipelines
                are unchanged. Supported in backend-free mode and with
                ``backend="native"``; every other backend samples through its own
                library and rejects a generator at construction rather than
                silently falling back to the global stream.
            fill: Constant value written outside the source canvas of every warp, in
                the image's own value range (``114`` for a uint8 grey, ``114 / 255`` as
                float); a scalar fills every channel, a sequence one channel each.
                ``None`` (default) keeps the plain zero border. Images only — auxiliary
                masks keep zero padding. Requires ``padding_mode="zeros"``.
            keypoint_flip_index: Keypoint pair permutation applied whenever the composed
                transform reverses orientation, forwarded to :meth:`__init__`; slot ``i``
                takes its value from slot ``flip_index[i]``. The pair table is dataset
                schema and stays with the caller.
            letterbox: Output canvas as ``(height, width)``, or one ``int`` for a square.
                Appends a deterministic aspect-preserving fit: content is scaled by
                ``r = min(height_out / H, width_out / W)`` and the slack is padded with
                ``fill``. It sits after the geometry and before any brightness/contrast, so
                the affine and the letterbox compose into a single matrix and one resample
                — which also means a colour op in the same call acts on the padded canvas,
                pad region included. Backend-free mode only. ``None`` (default) adds nothing.
            allow_upscale: Whether ``letterbox`` may enlarge content past its native size
                when the output canvas is larger than the source. ``False`` caps the ratio
                at ``1.0``, padding instead of magnifying. Ignored without ``letterbox``.
            route_coords_via_grid: Force coordinate auxiliary targets through
                the grid path for direct-param construction.
            specs: List of :class:`TransformSpec` objects. When provided,
                all other geometric keyword arguments must be at their
                defaults (mutually exclusive). Keyword-only.
            backend: Backend name (``"kornia"``, ``"torchvision"``,
                ``"albumentations"``, ``"native"``), or ``None`` for backend-free
                mode. When set, delegates to :meth:`from_config` semantics.
                Keyword-only.

        Note:
                    **Per-axis geometry requires backend-free mode.** The per-axis
                    kwargs ``scale_x``/``scale_y``, ``shear_x``/``shear_y`` and
                    ``translate_x``/``translate_y`` are only accepted when
                    ``backend=None`` (the direct-parameter engine samples them per
                    axis). They are rejected with a ``ValueError`` when *any*
                    ``backend`` is set -- including ``backend="native"``, even though
                    the native engine is the same direct-parameter engine as
                    ``backend=None`` -- because the ``backend=`` path routes through
                    :meth:`from_config`, whose canonical op vocabulary exposes only
                    the symmetric ``"shear"``/``"translate"`` ops, not the per-axis
                    variants. Use ``backend=None`` for per-axis geometry, or
                    :meth:`from_config` with an explicit affine spec.

        Returns:
            A configured ``FusedCompose`` instance ready for inference or training.

        Raises:
            ValueError: If ``specs`` is provided together with any geometric keyword
                argument (they are mutually exclusive).
            ValueError: If ``specs`` contains an op that is not supported in
                backend-free mode (i.e. not one of ``"rotation"``, ``"scale"``,
                ``"scale_x"``, ``"scale_y"``, ``"shear_x"``, ``"shear_y"``,
                ``"translate_x"``, ``"translate_y"``, ``"hflip"``, ``"vflip"``).
            ValueError: If per-axis ``scale_x``/``scale_y``, ``shear_x``/``shear_y``
                or ``translate_x``/``translate_y`` are passed together with any
                ``backend`` (use ``backend=None`` for per-axis geometry).
            ValueError: If a backend-native kwarg in ``TransformSpec.params``
                is not accepted by the backend constructor.

        Examples:
            ```pycon
            >>> import torch
            >>> from fused_transforms.compose import FusedCompose
            >>> pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
            >>> x = torch.zeros(2, 3, 64, 64)
            >>> out = pipe(x)
            >>> out.shape
            torch.Size([2, 3, 64, 64])

            ```

            Caller-owned randomness: two identically seeded generators give
            identical output, independent of the global torch stream.

            ```pycon
            >>> import torch
            >>> from fused_transforms.compose import FusedCompose
            >>> image = torch.rand(2, 3, 32, 32)
            >>> def run(seed):
            ...     gen = torch.Generator().manual_seed(seed)
            ...     return FusedCompose.from_params(rotation=(-30, 30), generator=gen)(image)
            >>> bool(torch.equal(run(0), run(0)))
            True
            >>> bool(torch.equal(run(0), run(1)))
            False

            ```

        """
        _validate_geometric_probability("rotation_p", rotation_p)
        _validate_geometric_probability("scale_p", scale_p)
        mask_fill = _validate_mask_fill(mask_fill)

        # --- specs= overload path ---
        # Note: specs= is a convenience alias for declarative pipeline construction.
        # In a future minor version this dual-path may be split into a from_specs()
        # classmethod; for now mutual exclusivity of specs and geometric kwargs enforces intent.
        if specs is not None:
            # Validate mutual exclusivity
            # Geometric tuple params default to None; use explicit is-not-None
            # checks so invalid falsey values (e.g. empty tuple/list) are still
            # treated as "provided" and rejected in specs mode.
            has_keyword_params = any((
                letterbox is not None,
                rotation is not None,
                scale is not None,
                scale_x is not None,
                scale_y is not None,
                shear_x is not None,
                shear_y is not None,
                translate_x is not None,
                translate_y is not None,
                hflip_p != 0.0,
                vflip_p != 0.0,
                rotation_p != 1.0,
                scale_p != 1.0,
                brightness is not None,
                contrast is not None,
            ))
            if has_keyword_params:
                msg = "specs and keyword params are mutually exclusive"
                raise ValueError(msg)

            if backend is not None:
                return cls.from_config(
                    specs=specs,
                    backend=backend,
                    generator=generator,
                    fill=fill,
                    keypoint_flip_index=keypoint_flip_index,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    reorder=reorder,
                    data_keys=data_keys,
                    output_backend=output_backend,
                    randomness=randomness,
                    clip_policy=clip_policy,
                    mask_interpolation=mask_interpolation,
                    mask_fill=mask_fill,
                    pipeline_dtype=pipeline_dtype,
                )

            return cls._from_param_specs(
                specs=specs,
                generator=generator,
                fill=fill,
                keypoint_flip_index=keypoint_flip_index,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
                randomness=_coerce_randomness_policy(randomness),
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
                mask_fill=mask_fill,
                pipeline_dtype=pipeline_dtype,
                route_coords_via_grid=route_coords_via_grid,
            )

        # When backend is set and geometric kwargs are provided (no specs),
        # convert the kwargs to TransformSpec objects and delegate to from_config.
        if backend is not None:
            if letterbox is not None:
                raise NotImplementedError(
                    "letterbox= is only available in backend-free mode (backend=None); the backend= path routes "
                    "through from_config, whose canonical op vocabulary has no letterbox op."
                )
            if backend != "native" and (brightness is not None or contrast is not None):
                raise NotImplementedError("brightness and contrast from_params require backend='native'")
            config_specs = cls._geometric_kwargs_to_specs(
                rotation=rotation,
                scale=scale,
                scale_x=scale_x,
                scale_y=scale_y,
                shear_x=shear_x,
                shear_y=shear_y,
                translate_x=translate_x,
                translate_y=translate_y,
                hflip_p=hflip_p,
                vflip_p=vflip_p,
                rotation_p=rotation_p,
                scale_p=scale_p,
            )
            if backend == "native":
                config_specs.extend(
                    TransformSpec(operation=name, params={"factor": value})
                    for name, value in cls._native_color_specs(brightness, contrast)
                )
            return cls.from_config(
                specs=config_specs,
                backend=backend,
                generator=generator,
                fill=fill,
                keypoint_flip_index=keypoint_flip_index,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
                randomness=randomness,
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
                mask_fill=mask_fill,
                pipeline_dtype=pipeline_dtype,
            )

        # Collect geometric param specs
        param_specs: dict[str, tuple[float, float]] = {}
        if rotation is not None:
            param_specs["rotation"] = rotation
        if scale is not None:
            param_specs["scale"] = scale
        if scale_x is not None:
            param_specs["scale_x"] = scale_x
        if scale_y is not None:
            param_specs["scale_y"] = scale_y
        if shear_x is not None:
            param_specs["shear_x"] = shear_x
        if shear_y is not None:
            param_specs["shear_y"] = shear_y
        if translate_x is not None:
            param_specs["translate_x"] = translate_x
        if translate_y is not None:
            param_specs["translate_y"] = translate_y

        color_specs = cls._native_color_specs(brightness, contrast)
        has_affine = bool(param_specs)
        has_flips = hflip_p > 0.0 or vflip_p > 0.0
        has_color = bool(color_specs)
        has_letterbox = letterbox is not None

        # NOTE: The identity path (all params None) returns a normal __init__ instance
        # with _adapter=None. The non-identity path uses _DirectParamAdapter.
        # Both handle empty _segments correctly; do not branch on isinstance(_adapter, ...).
        if not has_affine and not has_flips and not has_color and not has_letterbox:
            constructor = cast(Callable[..., object], cls)
            return constructor(
                transforms=[],
                generator=generator,
                fill=fill,
                keypoint_flip_index=keypoint_flip_index,
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                output_backend=output_backend,
                reorder=reorder,
                randomness=randomness,
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
                mask_fill=mask_fill,
                pipeline_dtype=pipeline_dtype,
            )

        # Build internal transforms and adapter
        adapter = _DirectParamAdapter()
        transforms: list[object] = []

        if has_affine:
            transforms.extend(_geometric_param_transforms(param_specs, rotation_p, scale_p))

        if hflip_p > 0.0:
            transforms.append(_DirectFlipTransform(flip_type="hflip", prob=hflip_p))

        if vflip_p > 0.0:
            transforms.append(_DirectFlipTransform(flip_type="vflip", prob=vflip_p))

        if letterbox is not None:
            height_out, width_out = (letterbox, letterbox) if isinstance(letterbox, int) else letterbox
            # Placed after the geometry and before any colour op: the crop-resize category is
            # a reorder barrier, so a colour op buffered ahead of it would be flushed between
            # the geometric run and the letterbox and break the single-segment fusion.
            transforms.append(_DirectLetterboxTransform(height_out, width_out, allow_upscale))

        transforms.extend(_DirectParamTransform(param_specs={key: value}, prob=1.0) for key, value in color_specs)

        # Build instance bypassing detect_backend
        fill_values = _validate_fill(fill, padding_mode)
        flip_index = _validate_keypoint_flip_index(keypoint_flip_index)
        instance = cls.__new__(cls)
        nn.Module.__init__(cast(nn.Module, instance))

        randomness_policy = _coerce_randomness_policy(randomness)
        segments = build_segments(
            transforms,
            adapter,
            interpolation,
            None if padding_mode == "per_transform" else padding_mode,
            randomness_policy,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            route_coords_via_grid=route_coords_via_grid or _has_coord_aux(data_keys),
            per_transform_padding=padding_mode == "per_transform",
            generator=generator,
            fill=fill_values,
            keypoint_flip_index=flip_index,
        )
        cast(Any, instance)._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
            output_backend=output_backend,
            randomness=randomness_policy,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            pipeline_dtype=pipeline_dtype,
            generator=generator,
            fill=fill_values,
            keypoint_flip_index=flip_index,
        )

        return instance

    @classmethod
    def _from_param_specs(
        cls,
        specs: list[TransformSpec],
        interpolation: InterpolationStr,
        padding_mode: ComposePaddingModeStr,
        reorder: ReorderPolicy,
        data_keys: list[str] | None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        pipeline_dtype: PipelineDtypeStr | None = None,
        route_coords_via_grid: bool = False,
        native: bool = False,
        generator: torch.Generator | None = None,
        fill: FillValue | None = None,
        keypoint_flip_index: Sequence[int] | None = None,
        mask_fill: MaskFillValue = 0,
    ) -> object:
        """Build a from_params pipeline from a list of TransformSpec objects.

        Each spec is converted to the appropriate internal direct-param transform (``_DirectParamTransform`` or
        ``_DirectFlipTransform``).

        """
        op_to_param_key: dict[str, str] = {
            "rotation": "rotation",
            "scale": "scale",
            "scale_x": "scale_x",
            "scale_y": "scale_y",
            "shear_x": "shear_x",
            "shear_y": "shear_y",
            "translate_x": "translate_x",
            "translate_y": "translate_y",
            "brightness": "brightness",
            "contrast": "contrast",
        }

        adapter = _DirectParamAdapter()
        transforms: list[object] = []

        for spec in specs:
            if spec.operation in ("hflip", "vflip"):
                transforms.append(
                    _DirectFlipTransform(flip_type=spec.operation, prob=spec.prob),  # type: ignore[arg-type]
                )
            elif spec.operation in op_to_param_key:
                param_key = op_to_param_key[spec.operation]
                # Resolve and validate one numeric range for this op.
                param_value = cls._extract_param_range_from_spec(spec)
                param_specs = {param_key: param_value}
                transforms.append(
                    _DirectParamTransform(param_specs=param_specs, prob=spec.prob),
                )
            elif native and spec.operation == "shear":
                transforms.append(
                    _DirectParamTransform(
                        param_specs={"shear_x": cls._extract_param_range_from_spec(spec)},
                        prob=spec.prob,
                    )
                )
            elif native and spec.operation == "translate":
                param_value = cls._extract_param_range_from_spec(spec)
                transforms.append(
                    _DirectParamTransform(
                        param_specs={"translate_x": param_value, "translate_y": param_value},
                        prob=spec.prob,
                    )
                )
            else:
                msg = f"Unsupported op for from_params: {spec.operation!r}"
                raise ValueError(msg)

        if not transforms:
            constructor = cast(Callable[..., object], cls)
            return constructor(
                transforms=[],
                generator=generator,
                fill=fill,
                keypoint_flip_index=keypoint_flip_index,
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                output_backend=output_backend,
                reorder=reorder,
                randomness=randomness,
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
                mask_fill=mask_fill,
                pipeline_dtype=pipeline_dtype,
            )

        fill_values = _validate_fill(fill, padding_mode)
        flip_index = _validate_keypoint_flip_index(keypoint_flip_index)
        instance = cls.__new__(cls)
        nn.Module.__init__(cast(nn.Module, instance))

        segments = build_segments(
            transforms,
            adapter,
            interpolation,
            None if padding_mode == "per_transform" else padding_mode,
            randomness,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            route_coords_via_grid=route_coords_via_grid or _has_coord_aux(data_keys),
            per_transform_padding=padding_mode == "per_transform",
            generator=generator,
            fill=fill_values,
            keypoint_flip_index=flip_index,
        )
        cast(Any, instance)._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
            output_backend=output_backend,
            randomness=randomness,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            mask_fill=mask_fill,
            pipeline_dtype=pipeline_dtype,
            generator=generator,
            fill=fill_values,
            keypoint_flip_index=flip_index,
        )

        return instance

    @staticmethod
    def _native_color_specs(
        brightness: float | tuple[float, float] | None,
        contrast: float | tuple[float, float] | None,
    ) -> list[tuple[str, tuple[float, float]]]:
        """Convert brightness/contrast settings into native factor ranges.

        A number ``d`` is a maximum deviation, sampling factors in ``[1 - d, 1 + d]``; a ``(low, high)`` pair is the
        factor range itself (TorchVision's ``ColorJitter`` convention), so ``0.2`` and ``(0.8, 1.2)`` agree.

        """
        specs: list[tuple[str, tuple[float, float]]] = []
        for name, setting in (("brightness", brightness), ("contrast", contrast)):
            if setting is None:
                continue
            if isinstance(setting, (int, float)):
                if setting < 0.0:
                    raise ValueError(f"{name} must be non-negative, got {setting!r}")
                specs.append((name, (1.0 - setting, 1.0 + setting)))
                continue
            if len(setting) != 2:
                raise ValueError(f"{name} range must be a (low, high) pair, got {setting!r}")
            low, high = float(setting[0]), float(setting[1])
            if not 0.0 <= low <= high:
                raise ValueError(f"{name} range must satisfy 0 <= low <= high, got {setting!r}")
            specs.append((name, (low, high)))
        return specs

    @staticmethod
    def _extract_param_range_from_spec(spec: TransformSpec) -> tuple[float, float]:
        """Extract one numeric range tuple from a TransformSpec params dict."""
        op_to_allowed_keys: dict[str, tuple[str, ...]] = {
            "rotation": ("degrees", "rotation"),
            "scale": ("factor", "scale"),
            "scale_x": ("factor", "scale_x"),
            "scale_y": ("factor", "scale_y"),
            "shear_x": ("degrees", "shear_x"),
            "shear_y": ("degrees", "shear_y"),
            "shear": ("degrees", "shear"),
            "translate_x": ("pixels", "translate_x"),
            "translate_y": ("pixels", "translate_y"),
            "translate": ("pixels", "translate"),
            "brightness": ("factor", "brightness"),
            "contrast": ("factor", "contrast"),
        }

        allowed_keys = op_to_allowed_keys.get(spec.operation, ())
        for key in allowed_keys:
            if key in spec.params:
                value = spec.params[key]
                if (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and isinstance(value[0], (int, float))
                    and isinstance(value[1], (int, float))
                ):
                    return float(value[0]), float(value[1])
                msg = f"Invalid range for {spec.operation!r}: expected tuple[float, float] in params[{key!r}]"
                raise ValueError(msg)

        msg = f"Missing required range for {spec.operation!r}; expected one of keys: {allowed_keys}"
        raise ValueError(msg)

    @staticmethod
    def _geometric_kwargs_to_specs(
        rotation: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        scale_x: tuple[float, float] | None = None,
        scale_y: tuple[float, float] | None = None,
        shear_x: tuple[float, float] | None = None,
        shear_y: tuple[float, float] | None = None,
        translate_x: tuple[float, float] | None = None,
        translate_y: tuple[float, float] | None = None,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        rotation_p: float = 1.0,
        scale_p: float = 1.0,
    ) -> list[TransformSpec]:
        """Convert geometric keyword arguments to a list of TransformSpec objects.

        Used internally by :meth:`from_params` when ``backend`` is set and geometric kwargs (rather than ``specs``) are
        provided. ``rotation_p`` and ``scale_p`` become the corresponding spec's ``prob``, the same field the flip
        probabilities already use.

        """
        if scale_x is not None or scale_y is not None:
            msg = (
                "scale_x and scale_y are not supported when backend= is set (including backend='native'). "
                "Use backend=None for the backend-free direct-parameter engine, which samples per-axis "
                "scale directly; or from_config() with an explicit RandomAffine spec for anisotropic scale."
            )
            raise ValueError(msg)

        if any(param is not None for param in (shear_x, shear_y, translate_x, translate_y)):
            msg = (
                "shear_x/shear_y and translate_x/translate_y are not supported when backend= is set "
                "(including backend='native'). Use backend=None for the backend-free direct-parameter "
                "engine, which samples per-axis shear/translation directly; or from_config() with an "
                "explicit affine spec for per-axis shear/translation."
            )
            raise ValueError(msg)

        specs: list[TransformSpec] = []

        # Map geometric tuple params to their canonical op, param key, and per-op probability
        _kwarg_to_op: dict[str, tuple[str, str, float]] = {
            "rotation": ("rotation", "degrees", rotation_p),
            "scale": ("scale", "factor", scale_p),
        }

        # Geometric tuple params
        for kwarg_name, value in [
            ("rotation", rotation),
            ("scale", scale),
        ]:
            if value is not None:
                op_name, param_key, probability = _kwarg_to_op[kwarg_name]
                specs.append(TransformSpec(operation=op_name, params={param_key: value}, prob=probability))

        # Flip params
        if hflip_p > 0.0:
            specs.append(TransformSpec(operation="hflip", params={}, prob=hflip_p))
        if vflip_p > 0.0:
            specs.append(TransformSpec(operation="vflip", params={}, prob=vflip_p))

        return specs


#: Geometric kwargs that describe one scaling of the same sample and therefore share ``scale_p``.
_SCALE_PARAM_KEYS = ("scale", "scale_x", "scale_y")


def _validate_geometric_probability(name: str, value: float) -> None:
    """Reject an out-of-range per-op probability at construction rather than at the first draw.

    Mirrors the bound :class:`~fused_transforms.types.TransformSpec` enforces, so the backend-free engine and the spec-
    building path refuse the same values.

    """
    if not (0.0 <= value <= 1.0):
        msg = f"{name} must be in [0.0, 1.0], got {value!r}"
        raise ValueError(msg)


def _geometric_param_transforms(
    param_specs: dict[str, tuple[float, float]],
    rotation_p: float,
    scale_p: float,
) -> list[_DirectParamTransform]:
    """Split geometric parameter ranges into direct-parameter transforms by per-op probability.

    With both probabilities at their ``1.0`` default every range stays in a single transform, which is the historical
    layout — the split would otherwise change how many probability draws the generator makes per call and move seeded
    output for pipelines that never asked for per-op probabilities. A range whose probability is below one moves into
    its own transform so its gate is sampled independently of the rest of the geometry.

    """
    if rotation_p == 1.0 and scale_p == 1.0:
        return [_DirectParamTransform(param_specs, prob=1.0)]

    # One transform per contiguous run of ranges sharing a gate, walking them in canonical order
    # (rotation -> scale -> shear -> translate). A gate is an op family, not a probability value: rotation and the
    # scale family draw independently even at equal probabilities, the scale family stays under one draw, and the
    # ungated ops keep their place after the gated ones instead of being hoisted ahead of them.
    transforms: list[_DirectParamTransform] = []
    current_gate: str | None = None
    for key, value in param_specs.items():
        if key == "rotation":
            family, probability = "rotation", rotation_p
        elif key in _SCALE_PARAM_KEYS:
            family, probability = "scale", scale_p
        else:
            family, probability = "always", 1.0
        gate = family if probability < 1.0 else "always"
        if gate != current_gate:
            transforms.append(_DirectParamTransform({}, prob=probability))
            current_gate = gate
        transforms[-1].param_specs[key] = value
    return transforms


class _DirectParamTransform:
    """Internal transform that holds parameter ranges for from_params().

    Not exported. Implements the minimal interface expected by _DirectParamAdapter.

    """

    def __init__(self, param_specs: dict[str, tuple[float, float]], prob: float = 1.0) -> None:
        self.param_specs = param_specs
        self.prob = prob
        self.same_on_batch = False


class _DirectFlipTransform:
    """Internal transform representing an hflip or vflip for from_params().

    Not exported.

    """

    def __init__(self, flip_type: Literal["hflip", "vflip"], prob: float = 0.5) -> None:
        self.flip_type: Literal["hflip", "vflip"] = flip_type
        self.prob = prob
        self.same_on_batch = False


class _DirectLetterboxTransform:
    """Internal deterministic letterbox op for from_params().

    Not exported. Classified ``CROP_RESIZE_FIXED`` so it reuses the shape-changing warp
    machinery: standalone it becomes a ``CropResizeSegment``, and after a geometric run it
    fuses into a ``_FusedGeoCropSegment`` -- one composed matrix, one resample from the
    source canvas straight to the letterboxed one.

    """

    _coordinate_matrix_recoverable = True

    def __init__(self, height_out: int, width_out: int, allow_upscale: bool = True) -> None:
        self.height_out = int(height_out)
        self.width_out = int(width_out)
        self.allow_upscale = bool(allow_upscale)
        #: Deterministic: applied to every sample, identically across the batch.
        self.prob = 1.0
        self.same_on_batch = True


class _DirectParamAdapter:
    """Internal adapter for from_params() that samples directly from param ranges.

    Not exported. Implements the TransformAdapter protocol for _DirectParamTransform and _DirectFlipTransform objects.

    """

    #: Marks this adapter as able to draw from a caller-supplied ``torch.Generator``.
    #: The backend adapters sample through their own libraries and leave it unset.
    supports_generator = True

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the TransformCategory for a direct-param transform."""
        if isinstance(transform, _DirectParamTransform):
            if "brightness" in transform.param_specs or "contrast" in transform.param_specs:
                return TransformCategory.POINTWISE_LINEAR
            return TransformCategory.GEOMETRIC_INTERP
        if isinstance(transform, _DirectFlipTransform):
            return TransformCategory.GEOMETRIC_EXACT
        if isinstance(transform, _DirectLetterboxTransform):
            return TransformCategory.CROP_RESIZE_FIXED
        return TransformCategory.SPATIAL_KERNEL

    @staticmethod
    def sample_params(
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample random parameters from the stored ranges.

        Args:
            transform: A ``_DirectParamTransform`` or ``_DirectFlipTransform``.
            input_shape: ``(batch_size, channels, height, width)`` shape tuple.
            device: Target device for the returned tensors.
            generator: Caller-owned generator driving every draw, or ``None`` for the global stream.

        Returns:
            Dict of canonical parameter tensors; a ``_batch_size`` sentinel for flips.

        """
        batch_size = input_shape[0]

        if isinstance(transform, _DirectFlipTransform):
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}

        if isinstance(transform, _DirectLetterboxTransform):
            # Deterministic op: the output size is configuration, not a draw, so the
            # generator is deliberately unused here rather than silently consumed.
            return {
                "target_h": torch.full((batch_size,), transform.height_out, device=device, dtype=torch.int64),
                "target_w": torch.full((batch_size,), transform.width_out, device=device, dtype=torch.int64),
            }

        if isinstance(transform, _DirectParamTransform):
            specs = transform.param_specs
            result: dict[str, torch.Tensor] = {}

            if "rotation" in specs:
                low, high = specs["rotation"]
                low_rad, high_rad = math.radians(low), math.radians(high)
                result["angle_rad"] = _uniform(batch_size, low_rad, high_rad, device=device, generator=generator)

            # Scale: if uniform 'scale' is set, use it for both axes
            # Individual scale_x/scale_y override uniform scale
            scale_x_range = specs.get("scale_x") or specs.get("scale")
            scale_y_range = specs.get("scale_y") or specs.get("scale")
            if scale_x_range is not None and scale_y_range is not None:
                if "scale_x" not in specs and "scale_y" not in specs:
                    # Uniform 'scale' promises isotropic scaling: one draw shared by both axes
                    scale = _uniform(batch_size, *scale_x_range, device=device, generator=generator)
                    result["scale_x"] = scale
                    result["scale_y"] = scale.clone()
                else:
                    result["scale_x"] = _uniform(batch_size, *scale_x_range, device=device, generator=generator)
                    result["scale_y"] = _uniform(batch_size, *scale_y_range, device=device, generator=generator)
            if scale_x_range is not None and scale_y_range is None:
                result["scale_x"] = _uniform(batch_size, *scale_x_range, device=device, generator=generator)
                result["scale_y"] = torch.ones(batch_size, device=device)
            if scale_y_range is not None and scale_x_range is None:
                result["scale_x"] = torch.ones(batch_size, device=device)
                result["scale_y"] = _uniform(batch_size, *scale_y_range, device=device, generator=generator)

            if "shear_x" in specs:
                low, high = specs["shear_x"]
                result["shear_x_rad"] = _uniform(
                    batch_size, math.radians(low), math.radians(high), device=device, generator=generator
                )

            if "shear_y" in specs:
                low, high = specs["shear_y"]
                result["shear_y_rad"] = _uniform(
                    batch_size, math.radians(low), math.radians(high), device=device, generator=generator
                )

            if "translate_x" in specs or "translate_y" in specs:
                if "translate_x" in specs:
                    result["translate_x"] = _uniform(
                        batch_size, *specs["translate_x"], device=device, generator=generator
                    )
                else:
                    result["translate_x"] = torch.zeros(batch_size, device=device)
                if "translate_y" in specs:
                    result["translate_y"] = _uniform(
                        batch_size, *specs["translate_y"], device=device, generator=generator
                    )
                else:
                    result["translate_y"] = torch.zeros(batch_size, device=device)

            if "brightness" in specs:
                result["brightness_factor"] = _uniform(
                    batch_size, *specs["brightness"], device=device, generator=generator
                )
            if "contrast" in specs:
                result["contrast_factor"] = _uniform(batch_size, *specs["contrast"], device=device, generator=generator)

            return result

        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build a (batch_size, 3, 3) forward affine matrix from sampled params."""
        batch_size: int | None = None
        if isinstance(transform, _DirectLetterboxTransform):
            reference = params["target_h"]
            return letterbox_matrix(
                height_in=height,
                width_in=width,
                height_out=transform.height_out,
                width_out=transform.width_out,
                allow_upscale=transform.allow_upscale,
                batch_size=int(reference.shape[0]),
                device=reference.device,
            )
        if isinstance(transform, _DirectFlipTransform):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            if transform.flip_type == "hflip":
                return hflip_matrix(width=width, batch_size=batch_size, device=device, dtype=torch.float32)
            return vflip_matrix(height=height, batch_size=batch_size, device=device, dtype=torch.float32)

        if isinstance(transform, _DirectParamTransform):
            # Determine batch size and device from any param
            batch_size = None
            device = torch.device("cpu")
            dtype = torch.float32
            for param_value in params.values():
                if isinstance(param_value, torch.Tensor):
                    batch_size = param_value.shape[0]
                    device = param_value.device
                    dtype = param_value.dtype
                    break
            if batch_size is None:
                return torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

            mtx_acc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

            if "angle_rad" in params:
                mtx_acc = matmul3x3(rotation_matrix(params["angle_rad"], height=height, width=width), mtx_acc)

            if "scale_x" in params and "scale_y" in params:
                mtx_acc = matmul3x3(
                    scale_matrix(params["scale_x"], params["scale_y"], height=height, width=width), mtx_acc
                )

            if "shear_x_rad" in params:
                shear_x_tan = torch.tan(params["shear_x_rad"])
                mtx_acc = matmul3x3(shear_x_matrix(shear_x_tan, height=height, width=width), mtx_acc)

            if "shear_y_rad" in params:
                shear_y_tan = torch.tan(params["shear_y_rad"])
                mtx_acc = matmul3x3(shear_y_matrix(shear_y_tan, height=height, width=width), mtx_acc)

            if "translate_x" in params and "translate_y" in params:
                mtx_acc = matmul3x3(translate_matrix(params["translate_x"], params["translate_y"]), mtx_acc)

            return mtx_acc

        msg = (
            f"_DirectParamAdapter.build_matrix: no recognized param keys in {list(params.keys())}. "
            "This is a bug — either sample_params() returned unexpected keys or the transform "
            "was constructed with an unknown operation name."
        )
        raise RuntimeError(msg)

    @staticmethod
    def build_color_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a homogeneous RGB matrix for native brightness or contrast."""
        factor = params.get("brightness_factor", params.get("contrast_factor"))
        if factor is None:
            if isinstance(transform, _DirectParamTransform):
                raise KeyError("native color factor is sampled at forward time")
            msg = f"build_color_matrix not supported for {type(transform).__name__!r}"
            raise NotImplementedError(msg)
        batch_size = factor.shape[0]
        matrix = torch.eye(4, device=factor.device, dtype=factor.dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        matrix[:, 0, 0] = factor
        matrix[:, 1, 1] = factor
        matrix[:, 2, 2] = factor
        if "contrast_factor" in params:
            midpoint = (
                mean.to(device=factor.device, dtype=factor.dtype) if mean is not None else torch.full_like(factor, 0.5)
            )
            bias = (1.0 - factor) * midpoint
            matrix[:, 0, 3] = bias
            matrix[:, 1, 3] = bias
            matrix[:, 2, 3] = bias
        return matrix

    @staticmethod
    def build_lut(
        transform: object,
        params: dict[str, torch.Tensor],
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Raise: the native/direct backend registers no lookup-table (gamma/solarize/posterize) ops."""
        del params, values
        msg = f"build_lut not supported for native transform {type(transform).__name__!r}"
        raise NotImplementedError(msg)

    #: Native brightness/contrast scale every channel by one factor about one midpoint, so
    #: ``FusedColorSegment`` applies them per channel to non-RGB images instead of skipping them.
    channel_uniform_color: bool = True

    @staticmethod
    def exact_flip_dims(transform: object) -> list[int]:
        """Return the spatial dims to flip for a _DirectFlipTransform."""
        if isinstance(transform, _DirectFlipTransform):
            if transform.flip_type == "hflip":
                return [3]
            return [2]
        msg = f"Cannot determine flip dims for {type(transform).__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def exact_apply(
        transform: object,
        image: torch.Tensor,
        *,
        params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly."""
        del params
        if isinstance(transform, _DirectFlipTransform):
            if transform.flip_type == "hflip":
                return image.flip(dims=[3])
            return image.flip(dims=[2])
        msg = f"Cannot apply exact op for {type(transform).__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Passthrough - direct-param transforms are always fusible."""
        return image
