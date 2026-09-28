# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions below `0.12.0` are `dev0` snapshots, each cut from its own `bump vX` commit's tree. `0.1.0.dev0` through `0.9.0.dev0` were batch-uploaded to PyPI together on 2026-07-11, catching up nine commits at once; `0.10.0.dev0` followed separately on 2026-08-02. `0.11.0.dev0` was never published to PyPI — only `0.12.0` and later exist there next to it. `0.12.0` is the first stable release and the first cut from a `vX.Y.Z` tag.

## [Unreleased]

### Breaking changes

- The distribution was renamed from `fuse-augmentations` to `vision-synth`. Install with `pip install vision-synth`; the old name receives no further releases and no alias distribution is published, so `pip install fuse-augmentations` keeps resolving to `0.14.0` and earlier only. The import packages were renamed alongside it — see the two entries below. The repository moved to `github.com/Borda/vision-synth` and the documentation to `borda.github.io/vision-synth` (GitHub redirects the old repository URLs). The name now covers both halves of the package: synthesizing labelled data and synthesizing augmentation pipelines, rather than only the augmentation fusion the old name described.

- The import package was renamed from `fuse_augmentations` to `fused_transforms`. No compatibility shim is installed, so `import fuse_augmentations` raises `ModuleNotFoundError`; rewrite the import (`from fused_transforms import Compose`). The rename ships in the same release as the distribution rename deliberately: the old distribution publishes no alias, so nobody reaches this release without already editing their install line, and a later rename would force the same users through a second migration.

- The `fuse_aug` short-name re-export package is removed. It re-exported `fuse_augmentations` unchanged; import from `fused_transforms` instead. Two import paths to one API cost more in documentation and tooling (IDE completion, mypy) than the four characters they saved.

- The third-party adapter entry-point group was renamed from `fuse_augmentations.adapters` to `fused_transforms.adapters`. Downstream packages exposing an adapter must update the group name in their own `pyproject.toml`; the old group is not read as a fallback. The mechanism stays flagged experimental.

- Objects pickled by an earlier release record `fuse_augmentations.*` module paths and cannot be unpickled after the rename. Regenerate them from a live pipeline rather than loading the old file.

- `torch` moved from a hard dependency to the `torch` extra. `pip install vision-synth` no longer installs torch; the augmentation engine (`Compose`, `FusedCompose`, `AugmentationSequential`, and everything else at the `fused_transforms` package root) now needs `pip install "vision-synth[torch]"` (or any adapter extra — `kornia`, `torchvision`, `albumentations`, `all` — which each now pull `torch` explicitly). Importing `fused_transforms` without torch installed raises a clear `ModuleNotFoundError` naming the extra, instead of a raw error from deep inside a submodule.

- The synthetic-dataset generator moved from `fuse_augmentations.data` to a new standalone top-level package, `synth_datasets` (`pip install vision-synth` alone is enough; `import synth_datasets`), and no part of the old path survives. `fused_transforms.data` and `fused_transforms.generate_dataset` are both removed, along with former submodule paths such as `fuse_augmentations.data.geometry`. Import everything dataset-related from `synth_datasets`, which is torch-free; the facade was not, since reaching it ran the torch-dependent augmentation package's `__init__`. Removing it in the same release as the import rename keeps the whole rebranding to one migration instead of two.

- `fused_transforms`'s flat module tree was regrouped by concern: `pipeline.py`, `factories.py`, `introspection.py`, `planner.py`, and `config_validation.py` moved under `fused_transforms.core`; `resolver.py`, `converters.py`, and `substitution.py` moved under `fused_transforms.dispatch`. No compatibility shim is installed at the old paths, so objects pickled by an earlier release — including a `FusedCompose` instance, or one holding a `_PassthroughSegment`/`_DirectParamAdapter` in its segment list — record the old module paths and cannot be unpickled after this change; regenerate them from a live pipeline instead. `fused_transforms.compose` (the historical pickle-compat shim for `pipeline.py`'s own earlier rename), `targets.py`, and `detection.py` did not move and keep working unchanged.

- `SyntheticIterableDataset` seeds every stream from `numpy.random.SeedSequence(seed, spawn_key=(rank, epoch, worker_id))`. Rank 0 / epoch 0 used to seed worker `w` with plain `seed + w`, so worker 1 at seed `s` replayed worker 0 at seed `s + 1` image for image, and two "independent" runs at adjacent seeds overlapped. Dropping that special case changes **every** seeded stream the dataset yields, including a single-process, `num_workers=0` one, which no longer matches `SyntheticGenerator(config).generate(n, seed=s)`; a run recorded against an earlier release's seed reproduces different images. `generate_dataset` and `SyntheticGenerator` are unaffected. `rank` and `world_size` now default to `None`, read from the default `torch.distributed` process group when it is initialized at construction, and fall back to `0` and `1` otherwise.

- Built-in writers refuse to write into a populated destination. A second `generate_dataset` into the same directory used to overwrite `img_000000..` in place and keep every higher-numbered file of the first run (20 images then 5 left 14 in `images/train`, under a `data.yaml` describing neither), silently mixing two datasets. `CocoWriter` and `YoloWriter` now raise `FileExistsError` before consuming a sample when any path they own for the run's splits — `<split>/` for COCO; `images/<split>`, `labels/<split>` and `data.yaml` for YOLO — already holds files. Each run records the paths it wrote in a `.vision-synth.json` manifest at the output root. Pass `generate_dataset(..., overwrite=True)` (`--overwrite` on the command line) to replace them: this removes the run's own paths plus every path the manifest lists, so a split an earlier run wrote and this run does not goes too, and touches nothing else under the output directory — ownership is never inferred from file names, so a curated dataset beside the output survives. The replacement is staged and swapped, never deleted first: the new dataset is written into a `.vision-synth-staging-*` directory inside the output directory, and only once it is complete are the earlier dataset's paths renamed into a `.vision-synth-backup-*` directory and the new entries renamed into place. Paths are compared as the filesystem stores them, so on a case-insensitive filesystem a split respelled only in case (`train` to `Train`) is replaced once and recorded with the new spelling. Once each new entry is confirmed in place, the overwrite commits by renaming the backup and staging directories to `.vision-synth-discard-*`, and only then are they removed. Each staging and backup directory carries a `.vision-synth-owner.json` ownership marker, written atomically, and the backup's marker records the paths the swap replaces and the ones it adds. A run that fails while generating (a fractional `distractors`, say) leaves the earlier dataset as it was. A swap stopped by a failed rename or by Ctrl+C is rolled back by renaming only, judged from what is on disk, and the backup directory is removed with `os.rmdir` alone once a fresh listing shows the earlier dataset fully back, or otherwise kept and named in the error. Nothing the call did not create is ever deleted. After an interrupted overwrite, every write into that directory refuses and classifies each leftover by prefix and marker. A marked discard directory means the overwrite completed and holds only the superseded copy. A marked backup means it did not commit, and the refusal gives restore steps per path, checked against the disk when it is raised: a path is replaced only when its backup copy exists (otherwise the original is still in place, and it says to leave it), and an added path is removed only when it is present and has no backup copy. A marked staging directory holds only never-promoted files. A reserved-looking directory without a valid marker is reported as not created by vision-synth, to be renamed or moved, never deleted (see "Recovering from an interrupted overwrite" in `docs/datasets/outputs.md`). Recovery never names a path the run did not record. A path whose metadata cannot be read (a permission or I/O error) is never taken for a missing one: planning, rollback and clean-up stop, a marker that cannot be read is reported as "ownership unknown", and a restore step for such a path is reported as "state unknown" rather than guessed. Every path is validated before anything is deleted: one that resolves outside the output directory or lies under a symlinked directory, or a `.vision-synth.json` that is not a regular file, makes `overwrite` raise `ValueError` and delete nothing.

- `SplitRatios` rejects a split name that is not one plain path component (empty, `.`, `..`, or holding `/`, `\`, `:` or NUL), or that Windows cannot create as a directory (a device name such as `CON`, `NUL`, `aux.txt`, `COM1`, `LPT9` or `COM¹` in any case, one of `< > " | ? *` or a control character, or a name ending in a dot or space), or that starts with `.vision-synth` in any case, which the writers keep for their manifest and their staging and backup directories. `SplitRatios.custom({"../escaped": 1.0})` used to be accepted, and a COCO export then wrote `escaped/` beside the output directory instead of inside it.

- `FusedCompose` raises `ValueError` for a `data_keys` entry outside `input`, `mask`, `bbox_xyxy`, `bbox_xywh`, `keypoints` and `rboxes`, with a did-you-mean hint (`"boxes"` suggests `"bbox_xyxy"`). It used to warn and pass the tensor through unchanged, so a target named after another library's convention stayed put while the image was warped.

- `nn.Module` transforms inside a pipeline, including Kornia and TorchVision transforms, are registered as submodules. `parameters()`, `state_dict()`, `.to()`, `.eval()` and `.train()` now reach a learnable or stateful custom transform, which they silently skipped. A model that embeds a pipeline gains `_module_transforms.*` keys in its `state_dict()`, so loading an earlier checkpoint with `strict=True` reports them as missing; load with `strict=False` or re-save.

- `Compose.from_params(rotation_p=..., scale_p=...)` gates rotation and the scale family with independent draws and keeps the order rotation, scale, shear, translate. Ranges were grouped by probability value, so equal probabilities shared one draw and ungated operations were hoisted ahead of gated ones. Seeded output changes for pipelines that set either probability below `1.0`; with both at the default `1.0` it is unchanged.

- A single-transform pipeline honours the `fill`, `padding_mode` and `interpolation` given to `Compose`. Its native fast path rendered with the transform's own settings instead, so the same options behaved differently with one transform than with two. `from_config` and `from_params(backend=...)` pass `interpolation="bilinear"` and `padding_mode="zeros"` by default, so their single-transform pipelines now render bilinear/zeros through the matrix path; output changes where the transform's native mode differs (TorchVision `RandomRotation` defaults to nearest).

### Added

- The `vision-synth` console script. `vision-synth generate ./ds 1000 --fmt yolo --task obb --shapes duck,camel --seed 0` runs `generate_dataset`; every other `--flag` is a `SyntheticConfig` field. It needs the `cli` extra (`pip install "vision-synth[cli]"`), which until now installed `fire` for no command at all, and says so when `fire` is missing.

- `SyntheticConfig(shapes=...)` and `distractor_shapes=...` accept shape names (`("duck", "camel")`, or one `"duck"`) as well as members, resolving each at construction, so a YAML- or CLI-driven config no longer maps names to enums itself. An unknown name raises `ValueError` listing every valid one; `synth_datasets.families.resolve_shape` does the lookup.

- Rectangular canvases: `SyntheticConfig(img_size=(width, height))`, read back as `config.canvas_size`. Placement, the boundary check, keypoint visibility, the occluder stencil, every built-in background and degradation, both writers, `SyntheticIterableDataset` and the command line (`--img_size 96,48`) work per axis; object size ratios apply to the shorter side, and `TextureBackground.frequency` counts features across the shorter side. `Background.render(rng, img_size)` receives a plain `int` for every square canvas, so an existing subclass keeps working, and `(width, height)` for a rectangular one. Square configurations produce the same pixels and annotations as before.

- `SyntheticConfig(colors=...)` (and every other fill) accepts colour names, in any case: `colors=("red", "blue")`, or `--colors red,blue` on the command line. An unknown name raises `ValueError` listing the valid ones.

- The `vision-synth` command line reports an invalid value, or an output directory that already holds a dataset, as one line on stderr with exit status 1 instead of a traceback, and names the `--overwrite` flag. An unknown flag lists the valid ones, and every value is checked against its option's type (the `SyntheticConfig` field annotation, or the command's own option) so that a misused one (`--min_objects bad`, `--min_objects None`, `--shapes 5`, `--split_ratios 5`, `--seed [1]`) names its flag instead of escaping as a `TypeError` traceback. Colour options are checked against every spelling they accept — comma-separated names, one `(r, g, b)` triple, or a list of names and triples (`--distractor_colors '[(255,215,0)]'`) — before they are normalized.

- `PrimitiveShape`, `AnimalShape`, `SymbolShape` and `LetterShape` are exported from `synth_datasets`. `SyntheticConfig` is documented against them, yet `from synth_datasets import AnimalShape` raised `ImportError`.

- `DatasetWriter.owned_paths` (a subclass hook, empty by default), `DatasetWriter.prepare_output` (refuses a populated destination; never deletes), `DatasetWriter.record_output` (which writes the `.vision-synth.json` manifest) and `DatasetWriter.write_replacing` (the staged replacement `overwrite=True` runs) let a custom writer opt into the same refuse-or-replace check the built-ins run.

- `docs/reference/datasets-generation.md` and `docs/reference/datasets-scenes.md` give `synth_datasets` a generated API reference. Until now the four reference pages covered `fused_transforms` only, so 48 exported dataset names and 121 documented public definitions rendered nowhere and were unsearchable on the site. `mkdocs.yml` also watches `src/synth_datasets`, so `mkdocs serve` reloads on a dataset docstring change.

- `docs/applications/generate-and-augment.md` documents the path from a generated sample to a fused pipeline. No page previously ran both packages in one example, even though their contracts already align: `Sample.image` is HWC `uint8`, and `Annotation.bbox_xyxy` is in the pixel-edge space `transform_bbox_xyxy` assumes, so a box crosses with no half-pixel adjustment.

- `docs/applications/synthetic-data-experiments.md` routes a question to the experiment that answers it, and checks COCO's 1-based categories against YOLO's 0-based classes over one seeded export.

- `Compose.from_params(brightness=..., contrast=...)` accepts a `(low, high)` factor range as well as a deviation, matching the tuple form of `rotation=` and `scale=`.

- `print(pipeline)` shows the fusion plan and the number of warps saved; it used to print `FusedCompose()`.

### Fixed

- `generate_dataset(fmt=...)` resolves every key registered with `register_writer`, as `register_writer`'s own documentation promised. It coerced `fmt` through `OutputFormat(fmt)` first, so a registered custom format failed with `'<key>' is not a valid OutputFormat`; an unknown key now fails with the list of registered formats instead.

- `SyntheticConfig.colors` documented that restricting colors never renumbers classes. Under `ClassMode.COLOR` and `ClassMode.SHAPE_COLOR` it does, exactly as restricting `shapes` does: `colors=(Color.BLUE,)` declares one class, `blue`, with id `0`. The docstring now says so and a doctest pins it.

- `docs/llms.txt` described the removed `fused_transforms.generate_dataset` alias as a working compatibility import needing the `torch` extra. An agent following it hit `ImportError`. The entry now states that `synth_datasets` is the only import path.

- Exact quarter-turn operations such as Kornia `RandomRotation90` keep an integer or boolean mask's dtype. The mask was concatenated with the float image and came back as `float32`, and its values came back rounded once they passed `2**53` (`2**53 + 1` became `2**53`; int64 max saturated). The mask now follows two planes encoding each pixel's integer index through the same draw and is gathered from them, so every label comes back exact on every device, MPS included; a canvas too large to index that way is refused with `ValueError` rather than routed inexactly.

- A standalone `FusedAffineSegment` pickled before the single-transform render-override flag existed restores with the rendering it had (its native fast path); only `FusedCompose` used to migrate it, so the bare segment raised `AttributeError` on its first call.

- `antialias=True` requires Kornia to import, not merely to be installed. A broken Kornia install was accepted and failed later, on the first downscale.

- `FusedCompose.inverse()` works with `padding_mode="per_transform"`; it passed the policy name straight to `grid_sample` and raised.

- Backend-free `brightness` and `contrast` apply to images with any channel count. On anything but three channels they silently did nothing.

- An unbatched `(C, H, W)` tensor passed to a fused pipeline raises `ValueError` suggesting `x[None]` instead of `not enough values to unpack`.

- Documentation corrections: keypoint exports declare only the run's own classes (`docs/datasets/tasks.md`); `numpy` and `numpy_hwc` outputs are one contract that squeezes a batch of one (`docs/getting-started/quickstart.md`); `augment_detection_batch` returns no keep mask (`docs/applications/detection-and-keypoints.md`); masks do not recover content warped off the canvas (`docs/applications/test-time-augmentation.md`); the worker-seeding recipe reseeds `pipeline.generator` (`docs/guides/reproducibility.md`); the README no longer claims shape families can be registered at runtime.

### Changed

- `import fused_transforms` no longer imports Kornia, TorchVision, Albumentations or OpenCV; each loads on first use. Import time drops from about 1.4 s to 0.4 s, roughly torch's own. A backend that is installed but fails to import emits a `UserWarning` carrying the original error instead of looking absent.
- Tests that need an optional module skip only when it is absent (`importorskip(..., exc_type=ModuleNotFoundError)`); a broken install now fails them.
- `README.md` reorders around the two packages: install, then synthetic datasets with a generator capability table, then the augmentation engine. The dataset section previously sat twelfth of sixteen despite the intro and routing table leading with generation. "Where it fits" and the documentation list now cover both halves.
- `docs/known-limitations.md` gained a synthetic-dataset limits section — scope, the absent `difficulty=` control, keypoint-family restrictions, split stability, occluder geometry, baked degradations, COCO export memory, and the training-free nature of the difficulty bands. The page previously carried eleven augmentation sections and no dataset limits.
- `docs/applications/index.md` no longer opens with "This package solves one problem", which described the augmentation engine alone on a landing page reached by a top-bar tab. It now splits into synthetic generation and fused augmentation.
- The package maturity classifier moved from `Development Status :: 3 - Alpha` to `Development Status :: 4 - Beta`, with the matching prose updated in `README.md`, `docs/index.md`, and `docs/getting-started/installation.md`. Advanced segment classes and the third-party adapter extension point stay flagged as provisional.
- The `torchvision` floor was raised from `>=0.15` to `>=0.17` in the `torchvision` and `all` extras, to match the `torch>=2.2` pin now explicit alongside it; `torchvision` 0.17 is torch 2.2's paired release, and keeping `0.15` would let the resolver pick a `torchvision` build against `torch` 1.x/2.0.
- The `torch` floor stays at `>=2.2`, accepted as-is: this range resolves two known advisories, [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) (CVE-2025-3000 / PYSEC-2025-194, fixed in `torch` 2.13.0) and CVE-2026-4538 / PYSEC-2026-139, which has no fix released for any `torch` version yet. Both are local-vector, low-privilege issues. Applications that need the patched `torch.jit.script` should pin `torch>=2.13` themselves.

## [0.14.0] - 2026-09-23

Synthetic-dataset scene realism: image-crop backgrounds, procedural background types, distractor clutter, occluders, pixel-value degradations, and a measured difficulty ladder; plus a 1px label-emission fix.

### Added

- `ImageBackground` crops the canvas from a directory of the caller's own pictures (`image_dir` required, no default; refuses a missing, empty, or corrupt-file directory at construction rather than rendering black). Recursive, sorted, cached file listing for reproducible seeding; upscales a picture smaller than the canvas. Source path recorded on `sample.scene.background_source` via new `Background.render_with_source`.
- `docs/datasets/difficulty.md` maps easy/moderate/hard configurations to a difficulty band, ranked by five deterministic proxy statistics (object area, small-object share, boundary contrast, background SNR, clutter coverage). Deliberately no `difficulty=` parameter — difficulty stays a property of the knob combination.
- `Sample.scene` carries a `SceneRecord` (typed side-car, defaults to a shared empty record, `eq=False`; re-clears mask writability after unpickling for `num_workers > 0`).
- `SyntheticConfig.occluders` draws unlabelled shapes over labelled objects; demotes a covered landmark from COCO visibility `2` to `1` (never promotes `0`). `sample.scene.occluder_mask` publishes the modal mask (read-only, `None` if unused). Ignores `boundary_tolerance`/`overlap_iou`.
- `SyntheticConfig.distractors` draws unlabelled shapes beneath labelled objects from `distractor_shapes`/`distractor_colors` (default: complement of `shapes`/`colors`, plus a new `DISTRACTOR_PALETTE`). No annotation, no IoU constraint, best-effort placement; raises at construction if a pool is empty. `resolved_distractor_shapes`/`resolved_distractor_colors` derive pools on read so `dataclasses.replace` can't stale them.
- `SyntheticConfig.degrade`: a tuple of pointwise finishing effects — `GaussianNoise`, `GaussianBlur`, `JPEG`, `Contrast`, `ColorCast`, `Vignette`, `Quantize` — applied in order, pixel values only, no label movement. `uint8` in/out; `degrade=()` is a no-op.
- Background, distractors, occluders, and degradations each now draw from their own side stream (`Generator.spawn`), so enabling one knob no longer perturbs the others' draws.
- `SyntheticConfig.background` accepts a `Background` object as well as a plain fill: `SolidBackground`, `GradientBackground`, `NoiseBackground`, `ImpulseNoiseBackground`, `TextureBackground`. A bare `(r, g, b)` normalizes to `SolidBackground`. **Breaking:** `config.background` reads back as a `Background` object, not the passed triple; a colour name (`"white"`, `"#204080"`) is now rejected at construction instead of silently rendering black. Pinned by digest fixtures across 25 configurations.
- Documented recipe for a tight rotated box: warp the polygon through `["input", "keypoints"]` and take its extent (0.96–0.98 IoU) instead of the corner-based `transform_bbox_xyxy` AABB (0.51–0.83 IoU at 37°). No API change.

### Fixed

- `polygon`, `keypoints`, and `obb_corners` are now emitted in pixel-centre coordinates (previously off by 1px in the rasterizer's edge space after any flip/quarter-turn; sub-pixel and unnoticed under generic rotation). `bbox_xyxy` and exported COCO/YOLO files are unchanged — writers still convert back to edge space at the file boundary.

## [0.13.0] - 2026-09-06

Training-loop boundaries (`mask_fill`, `augment_detection_batch`, dataset sharding), NumPy-native multi-target execution (`execution="auto"`, channel-last arrays), and the pixel-edge bounding-box migration. Performance figures below reflect their original measurement runs; current numbers live in `docs/research/benchmarks.md` — historical CUDA/MPS ratios need revalidation.

### Added

- Executable classification/segmentation model-step checks alongside the existing detector/pose/TTA/letterbox integration tests (losses, gradients, ignore-label routing — not accuracy).
- `mask_fill` on `Compose`: scalar auxiliary-mask border independent of image `fill`, including ignore labels `255`/`-1`.
- `augment_detection_batch`: packs per-image boxes/labels through the dense box pipeline, clips/filters via one survival mask, keeps `area`/`iscrowd`/`image_id` aligned.
- `SyntheticIterableDataset` accepts explicit `rank`, `world_size`, immutable `epoch` (`num_images` is per rank); recreate per epoch with `persistent_workers=False`.
- Exact geometric ops and deterministic letterbox expose their actual pixel-centre matrix via `return_matrix=True`.
- Multi-target path accepts channel-last NumPy images (`Compose(..., data_keys=[...])`); returns results in the caller's own dtype/layout. Tensor and array inputs can't mix in one call (`TypeError`).
- `rotation_p`/`scale_p` on `Compose.from_params` (keyword-only, default `1.0`, bit-identical at default): per-sample probability for the geometric ranges, matching `hflip_p`/`vflip_p`.
- Albumentations `RandomSizedCrop` registered as a crop-resize op, routing masks/boxes/keypoints like `RandomResizedCrop`. `RandomSizedBBoxSafeCrop` stays an unregistered barrier by design.
- `execution="auto"`: resolves cv2/torch per call by device (host → cv2, accelerator → `grid_sample`); never the default. New `resolved_execution` property reports the engine actually used.
- Multi-target NumPy calls now stay in NumPy/cv2 space instead of round-tripping through the tensor path — 1.05x native Albumentations at 640/1024px, 2.9x on a four-op chain (previously 0.48x).
- Multi-target pipeline accepts the image positionally alongside keyword auxiliary targets (`pipe(image, bboxes=...)`).

### Changed

- Malformed target shapes and non-involutive keypoint pair tables now fail at the boundary instead of silently misbehaving. Unknown spatial passthroughs are refused up front when auxiliary targets are supplied.
- Public docs cover pixel-edge boxes, mask padding, the ragged detector adapter, rank/epoch lifecycle, exact/letterbox metadata, CPU-only passthrough transfers.
- Albumentations affine/projective tensor paths reuse native NumPy matrix preparation; older perspective pickles rebuild the cache on restore.
- `antialias=True` now requires Kornia; filters aggressive crop-resize downscales per sample instead of per batch. Default `False` unchanged.
- **Bounding-box numerical migration:** xyxy/xywh now use pixel-edge extents `[0, W] x [0, H]`; full-frame boxes survive flips without a 1px shift. Convert rbox envelopes by `+0.5` against edge-space AABBs.
- NumPy auxiliary masks retain label values/dtype during layout conversion; integer masks reject bilinear sampling like tensor masks.
- Fused affine segments on an accelerator now sample parameters and build matrices on host, then transfer once — 1.51x faster on MPS at batch 8 (previously ~10 host-device copies per call).
- Under `execution="torch"`, image-only Albumentations crops route through `CropResizeSegment` instead of a native passthrough — 0.83x at batch 8, 1.17x at 32, 1.39x at 64 on MPS.
- `transform_bbox_xyxy`'s four-corner AABB reduction uses `torch.amin`/`amax` instead of `Tensor.min/max(dim=)` — detection-step cost fell from 0.40ms to 0.15ms. Backward-pass subgradient tie-breaking and NaN propagation change; forward values unaffected.
- Residual float32/float64 divergence between the NumPy and adapter matrix paths is now bounded (≤1 float32 epsilon) and tested (`TestEntryPointAgreement`).

### Fixed

- The NumPy image path now always draws one Bernoulli activation per transform, matching the tensor path (previously skipped the draw at `p=0.0`/`p=1.0`, desyncing the shared `numpy.random` stream).
- Auxiliary-target geometry no longer inherits the image's dtype (was rounding affine coefficients to low precision); routes in float64/float32 matching the image warp.
- A raw NumPy image on the multi-target path no longer crashes with a shape-unpacking error (`_forward_kwargs_dict`'s `cast` was a no-op at runtime).

## [0.12.0] - 2026-09-02

First release published from a tag rather than a batch upload, and the first non-`dev0` version. Adds keypoint mirroring, rotated boxes, letterbox with an exact inverse, instance-survival helpers, constant fill, and caller-owned generators.

### Added

- `keypoint_flip_index=` (`Compose`/`from_params`/`from_config`): caller-supplied keypoint pair permutation, swapped whenever the composed transform reverses orientation (read off the matrix determinant's sign, not a discrete flip op). Applies on every routing path, including the exact-flip and D4 fast paths. `orientation_reversed()`/`permute_keypoint_pairs()` expose the same decision. Default `None` = no swap.
- Rotated boxes are a first-class target: `"rboxes"` data key `(B, N, 5)` `(cx, cy, w, h, theta)`, routed by every existing box path. Public helpers: `rboxes_to_corners`, `corners_to_rboxes`, `transform_rboxes`, `mirror_rboxes`, `shift_rboxes`, `rbox_envelopes`. The corner re-fit is exact under rotation/scale/translation/mirror, approximate under shear. No canonical form imposed (`canonicalize=` callable available). Clipping not provided — use `rbox_envelopes` with `clip_bbox_xyxy`/`instance_keep_mask`.
- `letterbox=(height, width)` on `from_params`: single-ratio scale plus pad, exact inverse via `inv3x3`. Also public `letterbox_matrix()`/`letterbox_geometry()`/`LetterboxGeometry`. Fuses into the geometric chain (one `grid_sample`). `allow_upscale=False` caps the ratio at `1.0`. Backend-free only.
- `instance_keep_mask()` and `clip_bbox_xyxy()` (`fuse_augmentations.targets`, also top-level): the post-warp instance-survival rule. `clip_bbox_xyxy` clamps to the canvas extent; `instance_keep_mask(boxes, clipped_boxes, min_size=, min_visibility=)` returns a **mask**, never filtered boxes, keeping labels/keypoints/rboxes aligned.
- `fill=` (`Compose`/`from_params`/`from_config`): constant out-of-canvas colour in the image's own value range, per-channel or scalar. Requires `padding_mode="zeros"`; raises against other modes. Image only — routed masks keep zero padding.
- `generator=` (`Compose`/`from_params`/`from_config`): caller-owned `torch.Generator` drives every pipeline draw, isolating it from the global RNG stream. `None` (default) unchanged. Direct-parameter engine only — raises if combined with backend transforms (Kornia/TorchVision/Albumentations sample from their own RNGs). Pickled state carries the generator.

### Fixed

- Pipelines built with `generator=` now pickle on every supported torch version (`torch._C.Generator` was previously unpicklable); state is serialized and segments rebind to one restored instance.
- `_build_mixed_segments` now forwards `fill` and `keypoint_flip_index` (previously silently dropped on the mixed-backend planner path).

### Changed

- **Half-pixel convention pinned as a single, tested convention.** `align_corners=True` sampling against the derived normalization sandwich already matches `align_corners=False` implementations (TorchVision, Albumentations, YOLO pipelines) in pixel space; now asserted across shifts, scales, rotation, non-square canvases, and cross-size `normalize_matrix_io`. `padding_mode="reflection"` genuinely differs (pixel-centre vs. pixel-edge reflection), and canvases thinner than 2px on an axis are refused by name. No `align_corners` parameter added.
- **Releases now cut from a `vX.Y.Z` tag, not a batch upload.** `.github/workflows/release.yml` verifies tag/`__version__` agreement, runs `twine check`, and publishes via PyPI Trusted Publishing. `.github/CONTRIBUTING.md` documents the three release steps (close the changelog, bump `__version__`, tag and push).

## [0.11.0.dev0] - 2026-09-02

Three new shape vocabularies — animals, symbols, letters — each with a keypoint schema, followed by a `data` module restructure into one shape-family registry.

### Added

- `AnimalShape`: 12 animal side-profile silhouettes (duck, elephant, giraffe, fish, rabbit, camel, eagle, penguin, whale, kangaroo, flamingo, crocodile), none rotationally symmetric. `SyntheticConfig.shapes` restricts the drawn vocabulary (default: the original 4 geometric shapes, so existing seeded output is unchanged).
- `Task.KEYPOINTS`: 16-point anatomical schema (mouth, eye, ear, head, neck, body_top/bottom, tail, front_elbow/limb, hind_knee/limb pairs) shared across quadrupeds/birds/swimmers; `ear` and hind-leg points are optional (NaN when absent). COCO skeleton plus YOLO pose (`kpt_shape: [16, 3]`). One editable SVG per animal.
- `SymbolShape`: 7 analytically-computed straight-edge symbols (kite, trapezoid, house, arrow, cross, teardrop, anchor); 3 are concave. `Shape` widens to `GeomShape | AnimalShape | SymbolShape` (23 classes total).
- `Task.KEYPOINTS` for `SymbolShape`: 7-point schema (`center` mandatory; `apex`/`tail`/`flank_*`/`base_*` optional) with a genuine `flip_idx` swap.
- `SyntheticConfig.asymmetry_jitter` (default `0.0`, range `[0, 0.5)`): narrows a random half of each placed object to break left/right OBB symmetry; excludes `circle`.
- `examples/render_shape_reference.py`: one static reference image per shape with its detection box and, where applicable, keypoints/skeleton.
- `LetterShape`: 26 capital letters, authored skeleton-first (nodes plus edges, stroked into a polygon at import time) so every keypoint sits strictly inside the ink. Rounded stroke tips/corners; 7 letters split their counter-closing edge to keep one simple ring. `Shape` widens to include `LetterShape` (49 classes total).
- `Task.KEYPOINTS` for `LetterShape`: 15-point schema whose keypoints/skeleton are the stroke graph's own nodes/edges (`KeypointSchema.skeleton_by_value`/`skeleton_for()`). Mirroring is not label-preserving for letters (`flip_idx` still published for format completeness).
- Symbols'/letters' raw shape data packaged as JSON assets loaded at import time (superseded later this cycle by per-shape SVGs).

### Fixed

- `Annotation` now carries its own `KeypointSchema` instead of inferring the family from table length.
- `DatasetWriter`/`get_writer` no longer default `keypoint_schema` to the animal schema; `Task.KEYPOINTS` without one now raises.
- `class_names()`/`class_id()` require an explicit `shapes` vocabulary (default previously spanned all 49 shapes against a 4-shape config default).
- **Annotation class ids now resolve against the vocabulary written beside them.** A narrowed shape family (any animal/symbol/letter run) previously wrote renumbered categories but globally-numbered annotation ids, raising `KeyError` on read. `class_id_of`/`class_id` now take the same `shapes` argument as `class_names`. Non-prefix families' ids shift to start at `0` — compare datasets by class name, not raw id.
- A bare-string `class_mode` (e.g. `"shape"`) no longer selects the wrong vocabulary — `SyntheticConfig` and both vocabulary functions now coerce to a real `ClassMode` member.
- `polygon_to_obb`'s chosen orientation no longer flips between renders of the same shape at different angles (resolved outright by the upright-frame OBB semantics below).

### Changed

- **`fuse_augmentations.data` API restructure — breaking for former submodule imports; package-level facade retained:**
    - One shape-family registry (`data.families.SHAPE_FAMILIES`, `ShapeFamily`, `ALL_SHAPES`, `family_of`, `shape_outline`) replaces six independently-encoded family lists.
    - `generate_dataset` no longer takes `task=`/`class_mode=` — both are `SyntheticConfig` fields.
    - `class_vocabulary()` returns typed `ClassEntry` records instead of splitting class names on `"_"`.
    - `fuse_augmentations.data` resolves `SyntheticIterableDataset` lazily (PEP 562) — no torch import.
    - `register_writer()` and `SplitRatios.custom()` open extension points for output formats and split names.
    - Every family enum now derives from `data.shape_enum.ShapeEnum`; `Shape` **is** that base rather than a hand-written union.
    - `Fill` (frozen dataclass: RGB plus originating `Color`, if any) is the fill type everywhere past the config boundary; `SyntheticConfig` normalizes any spelling via `Fill.parse()`. **Breaking:** `config.colors` reads back as `Fill` objects.
- **Renames (breaking):** `GeomShape`→`PrimitiveShape` (→ `data.primitives`), `shape_polygon`→`shape_outline` (→ `data.families`), `class_id_of`→`class_id`, `data.landmarks`→`data.keypoints`, `Task.OBB`'s value `"oriented_bounding_boxes"`→`"obb"`. `data.shapes` shim removed; per-family `*_shapes(n)` helpers removed (use `tuple(Family)[:n]`).
- **Oriented boxes are now upright-frame, not minimum-area.** `Annotation.obb_corners` is the shape's own axis-aligned box rotated rigidly by the placement angle (previously the true minimum-area rectangle via rotating calipers, which leaned off-axis for shapes like `kite`/`arrow`/`teardrop`). New `Annotation.angle` field, inserted before `keypoints` — positional callers must switch to keywords. **Breaking** for consumers expecting minimum-area boxes.
- Symbol SVGs gain a `skeleton` visualization group, matching the animal assets.
- Synthetic-dataset docs split from one 800-line guide into a 5-page `docs/datasets/` section.
- Symbols and letters ship as editable per-shape SVGs (`data/symbols/`, `data/letters/`) instead of JSON, parsed by one `data.svgio` module and edited by `examples/edit_shape_keypoints.py`.
- `animate_synthetic_dataset.py` family previews now guarantee every shape in the family appears at least once (previously a lucky subset).
- `data.shapes` moved to `data.geometry` (deprecated re-export shim kept temporarily, removed above later this cycle).
- `Shape` widened progressively: `GeomShape | AnimalShape` → `+ SymbolShape` → `+ LetterShape` → replaced by the one-base-class registry.
- Animal/symbol outline normalization now centers on the polygon's area centroid instead of the vertex mean (small position shift; `GeomShape` unaffected).

## [0.10.0.dev0] - 2026-08-01

Synthetic dataset generator ships; test-time `inverse()` de-augmentation; Gaussian-blur and lookup-table fusion; per-transform padding mode.

### Added

- `fuse_augmentations.data`: standalone synthetic dataset generator — colored shapes on a canvas, COCO/YOLO export for detection/segmentation/OBB. `generate_dataset(...)` facade over `SyntheticGenerator`/`CocoWriter`/`YoloWriter`; fully seeded, streaming (memory-bounded), plus `SyntheticIterableDataset` for zero-round-trip `DataLoader` use. Pillow now a base dependency.
- `inverse()`: maps predictions (image, masks, boxes, keypoints) back to the original frame via the inverse fused matrix, from a `return_matrix=True` call. Supported only for a single affine/projective segment; boxes are AABB and inflate under rotation/shear.
- `pipeline_dtype="bfloat16"|"float16"`: half-precision warp/color ops on non-CPU (~2x bandwidth); matrix math stays float32/64. Default unchanged.
- Gaussian blur now folds and commutes instead of being a hard fusion barrier: consecutive blurs merge by variance addition; a blur before a fusible affine commutes to share one warp (axis-aligned, then general affine via the full covariance transform). Downscaling affines and rotated/sheared cv2-native runs keep it a barrier.
- Per-channel non-linear maps (`gamma`, `solarize`, `posterize`, `equalize`) fuse into a single lookup table (`POINTWISE_LUT`/`FusedLUTSegment`) instead of one pass each; `equalize`'s runtime histogram is built per call.
- `padding_mode="per_transform"`: honors each transform's own border mode instead of one pipeline-wide override; modes without an exact `grid_sample` equivalent stay a native passthrough with a warning.

### Fixed

- `inverse()` normalizes/inverts the paired matrix in full precision regardless of image dtype.
- `RandomRotate90`/D4 quarter-turn matrix direction corrected to match native `np.rot90`/`exact_apply`.
- Downscale antialias prefilter now reads the correct axis for anisotropic downscales; `Perspective(keep_size=False)` now raises `NotImplementedError` instead of silently mis-warping.
- Albumentations `RandomResizedCrop` with auxiliary targets now routes through a real `CropResizeSegment`.
- `DatasetWriter.write` documents that split iteration must happen once each, in order (shared lazy sample stream).
- Backend replay parity fixes: Kornia affine composition/quarter-turn direction, Albumentations keep-size perspective scaling, TorchVision rotation convention.
- Blur-commute singular-value guard now tolerates float32 rounding (`1e-6`); `clip_policy="per_op_parity"` contrast midpoint now recomputed from the clamped intermediate.

### Changed

- `compile=True` also wraps the fused color-matrix and lookup-table applies in their own `torch.compile` regions.
- Base dependency audit: dropped then reinstated `pillow` (needed by the data generator); dropped unused `rich`/`scipy`. CI now covers Python 3.11–3.14, including a previously-missing 3.12 leg.
- Pipeline pickling rebuilds derived dispatch attributes on unpickling instead of only at construction; `fusion_plan` reports backend-boundary `split_reason` and marks cv2 fused/projective segments as CPU-passthrough on non-CPU pipelines.

### Performance

- Oriented box now derives from the polygon on first access instead of at generation time — 75% of generation time on a mixed-family run (311ms → 56ms over 538 objects). Deleted outright in the next cycle's upright-frame rewrite.
- Several host-sync/FLOP reductions: skipped exact-D4 device readback off-CPU, `baddbmm` color-matrix apply (~25% fewer FLOPs), closed-form `normalize_matrix_io`, a batch-size sentinel avoiding a `.item()` sync, redundant `.copy()` removed from cv2 warp paths.

## [0.9.0.dev0] - 2026-07-11

Pluggable adapter registry, exact D4 execution, crop+resize fusion, and multi-target routing through Albumentations; opt-in `compile`/`antialias`/`clip_policy`/`mask_interpolation`.

### Added

- Pluggable adapter registry: public `register_adapter()` plus the `fuse_augmentations.adapters` entry-point group (experimental); `Compose.supported_ops(backend)` and `Compose.capability_matrix()` report config-time op coverage, and `from_config` aggregates all invalid specs in one error.
- Exact execution for composed flip / quarter-turn (90°/180°/270°) chains: dispatched via `tensor.flip`/`rot90` with zero interpolation error; auxiliary targets (masks, boxes, keypoints) fall back to the grid path automatically instead of raising.
- Crop+resize fusion: a geometric chain followed by `RandomResizedCrop` now fuses into a single warp at the target output size.
- `execution="cv2" | "torch"` flag on `Compose` for fused Albumentations segments: `"cv2"` (default) keeps per-sample cv2 warps bit-identical to earlier releases; `"torch"` opts into one batched `grid_sample` per segment.
- Multi-target `data_keys` with Albumentations fused segments: masks, bounding boxes, and keypoints are routed through the composed pixel matrix (previously a construction-time `ValueError`).
- Albumentations-style keyword calls on multi-target pipelines (`pipe(image=..., mask=..., bboxes=...)`) return a dict keyed by the caller's keyword names. Colliding keyword aliases raise `ValueError`.
- `output_backend="numpy"` now converts each convertible target of a multi-target output (image, mask); coordinate targets remain tensors.
- `Normalize` (Kornia, TorchVision v2, standard Albumentations) now fuses into the color matrix as a per-channel affine; the final gamut clamp is suppressed for normalized output.
- `clip_policy="final" | "per_op_parity"` on `Compose`: `"final"` (default) clamps once after the fused color matmul; `"per_op_parity"` splits the fused run wherever an intermediate would leave `[0, 1]`.
- Opt-in `compile=True`: wraps the warp core in `torch.compile` on torch ≥ 2.2. Opt-in `antialias=True`: crop-resize segments prefilter aggressive downscales. Opt-in `substitute_passthrough=True`: replaces registered non-fusible ops with an installed backend's torch-native equivalent (initially Albumentations `GaussianBlur` → Kornia `RandomGaussianBlur`); warns per substitution.
- Passthrough segments now cross the CPU boundary once per batch instead of per sample, with identical numerics.
- `fusion_plan` marks passthrough entries with `[CPU passthrough]` on non-CPU pipelines; `fusion_plan_descriptors` carries machine-readable `split_reason`/`barrier`/`refused` fields.
- Opt-in `mask_interpolation="bilinear"`: differentiable soft-mask sampling for auxiliary masks. Default `"nearest"` unchanged.
- Memory benchmark (`experiments/bench_memory.py`): peak memory and allocation counts, fused vs. native, per pipeline and batch size.
- `backend="native"` is now a first-class option for `from_config`: the zero-dependency, fully batched pure-torch engine. Opt-in — auto-detection remains the default.
- `return_matrix=True` per-call flag: returns `(output, matrix)` without reading shared instance state, making matrix retrieval thread-safe.
- One `finfo(dtype).eps`-scaled near-singular threshold shared by all three matrix-inversion paths; `fusion_plan`/`fusion_plan_descriptors` results are cached (device-aware, pickle-safe).

### Fixed

- Corrupt rotation matrix in the TorchVision batch-size-1 CPU cv2 fast path (`sin` computed as `cos`).
- `from_params(scale=...)` now draws a single isotropic factor shared by both axes, as documented; explicit `scale_x`/`scale_y` keep independent draws.
- cv2 `"reflection"` padding now maps to `BORDER_REFLECT_101`, matching torch `grid_sample(padding_mode="reflection", align_corners=True)`.
- Bounding-box zero-`w` guard uses `finfo.eps` (the previous `finfo.tiny` clamp overflowed float32 to `inf`).
- Near-singular affine matrices raise consistently across the torch and cv2 inversion paths.
- cv2 fast-path activation gates respond to `torch.manual_seed`; Albumentations segment `forward` no longer consumes RNG draws for inactive transforms.
- `uint16` NumPy inputs are normalised to `[0, 1]` (previously cast without rescaling).
- Albumentations native dict path raises instead of silently dropping non-image keys; unrecognised transforms are rejected in Albumentations-backed pipelines.
- `transform_matrix` resets to `None` at every forward, so exact/passthrough-only calls no longer report a stale matrix.
- `fuse_aug.__version__` is exported, matching `fuse_augmentations.__version__`.

### Changed

- **Fused contrast midpoint is now the per-image mean luminance** (matching native TorchVision/Kornia `ColorJitter` semantics) instead of a fixed `0.5`. Pin previous behavior only by comparing against your own stored baselines.
- Coordinate-changing passthrough ops (elastic/grid/optical distortion) now **raise `ValueError`** in a multi-target pipeline (previously a `UserWarning`). Kernel/pointwise passthrough (blur, noise) no longer warns — skipping them is correct.
- `same_on_batch=True` on Albumentations-backed fused segments now shares the sampled parameters across the batch, not just the activation decision.
- Documented color-fusion accuracy caveats (final-only clamping; fixed 0.5 contrast midpoint) and the seeding contract limits between warp backends.

## [0.8.0.dev0] - 2026-05-14

Single-op fast paths and a native Albumentations-dict I/O path.

### Added

- `FusedCompose.__call__` gains a native Albumentations-dict input/output fast path (`_forward_albu_native`).

### Fixed

- `RandomSaturation` and `HueSaturationValue` are now registered as `POINTWISE` (previously misclassified, bypassing color fusion).

### Changed

- CI gains a matrix strategy exercising the optional Kornia/TorchVision/Albumentations extras independently.

### Performance

- Single-op fast paths for `FusedAffineSegment` (Kornia/TorchVision) and the Albumentations numpy path skip matrix reconstruction and `grid_sample`/`cv2.warpAffine` entirely for one-transform chains, and bypass `nn.Module.__call__` in favor of direct `.forward()` dispatch.
- `matmul3x3` moved to `torch.bmm` and eager `inv3x3` to `torch.linalg.inv` (~150x and ~6x faster per call, measured); the Albumentations numpy path gained a closed-form Cramer's-rule 3x3 inverse and an `np.flip` bypass for pure flip chains, replacing `scipy.ndimage` with `cv2.warpAffine`.
- Fused sample+build helpers combine parameter sampling and matrix construction into one call on the Kornia and TorchVision cv2 fast paths.
- Pre-allocated matrix buffers, cached identity matrices, and pre-classified segment-dispatch tags remove per-call allocations and `isinstance` checks from the hot forward path.
- `experiments/optimize_score.py` grew from a 15-case to a 45-case benchmark with a computed theoretical-target ceiling per case; individual optimizations pinned to measured deltas against it.

## [0.7.0.dev0] - 2026-03-28

Declarative `from_config()` construction, crop-resize fusion, and color-matrix fusion.

### Added

- `Compose.from_config()` classmethod, backed by a backend resolver, an op-name registry, and a frozen `TransformSpec` dataclass.
- `output_backend` parameter on `Compose.__init__` for cross-backend output conversion (`NumpyToTorchConverter`/`TorchToNumpyConverter`, `BackendConverter` protocol).
- `CROP_RESIZE_FIXED` op category and `CropResizeSegment`, with adapter registrations across all three backends.
- `POINTWISE_LINEAR` color fusion: `build_color_matrix` per adapter, `FusedColorSegment`, `reorder_pointwise`/`build_segments` integration.
- `ReorderPolicy.AGGRESSIVE`, and extended `GEOMETRIC_EXACT` dispatch with an `exact_apply` protocol method.
- `fusion_plan_descriptors` property (`SegmentDescriptor` dataclass) and a `backend=` kwarg on `from_params` for full-parity delegation.

### Changed

- `ExactSegment` renamed to `ExactAffineSegment` (deprecation alias kept); expanded Kornia and Albumentations adapter coverage (`SafeRotate`, `RandomShear`, `RandomTranslate`).

### Fixed

- Aux-target corruption, batch-randomness, and backend-attribution bugs found across the review cycle; `_d4_matrix` now guards shape-changing D4 elements on non-square images.

## [0.6.0.dev0] - 2026-03-20

Fused perspective/projective warp chains.

### Added

- `ProjectiveSegment` and `AlbuProjectiveSegment` for fused perspective-warp chains, with perspective division applied to auxiliary targets (masks, boxes, keypoints).
- `RandomPerspective` / `Perspective` registered across all three adapters, wired into `Compose` via `ProjectiveSegment`.
- `PROJECTIVE` op-category enum and perspective matrix utilities.

## [0.5.0.dev0] - 2026-03-20

TorchVision adapter and mixed-backend pipelines.

### Added

- `TorchVisionAdapter` for TorchVision v1 and v2 transforms, wired into `Compose` dispatch.
- Mixed-backend restriction lifted: pipelines can now mix adapters per transform, dispatched individually.

### Fixed

- `RandomAffine` matrix composition corrected to match TorchVision semantics; TorchVision v2 batch semantics fixed.
- `id()`-keyed adapter map replaced with a stable lookup (fixes pickle-stability of passthrough adapter dispatch); `Backend.UNKNOWN` handling clarified.

## [0.4.0.dev0] - 2026-03-19

Albumentations adapter.

### Added

- `AlbumentationsAdapter` implementing the `TransformAdapter` protocol, wired into `Compose` and segment dispatch.
- `NumpyFusedAffineSegment` for the Albumentations (cv2) backend, plus `_np_matrix.py` matrix builders for `hflip`/`vflip`.

### Changed

- Affine engine restructured into an `affine/` subpackage; `cv2` replaced with `scipy` in the shared matrix path.

### Fixed

- `torch.from_numpy` incompatibility with NumPy 2.x.

## [0.3.0.dev0] - 2026-03-18

Auxiliary-target routing and `from_params()`.

### Added

- `data_keys` routing and auxiliary-target (mask/bbox/keypoint) transform helpers (`_targets.py`), wired through segments.
- `Compose.from_params()` classmethod.

### Fixed

- `transform_mask` now supports integer masks with dtype preservation (previously float32-only).
- Duplicate `data_keys` handling in the forward loop.

## [0.2.0.dev0] - 2026-03-18

Lossless exact-flip chains.

- `ExactSegment` for lossless flip-only chains, dispatched via `build_segments` detection of EXACT-only op chains.
- `ReorderPolicy.POINTWISE` reordering support.
- `same_on_batch` support verified and extended in `KorniaAdapter`.
- `FusedCompose` renamed and reworked for `Protocol` conformance.

## [0.1.0.dev0] - 2026-03-17

Initial release: fused-affine `Compose` orchestrator and the Kornia adapter.

- Initial `Compose` orchestrator (`fusion_plan`, `n_warps_saved`, `transform_matrix`) wired to `FusedAffineSegment` and `build_segments`.
- `KorniaAdapter` with corrected shear/rotation sign conventions.
- Core primitives: `TransformCategory`, `ReorderPolicy`, `InterpolationMode`, `PaddingMode`, `TransformAdapter` protocol (`_types.py`); matrix primitives `matmul3x3`, `inv3x3`, `normalize_matrix` (`_matrix.py`); interpolation and backend resolution (`_interpolation.py`, `_backend.py`).
- `fuse_aug` re-export package and public `__all__` surface.
- Test infrastructure scaffold.
