# Experiments

Benchmark and optimization scripts used during development to compare `vision-synth` with native Albumentations, Kornia, and TorchVision pipelines. These are developer tools outside the installed package. Results are specific to the measured workload and environment; see the [benchmark methodology](../docs/research/methodology.md) before citing them.

Install the optional backends and benchmark dependency group, then run a script:

```bash
uv sync --all-extras --group benchmark
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py --quick
```

`bench_augmentation_pipelines.py` and `bench_primitive_vs_affine.py` use [Jupytext percent-format](https://jupytext.readthedocs.io/) `# %%` cells. They also run as plain Python scripts. JupyterLab is in the benchmark group; Jupytext is not declared by the project, so install it separately if you want to convert either script to a notebook. The other experiment scripts are plain Python without notebook cells.

Most scripts write host-specific scratch results under ignored `experiments/results/`; `optimize_score.py` prints its score and writes JSON only when given `--details-json`. `bench_albu_preparation.py` requires an output path. The sample output below is historical and is not a performance claim for the current revision or another machine.

## Files

| File                              | Purpose                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `optimize_score.py`               | Fixed 45-case CPU score; median of three full passes by default.                         |
| `bench_augmentation_pipelines.py` | CPU latency across 28 sequences, three backends, and native/fused modes; visual figures. |
| `bench_primitive_vs_affine.py`    | Dedicated primitive versus generic affine cost and chain amortization.                   |
| `bench_gpu_batch.py`              | CPU/CUDA/MPS latency and throughput by batch size where supported.                       |
| `bench_memory.py`                 | Action-aware tensor memory and allocation counters by device and batch size.             |
| `bench_rfdetr_shape.py`           | Detector-shaped CPU image/box endpoint across four reproducible variants.                |
| `bench_albu_preparation.py`       | Albumentations matrix preparation and image/box endpoint comparison.                     |
| `bench_antialias.py`              | CPU cost of opt-in antialias filtering and its scale estimate.                           |

Run the three focused CPU probes with:

```bash
uv run --all-extras --group benchmark python experiments/bench_rfdetr_shape.py
NO_ALBUMENTATIONS_UPDATE=1 uv run --all-extras --group benchmark python experiments/bench_albu_preparation.py experiments/results/preparation.json --revision YOUR_SOURCE_REVISION
uv run --all-extras --group benchmark python experiments/bench_antialias.py
```

The preparation probe uses private implementation details and requires an output path plus a revision label. Its results, like the other probes, need the recorded source revision and environment to be interpreted.

______________________________________________________________________

## `optimize_score.py` — composite optimization metric

```bash
uv run --all-extras --group benchmark python experiments/optimize_score.py  # three complete scores by default
uv run --all-extras --group benchmark python experiments/optimize_score.py --repetitions 10  # PR gate setting
```

Times 45 cases (single-op baselines, pure-geometric chains, mixed geo+colour chains under aggressive reordering) across Kornia, TorchVision, and Albumentations, each native vs. fused. It prints the geometric mean of the 45 native/fused latency ratios and `theoretical_target`, the geometric mean of each case's geometric operation count. That target is an operation-count reference, not a speed ceiling. `--repetitions` repeats all 45 cases in one process (default 3) and reports the median score; individual scores go to standard error. Optional `--details-json PATH` records per-case timings. Standard output stays two lines.

**Sample output (short run):**

```
real_score=1.7601
theoretical_target=2.3752
```

This sample is a historical single-run score for one fixed CPU case bank, not a current or typical-user speedup. The `2.3752` value counts geometric operations in that bank and cannot bound runtime performance. The [methodology](../docs/research/methodology.md#interpret-the-fixed-bank-score) records a separately dated historical `1.7861x` result and the missing controls needed for a comparison claim.

## `bench_augmentation_pipelines.py` — full pipeline comparison

```bash
uv run --all-extras --group benchmark python experiments/bench_augmentation_pipelines.py
```

Runs 28 sequences (single-op `a*`, geometric `b*`, colour `c*`, mixed `d*` with `__pw`/`__agr` reorder variants) × 3 backends × native/fused = 168 timed benchmarks, then renders one visual-sanity PNG per sequence (native vs. fused output side by side, with a `max|native-fused|` diff annotation) and writes `experiments/results/benchmark_results.json`.

**Sample summary table (short run, 100 reps):**

```
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Sequence                          alb             |           kornia            |             tv
                      native ;   fused ;  boost   | native ;   fused ;  boost   | native ;   fused ;  boost
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
a01_rotate             0.224 ;   0.199 ;  x1.13 ✔ |  0.450 ;   0.480 ;  x0.94 ≈ |  0.726 ;   0.766 ;  x0.95 ≈
b02_geom_3             0.238 ;   0.192 ;  x1.24 ✔ |  1.126 ;   0.211 ;  x5.34 ✔ |  1.938 ;   0.214 ;  x9.05 ✔
b05_geom_5_warp        1.097 ;   0.260 ;  x4.23 ✔ |  4.295 ;   0.329 ; x13.05 ✔ |  3.626 ;   0.230 ; x15.78 ✔
c02_color_3            0.234 ;   0.239 ;  x0.98 ≈ |  3.406 ;   3.687 ;  x0.92 ≈ |  0.718 ;   0.726 ;  x0.99 ≈
d03_mixed_g4c3__agr    0.728 ;   0.465 ;  x1.57 ✔ |  4.381 ;   3.597 ;  x1.22 ✔ |  2.206 ;   0.924 ;  x2.39 ✔
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

(full table has 28 rows; `a*`/`c*` single-op and colour-only cases hover near 1×, `b*` pure-geometric chains show the largest wins since every fused chain collapses to one `grid_sample` regardless of length, `d*` mixed chains land in between depending on colour-op cost and reorder policy.)

Before the table, the script also prints one line per fused sequence/backend showing the chosen fusion plan, e.g.:

```
d03_mixed_g4c3__agr  albumentations  fused  →  fused(Rotate, HorizontalFlip, VerticalFlip, Rotate) →
  color(RandomBrightnessContrast, RandomBrightnessContrast) → passthrough(HueSaturationValue)  [4 warps saved]
```

**Illustration** — `experiments/results/visual_b02_geom_3.png` (Rotate, HFlip, Scale; all three backends, native on top and fused below). The panels display `max|native-fused|` as a visual diagnostic. Resetting seeds makes rows reproducible, but does not prove the paths sampled identical geometry or establish numeric parity:

![b02_geom_3 native vs fused](results/visual_b02_geom_3.png)

## `bench_primitive_vs_affine.py` — primitive vs. generic Affine

```bash
uv run --all-extras --group benchmark python experiments/bench_primitive_vs_affine.py
```

For each backend, times a dedicated primitive (`A.Rotate`, `K.RandomRotation`, …) against the backend's generic `Affine`/`RandomAffine` configured to the same effect. A ratio ≈ 1.0 means fuse-aug can freely route that op through `Affine` (and therefore fuse it into a chain) at no per-op cost; a ratio ≫ 1.0 means the dedicated primitive is meaningfully cheaper and the fused path pays a tax for single ops of that kind. Also times 2–6 op chains as dedicated primitives vs. one combined `Affine` call, which is the actual saving fusion delivers.

**Sample output (short run, 100 reps, `—` = backend has no dedicated op or no Affine equivalent):**

```
                            alb                     kornia                        tv
                          prims   affine    ratio    prims   affine   ratio    prims   affine    ratio
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  Geometric primitives
    HFlip                 0.023    0.213    9.27×    0.097        —       —    0.050        —        —
    Rotate 30°            0.221    0.225    1.02×    0.406    0.494   1.22×    0.733    0.720    0.98×
    Scale                 0.084    0.216    2.58×        —    0.610       —        —    0.877        —

  Multi-op sequence
    2-op chain            0.329    0.221    0.67×    1.286    0.510   0.40×    0.804        —        —
    5-op chain            0.577    0.223    0.39×    3.126        —       —    1.675        —        —
    6-op chain            0.659    0.228    0.35×    3.106        —       —    1.644        —        —
```

Reading this: Albumentations' `Affine` costs a near-fixed ~0.22 ms regardless of op count, so an N-op Albumentations chain routed through one `Affine` call gets cheaper (relative to N dedicated-primitive calls) as N grows — exactly the fusion win `FusedAffineSegment` exploits. Cheap ops with no dedicated primitive at all (`HFlip`/`VFlip` are near-free in Albumentations, ~0.01–0.02 ms) show a high single-op ratio, which is why fuse-aug keeps exact-primitive shortcuts for those rather than always routing through `Affine`.

Results: `experiments/results/bench_primitive_vs_affine.json`.

## `bench_gpu_batch.py` — device × batch-size sweep

```bash
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py            # full sweep
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py --quick     # fast smoke run
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py --batch-sizes '[1,8,32]'
```

Sweeps CPU (always) plus CUDA/MPS (auto-detected) at batch size 1 and 8 for a representative subset of sequences, reporting median/p10/p90 latency and throughput (img/s) for native vs. fused. Correct per-device synchronization (`torch.cuda.synchronize`/`torch.mps.synchronize`) is applied before/after timing so the numbers reflect real device execution, not async dispatch. Native Albumentations is CPU/NumPy-only, so it's skipped (recorded, not silently dropped) on `cuda`/`mps` device rows.

**Sample output (`--quick`, batch 8, CPU):**

```
| d03_mixed_g4c3__agr | kornia         | native | cpu | 8 | 10.430 | 9.900 | 11.305 | 767  | — |       |
| d03_mixed_g4c3__agr | kornia         | fused  | cpu | 8 |  8.387 | 8.125 |  8.883 | 954  | — | 1.24x |
| e01_geo_crop_fuse   | kornia         | native | cpu | 8 |  3.831 | 3.707 |  4.137 | 2088 | — |       |
| e01_geo_crop_fuse   | kornia         | fused  | cpu | 8 |  1.888 | 1.771 |  1.963 | 4237 | — | 2.03x |
```

**Sample output (`--quick`, batch 1, MPS):**

```
| b05_geom_5_warp | kornia      | native | mps | 1 | 20.080 | 16.281 | 20.331 |  50 | — |       |
| b05_geom_5_warp | kornia      | fused  | mps | 1 | 10.719 |  9.849 | 11.854 |  93 | — | 1.87x |
| b05_geom_5_warp | albumentations | native | mps | 1 | — | — | — | — | — | — | skip: native Albumentations is NumPy/CPU-bound; no GPU path |
```

`boost` (last numeric column) is throughput fused/native; `>1x` = fused faster. A quick smoke run reported `154 ok, 14 skipped` and wrote `experiments/results/bench_gpu_batch_darwin_arm64.json`.

## `bench_memory.py` — peak memory & allocation count

```bash
uv run --all-extras --group benchmark python experiments/bench_memory.py            # full sweep
uv run --all-extras --group benchmark python experiments/bench_memory.py --quick     # fast smoke subset
uv run --all-extras --group benchmark python experiments/bench_memory.py --json      # also write JSON
uv run --all-extras --group benchmark python experiments/bench_memory.py --devices '["cpu"]' --batch-sizes '[1,8]'
```

Measures memory counters on a sequence/device/batch sweep. CPU Torch profiling records live and incremental tensor peaks, preexisting baseline, and physical allocation events. CUDA uses peak allocator stats; MPS current allocation is a snapshot, not a transient peak. Unavailable counters are reported as null with an error. These metrics do not cover every allocator or total process memory; see [memory methodology](../docs/research/methodology.md#measure-memory-responsibly).

The [corrected CPU tensor-memory sweep](../docs/research/benchmarks.md#corrected-cpu-tensor-memory-sweep-september-6-2026) measured all 72 rows on September 6, 2026. Representative three-operation rows from that recorded environment are:

| Backend / batch | Native / fused live tensor peak (MiB) | Native / fused CREATE count |
| --------------- | ------------------------------------- | --------------------------- |
| Kornia / 1      | 2.751 / 1.500                         | 271 / 28                    |
| Kornia / 8      | 21.002 / 16.002                       | 429 / 260                   |
| TorchVision / 1 | 4.250 / 2.250                         | 35 / 23                     |
| TorchVision / 8 | 30.500 / 16.002                       | 35 / 231                    |

The earlier `117.5 MB → 38.0 MB` TorchVision ratio and other historical peak/allocation ratios are **withdrawn** because the original profiler timeline accounting was incorrect. The corrected table measures Torch tensor timeline events on CPU only. The profiler also warned about an allocation predating profiling whose size was unknown. No corrected CUDA/MPS memory result is included here.

## Notes

- Seeded visual rows are reproducible within each path. Native and fused paths can consume random draws differently; the timing cells do not replay paired geometry. A correctness claim needs matched transform parameters or matrices before comparing pixels.
- `optimize_score.py`'s 45-case bank and `bench_augmentation_pipelines.py`'s 28-case bank overlap but aren't identical; see `program.md` for the optimization-campaign context these scripts were built for.
- `bench_gpu_batch.py` and `bench_memory.py` import their sequence bank from `optimize_score.py` when available, falling back to an inline copy — console output notes which provenance was used.
