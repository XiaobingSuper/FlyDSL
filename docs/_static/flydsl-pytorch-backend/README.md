# PyTorch FlyDSL blog figures and benchmark data

This directory contains the architecture overview and five operator-performance
figures for [the blog](../../pytorch_flydsl_backend_blog.md), along with their
rendering script. The performance charts use the published measurements below;
the input data and methodology are recorded here. Regenerating the figures does
not run a benchmark.

## Publication scope

The September 15 revision is prepared for a PyTorch build containing
[MXFP scaled GEMM support, PR #196719](https://github.com/pytorch/pytorch/pull/196719).
The article assumes that PR has landed, as requested for the publication draft;
it was still open when reviewed. Support and usage were checked against commit
`126f5f30bb54d1b754659c6fd875e767caf16455`.

The MXFP implementation and tests cover NN, NT, TN, and TT layouts, with
layout-dependent alignment checks. Both formats require logical `K` to be a
multiple of 128, contiguous unswizzled E8M0 block scales, zero storage offsets,
and FP16/BF16 output without bias or fast accumulation. Eligible NT shapes can
have M/N tails. The benchmark comment's contract summary lists a broader K
granularity for MXFP8; the article follows the implementation's eligibility
checks. All published MXFP benchmark shapes satisfy the current requirements.

## Architecture overview

The diagram follows an application through two execution modes on AMD
MI350-series GPUs. It assumes the optional FlyDSL package is installed and the
compiled GEMM path has FlyDSL and autotuning enabled:

- Eager RMSNorm and TopK dispatch to FlyDSL for eligible inputs and retain ATen
  for unsupported inputs.
- TorchInductor selects the fastest measured eligible implementation. Dense and
  grouped GEMM can use enabled ATen, Triton, and FlyDSL candidates; MXFP scaled
  GEMM compares FlyDSL with an ATen candidate. The separately measured CK and
  AITER Triton kernels are comparison baselines, not additional candidates in
  the integrated MXFP path.

The diagram shows the public operations and selection policies. Compiler
internals and caches are omitted to keep the overview focused on application
behavior. Its source is the `architecture()` function in
[generate_figures.py](generate_figures.py).

## Sources

The original four operator source tables were retrieved on September 9, 2026;
the MXFP tables were retrieved on September 15. CSV values preserve the precision
of the published latency or throughput columns. Every case in each source suite
is included; no cases are removed based on performance.

| Data | Cases | Published source | Measurement setup |
|---|---:|---|---|
| [dense_gemm.csv](dense_gemm.csv) | 15 | [Dense GEMM benchmark comment](https://github.com/pytorch/pytorch/pull/190903#issuecomment-5061510962) | BF16 NT on MI355X (`gfx950`); graph replay; median of four accuracy-checked runs; FlyDSL/Triton `EXHAUSTIVE`, ATen default |
| [mxfp_gemm.csv](mxfp_gemm.csv) | 34 | [MXFP8/MXFP4 benchmark comment](https://github.com/pytorch/pytorch/pull/196719#issuecomment-5632055416) | MI355X (`gfx950`); NT layout, 17 shapes per format; MXFP8 vs ATen/CK, MXFP4 vs ATen/AITER Triton; CK graph timing includes dynamic A-scale shuffling and excludes one-time static B-scale preprocessing |
| [grouped_gemm.csv](grouped_gemm.csv) | 24 | [Grouped GEMM PR](https://github.com/pytorch/pytorch/pull/194032) | BF16 on `gfx950`; isolated process and fresh cache per backend/shape; steady-state TFLOP/s; output checked against eager |
| [rmsnorm.csv](rmsnorm.csv) | 22 | [RMSNorm PR](https://github.com/pytorch/pytorch/pull/191447) | FP16/BF16/FP32 on MI355X; FlyDSL 0.3.0; GPU events, 10 warmup and 50 timed iterations; one run per case |
| [topk.csv](topk.csv) | 33 | [TopK PR](https://github.com/pytorch/pytorch/pull/193548) | FP32 on MI355X; GPU events, 20 warmup and 100 timed iterations; median of three runs for each determinism setting |

The suites are separate experiments, not a single run under one common software
environment. Only RMSNorm's source specifies FlyDSL 0.3.0 for the reported table;
the CSV files do not infer missing version, clock, or power-setting information
from other suites. Compilation, first-call autotuning, and end-to-end model
latency are outside these measurements.

The MXFP source specifies supported output types but does not identify which
output dtype produced the tables, or provide timing iteration counts and a full
software version list. The figure therefore labels the operand formats and
hardware without assigning an output dtype or importing another suite's timing
protocol. CK uses pre-shuffled scales, so the comparison includes the dynamic
A-scale shuffle described by the source. The kernels compute equivalent scaled
matrix products with different scale layouts. Quantization and end-to-end model
costs are not characterized by these throughput tables.

## Calculations

- For latency data, speedup is `baseline_us / flydsl_us`.
- For throughput data, speedup is `flydsl_tflops / baseline_tflops`.
- Dense GEMM uses the faster baseline at each shape:
  `flydsl_tflops / max(aten_tflops, triton_tflops)`.
- A geometric mean is `exp(mean(log(speedup)))`, with equal weight per case.
  MXFP8 and MXFP4 are each aggregated over all 17 shapes, separately for ATen
  and the format's specialized baseline (CK or AITER Triton).
  Grouped GEMM is aggregated separately over 14 standard, five K/N-variant, and
  five ragged-M cases. TopK is aggregated separately by K band and determinism
  setting, with 10 register cases and 6/6/6/5 radix cases.
- Dense GEMM counts ratios in `[0.99, 1.01]` as ties. All 15 shapes remain in
  the chart and geometric mean, including the three ties and two losses.
- RMSNorm shows all cases individually on a common scale, split into aligned
  hidden dimensions and dimensions one element above an aligned size.
- MXFP panels show all 17 shapes per format on the same scale. The geometric
  means are 1.4350x vs ATen and 1.1078x vs CK for MXFP8, and 1.6027x vs ATen and
  0.9778x vs AITER Triton for MXFP4. Values below 1.0 remain visible. These NT
  measurements do not establish performance for NN, TN, or TT layouts.
- Chart axes start at zero. A dashed line at `1.0x` identifies the baseline.

Calculations use the published latency/throughput columns rather than their
separately rounded speedup columns. This can change the last displayed digit.
For example, the TopK register geometric mean with determinism off is 4.8157x,
displayed as **4.82x**; the earlier blog used **4.81x** from aggregating the
published speedup column. This is a rounding difference, not a new measurement.
The same convention is used consistently by the new figures and article.

The [all-layout GEMM extension](https://github.com/pytorch/pytorch/pull/194981)
reports a separate NT regression check: 1.206x versus the original Triton
reference, 1.165x versus the original ATen reference, and 1.115x versus the faster
original reference per shape. Its FlyDSL throughput is 1.07% higher in geometric
mean than the original FlyDSL run. These aggregate-only results are not used to
rescale the 15 individual measurements in `dense_gemm.csv`; no per-shape data or
NN/TN/TT performance numbers are inferred from them.

## Regenerating the figures

With Python 3, Matplotlib, and NumPy installed, run from the repository root:

```bash
python3 docs/_static/flydsl-pytorch-backend/generate_figures.py
```

The script writes PNG and SVG versions of the architecture overview and the
dense GEMM, MXFP scaled GEMM, grouped GEMM, RMSNorm, and TopK charts, and prints
the calculated performance aggregates. SVG text remains editable. All six
figures are used in the article.

The older `flydsl-mxfp8-performance` assets are retained for reference but are not
used in the publication article. They describe the earlier
[scaled GEMM prototype](https://github.com/pytorch/pytorch/pull/193527).
The current `flydsl-mxfp-gemm-performance` figure is generated from the 34
measurements published on PR #196719.
