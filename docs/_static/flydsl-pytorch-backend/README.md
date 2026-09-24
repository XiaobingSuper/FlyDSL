# PyTorch FlyDSL blog figures and benchmark data

This directory contains the architecture overview, five operator-performance
figures, and one end-to-end vLLM figure for
[the blog](../../pytorch_flydsl_backend_blog.md), along with their source data
and rendering material. Regenerating the operator figures does not run a
benchmark.

## Publication scope

The September 24 revision assumes a PyTorch build containing
[MXFP scaled GEMM support, PR #196719](https://github.com/pytorch/pytorch/pull/196719).
The article treats that PR as landed. Support and usage were checked against
commit `e297267ba4a23ee9e14d35d12145d7deb4d8fcd1`; the MXFP operator table uses
the final numbers published in the benchmark comment last edited on
September 20.

The MXFP implementation and tests cover NN, NT, TN, and TT layouts, with
layout-dependent alignment checks. Both formats require logical `K` to be a
multiple of 128, contiguous unswizzled E8M0 block scales, zero storage offsets,
and FP16/BF16 output without fast accumulation. An optional one-dimensional bias
is supported when its length is `N` and its dtype matches the output. Eligible NT
shapes can have M/N tails. All published MXFP benchmark shapes satisfy the
current requirements.

## Architecture overview

The diagram follows an application through two execution modes on AMD
MI350-series GPUs. It assumes the optional FlyDSL package is installed and the
compiled GEMM path has FlyDSL and autotuning enabled:

- Eager RMSNorm and TopK dispatch to FlyDSL for eligible inputs and retain ATen
  for unsupported inputs.
- TorchInductor selects the fastest measured eligible implementation. Dense and
  grouped GEMM can use enabled ATen, Triton, and FlyDSL candidates; MXFP scaled
  GEMM benchmarks all eligible candidates, including ATen where supported.

The diagram shows the public operations and selection policies. Compiler
internals and caches are omitted to keep the overview focused on application
behavior. Its source is the `architecture()` function in
[generate_figures.py](generate_figures.py).

## Sources

The original four operator source tables were retrieved on September 9, 2026.
The MXFP tables were refreshed on September 21 from the benchmark comment
updated on September 20. CSV values preserve the precision of the published
latency or throughput columns. Every case in each source suite is included; no
cases are removed based on performance.

| Data | Cases | Published source | Measurement setup |
| --- | ---: | --- | --- |
| [dense_gemm.csv](dense_gemm.csv) | 15 | [Dense GEMM benchmark comment](https://github.com/pytorch/pytorch/pull/190903#issuecomment-5061510962) | BF16 NT on MI355 (`gfx950`); graph replay; median of four accuracy-checked runs; FlyDSL/Triton `EXHAUSTIVE`, ATen default |
| [mxfp_gemm.csv](mxfp_gemm.csv) | 34 | [MXFP8/MXFP4 benchmark comment](https://github.com/pytorch/pytorch/pull/196719#issuecomment-5632055416) | MI355X (`gfx950`); NT layout, 17 shapes per format; FlyDSL versus ATen |
| [grouped_gemm.csv](grouped_gemm.csv) | 24 | [Grouped GEMM PR](https://github.com/pytorch/pytorch/pull/194032) | BF16 on `gfx950`; 14 uniform-group shapes, five K/N variants at `G=8, M=512`, and five ragged-M cases at `G=8, K=N=4096`; isolated process and fresh cache per backend/shape; steady-state TFLOP/s; output checked against eager |
| [rmsnorm.csv](rmsnorm.csv) | 22 | [RMSNorm PR](https://github.com/pytorch/pytorch/pull/191447) | FP16/BF16/FP32 on MI355X; FlyDSL 0.3.0; GPU events, 10 warmup and 50 timed iterations; one run per case |
| [topk.csv](topk.csv) | 33 | [TopK PR](https://github.com/pytorch/pytorch/pull/193548) | FP32 on MI355X; GPU events, 20 warmup and 100 timed iterations; median of three runs for each determinism setting |
| [vllm_e2e.csv](vllm_e2e.csv) | 18 BF16 + 18 MXFP8 | [End-to-end A/B report](vllm-bf16-mxfp8-report.html) | `vllm bench serve` on one MI355X; three models; concurrency 8–256; ISL 256, OSL 512; TP=1; fixed per-model KV cache |

The suites are separate experiments, not a single run under one common software
environment. Only RMSNorm's source specifies FlyDSL 0.3.0 for the reported table;
the CSV files do not infer missing version, clock, or power-setting information
from other suites. Compilation, first-call autotuning, and end-to-end model
latency are outside these measurements.

The MXFP source specifies supported output types but does not identify which
output dtype produced the tables, or provide timing iteration counts and a full
software version list. The figure therefore labels the operand formats and
hardware without assigning an output dtype or importing another suite's timing
protocol. Quantization and end-to-end model costs are not characterized by
these throughput tables.

The end-to-end report compares Llama-3.1-8B-Instruct, Qwen3-32B, and
Llama-3.3-70B-Instruct under fixed-length generation (`--ignore-eos`), with
`max_num_batched_tokens=2048`, piecewise CUDA graphs, and prefix caching
disabled. BF16 compares `ATEN,TRITON` with `ATEN,TRITON,FLYDSL`, uses at least
two ABBA-interleaved repetitions, and reports medians. MXFP8 compares ATen with
ATen plus FlyDSL; each cell has one measured run after a separate warmup. The
blog uses the report's whole-request latency results; the report also contains
output throughput, TPOT, and TTFT results. `vllm_e2e.csv` transcribes the
whole-request speedups used to rebuild the publication figure in the common
visual style.

## Calculations

- For latency data, speedup is `baseline_us / flydsl_us`.
- For throughput data, speedup is `flydsl_tflops / baseline_tflops`.
- Dense GEMM uses the faster baseline at each shape:
  `flydsl_tflops / max(aten_tflops, triton_tflops)`.
- A geometric mean is `exp(mean(log(speedup)))`, with equal weight per case.
  MXFP8 and MXFP4 are each aggregated over all 17 shapes relative to ATen.
  Grouped GEMM is aggregated separately over 14 uniform-group, five K/N-variant,
  and five ragged-M cases. The article reports geometric means against ATen and
  Triton separately; across all 24 cases these are 1.9505x and 1.2000x,
  respectively. The chart shows every case against its faster baseline.
  TopK uses `M` for row count, `N` for row width, and `K` for selected elements;
  it is aggregated separately by K band and determinism setting, with 10 register
  cases and 6/6/6/5 radix cases.
- TopK compares FlyDSL and ATen under the same
  `torch.use_deterministic_algorithms` setting. The deterministic radix path
  preserves finite-value tie order; the nondeterministic path uses atomic slot
  allocation. The register path is reproducible but can order ties differently
  from ATen.
- Dense GEMM counts ratios in `[0.99, 1.01]` as ties. All 15 shapes remain in
  the chart and geometric mean, including the three ties and two losses.
- RMSNorm shows all cases individually on a common scale, split into aligned
  hidden dimensions and dimensions one element above an aligned size.
- MXFP panels show all 17 shapes per format on the same scale. The geometric
  means are 1.5755x versus ATen for MXFP8 and 1.6804x for MXFP4. These NT
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
dense GEMM, MXFP scaled GEMM, grouped GEMM, RMSNorm, TopK, and end-to-end vLLM
charts, and prints the calculated performance aggregates. SVG text remains
editable. The self-contained report remains the source for all four vLLM
metrics and the detailed test configuration.

The older `flydsl-mxfp8-performance` assets are retained for reference but are not
used in the publication article. They describe the earlier
[scaled GEMM prototype](https://github.com/pytorch/pytorch/pull/193527).
The current `flydsl-mxfp-gemm-results` figure is generated from the 34
measurements published on PR #196719.
