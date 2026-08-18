# Exploring FlyDSL as a PyTorch Backend for ROCm Kernels

*This post reports ongoing work on an optional path for Python-authored,
architecture-specific ROCm kernels in eager PyTorch and `torch.compile`.*

> **Status (August 18, 2026):** This is a pre-publication prototype report for the
> [PyTorch RFC](https://github.com/pytorch/pytorch/issues/190875). The RMSNorm, TopK,
> dense GEMM, grouped GEMM, and scaled GEMM changes discussed below are in open or
> draft pull requests. They are not available as a complete feature in an official
> PyTorch release or nightly, and their APIs and performance may change before landing.

<!--
Publication checklist:
- Retitle as a launch post only after the operator/template PRs merge.
- Replace the status note with the first supported PyTorch nightly/release.
- Re-run one reproducible benchmark suite on the final merged revisions.
- Add author names and publication date in the PyTorch blog CMS.
-->

## Introduction

This article describes a proposed integration of
[FlyDSL](https://github.com/ROCm/FlyDSL) as an optional PyTorch kernel backend for AMD
GPUs. The work explores two independent extension points:

1. **Eager mode**, through targeted native operator overrides.
2. **`torch.compile`**, through TorchInductor templates and autotuning.

The prototype operator set is intentionally focused. The eager PRs implement RMSNorm
and TopK overrides. The TorchInductor PRs add FP16/BF16 dense GEMM, grouped GEMM, and
MXFP8/MXFP4 scaled GEMM templates. These kernels target AMD CDNA4 `gfx950` GPUs,
including the MI355X used for the measurements in this article.

The central design principle is **additive integration**. FlyDSL is not a required
PyTorch dependency and it does not replace ATen, Triton, Composable Kernel, CK-Tile,
or vendor libraries by policy. Unavailable or ineligible cases omit FlyDSL and retain
the existing PyTorch choices. Under TorchInductor, an eligible candidate is used only
if it wins autotuning.

## Why FlyDSL?

FlyDSL—Flexible Layout Python DSL—is a Python DSL and MLIR stack for authoring
high-performance GPU kernels with explicit layouts and tiling. A kernel author can
describe tensor shapes, strides, coordinate mappings, thread/value partitions, tiled
copies, LDS (AMD GPU shared memory) usage, and matrix fused multiply-add (MFMA)
instructions in Python while retaining control over the hardware execution hierarchy.
FlyDSL traces Python kernels into the Fly MLIR dialect, lowers through GPU/ROCDL and
LLVM AMDGPU, and emits an HSACO GPU code object.

This programming model is a useful match for performance-critical ROCm kernels.
High-level tensor semantics remain in Python, while architecture-specific choices such
as MFMA shapes, vectorized global-memory transfers, LDS staging, wave layouts, and
pipeline depth remain explicit and tunable.

FlyDSL also provides the runtime pieces needed by a framework integration: a HIP tensor
ABI, current-stream execution, JIT compilation, and artifact caching. The proposed
PyTorch integration keeps the compiler/runtime in the external package while placing
the reviewed PyTorch-facing wrappers and kernel snapshots in the PyTorch tree.

As with the [TorchInductor CuteDSL backend](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/),
the goal is not to generate every operation with a lower-level DSL. Existing PyTorch
backends already perform well over broad regions. A FlyDSL implementation should be
added only where an architecture-qualified kernel has a clear performance advantage
and a maintainable correctness and fallback story.

## Two Independent Integration Paths

Eager dispatch and TorchInductor selection solve different problems, so they remain
separate control planes. They share an optional-runtime policy and HIP tensor/stream
ABI, while routing, configuration, caches, tests, and rollback paths remain separate.
Enabling one path does not enable the other.

![FlyDSL integration in PyTorch](_static/flydsl-pytorch-backend/flydsl-pytorch-architecture.png)

*Figure 1. Eager dispatch and TorchInductor autotuning are independent selection
planes over the same optional FlyDSL compiler/runtime.*

For both paths:

- `import torch` does not import FlyDSL or initialize the ROCm runtime.
- PyTorch checks for a compatible optional FlyDSL 0.3.x package without eagerly
  importing it.
- PyTorch-facing wrappers and reviewed kernel snapshots live in the PyTorch tree;
  the compiler/runtime remains in the external FlyDSL package.
- Unavailable or ineligible calls omit the FlyDSL path and retain existing choices.
- Each integration path caches its own compiled specializations.

### Eager path

The eager PRs extend PyTorch's existing `torch._native` DSL mechanism. A cheap
predicate checks the support and performance region before lazily importing,
compiling, caching, and launching a FlyDSL kernel on the current ROCm stream. The
corresponding user control is exposed through
`torch.backends.python_native.flydsl`. Users can disable FlyDSL globally or in a
context manager without affecting TorchInductor.

### TorchInductor path

The compiler PRs add FlyDSL choices during ATen lowering, prune shape-incompatible
configurations, compile valid choices, and benchmark them with existing backends.
Tensor layouts and runtime values stay dynamic where possible, while tile shape,
pipeline depth, wave layout, and related parameters become compile-time
specializations.

## Benchmark Setup and Scope

The results below are **kernel-level measurements**, not end-to-end model results.
They come from separate pull requests rather than one unified benchmark run:

| Workload | Reported method | Cold compile/autotune included? |
|---|---|---|
| RMSNorm | 10 warmup, 50 GPU-event-timed iterations; one run per plotted row | No |
| TopK | 20 warmup, 100 timed iterations; median of three runs | No |
| Dense GEMM | Median of four accuracy-checked graph-replay runs | No |
| Grouped GEMM | Isolated process and fresh cache per shape; steady-state TFLOP/s | No |
| MXFP8 GEMM | 25 warmup, 100 timed iterations, four repeats; median graph throughput | No |

The RMSNorm, TopK, dense GEMM, and MXFP8 reports identify an AMD 256-CU MI355X
(`gfx950`); the grouped GEMM report identifies `gfx950` but does not record the exact
GPU model. ROCm, FlyDSL, and PyTorch revisions differ by pull request. Power mode,
clock policy, and repetition details were not reported consistently across all suites.
The linked PRs contain the available shape lists, commands, accuracy checks, and
version records. These results should therefore be compared only within each figure.
A publication-quality launch post should rerun the final merged revisions under one
fully specified environment and report cold compile, persistent-cache load, and
autotuning wall time separately.

## Eager Mode: Targeted Native Overrides

### RMSNorm

The RMSNorm PR is the first eager operator built on the proposed FlyDSL backend. It
accelerates the fused forward path; backward remains on the existing PyTorch
implementation. The kernel returns both the normalized output and the FP32 reciprocal
standard deviation required by the operator contract.

The initial support region is deliberately narrow:

- AMD `gfx950`;
- FP16, BF16, or FP32 input and weight;
- contiguous tensors with one normalized dimension;
- matching input/weight dtype and device;
- non-negative `eps`;
- measured `(M, N)` regions where the FlyDSL kernel wins.

N-dimensional inputs are logically flattened to `(M, N)`. The dispatcher predicate
checks the full contract before importing or compiling FlyDSL. Inputs outside the
support and performance gates use ATen.

The current performance gate covers the following flattened shapes:

| Normalized dimension `N` | Minimum row count `M` |
|---|---:|
| `4096 <= N < 8192` | `8192` |
| `8192 <= N < 16384` | `4096` |
| `16384 <= N <= 114688` | `2048` |

The kernel combines normalization, scaling, and output generation while using
architecture-specific reductions and vectorized loads. It also handles dimensions
that are not naturally aligned to the baseline kernel's preferred vector width. This
is especially helpful for the measured `N + 1` cases, where the speedup reaches 3.66x.

Across the representative measurements, aligned dimensions improve by 1.16x–1.54x
and off-by-one dimensions by 1.78x–3.66x. A separate local 60-case wall-to-sync suite
reported a **1.50x overall geometric-mean speedup**: 1.58x for FP16, 1.58x for BF16,
and 1.35x for FP32. That broader suite is summarized in the RFC but does not publish
the same per-shape detail as Figure 2.

![Eager RMSNorm speedup over ATen](_static/flydsl-pytorch-backend/flydsl-rmsnorm-performance.png)

*Figure 2. Warm eager RMSNorm speedup over ATen for every representative shape
reported in the RMSNorm pull request.*

The chart includes every representative row reported in the RMSNorm pull request.
Latency is measured with GPU events after 10 warmup iterations and over 50 timed
iterations. Every row was confirmed to dispatch to FlyDSL, and both output and `rstd`
were checked against ATen. The measurements represent warm execution and exclude
first-call compilation. Each plotted row is one run; repeating the smallest shape ten
times produced a 1.16x–1.26x speedup range.

### Other eager operator: TopK (summary)

The TopK PR uses two kernel families selected by the input shape and requested `K`:

- A **register kernel** for small fixed `K` values `{2, 4, 8, 16}`.
- A **radix-select kernel** for continuous ranges from `K=64` through `K=1024`.

The initial override supports contiguous FP32 `gfx950` inputs, reduction over the last
dimension, and `largest=True, sorted=True`. Both functional and `out=` variants are
covered. Deterministic mode preserves ATen's tie ordering, and the current MI355X
performance gate requires at least 256 rows.

| Kernel | `K` | Eligible last dimension `N` |
|---|---|---|
| Register | `{2, 4, 8, 16}` | Power of two, `1024 <= N <= 8192` |
| Radix | `64 <= K <= 256` | `8192 <= N <= 32768` |
| Radix | `257 <= K <= 383` | `16384 <= N <= 32768` |
| Radix | `384 <= K <= 831` | `32768 <= N <= 131072` |
| Radix | `832 <= K <= 1024` | `32768 <= N <= 262144` |

For non-power-of-two `K`, the radix path pads to the next complete bitonic sorting
network. This preserves correctness across continuous `K` ranges while keeping the
selection and final ordering inside a tuned GPU kernel.

Across all reported rows, the register family has a **4.81x sampled geomean** in
non-deterministic mode and **4.02x** in deterministic mode. The four radix bands
range from **1.63x–1.97x** and **1.40x–1.69x** respectively. The largest individual
speedups are 12.04x for the register kernel and 2.85x for radix select. The 12.04x
case reduces warm latency from 224.4 µs to 18.6 µs for
`M=16384, N=1024, K=2`.

![Eager TopK sampled geomean speedups](_static/flydsl-pytorch-backend/flydsl-topk-performance.png)

*Figure 3. Bars show geometric-mean TopK speedup over the sampled rows; whiskers show
the sampled minimum and maximum so outliers and the 0.99x case remain visible.*

TopK uses 20 warmup and 100 timed iterations and reports the median of three runs.
One smallest deterministic radix sample measured 0.99x. These sampled ranges are not
a whole-domain performance guarantee.

## TorchInductor: Templates, Caching, and Autotuning

### Dense FP16/BF16 GEMM

The dense GEMM PR targets static 2D `aten.mm(A, B.T)` on `gfx950`: row-major
`A[M, K]` is multiplied by `B[N, K].T` to produce `C[M, N]`. Its FlyDSL wrapper
adapts PyTorch's right-hand-side transpose view to the kernel's `[N, K]` contract
while preserving supported row strides and storage offsets.

Eligibility currently requires:

- FP16 or BF16 inputs and matching output;
- a static, non-empty 2D shape;
- the supported NT layout and vector-load alignment;
- `N` and `K` multiples of 32;
- `gfx950`;
- FlyDSL in the Inductor GEMM backend list;
- GEMM max-autotuning enabled.

The lowering filters configurations that cannot support the concrete shape before
benchmarking. The configuration space covers tile sizes, `K` depth, pipeline stages,
wave grids, output-tile grouping, and half-tile interleaving. By default, PyTorch can
use a single baseline configuration. Enabling FlyDSL autotuning exposes the curated
default set, while `EXHAUSTIVE` search explores the larger pruned space.

At the kernel level, workgroups stage `A` and `B` tiles through LDS, reuse those tiles
across waves, and issue MFMA operations into register-blocked accumulators. Tile
dimensions, pipeline depth, wave layout, `GROUP_M` swizzling, and half-tile
interleaving are template parameters. TorchInductor specializes and benchmarks these
parameters for the concrete problem shape instead of committing PyTorch to one
configuration for every GEMM.

Compiled dispatchers are cached by the constexpr configuration while tensor layouts
remain runtime inputs. The PR also routes FlyDSL's persistent artifact directory under
TorchInductor's cache root by default, while respecting an explicitly configured
`FLYDSL_RUNTIME_CACHE_DIR`.

On 15 BF16 NT GEMM shapes, FlyDSL achieved:

- **1.19x geomean over Triton**: 14 wins, 1 tie, 0 losses;
- **1.15x geomean over ATen**: 11 wins, 2 ties, 2 losses;
- **1.10x geomean over the faster baseline at each shape**: 10 wins, 3 ties,
  2 losses.

![TorchInductor BF16 dense GEMM speedup](_static/flydsl-pytorch-backend/flydsl-dense-gemm-performance.png)

*Figure 4. BF16 dense NT GEMM speedup over the faster ATen/Triton baseline at each
shape. A common scale is used across all panels; ratios within ±1% count as ties.*

The suite includes decode shapes from `M=8` upward, square GEMMs through
`8192 × 8192 × 8192`, LLM projection widths `N=14336` and `N=28672`, and a
`4096 × 256 × 4096` rectangular case. All GEMM shape labels use `M × N × K`.

### Other TorchInductor operators (summary)

**Grouped GEMM.** The grouped GEMM PR adds a
`torch.nn.functional.grouped_mm` template for ragged 2D `A` and grouped
`B[G, K, N]`. Its current gate requires FP16/BF16, static `N` and `K` divisible by
32, and eligible alignment and strides; scaled-grouped fusion is not enabled. On the
standard 14-shape BF16 suite it reaches a **1.21x geomean over Triton** and **2.12x
over ATen**, and is the best measured backend in 11 of 14 cases. The dense-`K/N` and
ragged-`M` suites reach 1.23x and 1.15x over Triton respectively; FlyDSL is best in all
five ragged-`M` cases.

![Grouped GEMM speedups](_static/flydsl-pytorch-backend/flydsl-grouped-gemm-performance.png)

*Figure 5. Geometric-mean grouped GEMM speedup within each reported suite. The PR
reports steady-state throughput from isolated processes with fresh caches, but does
not yet provide consistent warmup, repetition, or software-version details.*

**MXFP8/MXFP4 scaled GEMM.** The scaled GEMM PR adds dedicated BlockWise1x32 kernel
families. The published MXFP8 contract uses row-major E4M3FN `A`, column-major E4M3FN
`B`, E8M0FNU `NO_SWIZZLE` block scales along `K`, FP32 accumulation, FP16/BF16 output,
and `M`, `N`, and `K` divisible by 32. Across 13 shapes, FlyDSL reaches a **1.42x
geomean over ATen in the same harness**, including 2331.6 TFLOP/s at
`8192 × 8192 × 8192`.

![MXFP8 same-harness ATen speedups](_static/flydsl-pytorch-backend/flydsl-mxfp8-performance.png)

*Figure 6. Per-shape MXFP8 speedup over ATen. ATen and FlyDSL were measured
back-to-back through `aten._scaled_mm_v2` with the same graph-replay harness.*

The PR also reports a 1.15x geomean over a separately measured Composable Kernel
reference. That standalone C++ comparison is useful context but is not like-for-like,
so it is not plotted with Figure 6. MXFP4 has its own kernel and configuration family,
but the PR does not yet publish an equally detailed support matrix or full performance
table.

## Safety, Fallback, and Validation

An optional backend must remain invisible when it cannot run. The proposed gates omit
FlyDSL and retain existing choices for:

- CPU-only or CUDA-only PyTorch;
- ROCm PyTorch without FlyDSL installed;
- a missing FlyDSL MLIR runtime;
- a FlyDSL version outside the tested 0.3.x line;
- a non-`gfx950` device for the current PyTorch kernels;
- unsupported dtype, layout, shape, alignment, or operator options;
- a valid FlyDSL candidate that loses TorchInductor autotuning.

These statements cover unavailable runtimes, failed eligibility checks, and candidates
that lose autotuning; they are not a guarantee that every arbitrary compilation or
launch failure can be recovered transparently.

The eager and compiler controls remain independent. Disabling
`python_native.flydsl` restores eager ATen dispatch but does not remove FlyDSL from
TorchInductor. Conversely, removing `FLYDSL` from
`max_autotune_gemm_backends` disables the Inductor templates without changing eager
overrides.

The open PRs add tests for dependency and version detection, import laziness, eager
controls, cache keys, operator accuracy, wrapper generation, configuration filtering,
autotuning, and TopK deterministic ties. A separate open CI PR
([#193473](https://github.com/pytorch/pytorch/pull/193473)) adds a focused `gfx950`
shard and has produced a passing run; it is not yet standard merged CI coverage.

## How to Preview the PR Stacks

These are contributor preview instructions, not released user instructions. They
require a source build from the relevant PyTorch PR stack, ROCm matching that build,
an AMD `gfx950` GPU, and the FlyDSL version validated by that PR. The eager RMSNorm
results above use FlyDSL 0.3.1; other reports validate builds from the 0.3.x line.

```bash
python -m pip install "flydsl==0.3.1"
```

Before a launch post is published, replace this section with an exact PyTorch
release/nightly, ROCm version, FlyDSL build, and a verified installation command.

### Eager mode

Within a source build containing the eager PR stack, eligible operations dispatch
automatically. The `python_native` controller can inspect availability or provide an
explicit fallback context. ROCm builds continue to use PyTorch's `"cuda"` device
string:

```python
import torch
import torch.nn.functional as F
import torch.backends.python_native as pn

assert torch.version.hip is not None
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
assert arch == "gfx950"
print("FlyDSL available:", "flydsl" in pn.available_dsls)

x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

# Uses FlyDSL when every support and performance predicate passes.
y = F.rms_norm(x, (4096,), weight, eps=1e-5)

# Explicitly run the normal ATen path.
with pn.flydsl.disabled():
    y_aten = F.rms_norm(x, (4096,), weight, eps=1e-5)

torch.testing.assert_close(y, y_aten)
```

The same controller supports `enable()`, `disable()`, an `enabled` property, and a
`disabled()` context manager. Unsupported calls need no user handling—they retain
ATen behavior.

### TorchInductor

Within a source build containing the dense GEMM PR, add `FLYDSL` to the GEMM
autotuning backends and enable multiple FlyDSL configurations. These
`torch._inductor.config` controls are experimental/private and may change before
landing:

```python
import torch
import torch._inductor.config as config

config.max_autotune_gemm_backends = "ATEN,TRITON,FLYDSL"
config.flydsl_enable_autotuning = True

A = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
B = torch.randn(14336, 4096, device="cuda", dtype=torch.bfloat16)

@torch.compile(mode="max-autotune-no-cudagraphs")
def f(a, b):
    # The initial dense template targets A @ B.T.
    return a @ b.T

out = f(A, B)  # The first call compiles and benchmarks eligible choices.
```

The equivalent environment configuration is:

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 \
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,FLYDSL" \
FLYDSL_ENABLE_AUTOTUNING=1 \
python my_script.py
```

For the largest search space, additionally set:

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE
```

`EXHAUSTIVE` is useful for kernel evaluation, but it can compile many configurations
and substantially increase first-run autotuning time. The curated default search is
the practical starting point.

## Future Work

If the initial PRs land, follow-up work falls into three themes:

- **Broader qualification:** additional AMD GPU architectures, FP16 GEMM reporting,
  dynamic shapes, and wider layout/dtype coverage.
- **More kernel families:** grouped/expert GEMM, epilogue fusion, and attention
  templates, each with its own explicit support matrix.
- **Lower deployment cost:** bounded heuristic-guided search, parallel
  precompilation, persistent-cache measurements, and AOT-compatible packaging.

Each expansion should keep the same standard: an explicit support matrix, correctness
and fallback tests, cold and warm cost reporting, and performance evidence against
the fastest available PyTorch backend.

## Conclusion

FlyDSL is not proposed as a replacement for PyTorch's existing GPU backends. The
proposal would add architecture-specific candidates for selected `gfx950` workloads
while preserving existing choices elsewhere. The initial kernel-level results show
promising gains for RMSNorm, TopK, dense GEMM, grouped GEMM, and scaled GEMM, but
broader hardware coverage, cold-start measurements, merged CI, and end-to-end workload
evaluation are still required before this work can be presented as a shipped PyTorch
feature.

## References

- [RFC: Integrating FlyDSL with PyTorch DSL Extension Points](https://github.com/pytorch/pytorch/issues/190875)
- [FlyDSL native-op backend infrastructure](https://github.com/pytorch/pytorch/pull/191446)
- [Eager FlyDSL RMSNorm](https://github.com/pytorch/pytorch/pull/191447)
- [Eager FlyDSL TopK](https://github.com/pytorch/pytorch/pull/193548)
- [TorchInductor FlyDSL dense GEMM](https://github.com/pytorch/pytorch/pull/190903)
- [TorchInductor FlyDSL grouped GEMM](https://github.com/pytorch/pytorch/pull/191475)
- [TorchInductor FlyDSL MXFP8/MXFP4 scaled GEMM](https://github.com/pytorch/pytorch/pull/193527)
- [FlyDSL repository and documentation](https://github.com/ROCm/FlyDSL)
- [Generating State-of-the-Art GEMMs with TorchInductor's CuteDSL backend](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/)
