# Accelerating PyTorch on AMD MI350-Series GPUs with FlyDSL

*FlyDSL is now available as an optional PyTorch backend for targeted eager operators
and TorchInductor GEMM autotuning on `gfx950` GPUs.*

*Updated September 9, 2026. The coverage below describes changes landed in PyTorch
main; features still under review are identified separately.*

## Introduction

PyTorch now supports [FlyDSL](https://github.com/ROCm/FlyDSL) as an optional kernel
backend for AMD MI350-series GPUs. The integration works in the two execution modes
PyTorch users rely on:

1. **Eager mode**, where selected operators transparently dispatch to tuned FlyDSL
   kernels.
2. **`torch.compile`**, where TorchInductor benchmarks FlyDSL templates alongside
   existing backends and selects the fastest eligible implementation.

Users keep the same PyTorch APIs. FlyDSL is loaded only when it is installed and an
operator matches an explicitly tested support region. Unsupported eager calls continue
through ATen, while TorchInductor simply omits ineligible FlyDSL candidates or selects
another backend when it benchmarks faster.

The landed integration covers RMSNorm, TopK, and FP16/BF16 dense and grouped GEMM
on AMD `gfx950` GPUs. Dense GEMM now accepts all four row-major/column-major input
combinations, extending the original NT path to NN, TN, and TT. MXFP8/MXFP4 scaled
GEMM, attention, and epilogue fusion are active follow-up work.

The reported `gfx950` benchmarks show gains in the targeted shape regions, from
RMSNorm's fused reduction to TopK selection and GEMM autotuning. The Performance
Results section summarizes warm operator and kernel measurements from the linked
PRs; first-call compilation and autotuning are excluded.

## What's Available in PyTorch Main

| Path | Landed coverage | Upstream changes |
|---|---|---|
| Eager RMSNorm | Fused forward for contiguous FP16/BF16/FP32 inputs in the tuned shape region; existing PyTorch backward | [#191447](https://github.com/pytorch/pytorch/pull/191447) |
| Eager TopK | FP32 last-dimension selection, with register and radix-select kernels; functional and `out=` variants | [#193548](https://github.com/pytorch/pytorch/pull/193548) |
| TorchInductor dense GEMM | Static FP16/BF16 `torch.mm`, with NN/NT/TN/TT layouts and per-shape autotuning | [#190903](https://github.com/pytorch/pytorch/pull/190903), [#194981](https://github.com/pytorch/pytorch/pull/194981) |
| TorchInductor grouped GEMM | FP16/BF16 `F.grouped_mm` with ragged 2D activations, contiguous 3D weights, and group offsets | [#194032](https://github.com/pytorch/pytorch/pull/194032) |

These operators build on the landed
[native backend registration](https://github.com/pytorch/pytorch/pull/191446) and
[TorchInductor template infrastructure](https://github.com/pytorch/pytorch/pull/192877).
Availability depends on the PyTorch build containing the relevant changes and on
each operator's device, dtype, shape, and layout checks.

## Why FlyDSL?

FlyDSL—Flexible Layout Python DSL—is a Python DSL and MLIR stack for authoring
high-performance GPU kernels with explicit layouts and tiling. Kernel authors can
describe tensor shapes, strides, coordinate mappings, thread/value partitions, tiled
copies, LDS (AMD GPU shared memory) usage, and matrix fused multiply-add (MFMA)
instructions in Python while retaining control over the hardware execution hierarchy.

FlyDSL traces a Python kernel into the Fly MLIR dialect, lowers through GPU/ROCDL and
LLVM AMDGPU, and emits an HSACO GPU code object. This combines Python-level
productivity with the architecture control needed to tune data movement, wave layout,
shared-memory staging, and matrix-core scheduling on AMD GPUs.

## One Backend, Two Integration Paths

Eager dispatch and TorchInductor autotuning solve different problems, so they remain
independent control planes. They share FlyDSL's optional compiler/runtime and HIP
tensor/stream ABI, but keep separate routing, configuration, and caches.

![FlyDSL integration in PyTorch](_static/flydsl-pytorch-backend/flydsl-pytorch-architecture.png)

*Figure 1. Eager dispatch and TorchInductor autotuning independently use the optional
FlyDSL compiler/runtime.*

In eager mode, PyTorch checks the device, dtype, shape, and layout before lazily
compiling and launching a FlyDSL kernel. The override can be inspected or disabled
through `torch.backends.python_native.flydsl`.

With `torch.compile`, TorchInductor adds FlyDSL configurations only for eligible
problems, benchmarks them with existing choices, and caches the winner.

PyTorch owns the reviewed operator adapters and kernel source snapshots. The
optional `flydsl` package supplies the compiler and runtime, so kernel integration
can be reviewed in PyTorch while the DSL evolves independently.

FlyDSL remains optional: unsupported systems and operator inputs retain existing
PyTorch behavior. Eager controls and TorchInductor backend selection are independent,
so disabling one path does not affect the other.

### Compilation, Caching, and Validation

Repeated calls reuse compiled launchers. The eager path keeps an in-process cache
of operator specializations, backed by FlyDSL's persistent compiled-artifact cache.
For example, TopK specializes on `N` and `K`, and the radix path also keys on the
deterministic setting. Loaded callables are device-specific, while compatible
devices can reuse the same compiled artifacts.

TorchInductor has its own compilation and algorithm-selection path. It caches the
compiled FlyDSL dispatcher and, by default, places FlyDSL's disk artifacts under
the Inductor cache root in `flydsl_compile_cache`. An explicit
`FLYDSL_RUNTIME_CACHE_DIR` takes precedence. A warm launcher-cache hit avoids
re-entering compilation; a disk-cache hit in a new process still requires loading
the artifact. Neither removes the initial cost of benchmarking new GEMM choices.

A dedicated [gfx950 CI shard](https://github.com/pytorch/pytorch/pull/193473) installs
the pinned FlyDSL runtime and runs the Inductor template/GEMM suite. Its preflight
requires a working ROCm runtime and `gfx950` device so missing hardware cannot
silently turn GPU coverage into skipped tests. The operator PRs also include
focused correctness and eligibility tests, including all four dense GEMM layouts,
ragged groups, and eager fallback behavior.

## Performance Results

We first use RMSNorm and dense GEMM to explain the eager and TorchInductor paths in
detail, followed by TopK and grouped GEMM. Each chart summarizes the measurements
reported for that operator, with its own baseline and timing method; the suites
are not a single end-to-end model benchmark. Dense GEMM uses per-shape speedups,
while summary charts use geometric means by kernel family or suite. The scaled
GEMM preview appears separately under What's Next.

### Eager: RMSNorm

[RMSNorm](https://github.com/pytorch/pytorch/pull/191447) is the first detailed
example of the eager integration. The FlyDSL kernel implements the fused forward path
and returns both the normalized output and the FP32 reciprocal standard deviation
required by the operator contract. Backward continues through the existing PyTorch
implementation.

N-dimensional inputs are logically flattened to `(M, N)`. The current `gfx950`
performance gate supports contiguous FP16, BF16, or FP32 input and weight tensors,
one normalized dimension, matching dtype/device, non-negative `eps`, and these
measured shape regions:

| Normalized dimension `N` | Minimum row count `M` |
|---|---:|
| `4096 <= N < 8192` | `8192` |
| `8192 <= N < 16384` | `4096` |
| `16384 <= N <= 114688` | `2048` |

The kernel combines reduction, normalization, scaling, and output generation while
using architecture-specific reductions and vectorized loads. It also handles hidden
dimensions that are not naturally aligned to ATen's preferred vector width. This is
especially effective when `N` is one element larger than an aligned size, such as
`4097`, `8193`, or `16385`.

![Eager RMSNorm speedup over ATen](_static/flydsl-pytorch-backend/flydsl-rmsnorm-performance.png)

*Figure 2. Warm eager RMSNorm speedup over ATen from #191447, on MI355X with
FlyDSL 0.3.0, using GPU events after 10 warmup and 50 timed iterations. Each shape
is a single run. Shape labels are `M × N`; values above 1.0 favor FlyDSL.*

Aligned dimensions improve by **1.16x–1.54x**. For hidden dimensions one element
larger than an aligned size—for example, `4097` instead of `4096`—the measured
speedup reaches **3.66x**.

### TorchInductor: GEMM Autotuning

#### Dense FP16/BF16 GEMM

The original [dense GEMM integration](https://github.com/pytorch/pytorch/pull/190903)
targeted static 2D `aten.mm(A, B.T)` on `gfx950`. The landed
[layout extension](https://github.com/pytorch/pytorch/pull/194981) now consumes
logical `A[M, K]` and `B[K, N]` tensors directly, inferring each operand's layout
from its strides:

| Layout | Logical `A[M, K]` | Logical `B[K, N]` |
|---|---|---|
| NN | Row-major | Row-major |
| NT | Row-major | Column-major, such as `weight.T` |
| TN | Column-major | Row-major |
| TT | Column-major | Column-major |

The kernel selects the corresponding loads and LDS layouts, using transposed LDS
reads where needed. This lets eligible transpose views reach FlyDSL directly
without first making a contiguous copy. The layout is also part of kernel
specialization and caching.

Eligibility currently requires:

- FP16 or BF16 inputs and matching output dtype;
- a static, non-empty 2D shape and row-major output;
- row-major or column-major inputs with 16-byte-aligned origins and leading strides;
- `K` a multiple of 32 and `N` a multiple of 8; column-major `A` additionally
  requires `M` to be a multiple of 8;
- tensor byte spans that fit the kernel's 32-bit buffer addressing;
- an AMD `gfx950` GPU;
- GEMM max-autotuning with `FLYDSL` enabled as a candidate backend.

For each eligible shape, TorchInductor filters incompatible FlyDSL configurations,
benchmarks the remaining choices alongside existing backends, and caches the winner.
The search covers tile sizes, pipeline stages, wave layouts, tile ordering, and
half-tile-interleaved variants. Shape checks account for K tails and each
configuration's LDS requirements before it enters autotuning.

Across the original 15 BF16 NT GEMM shapes in #190903, FlyDSL achieves a **1.19x**
geomean over Triton, **1.15x** over ATen, and **1.10x** over the faster baseline at
each shape.

![TorchInductor BF16 dense GEMM speedup](_static/flydsl-pytorch-backend/flydsl-dense-gemm-performance.png)

*Figure 3. Original BF16 dense NT GEMM results from #190903, relative to the faster
ATen/Triton baseline. Shape labels are `M × N × K`; ratios within ±1% count as ties.*

Results are the median of four accuracy-checked graph-replay runs. FlyDSL and Triton
use the `EXHAUSTIVE` search space; ATen uses its default configuration.

The layout extension in #194981 also reports an NT regression check over those
same 15 shapes. Against the **fixed ATen/Triton reference measurements from
#190903**, the newer FlyDSL run reaches **1.206x** over Triton, **1.165x** over
ATen, and **1.115x** over the faster reference per shape. Its geometric-mean
throughput is **1.07% higher** than the original FlyDSL result. This supports
retaining NT performance while broadening layout coverage; Figure 3 remains the
original measurement set, and these numbers do not characterize NN/TN/TT speedups.

### Additional Operator Results

#### Eager: TopK

[TopK](https://github.com/pytorch/pytorch/pull/193548) uses a register kernel for
small fixed `K` values and radix-select kernels for larger continuous ranges. The
override supports contiguous FP32 `gfx950` inputs, reduction over the last
dimension, `largest=True`, and `sorted=True`. It requires at least one row per
compute unit (`M >= 256` on MI355X), with input and output buffers fitting the
kernel's 32-bit addressing. Functional and `out=` variants are both supported.

| Kernel family | `K` | Last dimension `N` |
|---|---|---|
| Register | `{2, 4, 8, 16}` | Power of two, `1024 <= N <= 8192` |
| Radix-select | `64–256` | `8192 <= N <= 32768` |
| Radix-select | `257–383` | `16384 <= N <= 32768` |
| Radix-select | `384–831` | `32768 <= N <= 131072` |
| Radix-select | `832–1024` | `32768 <= N <= 262144` |

The register kernel is reproducible and breaks equal-value ties by ascending
index, which can differ from ATen. For finite inputs, the radix kernel's
deterministic mode preserves ATen's tie ordering. NaN payload and index choices
can differ because the kernel maps NaNs to a single ordering key. Applications
should not assume identical tied indices across backends.

![Eager TopK speedup over ATen](_static/flydsl-pytorch-backend/flydsl-topk-performance.png)

*Figure 4. TopK geometric-mean speedup over ATen by kernel family from #193548.
MI355X timings use GPU events after 20 warmup and 100 timed iterations, taking the
median of three runs.*

The register family reaches **4.81x** and **4.02x** geometric-mean speedups in
non-deterministic and deterministic modes. Radix-select reaches a **1.40x–1.97x**
geometric-mean speedup across its tuned `K` bands.

#### TorchInductor: Grouped GEMM

The landed
[`torch.nn.functional.grouped_mm` template](https://github.com/pytorch/pytorch/pull/194032)
supports ragged 2D `A[total_M, K]` and grouped `B[G, K, N]`. Device-resident `int32`
offsets mark the end of each group in `A`, so experts can receive different
numbers of rows without padding every expert to the same size.

The current gate requires contiguous FP16/BF16 inputs and matching output dtype,
static `total_M`, `G`, `N`, and `K`, with `N` and `K` divisible by 32. It also
checks alignment and buffer spans. This template covers the unscaled 2D-by-3D
case without bias; dense GEMM's four-layout support does not extend this grouped
layout contract.

The persistent kernel supports uneven and empty groups, stages `B` through LDS,
and reuses the dense integration's configuration and caching infrastructure.
These choices target MoE workloads where expert token counts are imbalanced.

![Grouped GEMM speedups](_static/flydsl-pytorch-backend/flydsl-grouped-gemm-performance.png)

*Figure 5. Geometric-mean BF16 grouped GEMM speedup within each suite reported in
#194032 on `gfx950`. Backends were forced in isolated processes with a fresh cache
per shape, outputs checked against eager results, and steady-state throughput
measured after compilation.*

On the standard 14-shape suite, FlyDSL reaches a **1.21x** geomean over Triton and a
**2.12x** geomean over ATen, and is the best measured backend in 11 of 14 cases. It
is also the best backend in all five ragged-`M` cases.

## How to Try It

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) to
install a ROCm nightly or release that includes FlyDSL support, then install the tested
optional runtime. For all the landed features described here, use a build that
includes #194981 and its preceding integrations. Installing FlyDSL alone does not
add these changes to an older PyTorch build.

The dedicated upstream CI job pins FlyDSL 0.3.0; the current runtime gates accept
the 0.3.x release series:

```bash
python -m pip install "flydsl==0.3.0"
```

ROCm builds continue to use PyTorch's `"cuda"` device string.

### Eager RMSNorm

Eligible eager operations dispatch automatically:

```python
import torch
import torch.nn.functional as F
import torch.backends.python_native as pn

assert torch.version.hip is not None
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
assert arch == "gfx950"
print("FlyDSL available:", "flydsl" in pn.available_dsls)
print("Registered FlyDSL operations:", pn.get_dsl_operations("flydsl"))

x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

y = F.rms_norm(x, (4096,), weight, eps=1e-5)

with pn.flydsl.disabled():
    y_aten = F.rms_norm(x, (4096,), weight, eps=1e-5)

torch.testing.assert_close(y, y_aten)
```

The controller also supports `enable()`, `disable()`, an `enabled` property, and a
`disabled()` context manager. Runtime availability and registration do not mean
every call uses FlyDSL: the operator must also pass its eligibility checks.
Eligible `torch.topk` calls use the same eager controls.

### TorchInductor GEMM

Enable FlyDSL as a GEMM autotuning candidate:

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,FLYDSL" \
FLYDSL_ENABLE_AUTOTUNING=1 \
python my_script.py
```

`my_script.py` can use the normal `torch.compile` API:

```python
import torch

A = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
B = torch.randn(14336, 4096, device="cuda", dtype=torch.bfloat16)

@torch.compile(mode="max-autotune")
def f(a, b):
    return a @ b.T

out = f(A, B)  # The first call compiles and benchmarks eligible choices.
```

`max-autotune` enables template autotuning and CUDAGraphs on GPU. Use
`max-autotune-no-cudagraphs` for workloads that are incompatible with graph capture.

The two environment variables above serve different purposes:
`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` admits FlyDSL into backend selection,
while `FLYDSL_ENABLE_AUTOTUNING=1` expands FlyDSL's own candidate configurations.
Without the latter, FlyDSL offers a single baseline configuration when eligible.
Keeping `ATEN,TRITON` in the candidate list allows Inductor to choose either for
shapes that FlyDSL cannot serve or where another backend benchmarks faster.

The example uses NT. With a contiguous `B[K, N]`, use `a @ b` for NN; eligible
transposed left operands also enable TN and TT. Layout
selection is automatic and requires no extra backend flag.

For the largest search space, additionally set:

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE
```

`EXHAUSTIVE` can compile many configurations and substantially increase first-run
autotuning time. The curated search enabled by `FLYDSL_ENABLE_AUTOTUNING=1` is the
practical starting point. To verify selection, use `TORCH_LOGS="output_code"` and
inspect the generated wrapper for a FlyDSL kernel call. Enabling a candidate
alone does not establish that it won autotuning.

### TorchInductor Grouped GEMM

The same GEMM backend settings enable grouped GEMM. This example has four groups,
including an empty second group:

```python
import torch
import torch.nn.functional as F

# Group row counts: [512, 0, 256, 1024]; offsets are cumulative end positions.
a = torch.randn(1792, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4, 2048, 2048, device="cuda", dtype=torch.bfloat16)
offs = torch.tensor([512, 512, 768, 1792], device="cuda", dtype=torch.int32)

@torch.compile(mode="max-autotune")
def grouped(a, b, offs):
    return F.grouped_mm(a, b, offs=offs)

out = grouped(a, b, offs)  # Shape: [1792, 2048].
```

Here `b` is contiguous in `[G, K, N]` order, as required by the grouped template.
Repeated offsets represent empty groups; `offs` contains one end position per
group and no leading zero.

## What's Next

### Preview: MXFP8 and MXFP4 Scaled GEMM

The proposed [scaled GEMM integration](https://github.com/pytorch/pytorch/pull/194987)
adds MXFP8 and MXFP4 BlockWise1x32 kernel families. It remains under review as of
this update and is not part of the landed coverage or installation examples above.
MXFP8 uses E4M3 inputs, E8M0 block scales, FP32 accumulation, and FP16/BF16 output.

The earlier [#193527 prototype](https://github.com/pytorch/pytorch/pull/193527)
reported the following MXFP8 results:

![MXFP8 preview speedup over ATen](_static/flydsl-pytorch-backend/flydsl-mxfp8-performance.png)

*Figure 6. Prototype MXFP8 results from #193527. ATen and FlyDSL were measured
back-to-back through `aten._scaled_mm_v2` with the same graph-replay harness.
These are historical preview measurements, not results for the current main branch.*

Across 13 shapes, the prototype reaches a **1.42x** geomean over ATen. Its comparison
with a separately measured Composable Kernel reference shows a **1.15x** geomean
advantage, but that reference uses a standalone C++ harness. The chart measures
MXFP8 only; it does not establish MXFP4 performance or results for a future merged
revision.

### Work in Progress

The landed layout support, persistent artifact caching, and dedicated CI provide
the base for the next extensions:

- **More GEMM workloads.** Extend low-precision coverage through
  [scaled GEMM](https://github.com/pytorch/pytorch/pull/194987) and
  [MXFP8 grouped GEMM](https://github.com/pytorch/pytorch/pull/194303), with
  FP16/BF16 Split-K and additional scaling recipes as further targets.
- **Fusion and tuning.** The proposed
  [GEMM epilogue fusion](https://github.com/pytorch/pytorch/pull/196277) applies
  supported scalar pointwise operations and ReLU in the GEMM store path. Broader
  epilogues, fusion-aware benchmarking, better configuration pruning, and parallel
  precompilation remain follow-up work.
- **Attention and training.** A
  [FlexAttention forward backend](https://github.com/pytorch/pytorch/pull/194309)
  targets BF16 prefill and decode. Backward coverage for attention, RMSNorm, and
  grouped/scaled GEMM remains a separate expansion area.
- **Deployment and qualification.** The
  [AOTInductor compiler and export work](https://github.com/pytorch/pytorch/pull/194635)
  targets packaged FlyDSL launchers. Compile time, cache-hit latency, and
  end-to-end model performance need evaluation alongside warm kernel results.

These linked proposals are still under review. Their availability and supported
scope should be checked against the final landed revisions.

## Conclusion

FlyDSL gives PyTorch a Python-authored path to architecture-specific ROCm kernels
without replacing existing GPU backends or making the compiler a required dependency.
The first eager and TorchInductor integrations show that targeted kernels can deliver
meaningful gains while preserving PyTorch APIs, fallback behavior, and backend
selection.

We plan to grow coverage only where correctness, maintainability, and measured
performance justify it. Feedback and new workload ideas are welcome in the
[PyTorch RFC](https://github.com/pytorch/pytorch/issues/190875) and
[PyTorch issue tracker](https://github.com/pytorch/pytorch/issues/new/choose).
