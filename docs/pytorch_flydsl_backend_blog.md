# Accelerating PyTorch on AMD MI350-Series GPUs with FlyDSL

## Introduction

PyTorch now integrates [FlyDSL](https://github.com/ROCm/FlyDSL) as an optional
kernel backend for AMD MI350-series GPUs. The integration brings tuned kernels to
two familiar workflows: eager execution for RMSNorm and TopK, and
`torch.compile` for dense and grouped matrix multiplication.

FlyDSL gives kernel authors explicit control over tensor layouts, data movement,
and matrix instructions in Python. PyTorch connects those kernels to existing
operator APIs and TorchInductor's autotuning pipeline. Users can enable another
source of optimized kernels while continuing to write ordinary PyTorch code.

This post explains the integration, presents operator-level performance results,
and shows how to try it. Current coverage includes FP16/BF16 dense GEMM with all
four input layouts—NN, NT, TN, and TT—and grouped GEMM for workloads with uneven
numbers of tokens per expert.

## Why FlyDSL?

Efficient GPU kernels depend on how computation maps to the hardware. For GEMM,
tile sizes, shared-memory staging, wave layouts, and matrix-instruction scheduling
all affect utilization. For reductions and selection, vectorized memory access and
the way threads exchange values can determine whether a kernel performs well on a
particular shape.

FlyDSL, short for Flexible Layout Python DSL, exposes these choices through a
Python DSL and MLIR compiler stack. Kernel authors can describe tensor layouts,
partition work across threads and waves, stage data in LDS (AMD GPU shared memory),
and use matrix fused multiply-add (MFMA) instructions. The compiler lowers this
description to AMD GPU code.

The PyTorch integration targets workloads where this control has demonstrated
value: fused RMSNorm reductions, TopK selection, and matrix multiplications tuned
for `gfx950`. PyTorch maintains the operator adapters and reviewed kernel
snapshots; the FlyDSL compiler and runtime remain an optional external package.
This keeps the integration close to PyTorch's dispatch and compilation machinery
while allowing the DSL to evolve independently.

## How FlyDSL Fits into PyTorch

The integration uses two independent paths, shown below.

![Eager dispatch and TorchInductor autotuning with FlyDSL](_static/flydsl-pytorch-backend/flydsl-pytorch-architecture.png)

*Figure 1. Eager operators use an eligibility check and a cached FlyDSL launcher.
TorchInductor benchmarks eligible FlyDSL templates alongside other backends.*

In **eager mode**, PyTorch checks the operator's device, dtype, shape, and layout.
An eligible call dispatches to its FlyDSL kernel; other calls continue through
ATen. The first eligible call compiles a specialization, and subsequent calls
reuse the loaded launcher. The public control surface is
`torch.backends.python_native.flydsl`.

With **TorchInductor**, FlyDSL participates in GEMM autotuning. During lowering,
Inductor identifies the matrix dimensions and input strides, filters compatible
kernel configurations, and benchmarks them against other enabled backends.
Configurations vary in tile size, pipeline depth, wave layout, and tile ordering.
Inductor selects and caches the fastest measured choice for that problem.

The dense GEMM backend infers NN, NT, TN, or TT layout from the input strides.
It selects the corresponding loads and LDS layouts, including transposed LDS
reads where needed. Eligible transpose views therefore reach the kernel directly,
without requiring a contiguous copy.

Both paths reuse FlyDSL's persistent compilation artifacts as well as loaded
callables. Inductor places its FlyDSL artifacts under the Inductor cache root by
default. These caches reduce repeated compilation work; a new shape or
configuration can still incur compilation and autotuning costs. A dedicated
[gfx950 CI job](https://github.com/pytorch/pytorch/pull/193473) exercises the
Inductor integration with the optional runtime installed.

Eager controls and Inductor backend selection are independent. Adding `FLYDSL`
to the GEMM backend list makes it a candidate; the selected implementation can
still be ATen or Triton when either benchmarks faster.

## Performance Results

The following results cover four operator families on AMD `gfx950` GPUs. They
measure warm execution after compilation and, where applicable, autotuning.
Speedup is baseline latency divided by FlyDSL latency, equivalently FlyDSL
throughput divided by baseline throughput for the same work. Values above 1.0
favor FlyDSL.

Each suite uses its own measurement setup, described with its results. Geometric
means give each sampled case equal weight within that suite. These are operator
benchmarks; they do not measure end-to-end model speedup or first-call latency.

### Dense GEMM

Across 15 BF16 NT GEMM shapes, FlyDSL achieves a **1.19x geometric-mean speedup
over Triton**, **1.15x over ATen**, and **1.10x over the faster baseline at each
shape**.

The strongest gains over the faster baseline occur in smaller and medium-sized
matrix problems. For example, `M × N × K = 64 × 4096 × 4096` improves by
**1.33x**. The two largest square-output problems in this suite slightly favor
ATen, illustrating why backend selection remains useful.

![FlyDSL dense GEMM speedup for all 15 BF16 NT shapes, including ties and losses](_static/flydsl-pytorch-backend/flydsl-dense-gemm-performance.png)

*Figure 2. BF16 NT GEMM speedup relative to the faster ATen/Triton baseline for
each shape. All 15 cases are shown: 10 wins, three ties, and two losses using a
±1% tie band. The final bar is the geometric mean.*

These [measurements](https://github.com/pytorch/pytorch/pull/190903#issuecomment-5061510962)
use graph replay on MI355X and the median of four accuracy-checked runs.
FlyDSL and Triton use the `EXHAUSTIVE` search space; ATen uses its default
configuration. The chart reports NT performance. The
[all-layout implementation](https://github.com/pytorch/pytorch/pull/194981)
also validates NN, TN, and TT correctness; its NT regression check reports
1.07% higher geometric-mean throughput than this measurement set. That check
reuses the original ATen/Triton references, so it is separate from the chart.

### Grouped GEMM

Grouped GEMM is useful in mixture-of-experts models, where different experts
receive different numbers of tokens. FlyDSL uses a persistent kernel that
processes ragged activations against grouped weights, including groups with no
rows. The kernel stages weights through LDS and shares the dense GEMM
integration's tuning and caching infrastructure.

On the standard 14-shape BF16 suite, FlyDSL achieves a **1.21x geometric-mean
speedup over Triton** and **2.12x over ATen**. It is the fastest measured backend
in **11 of 14 cases**, and in all five ragged-`M` cases. The separate K/N sweep
also includes a small-`N` case where Triton is faster.

![Grouped GEMM geometric-mean speedup over Triton and ATen in three benchmark suites](_static/flydsl-pytorch-backend/flydsl-grouped-gemm-performance.png)

*Figure 3. BF16 grouped GEMM speedup, aggregated separately over the standard
14-shape suite, five K/N variants, and five ragged-M cases. Every case in each
suite contributes to its geometric mean.*

The [grouped GEMM benchmark](https://github.com/pytorch/pytorch/pull/194032)
forces each backend in an isolated process with a fresh cache per shape, checks
outputs against eager results, and reports steady-state throughput on `gfx950`.
Compilation time is excluded.

### RMSNorm

RMSNorm combines a reduction with normalization and scaling. The FlyDSL kernel
fuses these steps and returns both the normalized output and the FP32 reciprocal
standard deviation consumed by PyTorch's backward implementation.

For the measured aligned hidden dimensions, speedups over ATen are approximately
**1.2x–1.5x**. Dimensions one element above an aligned size—for example, `4097`
rather than `4096`—benefit more, reaching **3.66x**. The chart separates these
cases so the larger gains can be interpreted in the context of their shapes.

![RMSNorm speedups across 22 FP16, BF16, and FP32 cases, split by hidden-dimension alignment](_static/flydsl-pytorch-backend/flydsl-rmsnorm-performance.png)

*Figure 4. All 22 reported RMSNorm cases, grouped into aligned and off-by-one
hidden dimensions. Each label identifies dtype and M × N; both panels use the
same speedup scale.*

The [RMSNorm measurements](https://github.com/pytorch/pytorch/pull/191447) use
MI355X with FlyDSL 0.3.0 and GPU events after 10 warmup and 50 timed iterations.
Each case is one run, with output and reciprocal-standard-deviation correctness
checked against ATen. The benchmark also verifies that the FlyDSL override
actually dispatches.

### TopK

TopK uses two kernel families: a register kernel for small fixed `K`, and
radix-select kernels for larger ranges of `K`. This lets the implementation
adapt its selection strategy to the amount of output requested.

Across the sampled register-kernel cases, FlyDSL achieves a **4.82x
geometric-mean speedup over ATen with deterministic algorithms disabled**, and
**4.02x with them enabled**. The radix-select families achieve **1.40x–1.97x**
geometric-mean speedups across the reported bands and determinism settings.

![TopK geometric-mean speedup by kernel family and determinism setting](_static/flydsl-pytorch-backend/flydsl-topk-performance.png)

*Figure 5. FP32 TopK speedup over ATen under the same determinism setting.
Geometric means include all 33 reported cases, grouped by kernel family and K
band. Individual cases can be at parity even when the family average improves.*

The [TopK benchmark](https://github.com/pytorch/pytorch/pull/193548) uses MI355X
and GPU events after 20 warmup and 100 timed iterations, taking the median of
three runs. The register kernel is reproducible in either mode; the radix kernel
changes its gathering behavior when deterministic algorithms are enabled.

The plotted values and aggregates are reproducible from the accompanying
[benchmark data and plotting script](_static/flydsl-pytorch-backend/README.md).
Ratios are computed from published, rounded latency or throughput values, so
small rounding differences from the source reports are possible.

## Supported Features

The backend currently targets AMD MI350-series `gfx950` GPUs.

| Feature | Supported scope |
|---|---|
| Dense GEMM | Static 2D `torch.mm`; FP16/BF16 inputs and matching output dtype; NN, NT, TN, and TT input layouts |
| Grouped GEMM | `torch.nn.functional.grouped_mm`; FP16/BF16 ragged `A[total_M, K]`, contiguous `B[G, K, N]`, and device-resident group offsets; uneven and empty groups |
| Eager RMSNorm | Fused forward for contiguous FP16/BF16/FP32 input and explicit weight; one normalized dimension; existing PyTorch backward |
| Eager TopK | Contiguous FP32 input; last-dimension reduction with `largest=True` and `sorted=True`; functional and `out=` variants |
| GEMM autotuning | Selection alongside ATen and Triton; curated and exhaustive FlyDSL configuration sets |
| Compilation caching | Reuse of loaded launchers and persistent FlyDSL artifacts; Inductor-owned artifact-cache location by default |

For dense GEMM, N denotes a row-major operand and T a column-major operand,
such as a transpose view. The current gate requires `K` divisible by 32, `N`
divisible by 8, and `M` divisible by 8 when the left operand is column-major,
along with 16-byte input alignment and supported buffer spans. Grouped GEMM
requires static dimensions, `N` and `K` divisible by 32, and the contiguous
layout above; its current template handles the unscaled operation without bias.

Eager acceleration is restricted to tuned shape regions. RMSNorm starts at
`N = 4096` with large row counts. TopK's register path covers
`K = {2, 4, 8, 16}`, and radix-select covers tuned bands within
`64 <= K <= 1024`, with at least one row per compute unit. The
[RMSNorm](https://github.com/pytorch/pytorch/pull/191447) and
[TopK](https://github.com/pytorch/pytorch/pull/193548) support tables give the
complete shape boundaries; calls outside them retain ATen behavior.

For TopK, reproducibility does not imply identical tied indices across backends.
The register kernel orders equal values by ascending index. The radix kernel's
deterministic mode matches ATen's tie ordering for finite inputs; NaN payload and
index choices can differ.

## How to Try It

### Installation

Use a [ROCm PyTorch nightly](https://pytorch.org/get-started/locally/) that includes
the [all-layout GEMM support](https://github.com/pytorch/pytorch/pull/194981) and
the preceding integrations. Install the optional runtime used by the dedicated
upstream CI job:

```bash
python -m pip install "flydsl==0.3.0"
```

The integration accepts the FlyDSL 0.3.x release series. ROCm builds use
PyTorch's `"cuda"` device string; the examples below require a `gfx950` GPU.

### GEMM with TorchInductor

Add FlyDSL to the candidate backends and enable its configuration search:

```python
import torch
import torch._inductor.config as config

config.max_autotune_gemm_backends = "ATEN,TRITON,FLYDSL"
config.flydsl_enable_autotuning = True

a = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 14336, device="cuda", dtype=torch.bfloat16)

@torch.compile(mode="max-autotune")
def mm(a, b):
    return a @ b

out = mm(a, b)  # First call compiles and benchmarks eligible kernels.
```

This example uses NN layout. A column-major right operand, such as a transposed
`[N, K]` weight, uses NT; transposed left operands also enable TN and TT.
Inductor infers the layout automatically.

The same settings can be supplied through environment variables:

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,FLYDSL" \
FLYDSL_ENABLE_AUTOTUNING=1 \
python my_script.py
```

The backend list controls which implementations compete. The FlyDSL-specific
flag expands its candidate configurations; otherwise it offers a single baseline
configuration. The curated search is a practical starting point. Setting
`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE` explores more
configurations at the cost of longer first-run autotuning.

`max-autotune` also enables GPU graph capture. Use
`max-autotune-no-cudagraphs` when graph capture is unsuitable. To inspect which
implementation was selected, run with `TORCH_LOGS="output_code"` and look for a
FlyDSL kernel call in the generated wrapper.

The same backend settings apply to `F.grouped_mm`. For ragged inputs, `offs`
contains cumulative group-end positions: row counts `[512, 0, 256, 1024]` become
`int32` offsets `[512, 512, 768, 1792]`. Repeated offsets represent empty
groups, and the weights are contiguous in `[G, K, N]` order.

### Eager Operators

Eligible eager calls dispatch automatically once the optional runtime is
available. The controller provides a convenient way to compare against ATen:

```python
import torch
import torch.nn.functional as F
import torch.backends.python_native as pn

assert torch.version.hip is not None
assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"
assert pn.flydsl.available

x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

y = F.rms_norm(x, (4096,), weight, eps=1e-5)
with pn.flydsl.disabled():
    y_aten = F.rms_norm(x, (4096,), weight, eps=1e-5)

torch.testing.assert_close(y, y_aten)
```

The same controller applies to eligible `torch.topk` calls. It supports
`enable()`, `disable()`, the `enabled` property, and the `disabled()` context
manager. Availability indicates that the runtime is present; dispatch still
depends on each operator's eligibility checks.

## Future Work

The next extensions focus on expanding useful workloads and reducing the cost of
bringing tuned kernels into applications:

- **Low-precision GEMM.** MXFP8/MXFP4 scaled GEMM and MXFP8 grouped GEMM extend
  coverage to block-scaled inputs. Further targets include Split-K and additional
  scaling recipes.
- **Fusion and tuning.** GEMM epilogue fusion can combine supported pointwise
  operations with the output store. Broader fusion support, fusion-aware
  benchmarking, configuration pruning, and parallel precompilation are
  complementary directions.
- **Attention and training.** FlexAttention forward work targets BF16 prefill
  and decode; additional backward kernels would extend training coverage.
- **Deployment and evaluation.** AOTInductor integration aims to package FlyDSL
  launchers ahead of execution. Compile time, cache-hit latency, and end-to-end
  model performance remain important measurements alongside kernel throughput.

Work is underway on
[scaled GEMM](https://github.com/pytorch/pytorch/pull/194987),
[scaled grouped GEMM](https://github.com/pytorch/pytorch/pull/194303),
[epilogue fusion](https://github.com/pytorch/pytorch/pull/196277),
[FlexAttention](https://github.com/pytorch/pytorch/pull/194309), and
[AOTInductor](https://github.com/pytorch/pytorch/pull/194635).
These extensions are under review and are outside the supported features and
performance results presented here.

## Conclusion

FlyDSL brings Python-authored, architecture-specific kernels into PyTorch's eager
dispatcher and TorchInductor autotuning. The current integration covers
reductions, selection, and dense and grouped matrix multiplication, with measured
gains across the targeted AMD GPU workloads.

Try the backend on your own shapes and share results or feature requests through
the [PyTorch issue tracker](https://github.com/pytorch/pytorch/issues/new/choose).
The [integration RFC](https://github.com/pytorch/pytorch/issues/190875) provides
further background for contributors.
