# Accelerating PyTorch on AMD MI350-Series GPUs with FlyDSL

PyTorch users on AMD MI350-series GPUs can now use
[FlyDSL](https://github.com/ROCm/FlyDSL) kernels for dense and grouped matrix
multiplication, RMSNorm, and TopK. The integration covers both eager execution
and `torch.compile` through familiar PyTorch APIs.

FlyDSL is a Python-based language for writing GPU kernels, available in PyTorch
as an optional backend. In the reported operator benchmarks, it delivers a
**1.10x geometric-mean speedup for dense GEMM over the faster ATen/Triton
baseline** across the measured BF16 NT suite, and **4.82x for small-K TopK over
ATen with deterministic algorithms disabled**. These results illustrate benefits
across both matrix multiplication and selection workloads.

## How FlyDSL Fits into PyTorch

The integration makes FlyDSL kernels accessible through two execution paths.
In **eager mode**, eligible RMSNorm and TopK calls use FlyDSL automatically when
the optional runtime is installed and enabled. PyTorch checks the inputs and
retains its ATen implementation for unsupported cases.

With **`torch.compile`**, FlyDSL joins the set of implementations that
TorchInductor can benchmark for dense and grouped GEMM. When FlyDSL and GEMM
autotuning are enabled, Inductor compares eligible candidates from the configured
backends and selects the fastest measured implementation for each workload.

![PyTorch APIs feed two execution paths: eager dispatch chooses FlyDSL for eligible RMSNorm and TopK inputs or ATen otherwise; torch.compile benchmarks enabled ATen, Triton, and FlyDSL GEMM candidates and runs the fastest on an AMD MI350-series GPU.](_static/flydsl-pytorch-backend/flydsl-pytorch-architecture.png)

*Figure 1. FlyDSL in PyTorch, with the optional package installed and FlyDSL
enabled for GEMM autotuning. Eager execution dispatches by input support;
TorchInductor selects by measured performance. Both paths preserve the familiar
PyTorch operator APIs.*

## Supported Features

Current support targets AMD MI350-series GPUs with the `gfx950` architecture.
The following features are available in PyTorch:

| Operation | Execution mode | Data types | Supported scope |
|---|---|---|---|
| Dense GEMM | `torch.compile` | FP16, BF16 | Static 2D `torch.mm`; NN, NT, TN, and TT input layouts |
| Grouped GEMM | `torch.compile` | FP16, BF16 | Static-shape `F.grouped_mm` with ragged 2D activations and contiguous 3D weights; uneven and empty groups |
| RMSNorm | Eager | FP16, BF16, FP32 | Forward with contiguous input and explicit weight; one normalized dimension; tuned shape regions |
| TopK | Eager | FP32 | Contiguous input; last-dimension selection with `largest=True` and `sorted=True`; functional and `out=` variants in tuned shape regions |

Dense GEMM's [four-layout support](https://github.com/pytorch/pytorch/pull/194981)
accepts row-major and column-major inputs, including eligible transpose views.
Grouped GEMM supports different row counts per group, allowing experts to process
uneven token assignments. Its current input-layout requirements are separate
from dense GEMM's four-layout coverage. RMSNorm backward continues through
PyTorch's existing implementation.

Each operation has shape and alignment requirements. The detailed
[dense GEMM](https://github.com/pytorch/pytorch/pull/194981),
[grouped GEMM](https://github.com/pytorch/pytorch/pull/194032),
[RMSNorm](https://github.com/pytorch/pytorch/pull/191447), and
[TopK](https://github.com/pytorch/pytorch/pull/193548) support descriptions define
those boundaries. Unsupported eager calls retain ATen behavior; compiled GEMMs
can use the other enabled backends.

## Performance on Key Operators

The following benchmarks measure warm operator execution on AMD `gfx950` GPUs,
after compilation and autotuning. A speedup above 1.0 means FlyDSL is faster.
Geometric means give equal weight to each sampled case within a suite.

### Dense GEMM: Linear-Layer Workloads

Dense matrix multiplication is a building block of transformer projection and
feed-forward layers. Its dimensions vary substantially with the number of tokens
being processed, making performance across a range of shapes relevant to model
developers.

Across 15 BF16 NT shapes, FlyDSL delivers a **1.10x geometric-mean speedup over
the faster ATen/Triton baseline at each shape**. Gains are strongest in smaller
and medium-sized problems: `M × N × K = 64 × 4096 × 4096`, for example, improves
by **1.33x**.

![FlyDSL dense GEMM speedup for all 15 BF16 NT shapes](_static/flydsl-pytorch-backend/flydsl-dense-gemm-performance.png)

*Figure 2. BF16 NT GEMM on MI355X, relative to the faster ATen/Triton baseline at
each shape. All 15 cases are shown: 10 wins, three ties, and two losses using a
±1% tie band.*

The two largest square-output problems in this suite slightly favor ATen. Keeping
multiple backends enabled lets PyTorch make that choice for each workload.
These results characterize NT GEMM; NN, TN, and TT are also supported, with their
performance depending on the workload.

### Grouped GEMM: Uneven Work Across Experts

In mixture-of-experts models, experts can receive very different numbers of
tokens. Grouped GEMM processes these matrix multiplications together while
accommodating uneven—and sometimes empty—groups.

On the standard 14-shape BF16 suite, FlyDSL achieves a **1.21x geometric-mean
speedup over Triton** and **2.12x over ATen**. It is the fastest measured backend
in **11 of 14 cases**, and in all five separately measured ragged-`M` cases.

![Grouped GEMM geometric-mean speedup across standard, K/N-variant, and ragged-M suites](_static/flydsl-pytorch-backend/flydsl-grouped-gemm-performance.png)

*Figure 3. BF16 grouped GEMM on gfx950. Each pair of bars summarizes a complete
suite: 14 standard shapes, five K/N variants, or five ragged-M cases.*

The ragged cases include imbalanced token counts and empty groups, making them
particularly relevant to expert workloads. Gains vary by shape: the K/N sweep
also includes a small-`N` case where Triton is faster.

### RMSNorm: Gains Across Hidden Dimensions

RMSNorm normalizes and scales activations, a repeated step in many transformer
models. FlyDSL accelerates the forward operation while retaining the normal
PyTorch API and backward behavior.

For the measured aligned hidden dimensions, speedups over ATen are approximately
**1.2x–1.5x**. Dimensions one element above an aligned size—for example, `4097`
rather than `4096`—show larger gains, reaching **3.66x**.

![RMSNorm speedups for aligned and off-by-one hidden dimensions](_static/flydsl-pytorch-backend/flydsl-rmsnorm-performance.png)

*Figure 4. All 22 reported RMSNorm cases on MI355X. Labels identify dtype and
M × N, where M is the row count and N the normalized dimension. Both panels use
the same speedup scale.*

The split between aligned and off-by-one dimensions shows why the input shape
matters when interpreting the peak result. The 3.66x speedup applies to a
particular FP16 case; gains on aligned dimensions are more modest.

### TopK: Faster Selection for Small and Large K

TopK selects the highest-scoring elements from each row. The amount of output
requested changes the workload: selecting a handful of elements is different
from selecting hundreds. FlyDSL provides specialized paths for these regimes.

For small `K = {2, 4, 8, 16}`, FlyDSL achieves a **4.82x geometric-mean speedup
over ATen with deterministic algorithms disabled**, and **4.02x with them
enabled**. For the larger sampled K bands, the geometric-mean speedup ranges
from **1.40x to 1.97x**.

![TopK geometric-mean speedup by K band and determinism setting](_static/flydsl-pytorch-backend/flydsl-topk-performance.png)

*Figure 5. FP32 TopK on MI355X, compared with ATen under the same determinism
setting. The bars aggregate all 33 reported cases by kernel family and K band.*

The small-K register path shows the largest average gain. The radix-select paths
extend the benefit to larger K ranges, although some individual cases are near
parity. For applications that inspect indices, equal-value ties can be ordered
differently across backends; reproducibility does not guarantee identical tied
indices.

The four suites use separate measurement setups. Dense GEMM uses graph replay,
grouped GEMM reports steady-state throughput, and RMSNorm and TopK use GPU-event
timing. These are operator-level results, so model-level gains depend on how much
time an application spends in the supported operations. Full methodology,
measurement sources, CSV data, and plotting code are available in the
[benchmark notes](_static/flydsl-pytorch-backend/README.md).

## Try It in PyTorch

Use a [ROCm PyTorch nightly](https://pytorch.org/get-started/locally/) that includes
the [current FlyDSL support](https://github.com/pytorch/pytorch/pull/194981), then
install the tested optional runtime:

```bash
python -m pip install "flydsl==0.3.0"
```

The integration accepts FlyDSL 0.3.x. The examples below require a `gfx950` GPU;
ROCm builds use PyTorch's `"cuda"` device string.

### Dense and Grouped GEMM

Enable FlyDSL as a candidate and let `torch.compile` select the implementation:

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

out = mm(a, b)
```

This example uses NN layout. Eligible transposed inputs enable NT, TN, and TT
without another flag. The same settings apply to `F.grouped_mm`.

The first call includes compilation and tuning; later calls reuse the selected
implementation. Enabling FlyDSL gives PyTorch another candidate, so either ATen
or Triton may still be selected. Use `max-autotune-no-cudagraphs` when the
workload is incompatible with GPU graph capture.

### Eager RMSNorm and TopK

Eligible eager calls use FlyDSL automatically. The backend controller makes it
easy to compare with ATen:

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

The same `pn.flydsl.disabled()` context manager applies to eligible
`torch.topk` calls. Actual dispatch depends on the operation's shape, dtype, and
layout.

## What's Next

The next steps extend the range of workloads that can benefit from FlyDSL:

- **Low precision and attention:** work on
  [MXFP8/MXFP4 scaled GEMM](https://github.com/pytorch/pytorch/pull/194987),
  [MXFP8 grouped GEMM](https://github.com/pytorch/pytorch/pull/194303), and
  [FlexAttention](https://github.com/pytorch/pytorch/pull/194309) targets more
  inference workloads.
- **Fusion and training:** [GEMM epilogue fusion](https://github.com/pytorch/pytorch/pull/196277)
  aims to combine matrix multiplication with following pointwise operations;
  broader backward coverage would extend training support.
- **Deployment:** [AOTInductor support](https://github.com/pytorch/pytorch/pull/194635)
  aims to package kernels ahead of execution. End-to-end model benchmarks will
  help quantify how operator gains translate to application performance.

These extensions are in progress and are outside the current support and
performance results presented here.

The current integration connects FlyDSL kernel development to both eager
operators and compiled GEMMs, giving PyTorch users a practical way to benefit
from optimized AMD kernels. Try it on your workloads and share results or
feature requests through the
[PyTorch issue tracker](https://github.com/pytorch/pytorch/issues/new/choose).
