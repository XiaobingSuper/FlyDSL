# Accelerating PyTorch on AMD MI350-Series GPUs with FlyDSL

PyTorch users on AMD MI350-series GPUs can now use
[FlyDSL](https://github.com/ROCm/FlyDSL) kernels for dense and grouped matrix
multiplication, MXFP8/MXFP4 scaled matrix multiplication, RMSNorm, and TopK. The
integration covers both eager execution and `torch.compile` through PyTorch
operator APIs.

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
TorchInductor can benchmark for dense, grouped, and MXFP scaled GEMM. When
FlyDSL and GEMM autotuning are enabled, Inductor compares eligible candidates
and selects the fastest measured implementation for each workload. Dense and
grouped GEMM can choose among ATen, Triton, and FlyDSL; the MXFP scaled GEMM path
compares FlyDSL with ATen.

![PyTorch APIs feed two execution paths: eager dispatch chooses FlyDSL for eligible RMSNorm and TopK inputs or ATen otherwise; torch.compile benchmarks eligible implementations of dense, grouped, and MXFP8/MXFP4 scaled GEMM and runs the fastest on an AMD MI350-series GPU.](_static/flydsl-pytorch-backend/flydsl-pytorch-architecture.png)

*Figure 1. FlyDSL in PyTorch, with the optional package installed and FlyDSL
enabled for GEMM autotuning. Eager execution dispatches by input support;
TorchInductor selects by measured performance, with candidates depending on the
operation. Both paths use PyTorch operator APIs.*

## Supported Features

Current support targets AMD MI350-series GPUs with the `gfx950` architecture.
The following features are available in PyTorch:

| Operation | Execution mode | Data types | Supported scope |
|---|---|---|---|
| Dense GEMM | `torch.compile` | FP16, BF16 | Static 2D `torch.mm`; NN, NT, TN, and TT input layouts |
| MXFP scaled GEMM | `torch.compile` | MXFP8, MXFP4 | Static 2D `torch._scaled_mm_v2`; NN, NT, TN, and TT input layouts; block scales; FP16/BF16 output |
| Grouped GEMM | `torch.compile` | FP16, BF16 | Static-shape `F.grouped_mm` with ragged 2D activations and contiguous 3D weights; uneven and empty groups |
| RMSNorm | Eager | FP16, BF16, FP32 | Forward with contiguous input and explicit weight; one normalized dimension; tuned shape regions |
| TopK | Eager | FP32 | Contiguous input; last-dimension selection with `largest=True` and `sorted=True`; functional and `out=` variants in tuned shape regions |

Dense GEMM's [four-layout support](https://github.com/pytorch/pytorch/pull/194981)
accepts row-major and column-major inputs, including eligible transpose views.
Grouped GEMM supports different row counts per group, allowing experts to process
uneven token assignments. Its current input-layout requirements are separate
from dense GEMM's four-layout coverage. RMSNorm backward continues through
PyTorch's existing implementation.

MXFP scaled GEMM accepts FP8 E4M3 or packed FP4 E2M1 operands, with one E8M0
scale per 32 values along the reduction dimension (`BlockWise1x32`). Applications
supply the quantized operands and contiguous, unswizzled scale tensors. Both
formats use FP32 accumulation and return FP16 or BF16; the current path requires
`K` to be a multiple of 128 and has no bias support.

Each operation has shape and alignment requirements. The detailed
[dense GEMM](https://github.com/pytorch/pytorch/pull/194981),
[MXFP scaled GEMM](https://github.com/pytorch/pytorch/pull/196719),
[grouped GEMM](https://github.com/pytorch/pytorch/pull/194032),
[RMSNorm](https://github.com/pytorch/pytorch/pull/191447), and
[TopK](https://github.com/pytorch/pytorch/pull/193548) support descriptions define
those boundaries. Unsupported eager calls retain ATen behavior; compiled GEMMs
can use the other enabled backends.

## Performance on Key Operators

The following benchmarks compare operator execution on AMD `gfx950` GPUs.
A speedup above 1.0 means FlyDSL is faster than the named comparison backend.
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

### MXFP Scaled GEMM: Extending to Low-Precision Workloads

Microscaled formats combine low-precision values with a shared scale for each
small block of elements. They reduce operand storage while allowing matrix
multiplication to accumulate in FP32. FlyDSL's MXFP8 and MXFP4 support makes
these kernels available to quantized workloads through `torch.compile`.

On MI355X, the 17-shape NT suite shows a **1.44x geometric-mean speedup for
MXFP8 over ATen** and **1.11x over Composable Kernel (CK)**. For MXFP4, the
corresponding results are **1.60x over ATen** and **0.98x relative to AITER
Triton**. The latter places FlyDSL close to that specialized implementation on
average, with the fastest backend varying by shape.

![MXFP8 and MXFP4 scaled GEMM speedups for all 17 shapes per format, compared with ATen and with CK or AITER Triton.](_static/flydsl-pytorch-backend/flydsl-mxfp-gemm-performance.png)

*Figure 3. NT scaled GEMM on MI355X. Each panel shows all 17 cases and their
geometric mean. MXFP8 uses ATen and CK as baselines; MXFP4 uses ATen and AITER
Triton. CK timing includes dynamic activation-scale shuffling and excludes
one-time static weight-scale preprocessing.*

FlyDSL exceeds ATen throughput in every reported case for both formats.
Against the specialized baselines, the picture is more varied: MXFP8 has its
largest CK-relative gain at `32 × 4096 × 4096` (**1.39x**), while MXFP4 ranges
from **0.90x to 1.08x** relative to AITER Triton. These CK and AITER comparisons
provide context for kernel performance; the integrated MXFP autotuning path
currently chooses between ATen and FlyDSL. The measurements cover NT layout.

### Grouped GEMM: Uneven Work Across Experts

In mixture-of-experts models, experts can receive very different numbers of
tokens. Grouped GEMM processes these matrix multiplications together while
accommodating uneven—and sometimes empty—groups.

On the standard 14-shape BF16 suite, FlyDSL achieves a **1.21x geometric-mean
speedup over Triton** and **2.12x over ATen**. It is the fastest measured backend
in **11 of 14 cases**, and in all five separately measured ragged-`M` cases.

![Grouped GEMM geometric-mean speedup across standard, K/N-variant, and ragged-M suites](_static/flydsl-pytorch-backend/flydsl-grouped-gemm-performance.png)

*Figure 4. BF16 grouped GEMM on gfx950. Each pair of bars summarizes a complete
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

*Figure 5. All 22 reported RMSNorm cases on MI355X. Labels identify dtype and
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

*Figure 6. FP32 TopK on MI355X, compared with ATen under the same determinism
setting. The bars aggregate all 33 reported cases by kernel family and K band.*

The small-K register path shows the largest average gain. The radix-select paths
extend the benefit to larger K ranges, although some individual cases are near
parity. For applications that inspect indices, equal-value ties can be ordered
differently across backends; reproducibility does not guarantee identical tied
indices.

The suites use separate measurement setups. Dense GEMM uses graph replay,
the MXFP results report throughput with the CK preprocessing treatment described
above, grouped GEMM reports steady-state throughput, and RMSNorm and TopK use
GPU-event timing. These are operator-level results, so model-level gains depend
on how much time an application spends in the supported operations. Full methodology,
measurement sources, CSV data, and plotting code are available in the
[benchmark notes](_static/flydsl-pytorch-backend/README.md).

## Try It in PyTorch

Use a [ROCm PyTorch nightly](https://pytorch.org/get-started/locally/) that includes
the [MXFP scaled GEMM support](https://github.com/pytorch/pytorch/pull/196719)
alongside the existing FlyDSL operators, then install an optional runtime from
the supported 0.3.x series:

```bash
python -m pip install --upgrade "flydsl>=0.3.0,<0.4"
```

The examples below require a `gfx950` GPU; ROCm builds use PyTorch's `"cuda"`
device string.

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

### MXFP8 and MXFP4 Scaled GEMM

With the GEMM configuration above, compile a call on already quantized operands
and their block scales. This wrapper uses the `torch._scaled_mm_v2` API supported
by the integration:

```python
from torch.nn.functional import ScalingType, SwizzleType

@torch.compile(mode="max-autotune")
def mxfp_mm(a, b, scale_a, scale_b):
    recipe = [ScalingType.BlockWise1x32.value]
    swizzle = [SwizzleType.NO_SWIZZLE.value]
    return torch._scaled_mm_v2(
        a,
        b,
        [scale_a],
        recipe,
        swizzle,
        [scale_b],
        recipe,
        swizzle,
        bias=None,
        out_dtype=torch.bfloat16,
        contraction_dim=[],
        use_fast_accum=False,
    )
```

For an MXFP8 NT call, pass row-major `a[M, K]` and column-major `b[K, N]`
with dtype `torch.float8_e4m3fn`. Both scales have dtype `torch.float8_e8m0fnu`,
with contiguous shapes `[M, K // 32]` and `[N, K // 32]`.
For MXFP4, use packed `torch.float4_e2m1fn_x2` operands with storage shapes
`[M, K // 2]` and `[K // 2, N]`; the scale shapes use the logical `K`.
The wrapper supports both formats with the same backend settings.

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

- **More low-precision workloads and attention:** work on
  [MXFP8 grouped GEMM](https://github.com/pytorch/pytorch/pull/194303) and
  [FlexAttention](https://github.com/pytorch/pytorch/pull/194309) targets more
  inference workloads.
- **Fusion and training:** [GEMM epilogue fusion](https://github.com/pytorch/pytorch/pull/196277)
  aims to combine matrix multiplication with following pointwise operations;
  broader backward coverage would extend training support.
- **Deployment:** [AOTInductor support](https://github.com/pytorch/pytorch/pull/196805)
  aims to package kernels ahead of execution. End-to-end model benchmarks will
  help quantify how operator gains translate to application performance.

These extensions are in progress and are outside the current support and
performance results presented here.

The current integration connects FlyDSL kernel development to both eager
operators and compiled GEMMs, giving PyTorch users a practical way to benefit
from optimized AMD kernels. Try it on your workloads and share results or
feature requests through the
[PyTorch issue tracker](https://github.com/pytorch/pytorch/issues/new/choose).
