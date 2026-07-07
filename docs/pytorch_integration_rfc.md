# RFC: Integrating FlyDSL with PyTorch DSL Extension Points

Status: Draft  
Authors: FlyDSL contributors  
Companion analysis: `docs/cutedsl_pytorch_integration_report.md`

## Summary

This RFC proposes integrating FlyDSL as an optional ROCm Python DSL backend in
PyTorch through two independent planes:

- native/eager dispatcher overrides through `torch._native`;
- TorchInductor compiler templates through Inductor lowering, template choices,
  async compile/load, and autotune.

The first native/eager prototype targets RMSNorm. The first Inductor prototype
targets bf16 hgemm for `aten.mm(A, B.T)`. Future work should expand from this
foundation toward broader GEMM dtype/layout coverage, grouped/expert GEMM,
fusion/epilogues, cache/AOT, and workload integrations.

The core design principle is additive integration. FlyDSL should be a selectable
backend candidate with strict gates and fallback, not a required dependency and
not a replacement for existing Aten, Triton, CK, CKTile, or vendor backends.

## Motivation

PyTorch already has several extension points for Python GPU DSLs:

- `torch._native` for eager/native dispatcher overrides with aten fallback;
- `torch.backends.python_native` for native/eager DSL enable and disable controls;
- `torch._inductor.codegen.*` for template-based compiler backends used by
  `torch.compile`;
- optional dependency and test utilities for runtime gates and skip behavior.

CuteDSL uses these extension points on CUDA. FlyDSL is a ROCm-oriented Python DSL
with an MLIR lowering pipeline, tensor ABI support, stream support, and
JIT/runtime caching. FlyDSL should plug into PyTorch through the same style of
optional DSL architecture, while preserving PyTorch default behavior on systems
without FlyDSL.

## Goals

1. Add FlyDSL as an optional ROCm DSL runtime known to PyTorch.
2. Reuse PyTorch's existing native/eager and Inductor extension points.
3. Keep native/eager controls independent from Inductor compiler controls.
4. Keep the FlyDSL compiler/runtime outside PyTorch.
5. Preserve aten behavior for unsupported platforms, package versions, dtypes,
   shapes, layouts, and architectures.
6. Use RMSNorm as the first native/eager prototype.
7. Use bf16 hgemm for `aten.mm(A, B.T)` as the first Inductor template prototype.
8. Provide a staged rollout plan that can be reviewed as small, independently
   owned PRs.

## Non-Goals

| Non-goal | Reason |
|---|---|
| Make `flydsl` a required PyTorch dependency | PyTorch default installs must remain unchanged. |
| Vendor FlyDSL's MLIR compiler into PyTorch | The compiler/runtime should remain FlyDSL-owned. |
| Replace CK, CKTile, Triton, Aten, or vendor libraries | FlyDSL is an optional accelerator/backend candidate. |
| Land native/eager and Inductor support in one PR | They are different review surfaces with different failure modes. |
| Claim full GEMM backend coverage from the first hgemm prototype | The current Inductor prototype is intentionally narrow. |
| Claim MoE support from grouped GEMM primitives alone | MoE needs routing metadata, wrapper ABI, validation, and workload benchmarks. |

## Current Prototype Status

The current PyTorch prototypes are split into independent branches:

| Branch | Scope | Status |
|---|---|---|
| Native/eager | FlyDSL RMSNorm through `torch._native` and `python_native` controls | Implemented separately; not required by Inductor. |
| Inductor/compiler | FlyDSL bf16 hgemm template for `aten.mm(A, B.T)` on ROCm | Implemented as an Inductor-only prototype with focused tests and autotune coverage. |

The Inductor prototype intentionally does not depend on
`torch._native.dsl_registry` or `torch.backends.python_native.flydsl`. It uses
Inductor GEMM autotune backend controls plus an Inductor-local runtime gate. This
keeps eager/native rollback and compiler-template selection separate.

## Architecture Decision

FlyDSL integration should use two independent control planes:

```mermaid
flowchart LR
    A["Native Control<br/>dsl_registry + python_native"] --> B["Native / Eager"]
    C["Compiler Control<br/>Inductor config + template selection"] --> D["Compiler / Inductor"]
    E["Optional Runtime Policy"] -.-> A
    E -.-> C
    B --> F["Validation"]
    D --> F
```

| Plane | Existing PyTorch mechanism | FlyDSL addition |
|---|---|---|
| Native control | `torch._native.dsl_registry`, `torch.backends.python_native` | Register `flydsl` and expose native/eager controls. |
| Native/eager | `torch._native.registry`, per-op `cond` / `impl` wrappers | ROCm op adapters with lazy FlyDSL imports and aten fallback. |
| Compiler/Inductor | Inductor lowering, templates, scheduling, `async_compile.*`, autotune | `FlyDSLTemplate`, `FlyDSLScheduling`, `async_compile.flydsl`, and hgemm choices. |
| Validation | Optional package install, smoke tests, OpInfo, compiler tests | FlyDSL CI install and focused native/Inductor tests. |

Decision statement:

> Native `python_native.flydsl` controls must not be used as the Inductor
> template selector. Native/eager overrides and Inductor templates may share
> optional runtime policy, but their routing and rollback mechanisms must remain
> separate.

## Reference: CuteDSL in PyTorch

CuteDSL is the closest PyTorch precedent, but it is not a single mechanism. It
currently appears across multiple surfaces:

| Surface | CuteDSL examples | Lesson for FlyDSL |
|---|---|---|
| Native/eager | TopK, ScatterAdd, fused RMSNorm through QuACK | Optional runtime, cheap predicates, lazy imports, aten fallback. |
| Inductor templates | grouped GEMM, FlexAttention forward/backward, FlexGEMM epilogue | Template choices, generated wrappers, async compile/load, scheduling. |
| Universal GEMM | NVGEMM through `cutlass_api` for GEMM/scaled GEMM/grouped GEMM candidates | Registry-style kernel discovery and heuristic ranking can be a later maturity goal. |
| Runtime/cache policy | optional package installs, runtime gates, compile/cache helpers | Missing runtime must not change PyTorch default behavior. |

The reusable pattern is the optional DSL architecture, not CUDA-specific details.
FlyDSL should replace CUDA/CUTLASS assumptions with ROCm/HIP and `gfx`-specific
gates.

## Native / Eager Design

### Technical Principle

Native/eager integration is a dispatcher override with fallback:

```mermaid
flowchart LR
    A["import torch"] --> B["register DSL identity"]
    B --> C["register op override"]
    C --> D["user calls aten op"]
    D --> E["cheap cond"]
    E -->|true| F["lazy FlyDSL impl"]
    E -->|false| G["aten fallback"]
```

The key invariants are:

- `import torch` must not import FlyDSL or initialize ROCm runtime state.
- The predicate must be cheap, schema-compatible, and side-effect free.
- Unsupported cases return `False` and use aten fallback.
- The implementation may lazily import FlyDSL and use compile/cache helpers.
- Users must have a rollback path through `python_native`.

### FlyDSL Native Prototype

The first native prototype is RMSNorm:

```mermaid
flowchart LR
    A["FlyDSL runtime gate"] --> B["DSL identity registered"]
    B --> C["RMSNorm override"]
    C --> D["RMSNorm cond"]
    D --> E["lazy FlyDSL kernel import"]
    E --> F["compile cache + launch"]
```

RMSNorm is a good first native target because it has an existing aten surface,
bounded support predicates, existing FlyDSL tests, and a smaller behavioral
surface than GEMM or MoE.

Native follow-up work should include:

- stabilizing runtime and version gates;
- adding ROCm CI smoke coverage;
- adding OpInfo and user-control tests;
- measuring native cold compile, warm cache, and fallback behavior;
- evaluating additional native ops only when eager execution has clear value.

## Inductor / Compiler Design

### Technical Principle

Inductor integration is compile-time backend selection:

```mermaid
flowchart LR
    A["torch.compile"] --> B["FX graph"]
    B --> C["aten lowering"]
    C --> D["backend choices"]
    D --> E["benchmark requests"]
    E --> F["autotune winner"]
    F --> G["generated runtime code"]
```

A backend candidate needs:

| Object | Role |
|---|---|
| Lowering hook | Decides whether to append backend choices for an op. |
| ChoiceCaller | Represents one candidate implementation. |
| BenchmarkRequest | Runs the candidate during autotune. |
| TemplateBuffer | Stores the selected template output in Inductor IR. |
| Scheduling backend | Emits compile/load/runtime code for the selected template. |
| Async compile entry | Loads generated source and returns a runtime wrapper. |

### Current FlyDSL hgemm Prototype

The current Inductor prototype targets bf16 hgemm for `aten.mm(A, B.T)`:

```mermaid
flowchart LR
    A["aten.mm lowering"] --> B["FlyDSL eligibility gate"]
    B --> C["hgemm configs"]
    C --> D["Jinja wrapper"]
    D --> E["async_compile.flydsl"]
    E --> F["FlyDSL hgemm kernel"]
```

Current support matrix:

| Topic | Current decision |
|---|---|
| Op | `aten.mm(A, B.T)` style matmul. Inductor sees RHS as a `[K, N]` transpose view and the wrapper adapts it to FlyDSL's `[N, K]` expectation. |
| Dtype | Current report/prototype focus is bf16 hgemm. |
| Shape | Static 2D inputs. `N` must be divisible by `TILE_N`, `K` by `TILE_K`, and `K // SPLIT_K // TILE_K >= STAGES`. |
| Layout | `mat1` and output are row-major along K/N; RHS must match the transpose-view pattern. |
| Autotune | Multiple FlyDSL hgemm configs are emitted and benchmarked. |
| Fallback | If any gate fails, no FlyDSL choice is appended. Existing Inductor choices continue unchanged. |

### Kernel and Wrapper Boundary

The first hgemm template should be treated as the seed of a GEMM family. PyTorch
and FlyDSL should keep a clear boundary:

| Layer | Responsibility |
|---|---|
| Inductor lowering | Decide whether a FlyDSL family is eligible and append choices. |
| Template heuristics | Generate/prune configs before autotune. |
| Jinja wrapper | Adapt PyTorch tensors, layouts, streams, semaphores, and compile-time constants. |
| Vendored FlyDSL kernel | Implement the DSL kernel without PyTorch tensor-specific adaptation. |
| FlyDSL runtime | Own compiler/runtime internals and artifact format. |

## GEMM Family Expansion Design

Future GEMM expansion should be organized by kernel family and support matrix
instead of accumulating unrelated flags in one template.

```mermaid
flowchart LR
    A["bf16 hgemm"] --> B["fp16 hgemm"]
    B --> C["fp8 / scaled GEMM"]
    C --> D["layout variants"]
    D --> E["grouped / expert GEMM"]
    E --> F["MoE workload integration"]
```

| Area | Direction | Boundary |
|---|---|---|
| Dtype families | Add fp16, fp8, scaled GEMM, and mixed precision as separate families when tensor metadata or compile-time parameters differ. | PyTorch selects the family through dtype/layout gates; FlyDSL owns kernel implementation. |
| Layout families | Start with row-major A plus transpose-view B. Add contiguous B, prepacked B, and other stride forms only after wrapper ABI is explicit. | Jinja wrapper adapts PyTorch layouts; kernel code remains DSL-centric. |
| Shape regimes | Separate small-M decode, medium GEMM, and large prefill-like shapes in config heuristics. | Inductor prunes configs before autotune. |
| Config search | Move from a small manual config set to structured search over tile shape, stages, split-K, warp partitioning, and LDS policy. | PyTorch owns search/pruning policy; FlyDSL owns legal parameter space. |
| Grouped/expert GEMM | Add after single GEMM stabilizes. Define metadata for group offsets, per-group shapes, strides, and workspace. | Inductor wrapper owns metadata construction; FlyDSL kernel owns execution. |
| Fusion/epilogue | Add bias, activation, scale, and store epilogues after unfused GEMM is stable. | Inductor decides fusion profitability; FlyDSL exposes epilogue-capable APIs. |
| Cache/AOT | Define stable compile keys by dtype, layout family, tile config, arch, and kernel family. | FlyDSL owns artifact format; Inductor owns selection/generation cache. |

Implementation guidance:

- Keep PyTorch tensor adaptation in generated wrappers.
- Avoid one monolithic `compile_gemm_kernel(...)` with unrelated flags for bf16,
  fp8, scaled, and grouped kernels.
- Prefer a small family dispatcher in the wrapper that calls separate stable
  compile helpers.
- Add each dtype/layout family with focused correctness, generated-code, and
  autotune tests before broadening the lowering gate.

## Rollout Plan

Native/eager and Inductor/compiler work should be reviewed as separate tracks
while sharing runtime policy and CI setup.

| Stage | Native / eager track | Inductor / compiler track | Exit criteria |
|---:|---|---|---|
| 1 | Register FlyDSL as an optional DSL; add unavailable-runtime tests. | None. | Import safety and no default behavior change. |
| 2 | Add ROCm CI install helper and smoke test. | Reuse CI install path for compiler tests. | Package/runtime viability. |
| 3 | Add RMSNorm native adapter and kernel wrapper. | None. | Correctness, fallback, cache behavior. |
| 4 | Add OpInfo, user-control tests, and cold/warm cache reporting. | None. | Native test coverage and observable runtime behavior. |
| 5 | Keep native track stable and independent. | Land bf16 hgemm template prototype. | Template rendering, async compile/load, autotune, narrow support matrix. |
| 6 | Evaluate next native op only with benchmark evidence. | Expand GEMM dtype/layout support: fp16, fp8/scaled GEMM, more shapes/gfx targets. | Separate support matrices and config pruning. |
| 7 | Continue fallback and OpInfo coverage for any new native op. | Add grouped/expert GEMM template and metadata ABI. | Group offsets, per-group shapes, workspace, benchmarks. |
| 8 | Keep native wrappers focused on eager semantics. | Add fusion/epilogue, persistent cache, and AOT exploration. | Predictable compile/runtime behavior. |
| 9 | Revisit native expansion only for ops with real eager value. | Explore FlexAttention-style kernels, attention-adjacent matmul APIs, and MoE through grouped GEMM. | Maintainable workload-level path. |

## Validation and Acceptance Criteria

Validation should match existing optional DSL patterns:

| Gate | Minimum proof | Failure behavior |
|---|---|---|
| Native registry | Missing FlyDSL is silent; native controls behave correctly; no runtime import. | FlyDSL remains unavailable. |
| Smoke | A tiny FlyDSL kernel compiles and runs on selected ROCm CI jobs. | Skip outside supported jobs. |
| Native op | Supported cases match aten/reference; unsupported cases fall back. | Predicate returns `False`. |
| Compiler | Generated source, async compile/load, runtime launch, autotune, fallback by omission. | No FlyDSL choice is appended. |
| GEMM expansion | Each dtype/layout family has support matrix, config search space, selected configs, correctness, and fallback coverage. | Family remains disabled in lowering. |
| Grouped GEMM | Grouped/expert GEMM correctness, metadata ABI, workspace behavior, and serving-shape benchmarks. | No MoE-level performance claims. |

Performance reporting should separate:

- cold compile cost;
- persistent-cache load cost;
- warm runtime latency;
- autotune benchmarking time;
- selected config and backend winner.

## Compatibility

Default behavior should not change.

| Environment | Expected behavior |
|---|---|
| CPU-only PyTorch | FlyDSL unavailable; no import error. |
| CUDA/NVIDIA PyTorch | FlyDSL unavailable because `torch.version.hip is None`. |
| ROCm PyTorch without FlyDSL | FlyDSL unavailable; aten behavior unchanged. |
| ROCm PyTorch with supported FlyDSL | Eligible calls may use FlyDSL; unsupported calls use aten or existing Inductor choices. |
| User disables FlyDSL native overrides | Eager/native FlyDSL overrides are disabled; Inductor controls remain separate. |

## Implementation Checklist

Before opening PyTorch PRs, each stage should be able to answer:

| Check | Expected answer |
|---|---|
| Import safety | `import torch` does not import FlyDSL, initialize ROCm, or query device properties. |
| Optional dependency | Missing or unsupported FlyDSL leaves PyTorch behavior unchanged. |
| Native rollback | `torch.backends.python_native.flydsl.enabled = False` restores eager/native aten behavior. |
| First op schema | The native adapter names the exact aten op, overload, outputs, and autograd behavior. |
| Native support matrix | dtype, shape, layout, and `gfx` targets match tested coverage. |
| Native cache | Compile keys exclude runtime tensors and streams where possible. |
| Inductor hgemm | Focused `torch.compile` test covers generated source, `async_compile.flydsl`, runtime correctness, and multi-choice autotune. |
| GEMM family boundaries | bf16/fp16/fp8/scaled/grouped kernels have documented wrapper ABI, compile-time parameters, and tests before enabling in lowering. |
| Roadmap split | Native/eager and Inductor/compiler work are tracked independently with shared optional-runtime policy. |

## Resolution / Next Steps

If accepted, implementation should proceed through the rollout plan above. The
native/eager track should prove optional-runtime import safety, user controls,
schema-compatible adapters, and fallback. The Inductor/compiler track should
keep the hgemm support matrix narrow, host only reviewed template/kernel glue in
PyTorch, and rely on the optional FlyDSL package for compiler/runtime services.

The next major design work after the current prototypes is GEMM family expansion:
fp16, fp8/scaled GEMM, layout variants, structured config search, and
grouped/expert GEMM as the foundation for future workload integration.
