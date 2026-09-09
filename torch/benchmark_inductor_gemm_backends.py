import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code


SHAPES = [
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (32, 14336, 4096),
    (16, 28672, 4096),
    (4096, 256, 4096),
]


BACKEND_PATCHES = {
    "aten": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN",
        "max_autotune_gemm_search_space": "DEFAULT",
    },
    "triton": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
        "max_autotune_gemm_search_space": "EXHAUSTIVE",
    },
    "flydsl": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "FLYDSL",
        "max_autotune_gemm_search_space": "EXHAUSTIVE",
        "flydsl_enable_autotuning": True,
    },
}


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def mm_nt(a, b):
    return torch.mm(a, b.t())


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1.0e9)


def cuda_event_bench(fn, a, b, warmup, reps, rounds):
    samples = []
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()

    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn(a, b)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / reps)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def read_completed(path):
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(
            (row["backend"], row.get("dtype", "bfloat16"), row["m"], row["n"], row["k"])
        )
    return completed


def run_case(backend, m, n, k, args):
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    dtype = DTYPES[args.dtype]
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype)

    patch = dict(BACKEND_PATCHES[backend])
    compile_start = time.perf_counter()
    with inductor_config.patch(**patch):
        compiled = torch.compile(mm_nt, backend="inductor")
        result, source_codes = run_and_get_code(compiled, a, b)
    compile_s = time.perf_counter() - compile_start

    ref = mm_nt(a, b)
    max_abs = (result - ref).abs().max().item()
    ok = torch.allclose(result, ref, atol=3e-2, rtol=3e-2)
    source = "\n".join(source_codes)
    uses_flydsl = "_flydsl_mm" in source
    uses_triton = "triton_" in source or "@triton" in source
    uses_aten = "extern_kernels.mm" in source or "aten.mm" in source or "at::mm" in source

    bench = cuda_event_bench(
        compiled,
        a,
        b,
        warmup=args.warmup,
        reps=args.reps,
        rounds=args.rounds,
    )
    median_ms = bench["median_ms"]
    return {
        "backend": backend,
        "m": m,
        "n": n,
        "k": k,
        "dtype": args.dtype,
        "ok": bool(ok),
        "max_abs": max_abs,
        "compile_s": compile_s,
        "median_ms": median_ms,
        "mean_ms": bench["mean_ms"],
        "min_ms": bench["min_ms"],
        "samples_ms": bench["samples_ms"],
        "tflops": tflops(m, n, k, median_ms),
        "uses_flydsl": uses_flydsl,
        "uses_triton": uses_triton,
        "uses_aten": uses_aten,
        "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["aten", "triton", "flydsl", "all"],
        default="all",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--shape-index", type=int, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(out_path) if args.resume else set()
    backends = ["aten", "triton", "flydsl"] if args.backend == "all" else [args.backend]
    shapes = SHAPES if args.shape_index is None else [SHAPES[args.shape_index]]

    print("output", out_path, flush=True)
    print("backends", backends, flush=True)
    print("shapes", shapes, flush=True)

    with out_path.open("a", buffering=1) as f:
        for backend in backends:
            for m, n, k in shapes:
                key = (backend, args.dtype, m, n, k)
                if key in completed:
                    print("skip", key, flush=True)
                    continue
                print("run", key, flush=True)
                try:
                    row = run_case(backend, m, n, k, args)
                except Exception as exc:
                    row = {
                        "backend": backend,
                        "m": m,
                        "n": n,
                        "k": k,
                        "ok": False,
                        "error": repr(exc),
                        "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                    }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    "done",
                    key,
                    row.get("median_ms"),
                    row.get("tflops"),
                    row.get("error"),
                    flush=True,
                )


if __name__ == "__main__":
    main()
