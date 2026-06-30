# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""PyTorch-facing RMSNorm wrapper with FlyDSL-owned compile caching."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from threading import Lock

import torch

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.rmsnorm_kernel import EPS as _KERNEL_EPS
from kernels.rmsnorm_kernel import build_rmsnorm_module


_SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
_SUPPORTED_HIDDEN_SIZES = frozenset({128, 256, 512, 1024, 2000, 2048, 4096, 8192})

RmsNormCacheInfo = namedtuple("RmsNormCacheInfo", ["hits", "misses", "currsize"])


@dataclass(frozen=True)
class _RmsNormKey:
    n: int
    dtype: str
    arch: str
    backend: str
    variant: str = "forward"


_compiled_rmsnorm_cache: dict[_RmsNormKey, flyc.CompiledFunction] = {}
_cache_lock = Lock()
_cache_hits = 0
_cache_misses = 0


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _canonical_normalized_shape(normalized_shape) -> tuple[int, ...]:
    if isinstance(normalized_shape, torch.Size):
        return tuple(int(x) for x in normalized_shape)
    if isinstance(normalized_shape, (tuple, list)):
        return tuple(int(x) for x in normalized_shape)
    return (int(normalized_shape),)


def _eps_supported(eps: float | None) -> bool:
    return eps is not None and float(eps) == _KERNEL_EPS


def is_rmsnorm_supported(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> bool:
    """Return whether the current FlyDSL RMSNorm wrapper can handle this call."""

    shape = _canonical_normalized_shape(normalized_shape)
    if len(shape) != 1:
        return False
    n = shape[0]
    if n not in _SUPPORTED_HIDDEN_SIZES:
        return False
    if input.device.type != "cuda" or torch.version.hip is None:
        return False
    if input.requires_grad or (weight is not None and weight.requires_grad):
        return False
    if input.dtype not in _SUPPORTED_DTYPES:
        return False
    if input.ndim < 1 or input.shape[-1] != n:
        return False
    if not input.is_contiguous():
        return False
    if weight is None:
        return False
    if weight.shape != (n,) or weight.dtype != input.dtype or weight.device != input.device:
        return False
    if not weight.is_contiguous():
        return False
    if not _eps_supported(eps):
        return False
    return True


def _make_compile_arg(tensor: torch.Tensor):
    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


def _get_compiled(
    key: _RmsNormKey,
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    output_2d: torch.Tensor,
    rows_m: int,
    stream,
) -> flyc.CompiledFunction:
    global _cache_hits, _cache_misses

    compiled = _compiled_rmsnorm_cache.get(key)
    if compiled is not None:
        _cache_hits += 1
        return compiled

    with _cache_lock:
        compiled = _compiled_rmsnorm_cache.get(key)
        if compiled is not None:
            _cache_hits += 1
            return compiled

        _cache_misses += 1
        launch = build_rmsnorm_module(key.n, key.dtype)
        compiled = flyc.compile(
            launch,
            _make_compile_arg(input_2d),
            flyc.from_torch_tensor(weight),
            _make_compile_arg(output_2d),
            rows_m,
            stream,
        )
        _compiled_rmsnorm_cache[key] = compiled
        return compiled


def rmsnorm(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    """Run FlyDSL RMSNorm for a narrow, inference-only PyTorch-compatible surface."""

    if not is_rmsnorm_supported(input, normalized_shape, weight, eps):
        raise ValueError("unsupported FlyDSL RMSNorm input; PyTorch adapter should fall back to aten")

    shape = _canonical_normalized_shape(normalized_shape)
    n = shape[0]
    rows_m = input.numel() // n
    output = torch.empty_like(input)

    with torch.cuda.device(input.device):
        input_2d = input.reshape(rows_m, n)
        output_2d = output.reshape(rows_m, n)
        stream = torch.cuda.current_stream(input.device)
        key = _RmsNormKey(
            n=n,
            dtype=_dtype_str(input.dtype),
            arch=str(get_rocm_arch()),
            backend=flyc.compile_backend_name(),
        )
        compiled = _get_compiled(key, input_2d, weight, output_2d, rows_m, stream)
        compiled(input_2d, weight, output_2d, rows_m, stream)

    return output


def clear_rmsnorm_cache() -> None:
    global _cache_hits, _cache_misses
    with _cache_lock:
        _compiled_rmsnorm_cache.clear()
        _cache_hits = 0
        _cache_misses = 0


def rmsnorm_cache_info() -> RmsNormCacheInfo:
    with _cache_lock:
        return RmsNormCacheInfo(_cache_hits, _cache_misses, len(_compiled_rmsnorm_cache))
