#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for the PyTorch-facing FlyDSL RMSNorm wrapper."""

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU is required for FlyDSL RMSNorm wrapper tests.", allow_module_level=True)

pytest.importorskip("flydsl._mlir", reason="FlyDSL MLIR runtime is not built or installed")

from flydsl.torch import clear_rmsnorm_cache, is_rmsnorm_supported, rmsnorm, rmsnorm_cache_info


def _reference_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = input.float()
    w = weight.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * w).to(input.dtype)


def test_rmsnorm_support_predicate():
    x = torch.randn((8, 128), device="cuda", dtype=torch.float16)
    w = torch.randn((128,), device="cuda", dtype=torch.float16)

    assert is_rmsnorm_supported(x, (128,), w, 1e-5)
    assert not is_rmsnorm_supported(x, (128,), None, 1e-5)
    assert not is_rmsnorm_supported(x.t(), (8,), torch.randn((8,), device="cuda", dtype=torch.float16), 1e-5)
    assert not is_rmsnorm_supported(x, (128,), w, 1e-6)


def test_rmsnorm_wrapper_correctness_and_cache_reuse():
    clear_rmsnorm_cache()

    x = torch.randn((16, 128), device="cuda", dtype=torch.float16)
    w = torch.randn((128,), device="cuda", dtype=torch.float16)

    out = rmsnorm(x, (128,), w, 1e-5)
    torch.testing.assert_close(out, _reference_rmsnorm(x, w), atol=1e-2, rtol=1e-2)

    x2 = torch.randn((32, 128), device="cuda", dtype=torch.float16)
    out2 = rmsnorm(x2, (128,), w, 1e-5)
    torch.testing.assert_close(out2, _reference_rmsnorm(x2, w), atol=1e-2, rtol=1e-2)

    x3 = torch.randn((2, 16, 128), device="cuda", dtype=torch.float16)
    out3 = rmsnorm(x3, (128,), w, 1e-5)
    torch.testing.assert_close(out3, _reference_rmsnorm(x3, w), atol=1e-2, rtol=1e-2)

    info = rmsnorm_cache_info()
    assert info.misses == 1
    assert info.hits >= 2
    assert info.currsize == 1
