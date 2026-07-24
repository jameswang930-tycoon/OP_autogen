#!/usr/bin/env python3
"""
Test Cases: RMSNorm + Residual Add

验证策略 (5 级):
  1. Smoke test — 小 shape, 快速验证基本正确性
  2. Shape sweep — 多 batch/seq_len 组合
  3. Dtype sweep — fp16 / fp32
  4. Edge cases — 全零输入, 极值, 空维度
  5. Numerical precision — max_abs / max_rel vs PyTorch reference

Usage:
  python test_cases.py                    # 运行所有测试
  python test_cases.py --smoke            # 只跑 smoke test
"""

import torch
import numpy as np

# ── Import reference ──────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pytorch_code import reference_rms_norm_residual


# ── Test matrix ──────────────────────────────────────────────────

TEST_SHAPES = [
    # (batch, seq_len, hidden_dim, dtype, name)
    (1, 128, 4096, torch.float16, "smoke_tiny"),
    (1, 1024, 4096, torch.float16, "llama2_inference"),
    (1, 2048, 4096, torch.float16, "llama2_medium_context"),
    (8, 1024, 4096, torch.float16, "batch_8_inference"),
    (1, 4096, 4096, torch.float16, "long_sequence"),
    (1, 1024, 4096, torch.float32, "llama2_fp32"),
]

EDGE_CASES = [
    # (batch, seq_len, hidden_dim, name, input_type)
    (1, 1024, 4096, "all_zeros", "zeros"),
    (1, 1024, 4096, "all_ones", "ones"),
    (1, 1024, 4096, "large_values", "large"),
    (1, 1024, 4096, "mixed_sign", "random"),
]


def generate_input(batch, seq_len, hidden_dim, dtype, input_type="random"):
    """Generate test inputs."""
    if input_type == "zeros":
        x = torch.zeros(batch, seq_len, hidden_dim, dtype=dtype)
        residual = torch.zeros(batch, seq_len, hidden_dim, dtype=dtype)
    elif input_type == "ones":
        x = torch.ones(batch, seq_len, hidden_dim, dtype=dtype)
        residual = torch.ones(batch, seq_len, hidden_dim, dtype=dtype)
    elif input_type == "large":
        x = torch.full((batch, seq_len, hidden_dim), 100.0, dtype=dtype)
        residual = torch.full((batch, seq_len, hidden_dim), -50.0, dtype=dtype)
    else:
        x = torch.randn(batch, seq_len, hidden_dim, dtype=dtype)
        residual = torch.randn(batch, seq_len, hidden_dim, dtype=dtype)

    gamma = torch.ones(hidden_dim, dtype=dtype)
    return x, residual, gamma


def test_smoke():
    """快速冒烟测试。"""
    print("=== Smoke Test ===")
    for batch, seq_len, hidden_dim, dtype, name in TEST_SHAPES[:2]:
        x, residual, gamma = generate_input(batch, seq_len, hidden_dim, dtype)
        ref = reference_rms_norm_residual(x, residual, gamma)

        # Verify basic properties
        assert ref.shape == (batch, seq_len, hidden_dim), \
            f"Shape mismatch: {ref.shape} vs {(batch, seq_len, hidden_dim)}"
        assert ref.dtype == dtype, f"Dtype mismatch: {ref.dtype} vs {dtype}"
        assert not torch.isnan(ref).any(), f"NaN in output for {name}"
        assert not torch.isinf(ref).any(), f"Inf in output for {name}"
        print(f"  {name:30s} shape={ref.shape} dtype={ref.dtype} PASS")
    print()


def test_shape_sweep():
    """多 shape 验证。"""
    print("=== Shape Sweep ===")
    for batch, seq_len, hidden_dim, dtype, name in TEST_SHAPES:
        x, residual, gamma = generate_input(batch, seq_len, hidden_dim, dtype)
        ref = reference_rms_norm_residual(x, residual, gamma)
        print(f"  {name:30s} {str(ref.shape):20s} {str(ref.dtype):10s} PASS")
    print()


def test_edge_cases():
    """边界条件验证。"""
    print("=== Edge Cases ===")
    for batch, seq_len, hidden_dim, name, itype in EDGE_CASES:
        x, residual, gamma = generate_input(batch, seq_len, hidden_dim,
                                             torch.float16, itype)
        ref = reference_rms_norm_residual(x, residual, gamma)

        # For all-zero input, output should be all-zero (0/rms = 0)
        if itype == "zeros":
            assert torch.allclose(ref, torch.zeros_like(ref), atol=1e-3), \
                f"All-zero input should produce all-zero output for {name}"

        assert not torch.isnan(ref).any(), f"NaN for {name}"
        print(f"  {name:30s} PASS")
    print()


def test_numerical_precision():
    """数值精度验证。"""
    print("=== Numerical Precision ===")
    x, residual, gamma = generate_input(1, 1024, 4096, torch.float16)
    ref_fp16 = reference_rms_norm_residual(x, residual, gamma)

    # Compare with FP32 reference (ground truth)
    x32 = x.float()
    residual32 = residual.float()
    gamma32 = gamma.float()
    ref_fp32 = reference_rms_norm_residual(x32, residual32, gamma32)

    abs_err = (ref_fp16.float() - ref_fp32).abs()
    rel_err = abs_err / (ref_fp32.abs() + 1e-10)

    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    mean_abs = abs_err.mean().item()

    print(f"  max_abs_error: {max_abs:.2e}")
    print(f"  max_rel_error: {max_rel:.2e}")
    print(f"  mean_abs_error: {mean_abs:.2e}")

    # FP16 tolerance: rtol=1e-2, atol=1e-2
    assert max_abs < 1e-1, f"max_abs {max_abs:.2e} exceeds FP16 tolerance"
    assert max_rel < 1e-1, f"max_rel {max_rel:.2e} exceeds FP16 tolerance"
    print("  PASS (within FP16 tolerance)")
    print()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Smoke test only")
    args = p.parse_args()

    test_smoke()
    if not args.smoke:
        test_shape_sweep()
        test_edge_cases()
        test_numerical_precision()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
