#!/usr/bin/env python3
"""
Baseline Triton Kernel: RMSNorm + Residual Add — Llama-2-7B Transformer Norm

Multi-operator: 3 ops
  op0: gm_to_ub — load input + residual to UB
  op1: compute  — residual add + RMSNorm (two-pass: mean(x^2) + scale)
  op2: ub_to_gm — store result

Target: Huawei Ascend 910B3, fp16, hidden_dim=4096, BLOCK_SIZE=1024
"""

import triton
import triton.language as tl


@triton.jit
def add_rms_norm_kernel(
    x_ptr,          # input tensor [batch, seq_len, hidden_dim]
    residual_ptr,   # residual tensor [batch, seq_len, hidden_dim]
    gamma_ptr,      # gamma weight [hidden_dim]
    out_ptr,        # output tensor [batch, seq_len, hidden_dim]
    hidden_dim,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one row (seq position).
    Two-pass RMSNorm: sum(x^2) → rms → scale.
    """
    pid = tl.program_id(0)
    row_idx = pid

    # ── Pass 1: Load + Add + Compute sum(x^2) ──
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim

    x_offs = row_idx * hidden_dim + offs
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

    # Residual add
    combined = x + residual

    # Two-pass RMSNorm: compute mean of squares
    sum_sq = tl.sum(combined * combined, axis=0)
    n = tl.min(hidden_dim, BLOCK_SIZE)
    mean_sq = sum_sq / n
    rms = tl.sqrt(mean_sq + eps)

    # ── Pass 2: Normalize + Scale by gamma ──
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    normalized = (combined / rms) * gamma

    # Store
    out_offs = row_idx * hidden_dim + offs
    tl.store(out_ptr + out_offs, normalized.to(tl.float16), mask=mask)


@triton.jit
def rms_norm_kernel(
    x_ptr,
    gamma_ptr,
    out_ptr,
    hidden_dim,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm without residual — baseline for comparison."""
    pid = tl.program_id(0)
    row_idx = pid

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim

    x_offs = row_idx * hidden_dim + offs
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    n = tl.min(hidden_dim, BLOCK_SIZE)
    rms = tl.sqrt(sum_sq / n + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    normalized = (x / rms) * gamma

    out_offs = row_idx * hidden_dim + offs
    tl.store(out_ptr + out_offs, normalized.to(tl.float16), mask=mask)


# ── DSL Pipeline (for cost simulator analysis) ──────────────────
# 3 ops in our pipeline:
#   alloc(gm_in, 128KB)  alloc(gm_res, 128KB)  alloc(gm_gamma, 8KB)
#   alloc(ub_x, 4KB)  alloc(ub_res, 4KB)  alloc(ub_combined, 4KB)
#   gm_to_ub(ub_x, gm_in)
#   gm_to_ub(ub_res, gm_res)
#   vadd(ub_combined, ub_x, ub_res)    # residual add
#   compute_rms_norm(ub_combined)      # RMSNorm (custom)
#   ub_to_gm(gm_out, ub_combined)

DSL_PROGRAM = """
alloc(gm_in, 128KB) alloc(gm_residual, 128KB) alloc(gm_gamma, 8KB)
alloc(ub_x, 4KB) alloc(ub_res, 4KB) alloc(ub_combined, 4KB)
gm_to_ub(ub_x, gm_in)
gm_to_ub(ub_res, gm_residual)
vadd(ub_combined, ub_x, 1.0)
ub_to_gm(gm_out, ub_combined)
"""
