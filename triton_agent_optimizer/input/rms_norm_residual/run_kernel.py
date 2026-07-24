#!/usr/bin/env python3
"""
Profiling 启动脚本 — 带 warmup + repeat 基准测试。

Usage:
  msprof op simulator --kernel-name=add_rms_norm_kernel --soc-version=Ascend910B3 python3 run_kernel.py
  msprof op --kernel-name=add_rms_norm_kernel python3 run_kernel.py

  不带 msprof 时直接跑 benchmark:
  python3 run_kernel.py
"""

import time
import torch
import triton
from triton_kernel import add_rms_norm_kernel

WARMUP = 30
REPEAT = 200

B, S, H = 1, 1024, 4096
dtype = torch.float16
device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"

# Inputs
x = torch.randn(B * S * H, dtype=dtype, device=device)
residual = torch.randn(B * S * H, dtype=dtype, device=device)
gamma = torch.ones(H, dtype=dtype, device=device)
out = torch.empty(B * S * H, dtype=dtype, device=device)

BLOCK_SIZE = 1024
grid = (B * S,)


def sync():
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ── Warmup ─────────────────────────────────────────────────────
for _ in range(WARMUP):
    add_rms_norm_kernel[grid](x, residual, gamma, out, H, 1e-6, BLOCK_SIZE=BLOCK_SIZE)
sync()

# ── Benchmark ──────────────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(REPEAT):
    add_rms_norm_kernel[grid](x, residual, gamma, out, H, 1e-6, BLOCK_SIZE=BLOCK_SIZE)
sync()
elapsed_ms = (time.perf_counter() - t0) / REPEAT * 1e3

print(f"RMSNorm+Residual (B={B}, S={S}, H={H}, fp16)")
print(f"  warmup={WARMUP}  repeat={REPEAT}")
print(f"  avg latency: {elapsed_ms * 1e3:.1f} us")
print(f"  throughput:  {1 / elapsed_ms * 1e3:.0f} calls/s")
