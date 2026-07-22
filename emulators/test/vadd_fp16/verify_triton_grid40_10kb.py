#!/usr/bin/env python3
"""
vadd 上板验证 — grid=40 / tile=10KB / fp16 / 含 store
=====================================================
grid=40 (40 program), 每 program 10KB chunk (5120 elem fp16), BLOCK=8192 masked to 5120
总数据 40×10KB = 400KB
UB peak (ub_1+ub_2 = 20KB) < 192KB (不 overflow)

cost model 单核饱和曲线预测 (含 store): total 306.58 ns (≈0.31 us)
  load(GM→UB)  140.8 ns (46%)
  store(UB→GM) 133.6 ns (44%)
  vadd(VecUnit) 32.2 ns (11%)
注: 单核预测, grid=40 上板时 40 核并行, wall-clock ≈ 单 program + 多核开销

Run: python3 verify_triton_grid40_10kb.py
"""

import time
import torch
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:
    pass

DTYPE      = torch.float16
N_PER_PROG = 5120               # 10KB / 2B = 5120 elem per program
BLOCK      = 8192               # next_power_of_2(5120), mask 到 5120
GRID       = 40
SCALAR     = 3.0
N_TOTAL    = GRID * N_PER_PROG  # 204800 elem = 400KB


@triton.jit
def vadd_kernel(in_ptr, out_ptr, n_per_prog,
                SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    base = pid * n_per_prog
    offs = tl.arange(0, BLOCK)
    mask = offs < n_per_prog
    x = tl.load(in_ptr + base + offs, mask=mask)
    y = x + SCALAR
    tl.store(out_ptr + base + offs, y, mask=mask)


def pick_device():
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch, 'npu') and torch.npu.is_available(): return 'npu'
    raise SystemExit("No CUDA/NPU device.")


def sync(device):
    (torch.npu if device == 'npu' else torch.cuda).synchronize()


def bench(fn, device, warmup=50, rep=200):
    for _ in range(warmup): fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(rep): fn()
    sync(device)
    return (time.perf_counter() - t0) / rep * 1e3


def main():
    device = pick_device()
    torch.manual_seed(0)
    x   = torch.randn(N_TOTAL, dtype=DTYPE, device=device)
    out = torch.empty_like(x)

    def run():
        vadd_kernel[(GRID,)](x, out, N_PER_PROG, SCALAR=SCALAR, BLOCK=BLOCK)

    run(); sync(device)
    ref = x + SCALAR
    ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
    max_err = (out.float() - ref.float()).abs().max().item()
    ms = bench(run, device)
    print(f"device: {device}")
    print(f"config: grid={GRID}, tile=10KB (BLOCK={BLOCK} mask to {N_PER_PROG}), "
          f"total={N_TOTAL} elem ({N_TOTAL*2//1024}KB), fp16")
    print(f"correctness: {'PASS' if ok else 'FAIL'} (max_err={max_err:.2e})")
    print(f"实测 wall-clock: {ms*1e3:.3f} us")
    print(f"cost model 单核预测 total: 0.31 us (306.58 ns)")
    print(f"  load  (GM→UB): 140.8 ns (46%)")
    print(f"  store (UB→GM): 133.6 ns (44%)")
    print(f"  vadd  (Vec)  :  32.2 ns (11%)")


if __name__ == "__main__":
    main()
