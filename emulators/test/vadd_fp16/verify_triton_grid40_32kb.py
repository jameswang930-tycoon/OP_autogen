#!/usr/bin/env python3
"""
vadd 上板验证 — grid=40 / tile=32KB / fp16 / 含 store
=====================================================
grid=40 (40 program), 每 program 32KB chunk (16384 elem fp16), BLOCK=16384
总数据 40×32KB = 1.28MB
UB peak (ub_1+ub_2 = 64KB) < 192KB (不 overflow)

cost model 单核饱和曲线预测 (含 store): total 913.89 ns (≈0.91 us)
  load(GM→UB)  405.4 ns (44%)
  store(UB→GM) 427.4 ns (47%)
  vadd(VecUnit) 81.1 ns (9%)
注: 单核预测, grid=40 上板时 40 核并行, wall-clock ≈ 单 program + 多核开销

Run: python3 verify_triton_grid40_32kb.py
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
N_PER_PROG = 16384              # 32KB / 2B = 16384 elem per program
BLOCK      = 16384              # 32KB tile (saturated)
GRID       = 40
SCALAR     = 3.0
N_TOTAL    = GRID * N_PER_PROG  # 655360 elem = 1.28MB


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
    print(f"config: grid={GRID}, tile=32KB (BLOCK={BLOCK}), "
          f"total={N_TOTAL} elem ({N_TOTAL*2//1024}KB), fp16")
    print(f"correctness: {'PASS' if ok else 'FAIL'} (max_err={max_err:.2e})")
    print(f"实测 wall-clock: {ms*1e3:.3f} us")
    print(f"cost model 单核预测 total: 0.91 us (913.89 ns)")
    print(f"  load  (GM→UB): 405.4 ns (44%)")
    print(f"  store (UB→GM): 427.4 ns (47%)")
    print(f"  vadd  (Vec)  :  81.1 ns (9%)")


if __name__ == "__main__":
    main()
