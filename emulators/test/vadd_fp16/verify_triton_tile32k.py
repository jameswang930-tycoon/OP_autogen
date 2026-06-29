#!/usr/bin/env python3
"""
vadd 上板验证 — tile=32KB / fp16 / 含 store 完整通路
=====================================================
两个配置, 对照 cost model 单核口径预测 (各通路时延 + 占比):

  总32KB (grid=1): total 0.97 us
    load(GM→UB)  436.9 ns (45%)
    store(UB→GM) 448.6 ns (46%)
    vadd(Vec)     81.9 ns  (8%)

  总64KB (grid=2): total 1.40 us (单program搬64KB load; 实际grid=2并行应≈0.97)
    load(GM→UB)  873.8 ns (62%)
    store(UB→GM) 448.6 ns (32%)
    vadd(Vec)     81.9 ns  (6%)

Run: python3 verify_triton_tile32k.py   (需 torch + triton + 真实硬件 CUDA/NPU)
"""

import time

import torch
import triton
import triton.language as tl

try:                       # noqa: SIM105
    import torch_npu       # noqa: F401
except ImportError:
    pass

DTYPE  = torch.float16
BLOCK  = 16384          # 32KB tile (fp16, >= VecUnit 24KB 拐点, vec 吃满)
SCALAR = 3.0


@triton.jit
def vadd_kernel(in_ptr, out_ptr, n_per_prog,
                SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    base = pid * n_per_prog
    offs = tl.arange(0, BLOCK)
    mask = offs < n_per_prog
    x = tl.load(in_ptr + base + offs, mask=mask)      # gm_to_ub (32KB load)
    y = x + SCALAR                                     # vadd (32KB compute)
    tl.store(out_ptr + base + offs, y, mask=mask)      # ub_to_gm (32KB store)


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    raise SystemExit("No CUDA/NPU device found.")


def sync(device: str) -> None:
    (torch.npu if device == 'npu' else torch.cuda).synchronize()


def bench(fn, device: str, warmup: int = 50, rep: int = 200) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / rep * 1e3


def run_case(device, grid, n_per_prog, label, pred):
    n_total = grid * n_per_prog
    torch.manual_seed(0)
    x   = torch.randn(n_total, dtype=DTYPE, device=device)
    out = torch.empty_like(x)

    def run():
        vadd_kernel[(grid,)](x, out, n_per_prog, SCALAR=SCALAR, BLOCK=BLOCK)

    run()
    sync(device)
    ref = x + SCALAR
    ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
    max_err = (out.float() - ref.float()).abs().max().item()
    ms = bench(run, device)
    print(f"[{label}] grid={grid}, N={n_total} ({n_total*2//1024}KB), "
          f"BLOCK=16384(32KB), fp16")
    print(f"  correctness: {'PASS' if ok else 'FAIL'} (max_err={max_err:.2e})")
    print(f"  实测 wall-clock: {ms*1e3:.3f} us")
    print(f"  单核预测 total: {pred['total_us']} us")
    print(f"    load  (GM→UB): {pred['load_ns']:>6} ns  ({pred['load_pct']})")
    print(f"    store (UB→GM): {pred['store_ns']:>6} ns  ({pred['store_pct']})")
    print(f"    vadd  (Vec)  : {pred['vadd_ns']:>6} ns  ({pred['vadd_pct']})")
    print()


def main():
    device = pick_device()
    print(f"device: {device}\n")

    run_case(device, grid=1, n_per_prog=16384, label="总32KB", pred={
        'total_us': '0.97', 'load_ns': '436.9', 'load_pct': '45%',
        'store_ns': '448.6', 'store_pct': '46%',
        'vadd_ns': '81.9', 'vadd_pct': '8%'})

    run_case(device, grid=2, n_per_prog=16384, label="总64KB", pred={
        'total_us': '1.40', 'load_ns': '873.8', 'load_pct': '62%',
        'store_ns': '448.6', 'store_pct': '32%',
        'vadd_ns': '81.9', 'vadd_pct': '6%'})


if __name__ == "__main__":
    main()
