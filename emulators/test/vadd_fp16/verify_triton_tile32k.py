#!/usr/bin/env python3
"""
vadd 上板验证 — tile=32KB / fp16 / 含 store 完整通路
=====================================================
两个配置:
  总32KB: grid=1, BLOCK=16384 (32KB), N=16384
  总64KB: grid=2, BLOCK=16384 (32KB), N=32768

cost model 单核口径预测 (含 store):
  总32KB (grid=1): 0.97 us  (load 436.9 + store 448.6 + vadd 81.9)
  总64KB (grid=2): 1.40 us  (单program搬64KB load; 实际grid=2并行应≈0.97us)

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


def run_case(device, grid, n_per_prog, label, predict_us):
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
    print(f"  实测: {ms*1e3:.3f} us   单核预测: {predict_us} us")
    print()


def main():
    device = pick_device()
    print(f"device: {device}\n")
    run_case(device, grid=1, n_per_prog=16384, label="总32KB",
             predict_us="0.97")
    run_case(device, grid=2, n_per_prog=16384, label="总64KB",
             predict_us="1.40 (单program口径; grid=2并行应≈0.97)")


if __name__ == "__main__":
    main()
