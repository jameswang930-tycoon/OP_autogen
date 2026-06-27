#!/usr/bin/env python3
"""
vadd 上板验证 — grid=2 / tile=32KB / 总 64KB fp16 / 含 store 完整通路
=====================================================================
配置:
  grid=2 (2 program), 每 program 32KB chunk (16384 elem fp16)
  BLOCK=16384 (32KB tile), load + vadd + store 三段
  总数据 32768 elem = 64KB fp16
  UB 占用: load 32KB + vadd 32KB (中间) < 192KB (不 overflow)

cost model 单核口径预测 (含 store): total ~967 ns (0.97 us)
  - load 436.9 ns (45%), store 448.6 ns (46%), vadd 81.9 ns (8%)

对照: grid=2 上板时 2 核并行, 每核 1 个 32KB program.
如果 2 核全并行, wall-clock ≈ 单 program 时延 (~0.97 us 单核预测) + 多核开销.

Run: python3 verify_triton_grid2.py   (需 torch + triton + 真实硬件 CUDA/NPU)
"""

import time

import torch
import triton
import triton.language as tl

# Pull in the Ascend NPU torch backend if installed (registers torch.npu).
try:                       # noqa: SIM105
    import torch_npu       # noqa: F401
except ImportError:
    pass

# ── Geometry ───────────────────────────────────────────────────────────────────
DTYPE        = torch.float16
BYTES_PER_EL = 2
N_PER_PROG   = 16384                  # 32KB / 2B = 16384 elem per program
BLOCK        = 16384                  # 32KB tile (>= VecUnit 24KB 拐点, vec 吃满)
GRID         = 2
SCALAR       = 3.0
N_TOTAL      = GRID * N_PER_PROG      # 32768 elem = 64KB


@triton.jit
def vadd_kernel(in_ptr, out_ptr, n_per_prog,
                SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    base = pid * n_per_prog
    offs = tl.arange(0, BLOCK)                 # 16384 lanes
    mask = offs < n_per_prog
    x = tl.load(in_ptr + base + offs, mask=mask)      # gm_to_ub (32KB load)
    y = x + SCALAR                                     # vadd (32KB compute)
    tl.store(out_ptr + base + offs, y, mask=mask)      # ub_to_gm (32KB store)


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    raise SystemExit("No CUDA/NPU device found. Run on real hardware "
                     "(for Ascend, ensure torch_npu + Triton-Ascend installed).")


def sync(device: str) -> None:
    (torch.npu if device == 'npu' else torch.cuda).synchronize()


def bench(fn, device: str, warmup: int = 50, rep: int = 200) -> float:
    """Median-ish wall time per call in milliseconds."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / rep * 1e3


def main() -> None:
    device = pick_device()
    torch.manual_seed(0)
    x   = torch.randn(N_TOTAL, dtype=DTYPE, device=device)
    out = torch.empty_like(x)

    def run():
        vadd_kernel[(GRID,)](x, out, N_PER_PROG, SCALAR=SCALAR, BLOCK=BLOCK)

    # ── Correctness ────────────────────────────────────────────────────────────
    run()
    sync(device)
    ref = x + SCALAR
    ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
    max_err = (out.float() - ref.float()).abs().max().item()
    print(f"device: {device}")
    print(f"correctness: {'PASS' if ok else 'FAIL'} (max_err={max_err:.2e})")

    # ── Timing ─────────────────────────────────────────────────────────────────
    ms = bench(run, device)
    print(f"\nconfig: grid={GRID}, tile=32KB (BLOCK={BLOCK}), "
          f"total={N_TOTAL} elem ({N_TOTAL*BYTES_PER_EL//1024}KB), fp16")
    print(f"  vadd grid=2/32KB: {ms*1e3:9.3f} us")
    print(f"  cost model 单核预测 (含store): ~0.97 us (967 ns)")


if __name__ == "__main__":
    main()
