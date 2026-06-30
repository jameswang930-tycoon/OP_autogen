#!/usr/bin/env python3
"""
Verify the simulator's predicted ~11.2x speedup on REAL hardware via Triton.

Source DSL programs (each describes the work of ONE thread over a 10 KB fp16 buffer):

  before:
    alloc(gm_1, 10KB) alloc(ub_1, 1KB) alloc(ub_2, 1KB)
    for m in range(0, 10, 1) { gm_to_ub(ub_1, gm_1 + m * 1KB)  vadd(ub_2, ub_1, 3) }
        -> 10 separate 1 KB tile loads + 10 small vadds  (small-transfer pattern)

  after:
    alloc(gm_1, 10KB) alloc(ub_1, 10KB) alloc(ub_2, 10KB)
    gm_to_ub(ub_1, gm_1)  vadd(ub_2, ub_1, 3)
        -> one 10 KB load + one 10 KB vadd          (single-large-transfer pattern)

Both compute  out = in + 3  over 10 KB of fp16 data per thread.

Translation notes
-----------------
* The DSL program is "1 thread". We run THREADS (=40) parallel program instances,
  each owning its own 10 KB chunk:  grid = (THREADS,).  total data = THREADS * 10 KB.
  (40 program instances maps to 40 AI cores on an NPU; on a GPU it underfills the
  device, so the meaningful target here is the NPU.)
* dtype is fp16 (the VecUnit/vadd model is calibrated for fp16; 2 bytes/elem):
    1 KB tile  = 512 elems     10 KB chunk = 5120 elems per thread.
* The DSL keeps the vadd result in UB and never stores back. A real kernel whose
  output is never written would be optimized away, so we store the result to a
  global output tensor. We store at the SAME granularity as the load (per-tile in
  `before`, single in `after`) so the small-vs-large transfer pattern — the variable
  the predicted speedup depends on — applies to the store too, and both kernels
  produce identical, checkable output.

Hardware / backend
------------------
Runs on whatever Triton backend torch exposes. CUDA is auto-detected; on an Ascend
NPU make sure `torch_npu` is importable (this script imports it if present) and the
Triton-Ascend backend is installed. Timing uses a backend-agnostic warmup+sync loop
so it does not depend on triton.testing.do_bench being ported.

Run:  python3 verify_triton.py                 # 40 threads (default)
      python3 verify_triton.py --threads 40 --warps 4
"""

import argparse
import time

import torch
import triton
import triton.language as tl

# Pull in the Ascend NPU torch backend if it is installed (registers torch.npu).
try:                       # noqa: SIM105
    import torch_npu       # noqa: F401
except ImportError:
    pass

# ── Geometry derived from the DSL ───────────────────────────────────────────────
DTYPE        = torch.float16
BYTES_PER_EL = 2
TILE_BYTES   = 1024                          # 1 KB tile in `before`
CHUNK_BYTES  = 10 * 1024                     # 10 KB per thread (gm_1)
TILE_ELEMS   = TILE_BYTES  // BYTES_PER_EL   # 512
CHUNK_ELEMS  = CHUNK_BYTES // BYTES_PER_EL   # 5120
NUM_TILES    = CHUNK_BYTES // TILE_BYTES     # 10
SCALAR       = 3.0
# Next power-of-two >= CHUNK_ELEMS, for the single masked load in `after`.
CHUNK_BLOCK  = triton.next_power_of_2(CHUNK_ELEMS)   # 8192


# ── before: ten 1 KB tile load+add+store per thread ─────────────────────────────
@triton.jit
def before_kernel(in_ptr, out_ptr, n_per_prog,
                  TILE: tl.constexpr, NUM_TILES: tl.constexpr, SCALAR: tl.constexpr):
    pid  = tl.program_id(0)
    base = pid * n_per_prog
    for m in range(NUM_TILES):                       # 10 small transfers
        offs = base + m * TILE + tl.arange(0, TILE)  # one 1 KB tile (512 elems)
        x = tl.load(in_ptr + offs)                   # gm_to_ub(ub_1, gm_1 + m*1KB)
        y = x + SCALAR                               # vadd(ub_2, ub_1, 3)
        tl.store(out_ptr + offs, y)


# ── after: one 10 KB load+add+store per thread ──────────────────────────────────
@triton.jit
def after_kernel(in_ptr, out_ptr, n_per_prog,
                 BLOCK: tl.constexpr, SCALAR: tl.constexpr):
    pid  = tl.program_id(0)
    base = pid * n_per_prog
    offs = tl.arange(0, BLOCK)                        # 8192 lanes, masked to 5120
    mask = offs < n_per_prog
    x = tl.load(in_ptr + base + offs, mask=mask)      # gm_to_ub(ub_1, gm_1)  (10 KB)
    y = x + SCALAR                                     # vadd(ub_2, ub_1, 3)   (10 KB)
    tl.store(out_ptr + base + offs, y, mask=mask)


def pick_device() -> str:
    """Return the available Triton-capable accelerator: 'cuda' or 'npu'."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    raise SystemExit("No CUDA/NPU device found. Run this on the real hardware "
                     "(for Ascend, ensure torch_npu + Triton-Ascend are installed).")


def sync(device: str) -> None:
    (torch.npu if device == 'npu' else torch.cuda).synchronize()


def bench(fn, device: str, warmup: int = 50, rep: int = 200) -> float:
    """Median-ish wall time per call in milliseconds (warmup + synchronized reps)."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / rep * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=40,
                    help='number of parallel program instances (default 40)')
    ap.add_argument('--warps', type=int, default=4,
                    help='num_warps per program; same for both kernels for fairness')
    args = ap.parse_args()

    device  = pick_device()
    THREADS = args.threads
    n_total = THREADS * CHUNK_ELEMS
    grid    = (THREADS,)

    torch.manual_seed(0)
    x   = torch.randn(n_total, dtype=DTYPE, device=device)
    o_b = torch.empty_like(x)
    o_a = torch.empty_like(x)

    def run_before():
        before_kernel[grid](x, o_b, CHUNK_ELEMS,
                            TILE=TILE_ELEMS, NUM_TILES=NUM_TILES, SCALAR=SCALAR,
                            num_warps=args.warps)

    def run_after():
        after_kernel[grid](x, o_a, CHUNK_ELEMS,
                           BLOCK=CHUNK_BLOCK, SCALAR=SCALAR, num_warps=args.warps)

    # ── Correctness: both must equal x + 3 ──────────────────────────────────────
    # fp16 has ~3 significant digits; the kernel's add may round through fp32 while
    # the torch reference rounds in fp16, so compare with an fp16-appropriate
    # tolerance. A real offset/mask bug would read different random elements and
    # diverge by O(1), far above this tolerance, so it is still caught.
    run_before()
    run_after()
    sync(device)
    ref  = x + SCALAR
    ok_b = torch.allclose(o_b, ref, rtol=1e-2, atol=1e-2)
    ok_a = torch.allclose(o_a, ref, rtol=1e-2, atol=1e-2)
    max_err_b = (o_b.float() - ref.float()).abs().max().item()
    max_err_a = (o_a.float() - ref.float()).abs().max().item()
    print(f"device: {device}")
    print(f"correctness: before={'PASS' if ok_b else 'FAIL'} (max_err={max_err_b:.2e})  "
          f"after={'PASS' if ok_a else 'FAIL'} (max_err={max_err_a:.2e})")

    # ── Timing ──────────────────────────────────────────────────────────────────
    ms_before = bench(run_before, device)
    ms_after  = bench(run_after,  device)

    print(f"\nconfig: {THREADS} threads (programs), {CHUNK_BYTES//1024} KB/thread, "
          f"fp16, num_warps={args.warps}")
    print(f"  before (10x1KB tiles): {ms_before*1e3:9.3f} us")
    print(f"  after  (1x10KB)      : {ms_after*1e3:9.3f} us")
    speedup = ms_before / ms_after if ms_after else float('nan')
    print(f"  measured speedup     : {speedup:6.2f}x   (simulator predicted ~11.2x)")


if __name__ == '__main__':
    main()
