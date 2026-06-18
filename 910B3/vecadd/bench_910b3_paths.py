#!/usr/bin/env python3
"""
Ascend 910B3 - five datapath TILE sweep (fixed data=256MB)
Transfer paths grid=20 (AI Core); vec compute grid=40 (Vec Core).
Each path sweeps TILE to find the best transfer/compute granularity.

Outputs:
  - terminal tables (human readable)
  - bench_result.csv (structured, one row per (case,grid,TILE), for plotting)
"""

import csv
import time
import copy
import torch
import triton
import triton.language as tl

N_CORES    = 20      # AI Core (MTE2/MTE3 transfer)
N_VEC      = 40      # Vec Core (vector compute)
FREQ_GHZ   = 1.8
UB_KB      = 192
UB_BYTES   = UB_KB * 1024
L2_MB      = 192
DTYPE      = torch.float16
ELEM       = 2

WARMUP = 30
REPEAT = 200

# fixed data size = 256MB (> L2, DDR scenario)
DATA_MB = 256

# theoretical values (B/cyc/core)
T_DDR_R  = 32
T_DDR_W  = 32
T_L2_R   = 110
T_UB2L2  = 64
# Vec: 256B SIMD, 128 ops/cyc, 40 cores
T_VEC_TFLOPS = 128 * FREQ_GHZ * N_VEC / 1e3  # 9.216
T_VEC_BCYC   = 256  # B/cyc/vcore

# -- global CSV rows: collected across all sweeps, written at the end ----------
# columns: case, grid, tile, tile_kb, metric_kind, metric_value, secondary
#   metric_kind = "GB/s" for transfer, "TFLOPS" for vec
#   secondary   = B/cyc for transfer, FLOP/cyc/vcore for vec
CSV_ROWS = []

def add_row(case, grid, tile, metric_kind, metric_value, secondary):
    CSV_ROWS.append({
        "case": case,
        "grid": grid,
        "tile": tile,
        "tile_kb": tile * ELEM / 1024,
        "metric_kind": metric_kind,
        "metric_value": round(metric_value, 4),
        "secondary": round(secondary, 4) if secondary is not None else "",
    })


# -- kernel rename helper: give every TILE/grid config a unique name -----------
# msprof only records the kernel function name; same-named kernels (e.g.
# read_tile_kernel) called many times all look identical and can't be mapped
# back to a config. Here we clone the jit kernel and rename it -> msprof output
# carries the config tag directly.
_renamed_cache = {}

def named_kernel(base_kernel, tag):
    new_name = f"{base_kernel.__name__}_{tag}"
    if new_name in _renamed_cache:
        return _renamed_cache[new_name]
    k = copy.copy(base_kernel)
    try:
        k.fn = copy.copy(base_kernel.fn)
        k.fn.__name__ = new_name
    except AttributeError:
        pass
    for attr in ("__name__", "_fn_name", "fn_name"):
        if hasattr(k, attr):
            try:
                setattr(k, attr, new_name)
            except (AttributeError, TypeError):
                pass
    _renamed_cache[new_name] = k
    return k


def align_data(mb, grid, max_tile):
    """
    align data size to a multiple of grid x max_tile x 3
    the x3 lets factor-3 fine tiles (384/768/1536/3072...) divide XBLOCK evenly
    """
    unit = grid * max_tile * 3
    target = int(mb * 1024 * 1024 / ELEM)
    return max(unit, (target // unit) * unit)


# ===============================================================================
#  Kernels (TILE tunable)
# ===============================================================================

# pure read (fp16 accumulate)
@triton.jit
def read_kernel(x_ptr, out_ptr, XBLOCK: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * XBLOCK
    acc = tl.zeros([TILE], dtype=tl.float16)
    loops: tl.constexpr = XBLOCK // TILE
    for i in range(loops):
        offs = base + i * TILE + tl.arange(0, TILE)
        acc += tl.load(x_ptr + offs)
    tl.store(out_ptr + pid, tl.sum(acc.to(tl.float32), axis=0))


# -- L2 hit/miss TILE sweep kernels ----------------------------------------
# read_tile: standard read (used for miss), each core reads XBLOCK, inner TILE
@triton.jit
def read_tile_kernel(x_ptr, out_ptr, XBLOCK: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * XBLOCK
    acc = tl.zeros([TILE], dtype=tl.float16)
    loops: tl.constexpr = XBLOCK // TILE
    for i in range(loops):
        offs = base + i * TILE + tl.arange(0, TILE)
        acc += tl.load(x_ptr + offs)
    tl.store(out_ptr + pid, tl.sum(acc.to(tl.float32), axis=0))


# read_hit_tile: used for hit. each core reads a fixed block (BLK_LOOPS x TILE)
# repeated OUTER times -> block stays resident in L2. inner TILE tunable to
# measure transfer-granularity effect on L2->UB.
@triton.jit
def read_hit_tile_kernel(x_ptr, out_ptr,
                         TILE: tl.constexpr, BLK_LOOPS: tl.constexpr,
                         OUTER: tl.constexpr):
    pid = tl.program_id(0)
    blk_elems: tl.constexpr = BLK_LOOPS * TILE
    core_base = pid * blk_elems
    acc = tl.zeros([TILE], dtype=tl.float16)
    for _ in range(OUTER):
        for i in range(BLK_LOOPS):
            offs = core_base + i * TILE + tl.arange(0, TILE)
            acc += tl.load(x_ptr + offs)
    tl.store(out_ptr + pid, tl.sum(acc.to(tl.float32), axis=0))


# scalar-accumulate version: reduce each loaded tile to scalar immediately.
# saves the TILE-sized accumulator buffer; only the load buffer occupies UB,
# so the TILE ceiling roughly doubles.
@triton.jit
def read_hit_scalar_kernel(x_ptr, out_ptr,
                           TILE: tl.constexpr, BLK_LOOPS: tl.constexpr,
                           OUTER: tl.constexpr):
    pid = tl.program_id(0)
    blk_elems: tl.constexpr = BLK_LOOPS * TILE
    core_base = pid * blk_elems
    s = 0.0
    for _ in range(OUTER):
        for i in range(BLK_LOOPS):
            offs = core_base + i * TILE + tl.arange(0, TILE)
            x = tl.load(x_ptr + offs)              # only load buffer in UB
            s += tl.sum(x.to(tl.float32), axis=0)  # reduce to scalar at once
    tl.store(out_ptr + pid, s)


# pure write
@triton.jit
def write_kernel(out_ptr, XBLOCK: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * XBLOCK
    loops: tl.constexpr = XBLOCK // TILE
    for i in range(loops):
        offs = base + i * TILE + tl.arange(0, TILE)
        tl.store(out_ptr + offs, tl.zeros([TILE], dtype=tl.float16) + 1.0)


# copy (1R+1W)
@triton.jit
def copy_kernel(src_ptr, dst_ptr, XBLOCK: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * XBLOCK
    loops: tl.constexpr = XBLOCK // TILE
    for i in range(loops):
        offs = base + i * TILE + tl.arange(0, TILE)
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs))


# Vec compute (FMA: acc*k+k = 2 ops per iter)
@triton.jit
def compute_kernel(x_ptr, out_ptr, XBLOCK: tl.constexpr, TILE: tl.constexpr,
                   N_ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * XBLOCK
    loops: tl.constexpr = XBLOCK // TILE
    for i in range(loops):
        offs = base + i * TILE + tl.arange(0, TILE)
        acc = tl.load(x_ptr + offs)
        for _ in range(N_ITERS):
            acc = acc * 0.99 + 0.01
        tl.store(out_ptr + offs, acc)


# Vec compute (monadic add: acc+k = 1 op per iter)
@triton.jit
def compute_add_kernel(x_ptr, out_ptr, XBLOCK: tl.constexpr, TILE: tl.constexpr,
                       N_ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * XBLOCK
    loops: tl.constexpr = XBLOCK // TILE
    for i in range(loops):
        offs = base + i * TILE + tl.arange(0, TILE)
        acc = tl.load(x_ptr + offs)
        for _ in range(N_ITERS):
            acc = acc + 0.01
        tl.store(out_ptr + offs, acc)


# ===============================================================================
#  Generic sweep framework
# ===============================================================================

def bench(kernel_fn, make_args, grid, xblock, TILE, l2_warmup=False, tag=None, **kw):
    args = make_args()
    g = (grid,)
    if tag is not None:
        kernel_fn = named_kernel(kernel_fn, tag)
    try:
        nwarm = WARMUP * 2 if l2_warmup else WARMUP
        for _ in range(nwarm):
            kernel_fn[g](*args, XBLOCK=xblock, TILE=TILE, **kw)
        torch.npu.synchronize()
    except Exception as e:
        msg = str(e)
        if "ub overflow" in msg.lower() or "MLIR" in type(e).__name__:
            return None, "UB overflow"
        return None, msg[:40]

    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        kernel_fn[g](*args, XBLOCK=xblock, TILE=TILE, **kw)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / REPEAT * 1e3, None


SEP = "=" * 80

def sweep_path(title, case, kernel_fn, make_args, n_bufs, theory_bpc,
               bytes_per_elem_moved, grids=(N_CORES, N_VEC), l2_warmup=False,
               extra_ub_bytes=0, **kw):
    """
    Generic TILE x grid sweep.
      grids: grid list to compare (default 20 and 40)
      n_bufs: number of UB buffers the kernel uses (sets the TILE ceiling)
      bytes_per_elem_moved: bytes moved per element (for bandwidth)
      extra_ub_bytes: extra UB usage (e.g. fp32 accumulator 4B)
    """
    per_elem = n_bufs * ELEM + extra_ub_bytes
    tile_max = UB_BYTES // (per_elem * 2)
    tile_max = 1 << (tile_max.bit_length() - 1)

    # fine grain: powers of 2 + 1.5x midpoints (factor-3, divisible via align x3)
    tiles = [t for t in [256, 384, 512, 768, 1024, 1536, 2048, 3072,
                         4096, 6144, 8192, 12288, 16384, 24576, 32768,
                         49152, 65536]
             if t <= tile_max]

    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"  data={DATA_MB}MB  TILE_max={tile_max}  theory_per_core={theory_bpc} B/cyc")
    print(f"{SEP}")
    print(f"  {'grid':>5s}  {'TILE(elem)':>10s}  {'KB/xfer':>8s}  {'loops':>7s}  "
          f"{'ms':>10s}  {'agg_GB/s':>10s}  {'B/cyc/core':>11s}  {'util':>7s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*7}  "
          f"{'-'*10}  {'-'*10}  {'-'*11}  {'-'*7}")

    best = {"bw": 0, "tile": 0, "bpc": 0, "grid": 0}
    for grid in grids:
        N = align_data(DATA_MB, grid, max(tiles))
        for TILE in tiles:
            xblock = N // grid
            if xblock % TILE != 0:
                continue
            loops = xblock // TILE
            ms, err = bench(kernel_fn, lambda: make_args(N), grid, xblock, TILE,
                            l2_warmup=l2_warmup, tag=f"T{TILE}_g{grid}", **kw)
            if ms is None:
                print(f"  {grid:>5d}  {TILE:>10d}  SKIP: {err}")
                continue
            total_bytes = bytes_per_elem_moved * N
            bw = total_bytes / (ms / 1e3) / 1e9
            bpc = bw / N_CORES / FREQ_GHZ
            util = bpc / theory_bpc * 100
            if bw > best["bw"]:
                best = {"bw": bw, "tile": TILE, "bpc": bpc, "grid": grid}
            add_row(case, grid, TILE, "GB/s", bw, bpc)
            print(f"  {grid:>5d}  {TILE:>10d}  {TILE*ELEM/1024:>7.1f}K  {loops:>7d}  "
                  f"{ms:>10.4f}  {bw:>10.2f}  {bpc:>11.2f}  {util:>6.1f}%")
        print(f"  {'.'*78}")

    print(f"  -> best grid={best['grid']} TILE={best['tile']} "
          f"({best['tile']*ELEM//1024}KB)  "
          f"BW={best['bw']:.2f} GB/s  {best['bpc']:.2f} B/cyc/core")
    return best


def sweep_read_l2(grid=N_VEC):
    """
    [1] GM->UB pure read, L2 hit / miss, each sweeps TILE internally.
      hit:  total data < L2, repeated reads stay resident -> L2->UB bandwidth
      miss: total data 256/512/1024MB >> L2, each pass reads new region -> HBM->UB
    grid fixed at 40 (known best); primary metric aggregate GB/s.
    """
    AGG_DDR = N_CORES * T_DDR_R * FREQ_GHZ   # 1152 GB/s aggregate theory
    AGG_L2  = N_VEC   * T_L2_R * FREQ_GHZ    # 7920 GB/s (L2 not bottleneck, 40 conc.)

    tiles = [256, 384, 512, 768, 1024, 1536, 2048, 3072,
             4096, 6144, 8192, 12288, 16384]

    # -- L2 HIT: working set < L2, repeated read -> resident ------------
    # compare two kernels:
    #   vector-acc (acc+=): acc takes 1xTILE buffer -> TILE ceiling ~24576
    #   scalar-acc (s+=sum): no TILE accumulator -> TILE ceiling doubles
    HIT_TILES = [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152]
    BLK_ELEMS = 49152 * 3 * 10   # =1474560/core, working set ~112MB, divisible
    workset_mb = grid * BLK_ELEMS * ELEM / 1024 / 1024

    N_hit = grid * BLK_ELEMS
    x_hit = torch.randn(N_hit, device="npu", dtype=DTYPE)
    out_hit = torch.zeros(grid, device="npu", dtype=torch.float32)
    g = (grid,)
    OUTER_HIT = 256

    for kernel_fn, kname, case in [
            (read_hit_tile_kernel,   "vector-acc (acc+=)",  "1-HIT_vector"),
            (read_hit_scalar_kernel, "scalar-acc (s+=sum)", "1-HIT_scalar")]:
        print(f"\n{SEP}")
        print(f"  [1-HIT/{kname}] L2->UB read BW - TILE sweep (grid={grid})")
        print(f"          working_set={workset_mb:.0f}MB < L2 {L2_MB}MB, resident")
        print(f"          aggregate theory l2_read = {AGG_L2:.0f} GB/s")
        print(f"{SEP}")
        print(f"  {'TILE(elem)':>10s}  {'KB/xfer':>8s}  {'loops':>7s}  {'repeats':>8s}  "
              f"{'ms':>10s}  {'agg_GB/s':>10s}  {'%L2theory':>9s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}  "
              f"{'-'*10}  {'-'*10}  {'-'*9}")

        best_hit = {"bw": 0, "tile": 0}
        for TILE in HIT_TILES:
            if BLK_ELEMS % TILE != 0:
                continue
            blk_loops = BLK_ELEMS // TILE
            k = named_kernel(kernel_fn, f"HIT_T{TILE}_g{grid}")
            try:
                for _ in range(WARMUP):
                    k[g](x_hit, out_hit, TILE=TILE,
                         BLK_LOOPS=blk_loops, OUTER=OUTER_HIT)
                torch.npu.synchronize()
            except Exception:
                print(f"  {TILE:>10d}  FAIL: UB overflow")
                continue
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEAT // 4):
                k[g](x_hit, out_hit, TILE=TILE,
                     BLK_LOOPS=blk_loops, OUTER=OUTER_HIT)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            ms = (t1 - t0) / (REPEAT // 4) * 1e3
            total_bytes = grid * BLK_ELEMS * OUTER_HIT * ELEM
            bw = total_bytes / (ms / 1e3) / 1e9
            bpc = bw / N_CORES / FREQ_GHZ
            pct = bw / AGG_L2 * 100
            if bw > best_hit["bw"]:
                best_hit = {"bw": bw, "tile": TILE}
            add_row(case, grid, TILE, "GB/s", bw, bpc)
            print(f"  {TILE:>10d}  {TILE*ELEM/1024:>7.1f}K  {blk_loops:>7d}  "
                  f"{OUTER_HIT:>8d}  {ms:>10.4f}  {bw:>10.2f}  {pct:>8.1f}%")
        print(f"  -> best TILE={best_hit['tile']} ({best_hit['tile']*ELEM//1024}KB)  "
              f"{best_hit['bw']:.0f} GB/s")
    del x_hit, out_hit

    # -- L2 MISS: total data >> L2, each pass reads new region -> miss --
    for DATA_MISS_MB in [256, 512, 1024]:
        case = f"1-MISS_{DATA_MISS_MB}MB"
        print(f"\n{SEP}")
        print(f"  [1-MISS] GM->UB(HBM) read BW - data={DATA_MISS_MB}MB "
              f"({DATA_MISS_MB/L2_MB:.1f}xL2)  TILE sweep (grid={grid})")
        print(f"           data >> L2 -> hit rate ~0, aggregate theory ddr_read={AGG_DDR:.0f} GB/s")
        print(f"{SEP}")
        print(f"  {'TILE(elem)':>10s}  {'KB/xfer':>8s}  {'core_elems':>10s}  {'loops':>7s}  "
              f"{'ms':>10s}  {'agg_GB/s':>10s}  {'%DDRtheory':>10s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*7}  "
              f"{'-'*10}  {'-'*10}  {'-'*10}")

        unit = grid * 16384 * 3
        N = max(unit, int(DATA_MISS_MB * 1024 * 1024 / ELEM) // unit * unit)
        x = torch.randn(N, device="npu", dtype=DTYPE)
        out = torch.zeros(grid, device="npu", dtype=torch.float32)

        best_miss = {"bw": 0, "tile": 0}
        for TILE in tiles:
            xblock = N // grid
            if xblock % TILE != 0:
                continue
            loops = xblock // TILE
            k = named_kernel(read_tile_kernel, f"MISS{DATA_MISS_MB}_T{TILE}_g{grid}")
            try:
                for _ in range(WARMUP):
                    k[g](x, out, XBLOCK=xblock, TILE=TILE)
                torch.npu.synchronize()
            except Exception as e:
                print(f"  {TILE:>10d}  FAIL: {str(e)[:40]}")
                continue
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                k[g](x, out, XBLOCK=xblock, TILE=TILE)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            ms = (t1 - t0) / REPEAT * 1e3
            bw = N * ELEM / (ms / 1e3) / 1e9
            bpc = bw / N_CORES / FREQ_GHZ
            pct = bw / AGG_DDR * 100
            if bw > best_miss["bw"]:
                best_miss = {"bw": bw, "tile": TILE}
            add_row(case, grid, TILE, "GB/s", bw, bpc)
            print(f"  {TILE:>10d}  {TILE*ELEM/1024:>7.1f}K  {xblock:>10d}  "
                  f"{loops:>7d}  {ms:>10.4f}  {bw:>10.2f}  {pct:>9.1f}%")
        print(f"  -> best TILE={best_miss['tile']} "
              f"({best_miss['tile']*ELEM//1024}KB)  {best_miss['bw']:.0f} GB/s")
        del x, out


# ===============================================================================
#  Main
# ===============================================================================

def main():
    dev = torch.npu.get_device_name(0)
    print(f"\n  Device: {dev}")
    print(f"  fixed data={DATA_MB}MB (>{L2_MB}MB L2 -> DDR scenario), sweep TILE")
    print(f"  transfer grid={N_CORES} (AI Core), vec compute grid={N_VEC} (Vec Core)")
    print(f"  theory: ddr_read={T_DDR_R} l2_read={T_L2_R} ub_to_l2={T_UB2L2} "
          f"vec={T_VEC_TFLOPS:.2f}TFLOPS/{T_VEC_BCYC}B/cyc")

    # -- [1] GM->UB read: L2 hit vs L2 miss ---------------------------
    sweep_read_l2()

    # -- [2] UB->L2->GM pure write (1 buf) ----------------------------
    sweep_path(
        "[2] UB->L2->GM pure write (MTE3)", "2-write",
        write_kernel,
        lambda N: (torch.empty(N, device="npu", dtype=DTYPE),),
        n_bufs=1, theory_bpc=T_UB2L2, bytes_per_elem_moved=ELEM,
    )

    # -- [3] GM<->UB copy (2 buf) -------------------------------------
    sweep_path(
        "[3] GM<->UB copy (1R+1W)", "3-copy",
        copy_kernel,
        lambda N: (torch.randn(N, device="npu", dtype=DTYPE),
                   torch.empty(N, device="npu", dtype=DTYPE)),
        n_bufs=2, theory_bpc=T_UB2L2, bytes_per_elem_moved=2 * ELEM,
    )

    # -- [4] Vector compute: FMA (2 ops) vs monadic add (1 op) --------
    NITERS = 256
    tile_max_vec = UB_BYTES // (2 * ELEM * 2)
    tile_max_vec = 1 << (tile_max_vec.bit_length() - 1)
    tiles_vec = [t for t in [256, 384, 512, 768, 1024, 1536, 2048, 3072,
                             4096, 6144, 8192, 12288, 16384, 24576, 32768]
                 if t <= tile_max_vec]

    vec_variants = [
        (compute_kernel,     "FMA (acc*k+k, 2 ops)", 2, "FMA", "4-Vec_FMA"),
        (compute_add_kernel, "monadic add (acc+k, 1 op)", 1, "ADD", "4-Vec_ADD"),
    ]

    for vkernel, vlabel, ops_per_iter, vtag, case in vec_variants:
        print(f"\n{SEP}")
        print(f"  [4] Vector {vlabel} (N_ITERS={NITERS}) - sweep grid x TILE")
        print(f"  theory: {T_VEC_TFLOPS:.3f} TFLOPS = {T_VEC_BCYC} B/cyc/vcore")
        print(f"{SEP}")
        print(f"  {'grid':>5s}  {'TILE(elem)':>10s}  {'KB/xfer':>8s}  {'loops':>7s}  "
              f"{'ms':>10s}  {'TFLOPS':>9s}  {'FLOP/cyc/vc':>11s}  {'util':>7s}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*7}  "
              f"{'-'*10}  {'-'*9}  {'-'*11}  {'-'*7}")

        best_v = {"tf": 0, "tile": 0, "grid": 0}
        for grid in (N_CORES, N_VEC):
            N_vec = align_data(DATA_MB, grid, max(tiles_vec))
            for TILE in tiles_vec:
                xblock = N_vec // grid
                if xblock % TILE != 0:
                    continue
                loops = xblock // TILE
                x = torch.randn(N_vec, device="npu", dtype=DTYPE)
                o = torch.empty(N_vec, device="npu", dtype=DTYPE)
                ms, err = bench(vkernel, lambda: (x, o), grid, xblock, TILE,
                                tag=f"{vtag}_T{TILE}_g{grid}", N_ITERS=NITERS)
                if ms is None:
                    print(f"  {grid:>5d}  {TILE:>10d}  SKIP: {err}")
                    del x, o
                    continue
                tflops = N_vec * NITERS * ops_per_iter / (ms / 1e3) / 1e12
                fpc = tflops * 1e12 / N_VEC / (FREQ_GHZ * 1e9)
                util = tflops / T_VEC_TFLOPS * 100
                if tflops > best_v["tf"]:
                    best_v = {"tf": tflops, "tile": TILE, "grid": grid}
                add_row(case, grid, TILE, "TFLOPS", tflops, fpc)
                print(f"  {grid:>5d}  {TILE:>10d}  {TILE*ELEM/1024:>7.1f}K  {loops:>7d}  "
                      f"{ms:>10.4f}  {tflops:>9.2f}  {fpc:>11.2f}  {util:>6.1f}%")
                del x, o
            print(f"  {'.'*78}")
        print(f"  -> best grid={best_v['grid']} TILE={best_v['tile']} "
              f"({best_v['tile']*ELEM//1024}KB)  {best_v['tf']:.2f} TFLOPS")

    # -- write structured CSV for plotting ----------------------------
    out_csv = "bench_result.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "grid", "tile", "tile_kb",
                                          "metric_kind", "metric_value", "secondary"])
        w.writeheader()
        w.writerows(CSV_ROWS)
    print(f"\n{SEP}")
    print(f"  done. structured CSV -> {out_csv} ({len(CSV_ROWS)} rows)")
    print(f"  plot with: python plot_bench.py {out_csv}")
    print(f"{SEP}")


if __name__ == "__main__":
    main()
