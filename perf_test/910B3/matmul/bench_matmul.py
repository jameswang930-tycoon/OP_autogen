#!/usr/bin/env python3
"""
Ascend 910B3 - Cube Core matmul 算力测试 (TILE sweep)
照 vecadd/bench_910b3_paths.py 结构, 测 Cube 单元 (tl.dot) 的 TFLOPS。

Cube 数据路径: GM -> L1 -> L0A/L0B -> Cube(MMA 16x16x16) -> L0C -> UB -> GM
理论算力: 每 cube 16^3 FMA/cyc = 8192 FLOP/cyc; 20 cube x 1.8GHz ≈ 294.9 TFLOPS(fp16)

L0 容量约束 (选 tile 时已满足):
  L0A=64KB: BM*BK*2 <= 65536
  L0B=64KB: BK*BN*2 <= 65536
  L0C=128KB: BM*BN*4 <= 131072  (fp32 accumulator)

输出: 终端表 + bench_result.csv (兼容 plot_bench.py)
用法:
  python bench_matmul.py
  python plot_bench.py bench_result.csv
"""

import csv
import time
import torch
import triton
import triton.language as tl

N_CUBE   = 20
FREQ_GHZ = 1.8
FLOP_PER_CYC_PER_CUBE = 16 * 16 * 16 * 2          # 8192 (16^3 FMA = 2 ops)
T_CUBE_TFLOPS = N_CUBE * FLOP_PER_CYC_PER_CUBE * FREQ_GHZ / 1e3   # ~294.9

DTYPE = torch.float16
ELEM  = 2

WARMUP = 30
REPEAT = 50          # matmul 重, 少 repeat

# 矩阵尺寸 M=N=K (2 的幂, 被 tile 整除)
SIZES = [1024, 2048, 4096]

# tile 组合 (BM, BN, BK); 已满足 L0A/L0B/L0C 约束, 且均为 16 倍数 (cube m=n=k=16)
TILES = [(128, 128, 32), (128, 128, 64),
         (64, 128, 32), (128, 64, 32),
         (256, 128, 32), (128, 256, 32),
         (64, 64, 32)]

CSV_ROWS = []
def add_row(case, size, bm, bn, bk, tflops, fpc):
    # CSV 兼容 vecadd/plot_bench.py: grid=N_CUBE, tile=bm, tile_kb=tile 规模
    CSV_ROWS.append({
        "case": case,
        "grid": N_CUBE,
        "tile": bm,
        "tile_kb": bm * bn * bk * ELEM / 1024,
        "metric_kind": "TFLOPS",
        "metric_value": round(tflops, 4),
        "secondary": round(fpc, 4),
    })


# ===============================================================================
#  Matmul kernel: C[M,N] = A[M,K] @ B[K,N]  (fp16 input, fp32 accumulate -> cube)
# ===============================================================================
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  NUM_N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N          # NUM_N = N // BN, host 算好 (避免 kernel 内运行时除法)
    pid_n = pid % NUM_N
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak)
        b = tl.load(b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
        acc = tl.dot(a, b, acc)                       # -> cube MMA (fp16 x fp16, fp32 acc)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc.to(tl.float16))


SEP = "=" * 80


def _launch(a, b, c, M, N, K, BM, BN, BK):
    grid = (M // BM * N // BN,)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        NUM_N=N // BN, BM=BM, BN=BN, BK=BK)


def run_one(M, N, K, BM, BN, BK):
    """返回 (ms, tflops) 或 (None, err)."""
    a = torch.randn(M, K, device="npu", dtype=DTYPE)
    b = torch.randn(K, N, device="npu", dtype=DTYPE)
    c = torch.empty(M, N, device="npu", dtype=DTYPE)
    try:
        for _ in range(WARMUP):
            _launch(a, b, c, M, N, K, BM, BN, BK)
        torch.npu.synchronize()
    except Exception as e:
        msg = str(e)
        if "ub overflow" in msg.lower() or "MLIR" in type(e).__name__:
            return None, "UB overflow"
        return None, msg[:50]
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        _launch(a, b, c, M, N, K, BM, BN, BK)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / REPEAT * 1e3
    tflops = 2 * M * N * K / (ms / 1e3) / 1e12
    return ms, tflops


def check_correctness():
    """小矩阵对比 torch.matmul, 确保 tl.dot 配置正确触发 cube。"""
    M = N = K = 256
    BM = BN = 128
    BK = 32
    torch.manual_seed(0)
    a = torch.randn(M, K, device="npu", dtype=DTYPE)
    b = torch.randn(K, N, device="npu", dtype=DTYPE)
    c = torch.empty(M, N, device="npu", dtype=DTYPE)
    _launch(a, b, c, M, N, K, BM, BN, BK)
    torch.npu.synchronize()
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel = err / (ref_max + 1e-6)
    print(f"  correctness (256^3, BM=BN=128 BK=32): max_err={err:.2f} rel_err={rel:.3f}")
    if rel > 0.3:    # fp16 精度 rel~0.05; 若 tl.dot 配置错/没走 cube, rel 会 >>0.3
        print("  [WARN] matmul 相对误差过大, tl.dot 可能未正确配置或未走 cube!")
        return False
    return True


def sweep():
    for S in SIZES:
        M = N = K = S
        case = f"matmul_{S}x{S}x{S}"
        print(f"\n{SEP}")
        print(f"  [matmul] M=N=K={S}  (grid = M/BM x N/BN, 调度到 {N_CUBE} cube)")
        print(f"  理论 {T_CUBE_TFLOPS:.1f} TFLOPS (fp16, {N_CUBE} cube "
              f"x {FLOP_PER_CYC_PER_CUBE} FLOP/cyc x {FREQ_GHZ}GHz)")
        print(f"{SEP}")
        print(f"  {'BM':>5s} {'BN':>5s} {'BK':>5s} {'grid':>7s} {'ms':>10s} "
              f"{'TFLOPS':>9s} {'FLOP/cyc/cube':>13s} {'util':>7s}")
        print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*10} {'-'*9} {'-'*13} {'-'*7}")

        best = {"tf": 0.0, "tile": None}
        for BM, BN, BK in TILES:
            if M % BM or N % BN or K % BK:
                continue
            grid_n = M // BM * N // BN
            ms, tf_or_err = run_one(M, N, K, BM, BN, BK)
            if ms is None:
                print(f"  {BM:>5d} {BN:>5d} {BK:>5d} {grid_n:>7d}  SKIP: {tf_or_err}")
                continue
            tflops = tf_or_err
            fpc = tflops * 1e12 / N_CUBE / (FREQ_GHZ * 1e9)
            util = tflops / T_CUBE_TFLOPS * 100
            if tflops > best["tf"]:
                best = {"tf": tflops, "tile": (BM, BN, BK)}
            add_row(case, S, BM, BN, BK, tflops, fpc)
            print(f"  {BM:>5d} {BN:>5d} {BK:>5d} {grid_n:>7d} {ms:>10.4f} "
                  f"{tflops:>9.2f} {fpc:>13.2f} {util:>6.1f}%")
        if best["tile"] is not None:
            print(f"  -> best tile={best['tile']}  {best['tf']:.2f} TFLOPS "
                  f"({best['tf']/T_CUBE_TFLOPS*100:.1f}% theory)")


def main():
    dev = torch.npu.get_device_name(0)
    print(f"\n  Device: {dev}")
    print(f"  Cube: {N_CUBE} cores x {FREQ_GHZ}GHz, 理论 {T_CUBE_TFLOPS:.1f} TFLOPS (fp16)")
    print(f"  sizes: {SIZES}, tiles: {TILES}")

    if not check_correctness():
        print("  正确性检查失败, 终止 (检查 tl.dot / cube 配置)")
        return

    sweep()

    out_csv = "bench_result.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "grid", "tile", "tile_kb",
                                          "metric_kind", "metric_value", "secondary"])
        w.writeheader()
        w.writerows(CSV_ROWS)
    print(f"\n{SEP}")
    print(f"  done. CSV -> {out_csv} ({len(CSV_ROWS)} rows)")
    print(f"  plot: python plot_bench.py {out_csv}")
    print(f"{SEP}")


if __name__ == "__main__":
    main()
