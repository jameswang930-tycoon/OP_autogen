#!/usr/bin/env python3
"""910B3 硬件基准 kernels — 纯 triton, 自包含可跑. 每类 bench 多个变体 (尺寸×分块×精度扫描).

科学原则:
  1. 多变体扫描 (尺寸×分块×精度) — 单次测量是真值的下界, 取最大
  2. **一次 msprof 内循环跑 warmup+measure 次** — 跳过热身前奏, 平均稳态:
     - 默认 BENCH_WARMUP_ITERS=10 热身 + BENCH_MEASURE_ITERS=30 测量
       (msprof 测设备侧时间确定性高, 10 过 JIT/冷cache + 30 平均够稳, 无需 100)
     - run_bench.py 读 op_summary 跳过前 warmup 次, 平均后 measure 次 (与主循环同源 msprof)
  3. `--single` 单次 launch — 供 msprof op per-path (每路径带宽) 用

测什么 (6 类, 覆盖全链路带宽/算力):
  gm_read     GM 读带宽 (多尺寸×分块扫描)
  gm_write    GM 写带宽
  gm_copy     GM 拷贝 (读A写B)
  l2_read     L2 读带宽 (L2 内数组反复读)
  cube        cube 算力 (fp16/fp32 × 尺寸×分块扫描)
  vec         Vec 吞吐 (add/mul/fma)

用法 (910B3 服务器):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_kernels.py --list                          # 列出所有
  python3 bench_kernels.py --bench cube --variant 0        # 单变体 (10+30 循环, 打印 avg)
  python3 bench_kernels.py --bench cube --variant 0 --single  # 单次 (msprof op 用)
  python3 run_bench.py                                     # ★全套实测 (推荐)
  循环次数: BENCH_WARMUP_ITERS=10 BENCH_MEASURE_ITERS=30  (env 覆盖)
"""
import os
import sys
import time
import torch
try:
    import torch_npu          # 仅服务器有; --list/--help 不需要
except ImportError:
    pass
import triton
import triton.language as tl

# 配置注册表 + 静态字节/算力 (无 triton 依赖, 见 bench_config.py)
from bench_config import BENCHES, variant_bytes_flops  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════
#  kernels
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def read_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """纯读大数组 (归约防 load 被优化掉)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid, tl.sum(x))


@triton.jit
def write_kernel(x_ptr, N, BLOCK: tl.constexpr):
    """纯写大数组."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 1.0, mask=offs < N)


@triton.jit
def copy_kernel(a_ptr, b_ptr, N, BLOCK: tl.constexpr):
    """读A写B (拷贝)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(b_ptr + offs, tl.load(a_ptr + offs, mask=mask, other=0.0), mask=mask)


@triton.jit
def l2_read_kernel(x_ptr, out_ptr, N, ITERS, BLOCK: tl.constexpr):
    """L2 内小数组反复读 (测 L2 带宽, 数据不落 GM)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(ITERS):
        acc += tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid, tl.sum(acc))


@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """matmul (fp16/fp32, 视输入 dtype), fp32 累加."""
    pid = tl.program_id(0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def vec_kernel(a_ptr, b_ptr, c_ptr, N, OP: tl.constexpr, BLOCK: tl.constexpr):
    """向量运算: OP=0 add, 1 mul, 2 fma (a*b+a). 读2写1."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    if OP == 0:
        r = a + b
    elif OP == 1:
        r = a * b
    else:
        r = a * b + a
    tl.store(c_ptr + offs, r, mask=mask)


# ═══════════════════════════════════════════════════════════════════════
#  分配 + launch 分离 (循环复用同一批张量, 避免分配开销污染计时)
# ═══════════════════════════════════════════════════════════════════════

def _alloc(btype: str, v: dict) -> dict:
    npu = torch.device("npu")
    if btype == "read":
        return {"x": torch.empty(v["N"], dtype=torch.float32, device=npu),
                "out": torch.empty(v["N"] // v["BLOCK"], dtype=torch.float32, device=npu)}
    if btype == "write":
        return {"x": torch.empty(v["N"], dtype=torch.float32, device=npu)}
    if btype == "copy":
        return {"a": torch.empty(v["N"], dtype=torch.float32, device=npu),
                "b": torch.empty(v["N"], dtype=torch.float32, device=npu)}
    if btype == "l2":
        return {"x": torch.empty(v["N"], dtype=torch.float32, device=npu),
                "out": torch.empty(v["N"] // v["BLOCK"], dtype=torch.float32, device=npu)}
    if btype == "mm":
        dt = torch.float16 if v["dtype"] == "float16" else torch.float32
        return {"a": torch.rand(v["M"], v["K"], dtype=dt, device=npu),
                "b": torch.rand(v["K"], v["N"], dtype=dt, device=npu),
                "c": torch.empty(v["M"], v["N"], dtype=torch.float32, device=npu)}
    if btype == "vec":
        return {"a": torch.empty(v["N"], dtype=torch.float32, device=npu),
                "b": torch.empty(v["N"], dtype=torch.float32, device=npu),
                "c": torch.empty(v["N"], dtype=torch.float32, device=npu)}
    raise ValueError(f"未知 bench 类型: {btype}")


def _launch(btype: str, v: dict, t: dict):
    if btype == "read":
        read_kernel[(v["N"] // v["BLOCK"],)](t["x"], t["out"], v["N"], BLOCK=v["BLOCK"])
    elif btype == "write":
        write_kernel[(v["N"] // v["BLOCK"],)](t["x"], v["N"], BLOCK=v["BLOCK"])
    elif btype == "copy":
        copy_kernel[(v["N"] // v["BLOCK"],)](t["a"], t["b"], v["N"], BLOCK=v["BLOCK"])
    elif btype == "l2":
        l2_read_kernel[(v["N"] // v["BLOCK"],)](t["x"], t["out"], v["N"], v["ITERS"], BLOCK=v["BLOCK"])
    elif btype == "mm":
        grid = (triton.cdiv(v["M"], v["BM"]) * triton.cdiv(v["N"], v["BN"]),)
        mm_kernel[grid](t["a"], t["b"], t["c"], v["M"], v["N"], v["K"],
                        BLOCK_M=v["BM"], BLOCK_N=v["BN"], BLOCK_K=v["BK"])
    elif btype == "vec":
        vec_kernel[(v["N"] // v["BLOCK"],)](t["a"], t["b"], t["c"], v["N"], OP=v["OP"], BLOCK=v["BLOCK"])
    else:
        raise ValueError(f"未知 bench 类型: {btype}")


def run_variant(btype: str, v: dict, sync: bool = True) -> tuple:
    """分配一次 + launch (可选 sync). 返回 (bytes, flops)."""
    t = _alloc(btype, v)
    _launch(btype, v, t)
    if sync:
        torch.npu.synchronize()
    return variant_bytes_flops(btype, v)


def main():
    import argparse
    p = argparse.ArgumentParser(description="910B3 bench kernels")
    p.add_argument("--bench", type=str, help=f"bench 名: {list(BENCHES)}")
    p.add_argument("--variant", type=int, default=0, help="变体索引")
    p.add_argument("--single", action="store_true", help="单次 launch (msprof op per-path 用)")
    p.add_argument("--list", action="store_true", help="列出所有 bench + 变体")
    args = p.parse_args()
    if args.list:
        for n, b in BENCHES.items():
            print(f"  {n:10s} {b['desc']}  ({len(b['variants'])} 变体)")
        return
    if args.bench not in BENCHES:
        print(f"❌ 未知 bench: {args.bench}. 可选: {list(BENCHES)}")
        sys.exit(1)
    b = BENCHES[args.bench]
    if not (0 <= args.variant < len(b["variants"])):
        print(f"❌ variant 越界: 0~{len(b['variants'])-1}")
        sys.exit(1)
    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    v = b["variants"][args.variant]

    if args.single:
        bytes_total, flops = run_variant(b["type"], v, sync=True)
        print(f"[ok] {args.bench} v{args.variant}: single launch bytes={bytes_total}B flops={flops}")
        return

    # 循环模式: warmup + measure 次, 分配一次复用 (供 run_bench 的 msprof 一次采集)
    warmup = int(os.environ.get("BENCH_WARMUP_ITERS", "10"))
    measure = int(os.environ.get("BENCH_MEASURE_ITERS", "30"))
    t = _alloc(b["type"], v)
    for _ in range(warmup):
        _launch(b["type"], v, t)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure):
        _launch(b["type"], v, t)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    avg_us = (t1 - t0) / measure * 1e6
    print(f"[bench] {args.bench} v{args.variant}: warmup {warmup} + measure {measure} "
          f"→ avg {avg_us:.1f} us/run (host 计时, 仅供参考; run_bench 以 msprof 为准)")


if __name__ == "__main__":
    main()
