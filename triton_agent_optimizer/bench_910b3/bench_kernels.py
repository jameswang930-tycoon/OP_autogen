#!/usr/bin/env python3
"""910B3 硬件基准 kernels — 纯 triton (与 kernel_op.py 同风格), 自包含可跑。

每个 bench: 分配 NPU 张量 → 启动 kernel → 同步。返回 (bytes, flops) 供算带宽/算力。

═══ 怎么运行 (910B3 服务器) ═══
  单测某个 kernel 能跑 (不带 msprof):
    python3 bench_kernels.py --bench read_bw
  列出所有:
    python3 bench_kernels.py --list
  完整测量 (warmup+msprof 多轮): 用 run_bench.py (见其顶部教程)
  环境: conda activate triton-npu && source set_env.sh

  测什么 (6 个):
    read_bw  GM 读带宽 | write_bw  GM 写带宽 | copy_bw  GM 拷贝
    l2_bw    L2 读带宽 | mm       cube 算力  | vec      Vec 吞吐
  尺寸 env: BENCH_BW_N / BENCH_L2_N / BENCH_MM / BENCH_VEC_N
"""
import os
import sys
import torch
try:
    import torch_npu          # 仅服务器有; --list/--help 不需要
except ImportError:
    pass
import triton
import triton.language as tl

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ═══ 配置 (env 可覆盖) ═══
BW_N      = int(os.environ.get("BENCH_BW_N", 1 << 22))       # 4M 元素 = 16MB fp32
L2_N      = int(os.environ.get("BENCH_L2_N", 1 << 20))       # 1M 元素 = 4MB (L2 内)
L2_ITERS  = int(os.environ.get("BENCH_L2_ITERS", 64))        # L2 内反复读次数
MM        = int(os.environ.get("BENCH_MM", 4096))            # 4096³ matmul (compute-bound)
VEC_N     = int(os.environ.get("BENCH_VEC_N", 1 << 23))      # 8M 元素
BLOCK     = 1024
DTYPE     = torch.float32
MM_BLOCK_M, MM_BLOCK_N, MM_BLOCK_K = 128, 128, 64


# ═══════════════════════════════════════════════════════════════════════
#  kernels
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def read_bw_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid, tl.sum(x))   # 归约防 load 被优化掉


@triton.jit
def write_bw_kernel(x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(x_ptr + offs, 1.0, mask=mask)


@triton.jit
def copy_bw_kernel(a_ptr, b_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(b_ptr + offs, tl.load(a_ptr + offs, mask=mask, other=0.0), mask=mask)


@triton.jit
def l2_bw_kernel(x_ptr, out_ptr, N, ITERS, BLOCK: tl.constexpr):
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
def vec_bw_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=mask)


# ═══════════════════════════════════════════════════════════════════════
#  bench 运行函数 → (bytes, flops)
# ═══════════════════════════════════════════════════════════════════════

def run_read_bw():
    """GM 读带宽: 读 BW_N*4B, 写几乎为 0。"""
    x = torch.empty(BW_N, dtype=DTYPE, device="npu")
    out = torch.empty(BW_N // BLOCK, dtype=DTYPE, device="npu")
    grid = (BW_N // BLOCK,)
    read_bw_kernel[grid](x, out, BW_N, BLOCK=BLOCK)
    torch.npu.synchronize()
    return BW_N * 4, 0


def run_write_bw():
    """GM 写带宽: 写 BW_N*4B。"""
    x = torch.empty(BW_N, dtype=DTYPE, device="npu")
    grid = (BW_N // BLOCK,)
    write_bw_kernel[grid](x, BW_N, BLOCK=BLOCK)
    torch.npu.synchronize()
    return 0, BW_N * 4


def run_copy_bw():
    """GM 拷贝带宽: 读 A + 写 B, 共 2*BW_N*4B。"""
    a = torch.empty(BW_N, dtype=DTYPE, device="npu")
    b = torch.empty(BW_N, dtype=DTYPE, device="npu")
    grid = (BW_N // BLOCK,)
    copy_bw_kernel[grid](a, b, BW_N, BLOCK=BLOCK)
    torch.npu.synchronize()
    return BW_N * 4, BW_N * 4


def run_l2_bw():
    """L2 读带宽: 小数组 (L2 内) 反复读 L2_ITERS 次。"""
    x = torch.empty(L2_N, dtype=DTYPE, device="npu")
    out = torch.empty(L2_N // BLOCK, dtype=DTYPE, device="npu")
    grid = (L2_N // BLOCK,)
    l2_bw_kernel[grid](x, out, L2_N, L2_ITERS, BLOCK=BLOCK)
    torch.npu.synchronize()
    return L2_N * 4 * L2_ITERS, 0


def run_mm():
    """cube 算力: 4096³ fp16 matmul (compute-bound), flops=2*M*N*K。"""
    dtype = torch.float16
    a = torch.rand(MM, MM, dtype=dtype, device="npu")
    b = torch.rand(MM, MM, dtype=dtype, device="npu")
    c = torch.empty(MM, MM, dtype=torch.float32, device="npu")
    grid = ((MM // MM_BLOCK_M) * (MM // MM_BLOCK_N),)
    mm_kernel[grid](a, b, c, MM, MM, MM,
                    BLOCK_M=MM_BLOCK_M, BLOCK_N=MM_BLOCK_N, BLOCK_K=MM_BLOCK_K)
    torch.npu.synchronize()
    bytes_ = (MM * MM * 3) * 2   # 读A+读B+写C, fp16
    return bytes_, 0


def run_vec():
    """Vec 吞吐: 读 A+B + 写 C, 共 3*VEC_N*4B。"""
    a = torch.empty(VEC_N, dtype=DTYPE, device="npu")
    b = torch.empty(VEC_N, dtype=DTYPE, device="npu")
    c = torch.empty(VEC_N, dtype=DTYPE, device="npu")
    grid = (VEC_N // BLOCK,)
    vec_bw_kernel[grid](a, b, c, VEC_N, BLOCK=BLOCK)
    torch.npu.synchronize()
    return VEC_N * 4 * 2, VEC_N * 4


# ═══ bench 注册 (flops: 计算量, 算 TFLOPS 用; 其余为字节数) ═══
BENCHES = {
    "read_bw":   {"run": run_read_bw,  "desc": "GM 读带宽 (只读大数组)", "flops": 0},
    "write_bw":  {"run": run_write_bw, "desc": "GM 写带宽 (只写大数组)", "flops": 0},
    "copy_bw":   {"run": run_copy_bw,  "desc": "GM 拷贝带宽 (读A写B)", "flops": 0},
    "l2_bw":     {"run": run_l2_bw,    "desc": "L2 读带宽 (L2内小数组反复读)", "flops": 0},
    "mm":        {"run": run_mm,       "desc": "cube 算力 (4096³ fp16 matmul)", "flops": 2 * MM * MM * MM},
    "vec":       {"run": run_vec,      "desc": "Vec 吞吐 (大向量 add)", "flops": 0},
}


def main():
    import argparse
    p = argparse.ArgumentParser(description="910B3 bench kernels")
    p.add_argument("--bench", type=str, help=f"bench 名: {list(BENCHES)}")
    p.add_argument("--list", action="store_true", help="列出所有 bench")
    args = p.parse_args()
    if args.list:
        for n, b in BENCHES.items():
            print(f"  {n:10s} {b['desc']}")
        return
    if args.bench not in BENCHES:
        print(f"❌ 未知 bench: {args.bench}. 可选: {list(BENCHES)}")
        sys.exit(1)
    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    rb, wb = BENCHES[args.bench]["run"]()
    print(f"[ok] {args.bench}: read={rb}B write={wb}B")


if __name__ == "__main__":
    main()
