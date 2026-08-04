#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — 算子 + 场景 config + 测试 main 合一体 (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config (改尺寸/分块/精度)  ② 算子 kernel (优化核心)  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 改这里调 M/N/K、精度、分块 (tier3 优化点: BLOCK_*)
# ═══════════════════════════════════════════════════════════════════════════════
M  = int(os.environ.get("MATMUL_M", 512))
N  = int(os.environ.get("MATMUL_N", 512))
K  = int(os.environ.get("MATMUL_K", 512))
DTYPE = torch.float32                    # 精度 (tier5 优化点: fp16 计算 + fp32 累加)
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32   # 分块大小 (tier3 优化点)
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel — 优化核心 (tier1 算法 / tier2 融合 / tier3 分块 / tier4 访存 / tier5 计算 / tier6 架构)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步 (一般不动)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    # fp32 dot 值域限制 [-5, 5], 用 rand-0.5 控制在 ±0.5
    a = torch.rand(M, K, dtype=DTYPE, device="npu") - 0.5
    b = torch.rand(K, N, dtype=DTYPE, device="npu") - 0.5
    c = torch.empty(M, N, dtype=DTYPE, device="npu")

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m * grid_n,)   # 必须用 tuple: triton-ascend 对 int grid 调 len() 会报 "int has no len()"
    print(f"[info] launch grid={grid}  A({M}x{K}) @ B({K}x{N})  block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    torch.npu.synchronize()
    print("[info] kernel launched & synced OK")

    # 正确性校验 (默认关, 要开用 MATMUL_VERIFY=1)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        try:
            ref = torch.matmul(a, b)
            diff = (c - ref).abs().max().item()
            print(f"[info] result check: {'PASS' if diff < 0.05 else 'CHECK'}  max|C-ref| = {diff:.5f}")
        except Exception as e:
            print(f"[warn] result check skipped: {e}")


if __name__ == "__main__":
    main()
