#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — SwiGLU MLP (LLaMA FFN) (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链: Y = (SiLU(X@W1) * (X@W2)) @ W3
#  算法: up/gate 两个 matmul → 逐元素 silu×门控 → down matmul
#  (中上水平基线: 标准 GEMM + 逐元素融合; 优化空间: 门控并入 epilogue / 分块)
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块 (与 bench_910b3 _shapes 对齐)
# ═══════════════════════════════════════════════════════════════════════════════
S = int(os.environ.get("SM_SEQ", 2048))       # 序列长 (M 维)
D = int(os.environ.get("SM_DIM", 1024))       # 输入/输出维
FFN = int(os.environ.get("SM_FFN", 4096))     # 中间维
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # matmul 分块
BLOCK_EL = 1024                               # 逐元素分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# 标准 GEMM (up/gate/down 共用; 独立函数名防 msprof 同名聚合)
@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr,
              M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 逐元素: gated = silu(u) * g  (融合门控)

@triton.jit
def mm_kernel2(a_ptr, b_ptr, c_ptr,
              M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 逐元素: gated = silu(u) * g  (融合门控)

@triton.jit
def mm_kernel3(a_ptr, b_ptr, c_ptr,
              M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 逐元素: gated = silu(u) * g  (融合门控)
@triton.jit
def silu_gate_kernel(u_ptr, g_ptr, y_ptr,
                     n_elements,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    u = tl.load(u_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    s = u / (1.0 + tl.exp(-u))                # SiLU (与 torch F.silu 一致)
    tl.store(y_ptr + offs, s * g, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    import torch.nn.functional as F
    npu = torch.device("npu")
    x = (torch.randn(S, D, dtype=DTYPE, device=npu) * 0.1)
    w1 = (torch.randn(D, FFN, dtype=DTYPE, device=npu) * 0.1)    # up
    w2 = (torch.randn(D, FFN, dtype=DTYPE, device=npu) * 0.1)    # gate
    w3 = (torch.randn(FFN, D, dtype=DTYPE, device=npu) * 0.1)    # down
    u = torch.empty(S, FFN, dtype=DTYPE, device=npu)
    g = torch.empty(S, FFN, dtype=DTYPE, device=npu)
    act = torch.empty(S, FFN, dtype=DTYPE, device=npu)
    y = torch.empty(S, D, dtype=DTYPE, device=npu)

    grid_mm = (triton.cdiv(S, BLOCK_M) * triton.cdiv(FFN, BLOCK_N),)
    grid_dn = (triton.cdiv(S, BLOCK_M) * triton.cdiv(D, BLOCK_N),)
    grid_el = (triton.cdiv(S * FFN, BLOCK_EL),)
    print(f"[info] SwiGLU MLP S={S} D={D} FFN={FFN} kernels=4 grid_mm={grid_mm[0]} "
          f"grid_el={grid_el[0]} block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        # up: u = X @ W1
        mm_kernel[grid_mm](x, w1, u, S, FFN, D,
                           x.stride(0), x.stride(1), w1.stride(0), w1.stride(1),
                           u.stride(0), u.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # gate: g = X @ W2
        mm_kernel2[grid_mm](x, w2, g, S, FFN, D,
                           x.stride(0), x.stride(1), w2.stride(0), w2.stride(1),
                           g.stride(0), g.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # act = silu(u) * g
        silu_gate_kernel[grid_el](u, g, act, S * FFN, BLOCK=BLOCK_EL)
        # y = act @ W3
        mm_kernel3[grid_dn](act, w3, y, S, D, FFN,
                           act.stride(0), act.stride(1), w3.stride(0), w3.stride(1),
                           y.stride(0), y.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()
    print("[info] SwiGLU MLP launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        ref = (F.silu(x @ w1) * (x @ w2)) @ w3
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
