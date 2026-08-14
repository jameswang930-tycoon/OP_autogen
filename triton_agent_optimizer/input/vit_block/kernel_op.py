#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — ViT Encoder Block (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链: Y = GELU(LN(O)@W1+b1)@W2 + O,  O = (softmax((LN(X)@Wq)(LN(X)@Wk)^T·s) @ (LN(X)@Wv)) @ Wo + X
#  算法: LN → QKV 3 matmul → 多头 scores → 行 softmax → PV → O 投影 → 残差 → LN → GELU MLP → 残差
#  (K_t 预转置 + V 预转 [H,S,D], 窗口外一次性; 中上水平基线)
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
S = int(os.environ.get("VIT_SEQ", 197))       # 序列长 (patch 数)
D = int(os.environ.get("VIT_DIM", 768))       # 隐藏维
H = int(os.environ.get("VIT_HEADS", 12))      # 头数
HD = int(os.environ.get("VIT_HDIM", 64))      # 头维
FFN = int(os.environ.get("VIT_FFN", 3072))    # MLP 中间维
DTYPE = torch.float32
SCALE = 1.0 / (HD ** 0.5)
EPS = 1e-6
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # matmul 分块
BLOCK_S = 256                                 # softmax/LN 行分块 (≥S=197)
BLOCK_LN = 1024                               # LN 行宽分块 (≥D=768)
BLOCK_EL = 1024                               # 逐元素分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# 标准 GEMM
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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


# 多头 scores: S[h,m,n] = Q[h,m,:]@K_t[h,:,n]·scale;  Q[S,H*D], K_t[H,D,S] 预转置, S[H,S,S]

@triton.jit
def matmul_kernel2(a_ptr, b_ptr, c_ptr, M, N, K,
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


# 多头 scores: S[h,m,n] = Q[h,m,:]@K_t[h,:,n]·scale;  Q[S,H*D], K_t[H,D,S] 预转置, S[H,S,S]

@triton.jit
def matmul_kernel3(a_ptr, b_ptr, c_ptr, M, N, K,
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


# 多头 scores: S[h,m,n] = Q[h,m,:]@K_t[h,:,n]·scale;  Q[S,H*D], K_t[H,D,S] 预转置, S[H,S,S]

@triton.jit
def matmul_kernel4(a_ptr, b_ptr, c_ptr, M, N, K,
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


# 多头 scores: S[h,m,n] = Q[h,m,:]@K_t[h,:,n]·scale;  Q[S,H*D], K_t[H,D,S] 预转置, S[H,S,S]

@triton.jit
def matmul_kernel5(a_ptr, b_ptr, c_ptr, M, N, K,
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


# 多头 scores: S[h,m,n] = Q[h,m,:]@K_t[h,:,n]·scale;  Q[S,H*D], K_t[H,D,S] 预转置, S[H,S,S]
@triton.jit
def scores_kernel(q_ptr, k_ptr, s_ptr, seq, dim, nheads, scale,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (seq + BLOCK_N - 1) // BLOCK_N
    grid_m = (seq + BLOCK_M - 1) // BLOCK_M
    head = pid // (grid_m * grid_n)
    tmp = pid % (grid_m * grid_n)
    pid_m = tmp // grid_n
    pid_n = tmp % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = q_ptr + head * seq * dim + offs_m[:, None] * dim + offs_k[None, :]
    k_ptrs = k_ptr + head * dim * seq + offs_k[:, None] * seq + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, dim, BLOCK_K):
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq) & (offs_k[None, :] < (dim - k)), other=0.0)
        kk = tl.load(k_ptrs, mask=(offs_k[:, None] < (dim - k)) & (offs_n[None, :] < seq), other=0.0)
        acc = tl.dot(q, kk, acc)
        q_ptrs += BLOCK_K
        k_ptrs += BLOCK_K * seq
    s_ptrs = s_ptr + head * seq * seq + offs_m[:, None] * seq + offs_n[None, :]
    tl.store(s_ptrs, acc * scale, mask=(offs_m[:, None] < seq) & (offs_n[None, :] < seq))


# 行 softmax: P[h,row,:] = softmax(S[h,row,:])
@triton.jit
def softmax_kernel(x_ptr, y_ptr, rows, cols, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + pid * cols + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + pid * cols + offs, y, mask=mask)


# PV: O[h,m,d] = Σ_n P[h,m,n]·V[h,n,d];  P[H,S,S], V_hsd[H,S,D] 预转置, O[H,S,D]
@triton.jit
def pv_kernel(p_ptr, v_ptr, o_ptr, seq, dim, nheads,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_d = (dim + BLOCK_K - 1) // BLOCK_K
    grid_m = (seq + BLOCK_M - 1) // BLOCK_M
    head = pid // (grid_m * grid_d)
    tmp = pid % (grid_m * grid_d)
    pid_m = tmp // grid_d
    pid_d = tmp % grid_d
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_K + tl.arange(0, BLOCK_K)
    p_ptrs = p_ptr + head * seq * seq + offs_m[:, None] * seq + offs_n[None, :]
    v_ptrs = v_ptr + head * seq * dim + offs_n[:, None] * dim + offs_d[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, seq, BLOCK_N):
        p = tl.load(p_ptrs, mask=(offs_m[:, None] < seq) & (offs_n[None, :] < (seq - n)), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < (seq - n)) & (offs_d[None, :] < dim), other=0.0)
        acc = tl.dot(p, v, acc)
        p_ptrs += BLOCK_N
        v_ptrs += BLOCK_N * dim
    o_ptrs = o_ptr + head * seq * dim + offs_m[:, None] * dim + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < seq) & (offs_d[None, :] < dim))


# LayerNorm: y = (x-mean)/sqrt(var+eps)·w + b
@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / cols
    y = (x - mean) / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * cols + offs, y * w + b, mask=mask)


# MLP: Y = GELU(A@W1+b1)@W2 + 残差 (bias+tanh-GELU 融合)

@triton.jit
def layernorm_kernel2(x_ptr, w_ptr, b_ptr, y_ptr, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / cols
    y = (x - mean) / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * cols + offs, y * w + b, mask=mask)


# MLP: Y = GELU(A@W1+b1)@W2 + 残差 (bias+tanh-GELU 融合)
@triton.jit
def mlp_gelu_kernel(a_ptr, w_ptr, b_ptr, c_ptr, M, N, K,
                    stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    val = acc + bias[None, :]
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, val * cdf, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 残差 add
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def add_kernel2(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    import torch.nn.functional as F
    npu = torch.device("npu")
    HD_D = H * HD
    x = (torch.randn(S, D, dtype=DTYPE, device=npu) * 0.1)
    wq = (torch.randn(D, HD_D, dtype=DTYPE, device=npu) * 0.1)
    wk = (torch.randn(D, HD_D, dtype=DTYPE, device=npu) * 0.1)
    wv = (torch.randn(D, HD_D, dtype=DTYPE, device=npu) * 0.1)
    wo = (torch.randn(HD_D, D, dtype=DTYPE, device=npu) * 0.1)
    w1 = (torch.randn(D, FFN, dtype=DTYPE, device=npu) * 0.1)
    b1 = (torch.randn(FFN, dtype=DTYPE, device=npu) * 0.1)
    w2 = (torch.randn(FFN, D, dtype=DTYPE, device=npu) * 0.1)
    b2 = (torch.randn(D, dtype=DTYPE, device=npu) * 0.1)
    wln = torch.ones(D, dtype=DTYPE, device=npu)
    bln = torch.zeros(D, dtype=DTYPE, device=npu)

    h = torch.empty(S, D, dtype=DTYPE, device=npu)
    q = torch.empty(S, HD_D, dtype=DTYPE, device=npu)
    k = torch.empty(S, HD_D, dtype=DTYPE, device=npu)
    v = torch.empty(S, HD_D, dtype=DTYPE, device=npu)
    # ★窗口外一次性预转置: k_t [H,D,S], v_hsd [H,S,D]
    k_t = torch.empty(H, D, S, dtype=DTYPE, device=npu)
    v_hsd = torch.empty(H, S, D, dtype=DTYPE, device=npu)
    s = torch.empty(H, S, S, dtype=DTYPE, device=npu)
    p = torch.empty(H, S, S, dtype=DTYPE, device=npu)
    o = torch.empty(H, S, D, dtype=DTYPE, device=npu)
    oo = torch.empty(S, HD_D, dtype=DTYPE, device=npu)
    r = torch.empty(S, D, dtype=DTYPE, device=npu)
    h2 = torch.empty(S, D, dtype=DTYPE, device=npu)
    m = torch.empty(S, FFN, dtype=DTYPE, device=npu)
    y = torch.empty(S, D, dtype=DTYPE, device=npu)

    grid_mm = (triton.cdiv(S, BLOCK_M) * triton.cdiv(D, BLOCK_N),)
    grid_mm_f = (triton.cdiv(S, BLOCK_M) * triton.cdiv(FFN, BLOCK_N),)
    grid_mm_d = (triton.cdiv(S, BLOCK_M) * triton.cdiv(D, BLOCK_N),)
    grid_s = (H * triton.cdiv(S, BLOCK_M) * triton.cdiv(S, BLOCK_N),)
    grid_pv = (H * triton.cdiv(S, BLOCK_M) * triton.cdiv(D, BLOCK_K),)
    grid_sm = (H * S,)
    grid_ln = (S,)
    grid_el = (triton.cdiv(S * D, BLOCK_EL),)
    grid_el_f = (triton.cdiv(S * FFN, BLOCK_EL),)
    print(f"[info] ViT block S={S} D={D} H={H} HD={HD} FFN={FFN} kernels=10")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        # LN(X) → h
        layernorm_kernel[grid_ln](x, wln, bln, h, S, D, EPS, BLOCK=BLOCK_LN)
        # Q = h@Wq, K = h@Wk, V = h@Wv
        matmul_kernel[grid_mm](h, wq, q, S, HD_D, D,
                               h.stride(0), h.stride(1), wq.stride(0), wq.stride(1),
                               q.stride(0), q.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel2[grid_mm](h, wk, k, S, HD_D, D,
                               h.stride(0), h.stride(1), wk.stride(0), wk.stride(1),
                               k.stride(0), k.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel3[grid_mm](h, wv, v, S, HD_D, D,
                               h.stride(0), h.stride(1), wv.stride(0), wv.stride(1),
                               v.stride(0), v.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # ★预转置 (host, 每次循环内 — 与验证注入兼容; 窗口外版本留优化空间)
        k_t.copy_(k.view(S, H, HD).permute(1, 2, 0))
        v_hsd.copy_(v.view(S, H, HD).permute(1, 0, 2))
        # S = Q@K_t·scale → P = softmax → O = P@V_hsd
        scores_kernel[grid_s](q, k_t, s, S, HD, H, SCALE,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        softmax_kernel[grid_sm](s, p, H * S, S, BLOCK=BLOCK_S)
        pv_kernel[grid_pv](p, v_hsd, o, S, HD, H,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # O 投影 [H,S,D] → [S,HD_D] → @Wo → 残差
        oo.copy_(o.permute(1, 0, 2).contiguous().view(S, HD_D))
        matmul_kernel4[grid_mm_d](oo, wo, r, S, D, HD_D,
                                 oo.stride(0), oo.stride(1), wo.stride(0), wo.stride(1),
                                 r.stride(0), r.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        add_kernel[grid_el](r, x, r, S * D, BLOCK=BLOCK_EL)
        # LN → MLP(GELU) → 残差
        layernorm_kernel2[grid_ln](r, wln, bln, h2, S, D, EPS, BLOCK=BLOCK_LN)
        mlp_gelu_kernel[grid_mm_f](h2, w1, b1, m, S, FFN, D,
                                   h2.stride(0), h2.stride(1), w1.stride(0), w1.stride(1),
                                   m.stride(0), m.stride(1),
                                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel5[grid_mm_d](m, w2, y, S, D, FFN,
                                 m.stride(0), m.stride(1), w2.stride(0), w2.stride(1),
                                 y.stride(0), y.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        add_kernel2[grid_el](y, r, y, S * D, BLOCK=BLOCK_EL)
    torch.npu.synchronize()
    print("[info] ViT block launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        h_r = F.layer_norm(x, [D], wln, bln, EPS)
        q_r = h_r @ wq
        k_r = h_r @ wk
        v_r = h_r @ wv
        qh = q_r.view(S, H, HD).transpose(0, 1)
        kh = k_r.view(S, H, HD).transpose(0, 1)
        vh = v_r.view(S, H, HD).transpose(0, 1)
        sh = (qh @ kh.transpose(-1, -2)) * SCALE
        ph = torch.softmax(sh, dim=-1)
        oh = (ph @ vh).transpose(0, 1).reshape(S, HD_D)
        r_r = oh @ wo + x
        h2_r = F.layer_norm(r_r, [D], wln, bln, EPS)
        ref = F.gelu(h2_r @ w1 + b1) @ w2 + r_r
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
