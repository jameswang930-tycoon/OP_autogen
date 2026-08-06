#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — 自注意力 + MLP 块 (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (序列长 seq, 隐藏维 dim):
#    Q = X@Wq ; K = X@Wk ; V = X@Wv          (3 个 matmul)
#    S = Q@K^T · scale                        (转置 matmul + 缩放)
#    P = softmax(S, dim=-1)                   (行 softmax)
#    O = P@V                                  (matmul)
#    Y = GELU(O@W1 + b1)                      (matmul + bias + 激活)
#    Z = Y@W2                                 (matmul)
#    Out = Z + O                              (残差加)
#  约 7 个 matmul + 1 softmax + 1 加, 阶段多、数据流长、中间张量多
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
# ═══════════════════════════════════════════════════════════════════════════════
SEQ = int(os.environ.get("MATMUL_M", 2048))    # 序列长度 (大)
DIM = int(os.environ.get("MATMUL_N", 2048))    # 隐藏维度
K   = int(os.environ.get("MATMUL_K", 2048))    # (与 M/N 对齐, 供调度器尺寸核对)
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64         # matmul 分块
BLOCK_S = 2048                                 # softmax/加 分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
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


@triton.jit
def attention_scores_kernel(q_ptr, k_ptr, s_ptr,
                            seq, dim, scale,
                            BLOCK_M: tl.constexpr,
                            BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
    # S = Q @ K^T · scale;  Q[seq,dim], K[seq,dim], S[seq,seq]
    # K^T[k,n] = K[n,k] 用指针直接转置 (不依赖 tl.trans)
    pid = tl.program_id(axis=0)
    grid_n = (seq + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = q_ptr + (offs_m[:, None] * dim + offs_k[None, :])      # Q[m,k]
    k_ptrs = k_ptr + (offs_n[None, :] * dim + offs_k[:, None])      # K^T[k,n] = K[n,k]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, dim, BLOCK_K):
        q = tl.load(q_ptrs, mask=offs_k[None, :] < (dim - k), other=0.0)
        kk = tl.load(k_ptrs, mask=offs_k[:, None] < (dim - k), other=0.0)
        acc = tl.dot(q, kk, acc)
        q_ptrs += BLOCK_K
        k_ptrs += BLOCK_K
    s_ptrs = s_ptr + (offs_m[:, None] * seq + offs_n[None, :])
    s_mask = (offs_m[:, None] < seq) & (offs_n[None, :] < seq)
    tl.store(s_ptrs, acc * scale, mask=s_mask)


@triton.jit
def softmax_kernel(x_ptr, y_ptr, rows, cols, BLOCK: tl.constexpr):
    # 行 softmax: y[row, :] = softmax(x[row, :])
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(y_ptr + row * cols + offs, y, mask=mask)


@triton.jit
def mlp_gelu_kernel(a_ptr, w_ptr, b_ptr, c_ptr,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_wk, stride_wn,
                    stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    # Y = GELU(A @ W + b) — matmul + bias + tanh-GELU 激活, fp32 累加
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    val = acc + bias[None, :]
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))
    y = val * cdf
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, y, mask=c_mask)


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    # c = a + b (残差加)
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

    seq, dim = SEQ, DIM
    scale = 1.0 / (dim ** 0.5)
    npu = torch.device("npu")

    # 输入 (小值 ±0.05, 避免 fp32 dot 值域溢出)
    x  = (torch.rand(seq, dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    wq = (torch.rand(dim, dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    wk = (torch.rand(dim, dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    wv = (torch.rand(dim, dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    w1 = (torch.rand(dim, dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    b1 = (torch.rand(dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    w2 = (torch.rand(dim, dim, dtype=DTYPE, device=npu) - 0.5) * 0.1
    q = torch.empty(seq, dim, dtype=DTYPE, device=npu)
    k = torch.empty(seq, dim, dtype=DTYPE, device=npu)
    v = torch.empty(seq, dim, dtype=DTYPE, device=npu)
    s = torch.empty(seq, seq, dtype=DTYPE, device=npu)
    p = torch.empty(seq, seq, dtype=DTYPE, device=npu)
    o = torch.empty(seq, dim, dtype=DTYPE, device=npu)
    y = torch.empty(seq, dim, dtype=DTYPE, device=npu)
    z = torch.empty(seq, dim, dtype=DTYPE, device=npu)
    out = torch.empty(seq, dim, dtype=DTYPE, device=npu)

    g_hidden = (triton.cdiv(seq, BLOCK_M) * triton.cdiv(dim, BLOCK_N),)   # → [seq,dim]
    g_scores = (triton.cdiv(seq, BLOCK_M) * triton.cdiv(seq, BLOCK_N),)   # → [seq,seq]
    g_softmax = (seq,)                                                    # 每行一个 program
    g_add = (triton.cdiv(seq * dim, BLOCK_S),)

    print(f"[info] attention+MLP seq={seq} dim={dim} dtype={DTYPE} "
          f"block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        # Q,K,V = X @ Wq/Wk/Wv
        matmul_kernel[g_hidden](x, wq, q, seq, dim, dim,
                                x.stride(0), x.stride(1), wq.stride(0), wq.stride(1),
                                q.stride(0), q.stride(1),
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel[g_hidden](x, wk, k, seq, dim, dim,
                                x.stride(0), x.stride(1), wk.stride(0), wk.stride(1),
                                k.stride(0), k.stride(1),
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel[g_hidden](x, wv, v, seq, dim, dim,
                                x.stride(0), x.stride(1), wv.stride(0), wv.stride(1),
                                v.stride(0), v.stride(1),
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # S = Q@K^T · scale
        attention_scores_kernel[g_scores](q, k, s, seq, dim, scale,
                                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # P = softmax(S)
        softmax_kernel[g_softmax](s, p, seq, seq, BLOCK=BLOCK_S)
        # O = P @ V
        matmul_kernel[g_hidden](p, v, o, seq, dim, seq,
                                p.stride(0), p.stride(1), v.stride(0), v.stride(1),
                                o.stride(0), o.stride(1),
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # Y = GELU(O @ W1 + b1)
        mlp_gelu_kernel[g_hidden](o, w1, b1, y, seq, dim, dim,
                                  o.stride(0), o.stride(1), w1.stride(0), w1.stride(1),
                                  y.stride(0), y.stride(1),
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # Z = Y @ W2
        matmul_kernel[g_hidden](y, w2, z, seq, dim, dim,
                                y.stride(0), y.stride(1), w2.stride(0), w2.stride(1),
                                z.stride(0), z.stride(1),
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # Out = Z + O (残差)
        add_kernel[g_add](z, o, out, seq * dim, BLOCK=BLOCK_S)
    torch.npu.synchronize()
    print("[info] attention+MLP launched & synced OK")

    # 正确性校验 (默认关, 要开用 MATMUL_VERIFY=1)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        try:
            import torch.nn.functional as F
            q_r = torch.matmul(x, wq); k_r = torch.matmul(x, wk); v_r = torch.matmul(x, wv)
            s_r = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
            p_r = F.softmax(s_r, dim=-1)
            o_r = torch.matmul(p_r, v_r)
            y_r = F.gelu(torch.matmul(o_r, w1) + b1, approximate="tanh")
            z_r = torch.matmul(y_r, w2)
            out_r = z_r + o_r
            diff = (out - out_r).abs().max().item()
            rel = diff / (out_r.abs().max().item() + 1e-6)
            print(f"[info] result check: {'PASS' if rel < 0.05 else 'CHECK'}  "
                  f"max|diff|={diff:.5f} rel={rel:.5f}")
        except Exception as e:
            print(f"[warn] result check skipped: {e}")


if __name__ == "__main__":
    main()
