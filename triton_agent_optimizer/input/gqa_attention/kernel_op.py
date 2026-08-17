#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — GQA Attention + RoPE (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (LLaMA/DeepSeek 推理核心, 无 KV cache):
#    Q = X@Wq [S,H·HD];  K = X@Wk, V = X@Wv [S,KV·HD]   (KV 头数 < Q 头数)
#    RoPE(Q), RoPE(K);  K/V 组内复制 → 标准多头注意力
#    O = (softmax(Q@K^T·s) @ V) @ Wo + X
#  算法: 与 input/vit_block 同套 MHA kernel + RoPE 逐元素 + KV 组内 repeat
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
S = int(os.environ.get("GQA_SEQ", 2048))      # 序列长
D = int(os.environ.get("GQA_DIM", 1024))      # 隐藏维
H = int(os.environ.get("GQA_HEADS", 16))      # Q 头数
HD = int(os.environ.get("GQA_HDIM", 64))      # 头维
KV = int(os.environ.get("GQA_KV", 4))         # KV 头数 (GQA 组数)
DTYPE = torch.float32
SCALE = 1.0 / (HD ** 0.5)
EPS = 1e-6
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # matmul 分块
BLOCK_S = 2048                                # softmax 行分块 (≥S)
BLOCK_ROPE = 1024                             # RoPE 行分块 (≥H*HD=1024)
BLOCK_EL = 1024                               # 逐元素分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

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
    q_ptrs = q_ptr + offs_m[:, None] * (nheads * dim) + head * dim + offs_k[None, :]
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


# RoPE 旋转 (cos/sin 表 [S, HD])
@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                seq, dim, hd,
                BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0)
    d = offs % hd
    cos = tl.load(cos_ptr + row * hd + d, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + row * hd + d, mask=mask, other=0.0)
    d2 = tl.where(d < hd // 2, d + hd // 2, d - hd // 2)
    x2 = tl.load(x_ptr + row * dim + (offs - d) + d2, mask=mask, other=0.0)
    sign = tl.where(d < hd // 2, -1.0, 1.0)
    tl.store(y_ptr + row * dim + offs, x * cos + sign * x2 * sin, mask=mask)



@triton.jit
def rope_kernel2(x_ptr, cos_ptr, sin_ptr, y_ptr,
                seq, dim, hd,
                BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0)
    d = offs % hd
    cos = tl.load(cos_ptr + row * hd + d, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + row * hd + d, mask=mask, other=0.0)
    d2 = tl.where(d < hd // 2, d + hd // 2, d - hd // 2)
    x2 = tl.load(x_ptr + row * dim + (offs - d) + d2, mask=mask, other=0.0)
    sign = tl.where(d < hd // 2, -1.0, 1.0)
    tl.store(y_ptr + row * dim + offs, x * cos + sign * x2 * sin, mask=mask)


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
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    QD = H * HD
    KVD = KV * HD
    x = (torch.randn(S, D, dtype=DTYPE, device=npu) * 0.1)
    wq = (torch.randn(D, QD, dtype=DTYPE, device=npu) * 0.1)
    wk = (torch.randn(D, KVD, dtype=DTYPE, device=npu) * 0.1)
    wv = (torch.randn(D, KVD, dtype=DTYPE, device=npu) * 0.1)
    wo = (torch.randn(QD, D, dtype=DTYPE, device=npu) * 0.1)
    # ★RoPE 表 [S, HD] (窗口外一次性)
    _freq = 1.0 / (10000 ** (torch.arange(0, HD // 2, dtype=torch.float32, device=npu) / (HD // 2)))
    _t = torch.arange(S, dtype=torch.float32, device=npu)
    _ang = _t[:, None] * _freq[None, :]
    cos_t = torch.cos(_ang).repeat(1, 2)
    sin_t = torch.sin(_ang).repeat(1, 2)

    q = torch.empty(S, QD, dtype=DTYPE, device=npu)
    k = torch.empty(S, KVD, dtype=DTYPE, device=npu)
    v = torch.empty(S, KVD, dtype=DTYPE, device=npu)
    # ★KV 组内复制 → [H, ...] 布局 (窗口外一次性; 组内共享同一 KV)
    k_h = torch.empty(H, S, HD, dtype=DTYPE, device=npu)
    v_h = torch.empty(H, S, HD, dtype=DTYPE, device=npu)
    k_t = torch.empty(H, HD, S, dtype=DTYPE, device=npu)
    v_hsd = torch.empty(H, S, HD, dtype=DTYPE, device=npu)
    s = torch.empty(H, S, S, dtype=DTYPE, device=npu)
    p = torch.empty(H, S, S, dtype=DTYPE, device=npu)
    o = torch.empty(H, S, HD, dtype=DTYPE, device=npu)
    oo = torch.empty(S, QD, dtype=DTYPE, device=npu)
    y = torch.empty(S, D, dtype=DTYPE, device=npu)

    grid_mm = (triton.cdiv(S, BLOCK_M) * triton.cdiv(D, BLOCK_N),)
    grid_s = (H * triton.cdiv(S, BLOCK_M) * triton.cdiv(S, BLOCK_N),)
    grid_pv = (H * triton.cdiv(S, BLOCK_M) * triton.cdiv(HD, BLOCK_K),)
    grid_sm = (H * S,)
    grid_rope = (S,)
    grid_el = (triton.cdiv(S * D, BLOCK_EL),)
    print(f"[info] GQA S={S} D={D} H={H} HD={HD} KV={KV} kernels=9")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        matmul_kernel[grid_mm](x, wq, q, S, QD, D,
                               x.stride(0), x.stride(1), wq.stride(0), wq.stride(1),
                               q.stride(0), q.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel2[grid_mm](x, wk, k, S, KVD, D,
                               x.stride(0), x.stride(1), wk.stride(0), wk.stride(1),
                               k.stride(0), k.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel3[grid_mm](x, wv, v, S, KVD, D,
                               x.stride(0), x.stride(1), wv.stride(0), wv.stride(1),
                               v.stride(0), v.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        rope_kernel[grid_rope](q, cos_t, sin_t, q, S, QD, HD, BLOCK=BLOCK_ROPE)
        rope_kernel2[grid_rope](k, cos_t, sin_t, k, S, KVD, HD, BLOCK=BLOCK_ROPE)
        # ★GQA 组内复制 (host, body 内): KV 头 → H 头
        k_h.copy_(k.view(S, KV, HD).transpose(0, 1).repeat_interleave(H // KV, dim=0))
        v_h.copy_(v.view(S, KV, HD).transpose(0, 1).repeat_interleave(H // KV, dim=0))
        k_t.copy_(k_h.permute(0, 2, 1))
        v_hsd.copy_(v_h)
        scores_kernel[grid_s](q, k_t, s, S, HD, H, SCALE,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        softmax_kernel[grid_sm](s, p, H * S, S, BLOCK=BLOCK_S)
        pv_kernel[grid_pv](p, v_hsd, o, S, HD, H,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        oo.copy_(o.permute(1, 0, 2).contiguous().view(S, QD))
        matmul_kernel4[grid_mm](oo, wo, y, S, D, QD,
                               oo.stride(0), oo.stride(1), wo.stride(0), wo.stride(1),
                               y.stride(0), y.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        add_kernel[grid_el](y, x, y, S * D, BLOCK=BLOCK_EL)
    torch.npu.synchronize()
    print("[info] GQA launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        def _rope_ref(t, hd, heads):
            th = t.view(S, heads, hd)
            x1 = th[..., : hd // 2]
            x2 = th[..., hd // 2:]
            return (th * cos_t[:, None, :] + torch.cat([-x2, x1], dim=-1) * sin_t[:, None, :]
                    ).reshape(S, heads * hd)
        q_r = _rope_ref(x @ wq, HD, H)
        k_r = _rope_ref(x @ wk, HD, KV)
        v_r = x @ wv
        qh = q_r.view(S, H, HD).transpose(0, 1)
        kh = k_r.view(S, KV, HD).transpose(0, 1).repeat_interleave(H // KV, dim=0)
        vh = v_r.view(S, KV, HD).transpose(0, 1).repeat_interleave(H // KV, dim=0)
        ph = torch.softmax((qh @ kh.transpose(-1, -2)) * SCALE, dim=-1)
        ref = (ph @ vh).transpose(0, 1).reshape(S, QD) @ wo + x
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
