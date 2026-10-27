#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Mixture of Experts (MoE, top-k) (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (SwiGLU experts + top-2 router):
#    logits = X @ Router;  idx, w = topk(softmax(logits), k=2)
#    u = X@W1, g = X@W2 (全量 E 个 expert, 扁平 [E*FFN]);  act = silu(u)·g
#    Y_e = act_e @ W3_e  (per-expert bmm);  Y = Σ_k w[k]·Y_e[idx[k]]  (topk 加权)
#  算法: 全量 experts (基准语义正确优先; 稀疏路由留优化空间), topk 用 torch (body 内)
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
S = int(os.environ.get("MOE_SEQ", 1024))      # 序列长
D = int(os.environ.get("MOE_DIM", 1024))      # 隐藏维
E = int(os.environ.get("MOE_NEXP", 8))        # expert 数
FFN = int(os.environ.get("MOE_FFN", 2048))    # expert FFN 中间维
TOPK = 2
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # matmul 分块
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


# 逐元素: gated = silu(u) * g  (扁平 [S, E*FFN])

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


# 逐元素: gated = silu(u) * g  (扁平 [S, E*FFN])

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


# 逐元素: gated = silu(u) * g  (扁平 [S, E*FFN])
@triton.jit
def silu_gate_kernel(u_ptr, g_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    u = tl.load(u_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    s = u / (1.0 + tl.exp(-u))
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
    router_w = (torch.randn(D, E, dtype=DTYPE, device=npu) * 0.1)
    w1f = (torch.randn(D, E * FFN, dtype=DTYPE, device=npu) * 0.1)   # up 扁平
    w2f = (torch.randn(D, E * FFN, dtype=DTYPE, device=npu) * 0.1)   # gate 扁平
    w3 = (torch.randn(E, FFN, D, dtype=DTYPE, device=npu) * 0.1)     # down [E,FFN,D]

    logits = torch.empty(S, E, dtype=DTYPE, device=npu)
    u = torch.empty(S, E * FFN, dtype=DTYPE, device=npu)
    g = torch.empty(S, E * FFN, dtype=DTYPE, device=npu)
    act = torch.empty(S, E * FFN, dtype=DTYPE, device=npu)
    act_t = torch.empty(E, S, FFN, dtype=DTYPE, device=npu)          # [E,S,FFN] (bmm 输入)
    y_e = torch.empty(E, S, D, dtype=DTYPE, device=npu)              # [E,S,D]
    y_esd = torch.empty(S, E, D, dtype=DTYPE, device=npu)            # [S,E,D]
    y = torch.empty(S, D, dtype=DTYPE, device=npu)

    grid_mm_e = (triton.cdiv(S, BLOCK_M) * triton.cdiv(E * FFN, BLOCK_N),)
    grid_mm_r = (triton.cdiv(S, BLOCK_M) * triton.cdiv(E, BLOCK_N),)
    grid_el = (triton.cdiv(S * E * FFN, BLOCK_EL),)
    print(f"[info] MoE S={S} D={D} E={E} FFN={FFN} topk={TOPK} kernels=5 (bmm/topk 走 torch)")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        # router
        matmul_kernel[grid_mm_r](x, router_w, logits, S, E, D,
                                 x.stride(0), x.stride(1), router_w.stride(0), router_w.stride(1),
                                 logits.stride(0), logits.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # topk 权重 (torch, body 内)
        _tv, _ti = torch.topk(logits, TOPK, dim=-1)
        w = F.softmax(_tv, dim=-1)                                    # [S, TOPK]
        # up/gate (全量 experts)
        matmul_kernel2[grid_mm_e](x, w1f, u, S, E * FFN, D,
                                 x.stride(0), x.stride(1), w1f.stride(0), w1f.stride(1),
                                 u.stride(0), u.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        matmul_kernel3[grid_mm_e](x, w2f, g, S, E * FFN, D,
                                 x.stride(0), x.stride(1), w2f.stride(0), w2f.stride(1),
                                 g.stride(0), g.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        silu_gate_kernel[grid_el](u, g, act, S * E * FFN, BLOCK=BLOCK_EL)
        # per-expert down (bmm): [E,S,FFN] @ [E,FFN,D] → [E,S,D] → [S,E,D]
        act_t.copy_(act.view(S, E, FFN).transpose(0, 1).contiguous())
        y_e = torch.bmm(act_t, w3)
        y_esd.copy_(y_e.transpose(0, 1).contiguous())
        # topk 加权合并 (torch, body 内)
        _idx = _ti.unsqueeze(-1).expand(-1, -1, D)
        y = (torch.gather(y_esd, 1, _idx) * w.unsqueeze(-1)).sum(1)
    torch.npu.synchronize()
    print("[info] MoE launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        logits_r = x @ router_w
        tv_r, ti_r = torch.topk(logits_r, TOPK, dim=-1)
        w_r = F.softmax(tv_r, dim=-1)
        u_r = (x @ w1f).view(S, E, FFN)
        g_r = (x @ w2f).view(S, E, FFN)
        act_r = F.silu(u_r) * g_r
        y_e_r = torch.bmm(act_r.transpose(0, 1), w3).transpose(0, 1)   # [S,E,D]
        idx_r = ti_r.unsqueeze(-1).expand(-1, -1, D)
        ref = (torch.gather(y_e_r, 1, idx_r) * w_r.unsqueeze(-1)).sum(1)
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
