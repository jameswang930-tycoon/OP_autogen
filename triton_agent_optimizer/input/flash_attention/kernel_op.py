#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Flash Attention 单头 (online softmax 融合版) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (单头, 序列长 seq, 头维 dim):
#    S = Q@K^T · scale          (scale = 1/sqrt(dim))
#    P = softmax(S, dim=-1)     (★online softmax: 逐 key 块更新 running max/sum, 不物化全 S)
#    O = P@V
#  一个 kernel 融合全部: 每个 query 块 loop 所有 key 块, 边算 S 边更新 m/l/acc.
#  (Tier1 算法层: online softmax 避免 O(seq²) 中间张量写 GM — flash attention 的核心)
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
#  形状: Q/K/V[seq, dim], O[seq, dim]
# ═══════════════════════════════════════════════════════════════════════════════
SEQ = int(os.environ.get("FA_SEQ", 2048))     # 序列长度
DIM = int(os.environ.get("FA_DIM", 64))       # 头维 (head_dim)
DTYPE = torch.float32
SCALE = 1.0 / (DIM ** 0.5)
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # 查询块/键块/头维块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# Flash Attention 前向 (单头): 每个查询块一个 program, 边 loop key 块边 online softmax
@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                      seq, dim, scale,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < seq
    k_mask = offs_k < dim

    # Q 块固定加载一次: Q[m, k]  [BLOCK_M, BLOCK_K]
    q_ptrs = q_ptr + offs_m[:, None] * dim + offs_k[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)   # O 累加 [BLOCK_M, 头维]
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                 # running sum(exp)

    for start in range(0, seq, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq
        # K^T[k, n] = K[n, k]  (指针直接转置, 同 attention_scores 的写法)
        k_ptrs = k_ptr + offs_n[None, :] * dim + offs_k[:, None]
        kk = tl.load(k_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        s = tl.dot(q, kk) * scale                       # S = Q@K^T·scale [BLOCK_M, BLOCK_N]

        m_curr = tl.maximum(tl.max(s, axis=1), m_i)     # running max [BLOCK_M]
        p = tl.exp(s - m_curr[:, None])                 # [BLOCK_M, BLOCK_N]
        alpha = tl.exp(m_i - m_curr)                    # 旧块衰减因子 [BLOCK_M]
        l_i = alpha * l_i + tl.sum(p, axis=1)           # running sum [BLOCK_M]

        v_ptrs = v_ptr + offs_n[:, None] * dim + offs_k[None, :]
        vv = tl.load(v_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, vv)      # O 累加 [BLOCK_M, BLOCK_K]
        m_i = m_curr

    o = acc / l_i[:, None]                              # 归一化
    o_ptrs = o_ptr + offs_m[:, None] * dim + offs_k[None, :]
    tl.store(o_ptrs, o, mask=m_mask[:, None] & k_mask[None, :])


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    seq, dim = SEQ, DIM
    q = (torch.randn(seq, dim, dtype=DTYPE, device=npu)) * 0.1
    k = (torch.randn(seq, dim, dtype=DTYPE, device=npu)) * 0.1
    v = (torch.randn(seq, dim, dtype=DTYPE, device=npu)) * 0.1
    o = torch.empty(seq, dim, dtype=DTYPE, device=npu)

    grid = (triton.cdiv(seq, BLOCK_M),)
    print(f"[info] flash_attn seq={seq} dim={dim} scale={SCALE} "
          f"grid={grid[0]} block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        flash_attn_kernel[grid](q, k, v, o, seq, dim, SCALE,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()
    print("[info] flash_attn launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        s = (q @ k.transpose(-2, -1)) * SCALE
        p = torch.softmax(s, dim=-1)
        ref = p @ v
        abs_diff = (o - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|O-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
