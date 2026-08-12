#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Multi-Head Flash Attention (online softmax 融合, 因果 mask) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (多头, 序列长 seq, 头数 nheads, 头维 dim, 因果):
#    S[h] = Q[h]@K[h]^T · scale      (scale = 1/sqrt(dim))
#    P[h] = softmax(S[h] + causal_mask, dim=-1)   (因果: 只关注 key≤query)
#    O[h] = P[h]@V[h]
#  单 kernel 融合全部: 每个 (头, 查询块) 一个 program, loop 所有 key 块, 边算 S 边更新 m/l/acc.
#  (Tier1 算法层: online softmax 避免物化 O(seq²×heads) 中间张量 — flash attention 核心)
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
#  形状: Q/K/V/O[seq, nheads, dim]
# ═══════════════════════════════════════════════════════════════════════════════
SEQ = int(os.environ.get("FA_SEQ", 2048))       # 序列长度
NHEADS = int(os.environ.get("FA_HEADS", 8))     # 头数
DIM = int(os.environ.get("FA_DIM", 64))         # 头维 (head_dim)
# ★fp16 输入 (与工业级 CANN FA 对齐 — 910B3 FA 天花板即 fp16; cube fp16 算力是 fp32 的 2×);
#   累加 acc/m_i/l_i 保持 fp32 (FA 标准, 精度不丢)
DTYPE = torch.float16
SCALE = 1.0 / (DIM ** 0.5)
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64          # 查询块/键块/头维块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# Multi-Head Flash Attention (因果): 每个 (头, 查询块) 一个 program, 边 loop key 块边 online softmax
# ★K 输入预转置为 [nheads, dim, seq] (=每头 K^T): 保证 inner-loop K 加载 innermost stride=1
#   (Q/V 用 [nheads, seq, dim], 原本就是连续加载; 只有 K 需要转置)
@triton.jit
def flash_attn_mha_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                          seq, nheads, dim, scale,
                          BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_m = (seq + BLOCK_M - 1) // BLOCK_M
    head = pid // num_m                            # 第几个头
    m_block = pid % num_m                          # 第几个查询块

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < seq
    k_mask = offs_k < dim

    # Q[h, m, k]: h*(seq*dim) + m*dim + k  (布局 [nheads, seq, dim])
    q_ptrs = q_ptr + head * (seq * dim) + offs_m[:, None] * dim + offs_k[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start in range(0, seq, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq
        # K_t[h, k, n]: h*(dim*seq) + k*seq + n  → innermost(n) stride=1 连续✓
        #   (K_t = 每头 K^T 预转置, 和 attention_mlp 的 attention_scores 同款修法)
        k_ptrs = k_ptr + head * (dim * seq) + offs_k[:, None] * seq + offs_n[None, :]
        kk = tl.load(k_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        s = tl.dot(q, kk) * scale                  # S = Q@K^T·scale [BLOCK_M, BLOCK_N]

        # ★因果 mask: key n 只允许 ≤ query m
        causal = offs_n[None, :] <= offs_m[:, None]
        s = tl.where(causal, s, float("-inf"))

        m_curr = tl.maximum(tl.max(s, axis=1), m_i)   # running max [BLOCK_M]
        # ★P 转 fp16: tl.dot 两个输入必须同 dtype — p 是 exp 结果(fp32), vv 是 fp16 输入,
        #   dot(fp32, fp16) 会编译失败 → kernel 起不来 → msprof 采不到 kernel 名.
        #   FA 标准做法: P 矩阵用 fp16 (精度损失可忽略), acc 累加保持 fp32.
        p = tl.exp(s - m_curr[:, None]).to(tl.float16)  # [BLOCK_M, BLOCK_N]
        alpha = tl.exp(m_i - m_curr)                  # 旧块衰减 [BLOCK_M]
        # ★l_i 求和用 fp32 (p 已是 fp16, 直接 sum 会 fp16 累加掉精度; FA 标准: 归一化分母保持 fp32)
        l_i = alpha * l_i + tl.sum(p.to(tl.float32), axis=1)  # running sum [BLOCK_M]

        # V[h, n, d]: h*(seq*dim) + n*dim + d  → 行跨度=1 (连续✓)
        v_ptrs = v_ptr + head * (seq * dim) + offs_n[:, None] * dim + offs_k[None, :]
        vv = tl.load(v_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, vv)    # O 累加 [BLOCK_M, BLOCK_K]
        m_i = m_curr

    o = acc / l_i[:, None]                            # 归一化
    # O[h, m, k]: h*(seq*dim) + m*dim + k
    o_ptrs = o_ptr + head * (seq * dim) + offs_m[:, None] * dim + offs_k[None, :]
    tl.store(o_ptrs, o, mask=m_mask[:, None] & k_mask[None, :])


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    seq, nh, dim = SEQ, NHEADS, DIM
    # ★Q/V 用 [nheads, seq, dim] (原始连续); K 预转置成 [nheads, dim, seq] (=每头 K^T)
    #   保证 kernel 内 K 加载 innermost stride=1 (DMA 连续突发)
    q_t = (torch.randn(seq, nh, dim, dtype=DTYPE, device=npu) * 0.1).permute(1, 0, 2).contiguous()
    k_t = (torch.randn(seq, nh, dim, dtype=DTYPE, device=npu) * 0.1).permute(1, 2, 0).contiguous()
    v_t = (torch.randn(seq, nh, dim, dtype=DTYPE, device=npu) * 0.1).permute(1, 0, 2).contiguous()
    o = torch.empty(nh, seq, dim, dtype=torch.float32, device=npu)  # ★输出保持 fp32 (acc 即 fp32, 精度更稳)

    grid = (triton.cdiv(seq, BLOCK_M) * nh,)
    print(f"[info] flash_attn_mha seq={seq} heads={nh} dim={dim} causal=1 "
          f"grid={grid[0]} block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        flash_attn_mha_kernel[grid](q_t, k_t, v_t, o, seq, nh, dim, SCALE,
                                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()
    print("[info] flash_attn_mha launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考, 逐头算因果注意力)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        # o=[nheads,seq,dim] → back to [seq,nheads,dim] for comparison
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=npu), diagonal=1)  # 因果
        # ★参考升 fp32 计算 (输入是 fp16, 参考用高精度避免 fp16 参考误差干扰判定; 我们输出 o 也是 fp32)
        q_ref = q_t.permute(1, 0, 2).float()   # [nh,seq,dim] → [seq,nh,dim]
        k_ref = k_t.permute(2, 0, 1).float()   # [nh,dim,seq] → [seq,nh,dim]
        v_ref = v_t.permute(1, 0, 2).float()   # [nh,seq,dim] → [seq,nh,dim]
        o_ref = torch.empty_like(q_ref)
        for h in range(nh):
            sh = (q_ref[:, h, :] @ k_ref[:, h, :].t()) * SCALE
            sh = sh.masked_fill(mask, float("-inf"))
            ph = torch.softmax(sh, dim=-1)
            o_ref[:, h, :] = ph @ v_ref[:, h, :]
        o_cmp = o.permute(1, 0, 2)     # [nheads,seq,dim] → [seq,nheads,dim]
        abs_diff = (o_cmp - o_ref).abs().max().item()
        rel_diff = abs_diff / (o_ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|O-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
