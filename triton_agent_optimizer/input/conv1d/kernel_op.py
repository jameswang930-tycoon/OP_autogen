#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Conv1d (1D 卷积, valid padding) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (NCHW 的 1D 版 [N, Cin, L]):
#    y[n,co,l] = Σ_{ci,k} x[n,ci, l+k] * w[co,ci,k] + b[co]
#    L_out = L - KL + 1
#  优化提示: ★Tier1 结构层 (必做): 当前是 vector 外积累加 (不上 cube),
#            改软件 im2col + tl.dot 走 cube (见 playbook_tier1 情况G/教学2, 1D 版 BLOCK_CRS=Cin*KL);
#            Tier3 分块 (BLOCK_CO/BLOCK_L); Tier2 与后续激活/池化融合
# ═══════════════════════════════════════════════════════════════════════════════
import os
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
# ═══════════════════════════════════════════════════════════════════════════════
N   = int(os.environ.get("C1_N", 1))       # batch
CIN = int(os.environ.get("C1_CIN", 8))     # 输入通道
L   = int(os.environ.get("C1_L", 256))     # 输入长度
COUT= int(os.environ.get("C1_COUT", 32))   # 输出通道
KL  = int(os.environ.get("C1_KL", 3))      # 卷积核长度
DTYPE = torch.float32
BLOCK_CO = 32                              # 输出通道分块 (tile 宽)
BLOCK_L  = 64                              # 输出长度分块 (tile 高)
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# Conv1d (基础版): 每个 program 算 [BLOCK_CO 输出通道 × BLOCK_L 输出长度],
# 循环 ci*k 向量外积累加 (★不上 cube — Tier1 im2col+tl.dot 是直接可做优化)
@triton.jit
def conv1d_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                  L, CIN, COUT, KL, LOUT,
                  BLOCK_CO: tl.constexpr, BLOCK_L: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_l = (LOUT + BLOCK_L - 1) // BLOCK_L
    lb = pid % total_l
    tmp = pid // total_l
    cob = tmp % (triton.cdiv(COUT, BLOCK_CO))
    n = tmp // (triton.cdiv(COUT, BLOCK_CO))

    offs_co = cob * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    co_mask = offs_co < COUT
    l_mask = offs_l < LOUT

    acc = tl.zeros((BLOCK_CO, BLOCK_L), dtype=tl.float32)
    for ci in range(CIN):
        for k in range(KL):
            # x[n, ci, l+k]: [BLOCK_L]  innermost(l) stride=1 连续
            xv = tl.load(x_ptr + n * CIN * L + ci * L + offs_l + k,
                         mask=l_mask, other=0.0)
            # w[co, ci, k]: [BLOCK_CO]
            wv = tl.load(w_ptr + offs_co * (CIN * KL) + ci * KL + k,
                         mask=co_mask, other=0.0)
            acc += wv[:, None] * xv[None, :]        # 向量外积 (不上 cube)

    bias = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0)
    acc += bias[:, None]
    y_ptrs = y_ptr + n * COUT * LOUT + offs_co[:, None] * LOUT + offs_l[None, :]
    tl.store(y_ptrs, acc, mask=co_mask[:, None] & l_mask[None, :])


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    LOUT = L - KL + 1
    x = (torch.randn(N, CIN, L, dtype=DTYPE, device=npu) * 0.1)
    w = (torch.randn(COUT, CIN, KL, dtype=DTYPE, device=npu) * 0.1)
    b = (torch.randn(COUT, dtype=DTYPE, device=npu) * 0.1)
    y = torch.empty(N, COUT, LOUT, dtype=DTYPE, device=npu)

    total_l = triton.cdiv(LOUT, BLOCK_L)
    grid = (N * triton.cdiv(COUT, BLOCK_CO) * total_l,)
    print(f"[info] Conv1d N={N} Cin={CIN} L={L} Cout={COUT} KL={KL} → LOUT={LOUT} "
          f"grid={grid[0]} block={BLOCK_CO}x{BLOCK_L}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        conv1d_kernel[grid](x, w, b, y, L, CIN, COUT, KL, LOUT,
                            BLOCK_CO=BLOCK_CO, BLOCK_L=BLOCK_L)
    torch.npu.synchronize()
    print("[info] conv1d launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        import torch.nn.functional as F
        y_ref = F.conv1d(x, w, b)
        abs_diff = (y - y_ref).abs().max().item()
        rel_diff = abs_diff / (y_ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-3 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
