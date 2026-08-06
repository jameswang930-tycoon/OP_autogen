#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Conv2D 直接卷积 (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: Y[n,k,oh,ow] = Σ_{c,r,s} X[n,c,oh+r-P, ow+s-P] · W[k,c,r,s]   (stride=1)
#  直接卷积 (非 im2col): 每个 program 算一个 (n, oh) 的 [BLOCK_K 输出通道 × BLOCK_OW 空间] 块,
#  loop c/r/s 做外积累加. (内存瓶颈: 每个输出像素要读 C·R·S 个输入 — Tier4/5 优化目标)
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
#  形状: X[N,C,H,W], W[K,C,R,S], Y[N,K,OH,OW];  stride=1, padding=P
# ═══════════════════════════════════════════════════════════════════════════════
N_B = int(os.environ.get("CONV_N", 1))        # batch
C_IN = int(os.environ.get("CONV_C", 8))       # 输入通道
H = int(os.environ.get("CONV_H", 64))         # 输入高
W = int(os.environ.get("CONV_W", 64))         # 输入宽
K_OUT = int(os.environ.get("CONV_K", 32))     # 输出通道
R = int(os.environ.get("CONV_R", 3))          # 卷积核高
S = int(os.environ.get("CONV_S", 3))          # 卷积核宽
PAD = int(os.environ.get("CONV_P", 1))        # padding
OH = (H + 2 * PAD - R) // 1 + 1               # 输出高
OW = (W + 2 * PAD - S) // 1 + 1               # 输出宽
DTYPE = torch.float32
BLOCK_K = 32                                  # 输出通道块 (≥K_OUT 的 2 幂)
BLOCK_OW = 64                                 # 空间块 (≥OW 的 2 幂)
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# 直接卷积: 每 program 算 (n, oh) 的 [BLOCK_K 通道 × BLOCK_OW 空间] 块, loop c/r/s 外积累加
@triton.jit
def conv2d_kernel(x_ptr, w_ptr, y_ptr,
                  N, H, W, K, OH, OW,
                  BLOCK_K: tl.constexpr, BLOCK_OW: tl.constexpr,
                  C: tl.constexpr, R: tl.constexpr, S: tl.constexpr, PAD: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % total_ow
    tmp = pid // total_ow
    oh = tmp % OH
    n = tmp // OH

    offs_k = tl.arange(0, BLOCK_K)
    offs_ow = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    k_mask = offs_k < K
    ow_mask = offs_ow < OW

    acc = tl.zeros((BLOCK_K, BLOCK_OW), dtype=tl.float32)
    for c in range(C):
        for r in range(R):
            for s in range(S):
                ih = oh + r - PAD
                iw = offs_ow + s - PAD
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & ow_mask
                xv = tl.load(x_ptr + n * C * H * W + c * H * W + ih * W + iw,
                             mask=valid, other=0.0)                      # [BLOCK_OW]
                wv = tl.load(w_ptr + offs_k * C * R * S + c * R * S + r * S + s,
                             mask=k_mask, other=0.0)                     # [BLOCK_K]
                acc += wv[:, None] * xv[None, :]                          # 外积累加

    y_ptrs = y_ptr + n * K * OH * OW + offs_k[:, None] * OH * OW + oh * OW + offs_ow[None, :]
    tl.store(y_ptrs, acc, mask=k_mask[:, None] & ow_mask[None, :])


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(N_B, C_IN, H, W, dtype=DTYPE, device=npu)) * 0.1
    w = (torch.randn(K_OUT, C_IN, R, S, dtype=DTYPE, device=npu)) * 0.1
    y = torch.empty(N_B, K_OUT, OH, OW, dtype=DTYPE, device=npu)

    grid = (N_B * OH * triton.cdiv(OW, BLOCK_OW),)
    print(f"[info] conv2d N={N_B} C={C_IN} HxW={H}x{W} K={K_OUT} R×S={R}x{S} pad={PAD} "
          f"OH×OW={OH}x{OW} grid={grid[0]} block_K={BLOCK_K} block_ow={BLOCK_OW}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        conv2d_kernel[grid](x, w, y, N_B, H, W, K_OUT, OH, OW,
                            BLOCK_K=BLOCK_K, BLOCK_OW=BLOCK_OW,
                            C=C_IN, R=R, S=S, PAD=PAD)
    torch.npu.synchronize()
    print("[info] conv2d launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch conv2d 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        import torch.nn.functional as F
        ref = F.conv2d(x, w, padding=PAD)
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
