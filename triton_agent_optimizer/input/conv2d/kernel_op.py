#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Conv2D (软件 im2col implicit GEMM) (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: Y[n,k,oh,ow] = Σ_{c,r,s} X[n,c,oh+r-P, ow+s-P] · W[k,c,r,s]   (stride=1)
#  算法: 软件 im2col 重写成 GEMM — M=输出空间(n*OH*OW), N=输出通道 K, K_dim=C*R*S
#        每 program 一次 tl.dot: [K, CRS] @ [CRS, OW] → [K, OW]  (走 cube 引擎)
#        (中上水平基线: 正常工程师的第一版就该走 cube, 而非向量外积)
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
CRS = C_IN * R * S                            # GEMM K 维 = 滤波 tap × 输入通道
DTYPE = torch.float32
BLOCK_K = 32                                  # 输出通道块 (≥K_OUT 的 2 幂)
BLOCK_OW = 64                                 # 空间块 (≥OW 的 2 幂)
BLOCK_CRS = 128                               # ≥CRS(72) 的 2 幂 (tl.arange 要求, 多出 tap mask 置 0)
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# 软件 im2col implicit GEMM: 每 program 算 (n, oh) 的 [K × OW] 块,
# patch[crs, ow] = X[n, c, ih, iw]  (padding/越界 → 0), 一次 tl.dot 走 cube
@triton.jit
def conv2d_kernel(x_ptr, w_ptr, y_ptr,
                  N, H, W, K, OH, OW,
                  BLOCK_K: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_CRS: tl.constexpr,
                  C: tl.constexpr, R: tl.constexpr, S: tl.constexpr, PAD: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % total_ow
    tmp = pid // total_ow
    oh = tmp % OH
    n = tmp // OH

    offs_k = tl.arange(0, BLOCK_K)
    offs_ow = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    offs_crs = tl.arange(0, BLOCK_CRS)          # ≥C*R*S 的 2 幂 (72 → 128)
    c = offs_crs // (R * S)
    r = (offs_crs % (R * S)) // S
    s = offs_crs % S
    ih = oh + r - PAD                            # [CRS]
    iw = offs_ow[None, :] + s[:, None] - PAD     # [CRS, OW]

    # 软件 im2col: patch[crs, ow] = X[n, c, ih, iw]; padding/越界 → 0
    valid = (offs_crs[:, None] < C * R * S) & (ih[:, None] >= 0) & (ih[:, None] < H) \
            & (iw >= 0) & (iw < W)
    patch = tl.load(x_ptr + n * C * H * W + c[:, None] * H * W + ih[:, None] * W + iw,
                    mask=valid, other=0.0)        # [CRS, OW]  innermost(ow) stride=1 连续

    # W 拍平 [K, C*R*S]: wtile[k, crs] = W[k, c, r, s]
    wtile = tl.load(w_ptr + offs_k[:, None] * (C * R * S) + offs_crs[None, :],
                    mask=(offs_crs[None, :] < C * R * S) & (offs_k[:, None] < K),
                    other=0.0)                    # [K, CRS]

    acc = tl.zeros((BLOCK_K, BLOCK_OW), dtype=tl.float32)
    acc = tl.dot(wtile, patch, acc)               # [K,CRS]@[CRS,OW]→[K,OW]
    y_ptrs = y_ptr + n * K * OH * OW + offs_k[:, None] * OH * OW + oh * OW + offs_ow[None, :]
    tl.store(y_ptrs, acc, mask=(offs_k[:, None] < K) & (offs_ow[None, :] < OW))


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
          f"OH×OW={OH}x{OW} grid={grid[0]} block_K={BLOCK_K} block_ow={BLOCK_OW} "
          f"block_crs={BLOCK_CRS} (im2col implicit GEMM)")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        conv2d_kernel[grid](x, w, y, N_B, H, W, K_OUT, OH, OW,
                            BLOCK_K=BLOCK_K, BLOCK_OW=BLOCK_OW, BLOCK_CRS=BLOCK_CRS,
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
