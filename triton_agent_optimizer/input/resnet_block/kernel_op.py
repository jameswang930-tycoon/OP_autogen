#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — ResNet 残差块 (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链: Y = ReLU( BN(Conv2d( ReLU(BN(Conv2d(X))) )) + X )
#  算法: host im2col (F.unfold 一次性, 窗口外) + GEMM 走 cube; BN 推理逐元素; 残差 add
#  (与 input/conv2d 同款 unfold+GEMM 模式 — 规避 HIVM root alloc 对 gather 的分析失败)
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
N_B = int(os.environ.get("RB_N", 8))          # batch
C_IN = int(os.environ.get("RB_C", 64))        # 输入通道 (= 输出通道 K, 残差要求同形状)
H = int(os.environ.get("RB_H", 64))           # 高
W = int(os.environ.get("RB_W", 64))           # 宽
K_OUT = int(os.environ.get("RB_K", 64))       # 输出通道 (== C_IN)
R, S, PAD = 3, 3, 1                           # 3x3, padding=1 (与 bench resnet_block 一致)
OH = (H + 2 * PAD - R) // 1 + 1
OW = (W + 2 * PAD - S) // 1 + 1
CRS = C_IN * R * S
LOUT = OH * OW
EPS = 1e-5
DTYPE = torch.float32
BLOCK_K = 64                                  # 输出通道块 (≥K_OUT 的 2 幂)
BLOCK_OW = 64                                 # 空间块
BLOCK_CRS = 1024                              # ≥CRS(576) 的 2 幂 (512<576 会丢 64 项规约 → 数值必错)
BLOCK_EL = 1024                               # 逐元素分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# 标准 GEMM (host im2col): 每 program 算 (n, oh) 的 [K × OW] 块
@triton.jit
def conv2d_kernel(x_col_ptr, w_ptr, b_ptr, y_ptr,
                  N, K, LOUT, OW,
                  BLOCK_K: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_CRS: tl.constexpr,
                  CRS: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % total_ow
    tmp = pid // total_ow
    oh = tmp % (LOUT // OW)
    n = tmp // (LOUT // OW)
    offs_k = tl.arange(0, BLOCK_K)
    offs_crs = tl.arange(0, BLOCK_CRS)
    offs_l = oh * OW + owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    l_ok = offs_l < (oh + 1) * OW    # 行界: OW 非 BLOCK_OW 整数倍时不越行 (env 改尺寸防串行)
    patch = tl.load(x_col_ptr + n * CRS * LOUT + offs_crs[:, None] * LOUT + offs_l[None, :],
                    mask=(offs_crs[:, None] < CRS) & l_ok[None, :], other=0.0)
    wtile = tl.load(w_ptr + offs_k[:, None] * CRS + offs_crs[None, :],
                    mask=(offs_crs[None, :] < CRS) & (offs_k[:, None] < K),
                    other=0.0)
    acc = tl.zeros((BLOCK_K, BLOCK_OW), dtype=tl.float32)
    acc = tl.dot(wtile, patch, acc)
    bias = tl.load(b_ptr + offs_k, mask=offs_k < K, other=0.0)
    acc = acc + bias[:, None]
    y_ptrs = y_ptr + n * K * LOUT + offs_k[:, None] * LOUT + offs_l[None, :]
    tl.store(y_ptrs, acc, mask=(offs_k[:, None] < K) & l_ok[None, :])


# BatchNorm 推理 (逐元素, 按通道): y = (x - rm) / sqrt(rv+eps) * g + b

@triton.jit
def conv2d_kernel2(x_col_ptr, w_ptr, b_ptr, y_ptr,
                   N, K, LOUT, OW,
                   BLOCK_K: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_CRS: tl.constexpr,
                   CRS: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % total_ow
    tmp = pid // total_ow
    oh = tmp % (LOUT // OW)
    n = tmp // (LOUT // OW)
    offs_k = tl.arange(0, BLOCK_K)
    offs_crs = tl.arange(0, BLOCK_CRS)
    offs_l = oh * OW + owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    l_ok = offs_l < (oh + 1) * OW
    patch = tl.load(x_col_ptr + n * CRS * LOUT + offs_crs[:, None] * LOUT + offs_l[None, :],
                    mask=(offs_crs[:, None] < CRS) & l_ok[None, :], other=0.0)
    wtile = tl.load(w_ptr + offs_k[:, None] * CRS + offs_crs[None, :],
                    mask=(offs_crs[None, :] < CRS) & (offs_k[:, None] < K),
                    other=0.0)
    acc = tl.zeros((BLOCK_K, BLOCK_OW), dtype=tl.float32)
    acc = tl.dot(wtile, patch, acc)
    bias = tl.load(b_ptr + offs_k, mask=offs_k < K, other=0.0)
    acc = acc + bias[:, None]
    y_ptrs = y_ptr + n * K * LOUT + offs_k[:, None] * LOUT + offs_l[None, :]
    tl.store(y_ptrs, acc, mask=(offs_k[:, None] < K) & l_ok[None, :])


# BatchNorm 推理 (逐元素, 按通道): y = (x - rm) / sqrt(rv+eps) * g + b
@triton.jit
def bn_kernel(x_ptr, y_ptr, rm_ptr, rv_ptr, g_ptr, b_ptr,
              n_elements, K, HW,
              eps,
              BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    ch = (offs // HW) % K
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    rm = tl.load(rm_ptr + ch, mask=mask, other=0.0)
    rv = tl.load(rv_ptr + ch, mask=mask, other=0.0)
    g = tl.load(g_ptr + ch, mask=mask, other=0.0)
    b = tl.load(b_ptr + ch, mask=mask, other=0.0)
    y = (x - rm) / tl.sqrt(rv + eps) * g + b
    tl.store(y_ptr + offs, y, mask=mask)


# ReLU

@triton.jit
def bn_kernel2(x_ptr, y_ptr, rm_ptr, rv_ptr, g_ptr, b_ptr,
              n_elements, K, HW,
              eps,
              BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    ch = (offs // HW) % K
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    rm = tl.load(rm_ptr + ch, mask=mask, other=0.0)
    rv = tl.load(rv_ptr + ch, mask=mask, other=0.0)
    g = tl.load(g_ptr + ch, mask=mask, other=0.0)
    b = tl.load(b_ptr + ch, mask=mask, other=0.0)
    y = (x - rm) / tl.sqrt(rv + eps) * g + b
    tl.store(y_ptr + offs, y, mask=mask)


# ReLU
@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


# 残差 add: y = x + res

@triton.jit
def relu_kernel2(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


# 残差 add: y = x + res
@triton.jit
def add_kernel(x_ptr, res_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    r = tl.load(res_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x + r, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    import torch.nn.functional as F
    npu = torch.device("npu")
    K = K_OUT
    x = (torch.randn(N_B, C_IN, H, W, dtype=DTYPE, device=npu) * 0.1)
    w1 = (torch.randn(K, C_IN, R, S, dtype=DTYPE, device=npu) * 0.1)
    b1 = (torch.randn(K, dtype=DTYPE, device=npu) * 0.1)
    w2 = (torch.randn(K, K, R, S, dtype=DTYPE, device=npu) * 0.1)
    b2 = (torch.randn(K, dtype=DTYPE, device=npu) * 0.1)
    g1 = (torch.rand(K, dtype=DTYPE, device=npu) * 0.5 + 0.5)
    bn1 = (torch.randn(K, dtype=DTYPE, device=npu) * 0.1)
    g2 = (torch.rand(K, dtype=DTYPE, device=npu) * 0.5 + 0.5)
    bn2 = (torch.randn(K, dtype=DTYPE, device=npu) * 0.1)
    rm1 = (torch.rand(K, dtype=DTYPE, device=npu) * 0.5 + 0.5)
    rv1 = (torch.rand(K, dtype=DTYPE, device=npu) * 0.5 + 0.5)
    rm2 = (torch.rand(K, dtype=DTYPE, device=npu) * 0.5 + 0.5)
    rv2 = (torch.rand(K, dtype=DTYPE, device=npu) * 0.5 + 0.5)

    # ★host im2col (一次性, 窗口外)
    x_col = F.unfold(x, (R, S), padding=PAD).contiguous()          # [N, CRS, LOUT]
    w1_flat = w1.reshape(K, C_IN * R * S).contiguous()
    w2_flat = w2.reshape(K, K * R * S).contiguous()

    y1 = torch.empty(N_B, K, OH, OW, dtype=DTYPE, device=npu)
    a1 = torch.empty(N_B, K, OH, OW, dtype=DTYPE, device=npu)
    y2 = torch.empty(N_B, K, OH, OW, dtype=DTYPE, device=npu)
    a2 = torch.empty(N_B, K, OH, OW, dtype=DTYPE, device=npu)
    y = torch.empty(N_B, K, OH, OW, dtype=DTYPE, device=npu)
    n_el = N_B * K * OH * OW

    grid_c = (N_B * OH * triton.cdiv(OW, BLOCK_OW),)
    grid_el = (triton.cdiv(n_el, BLOCK_EL),)
    print(f"[info] ResNet block N={N_B} C={C_IN} HxW={H}x{W} K={K} "
          f"OHxOW={OH}x{OW} kernels=7 grid_c={grid_c[0]}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        conv2d_kernel[grid_c](x_col, w1_flat, b1, y1, N_B, K, LOUT, OW,
                              BLOCK_K=BLOCK_K, BLOCK_OW=BLOCK_OW, BLOCK_CRS=BLOCK_CRS,
                              CRS=C_IN * R * S)
        bn_kernel[grid_el](y1, a1, rm1, rv1, g1, bn1, n_el, K, OH * OW, EPS, BLOCK=BLOCK_EL)
        relu_kernel[grid_el](a1, y1, n_el, BLOCK=BLOCK_EL)
        # ★第二层 conv 的 im2col: 依赖 conv1+relu 的输出, 必须在循环内做 (传激活图本身会越界+语义错)
        y1_col = F.unfold(y1, (R, S), padding=PAD).contiguous()
        conv2d_kernel2[grid_c](y1_col, w2_flat, b2, y2, N_B, K, LOUT, OW,
                              BLOCK_K=BLOCK_K, BLOCK_OW=BLOCK_OW, BLOCK_CRS=BLOCK_CRS,
                              CRS=K * R * S)
        bn_kernel2[grid_el](y2, a2, rm2, rv2, g2, bn2, n_el, K, OH * OW, EPS, BLOCK=BLOCK_EL)
        add_kernel[grid_el](a2, x, y, n_el, BLOCK=BLOCK_EL)
        relu_kernel2[grid_el](y, y, n_el, BLOCK=BLOCK_EL)
    torch.npu.synchronize()
    print("[info] ResNet block launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        def _bn(t, rm, rv, g, b):
            return F.batch_norm(t, rm, rv, g, b, training=False, momentum=0.0, eps=EPS)
        ref = F.relu(_bn(F.conv2d(F.relu(_bn(F.conv2d(x, w1, b1, padding=PAD),
                                             rm1, rv1, g1, bn1)), w2, b2, padding=PAD),
                         rm2, rv2, g2, bn2) + x)
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
