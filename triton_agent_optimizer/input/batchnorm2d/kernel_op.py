#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — BatchNorm2d (推理模式) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (NCHW [N,C,H,W], 按通道归一化):
#    y[n,c,h,w] = (x[n,c,h,w] - running_mean[c]) / sqrt(running_var[c] + eps)
#                 * gamma[c] + beta[c]
#  优化提示: Tier1 除法→乘 rsqrt (y=(x-mean)*rsqrt(var+eps)*gamma+beta, 一次乘链);
#            Tier4 逐元素连续访存 (innermost w stride=1); Tier5 减少标量/索引计算
# ═══════════════════════════════════════════════════════════════════════════════
import os
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
# ═══════════════════════════════════════════════════════════════════════════════
N  = int(os.environ.get("BN_N", 1))      # batch
C  = int(os.environ.get("BN_C", 8))      # 通道数
H  = int(os.environ.get("BN_H", 64))     # 高
W  = int(os.environ.get("BN_W", 64))     # 宽
DTYPE = torch.float32
EPS = 1e-5
BLOCK_SIZE = 1024                        # 逐元素分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# BatchNorm2d 推理: 每个 program 处理 BLOCK_SIZE 个连续元素 (N*C*H*W 展平),
# 按通道索引 c = (offs // (H*W)) % C 取 mean/var/gamma/beta (算术推导索引, 非数据依赖 → 合法)
@triton.jit
def bn2d_kernel(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, y_ptr,
                n_elements, HW, C, eps,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = (offs // HW) % C                       # 通道索引 (算术推导)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.load(mean_ptr + c, mask=mask, other=0.0)
    v = tl.load(var_ptr + c, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + c, mask=mask, other=0.0)
    b = tl.load(beta_ptr + c, mask=mask, other=0.0)
    y = (x - m) / tl.sqrt(v + eps) * g + b     # 除法版 (Tier1 可改 rsqrt 乘链)
    tl.store(y_ptr + offs, y, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(N, C, H, W, dtype=DTYPE, device=npu) * 0.1)
    running_mean = torch.randn(C, dtype=DTYPE, device=npu) * 0.1
    running_var = torch.rand(C, dtype=DTYPE, device=npu) * 0.5 + 0.5
    gamma = torch.randn(C, dtype=DTYPE, device=npu) * 0.1 + 1.0
    beta = torch.randn(C, dtype=DTYPE, device=npu) * 0.1
    y = torch.empty_like(x)

    n_elements = N * C * H * W
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    print(f"[info] BatchNorm2d N={N} C={C} H={H} W={W}  n_elements={n_elements} "
          f"grid={grid[0]} block={BLOCK_SIZE}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        bn2d_kernel[grid](x, running_mean, running_var, gamma, beta, y,
                          n_elements, H * W, C, EPS, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()
    print("[info] bn2d launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        import torch.nn.functional as F
        y_ref = F.batch_norm(x, running_mean, running_var, gamma, beta,
                             training=False, momentum=0.0, eps=EPS)
        abs_diff = (y - y_ref).abs().max().item()
        rel_diff = abs_diff / (y_ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-3 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
