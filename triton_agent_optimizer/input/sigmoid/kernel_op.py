#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Sigmoid (纯逐元素) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: Y = 1 / (1 + exp(-X))   (逐元素)
#  单 kernel, 纯逐元素 (无归约/无 matmul)
#
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
#  形状: X[N] → Y[N] (1D 扁平; 也可当 2D flatten)
# ═══════════════════════════════════════════════════════════════════════════════
N = int(os.environ.get("SIGMOID_N", 4 * 1024 * 1024))   # 元素数 (4M, 够大才有带宽压力)
DTYPE = torch.float32
BLOCK_SIZE = 512                            # ★偏小 (留 Tier3 调大空间)
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# Sigmoid: 逐元素. ★用 1/(1+exp(-x)) 两步 (留 Tier5 用 tl.sigmoid 原生指令空间)
@triton.jit
def sigmoid_kernel(x_ptr, y_ptr,
                   n_elements,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # ★两步算 (非原生): 1.0 / (1.0 + tl.exp(-x)) → 可换 tl.sigmoid(x) 省指令 (Tier5)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offs, y, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(N, dtype=DTYPE, device=npu)) * 0.1
    y = torch.empty(N, dtype=DTYPE, device=npu)

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    print(f"[info] sigmoid N={N} dtype={DTYPE} grid={grid[0]} block={BLOCK_SIZE}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        sigmoid_kernel[grid](x, y, N, BLOCK=BLOCK_SIZE)
    torch.npu.synchronize()
    print("[info] sigmoid launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        ref = torch.sigmoid(x)
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-4 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
