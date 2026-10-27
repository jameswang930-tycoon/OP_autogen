#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — MaxPool2d (2D 最大池化, stride/padding) (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (NCHW):
#    y[n,c,oh,ow] = max_{kh,kw} x[n,c, oh*stride_h+kh-pad_h, ow*stride_w+kw-pad_w]
#    H_out = (H + 2*pad - kH) // stride + 1
#  算法: main 里 F.pad 预处理输入 (-inf 填充 padding 区域, 一次性, 测量窗口外)
#        → kernel 内无边界 mask, 寻址完全规则 — 规避 HIVM root alloc 对复杂 mask
#          gather load 的分析失败 (实测编译报 "hivm.hir.load op unsupported")
#  优化空间: Tier4 访存连续化 (ow 方向 stride=1 连续突发, 已保持);
#            Tier5 max 链用 tl.maximum 合并; 窗口内共用地址 (kh 方向) 可优化加载次数
# ═══════════════════════════════════════════════════════════════════════════════
import os
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
# ═══════════════════════════════════════════════════════════════════════════════
N   = int(os.environ.get("MP_N", 1))     # batch
C   = int(os.environ.get("MP_C", 8))     # 通道数
H   = int(os.environ.get("MP_H", 64))    # 输入高
W   = int(os.environ.get("MP_W", 64))    # 输入宽
KH  = int(os.environ.get("MP_KH", 3))    # 窗口高
KW  = int(os.environ.get("MP_KW", 3))    # 窗口宽
SH  = int(os.environ.get("MP_SH", 2))    # stride 高
SW  = int(os.environ.get("MP_SW", 2))    # stride 宽
PAD = int(os.environ.get("MP_PAD", 1))   # padding (同高宽)
DTYPE = torch.float32
BLOCK_OW = 64                            # 输出宽分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# MaxPool2d: 每个 program 处理一个 (n,c,oh) 行的 BLOCK_OW 个输出宽,
# 输入已 pad (-inf), 无边界 mask, 寻址规则: ih = oh*SH+kh, iw = ow*SW+kw
@triton.jit
def maxpool2d_kernel(x_ptr, y_ptr,
                     HP, WP, OH, OW,
                     KH: tl.constexpr, KW: tl.constexpr,
                     SH: tl.constexpr, SW: tl.constexpr,
                     BLOCK_OW: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % total_ow
    tmp = pid // total_ow
    oh = tmp % OH
    nc = tmp // OH                    # (n,c) 由 grid 拆: grid = N*C*OH*total_ow
    n = nc // C
    c = nc % C

    offs_ow = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = offs_ow < OW
    acc = tl.full((BLOCK_OW,), float("-inf"), dtype=tl.float32)

    base = x_ptr + n * C * HP * WP + c * HP * WP
    for kh in range(KH):
        row = base + (oh * SH + kh) * WP
        for kw in range(KW):
            iw = offs_ow * SW + kw
            v = tl.load(row + iw, mask=ow_mask, other=float("-inf"))
            acc = tl.maximum(acc, v)

    y_ptrs = y_ptr + n * C * OH * OW + c * OH * OW + oh * OW + offs_ow
    tl.store(y_ptrs, acc, mask=ow_mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    import torch.nn.functional as F
    npu = torch.device("npu")
    OH = (H + 2 * PAD - KH) // SH + 1
    OW = (W + 2 * PAD - KW) // SW + 1
    HP, WP = H + 2 * PAD, W + 2 * PAD
    x = (torch.randn(N, C, H, W, dtype=DTYPE, device=npu)) * 0.1
    # ★pad 预处理 (-inf 填充 padding 区域, 一次性, 测量窗口外) → kernel 无边界 mask
    xp = F.pad(x, (PAD, PAD, PAD, PAD), value=float("-inf")).contiguous()   # [N, C, HP, WP]
    y = torch.empty(N, C, OH, OW, dtype=DTYPE, device=npu)

    total_ow = triton.cdiv(OW, BLOCK_OW)
    grid = (N * C * OH * total_ow,)
    print(f"[info] MaxPool2d N={N} C={C} H={H} W={W}  K={KH}x{KW} S={SH}x{SW} PAD={PAD} "
          f"→ OH={OH} OW={OW} grid={grid[0]} block_ow={BLOCK_OW} (pad 预处理)")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        maxpool2d_kernel[grid](xp, y, HP, WP, OH, OW,
                               KH=KH, KW=KW, SH=SH, SW=SW, BLOCK_OW=BLOCK_OW)
    torch.npu.synchronize()
    print("[info] maxpool2d launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        y_ref = F.max_pool2d(x, (KH, KW), stride=(SH, SW), padding=PAD)
        abs_diff = (y - y_ref).abs().max().item()
        rel_diff = abs_diff / (y_ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-5 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
