#!/usr/bin/env python3
"""PyTorch Conv2D 基准线 — 与 input/conv2d/kernel_op.py **同形状同运算同精度**:

  Y[n,k,oh,ow] = Σ_{c,r,s} X[n,c,oh+r-PAD, ow+s-PAD] · W[k,c,r,s]   (stride=1, padding=PAD)
  (N=1, C=8, H=W=64, K=32, R×S=3×3, pad=1, fp32)

  用途:
    - 跑 torch 端到端时间 → 与我们优化后的 Triton conv2d 对比
      speedup vs PyTorch = torch_conv_time / triton_conv_time
    - 输出 pytorch_conv2d_tflops.json

  FLOPs 口径: 2·N·K·OH·OW·C·R·S  (直接卷积真实乘加)

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_conv2d.py          # 默认尺寸 fp32
  CONV_N=2 CONV_C=16 CONV_K=64 python3 bench_910b3/bench_pytorch_conv2d.py   # 改尺寸
  python3 bench_910b3/bench_pytorch_conv2d.py --dtype float16
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
try:
    import torch_npu
except ImportError:
    pass

OUT = Path(__file__).resolve().parent / "pytorch_conv2d_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch Conv2D 基准线 (同 input/conv2d 场景)")
    # ★默认对齐 input/conv2d/kernel_op.py: N=1 C=8 H=W=64 K=32 R×S=3×3 pad=1 fp32
    p.add_argument("--n", type=int, default=int(os.environ.get("CONV_N", "1")))
    p.add_argument("--c", type=int, default=int(os.environ.get("CONV_C", "8")))
    p.add_argument("--h", type=int, default=int(os.environ.get("CONV_H", "64")))
    p.add_argument("--w", type=int, default=int(os.environ.get("CONV_W", "64")))
    p.add_argument("--k", type=int, default=int(os.environ.get("CONV_K", "32")))
    p.add_argument("--r", type=int, default=int(os.environ.get("CONV_R", "3")))
    p.add_argument("--s", type=int, default=int(os.environ.get("CONV_S", "3")))
    p.add_argument("--pad", type=int, default=int(os.environ.get("CONV_P", "1")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float32"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")),
                   help="一次 Event 窗口内 forward 次数, ÷N 求单次平均 (默认 30)")
    args = p.parse_args()

    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    NB, C, H, W, K = args.n, args.c, args.h, args.w, args.k
    R, S, PAD = args.r, args.s, args.pad
    OH, OW = (H + 2 * PAD - R) // 1 + 1, (W + 2 * PAD - S) // 1 + 1
    npu = torch.device("npu")

    # 与 kernel_op.py 同款输入 (小值 ±0.1)
    x = (torch.randn(NB, C, H, W, dtype=dt, device=npu)) * 0.1
    w = (torch.randn(K, C, R, S, dtype=dt, device=npu)) * 0.1

    import torch.nn.functional as F

    def forward():
        return F.conv2d(x, w, padding=PAD)

    # warmup (JIT/图编译预热)
    for _ in range(args.warmup):
        y = forward()
        torch.npu.synchronize()

    # measure: 一次 Event 窗口内 forward measure 次, ÷measure = 单次平均
    st = torch.npu.Event(enable_timing=True)
    en = torch.npu.Event(enable_timing=True)
    st.record()
    for _ in range(args.measure):
        y = forward()
    en.record()
    torch.npu.synchronize()
    avg_s = st.elapsed_time(en) / 1000.0 / args.measure   # ms → s → ÷N
    print(f"    {args.measure} 次窗口平均: {avg_s*1e6:.1f}us/次")

    flops = 2 * NB * K * OH * OW * C * R * S            # 直接卷积真实 FLOPs
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch conv2d({NB}x{C}x{H}x{W} → {K}x{OH}x{OW}, {R}x{S} pad{PAD}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "N": NB, "C": C, "H": H, "W": W, "K": K, "R": R, "S": S, "PAD": PAD,
        "OH": OH, "OW": OW, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "measure": args.measure,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
