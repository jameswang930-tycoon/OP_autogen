#!/usr/bin/env python3
"""PyTorch RMSNorm 基准线 — 与 input/rms_norm/kernel_op.py **同形状同运算同精度**:

  Y[row, :] = X[row, :] / sqrt(mean(X[row,:]^2) + eps) * weight[:]   (M=N=2048, fp32)

  用途:
    - 跑 torch 端到端时间 → 与我们优化后的 Triton RMSNorm 对比
      speedup vs PyTorch = torch_rmsnorm_time / triton_rmsnorm_time
    - 输出 pytorch_rms_norm_tflops.json (time_us/tflops, 供 trajectory_chart / scheduler 读)

  FLOPs 口径: 每输出元素 ≈ 4 FLOP (x² 乘 + 均值加 + rstd 乘 + weight 乘) → 4·M·N
  (RMSNorm 是带宽瓶颈, TFLOPS 低属正常; 加速比只看时间比)

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_rms_norm.py          # M=N=2048 fp32, warmup3 + 30 次窗口
  RMS_M=1024 RMS_N=4096 python3 bench_910b3/bench_pytorch_rms_norm.py   # 改尺寸
  python3 bench_910b3/bench_pytorch_rms_norm.py --dtype float16
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

OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_rms_norm_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch RMSNorm 基准线 (同 input/rms_norm 场景)")
    # ★默认对齐 input/rms_norm/kernel_op.py: M=N=2048, fp32
    p.add_argument("--m", type=int, default=int(os.environ.get("RMS_M", "2048")))
    p.add_argument("--n", type=int, default=int(os.environ.get("RMS_N", "2048")))
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
    M, N = args.m, args.n
    EPS = 1e-6
    npu = torch.device("npu")

    # 与 kernel_op.py 同款输入
    x = (torch.randn(M, N, dtype=dt, device=npu)) * 0.1
    w = (torch.randn(N, dtype=dt, device=npu)) * 0.1

    def forward():
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w

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

    flops = 4 * M * N                                   # 每元素 ≈4 FLOP (x²/mean/rstd·x/·w)
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch RMSNorm({M}x{N}, {args.dtype}): avg={avg_s*1e6:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "M": M, "N": N, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "measure": args.measure,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
