#!/usr/bin/env python3
"""测 torch.matmul 的 PyTorch 基准线 → 写 pytorch_tflops.json。

轨迹图 (trajectory_chart.py) 用这个当"PyTorch eager 基准线"虚线;
scheduler (main.py) 启动时会读它存进 trajectory state。

═══ 怎么运行 (910B3 服务器) ═══
  conda activate triton-npu
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_pytorch.py                  # M=N=K=512 fp16, warmup 2 + 5 轮取平均
  M=1024 python3 bench_pytorch.py           # 改尺寸
  python3 bench_pytorch.py --dtype float32  # 测 fp32
  python3 bench_pytorch.py --rounds 10      # 调轮数
  输出: pytorch_tflops.json (轨迹图/scheduler 读)

  ⚠ 顺序: 先跑这个生成 pytorch_tflops.json, 再跑 main.py (scheduler 读它), 再画轨迹图
  加速比 = 时间比 (baseline_time / current_time); 图上 TFLOPS 是同一比值的换算
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
try:
    import torch_npu
except ImportError:
    pass

OUT = Path(__file__).resolve().parent / "pytorch_tflops.json"


def main():
    p = argparse.ArgumentParser(description="torch.matmul PyTorch 基准线")
    p.add_argument("--m", type=int, default=int(os.environ.get("M", "512")))
    p.add_argument("--n", type=int, default=int(os.environ.get("N", "512")))
    p.add_argument("--k", type=int, default=int(os.environ.get("K", "512")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float16"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")),
                   help="一次 Event 窗口内 matmul 次数, ÷N 求单次平均 (默认 30)")
    args = p.parse_args()

    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32

    M, N, K = args.m, args.n, args.k
    a = torch.rand(M, K, dtype=dt, device="npu")
    b = torch.rand(K, N, dtype=dt, device="npu")

    # warmup
    for _ in range(args.warmup):
        c = torch.matmul(a, b)
        torch.npu.synchronize()

    # measure: 一次 Event 窗口内 matmul measure 次, ÷measure = 单次平均
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.measure):
        c = torch.matmul(a, b)
    end.record()
    torch.npu.synchronize()
    avg_s = start.elapsed_time(end) / 1000.0 / args.measure   # ms → s → ÷N
    print(f"    {args.measure} 次窗口平均: {avg_s*1e6:.1f}us/次")
    flops = 2 * M * N * K
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch.matmul({M}x{K}@{K}x{N}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "M": M, "N": N, "K": K, "dtype": args.dtype,
        "measured_at": datetime.now().isoformat(),
        "measure": args.measure,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
