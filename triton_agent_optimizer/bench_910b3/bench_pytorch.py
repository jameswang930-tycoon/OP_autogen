#!/usr/bin/env python3
"""测 torch.matmul 的 PyTorch 基准线 → 写 pytorch_tflops.json。

轨迹图 (trajectory_chart.py) 用这个当"PyTorch eager 基准线"虚线;
scheduler (main.py) 启动时会读它存进 trajectory state。

═══ 怎么运行 (910B3 服务器) ═══
  conda activate triton-npu
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_pytorch.py                  # M=N=K=2048 fp32 (对齐优化算子), do_bench 同款: 多窗口 median + 轮换破 L2
  MATMUL_M=1024 python3 bench_pytorch.py           # 改尺寸
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # ★供 from bench_910b3.bench_common import

OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_tflops.json"


def main():
    p = argparse.ArgumentParser(description="torch.matmul PyTorch 基准线")
    # ★H5/E4: 默认对齐优化算子 (input/matmul/kernel_op.py = 2048³ fp32), 避免跨尺寸/跨精度比较失真
    p.add_argument("--m", type=int, default=int(os.environ.get("MATMUL_M", os.environ.get("M", "2048"))))
    p.add_argument("--n", type=int, default=int(os.environ.get("MATMUL_N", os.environ.get("N", "2048"))))
    p.add_argument("--k", type=int, default=int(os.environ.get("MATMUL_K", os.environ.get("K", "2048"))))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float32"))
    p.add_argument("--warmup-ms", type=int, default=int(os.environ.get("BENCH_WARMUP_MS", "25")),
                   help="warmup 时间预算 (ms, do_bench 同款)")
    p.add_argument("--rep-ms", type=int, default=int(os.environ.get("BENCH_REP_MS", "100")),
                   help="测量时间预算 (ms, do_bench 同款)")
    p.add_argument("--n-buf", type=int, default=32,
                   help="轮换输入 buffer 组数 (破 L2 复用; 910B3 L2=192MB)")
    args = p.parse_args()

    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32

    M, N, K = args.m, args.n, args.k
    n = args.n_buf
    # ★轮换 buffer: 连续 forward 同一批张量 → 后 N 次全 L2 命中 (工作集 16MB << L2 192MB) → 数字虚高
    bufs = [((torch.rand(M, K, dtype=dt, device="npu") - 0.5) * 0.1,
             (torch.rand(K, N, dtype=dt, device="npu") - 0.5) * 0.1) for _ in range(n)]

    def fwd(i):
        a, b = bufs[i % len(bufs)]
        return torch.matmul(a, b)

    # ★do_bench 同款: 时间预算自适应 + 多窗口 median + 轮换破 L2
    from bench_910b3.bench_common import measure_event
    m = measure_event(fwd, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6   # us → s
    print(f"    {m['rep']} 个窗口 median: {m['median_us']:.1f}us/次 (min {m['min_us']}us)")
    flops = 2 * M * N * K
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch.matmul({M}x{K}@{K}x{N}, {args.dtype}): "
          f"median={m['median_us']:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": n,
        "M": M, "N": N, "K": K, "dtype": args.dtype,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
