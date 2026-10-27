#!/usr/bin/env python3
"""PyTorch MLP 基准线 — 用 torch 跑我们两层 MLP (X@W1+b1 → GELU → @W2) 的端到端时间/TFLOPS.

与 input/matmul/kernel_op.py (Triton 版 MLP) **同形状同运算**:
  Y = GELU(X@W1 + b1) @ W2,  X[M,K] W1[K,H] b1[H] W2[H,N]
  FLOPs = 2·M·K·H + 2·M·H·N  (两个 matmul 的真实算量)

用途:
  - 测 torch 跑原算子的耗时 → 与我们优化后的 Triton MLP 对比:
      speedup vs PyTorch = torch_mlp_time / triton_mlp_time
  - 输出 pytorch_mlp_tflops.json (供轨迹图 vs-PyTorch 参考)
  - ★env 名与 kernel_op.py 一致 (MATMUL_M/N/K, MLP_HIDDEN), 改尺寸两边同步

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_mlp.py                          # 2048³ fp32, do_bench 同款: 多窗口 median + 轮换破 L2
  MATMUL_M=1024 MATMUL_K=1024 MLP_HIDDEN=1024 MATMUL_N=1024 python3 bench_910b3/bench_pytorch_mlp.py
  python3 bench_910b3/bench_pytorch_mlp.py --dtype float16
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

OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_mlp_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch MLP 基准线 (与 Triton kernel_op.py 同形状)")
    p.add_argument("--m", type=int, default=int(os.environ.get("MATMUL_M", os.environ.get("M", "2048"))))
    p.add_argument("--k", type=int, default=int(os.environ.get("MATMUL_K", os.environ.get("K", "2048"))))
    p.add_argument("--n", type=int, default=int(os.environ.get("MATMUL_N", os.environ.get("N", "2048"))))
    p.add_argument("--hidden", type=int, default=int(os.environ.get("MLP_HIDDEN", os.environ.get("HIDDEN", "2048"))))
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
    M, K, N, H = args.m, args.k, args.n, args.hidden

    # ★轮换 buffer: 连续 forward 同一批张量 → 后 N 次全 L2 命中 (工作集 ~96MB < L2 192MB) → 数字虚高
    bufs = [((torch.rand(M, K, dtype=dt, device="npu") - 0.5) * 0.1,
             (torch.rand(K, H, dtype=dt, device="npu") - 0.5) * 0.1,
             (torch.rand(H, dtype=dt, device="npu") - 0.5) * 0.1,
             (torch.rand(H, N, dtype=dt, device="npu") - 0.5) * 0.1) for _ in range(args.n_buf)]

    import torch.nn.functional as F

    def forward(i):
        x, w1, b1, w2 = bufs[i % len(bufs)]
        z = torch.matmul(x, w1)
        h = F.gelu(z + b1, approximate="tanh")
        return torch.matmul(h, w2)

    # ★do_bench 同款: 时间预算自适应 + 多窗口 median + 轮换破 L2
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench_910b3.bench_common import measure_event
    m = measure_event(forward, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6
    print(f"    {m['rep']} 个窗口 median: {m['median_us']:.1f}us/次 (min {m['min_us']}us)")
    flops = 2 * M * K * H + 2 * M * H * N            # 两个 matmul 真实 FLOPs
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch MLP({M}x{K}@{K}x{H}→GELU→{H}x{N}, {args.dtype}): "
          f"median={m['median_us']:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": args.n_buf,
        "M": M, "K": K, "N": N, "H": H, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
