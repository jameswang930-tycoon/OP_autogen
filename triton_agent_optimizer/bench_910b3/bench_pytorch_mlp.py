#!/usr/bin/env python3
"""PyTorch MLP 基准线 — 用 torch 跑我们两层 MLP (X@W1+b1 → GELU → @W2) 的端到端时间/TFLOPS.

与 input/matmul/kernel_op.py (Triton 版 MLP) **同形状同运算**:
  Y = GELU(X@W1 + b1) @ W2,  X[M,K] W1[K,H] b1[H] W2[H,N]
  FLOPs = 2·M·K·H + 2·M·H·N  (两个 matmul 的真实算量)

用途:
  - 测 torch 跑原算子的耗时 → 与我们优化后的 Triton MLP 对比:
      speedup vs PyTorch = torch_mlp_time / triton_mlp_time
  - 输出 pytorch_mlp_tflops.json (供轨迹图 vs-PyTorch 参考)

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_mlp.py             # 2048³ fp32, warmup2 + 5 轮
  M=1024 K=1024 HIDDEN=1024 N=1024 python3 bench_910b3/bench_pytorch_mlp.py   # 改尺寸
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

OUT = Path(__file__).resolve().parent / "pytorch_mlp_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch MLP 基准线 (与 Triton kernel_op.py 同形状)")
    p.add_argument("--m", type=int, default=int(os.environ.get("M", "2048")))
    p.add_argument("--k", type=int, default=int(os.environ.get("K", "2048")))
    p.add_argument("--n", type=int, default=int(os.environ.get("N", "2048")))
    p.add_argument("--hidden", type=int, default=int(os.environ.get("HIDDEN", "2048")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float32"))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()

    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    M, K, N, H = args.m, args.k, args.n, args.hidden

    # 与 kernel_op.py 同款输入 (小值, 避免溢出)
    x  = (torch.rand(M, K, dtype=dt, device="npu") - 0.5) * 0.1
    w1 = (torch.rand(K, H, dtype=dt, device="npu") - 0.5) * 0.1
    b1 = (torch.rand(H, dtype=dt, device="npu") - 0.5) * 0.1
    w2 = (torch.rand(H, N, dtype=dt, device="npu") - 0.5) * 0.1

    import torch.nn.functional as F

    def forward():
        z = torch.matmul(x, w1)
        h = F.gelu(z + b1, approximate="tanh")
        return torch.matmul(h, w2)

    # warmup (JIT/图编译预热)
    for _ in range(args.warmup):
        y = forward()
        torch.npu.synchronize()

    # measure (torch.npu.Event 计时)
    times_s = []
    for _ in range(args.rounds):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        y = forward()
        e.record()
        torch.npu.synchronize()
        times_s.append(s.elapsed_time(e) / 1000.0)   # ms → s
        print(f"    run: {times_s[-1]*1e6:.1f}us")

    avg_s = sum(times_s) / len(times_s)
    flops = 2 * M * K * H + 2 * M * H * N            # 两个 matmul 真实 FLOPs
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch MLP({M}x{K}@{K}x{H}→GELU→{H}x{N}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "M": M, "K": K, "N": N, "H": H, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "rounds": args.rounds,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
