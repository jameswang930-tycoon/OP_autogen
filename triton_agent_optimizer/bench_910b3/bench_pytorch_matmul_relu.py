#!/usr/bin/env python3
"""PyTorch Matmul+ReLU 基准线 — 与 input/matmul_relu/kernel_op.py 同形状同运算:
  Z = X@W; Y = relu(Z)  (M=N=K=2048, fp32)
  输出 pytorch_matmul_relu_tflops.json (time_us/tflops)
"""
import argparse, json, os, sys
from datetime import datetime
from pathlib import Path
import torch
try:
    import torch_npu
except ImportError:
    pass
OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_matmul_relu_tflops.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=int(os.environ.get("MATMUL_M", "2048")))
    p.add_argument("--n", type=int, default=int(os.environ.get("MATMUL_N", "2048")))
    p.add_argument("--k", type=int, default=int(os.environ.get("MATMUL_K", "2048")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float32"))
    p.add_argument("--warmup-ms", type=int, default=int(os.environ.get("BENCH_WARMUP_MS", "25")),
                   help="warmup 时间预算 (ms, do_bench 同款)")
    p.add_argument("--rep-ms", type=int, default=int(os.environ.get("BENCH_REP_MS", "100")),
                   help="测量时间预算 (ms, do_bench 同款)")
    p.add_argument("--n-buf", type=int, default=32,
                   help="轮换输入 buffer 组数 (破 L2 复用; 910B3 L2=192MB)")
    args = p.parse_args()
    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用"); sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    npu = torch.device("npu")
    import torch.nn.functional as F
    # ★轮换 buffer: 工作集 32MB < L2 192MB → 不清会部分 L2 命中虚高
    bufs = [((torch.randn(args.m, args.k, dtype=dt, device=npu)) * 0.1,
             (torch.randn(args.k, args.n, dtype=dt, device=npu)) * 0.1) for _ in range(args.n_buf)]

    def forward(i):
        x, w = bufs[i % len(bufs)]
        return F.relu(torch.matmul(x, w))

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench_910b3.bench_common import measure_event
    m = measure_event(forward, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6
    flops = 2 * args.m * args.n * args.k
    tflops = flops / 1e12 / avg_s
    print(f"  torch matmul+relu({args.m}x{args.n}x{args.k}, {args.dtype}): "
          f"median={m['median_us']:.1f}us -> {tflops:.1f} TFLOPS")
    OUT.write_text(json.dumps({"tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": args.n_buf,
        "M": args.m, "N": args.n, "K": args.k, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
