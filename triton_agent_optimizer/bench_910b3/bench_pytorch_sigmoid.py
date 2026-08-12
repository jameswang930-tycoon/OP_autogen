#!/usr/bin/env python3
"""PyTorch Sigmoid 基准线 — 与 input/sigmoid/kernel_op.py 同形状:
  Y = sigmoid(X)  (N=4M, fp32)
  输出 pytorch_sigmoid_tflops.json
"""
import argparse, json, os, sys
from datetime import datetime
from pathlib import Path
import torch
try:
    import torch_npu
except ImportError:
    pass
OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_sigmoid_tflops.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=int(os.environ.get("SIGMOID_N", str(4 * 1024 * 1024))))
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
    # ★轮换 buffer: 工作集 16MB << L2 192MB → 不清会全 L2 命中虚高
    bufs = [((torch.randn(args.n, dtype=dt, device=npu)) * 0.1,) for _ in range(args.n_buf)]

    def forward(i):
        return torch.sigmoid(bufs[i % len(bufs)][0])

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench_910b3.bench_common import measure_event
    m = measure_event(forward, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6
    flops = 2 * args.n  # exp + div
    tflops = flops / 1e12 / avg_s
    print(f"  torch sigmoid(N={args.n}, {args.dtype}): "
          f"median={m['median_us']:.1f}us -> {tflops:.1f} TFLOPS")
    OUT.write_text(json.dumps({"tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": args.n_buf,
        "N": args.n, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
