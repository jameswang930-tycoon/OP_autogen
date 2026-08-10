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
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")))
    args = p.parse_args()
    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用"); sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    npu = torch.device("npu")
    x = (torch.randn(args.n, dtype=dt, device=npu)) * 0.1

    def forward():
        return torch.sigmoid(x)

    for _ in range(args.warmup):
        forward(); torch.npu.synchronize()
    st = torch.npu.Event(enable_timing=True); en = torch.npu.Event(enable_timing=True)
    st.record()
    for _ in range(args.measure):
        forward()
    en.record(); torch.npu.synchronize()
    avg_s = st.elapsed_time(en) / 1000.0 / args.measure
    flops = 2 * args.n  # exp + div
    tflops = flops / 1e12 / avg_s
    print(f"  torch sigmoid(N={args.n}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us -> {tflops:.1f} TFLOPS")
    OUT.write_text(json.dumps({"tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "N": args.n, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(), "measure": args.measure},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
