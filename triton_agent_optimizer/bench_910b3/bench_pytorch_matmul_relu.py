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
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")))
    args = p.parse_args()
    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用"); sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    npu = torch.device("npu")
    import torch.nn.functional as F
    x = (torch.randn(args.m, args.k, dtype=dt, device=npu)) * 0.1
    w = (torch.randn(args.k, args.n, dtype=dt, device=npu)) * 0.1

    def forward():
        return F.relu(torch.matmul(x, w))

    for _ in range(args.warmup):
        forward(); torch.npu.synchronize()
    st = torch.npu.Event(enable_timing=True); en = torch.npu.Event(enable_timing=True)
    st.record()
    for _ in range(args.measure):
        forward()
    en.record(); torch.npu.synchronize()
    avg_s = st.elapsed_time(en) / 1000.0 / args.measure
    flops = 2 * args.m * args.n * args.k
    tflops = flops / 1e12 / avg_s
    print(f"  torch matmul+relu({args.m}x{args.n}x{args.k}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us -> {tflops:.1f} TFLOPS")
    OUT.write_text(json.dumps({"tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "M": args.m, "N": args.n, "K": args.k, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(), "measure": args.measure},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
