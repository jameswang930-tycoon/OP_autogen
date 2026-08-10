#!/usr/bin/env python3
"""PyTorch LayerNorm 基准线 — 与 input/layernorm/kernel_op.py 同形状:
  Y = layer_norm(X, (N,), gamma, beta, eps)  (M=N=2048, fp32)
  输出 pytorch_layernorm_tflops.json
"""
import argparse, json, os, sys
from datetime import datetime
from pathlib import Path
import torch
try:
    import torch_npu
except ImportError:
    pass
OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_layernorm_tflops.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=int(os.environ.get("LAYERNORM_M", "2048")))
    p.add_argument("--n", type=int, default=int(os.environ.get("LAYERNORM_N", "2048")))
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
    x = (torch.randn(args.m, args.n, dtype=dt, device=npu)) * 0.1
    gamma = (torch.randn(args.n, dtype=dt, device=npu)) * 0.1
    beta = (torch.randn(args.n, dtype=dt, device=npu)) * 0.1

    def forward():
        return F.layer_norm(x, (args.n,), gamma, beta, 1e-5)

    for _ in range(args.warmup):
        forward(); torch.npu.synchronize()
    st = torch.npu.Event(enable_timing=True); en = torch.npu.Event(enable_timing=True)
    st.record()
    for _ in range(args.measure):
        forward()
    en.record(); torch.npu.synchronize()
    avg_s = st.elapsed_time(en) / 1000.0 / args.measure
    flops = 4 * args.m * args.n
    tflops = flops / 1e12 / avg_s
    print(f"  torch layernorm({args.m}x{args.n}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us -> {tflops:.1f} TFLOPS")
    OUT.write_text(json.dumps({"tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "M": args.m, "N": args.n, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(), "measure": args.measure},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
