#!/usr/bin/env python3
"""PyTorch Conv2D+Bias+ReLU 基准线 — 与 input/conv_bias_relu/kernel_op.py
**同形状同运算同精度** (3 步: conv → +bias → relu):

  Yc = Conv2d(X, W, padding=PAD);  Yb = Yc + bias[K,1,1];  Y = relu(Yb)
  (N=1, C=8, H=W=64, K=32, R×S=3×3, pad=1, fp32)

  用途:
    - 跑 torch 端到端时间 → 与我们优化后的 Triton conv_bias_relu 对比
      speedup vs PyTorch = torch_time / triton_time
    - 输出 pytorch_conv_bias_relu_tflops.json

  FLOPs 口径: conv = 2·N·K·OH·OW·C·R·S;  +bias 与 relu 逐元素 ≈ 2·N·K·OH·OW

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_conv_bias_relu.py   # 默认尺寸 fp32
  CONV_N=2 CONV_C=16 CONV_K=64 python3 bench_910b3/bench_pytorch_conv_bias_relu.py
  python3 bench_910b3/bench_pytorch_conv_bias_relu.py --dtype float16
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

OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_conv_bias_relu_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch Conv2D+Bias+ReLU 基准线 (同 input/conv_bias_relu 场景)")
    # ★默认对齐 input/conv_bias_relu/kernel_op.py: N=1 C=8 H=W=64 K=32 R×S=3×3 pad=1 fp32
    p.add_argument("--n", type=int, default=int(os.environ.get("CONV_N", "1")))
    p.add_argument("--c", type=int, default=int(os.environ.get("CONV_C", "8")))
    p.add_argument("--h", type=int, default=int(os.environ.get("CONV_H", "64")))
    p.add_argument("--w", type=int, default=int(os.environ.get("CONV_W", "64")))
    p.add_argument("--k", type=int, default=int(os.environ.get("CONV_K", "32")))
    p.add_argument("--r", type=int, default=int(os.environ.get("CONV_R", "3")))
    p.add_argument("--s", type=int, default=int(os.environ.get("CONV_S", "3")))
    p.add_argument("--pad", type=int, default=int(os.environ.get("CONV_P", "1")))
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
    NB, C, H, W, K = args.n, args.c, args.h, args.w, args.k
    R, S, PAD = args.r, args.s, args.pad
    OH, OW = (H + 2 * PAD - R) // 1 + 1, (W + 2 * PAD - S) // 1 + 1
    npu = torch.device("npu")

    # ★轮换 buffer: 小 conv 工作集 << L2 192MB → 不清会全 L2 命中虚高
    bufs = [((torch.randn(NB, C, H, W, dtype=dt, device=npu)) * 0.1,
             (torch.randn(K, C, R, S, dtype=dt, device=npu)) * 0.1,
             (torch.randn(K, dtype=dt, device=npu)) * 0.1) for _ in range(args.n_buf)]

    import torch.nn.functional as F

    def forward(i):
        x, w, bias = bufs[i % len(bufs)]
        yc = F.conv2d(x, w, padding=PAD)
        yb = yc + bias.view(1, -1, 1, 1)
        return F.relu(yb)

    # ★do_bench 同款: 时间预算自适应 + 多窗口 median + 轮换破 L2
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench_910b3.bench_common import measure_event
    m = measure_event(forward, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6
    print(f"    {m['rep']} 个窗口 median: {m['median_us']:.1f}us/次 (min {m['min_us']}us)")

    n_el = NB * K * OH * OW
    flops = 2 * NB * K * OH * OW * C * R * S + 2 * n_el     # conv + (bias+relu 逐元素)
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch conv+bias+relu({NB}x{C}x{H}x{W} → {K}x{OH}x{OW}, {R}x{S} pad{PAD}, {args.dtype}): "
          f"median={m['median_us']:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": args.n_buf,
        "N": NB, "C": C, "H": H, "W": W, "K": K, "R": R, "S": S, "PAD": PAD,
        "OH": OH, "OW": OW, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
