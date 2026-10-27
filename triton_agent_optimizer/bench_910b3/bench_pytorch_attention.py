#!/usr/bin/env python3
"""PyTorch 自注意力+MLP 基准线 — 与 input/attention_mlp/kernel_op.py **同场景同运算同精度**:

  Q=X@Wq; K=X@Wk; V=X@Wv → S=Q@K^T·scale → P=softmax(S) → O=P@V
  → Y=GELU(O@W1+b1) → Z=Y@W2 → Out=Z+O   (seq=dim=2048, fp32)

  用途:
    - 跑 torch 端到端时间 → 与我们优化后的 Triton attention_mlp 对比
    - 输出 pytorch_attention_tflops.json (time_us/tflops, 供轨迹图/对比)

  真实 FLOPs (7 个 matmul):
    q,k,v = 3×2·seq·dim·dim;  S/O = 2×2·seq·seq·dim;  Y/Z = 2×2·seq·dim·dim
    = 10·seq·dim² + 4·seq²·dim   (seq=dim=2048 → 14·2048³ ≈ 120 TFLOPS 口径)

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_attention.py              # 2048² fp32, do_bench 同款: 多窗口 median + 轮换破 L2
  MATMUL_M=1024 MATMUL_N=1024 python3 bench_910b3/bench_pytorch_attention.py   # 改尺寸
  python3 bench_910b3/bench_pytorch_attention.py --dtype float16
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

OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_attention_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch 自注意力+MLP 基准线 (同 attention_mlp 场景)")
    # ★默认对齐 input/attention_mlp/kernel_op.py: seq=dim=2048, fp32
    p.add_argument("--seq", type=int, default=int(os.environ.get("MATMUL_M", os.environ.get("SEQ", "2048"))))
    p.add_argument("--dim", type=int, default=int(os.environ.get("MATMUL_N", os.environ.get("DIM", "2048"))))
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
    seq, dim = args.seq, args.dim
    scale = 1.0 / (dim ** 0.5)
    npu = torch.device("npu")

    # ★轮换 buffer: 工作集 (x+wq+wk+wv+w1+w2 ≈ 96MB) < L2 192MB → 不清会全 L2 命中虚高
    bufs = [((torch.rand(seq, dim, dtype=dt, device=npu) - 0.5) * 0.1,
             (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1,
             (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1,
             (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1,
             (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1,
             (torch.rand(dim, dtype=dt, device=npu) - 0.5) * 0.1,
             (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1) for _ in range(args.n_buf)]

    import torch.nn.functional as F

    def forward(i):
        x, wq, wk, wv, w1, b1, w2 = bufs[i % len(bufs)]
        q = x @ wq
        k = x @ wk
        v = x @ wv
        s = (q @ k.transpose(-2, -1)) * scale
        p = F.softmax(s, dim=-1)
        o = p @ v
        y = F.gelu(o @ w1 + b1, approximate="tanh")
        z = y @ w2
        return z + o

    # ★do_bench 同款: 时间预算自适应 + 多窗口 median + 轮换破 L2
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench_910b3.bench_common import measure_event
    m = measure_event(forward, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6
    print(f"    {m['rep']} 个窗口 median: {m['median_us']:.1f}us/次 (min {m['min_us']}us)")

    flops = 10 * seq * dim * dim + 4 * seq * seq * dim   # 真实 7 matmul FLOPs
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch attention+MLP({seq}x{dim}, {args.dtype}): "
          f"median={m['median_us']:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": args.n_buf,
        "seq": seq, "dim": dim, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
