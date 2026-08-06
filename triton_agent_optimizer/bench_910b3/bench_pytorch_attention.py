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
  python3 bench_910b3/bench_pytorch_attention.py              # 2048² fp32, warmup3+30次窗口
  SEQ=1024 DIM=1024 python3 bench_910b3/bench_pytorch_attention.py   # 改尺寸
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

OUT = Path(__file__).resolve().parent / "pytorch_attention_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch 自注意力+MLP 基准线 (同 attention_mlp 场景)")
    # ★默认对齐 input/attention_mlp/kernel_op.py: seq=dim=2048, fp32
    p.add_argument("--seq", type=int, default=int(os.environ.get("SEQ", "2048")))
    p.add_argument("--dim", type=int, default=int(os.environ.get("DIM", "2048")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float32"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")),
                   help="一次 Event 窗口内 forward 次数, ÷N 求单次平均 (默认 30)")
    args = p.parse_args()

    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    seq, dim = args.seq, args.dim
    scale = 1.0 / (dim ** 0.5)
    npu = torch.device("npu")

    # 与 kernel_op.py 同款输入 (小值 ±0.05, 避免 fp32 dot 值域溢出)
    x  = (torch.rand(seq, dim, dtype=dt, device=npu) - 0.5) * 0.1
    wq = (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1
    wk = (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1
    wv = (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1
    w1 = (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1
    b1 = (torch.rand(dim, dtype=dt, device=npu) - 0.5) * 0.1
    w2 = (torch.rand(dim, dim, dtype=dt, device=npu) - 0.5) * 0.1

    import torch.nn.functional as F

    def forward():
        q = x @ wq
        k = x @ wk
        v = x @ wv
        s = (q @ k.transpose(-2, -1)) * scale
        p = F.softmax(s, dim=-1)
        o = p @ v
        y = F.gelu(o @ w1 + b1, approximate="tanh")
        z = y @ w2
        return z + o

    # warmup (JIT/图编译预热)
    for _ in range(args.warmup):
        out = forward()
        torch.npu.synchronize()

    # measure: 一次 Event 窗口内 forward measure 次, ÷measure = 单次平均 (摊薄 Event 开销/抖动)
    st = torch.npu.Event(enable_timing=True)
    en = torch.npu.Event(enable_timing=True)
    st.record()
    for _ in range(args.measure):
        out = forward()
    en.record()
    torch.npu.synchronize()
    avg_s = st.elapsed_time(en) / 1000.0 / args.measure   # ms → s → ÷N
    print(f"    {args.measure} 次窗口平均: {avg_s*1e6:.1f}us/次")

    flops = 10 * seq * dim * dim + 4 * seq * seq * dim   # 真实 7 matmul FLOPs
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch attention+MLP({seq}x{dim}, {args.dtype}): "
          f"avg={avg_s*1e6:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(avg_s * 1e6, 1),
        "seq": seq, "dim": dim, "dtype": args.dtype, "flops": flops,
        "measured_at": datetime.now().isoformat(),
        "measure": args.measure,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
