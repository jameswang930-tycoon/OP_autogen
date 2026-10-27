#!/usr/bin/env python3
"""PyTorch Multi-Head Flash Attention 基准线 — 与 input/flash_attention/kernel_op.py
**同形状同运算同精度同因果 mask**:

  Q/K/V/O[seq, nheads, dim];  每头:
    S = Q@K^T · scale  (scale = 1/sqrt(dim)),  causal mask (只关注 key≤query)
    P = softmax(S, dim=-1);  O = P@V
  (seq=2048, heads=8, dim=64, fp32, 因果)

  用途:
    - 跑 torch 端到端时间 → 与我们优化后的 Triton flash_attention 对比
      speedup vs PyTorch = torch_fa_time / triton_fa_time
    - 输出 pytorch_flash_attention_tflops.json

  FLOPs 口径: 每头 2 个 matmul (S=Q@K^T, O=P@V) → 2×2·seq·seq·dim·nheads = 4·seq²·dim·nheads
  ★用显式 matmul+mask+softmax 链 (非 F.scaled_dot_product_attention), 与 kernel_op 数学完全一致

用法 (910B3):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_910b3/bench_pytorch_flash_attention.py   # seq=2048 heads=8 dim=64 fp32
  FA_SEQ=1024 FA_HEADS=4 FA_DIM=64 python3 bench_910b3/bench_pytorch_flash_attention.py
  python3 bench_910b3/bench_pytorch_flash_attention.py --dtype float16
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

OUT = Path(__file__).resolve().parent / "outputs" / "pytorch_flash_attention_tflops.json"


def main():
    p = argparse.ArgumentParser(description="PyTorch 多头因果 Flash Attention 基准线")
    # ★默认对齐 input/flash_attention/kernel_op.py: seq=2048, heads=8, dim=64, fp16 (FA 工业级即 fp16)
    p.add_argument("--seq", type=int, default=int(os.environ.get("FA_SEQ", "2048")))
    p.add_argument("--heads", type=int, default=int(os.environ.get("FA_HEADS", "8")))
    p.add_argument("--dim", type=int, default=int(os.environ.get("FA_DIM", "64")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "float16"))
    p.add_argument("--warmup-ms", type=int, default=int(os.environ.get("BENCH_WARMUP_MS", "25")),
                   help="warmup 时间预算 (ms, do_bench 同款)")
    p.add_argument("--rep-ms", type=int, default=int(os.environ.get("BENCH_REP_MS", "100")),
                   help="测量时间预算 (ms, do_bench 同款)")
    p.add_argument("--n-buf", type=int, default=64,
                   help="轮换输入 buffer 组数 (破 L2 复用; FA 工作集小, 需更多组)")
    args = p.parse_args()

    if not torch.npu.is_available():
        print("[FATAL] torch.npu 不可用")
        sys.exit(1)
    torch.npu.set_device(0)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    seq, nh, dim = args.seq, args.heads, args.dim
    scale = 1.0 / (dim ** 0.5)
    npu = torch.device("npu")

    # ★轮换 buffer: FA 工作集 (q+k+v ≈ 6MB) << L2 192MB → 不清会全 L2 命中虚高 (64 组 ≈ 384MB)
    bufs = [((torch.randn(seq, nh, dim, dtype=dt, device=npu)) * 0.1,
             (torch.randn(seq, nh, dim, dtype=dt, device=npu)) * 0.1,
             (torch.randn(seq, nh, dim, dtype=dt, device=npu)) * 0.1) for _ in range(args.n_buf)]
    # 因果 mask: 上三角 (key n > query m 置 -inf), 与 kernel_op 一致 (seq 固定, 不轮换)
    causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=npu), diagonal=1)

    def forward(i):
        # ★布局 [seq,nh,dim] 的 batch 维是 seq, 直接 @ 会把 seq 当 batch → 必须先 permute 成 [nh,seq,dim]
        q, k, v = bufs[i % len(bufs)]
        q_h = q.permute(1, 0, 2)    # [nh, seq, dim]
        k_h = k.permute(1, 0, 2)
        v_h = v.permute(1, 0, 2)
        s = (q_h @ k_h.transpose(-2, -1)) * scale   # [nh, seq, seq]
        s = s.masked_fill(causal, float("-inf"))    # causal [seq,seq] 沿 nh 广播
        p = torch.softmax(s, dim=-1)
        o = p @ v_h                                 # [nh, seq, dim]
        return o.permute(1, 0, 2)                   # 回 [seq, nh, dim]

    # ★do_bench 同款: 时间预算自适应 + 多窗口 median + 轮换破 L2
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench_910b3.bench_common import measure_event
    m = measure_event(forward, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    avg_s = m["median_us"] / 1e6
    print(f"    {m['rep']} 个窗口 median: {m['median_us']:.1f}us/次 (min {m['min_us']}us)")

    flops = 4 * seq * seq * dim * nh                    # 每头 2 matmul (S 和 O)
    tflops = flops / 1e12 / avg_s
    print(f"\n  torch flash_attn({seq}x{nh}x{dim}, causal, {args.dtype}): "
          f"median={m['median_us']:.1f}us → {tflops:.1f} TFLOPS")

    OUT.write_text(json.dumps({
        "tflops": round(tflops, 2), "time_us": round(m["median_us"], 1),
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": args.n_buf,
        "seq": seq, "heads": nh, "dim": dim, "dtype": args.dtype, "flops": flops,
        "causal": True,
        "measured_at": datetime.now().isoformat(),
        "note": "Event 多窗口median+输入轮换破L2 (do_bench 同款)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
