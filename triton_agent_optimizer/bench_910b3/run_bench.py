#!/usr/bin/env python3
"""run_bench — 自动跑所有 910B3 基准, 输出实测结果 results.json。

流程 (每个 bench):
  ① warmup 裸跑 (JIT 编译/冷cache 预热)
  ② rounds 次 msprof → 读 op_summary 目标 kernel Task Duration(us)
  ③ 取平均 → 用 bench 的 bytes/flops 算 GB/s / TFLOPS

输出:
  results.json  结构化实测结果
  results.txt   可读表格

═══ 怎么运行 (910B3 服务器) ═══
  conda activate triton-npu
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd bench_910b3
  python3 run_bench.py                  # 全测 (6 个 bench, warmup1 + 3轮msprof平均)
  python3 run_bench.py --bench mm       # 只测 cube 算力
  python3 run_bench.py --rounds 5 --warmup 2   # 调轮数
  尺寸 env 覆盖: BENCH_BW_N / BENCH_MM / BENCH_VEC_N (见 bench_kernels.py 顶部)

  PyTorch 基准线 (轨迹图用): python3 bench_pytorch.py
  轨迹图:                python3 ../feedback/trajectory_chart.py ../outputs/matmul
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import measure_kernel  # noqa: E402
from bench_kernels import BENCHES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="910B3 基准测量")
    p.add_argument("--rounds", type=int, default=int(os.environ.get("BENCH_ROUNDS", "3")))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP", "1")))
    p.add_argument("--bench", type=str, default=None, help="只跑单个 bench (默认全跑)")
    args = p.parse_args()

    targets = [args.bench] if args.bench else list(BENCHES)
    results = {"measured_at": datetime.now().isoformat(),
               "rounds": args.rounds, "warmup": args.warmup,
               "env": {"M/N/K": "512³(非bench), bench 用自身尺寸",
                       "bench_common.py 版本": "v1"},
               "results": {}}

    for name in targets:
        if name not in BENCHES:
            print(f"❌ 未知 bench: {name}")
            sys.exit(1)
        b = BENCHES[name]
        print(f"\n══ bench: {name} — {b['desc']} ══")
        app = f"python3 {OUT_DIR / 'bench_kernels.py'} --bench {name}"
        work = OUT_DIR / "out" / name
        try:
            avg_us, durations = measure_kernel(app, warmup=args.warmup,
                                               rounds=args.rounds, work_dir=work)
        except Exception as e:
            print(f"  ❌ {name} 失败: {str(e)[:200]}")
            results["results"][name] = {"error": str(e)[:200]}
            continue
        rb, wb = b["run"]()   # (read_bytes, write_bytes), 只拿尺寸, 不计时
        bytes_total = rb + wb
        seconds = avg_us / 1e6
        bw_gb_s = (bytes_total / 1e9) / seconds if bytes_total else None
        tflops = None
        if b.get("flops"):
            tflops = (b["flops"] / 1e12) / seconds
        print(f"  平均 {avg_us:.1f}us | bytes={bytes_total/1e6:.1f}MB "
              f"| bw={bw_gb_s:.1f} GB/s"
              + (f" | {tflops:.1f} TFLOPS" if tflops else ""))
        results["results"][name] = {
            "desc": b["desc"],
            "avg_us": round(avg_us, 1),
            "durations_us": [round(d, 1) for d in durations],
            "bytes_total": bytes_total,
            "bw_gb_s": round(bw_gb_s, 1) if bw_gb_s else None,
            "tflops": round(tflops, 1) if tflops else None,
        }

    # 写结果
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(OUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("══ 910B3 实测结果 ══\n")
        for n, r in results["results"].items():
            if "error" in r:
                f.write(f"  {n:10s} ❌ {r['error']}\n")
                continue
            line = f"  {n:10s} {r['avg_us']:>10.1f}us"
            if r.get("bw_gb_s"): line += f"  {r['bw_gb_s']:>10.1f} GB/s"
            if r.get("tflops"): line += f"  {r['tflops']:>10.1f} TFLOPS"
            f.write(line + "\n")
    print(f"\n✅ 结果: {OUT_DIR / 'results.json'} + {OUT_DIR / 'results.txt'}")


if __name__ == "__main__":
    main()
