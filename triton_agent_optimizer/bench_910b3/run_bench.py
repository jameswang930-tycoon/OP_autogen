#!/usr/bin/env python3
"""run_bench — 910B3 全套硬件基准实测 (多策略 × 取最大, 全链路 per-path).

科学原则:
  1. 每类 bench 跑多个变体 (尺寸×分块×精度扫描) — 任何单次测量是真值的下界
  2. 每个度量取**全变体最大值** = 该路径峰值的最佳估计
  3. 聚合峰值 (GB/s/TFLOPS) 走通用 msprof (与主优化循环同源);
     per-path 带宽 (l0a/l0b feed, gm_to_ub, mte1/mte2 ...) 走 msprof op + board.json

输出:
  results.json        每个变体 + 每度量最大值 (完整)
  hardware_peak.json  校准峰值 (integrate.py 读取 → 替换硬编码 1800/294.9)
  results.txt         可读表格

═══ 怎么运行 (910B3 服务器) ═══
  conda activate triton-npu
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd bench_910b3
  python3 run_bench.py                      # 全套 (~30-45min, 含 msprof op per-path)
  python3 run_bench.py --skip-op            # 快速: 只聚合峰值, 不跑 per-path (~15min)
  python3 run_bench.py --bench cube         # 只测 cube
  python3 run_bench.py --rounds 5 --warmup 2
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import measure_msprof, measure_msprof_op, flatten_per_path  # noqa: E402
from bench_config import BENCHES, variant_bytes_flops  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def _merge_max(peak: dict, m: dict):
    """把 m 的非 None 数值并入 peak, 取最大值."""
    for k, v in m.items():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        peak[k] = max(peak.get(k, -1.0), fv)


def main():
    p = argparse.ArgumentParser(description="910B3 硬件基准实测")
    p.add_argument("--rounds", type=int, default=int(os.environ.get("BENCH_MEASURE_ITERS", "30")),
                   help="测量 launch 次数 (默认 30, 设备时间确定性高, 够稳)")
    p.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP_ITERS", "10")),
                   help="热身 launch 次数 (默认 10, 过 JIT/冷cache)")
    p.add_argument("--bench", type=str, default=None, help="只测单个 bench")
    p.add_argument("--skip-op", action="store_true",
                   help="跳过 msprof op per-path (只测聚合峰值, 快)")
    args = p.parse_args()

    targets = [args.bench] if args.bench else list(BENCHES)
    results = {"measured_at": datetime.now().isoformat(),
               "warmup_launches": args.warmup, "measure_launches": args.rounds,
               "method": "一次 msprof 内循环 warmup+measure, 跳热身平均稳态",
               "skip_op": args.skip_op,
               "results": {}, "per_path": {}}
    peak = {}          # 每度量最大值 (跨所有变体)

    for name in targets:
        if name not in BENCHES:
            print(f"❌ 未知 bench: {name}")
            sys.exit(1)
        bench = BENCHES[name]
        btype, kname = bench["type"], bench["kernel_name"]
        print(f"\n══ bench: {name} — {bench['desc']} ({len(bench['variants'])} 变体) ══")
        variants_out = []

        for vi, v in enumerate(bench["variants"]):
            app = f"python3 {OUT_DIR / 'bench_kernels.py'} --bench {name} --variant {vi}"
            work = OUT_DIR / "out" / name / f"v{vi}"
            try:
                # 聚合: 一次 msprof 内循环 (30+100) → 稳态平均 → GB/s / TFLOPS
                avg_us, durations = measure_msprof(app, kname,
                                                   warmup=args.warmup, measure=args.rounds,
                                                   work_dir=work)
                bytes_total, flops = variant_bytes_flops(btype, v)   # ★静态算, 不二次跑 kernel
                seconds = avg_us / 1e6
                bw = (bytes_total / 1e9) / seconds if bytes_total else None
                tf = (flops / 1e12) / seconds if flops else None
                entry = {"variant": vi, "params": v,
                         "avg_us": round(avg_us, 1),
                         "durations_us": [round(d, 1) for d in durations],
                         "bytes_total": bytes_total,
                         "bw_gb_s": round(bw, 1) if bw else None,
                         "tflops": round(tf, 1) if tf else None}
                if bw: peak["bw_gb_s"] = max(peak.get("bw_gb_s", -1), bw)
                if tf: peak["tflops"] = max(peak.get("tflops", -1), tf)
                print(f"    v{vi} {json.dumps(v, ensure_ascii=False)}: {avg_us:.1f}us "
                      f"bw={bw:.1f} GB/s" + (f" tflops={tf:.1f}" if tf else ""))

                # per-path: msprof op (仅 mm/vec 有意义; --skip-op 可跳过) — 用 --single 单次 launch
                if not args.skip_op and btype in ("mm", "vec"):
                    board = measure_msprof_op(app + " --single", kname, work, 0)
                    pp = flatten_per_path(board)
                    results["per_path"].setdefault(name, []).append({"variant": vi, "pp": pp})
                    _merge_max(peak, pp)
                    print(f"      per-path: l0a={pp.get('l0a_read_gb_s')} "
                          f"l0b={pp.get('l0b_read_gb_s')} gm_to_ub={pp.get('gm_to_ub_gb_s')} "
                          f"mte1={pp.get('mte1_ratio')} cube={pp.get('cube_ratio')}")
            except Exception as e:
                print(f"    ❌ v{vi} 失败: {str(e)[:200]}")
                entry = {"variant": vi, "params": v, "error": str(e)[:200]}
            variants_out.append(entry)

        results["results"][name] = variants_out

    # ── 命名峰值 (读起来友好) ──
    named_peak = {
        "gm_bw_gb_s": peak.get("bw_gb_s"),            # 聚合 GM/vec 峰值 (跨类 max)
        "main_mem_read_gb_s": peak.get("main_mem_read_gb_s"),
        "main_mem_write_gb_s": peak.get("main_mem_write_gb_s"),
        "l2_read_gb_s": peak.get("l2_read_gb_s"),
        "gm_to_ub_gb_s": peak.get("gm_to_ub_gb_s"),
        "ub_to_gm_gb_s": peak.get("ub_to_gm_gb_s"),
        "l0a_feed_gb_s": peak.get("l0a_read_gb_s"),
        "l0b_feed_gb_s": peak.get("l0b_read_gb_s"),
        "l1_read_gb_s": peak.get("l1_read_gb_s"),
        "mte1_ratio_max": peak.get("mte1_ratio"),
        "mte2_ratio_max": peak.get("mte2_ratio"),
        "cube_ratio_max": peak.get("cube_ratio"),
        "vec_ratio_max": peak.get("vec_ratio"),
    }
    # 单拆 fp16/fp32 cube TFLOPS (从 cube 变体按 dtype 分)
    cube_variants = results["results"].get("cube", [])
    fp16 = [v["tflops"] for v in cube_variants
            if "tflops" in v and v["params"].get("dtype") != "float32"]
    fp32 = [v["tflops"] for v in cube_variants
            if "tflops" in v and v["params"].get("dtype") == "float32"]
    named_peak["cube_fp16_tflops"] = max(fp16) if fp16 else None
    named_peak["cube_fp32_tflops"] = max(fp32) if fp32 else None

    hardware = {"measured_at": datetime.now().isoformat(),
                "note": "910B3 实测峰值 (每度量取全变体 max); integrate.py 读取此文件校准 roofline",
                "peak": {k: (round(v, 2) if isinstance(v, (int, float)) else v)
                         for k, v in named_peak.items() if v is not None}}
    (OUT_DIR / "hardware_peak.json").write_text(
        json.dumps(hardware, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # results.txt 可读表格
    with open(OUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("══ 910B3 实测结果 (多变体取最大) ══\n")
        for n, vs in results["results"].items():
            f.write(f"\n[{n}] {BENCHES[n]['desc']}\n")
            for v in vs:
                if "error" in v:
                    f.write(f"  v{v['variant']} ❌ {v['error']}\n")
                    continue
                line = f"  v{v['variant']} {json.dumps(v['params'], ensure_ascii=False)}: {v['avg_us']:>9.1f}us"
                if v.get("bw_gb_s"): line += f"  {v['bw_gb_s']:>10.1f} GB/s"
                if v.get("tflops"): line += f"  {v['tflops']:>9.1f} TFLOPS"
                f.write(line + "\n")
        f.write("\n══ 峰值 (max, 供校准) ══\n")
        for k, v in hardware["peak"].items():
            f.write(f"  {k:24s} {v}\n")

    print(f"\n✅ 结果: {OUT_DIR / 'results.json'} + {OUT_DIR / 'results.txt'}")
    print(f"   🔧 校准峰值 → {OUT_DIR / 'hardware_peak.json'} (integrate.py 自动读)")
    print("   峰值摘要:")
    for k, v in hardware["peak"].items():
        print(f"      {k:24s} {v}")


if __name__ == "__main__":
    main()
