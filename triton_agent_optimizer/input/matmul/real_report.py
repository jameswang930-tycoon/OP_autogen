#!/usr/bin/env python3
"""真实数据诊断报告 — 读 diagnosis.json (4 源整合), 按优化策略显示瓶颈。

用法:
  python real_report.py [diagnosis.json] [--llm]
  默认找 input/matmul/e2e_run/06_diagnosis/diagnosis.json

输出:
  默认: summary(真实端到端/L2/核数) + 每通路真实带宽利用率 + 每op表 + 每Tier瓶颈信号
  --llm: 紧凑文本 (EXECUTION SUMMARY / TRANSFER PATHS / OPS / BOTTLENECKS)
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _f(v, nd=2):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_ns(ns):
    if ns is None:
        return "-"
    return f"{ns/1000:.3g} µs" if ns >= 1000 else f"{ns:.3g} ns"


def render(d, llm=False):
    ks = d.get("kernel_summary", {})
    ro = d.get("roofline", {})
    eu = d.get("engine_utilization", {})
    paths = d.get("transfer_paths", [])
    mi = d.get("memory_issues", {})
    comp = d.get("compute", {})
    bns = d.get("bottlenecks", {})

    if llm:
        print("=== KERNEL SUMMARY ===")
        print(f"kernel: {ks.get('kernel_name')}  total_ns: {_f(ks.get('total_ns'))}  "
              f"cores: {_f(ks.get('num_cores'))}  kernels: {ks.get('num_kernels')}  "
              f"api_us: {_f(ks.get('api_overhead_total_us'))}  l2: {_f(ks.get('l2_hit_rate'))}")
        print()
        print("=== ROOFLINE ===")
        print(f"bottleneck_type: {ro.get('bottleneck_type')}  "
              f"memory_util: {_f(ro.get('memory_utilization'),3)}  "
              f"compute_util: {_f(ro.get('compute_utilization'),3)}  "
              f"mem_bw: {_f(ro.get('achieved_memory_bw_gb_s'))}GB/s")
        print()
        print("=== TRANSFER PATHS ===")
        for p in paths:
            print(f"{p['path']}: real_bw={_f(p.get('real_bw_gb_s'))} GB/s")
        print()
        print("=== BOTTLENECKS ===")
        for k, v in bns.items():
            print(f"{k}: {v.get('hint')}")
        return

    print()
    print("┌─ 真实数据诊断报告 (msprof + msprof op, roofline 核心) ─┐")
    print()
    print("=== KERNEL SUMMARY ===")
    print(f"  kernel        = {ks.get('kernel_name')}")
    print(f"  total_ns      = {_fmt_ns(ks.get('total_ns'))}")
    print(f"  num_cores     = {_f(ks.get('num_cores'))}    freq={_f(ks.get('freq_mhz'))}MHz")
    print(f"  num_kernels   = {ks.get('num_kernels')}    api_overhead={_f(ks.get('api_overhead_total_us'))}us")
    print(f"  input_shapes  = {ks.get('input_shapes')}    output_shapes = {ks.get('output_shapes')}")
    print(f"  l2_hit_rate   = {_f(ks.get('l2_hit_rate'))}")
    print()

    print("=== ROOFLINE — 一针见血判瓶颈类型 ===")
    print(f"  实测内存带宽 = {_f(ro.get('achieved_memory_bw_gb_s'))} GB/s  (峰值 {ro.get('peak_memory_bw_gb_s')})  "
          f"利用率 {_f(ro.get('memory_utilization'),3)}")
    print(f"  实测算力     = {_f(ro.get('achieved_compute_tflops'))} TFLOPS (峰值 {ro.get('peak_compute_tflops')})  "
          f"利用率 {_f(ro.get('compute_utilization'),3)}")
    print(f"  → 瓶颈类型: {ro.get('bottleneck_type')}  "
          f"(memory_bound=访存, compute_bound=计算, latency_bound=并行不足, balanced=均衡)")
    print()

    print("=== ENGINE UTILIZATION — 哪个引擎忙 ===")
    for k, v in eu.items():
        print(f"  {k:8s} : {_f(v,3)}")
    print()

    print("=== TRANSFER PATHS — 每通路真实带宽 ===")
    for p in paths:
        print(f"  {p['path']:8s} : {_f(p.get('real_bw_gb_s'))} GB/s  ({p.get('source','')})")
    print()

    print("=== MEMORY ISSUES + COMPUTE ===")
    print(f"  L2命中={_f(mi.get('l2_hit_rate'))}  UB bank冲突={_f(mi.get('ub_bank_conflict_ratio'),4)}  "
          f"bankgroup冲突={_f(mi.get('ub_bankgroup_conflict_ratio'),4)}")
    print(f"  cube_fops={_f(comp.get('cube_fops'))}  vec_fops={_f(comp.get('vector_fops'))}  "
          f"cube_ratio={_f(comp.get('cube_ratio'),3)}  vec_ratio={_f(comp.get('vec_ratio'),3)}")
    print()

    print("=== BOTTLENECKS — 每优化策略的处方化提示 ===")
    for k, v in bns.items():
        print(f"  [{k}]")
        for fk, fv in v.items():
            if fk != "hint" and fv:
                print(f"      {fk}: {json.dumps(fv, ensure_ascii=False)}")
        print(f"      → {v.get('hint')}")
    print()
    print("└" + "─" * 52 + "┘")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    llm = "--llm" in args
    path = next((a for a in args if not a.startswith("--")), None)
    if not path:
        path = Path(__file__).resolve().parent / "e2e_run/06_diagnosis/diagnosis.json"
    p = Path(path)
    if not p.exists():
        sys.exit(f"❌ 找不到 {p}\n  先跑: bash analyzers/run_server_flow.sh 生成 diagnosis.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    render(d, llm)
