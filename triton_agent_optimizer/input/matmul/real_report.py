#!/usr/bin/env python3
"""真实数据诊断报告 — 读 diagnosis.json (骨架 + 每kernel deep), 按优化顺序显示。

用法:
  python real_report.py [diagnosis.json] [--llm]
  默认找 input/matmul/e2e_run/06_diagnosis/diagnosis.json

新 schema (v4): summary + kernels[] (task 骨架 + deep msprof op + per-kernel roofline)
旧 schema 自动兜底 (kernel_summary/roofline/transfer_paths/...)
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


def _render_kernel(k, llm):
    """一个 kernel: task 骨架 + deep"""
    name = k.get("kernel_name")
    task = k.get("task", {})
    deep = k.get("deep")
    if llm:
        print(f"### KERNEL {name}  (launch={k.get('launch_count')}  filled_by={k.get('filled_by')})")
        print(f"  task: dur={_f(task.get('task_duration_us'))}us  block={_f(task.get('block_dim'))}  "
              f"type={task.get('task_type')}  shapes_in={task.get('input_shapes')} {task.get('input_dtypes')}")
        if task.get("pipes_us"):
            print(f"  pipes_us: " + ", ".join(f"{kk}={_f(vv)}" for kk, vv in task["pipes_us"].items()))
        if deep:
            ro = deep.get("roofline", {})
            print(f"  deep: roofline={ro.get('bottleneck_type')}  mem_util={_f(ro.get('memory_utilization'),3)}  "
                  f"comp_util={_f(ro.get('compute_utilization'),3)}  "
                  f"bw={_f(ro.get('achieved_memory_bw_gb_s'))}GB/s  freq={_f(deep.get('freq_mhz'))}MHz")
            bw = deep.get("bandwidth_gb_s", {})
            if bw:
                print("  bw: " + ", ".join(f"{kk}={_f(vv)}" for kk, vv in bw.items() if vv))
            print(f"  engine: " + ", ".join(f"{kk}={_f(vv)}" for kk, vv in deep.get("engine_utilization", {}).items()))
        else:
            print("  deep: (未填 — msprof only)")
        return
    print(f"  {name}  launch={k.get('launch_count')}  filled_by={k.get('filled_by')}")
    print(f"    task: dur={_f(task.get('task_duration_us'))}us  block={_f(task.get('block_dim'))}  "
          f"type={task.get('task_type')}  cores=... ")
    print(f"      in={task.get('input_shapes')} {task.get('input_dtypes')}  "
          f"out={task.get('output_shapes')} {task.get('output_dtypes')}")
    if task.get("pipes_us"):
        print("      pipes_us: " + ", ".join(f"{kk}={_f(vv)}" for kk, vv in task["pipes_us"].items()))
    for tr in task.get("transfers", []):
        if "bw_gb_s" in tr:
            print(f"      {tr['path']:18s} {_f(tr.get('bytes'))}B / {_f(tr.get('time_us'))}us → {_f(tr.get('bw_gb_s'))} GB/s")
        elif "tflops" in tr:
            print(f"      {tr['path']:18s} {_f(tr.get('macs'))} MAC / {_f(tr.get('time_us'))}us → {_f(tr.get('tflops'))} TFLOPS")
    if not deep:
        print("    deep: (未填 — msprof only)")
        return
    ro = deep.get("roofline", {})
    print(f"    → 瓶颈类型: {ro.get('bottleneck_type')}  "
          f"(mem={_f(ro.get('memory_utilization'),3)} comp={_f(ro.get('compute_utilization'),3)})")
    print(f"      实测带宽 {_f(ro.get('achieved_memory_bw_gb_s'))}GB/s / 峰值 {ro.get('peak_memory_bw_gb_s')}  "
          f"实测算力 {_f(ro.get('achieved_compute_tflops'))}TFLOPS / 峰值 {ro.get('peak_compute_tflops')}")
    print("    ENGINE: " + ", ".join(f"{kk}={_f(vv)}" for kk, vv in deep.get("engine_utilization", {}).items()))
    bw = deep.get("bandwidth_gb_s", {})
    for kk, vv in bw.items():
        if vv:
            print(f"      {kk:24s} {_f(vv)} GB/s")
    comp = deep.get("compute", {})
    print(f"    compute: cube_fops={_f(comp.get('cube_fops'))} vec_fops={_f(comp.get('vector_fops'))}  "
          f"l2={_f(deep.get('l2_hit_rate'))}")
    cfl = deep.get("conflict", {})
    if cfl:
        print("    conflict: " + ", ".join(f"{kk}={_f(vv,3)}" for kk, vv in cfl.items() if vv))


def render_v4(d, llm=False):
    s = d.get("summary", {})
    if llm:
        print("=== SUMMARY ===")
        print(f"num_kernels={s.get('num_kernels')}  filled={s.get('filled_kernels')}  "
              f"total_ns={_f(s.get('total_ns'))}  cores={_f(s.get('num_cores'))}  "
              f"api_us={_f(s.get('api_overhead_total_us'))}  l2={_f(s.get('l2_hit_rate'))}")
        for k in d.get("kernels", []):
            print()
            _render_kernel(k, llm=True)
        return
    print()
    print("┌─ 真实数据诊断报告 (msprof 骨架 + msprof op 每kernel deep) ─┐")
    print()
    print("=== SUMMARY ===")
    print(f"  num_kernels   = {s.get('num_kernels')}    filled = {s.get('filled_kernels')}")
    print(f"  total_ns      = {_fmt_ns(s.get('total_ns'))}    num_cores = {_f(s.get('num_cores'))}")
    print(f"  api_overhead  = {_f(s.get('api_overhead_total_us'))}us    l2_hit_rate = {_f(s.get('l2_hit_rate'))}")
    print()
    print("=== KERNELS ===")
    for k in d.get("kernels", []):
        _render_kernel(k, llm=False)
        print()
    print("=== API OVERHEAD (launch 开销) ===")
    for a in d.get("api_overhead", [])[:10]:
        print(f"  {a.get('api_name')}  {_f(a.get('total_us'))}us  x{a.get('count')}")
    print()
    print("=== MULTI KERNEL (类型分解) ===")
    for m in d.get("multi_kernel", [])[:10]:
        print(f"  {m.get('op_type')}  x{m.get('count')}  {_f(m.get('total_time_us'))}us  "
              f"ratio={_f(m.get('ratio'))}")
    print()
    print("└" + "─" * 52 + "┘")
    print()


def render_v3(d, llm=False):
    """旧 schema 兜底"""
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
        print("=== ROOFLINE ===")
        print(f"bottleneck_type: {ro.get('bottleneck_type')}  "
              f"memory_util: {_f(ro.get('memory_utilization'),3)}  "
              f"compute_util: {_f(ro.get('compute_utilization'),3)}  "
              f"mem_bw: {_f(ro.get('achieved_memory_bw_gb_s'))}GB/s")
        print("=== TRANSFER PATHS ===")
        for p in paths:
            print(f"{p['path']}: real_bw={_f(p.get('real_bw_gb_s'))} GB/s")
        print("=== BOTTLENECKS ===")
        for k, v in bns.items():
            print(f"{k}: {v.get('hint')}")
        return
    print()
    print("=== KERNEL SUMMARY ===")
    print(f"  kernel        = {ks.get('kernel_name')}")
    print(f"  total_ns      = {_fmt_ns(ks.get('total_ns'))}")
    print(f"  num_cores     = {_f(ks.get('num_cores'))}    freq={_f(ks.get('freq_mhz'))}MHz")
    print(f"  num_kernels   = {ks.get('num_kernels')}    api_overhead={_f(ks.get('api_overhead_total_us'))}us")
    print(f"  l2_hit_rate   = {_f(ks.get('l2_hit_rate'))}")
    print()
    print("=== ROOFLINE ===")
    print(f"  实测内存带宽 = {_f(ro.get('achieved_memory_bw_gb_s'))} GB/s  (峰值 {ro.get('peak_memory_bw_gb_s')})  "
          f"利用率 {_f(ro.get('memory_utilization'),3)}")
    print(f"  实测算力     = {_f(ro.get('achieved_compute_tflops'))} TFLOPS (峰值 {ro.get('peak_compute_tflops')})  "
          f"利用率 {_f(ro.get('compute_utilization'),3)}")
    print(f"  → 瓶颈类型: {ro.get('bottleneck_type')}")
    print()
    print("=== ENGINE UTILIZATION ===")
    for k, v in eu.items():
        print(f"  {k:8s} : {_f(v,3)}")
    print()
    print("=== TRANSFER PATHS ===")
    for p in paths:
        print(f"  {p['path']:8s} : {_f(p.get('real_bw_gb_s'))} GB/s")
    print()
    print("=== MEMORY ISSUES + COMPUTE ===")
    print(f"  L2命中={_f(mi.get('l2_hit_rate'))}  UB bank冲突={_f(mi.get('ub_bank_conflict_ratio'),4)}  "
          f"bankgroup冲突={_f(mi.get('ub_bankgroup_conflict_ratio'),4)}")
    print(f"  cube_fops={_f(comp.get('cube_fops'))}  vec_fops={_f(comp.get('vector_fops'))}  "
          f"cube_ratio={_f(comp.get('cube_ratio'),3)}  vec_ratio={_f(comp.get('vec_ratio'),3)}")
    print()
    print("=== BOTTLENECKS ===")
    for k, v in bns.items():
        print(f"  [{k}] → {v.get('hint')}")
    print()


def render(d, llm=False):
    if d.get("kernels") and d.get("summary"):
        render_v4(d, llm)
    else:
        render_v3(d, llm)


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
