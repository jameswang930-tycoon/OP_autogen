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
    s = d.get("summary", {})
    paths = d.get("transfer_paths", [])
    ops = d.get("ops", [])
    bns = d.get("bottlenecks", {})

    if llm:
        print("=== EXECUTION SUMMARY ===")
        print(f"total_ns: {_f(s.get('total_ns'))}   num_cores: {_f(s.get('num_cores'))}   "
              f"l2_hit_rate: {_f(s.get('l2_hit_rate'))}")
        print(f"execution_mode: {s.get('execution_mode')}   kernel: {s.get('kernel_name')}")
        print(f"engine_utilization: {json.dumps(s.get('engine_utilization'), ensure_ascii=False)}")
        print()
        print("=== TRANSFER PATHS (真实带宽) ===")
        for p in paths:
            print(f"{p['path']}: ops={p['num_ops']} size_kb={_f(p.get('total_size_kb'))} "
                  f"real_bw={_f(p.get('real_bw_gb_s'))} GB/s  eff_bw={_f(p.get('effective_bw_gb_s'))}  "
                  f"util={_f(p.get('bw_utilization'),3) if p.get('bw_utilization') is not None else '未校准'}")
        print()
        print("=== OPS ===")
        for o in ops:
            print(f"op{o['op_id']} {o['op_type']}: path={o['transfer_path']} "
                  f"size={_f(o.get('size_kb'))}KB sim_dur={_f(o.get('duration_ns'))}ns "
                  f"real_dur={_f(o.get('real_duration_ns'))}ns real_bw={_f(o.get('real_bw_gb_s'))} "
                  f"deps={len(o.get('dependencies') or [])}")
        print()
        print("=== BOTTLENECKS (每Tier) ===")
        for k, v in bns.items():
            print(f"{k}: {v.get('hint')}")
        return

    print()
    print("┌─ 真实数据诊断报告 (4源整合: HIVM+simulator+真机op_summary+msprof op) ─┐")
    print()
    print("=== EXECUTION SUMMARY ===")
    print(f"  total_ns      = {_fmt_ns(s.get('total_ns'))}   (真机 op_summary Task Duration)")
    print(f"  num_cores     = {_f(s.get('num_cores'))}")
    print(f"  l2_hit_rate   = {_f(s.get('l2_hit_rate'))}")
    print(f"  execution_mode= {s.get('execution_mode')}    kernel={s.get('kernel_name')}")
    print(f"  engine_util   = {json.dumps(s.get('engine_utilization'), ensure_ascii=False)}  (真机 pipe ratios)")
    print()

    print("=== TRANSFER PATHS — 每通路真实带宽 (找瓶颈: 哪条饱和) ===")
    print(f"  {'路径':<9} {'op数':<4} {'总size':<9} {'真实带宽':<12} {'有效带宽':<11} {'利用率':<9} 说明")
    print("  " + "-" * 78)
    for p in paths:
        util = (f"{p['bw_utilization']:.0%}" if p.get('bw_utilization') is not None
                else "需校准peak")
        print(f"  {p['path']:<9} {p['num_ops']:<4} {_f(p.get('total_size_kb')):<9} "
              f"{_f(p.get('real_bw_gb_s'))+' GB/s':<12} {_f(p.get('effective_bw_gb_s')):<11} "
              f"{util:<9} {p.get('desc','')}")
    print()

    print("=== OPS — 结构 + 指令时序 + 真机指标 ===")
    print(f"  {'op':<3} {'类型':<12} {'通路':<10} {'size':<7} {'sim_dur':<12} "
          f"{'real_dur':<12} {'real_bw':<9} 依赖")
    print("  " + "-" * 88)
    for o in ops:
        deps = ",".join(f"op{d.get('from_op')}" for d in (o.get('dependencies') or [])) or "—"
        print(f"  op{o['op_id']:<1} {str(o['op_type'])[:12]:<12} {str(o['transfer_path'])[:10]:<10} "
              f"{_f(o.get('size_kb'))+'KB':<7} {_fmt_ns(o.get('duration_ns')):<12} "
              f"{_fmt_ns(o.get('real_duration_ns')):<12} {_f(o.get('real_bw_gb_s')):<9} {deps}")
    print()

    print("=== BOTTLENECKS — 每优化策略的信号 + 提示 ===")
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
