#!/usr/bin/env python3
"""验收报告 — 提取每个算子的优化结果 vs 工业级最优, 算验收比值.

对 outputs/ 下每个算子提取 (来源优先 final_output/final_summary.json, 缺则回退
optimization_trajectory.json 的 state):
  1. 我们最优 Event 端到端耗时 (us)  — final_summary.our_best_e2e_event_ns (或
     trajectory state.best_e2e_event_ns / baseline_e2e_event_ns÷best_speedup 反推)
  2. 工业级最优端到端耗时 (us)        — final_summary.industrial_time_us
  3. 验收比值 = 工业级最优 ÷ 我们最优 (vs_industrial_speedup)
       ≥ 1.0  → 快于工业级 (融合/算法层胜利)   ✅ 优秀
       0.8~1.0 → 打平区间 (大算子/带宽型正常)   🟢 良好
       < 0.8  → 还有明显空间 (继续优化/查瓶颈)  🟡 有空间

终端表格 + outputs/acceptance_summary.json。

用法 (在仓库根 triton_agent_optimizer/ 下):
  python3 feedback/acceptance_report.py                 # 全部算子
  python3 feedback/acceptance_report.py --op matmul     # 只算一个
  python3 feedback/acceptance_report.py --md            # 额外写 outputs/acceptance_report.md
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))   # ★仓库相对路径导入 (feedback/ → 仓库根)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load(kernel_dir: Path) -> dict:
    """提取一个算子的验收数据: 优先 final_summary.json, 缺字段回退 trajectory state."""
    out = {"op": kernel_dir.name, "kernel_dir": str(kernel_dir)}
    fs_p = kernel_dir / "final_output" / "final_summary.json"
    traj_p = kernel_dir / "optimization_trajectory.json"
    fs = {}
    st = {}
    if fs_p.exists():
        try:
            fs = json.loads(fs_p.read_text(encoding="utf-8"))
        except Exception:
            fs = {}
    if traj_p.exists():
        try:
            st = json.loads(traj_p.read_text(encoding="utf-8")).get("state", {})
        except Exception:
            st = {}

    def _g(key):
        return fs.get(key, st.get(key))

    # 1) 我们最优 Event 端到端 (ns → us)
    our_ns = _g("our_best_e2e_event_ns") or _g("best_e2e_event_ns")
    if not our_ns and st.get("baseline_e2e_event_ns") and st.get("best_speedup"):
        our_ns = st["baseline_e2e_event_ns"] / st["best_speedup"]   # 反推
    out["our_event_us"] = round(our_ns / 1000.0, 2) if our_ns else None

    # 2) 工业级最优端到端 (us)
    ind_us = _g("industrial_time_us")
    out["industrial_us"] = round(ind_us, 2) if ind_us else None
    out["industrial_baseline"] = _g("industrial_baseline")

    # 3) 验收比值 = 工业级 ÷ 我们 (vs_industrial_speedup)
    if out["our_event_us"] and out["industrial_us"]:
        out["acceptance_x"] = round(out["industrial_us"] / out["our_event_us"], 3)
        out["our_over_industrial"] = round(out["our_event_us"] / out["industrial_us"], 3)
        if out["acceptance_x"] >= 1.0:
            out["verdict"] = "✅ 快于工业级"
        elif out["acceptance_x"] >= 0.8:
            out["verdict"] = "🟢 打平 (良好)"
        else:
            out["verdict"] = "🟡 有空间"
    else:
        out["acceptance_x"] = None
        out["verdict"] = "— 缺数据 (无工业级基准或无 Event 最优)"

    out["best_speedup"] = _g("best_speedup")
    out["total_rounds"] = _g("total_rounds")
    out["final_tier"] = _g("final_tier")
    return out


def main():
    p = argparse.ArgumentParser(description="验收报告: 我们 Event vs 工业级最优 ÷ 比值")
    p.add_argument("--op", type=str, default=None, help="只算指定算子 (缺省=全部)")
    p.add_argument("--md", action="store_true", help="额外写 outputs/acceptance_report.md")
    args = p.parse_args()

    out_root = _PROJECT_DIR / "outputs"
    if not out_root.exists():
        print(f"❌ 无 outputs/ 目录: {out_root} (先跑过 main.py 优化)")
        return 1

    dirs = [d for d in sorted(out_root.iterdir()) if d.is_dir()]
    if args.op:
        dirs = [d for d in dirs if d.name == args.op]
    rows = []
    for d in dirs:
        if not (d / "optimization_trajectory.json").exists() and \
           not (d / "final_output" / "final_summary.json").exists():
            continue
        rows.append(_load(d))
    if not rows:
        print("⚠ 没有找到任何算子的优化结果 (需 optimization_trajectory.json 或 final_summary.json)")
        return 1

    # ── 终端表格 ──
    print("═" * 104)
    print("  验收报告 — 我们最优 Event 端到端 vs 工业级最优端到端 (验收 = 工业级 ÷ 我们)")
    print("═" * 104)
    print(f"  {'算子':<18}{'我们Event(us)':>13}{'工业级(us)':>12}{'验收比值':>10}   {'判定':<14}"
          f"{'best加速比':>10}{'轮次':>6}")
    print("  " + "-" * 102)
    for r in rows:
        _o = f"{r['our_event_us']:.1f}" if r["our_event_us"] is not None else "—"
        _i = f"{r['industrial_us']:.1f}" if r["industrial_us"] is not None else "—"
        _a = f"{r['acceptance_x']:.2f}x" if r["acceptance_x"] is not None else "—"
        _b = f"{r['best_speedup']:.2f}x" if r["best_speedup"] else "—"
        _n = str(r["total_rounds"]) if r["total_rounds"] is not None else "—"
        print(f"  {r['op']:<18}{_o:>13}{_i:>12}{_a:>10}   {r['verdict']:<14}{_b:>10}{_n:>6}")
    ok = [r for r in rows if r.get("acceptance_x") is not None]
    if ok:
        _vals = [r["acceptance_x"] for r in ok]
        _avg = sum(_vals) / len(_vals)
        _mx = max(_vals); _mn = min(_vals)
        n_fast = sum(1 for v in _vals if v >= 1.0)
        n_good = sum(1 for v in _vals if 0.8 <= v < 1.0)
        n_slow = sum(1 for v in _vals if v < 0.8)
        print("  " + "-" * 102)
        print(f"  验收汇总 ({len(ok)} 个有数据): 平均 {_avg:.2f}x | 最快 {_mx:.2f}x | 最慢 {_mn:.2f}x")
        print(f"  ✅ 快于工业级: {n_fast} | 🟢 打平 0.8~1.0: {n_good} | 🟡 有空间 <0.8: {n_slow}")
        if n_slow:
            print(f"  ★有空间的算子: {[r['op'] for r in ok if r['acceptance_x'] < 0.8]}")
    print("═" * 104)

    # ── 写 json ──
    out_json = out_root / "acceptance_summary.json"
    out_json.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"),
         "method": "验收 = 工业级最优端到端(us) ÷ 我们最优Event端到端(us); ≥1.0=快于工业级, 0.8~1.0=打平, <0.8=有空间",
         "ops": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  明细 → {out_json}")

    # ── 可选 markdown 报告 ──
    if args.md:
        L = [f"# 验收报告 — {datetime.now().isoformat(timespec='seconds')}",
             "",
             "| 算子 | 我们Event(us) | 工业级(us) | 验收比值 | 判定 | best加速比 | 轮次 |",
             "|---|---|---|---|---|---|---|"]
        for r in rows:
            _o = f"{r['our_event_us']:.1f}" if r["our_event_us"] is not None else "—"
            _i = f"{r['industrial_us']:.1f}" if r["industrial_us"] is not None else "—"
            _a = f"{r['acceptance_x']:.2f}x" if r["acceptance_x"] is not None else "—"
            _b = f"{r['best_speedup']:.2f}x" if r["best_speedup"] else "—"
            _n = str(r["total_rounds"]) if r["total_rounds"] is not None else "—"
            L.append(f"| {r['op']} | {_o} | {_i} | {_a} | {r['verdict']} | {_b} | {_n} |")
        L.append("")
        L.append("> 验收 = 工业级最优端到端 ÷ 我们最优 Event 端到端 (两端同口径: Event 设备侧, 破 L2)。"
                 "≥1.0 = 快于工业级; 0.8~1.0 = 打平; <0.8 = 还有空间。")
        md_p = out_root / "acceptance_report.md"
        md_p.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"  Markdown → {md_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
