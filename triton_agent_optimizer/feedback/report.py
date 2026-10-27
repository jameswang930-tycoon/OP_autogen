#!/usr/bin/env python3
"""统一优化报告 — 把 final_summary + trajectory + 诊断画像 + 工业级明细 + 最终 diff 汇总成一份 REPORT.md。

产出: <kernel_dir>/final_output/REPORT.md
内容:
  1. 头部: 算子 / 总轮次 / 有效优化 / best_speedup / 生成时间
  2. 最终结果表: msprof 端到端基线→最终 / Event 我们最优 / 工业级最优 / vs 工业级
  3. 工业级各 mode 明细 (time_us / 是否真正执行)
  4. 最终诊断画像 (最后轮 bottleneck / 引擎利用率 / roofline) — "还有没有优化空间"
  5. 最终改动 (final_diff.patch 行数统计 + diff 全文引用)
  6. 成功策略清单 (从 trajectory 提取, 与 strategy_summary 同判定)
  7. 产物清单 (图 / rounds.csv / final_diff.patch / 策略 md / 最终 kernel)

用法 (库, scheduler 优化结束时自动调):
  from feedback.report import generate
  generate(kernel_dir)

命令行 (手动重跑):
  python3 feedback/report.py outputs/matmul
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _fnum(v, nd=1):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _is_successful(h: dict) -> bool:
    """成功优化 = 采纳(KEEP) + 有效(OK) + 非采集失败轮 + 严格超越上一轮 (与 strategy_summary 同判定)."""
    if h.get("decision") != "KEEP":
        return False
    if h.get("result") not in ("OK", None):
        return False
    if "采集失败" in (h.get("strategy") or ""):
        return False
    sp = h.get("speedup") or 0.0
    prev = h.get("prev_speedup") or 0.0
    return sp > prev + 1e-9


def _latest_diagnosis(kernel_dir: Path) -> dict:
    """找最新一轮的 diagnosis.json (输出画像: bottleneck/引擎利用率/roofline), 无则 {}."""
    cands = sorted(kernel_dir.glob("*/round*/06_diagnosis/diagnosis.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return {}
    try:
        return json.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _industrial_modes(op: str) -> list:
    """工业级各 mode 明细 (time_us/actual_mode), 无 json 的 mode 不列."""
    out = []
    for _m in ("eager", "compile", "cann-fused", "fa"):
        _p = _PROJECT_DIR / "bench_910b3" / "outputs" / f"industrial_{op}_{_m}_tflops.json"
        if not _p.exists():
            continue
        try:
            _d = json.loads(_p.read_text(encoding="utf-8"))
            out.append({"mode": _m, "time_us": _d.get("time_us"),
                        "actual_mode": _d.get("actual_mode", _m),
                        "rep": _d.get("rep"), "n_buf": _d.get("n_buf")})
        except Exception:
            continue
    return out


def generate(kernel_dir: Path, output_dir: Path | None = None) -> Path:
    traj_file = kernel_dir / "optimization_trajectory.json"
    if not traj_file.exists():
        raise FileNotFoundError(f"Not found: {traj_file}")
    traj = json.loads(traj_file.read_text(encoding="utf-8"))
    history = traj.get("history", [])
    state = traj.get("state", {})
    op = kernel_dir.name
    out_dir = output_dir or (kernel_dir / "final_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # final_summary 若存在 (scheduler 写) 优先; 否则从 state 兜底
    fs_p = out_dir / "final_summary.json"
    fs = {}
    if fs_p.exists():
        try:
            fs = json.loads(fs_p.read_text(encoding="utf-8"))
        except Exception:
            fs = {}

    def _g(key):
        return fs.get(key, state.get(key))

    n_keep = sum(1 for h in history if h.get("decision") == "KEEP")
    n_rev = sum(1 for h in history if h.get("decision") == "REVERT")
    n_fail = sum(1 for h in history if h.get("decision") == "FAIL")
    n_prom = _g("promote_rounds") or sum(1 for h in history
                                         if h.get("decision") == "KEEP" and h.get("result") == "OK"
                                         and (h.get("speedup") or 0) <= (h.get("prev_speedup") or 0) + 1e-9)
    n_succ = sum(1 for h in history if _is_successful(h))
    gen_at = datetime.now().isoformat(timespec="seconds")

    L = []
    L.append(f"# 优化报告 — {op}")
    L.append("")
    L.append(f"> 生成: {gen_at} | 总轮次: {len(history)} "
             f"(KEEP {n_keep} / REVERT {n_rev} / FAIL {n_fail} / promote {n_prom}) | "
             f"成功策略: {n_succ} | best_speedup: {_g('best_speedup')}x (R{_g('best_round')})")
    L.append("")

    # ═══ 1. 最终结果 ═══
    L.append("## 1. 最终结果")
    L.append("")
    L.append("| 指标 | 值 | 口径 |")
    L.append("|---|---|---|")
    _be2e = _g("baseline_e2e_ns")
    _ce2e = _g("final_e2e_ns")
    L.append(f"| 端到端 baseline → final | {_fnum(_be2e/1000 if _be2e else None)}us → "
             f"{_fnum(_ce2e/1000 if _ce2e else None)}us | msprof (Σ含框架) |")
    _bs = _g("baseline_ns")
    _cn = _g("final_ns")
    L.append(f"| 纯 kernel baseline → final | {_fnum(_bs/1000 if _bs else None)}us → "
             f"{_fnum(_cn/1000 if _cn else None)}us | msprof (Σ非aclnn) |")
    L.append(f"| 加速比 (端到端主口径) | {_g('current_speedup')}x (best {_g('best_speedup')}x) | msprof |")
    _our = _g("our_best_e2e_event_ns")
    if not _our:
        # 独立重跑 (无 final_summary) 时从 history 的 best_round 推导
        _br = _g("best_round")
        for _h in history:
            if _h.get("round") == _br and _h.get("e2e_event_ns"):
                _our = _h["e2e_event_ns"]
                break
    _ind = _g("industrial_time_us")
    L.append(f"| 我们最优 Event | {_fnum(_our/1000 if _our else None)}us | torch.npu.Event (工业级口径) |")
    L.append(f"| 工业级最优 Event | {_fnum(_ind)}us ({_g('industrial_baseline')}) | Event, 各 mode median 取最小 |")
    _vsr = _g("vs_industrial_ratio")
    _vss = _g("vs_industrial_speedup")
    if _vsr is not None:
        _verdict = "✓ 快于工业级" if _vss and _vss > 1 else "✗ 慢于工业级"
        L.append(f"| **vs 工业级** | 我们/工业级 = {_vsr}x → 工业级/我们 = {_vss}x {_verdict} | Event 同口径 |")
    else:
        L.append("| vs 工业级 | 缺数据 (无工业级基准) | — |")
    L.append("")

    # ═══ 2. 工业级各 mode 明细 ═══
    modes = _industrial_modes(op)
    if modes:
        L.append("## 2. 工业级各 mode 明细")
        L.append("")
        L.append("| mode | 端到端 median (us) | 实际执行 | 来源 json |")
        L.append("|---|---|---|---|")
        for m in modes:
            _act = "✅ 真正执行" if m["actual_mode"] == m["mode"] else f"⚠ 回退 {m['actual_mode']}"
            L.append(f"| {m['mode']} | {_fnum(m['time_us'])} | {_act} | industrial_{op}_{m['mode']}_tflops.json |")
        L.append("")
    else:
        L.append("## 2. 工业级各 mode 明细")
        L.append("")
        L.append("_(无 industrial json — 主循环 AUTO_RUN_IND_BENCH 会自动补跑; 或手动 bench_all)_")
        L.append("")

    # ═══ 3. 最终诊断画像 ═══
    dg = _latest_diagnosis(kernel_dir)
    L.append("## 3. 最终诊断画像 (最后一轮, 判断剩余优化空间)")
    L.append("")
    if dg:
        ks = dg.get("kernels") or []
        if ks:
            L.append("| kernel | 占比 | bottleneck | 算力利用 | 访存利用 | 读冗余 | L2 |")
            L.append("|---|---|---|---|---|---|---|")
            total_us = (dg.get("summary") or {}).get("total_ns")
            total_us = total_us / 1000 if total_us else None
            for k in ks:
                _dur = ((k.get("task") or {}).get("task_duration_us") or 0) * (k.get("launch_count") or 1)
                _pct = f"{_dur / total_us * 100:.1f}%" if (_dur and total_us) else "—"
                _d = k.get("deep") or {}
                _rl = _d.get("roofline") or {}
                _cu = _rl.get("compute_utilization")
                _mu = _rl.get("memory_utilization")
                _rd = _rl.get("traffic_redundancy_read")
                _l2 = _d.get("l2_hit_rate")
                L.append(f"| {k.get('kernel_name','?')} | {_pct} | {_rl.get('bottleneck_type','?')} | "
                         f"{_fnum(_cu,2)} | {_fnum(_mu,2)} | {_fnum(_rd,2)} | {_fnum(_l2,2)} |")
            L.append("")
            L.append("> 读冗余 >1.5 = 分块复用差 (Tier3/4 还有空间); bottleneck=memory/compute 高利用 = 已到头; "
                     "低利用 = 前层 (算法/融合) 还有空间.")
            L.append("")
        else:
            L.append("_(diagnosis 无 kernel 数据)_")
            L.append("")
    else:
        L.append("_(未找到任何轮次的 diagnosis.json — 没跑过采集)_")
        L.append("")

    # ═══ 4. 最终改动 (diff 统计) ═══
    dp = out_dir / "final_diff.patch"
    if dp.exists():
        _txt = dp.read_text(encoding="utf-8", errors="replace")
        _add = sum(1 for ln in _txt.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        _del = sum(1 for ln in _txt.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        L.append("## 4. 最终改动 (baseline → 最优 kernel)")
        L.append("")
        L.append(f"- 文件: `final_diff.patch` ({_add} 行新增 / {_del} 行删除)")
        L.append("- 全量 diff 见同目录 `final_diff.patch`; 成功策略明细见 `successful_strategies.md`")
        L.append("")
        L.append("```diff")
        _shown = [ln for ln in _txt.splitlines() if not ln.startswith("Index:")][:80]
        L.extend(_shown)
        if len(_txt.splitlines()) > 80:
            L.append("... (截断, 全文见 final_diff.patch)")
        L.append("```")
        L.append("")

    # ═══ 5. 成功策略 ═══
    succ = [h for h in history if _is_successful(h)]
    L.append("## 5. 成功策略 (KEEP + 有效 + 严格超越)")
    L.append("")
    if not succ:
        L.append("_(无成功采纳的优化轮次)_")
    for h in succ:
        sp = h.get("speedup")
        prev = h.get("prev_speedup")
        L.append(f"- **R{h.get('round')} T{h.get('tier')}**: {h.get('strategy','?')} "
                 f"→ {_fnum(prev)}x → {_fnum(sp)}x | 改动: {(h.get('change') or '')[:80]}")
    L.append("")

    # ═══ 6. 产物清单 ═══
    L.append("## 6. 产物清单 (final_output/)")
    L.append("")
    L.append("| 文件 | 内容 |")
    L.append("|---|---|")
    L.append("| `trajectory_chart.png` | 加速比轨迹图 (KEEP/REVERT/FAIL 标记 + PyTorch/工业级虚线) |")
    L.append("| `rounds.csv` | 每轮明细 (轮次/策略/加速比/Event 耗时/error, Excel 可读) |")
    L.append("| `final_diff.patch` | baseline → 最优 kernel 完整 diff |")
    L.append("| `final_summary.json` | 最终数值摘要 (机器可读) |")
    L.append("| `kernel_op.py` / `baseline_kernel.py` | 最优 kernel / 基线 kernel 副本 |")
    L.append("| `all_strategies.md` / `successful_strategies.md` | 全部轮次 / 成功策略复盘 |")
    L.append("| `REPORT.md` | 本报告 |")
    L.append("")

    out_p = out_dir / "REPORT.md"
    out_p.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] → {out_p} ({len(history)} 轮, best {_g('best_speedup')}x, 成功策略 {n_succ})")
    return out_p


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate(Path(sys.argv[1]))
    else:
        print("用法: python3 feedback/report.py <outputs/<op> 目录>")
