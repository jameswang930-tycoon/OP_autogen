#!/usr/bin/env python3
"""策略摘要 — 从 optimization_trajectory.json 抽出每轮 strategy, 产出到 final_output/.

产两个文件 (都写到 <kernel_dir>/final_output/):
  1. all_strategies.md        — 全部轮次 (KEEP/REVERT/FAIL/promote/采集失败 都收, 标注 decision)
  2. successful_strategies.md — 仅"成功优化"轮次 (采纳且有效; 排除 promote/REVERT/FAIL/采集失败)

"成功优化" 判定 (与 scheduler strict-best 一致):
  decision == "KEEP"  AND  result == "OK"  AND  strategy != "采集失败跳过"
  AND speedup > prev_speedup  (严格超越上一轮; promote 轮 speedup==prev 被排除)

每轮记: round/tier + strategy(策略表述) + change(具体改动) + speedup + 瓶颈(若有).

用法 (库, scheduler 每轮自动调):
  from feedback.strategy_summary import generate
  generate(kernel_dir)        # → kernel_dir/final_output/{all,successful}_strategies.md

命令行 (手动重跑):
  python3 feedback/strategy_summary.py outputs/matmul
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _is_successful(h: dict) -> bool:
    """成功优化 = 采纳(KEEP) + 有效(OK) + 非采集失败轮 + 严格超越上一轮.
    promote 轮 (speedup==prev_speedup, 原样拷贝) 不算优化 → 排除."""
    if h.get("decision") != "KEEP":
        return False
    if h.get("result") not in ("OK", None):
        return False
    if "采集失败" in (h.get("strategy") or ""):
        return False
    sp = h.get("speedup") or 0.0
    prev = h.get("prev_speedup") or 0.0
    return sp > prev + 1e-9   # 严格超越 (promote 轮 sp==prev → 排除)


def _bottleneck(h: dict) -> str:
    """从 history 取瓶颈 (没存则空). hist 没直接存 bottleneck, 用 tier 名兜底."""
    return ""


def generate(kernel_dir: Path, output_dir: Path | None = None) -> tuple:
    """读 optimization_trajectory.json → 写 all/successful 策略 md. 返回 (all_path, succ_path)."""
    traj_file = kernel_dir / "optimization_trajectory.json"
    if not traj_file.exists():
        raise FileNotFoundError(f"Not found: {traj_file}")
    traj = json.loads(traj_file.read_text(encoding="utf-8"))
    history = traj.get("history", [])
    state = traj.get("state", {})
    op = kernel_dir.name

    out_dir = output_dir or (kernel_dir / "final_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 统计 ──
    n_keep = sum(1 for h in history if h.get("decision") == "KEEP")
    n_revert = sum(1 for h in history if h.get("decision") == "REVERT")
    n_fail = sum(1 for h in history if h.get("decision") == "FAIL")
    n_succ = sum(1 for h in history if _is_successful(h))
    gen_at = datetime.now().isoformat(timespec="seconds")

    # ════════ 1. all_strategies.md (全部轮次) ════════
    lines = [
        f"# 全部优化轮次策略 — {op}",
        f"",
        f"> 生成: {gen_at} | 总轮次: {len(history)} | 采纳 KEEP: {n_keep} | 回退 REVERT: {n_revert} | 失败 FAIL: {n_fail}",
        f"> best_speedup: {state.get('best_speedup')}x (best_round={state.get('best_round')})",
        f"",
        f"| Round | Tier | Decision | Result | Speedup | Strategy | Change |",
        f"|---|---|---|---|---|---|---|",
    ]
    for h in history:
        r = h.get("round", "?")
        t = h.get("tier", "?")
        dec = h.get("decision", "?")
        res = h.get("result", "?")
        sp = h.get("speedup")
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "?"
        strat = (h.get("strategy") or "").replace("|", "/")[:60]
        chg = (h.get("change") or "").replace("|", "/")[:50]
        lines.append(f"| {r} | {t} | {dec} | {res} | {sp_s} | {strat} | {chg} |")
    all_path = out_dir / "all_strategies.md"
    all_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ════════ 2. successful_strategies.md (仅成功优化) ════════
    succ = [h for h in history if _is_successful(h)]
    slines = [
        f"# 成功优化策略 — {op}",
        f"",
        f"> 仅「采纳且有效」的轮次 (KEEP + 严格超越上一轮; 排除 promote/REVERT/FAIL/采集失败).",
        f"> 生成: {gen_at} | 共 {len(succ)} 条成功策略 | best_speedup: {state.get('best_speedup')}x",
        f"",
    ]
    if not succ:
        slines.append("_(暂无成功采纳的优化轮次)_")
    for h in succ:
        r = h.get("round", "?")
        t = h.get("tier", "?")
        sp = h.get("speedup")
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "?"
        prev = h.get("prev_speedup")
        prev_s = f"{prev:.2f}x" if isinstance(prev, (int, float)) else "?"
        strat = h.get("strategy") or "(无)"
        chg = h.get("change") or "(无)"
        ei = h.get("expected_impact") or ""
        tf = h.get("tflops")
        tf_s = f"{tf:.1f}" if isinstance(tf, (int, float)) else "?"
        first = " (首次采纳)" if (prev_s == "?") else ""
        slines.append(f"## R{r} Tier{t} — {sp_s}{first}")
        slines.append(f"- **策略**: {strat}")
        slines.append(f"- **改动**: {chg}")
        if ei:
            slines.append(f"- **预期**: {ei}")
        slines.append(f"- **加速比**: {prev_s} → {sp_s} | TFLOPS: {tf_s}")
        # 完整 changes[] (old→new, 复盘用)
        cf = h.get("changes_full") or []
        if cf:
            slines.append(f"- **changes[]**:")
            for c in cf:
                slines.append(f"    - `{(c.get('old_code') or '').strip()[:80]}` → `{(c.get('new_code') or '').strip()[:80]}`")
        slines.append("")
    succ_path = out_dir / "successful_strategies.md"
    succ_path.write_text("\n".join(slines) + "\n", encoding="utf-8")

    return all_path, succ_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        kd = Path(sys.argv[1])
        a, s = generate(kd)
        hist = json.loads((kd / "optimization_trajectory.json").read_text(encoding="utf-8"))["history"]
        n_succ = sum(1 for h in hist if _is_successful(h))
        print(f"[strategy] all → {a}")
        print(f"[strategy] successful → {s} ({n_succ} 条成功策略)")
    else:
        print("用法: python3 feedback/strategy_summary.py <outputs/<op> 目录>")
