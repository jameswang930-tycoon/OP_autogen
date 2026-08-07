#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""优秀优化案例自动记录 — 每个 Tier 一个 JSON.

★触发条件: 某轮优化 (相对【上一最优累计加速比】, 不是相对基准) 超过阈值时,
  把该轮的 所属策略/瓶颈/解决方案/修改前后代码/加速比 记为该 Tier 的优秀案例.
    speedup_after > best_before × EXCELLENT_THRESHOLD   (默认 1.3)

★存储: memory/tier{N}_cases.json  (N=1..6, 每策略层一个, 纯追加不覆盖)
★读取: planner 优化前 load(tier) 读最近案例作参考学习.
★健壮性: 任何异常不抛 (损坏则重置), 同 (op, round) 去重, 并发安全(单进程).
"""
import json
import os
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
CASES_DIR = _PROJECT / "memory"
# 阈值: 本轮累计加速比 > 上一最优 × EXCELLENT_THRESHOLD 才算优秀案例 (env 可调)
EXCELLENT_THRESHOLD = float(os.environ.get("EXCELLENT_CASE_THRESHOLD", "1.3"))
# 默认最近读多少条给 planner
DEFAULT_LIMIT = int(os.environ.get("EXCELLENT_CASE_LIMIT", "5"))


def cases_path(tier: int) -> Path:
    """tier 1~6 各自的 JSON 文件路径."""
    return CASES_DIR / f"tier{int(tier)}_cases.json"


def _load(tier: int) -> list:
    """读某 Tier 案例列表; 文件缺失/损坏 → 空列表 (不抛)."""
    p = cases_path(tier)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, list):
                return d
        except Exception:
            try:
                print(f"  [excellent-case] tier{tier} 文件损坏, 重置")
            except Exception:
                pass   # 控制台编码问题也别崩
    return []


def is_excellent(best_before, speedup) -> bool:
    """判定: 本轮 (相对上一最优) 是否构成优秀案例."""
    try:
        bb = float(best_before)
        sp = float(speedup)
    except (TypeError, ValueError):
        return False
    # 上一最优至少是基准(1.0), 本轮必须严格超过 上一最优×阈值
    return bb >= 1.0 and sp > bb * EXCELLENT_THRESHOLD


def record(tier: int, case: dict) -> bool:
    """追加一个优秀案例 (同 op+round 去重). 永不抛异常."""
    try:
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        cases = _load(tier)
        key = (case.get("op"), case.get("round"))
        cases = [c for c in cases if (c.get("op"), c.get("round")) != key]
        cases.append(case)
        cases_path(tier).write_text(
            json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except Exception as e:
        try:
            print(f"  [excellent-case] 记录失败: {str(e)[:120]}")
        except Exception:
            pass
        return False


def load(tier: int, limit: int = DEFAULT_LIMIT) -> list:
    """读某 Tier 最近 limit 条优秀案例 → planner 参考学习."""
    cases = _load(tier)
    if not cases:
        return []
    return cases[-limit:]


def format_for_planner(tier: int, limit: int = DEFAULT_LIMIT) -> str:
    """把某 Tier 的优秀案例格式化成一串文本 (进 planner prompt, 截断防爆)."""
    cases = load(tier, limit)
    if not cases:
        return "(暂无本层优秀案例)"
    lines = [f"## 本层优秀案例 (历史大加速比轮次, ★参考学习, 别重复发明) — {len(cases)} 条:"]
    for c in cases:
        imp = c.get("improvement_x")
        lines.append(f"- R{c.get('round')} [{c.get('op')}] {c.get('strategy','?')}: "
                     f"{c.get('speedup_before')}x→{c.get('speedup_after')}x "
                     f"(本层提速 {imp}x) 瓶颈={c.get('bottleneck','?')}")
        for ch in (c.get("changes") or [])[:2]:
            lines.append(f"    改: {str(ch.get('old_code',''))[:80]} → {str(ch.get('new_code',''))[:80]}")
    return "\n".join(lines)[:2500]
