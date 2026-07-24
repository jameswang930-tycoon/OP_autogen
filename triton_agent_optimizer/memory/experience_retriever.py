#!/usr/bin/env python3
"""
经验检索器 — 分 Tier 存储 + 精确键匹配 + 3 级放宽检索。

═══════════════════════════════════════════════════════════════════════════════
  存储结构
═══════════════════════════════════════════════════════════════════════════════

  memory/experiences/
    tier1_algorithm.json    ← Algorithm 成功/失败案例
    tier2_fusion.json       ← Fusion 案例
    tier3_tiling.json       ← Tiling 案例
    tier4_memory.json       ← Memory 案例
    tier5_compute.json      ← Compute 案例
    tier6_architecture.json ← Architecture 案例

  每个文件是 JSON 数组:
  [{
    "fingerprint": {"op_type", "bottleneck_type", "engine", "tier"},
    "strategy": "merge_small_transfers",
    "speedup": 1.30,
    "status": "SUCCESS",
    "description": "合并 4×1KB gm_to_ub → 1×4KB",
    "timestamp": "2026-07-23T16:00:00"
  }, ...]

═══════════════════════════════════════════════════════════════════════════════
  匹配逻辑 (3 级)
═══════════════════════════════════════════════════════════════════════════════

  Level 1: 精确匹配 {op_type, bottleneck_type, engine, tier}
  Level 2: 放宽 {bottleneck_type, tier}
  Level 3: 再放宽 {engine, tier}
  每级最多返回 3 条, 总共最多 5 条

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  from memory.experience_retriever import retrieve, record

  # 检索
  cases = retrieve(op_type="gm_to_ub", bottleneck_type="memory_latency",
                    engine="GM→UB", tier=3)

  # 记录
  record(tier=3, fingerprint={...}, strategy="merge_small_transfers",
         speedup=1.30, status="SUCCESS", description="...")
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

_EXPERIENCES_DIR = Path(__file__).resolve().parent / "experiences"

TIER_FILES = {
    1: "tier1_algorithm.json",
    2: "tier2_fusion.json",
    3: "tier3_tiling.json",
    4: "tier4_memory.json",
    5: "tier5_compute.json",
    6: "tier6_architecture.json",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  检索
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(
    op_type: str = "",
    bottleneck_type: str = "",
    engine: str = "",
    tier: int = 1,
    max_results: int = 5,
) -> List[dict]:
    """检索相似优化案例。3 级匹配。

    Args:
        op_type: 瓶颈 op 类型 (e.g. "gm_to_ub")
        bottleneck_type: 瓶颈类型 (e.g. "memory_latency")
        engine: 引擎名 (e.g. "GM→UB")
        tier: 当前优化层级
        max_results: 最多返回条数

    Returns:
        [{strategy, speedup, status, description, match_level}, ...]
    """
    entries = _load_tier(tier)
    if not entries:
        return []

    results: List[dict] = []
    seen: set = set()

    # Level 1: 精确匹配 (op_type + bottleneck_type + engine)
    for e in entries:
        fp = e.get("fingerprint", {})
        if (fp.get("op_type") == op_type
                and fp.get("bottleneck_type") == bottleneck_type
                and fp.get("engine") == engine):
            key = f"{e.get('strategy')}_{e.get('status')}"
            if key not in seen:
                results.append({**e, "match_level": 1})
                seen.add(key)

    # Level 2: bottleneck_type + tier
    if len(results) < max_results:
        for e in entries:
            fp = e.get("fingerprint", {})
            if fp.get("bottleneck_type") == bottleneck_type:
                key = f"{e.get('strategy')}_{e.get('status')}"
                if key not in seen:
                    results.append({**e, "match_level": 2})
                    seen.add(key)

    # Level 3: engine
    if len(results) < max_results:
        for e in entries:
            fp = e.get("fingerprint", {})
            if fp.get("engine") == engine:
                key = f"{e.get('strategy')}_{e.get('status')}"
                if key not in seen:
                    results.append({**e, "match_level": 3})
                    seen.add(key)

    # 排序: 成功案例优先, 然后按 speedup 降序
    results.sort(key=lambda r: (
        0 if r.get("status") == "SUCCESS" else 1,
        -r.get("speedup", 1.0),
    ))

    return results[:max_results]


# ═══════════════════════════════════════════════════════════════════════════════
#  记录
# ═══════════════════════════════════════════════════════════════════════════════

def record(
    tier: int,
    fingerprint: dict,
    strategy: str,
    speedup: float,
    status: str,           # "SUCCESS" | "FAIL"
    description: str = "",
    decision_reason: str = "",
):
    """记录优化经验到对应 Tier 文件。

    Args:
        tier: 优化层级 (1~6)
        fingerprint: {op_type, bottleneck_type, engine}
        strategy: 优化策略名
        speedup: 加速比
        status: "SUCCESS" (>5% 提升) | "FAIL" (<2% 倒退)
        description: 优化描述
        decision_reason: 失败原因 (FAIL 时)
    """
    entries = _load_tier(tier)

    entry = {
        "fingerprint": fingerprint,
        "strategy": strategy,
        "speedup": round(speedup, 4),
        "status": status,
        "description": description[:300],
        "timestamp": datetime.now().isoformat(),
    }
    if decision_reason and status == "FAIL":
        entry["decision_reason"] = decision_reason[:200]

    entries.append(entry)

    # 只保留最近 50 条 (避免文件过大)
    if len(entries) > 50:
        entries = entries[-50:]

    _save_tier(tier, entries)


# ═══════════════════════════════════════════════════════════════════════════════
#  格式化 (注入 Planner prompt)
# ═══════════════════════════════════════════════════════════════════════════════

def format_for_prompt(cases: List[dict]) -> str:
    """将检索结果格式化为 LLM prompt 可读文本。"""
    if not cases:
        return "(no similar cases found — this may be the first optimization of this type)"

    successes = [c for c in cases if c.get("status") == "SUCCESS"]
    failures = [c for c in cases if c.get("status") == "FAIL"]

    lines = []

    if successes:
        lines.append("### Successful Strategies (reference)")
        for c in successes[:3]:
            level = "★" if c.get("match_level") == 1 else "☆"
            lines.append(
                f"- {level} `{c.get('strategy','?')}` → "
                f"{c.get('speedup',1.0):.2f}x | "
                f"{c.get('description','')[:120]}"
            )

    if failures:
        lines.append("")
        lines.append("### Failed Strategies (avoid)")
        for c in failures[:2]:
            lines.append(
                f"- ✗ `{c.get('strategy','?')}` → "
                f"{c.get('speedup',1.0):.2f}x | "
                f"{c.get('decision_reason', c.get('description',''))[:120]}"
            )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  IO
# ═══════════════════════════════════════════════════════════════════════════════

def _load_tier(tier: int) -> list:
    fname = TIER_FILES.get(tier)
    if fname is None:
        return []
    fpath = _EXPERIENCES_DIR / fname
    if fpath.exists():
        try:
            return json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_tier(tier: int, entries: list):
    fname = TIER_FILES.get(tier)
    if fname is None:
        return
    _EXPERIENCES_DIR.mkdir(parents=True, exist_ok=True)
    (_EXPERIENCES_DIR / fname).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    # 记录几条经验
    record(tier=3, fingerprint={"op_type": "gm_to_ub", "bottleneck_type": "memory_latency", "engine": "GM→UB"},
           strategy="increase_tile_size", speedup=1.30, status="SUCCESS",
           description="BLOCK_SIZE 256 → 8192, bw_util 21% → 90%")
    record(tier=3, fingerprint={"op_type": "gm_to_ub", "bottleneck_type": "memory_latency", "engine": "GM→UB"},
           strategy="merge_small_transfers", speedup=1.15, status="SUCCESS",
           description="合并 4×1KB → 1×4KB gm_to_ub")
    record(tier=3, fingerprint={"op_type": "gm_to_ub", "bottleneck_type": "memory_bandwidth", "engine": "UB→GM"},
           strategy="double_buffering", speedup=0.97, status="FAIL",
           description="UB 溢出, 无法 double buffer", decision_reason="UB capacity 不足")

    # 检索
    cases = retrieve(op_type="gm_to_ub", bottleneck_type="memory_latency",
                     engine="GM→UB", tier=3)
    print(f"Retrieved {len(cases)} cases:")
    for c in cases:
        print(f"  L{c['match_level']} [{c['status']}] {c['strategy']} → {c['speedup']}x")

    # 格式化
    prompt = format_for_prompt(cases)
    print(f"\nPrompt injection:\n{prompt[:500]}")

    print("[experience_retriever] OK")


if __name__ == "__main__":
    _self_test()
