#!/usr/bin/env python3
"""
瓶颈诊断器 —— 从 merged_report.json 提取瓶颈信息 (脚本, 规则引擎)。

═══════════════════════════════════════════════════════════════════════════════
  定位: 分析层 (analyzers/), 非智能体层。
  输入: merged/merged_report.json (29 字段完整报告)
  输出: BottleneckDiagnosis (结构化 JSON, ~1KB)
  消费者: agents/planner.py → 注入 LLM prompt 生成优化计划

═══════════════════════════════════════════════════════════════════════════════
  设计原则
═══════════════════════════════════════════════════════════════════════════════

  1. 规则引擎: 确定性阈值分类, 不依赖 LLM
  2. Tier-aware: 不同优化层级关注不同瓶颈类型
  3. 数据压缩: 完整报告可能有几十 KB → 诊断结果 ~1KB
  4. 标注可靠性: placeholder engines 的结果标注 UNCERTAIN

═══════════════════════════════════════════════════════════════════════════════
  Tier × 瓶颈类型矩阵
═══════════════════════════════════════════════════════════════════════════════

  Tier 1 (Algorithm): 关注结构 — execution_mode, op_count, algorithm_pattern
  Tier 2 (Fusion):    关注依赖 — RAW chains, WAR avoidable, same-pipeline adjacencies
  Tier 3 (Tiling):    关注传输 — bw_util < 70% on critical_path
  Tier 4 (Memory):    关注饱和 — regime=saturated ops (已达峰值)
  Tier 5 (Compute):   关注计算 — VecUnit/CubeUnit time_ratio
  Tier 6 (Arch):      关注引擎 — engine_utilization imbalance, placeholder engines

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  python analyzers/bottleneck_diagnoser.py outputs/.../round0/merged/merged_report.json
  python analyzers/bottleneck_diagnoser.py outputs/.../round0/merged/merged_report.json --tier 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  硬件参数 (从 simulator.py 同源加载)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "simulator", str(Path(__file__).resolve().parent.parent.parent
                          / "costModel" / "cost_emulator" / "simulator.py"))
    if _spec and _spec.loader:
        _sim = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_sim)
        SATURATION_PARAMS = _sim.SATURATION_PARAMS
        ENG_NAME = _sim.ENG_NAME
    else:
        raise ImportError
except Exception:
    SATURATION_PARAMS = {
        0: {"vpeak": 121.08, "k0": 6.65, "peak_clamp": 80.83},
        1: {"vpeak": 190.19, "k0": 10.72, "peak_clamp": 76.67},
        2: {"vpeak": 461.0,  "k0": 4.50, "peak_clamp": 404.0},
        3: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},
        4: {"vpeak": 100.0,  "k0": 6.65, "peak_clamp": 100.0},
        5: {"vpeak": 150.0,  "k0": 0,    "peak_clamp": 150.0},
        6: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},
    }
    ENG_NAME = {0: "GM→UB", 1: "UB→GM", 2: "VecUnit", 3: "GM→L1",
                4: "L1→L0", 5: "CubeUnit", 6: "L0→GM"}

ENGINE_TO_ID = {v: k for k, v in ENG_NAME.items()}
MEASURED_ENGINES = {0, 1, 2}  # engines with real SATURATION_PARAMS
PLACEHOLDER_ENGINES = {3, 4, 5, 6}

# Tier → 关注的瓶颈类型优先级
TIER_BOTTLENECK_PRIORITY = {
    1: ["structural"],
    2: ["dependency", "memory_bandwidth", "memory_latency"],
    3: ["memory_latency", "memory_bandwidth"],
    4: ["memory_bandwidth", "memory_latency", "dependency"],
    5: ["compute_vec", "compute_cube"],
    6: ["engine_contention", "compute_vec", "compute_cube", "memory_bandwidth"],
}


@dataclass
class BottleneckDiagnosis:
    """瓶颈诊断结果 (~1KB, 注入 Planner LLM prompt)。"""
    # 瓶颈 op
    bottleneck_op_id: int
    bottleneck_op_type: str
    bottleneck_engine: str
    bottleneck_time_ratio: float
    bottleneck_bw_utilization: float
    bottleneck_regime: str
    on_critical_path: bool

    # 分类
    bottleneck_type: str  # memory_bandwidth / memory_latency / compute_vec / compute_cube / dependency / engine_contention / structural
    bottleneck_category: str  # MEMORY / COMPUTE / DEPENDENCY / ENGINE / STRUCTURAL

    # 优化空间
    optimization_headroom: str  # HIGH / MEDIUM / LOW / NONE / UNCERTAIN
    headroom_reason: str

    # 当前 Tier 信息
    current_tier: int
    tier_name: str

    # 建议
    suggested_strategies: List[str] = field(default_factory=list)
    suggested_playbook_sections: List[str] = field(default_factory=list)

    # 辅助数据 (供 Planner 参考)
    all_bottleneck_candidates: List[dict] = field(default_factory=list)
    aggregated_by_type: List[dict] = field(default_factory=list)  # ★ 聚合瓶颈
    engine_utilization_top3: List[dict] = field(default_factory=list)
    parallelism_info: dict = field(default_factory=dict)
    dependency_issues: dict = field(default_factory=dict)
    structural_issues: List[str] = field(default_factory=list)

    # 原始数据引用
    total_ns: float = 0.0
    num_ops: int = 0
    execution_mode: str = ""


def diagnose(merged_report: dict, current_tier: int = 1) -> BottleneckDiagnosis:
    """主入口: 从合并后的完整报告诊断瓶颈。

    Args:
        merged_report: merged_report.json 的内容 (dict)
        current_tier: 当前优化层级 (1~6)

    Returns:
        BottleneckDiagnosis 结构化诊断结果
    """
    ops = merged_report.get("per_op_statistics", [])
    if not ops:
        print("  [DIAG] WARNING: no ops in merged_report, skipping diagnosis")
        return BottleneckDiagnosis(
            bottleneck_op_id=-1, bottleneck_op_type="?", bottleneck_engine="?",
            bottleneck_time_ratio=0, bottleneck_bw_utilization=0,
            bottleneck_regime="?", on_critical_path=False,
            bottleneck_type="unknown", bottleneck_category="UNKNOWN",
            optimization_headroom="NONE", headroom_reason="No ops to diagnose",
            current_tier=current_tier, tier_name=f"Tier {current_tier}",
        )
    summary = merged_report.get("execution_summary", {})
    engine_util = merged_report.get("engine_utilization", {})
    cp = merged_report.get("critical_path", {})
    deps = merged_report.get("dependencies_summary", {})
    parallelism = merged_report.get("parallelism", {})

    total_ns = summary.get("total_ns", 0)
    cp_ops = set(cp.get("path", []))
    execution_mode = summary.get("execution_mode", "unknown")

    # ── Step 1: 按当前 Tier 找出候选瓶颈 ──
    candidates = _find_candidates(ops, cp_ops, engine_util, deps, parallelism,
                                   execution_mode, current_tier)

    # ── Step 1.5: 聚合分析 (同类型 op 合并统计) ──
    aggregated = _aggregate_by_type(ops, cp_ops, total_ns)

    # ── Step 2: 选主瓶颈 ──
    primary = _select_primary(candidates, aggregated, current_tier)

    # ── Step 3: 分类 + 评估优化空间 ──
    btype = _classify_bottleneck(primary, deps)
    category = _bottleneck_category(btype)
    headroom, reason = _assess_headroom(primary, btype)

    # ── Step 4: 建议策略 + playbook ──
    strategies = _suggest_strategies(btype, headroom, current_tier)
    playbook_sections = _suggest_playbook(btype, current_tier)

    # ── Step 5: 构建诊断结果 ──
    tier_names = {
        1: "Algorithmic Structure", 2: "Operator Fusion",
        3: "Tiling & Block Config", 4: "Memory Access",
        5: "Compute & Occupancy", 6: "910B3 Architecture",
    }

    diagnosis = BottleneckDiagnosis(
        bottleneck_op_id=primary.get("op_id", -1),
        bottleneck_op_type=primary.get("op_type", "?"),
        bottleneck_engine=primary.get("engine", "?"),
        bottleneck_time_ratio=primary.get("time_ratio", 0),
        bottleneck_bw_utilization=primary.get("bw_utilization", 0) if isinstance(primary.get("bw_utilization"), (int, float)) else 0,
        bottleneck_regime=str(primary.get("regime", "?")),
        on_critical_path=primary.get("op_id", -1) in cp_ops,
        bottleneck_type=btype,
        bottleneck_category=category,
        optimization_headroom=headroom,
        headroom_reason=reason,
        current_tier=current_tier,
        tier_name=tier_names.get(current_tier, "Unknown"),
        suggested_strategies=strategies,
        suggested_playbook_sections=playbook_sections,
        all_bottleneck_candidates=candidates[:5],
        aggregated_by_type=aggregated[:10],
        engine_utilization_top3=_top3_engines(engine_util),
        parallelism_info=parallelism if isinstance(parallelism, dict) else {},
        dependency_issues={
            "raw_count": len(deps.get("raw", [])),
            "war_count": len(deps.get("war", [])),
            "waw_count": len(deps.get("waw", [])),
            "has_avoidable_war": any(
                d.get("type") == "WAR" for d in deps.get("war", [])
            ),
        },
        structural_issues=_detect_structural_issues(ops, execution_mode, total_ns),
        total_ns=total_ns,
        num_ops=summary.get("num_ops", len(ops)),
        execution_mode=execution_mode,
    )

    return diagnosis


# ═══════════════════════════════════════════════════════════════════════════════
#  候选瓶颈查找 (Tier-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_candidates(
    ops: list, cp_ops: set, engine_util: dict,
    deps: dict, parallelism: dict, exec_mode: str, tier: int,
) -> list:
    """按 Tier 优先级查找候选瓶颈 op。"""
    candidates = []

    for op in ops:
        if tier == 3:
            # Tier 3 (Tiling): 关注传输 op. 优先低利用率, 全饱和时回退到全部
            if op.get("engine") in ("GM→UB", "UB→GM", "GM→L1", "L1→L0", "L0→GM"):
                bw = op.get("bw_utilization", 0)
                if isinstance(bw, (int, float)) and bw < 0.70:
                    candidates.append(_candidate(op, "memory_latency", bw))
                else:
                    candidates.append(_candidate(op, "memory_bandwidth", bw))
        elif tier == 4:
            # Tier 4 (Memory): 关注所有传输 op (包括已饱和的)
            if op.get("engine") in ("GM→UB", "UB→GM", "GM→L1", "L1→L0", "L0→GM"):
                candidates.append(_candidate(op, "memory_bandwidth",
                                              op.get("bw_utilization", 0)))
        elif tier == 5:
            # Tier 5 (Compute): 关注 VecUnit/CubeUnit
            if op.get("engine") in ("VecUnit", "CubeUnit"):
                candidates.append(_candidate(op, "compute_vec",
                                              op.get("bw_utilization", 0)))
        else:
            # Tier 1/2/6: 全面关注
            candidates.append(_candidate(op, "general",
                                          op.get("bw_utilization", 0)))

    # 在关键路径上的排前面
    for c in candidates:
        if c["op_id"] in cp_ops:
            c["score"] = c.get("score", 0) + 100
    # 时间占比大的排前面
    for c in candidates:
        tr = c.get("time_ratio", 0)
        if isinstance(tr, str):
            try:
                tr = float(tr.rstrip("%")) / 100.0
            except ValueError:
                tr = 0.0
        c["score"] = c.get("score", 0) + float(tr) * 200

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _candidate(op: dict, btype: str, bw: float) -> dict:
    eid = ENGINE_TO_ID.get(op.get("engine", ""))
    is_measured = eid in MEASURED_ENGINES if eid is not None else False
    return {
        "op_id": op.get("op_id"), "op_type": op.get("op_type"),
        "engine": op.get("engine"), "time_ratio": op.get("time_ratio", 0),
        "bw_utilization": bw if isinstance(bw, (int, float)) else 0,
        "regime": op.get("regime", "?"), "bottleneck_type": btype,
        "is_measured_engine": is_measured,
        "score": 0,
    }


def _aggregate_by_type(ops: list, cp_ops: set, total_ns: float) -> list:
    """按 op_type + engine 聚合统计 (发现聚合瓶颈)。

    例如: 50 个 gm_to_ub 各占 2%, 聚合后占 100% → 这才是真正的瓶颈。
    """
    groups: Dict[str, dict] = {}
    for op in ops:
        key = f"{op.get('op_type')}|{op.get('engine')}"
        if key not in groups:
            groups[key] = {
                "op_type": op.get("op_type"), "engine": op.get("engine"),
                "count": 0, "total_duration_ns": 0, "total_time_ratio": 0,
                "on_critical_path_count": 0, "avg_bw_util": 0,
                "bw_utils": [], "regimes": [],
            }
        g = groups[key]
        g["count"] += 1
        dur = op.get("duration_ns", 0)
        if isinstance(dur, (int, float)):
            g["total_duration_ns"] += dur
            g["total_time_ratio"] += dur / total_ns if total_ns > 0 else 0
        if op.get("op_id") in cp_ops:
            g["on_critical_path_count"] += 1
        bw = op.get("bw_utilization", 0)
        if isinstance(bw, (int, float)):
            g["bw_utils"].append(bw)
        g["regimes"].append(str(op.get("regime", "")))

    # 计算平均
    result = []
    for key, g in groups.items():
        if g["bw_utils"]:
            g["avg_bw_util"] = sum(g["bw_utils"]) / len(g["bw_utils"])
        result.append(g)

    result.sort(key=lambda g: g["total_time_ratio"], reverse=True)
    return result


def _select_primary(candidates: list, aggregated: list, tier: int) -> dict:
    """从候选中选主瓶颈。"""
    if not candidates:
        return {"op_id": -1, "op_type": "?", "engine": "?",
                "time_ratio": 0, "bw_utilization": 0, "regime": "?"}

    # Tier 1: 关注结构性, 不选特定 op
    if tier == 1:
        return candidates[0]  # 返回 time_ratio 最高的供参考

    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  瓶颈分类
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_bottleneck(op: dict, deps: dict) -> str:
    """分类瓶颈类型。"""
    engine = op.get("engine", "")
    bw = op.get("bw_utilization", 0)
    regime = str(op.get("regime", ""))
    is_measured = ENGINE_TO_ID.get(engine) in MEASURED_ENGINES

    # 传输引擎
    if engine in ("GM→UB", "UB→GM", "GM→L1", "L1→L0", "L0→GM"):
        if not is_measured:
            return "memory_bandwidth"  # placeholder → 默认当作带宽瓶颈
        if isinstance(bw, (int, float)):
            if regime in ("floor", "ramp") and bw < 0.70:
                return "memory_latency"
            return "memory_bandwidth"
        return "memory_bandwidth"

    # 计算引擎
    if engine in ("VecUnit",):
        return "compute_vec"
    if engine in ("CubeUnit",):
        return "compute_cube"

    return "unknown"


def _bottleneck_category(btype: str) -> str:
    if btype in ("memory_bandwidth", "memory_latency"): return "MEMORY"
    if btype in ("compute_vec", "compute_cube"): return "COMPUTE"
    if btype == "dependency": return "DEPENDENCY"
    if btype == "engine_contention": return "ENGINE"
    if btype == "structural": return "STRUCTURAL"
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════════
#  优化空间评估
# ═══════════════════════════════════════════════════════════════════════════════

def _assess_headroom(op: dict, btype: str) -> Tuple[str, str]:
    """评估可优化空间。"""
    engine = op.get("engine", "")
    eid = ENGINE_TO_ID.get(engine)
    bw = op.get("bw_utilization", 0)
    regime = str(op.get("regime", ""))
    is_measured = eid in MEASURED_ENGINES if eid is not None else False

    if not is_measured:
        return ("UNCERTAIN",
                f"engine '{engine}' uses PLACEHOLDER params — optimization advice is speculative")

    if btype == "memory_latency":
        if isinstance(bw, (int, float)):
            if bw < 0.40:
                return ("HIGH",
                        f"BW utilization={bw:.0%}, regime={regime} — large room to increase tile size")
            return ("MEDIUM",
                    f"BW utilization={bw:.0%}, approaching saturation — moderate room")
        return ("MEDIUM", "BW data unclear — moderate room assumed")

    if btype == "memory_bandwidth":
        if isinstance(bw, (int, float)) and bw > 0.95:
            return ("LOW",
                    f"BW utilization={bw:.0%}, regime={regime} — already at peak, cannot increase throughput by enlarging tile")
        return ("MEDIUM",
                f"BW utilization={bw:.0%} — some room but limited")

    if btype == "compute_vec":
        if isinstance(bw, (int, float)):
            if bw > 0.90:
                return ("LOW",
                        f"VecUnit at {bw:.0%} utilization — near peak, focus on overlap or reduce data")
            return ("MEDIUM",
                    f"VecUnit at {bw:.0%} — moderate room")

    if btype == "compute_cube":
        return ("LOW", "CubeUnit is flat (size-independent) — limited tuning options")

    if btype == "dependency":
        return ("HIGH", "Dependency bottleneck — WAR may be avoidable, RAW chain may be fusible")

    return ("MEDIUM", "Unknown bottleneck type — moderate room assumed")


# ═══════════════════════════════════════════════════════════════════════════════
#  策略 + Playbook 建议
# ═══════════════════════════════════════════════════════════════════════════════

def _suggest_strategies(btype: str, headroom: str, tier: int) -> List[str]:
    """建议优化策略列表 (给 Planner LLM 参考)。"""
    strategies = []

    if tier == 1:
        strategies.append("evaluate_algorithm_choice")
    elif tier == 2:
        if btype == "dependency":
            strategies.extend(["fuse_elementwise_ops", "break_war_with_new_buffer"])
        else:
            strategies.append("identify_fusion_opportunities")
    elif tier == 3:
        if btype in ("memory_latency", "memory_bandwidth"):
            strategies.append("increase_tile_size")
        strategies.append("tune_block_config")
    elif tier == 4:
        if btype == "memory_latency":
            strategies.append("merge_small_transfers")
        if btype == "memory_bandwidth":
            strategies.extend(["reduce_data_volume", "double_buffering", "use_faster_engine"])
    elif tier == 5:
        strategies.extend(["overlap_compute_transfer", "optimize_vectorization"])
    elif tier == 6:
        strategies.extend(["adjust_grid_count", "switch_pipeline", "l2_residency"])

    if not strategies:
        strategies.append("analyze_deeper")
    return strategies


def _suggest_playbook(btype: str, tier: int) -> List[str]:
    """建议注入的 Playbook 章节。"""
    mapping = {
        1: ["playbook_algorithmic.md"],
        2: ["playbook_fusion.md"],
        3: ["playbook_tiling.md", "playbook_memory.md §1"],
        4: ["playbook_memory.md §2", "playbook_memory.md §3", "playbook_memory.md §4"],
        5: ["playbook_compute.md §1", "playbook_compute.md §3"],
        6: ["playbook_910b3_arch.md §1", "playbook_910b3_arch.md §3", "playbook_910b3_arch.md §4"],
    }
    return mapping.get(tier, ["optimization_playbook.md"])


# ═══════════════════════════════════════════════════════════════════════════════
#  结构性诊断
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_structural_issues(
    ops: list, exec_mode: str, total_ns: float,
) -> List[str]:
    """检测结构性问题 (给 Tier 1 用)。"""
    issues = []
    if exec_mode == "sequential" and len(ops) > 5:
        issues.append(f"Fully sequential with {len(ops)} ops — consider persistent kernel or pipeline overlap")
    if len(ops) > 20:
        issues.append(f"Large op count ({len(ops)}) — consider algorithmic restructuring")
    if total_ns > 10000:
        issues.append(f"High latency ({total_ns:.0f}ns) — check for unnecessary sync barriers")
    return issues


def _top3_engines(engine_util: dict) -> List[dict]:
    if not engine_util:
        return []
    sorted_eng = sorted(engine_util.items(), key=lambda x: x[1], reverse=True)
    return [{"engine": e, "utilization": round(u, 4)} for e, u in sorted_eng[:3]]


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口 & 自测
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_round(round_dir: Path, current_tier: int = 1) -> BottleneckDiagnosis:
    """诊断一个 round 目录的瓶颈。"""
    merged_file = round_dir / "merged" / "merged_report.json"
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged report not found: {merged_file}\n"
                                f"Run dsl_merger.py first.")
    with open(merged_file, encoding="utf-8") as f:
        report = json.load(f)
    return diagnose(report, current_tier)


def _self_test():
    outputs_root = Path(__file__).resolve().parent.parent / "outputs"
    round_dir = outputs_root / "vector_add_fp16_N65536" / "round0"

    if not (round_dir / "merged" / "merged_report.json").exists():
        print("[diagnoser] SKIP: merged_report.json not found. Run dsl_merger.py first.")
        return

    print("=" * 60)
    print("BottleneckDiagnoser Self-Test")
    print("=" * 60)

    for tier in range(1, 7):
        d = diagnose_round(round_dir, current_tier=tier)
        print(f"\n─── Tier {tier}: {d.tier_name} ───")
        print(f"  Bottleneck:  op{d.bottleneck_op_id} ({d.bottleneck_op_type}, {d.bottleneck_engine})")
        print(f"  Type:        {d.bottleneck_type} ({d.bottleneck_category})")
        print(f"  Time ratio:  {d.bottleneck_time_ratio:.2%}")
        print(f"  BW util:     {d.bottleneck_bw_utilization:.2%}" if d.bottleneck_bw_utilization else "  BW util:     N/A")
        print(f"  Regime:      {d.bottleneck_regime}")
        print(f"  Headroom:    {d.optimization_headroom} — {d.headroom_reason}")
        print(f"  Strategies:  {d.suggested_strategies}")
        print(f"  Playbook:    {d.suggested_playbook_sections}")
        print(f"  Structural:  {d.structural_issues}")
        print(f"  Engines top: {d.engine_utilization_top3}")
        print(f"  Dependencies: RAW={d.dependency_issues['raw_count']} "
              f"WAR={d.dependency_issues['war_count']} "
              f"WAW={d.dependency_issues['waw_count']}")

    # 验证
    d3 = diagnose_round(round_dir, current_tier=3)
    assert "memory" in d3.bottleneck_type, \
        f"Tier 3 should be memory bottleneck, got {d3.bottleneck_type}"
    assert d3.bottleneck_op_id >= 0, "Should have valid bottleneck op"
    print(f"\n{'=' * 60}")
    print("ALL TIER DIAGNOSES PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        tier = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        if arg.is_dir():
            d = diagnose_round(arg, tier)
        else:
            with open(arg, encoding="utf-8") as f:
                d = diagnose(json.load(f), tier)
        print(json.dumps({
            "bottleneck": {
                "op_id": d.bottleneck_op_id, "op_type": d.bottleneck_op_type,
                "engine": d.bottleneck_engine, "type": d.bottleneck_type,
                "category": d.bottleneck_category, "headroom": d.optimization_headroom,
                "time_ratio": d.bottleneck_time_ratio,
                "bw_utilization": d.bottleneck_bw_utilization,
                "regime": d.bottleneck_regime,
            },
            "tier": d.current_tier,
            "strategies": d.suggested_strategies,
            "playbook": d.suggested_playbook_sections,
            "structural_issues": d.structural_issues,
        }, indent=2, ensure_ascii=False))
    else:
        _self_test()
