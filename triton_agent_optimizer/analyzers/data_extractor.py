#!/usr/bin/env python3
"""
按需数据提取器 — 从完整 merged_report.json 中提取 Tier 相关的关键数据。

═══════════════════════════════════════════════════════════════════════════════
  设计原则 (OPAL 2025 模式: 990MB → 6KB)
═══════════════════════════════════════════════════════════════════════════════

  1. 不同 Tier 关注不同列 — 不是所有列每轮都需要
  2. 不同 Tier 关注不同行 — 不是所有 op 每轮都需要
  3. 提取 = 行过滤 + 列过滤 + 聚合计算
  4. 输出 ≤ 5KB (LLM prompt 友好)

═══════════════════════════════════════════════════════════════════════════════
  Tier × 提取规则矩阵
═══════════════════════════════════════════════════════════════════════════════

  Tier 1 (Algorithm):  全部 op 概览, 只看结构 → top 10 by time_ratio, 8列
  Tier 2 (Fusion):     依赖链 + buffer 生命周期 → critical_path ops, 12列
  Tier 3 (Tiling):     传输 op 的 bw_util → critical_path 传输 ops, 10列 + k0
  Tier 4 (Memory):     传输 op 的饱和状态 → 所有传输 ops, 8列 + engine_util
  Tier 5 (Compute):    计算 op 的利用率 → VecUnit/Cube ops, 6列
  Tier 6 (Arch):       引擎负载均衡 → 聚合按引擎, 5列 + placeholder 标注

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  from analyzers.data_extractor import extract
  text = extract(merged_report, bottleneck_diagnosis)
  # → ~2-5KB 文本, 直接注入 Planner LLM prompt
"""

from __future__ import annotations

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

TBD = "待补充"

def _safe_float(v, default=0.0):
    if v is None or v == TBD: return default
    try: return float(v)
    except (ValueError, TypeError): return default


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 提取配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TierExtractConfig:
    """单个 Tier 的提取规则。"""
    tier: int
    # 行过滤
    op_filter: str          # 'all' | 'critical_path' | 'transfer_only' | 'compute_only' | 'top10'
    max_ops: int = 10
    # 列过滤 — 每个 op 保留哪些字段
    op_columns: List[str] = field(default_factory=list)
    # summary sections 保留哪些
    include_exec_summary: bool = True
    include_engine_util: bool = True
    include_deps_summary: bool = True
    include_parallelism: bool = False
    include_critical_path: bool = True
    include_buffers: bool = False
    include_aggregated: bool = True
    # 是否附加 k0 参考值
    add_k0_reference: bool = False
    # 是否标注 placeholder engines
    mark_placeholder: bool = False


# 完整的提取规则表
TIER_CONFIGS: Dict[int, TierExtractConfig] = {
    1: TierExtractConfig(
        tier=1,
        op_filter="top10",
        max_ops=10,
        op_columns=["op_id", "op_type", "engine", "instruction",
                     "duration_ns", "time_ratio", "regime", "dependencies"],
        include_aggregated=True,
        include_buffers=False,
    ),
    2: TierExtractConfig(
        tier=2,
        op_filter="critical_path",
        max_ops=15,
        op_columns=["op_id", "op_type", "engine", "instruction",
                     "dst", "src", "size_kb", "memory_region",
                     "duration_ns", "time_ratio", "blocked_by", "dependencies"],
        include_buffers=True,
        include_aggregated=False,
    ),
    3: TierExtractConfig(
        tier=3,
        op_filter="critical_path",
        max_ops=10,
        op_columns=["op_id", "op_type", "engine", "size_kb",
                     "duration_ns", "time_ratio", "bw_utilization",
                     "regime", "effective_bw_gb_s", "peak_bw_gb_s"],
        add_k0_reference=True,
        include_buffers=False,
        include_aggregated=True,
    ),
    4: TierExtractConfig(
        tier=4,
        op_filter="transfer_only",
        max_ops=15,
        op_columns=["op_id", "op_type", "engine", "instruction",
                     "size_kb", "duration_ns", "time_ratio",
                     "bw_utilization", "regime"],
        include_parallelism=True,
        include_aggregated=True,
        mark_placeholder=True,
    ),
    5: TierExtractConfig(
        tier=5,
        op_filter="compute_only",
        max_ops=8,
        op_columns=["op_id", "op_type", "engine", "instruction",
                     "size_kb", "duration_ns", "time_ratio",
                     "bw_utilization", "regime"],
        include_aggregated=True,
        mark_placeholder=True,
    ),
    6: TierExtractConfig(
        tier=6,
        op_filter="all",
        max_ops=10,
        op_columns=["op_id", "op_type", "engine",
                     "duration_ns", "time_ratio", "bw_utilization", "regime"],
        include_aggregated=True,
        mark_placeholder=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  K0 参考值 (半饱和点)
# ═══════════════════════════════════════════════════════════════════════════════

ENGINE_K0 = {
    "GM→UB":  6.65,
    "UB→GM":  10.72,
    "VecUnit": 4.50,
    "GM→L1":  6.65,
    "L1→L0":  6.65,
    "CubeUnit": 0,    # flat
    "L0→GM":  6.65,
}

PLACEHOLDER_ENGINES = {"GM→L1", "L1→L0", "CubeUnit", "L0→GM"}


# ═══════════════════════════════════════════════════════════════════════════════
#  主提取函数
# ═══════════════════════════════════════════════════════════════════════════════

def extract(merged_report: dict, diagnosis: dict | None = None,
            tier: int = 1) -> str:
    """从完整合并报告中提取 Tier 相关的关键数据。

    Args:
        merged_report: merged_report.json 的内容
        diagnosis: BottleneckDiagnosis (可选, 有则附加诊断摘要)
        tier: 当前优化层级 (1~6)

    Returns:
        精简的文本 (~2-5KB), 可直接注入 LLM prompt
    """
    cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS[1])
    ops = merged_report.get("per_op_statistics", [])
    summary = merged_report.get("execution_summary", {})
    cp_path = set(merged_report.get("critical_path", {}).get("path", []))
    total_ns = summary.get("total_ns", 0)

    # ── 1. 行过滤 ──
    filtered_ops = _filter_ops(ops, cp_path, cfg)

    # ── 2. 列过滤 ──
    compact_ops = _select_columns(filtered_ops, cfg.op_columns)

    # ── 3. 构建输出 ──
    lines = []

    # Header
    lines.append(f"=== TIER {tier} EXTRACTED DATA ===")
    lines.append(f"filter: {cfg.op_filter} ({len(filtered_ops)}/{len(ops)} ops)")

    # Execution summary
    if cfg.include_exec_summary and summary:
        lines.append("")
        lines.append("--- EXECUTION SUMMARY ---")
        lines.append(f"total_ns={total_ns:.1f}  num_ops={summary.get('num_ops',len(ops))}  "
                     f"mode={summary.get('execution_mode','?')}  "
                     f"cores={summary.get('num_cores','?')}")

    # Per-op compact table
    if compact_ops:
        lines.append("")
        lines.append(f"--- OPS ({len(compact_ops)} shown) ---")
        # table header
        header = " | ".join(cfg.op_columns)
        lines.append(header)
        lines.append("-" * len(header))
        for op in compact_ops:
            row = " | ".join(str(op.get(c, "")) for c in cfg.op_columns)
            lines.append(row)

    # K0 reference
    if cfg.add_k0_reference:
        lines.append("")
        lines.append("--- K0 REFERENCE (half-saturation point) ---")
        lines.append("tile > k0×2 → saturated region (peak bandwidth)")
        for eng_name, k0 in ENGINE_K0.items():
            if k0 > 0:
                lines.append(f"  {eng_name:10s}: k0={k0:.1f}KB  "
                             f"(> {k0*2:.0f}KB → saturated)")

    # Aggregated by type
    if cfg.include_aggregated:
        agg = _compute_aggregated(ops, cp_path, total_ns)
        if agg:
            lines.append("")
            lines.append("--- AGGREGATED BY TYPE ---")
            lines.append("op_type      | engine     | count | total_ratio | avg_bw | on_cp")
            lines.append("-" * 70)
            for g in agg[:8]:
                lines.append(
                    f"{g['op_type']:12s} | {g['engine']:10s} | {g['count']:5d} | "
                    f"{g['total_time_ratio']:11.2%} | {g['avg_bw_util']:5.1%} | "
                    f"{g['on_critical_path_count']:5d}"
                )

    # Engine utilization
    if cfg.include_engine_util:
        eu = merged_report.get("engine_utilization", {})
        if eu:
            lines.append("")
            lines.append("--- ENGINE UTILIZATION ---")
            for eng_name in ["GM→UB", "UB→GM", "VecUnit", "GM→L1", "L1→L0", "CubeUnit", "L0→GM"]:
                ratio = eu.get(eng_name, 0)
                marker = " [PLACEHOLDER]" if cfg.mark_placeholder and eng_name in PLACEHOLDER_ENGINES else ""
                bar = "█" * int(ratio * 20) + "." * (20 - int(ratio * 20))
                lines.append(f"  {eng_name:10s} [{bar}] {ratio:.0%}{marker}")

    # Dependencies summary
    if cfg.include_deps_summary:
        deps = merged_report.get("dependencies_summary", {})
        if deps:
            lines.append("")
            lines.append("--- DEPENDENCIES ---")
            lines.append(f"RAW={len(deps.get('raw',[]))}  "
                         f"WAR={len(deps.get('war',[]))}  "
                         f"WAW={len(deps.get('waw',[]))}")
            # 列出 WAR (可避免的)
            wars = deps.get("war", [])
            if wars:
                lines.append("  Avoidable WAR dependencies:")
                for w in wars[:5]:
                    lines.append(f"    op{w.get('from_op','?')} -> op{w.get('to_op','?')} "
                                 f"on '{w.get('buffer','?')}'")

    # Buffers (仅 Tier 2)
    if cfg.include_buffers:
        bufs = merged_report.get("buffers", {})
        if bufs:
            lines.append("")
            lines.append("--- BUFFERS (producers → consumers) ---")
            for name, info in bufs.items():
                producers = info.get("producers", [])
                consumers = info.get("consumers", [])
                region = info.get("region", "?")
                size = info.get("size_kb", 0)
                lines.append(f"  {name:10s} region={region:3s} size={size:6.0f}KB  "
                             f"writers={producers}  readers={consumers}")

    # Critical path
    if cfg.include_critical_path:
        cp = merged_report.get("critical_path", {})
        path = cp.get("path", [])
        if path:
            lines.append("")
            chain = " -> ".join(f"op{i}" for i in path)
            lines.append(f"--- CRITICAL PATH ---")
            lines.append(f"  chain: {chain}")
            lines.append(f"  length_ns={cp.get('length_ns',total_ns):.1f}  "
                         f"fraction={cp.get('fraction','?')}")

    # Parallelism
    if cfg.include_parallelism:
        para = merged_report.get("parallelism", {})
        pairs = para.get("parallel_pairs", para.get("pairs", []))
        lines.append("")
        lines.append(f"--- PARALLELISM ({len(pairs)} pairs) ---")
        for p in pairs[:5]:
            lines.append(f"  op{p.get('op_a','?')} || op{p.get('op_b','?')}  "
                         f"overlap={p.get('overlap_ns',0):.0f}ns")

    # Diagnosis summary (if provided)
    if diagnosis:
        d = diagnosis
        lines.append("")
        lines.append("--- BOTTLENECK DIAGNOSIS ---")
        bn = d.get("bottleneck", d)
        lines.append(f"  op:      op{bn.get('op_id','?')} ({bn.get('op_type','?')}, {bn.get('engine','?')})")
        lines.append(f"  type:    {bn.get('type','?')} ({bn.get('category','?')})")
        lines.append(f"  headroom: {bn.get('headroom','?')}")
        tr = bn.get("time_ratio", 0)
        if isinstance(tr, str):
            try: tr = float(tr.rstrip("%")) / 100.0
            except: tr = 0.0
        lines.append(f"  time_ratio: {float(tr):.2%}")
        bu = bn.get("bw_utilization", 0)
        if isinstance(bu, str):
            try: bu = float(bu.rstrip("%")) / 100.0
            except: bu = 0.0
        if bu:
            lines.append(f"  bw_util:   {float(bu):.2%}")
        lines.append(f"  regime:    {bn.get('regime','?')}")
        strategies = d.get("strategies", d.get("suggested_strategies", []))
        if strategies:
            lines.append(f"  strategies: {strategies}")
        structural = d.get("structural_issues", [])
        if structural:
            lines.append(f"  structural_issues: {structural}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  行过滤
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_ops(ops: list, cp_path: set, cfg: TierExtractConfig) -> list:
    """按 Tier 规则过滤 op 行。"""
    if cfg.op_filter == "all":
        result = list(ops)
    elif cfg.op_filter == "critical_path":
        result = [op for op in ops if op.get("op_id") in cp_path]
        if not result:
            result = list(ops)  # 回退: 没有关键路径数据
    elif cfg.op_filter == "transfer_only":
        result = [op for op in ops
                  if op.get("engine") in ("GM→UB", "UB→GM", "GM→L1", "L1→L0", "L0→GM")]
    elif cfg.op_filter == "compute_only":
        result = [op for op in ops
                  if op.get("engine") in ("VecUnit", "CubeUnit")]
    elif cfg.op_filter == "top10":
        result = sorted(ops, key=lambda o: _safe_float(o.get("time_ratio", 0)), reverse=True)
    else:
        result = list(ops)

    # 截断到 max_ops
    if len(result) > cfg.max_ops:
        result = result[:cfg.max_ops]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  列过滤
# ═══════════════════════════════════════════════════════════════════════════════

def _select_columns(ops: list, columns: List[str]) -> List[dict]:
    """只保留指定的列, 并格式化数值。"""
    result = []
    for op in ops:
        row = {}
        for col in columns:
            val = op.get(col, "")
            # 格式化
            if isinstance(val, float):
                if col in ("time_ratio", "bw_utilization"):
                    val = f"{val:.2%}"
                elif col in ("size_kb", "duration_ns"):
                    val = f"{val:.1f}"
                elif "bw" in col.lower():
                    val = f"{val:.2f}"
                else:
                    val = f"{val:.4g}"
            elif isinstance(val, list):
                if col == "dependencies":
                    parts = []
                    for d in val[:3]:
                        if isinstance(d, dict):
                            oid = d.get("from_op_id", "?")
                            tp = d.get("type", "?")
                            parts.append(f"op{oid}({tp})")
                        elif isinstance(d, (list, tuple)) and len(d) >= 2:
                            parts.append(f"op{d[0]}({d[1]})")
                        else:
                            parts.append(str(d)[:20])
                    val = ",".join(parts)
                elif col == "blocked_by":
                    val = str(val)[:60] if val else ""
                else:
                    val = str(val)[:40] if val else ""
            elif isinstance(val, str):
                val = val[:50]  # 截断长字符串
            else:
                val = str(val)[:20] if val else ""
            row[col] = val
        result.append(row)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  聚合计算
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_aggregated(ops: list, cp_path: set, total_ns: float) -> list:
    """按 op_type + engine 聚合统计。"""
    groups: Dict[str, dict] = {}
    for op in ops:
        key = f"{op.get('op_type')}|{op.get('engine')}"
        if key not in groups:
            groups[key] = {
                "op_type": op.get("op_type"),
                "engine": op.get("engine"),
                "count": 0,
                "total_duration_ns": 0.0,
                "total_time_ratio": 0.0,
                "on_critical_path_count": 0,
                "bw_utils": [],
            }
        g = groups[key]
        g["count"] += 1
        dur = op.get("duration_ns", 0)
        if isinstance(dur, (int, float)):
            g["total_duration_ns"] += dur
            g["total_time_ratio"] += dur / total_ns if total_ns > 0 else 0
        if op.get("op_id") in cp_path:
            g["on_critical_path_count"] += 1
        bw = op.get("bw_utilization", 0)
        if isinstance(bw, (int, float)):
            g["bw_utils"].append(bw)

    result = []
    for g in groups.values():
        if g["bw_utils"]:
            g["avg_bw_util"] = sum(g["bw_utils"]) / len(g["bw_utils"])
        else:
            g["avg_bw_util"] = 0.0
        result.append(g)

    result.sort(key=lambda g: _safe_float(g.get("total_time_ratio", 0)), reverse=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  自测 — 展示 6 个 Tier 各自的提取结果
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    import json
    from pathlib import Path

    outputs_root = Path(__file__).resolve().parent.parent / "outputs"
    merged_file = outputs_root / "vector_add_fp16_N65536" / "round0" / "merged" / "merged_report.json"

    if not merged_file.exists():
        print("[extractor] SKIP: merged_report.json not found. Run dsl_merger.py first.")
        return

    with open(merged_file, encoding="utf-8") as f:
        report = json.load(f)

    # 模拟诊断
    diag = {
        "bottleneck": {"op_id": 2, "op_type": "ub_to_gm", "engine": "UB→GM",
                        "type": "memory_bandwidth", "category": "MEMORY",
                        "headroom": "LOW", "time_ratio": 0.4677,
                        "bw_utilization": 1.0, "regime": "saturated"},
        "strategies": ["reduce_data_volume", "double_buffering"],
        "structural_issues": [],
    }

    print("=" * 60)
    print("DataExtractor — All 6 Tiers")
    print("=" * 60)

    for tier in range(1, 7):
        text = extract(report, diag, tier=tier)
        print(f"\n{'='*60}")
        print(f"TIER {tier} — {len(text)} chars ({len(text.encode())/1024:.1f} KB)")
        print(f"{'='*60}")
        print(text)

    # 验证: 每个 Tier 输出 ≤ 5KB
    for tier in range(1, 7):
        text = extract(report, diag, tier=tier)
        size = len(text.encode())
        assert size < 6000, f"Tier {tier} output too large: {size} bytes"
    print(f"\nAll tiers pass: output < 5KB per tier [OK]")


if __name__ == "__main__":
    _self_test()
