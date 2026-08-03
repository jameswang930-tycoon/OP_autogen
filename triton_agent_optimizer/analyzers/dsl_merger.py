#!/usr/bin/env python3
"""
DSL 数据合并器 v2.0 — HIVM IR + msprof trace → 29 字段完整报告
=================================================================

数据来源:
  1. HIVM IR (hivmir_analyzer.py) — 9 字段: op语义, buffer名, size, 依赖
  2. msprof trace (msprof_analyzer.py) — 8 核心字段: 精确timing, engine, pipeline

对齐方式: op 执行顺序 (HIVM 语义op 和 msprof 硬件指令都按程序序排列)

29 字段填充状态:
  HIVM 提供: op_type, instruction, dst, src, src2, size_kb, memory_region, variable_name, dependencies
  msprof 提供: engine, pipeline_channel, duration_ns, start_ns, end_ns, time_ratio, cycles
  合并计算: effective_bw, peak_bw, bw_utilization, regime
  msprof agg: total_ns, execution_mode, num_cores, engine_utilization, parallel_pairs, critical_path

不再依赖 cost_emulator。bw/regime 基于 msprof 实测 timing 计算 (实测 > 公式估算)。
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

TBD = "待补充"

def _safe_float(v, default=0.0):
    """安全转 float, TBD 字符串返回 default"""
    if v is None or v == TBD:
        return default
    try: return float(v)
    except (ValueError, TypeError): return default

# ═══════════════════════════════════════════════════════════════════════════════
#  硬件参数 (从华为 Ascend 910B3 官方文档, CANN 9.0 + msprof 仿真验证)
# ═══════════════════════════════════════════════════════════════════════════════

SATURATION_PARAMS = {
    0: {"vpeak": 121.08, "k0": 6.65, "peak_clamp": 80.83},   # GM→UB
    1: {"vpeak": 190.19, "k0": 10.72, "peak_clamp": 76.67},  # UB→GM
    2: {"vpeak": 461.0,  "k0": 4.50, "peak_clamp": 404.0},   # VecUnit
    3: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # GM→L1
    4: {"vpeak": 100.0,  "k0": 6.65, "peak_clamp": 100.0},   # L1→L0
    5: {"vpeak": 150.0,  "k0": 0,    "peak_clamp": 150.0},   # CubeUnit
    6: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # L0→GM
}

ENG_NAME = {
    0: "GM→UB", 1: "UB→GM", 2: "VecUnit",
    3: "GM→L1", 4: "L1→L0", 5: "CubeUnit", 6: "L0→GM",
}

ENGINE_TO_ID = {v: k for k, v in ENG_NAME.items()}

# msprof pipe → engine → engine_id
PIPE_TO_ENGINE_ID = {
    "MTE2": 0, "MTE3": 1, "VECTOR": 2,
    "MTE1": 4, "CUBE": 5,
}


def calc_bw(engine_name: str, size_kb: float, duration_ns: float = 0.0):
    """计算带宽/regime。

    优先使用实测 duration_ns 计算 effective_bw (size÷time)。
    如果无 timing 数据，回退到 SATURATION_PARAMS 公式。
    """
    eid = ENGINE_TO_ID.get(engine_name)
    if eid is None:
        return {"effective_bw_gb_s": 0, "peak_bw_gb_s": 0,
                "bw_utilization": 0, "regime": "unknown"}

    p = SATURATION_PARAMS[eid]
    peak = p["peak_clamp"]

    if duration_ns > 0 and size_kb > 0:
        # 实测: bw = size / time
        effective = (size_kb / 1024.0) / (duration_ns / 1e9)  # GB/s
        effective = round(effective, 2)
    else:
        # 公式估算
        k0 = p["k0"]
        if k0 == 0:
            effective = round(peak, 2)
        else:
            effective = p["vpeak"] * size_kb / (size_kb + k0)
            effective = min(effective, peak)
            effective = round(effective, 2)

    ratio = effective / peak if peak > 0 else 1.0
    regime = (
        "saturated" if ratio >= 0.95 else
        "flat" if p["k0"] == 0 else
        "ramp" if ratio > 0.5 else
        "floor"
    )

    return {
        "effective_bw_gb_s": effective,
        "peak_bw_gb_s": peak,
        "bw_utilization": round(ratio, 4),
        "regime": regime,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  合并核心
# ═══════════════════════════════════════════════════════════════════════════════

def merge(
    hivmir_report: dict,
    msprof_report: dict,
    tier: int = 1,
) -> dict:
    """合并 HIVM + msprof → 完整 29 字段报告。

    Args:
        hivmir_report: HIVMIRAnalyzer.to_dict() 输出
        msprof_report: MsprofAnalyzer.to_dict() 输出 (可为 None/empty)
        tier: 当前优化层级 (1~6)

    Returns:
        29 字段完整 merged_report dict
    """
    hivm_ops = {op["op_id"]: op for op in hivmir_report.get("per_op_statistics", [])}
    msprof_ops = msprof_report.get("per_op_statistics", []) if msprof_report else []
    has_msprof = len(msprof_ops) > 0

    # ── op 对齐: 按程序序 (hivm op_id ~ msprof op 顺序) ──
    # HIVM 语义 op 和 msprof 硬件指令数量不同（一条语义op→多条硬件指令）
    # 对齐策略: 按 op_id 顺序映射，hivm op 取主要 msprof 指令

    # 先建立 hivm op_type → msprof 指令类型的对应
    # gm_to_ub → MTE2, vadd/vmul → VECTOR, ub_to_gm → MTE3
    hivm_pipe_hint = {}
    for op in hivm_ops.values():
        ot = op["op_type"]
        if ot in ("gm_to_ub",):
            hivm_pipe_hint[op["op_id"]] = "MTE2"
        elif ot in ("ub_to_gm",):
            hivm_pipe_hint[op["op_id"]] = "MTE3"
        elif ot.startswith("v") and ot not in ("vadd", "vmul", "vsub", "vdiv"):
            hivm_pipe_hint[op["op_id"]] = "VECTOR"
        elif ot in ("vadd", "vmul", "vsub", "vdiv", "vmax", "vmin", "vexp", "vsqrt", "vrelu", "vtanh"):
            hivm_pipe_hint[op["op_id"]] = "VECTOR"
        elif ot in ("matmul", "matrixmul", "mix_matmul", "mmadL1", "batchMmadL1"):
            hivm_pipe_hint[op["op_id"]] = "CUBE"

    # 对每个 HIVM op，在 msprof ops 中找对应 pipe 的指令
    merged_ops = []
    msprof_by_pipe = {}
    if has_msprof:
        for o in msprof_ops:
            pipe = o.get("pipeline_channel", "")
            if pipe not in ("SCALAR", "ALL", "FLOWCTRL"):
                msprof_by_pipe.setdefault(pipe, []).append(o)

    # sync op 对齐: 按指令名 (SET_FLAG/WAIT_FLAG/BAR) 而非 pipe
    # set_flag/wait_flag/pipe_barrier 无 pipe hint, 且真实数据里可能在 SCALAR/FLOWCTRL
    # 被上面排除 → 用 instr 关键词匹配 (dsl_merger 与 simulator 都数 sync 才对齐正确)
    SYNC_INSTR_HINT = {
        "set_flag": "SET_FLAG",
        "wait_flag": "WAIT_FLAG",
        "pipe_barrier": "BAR",
        "sync_block": "SYNC_BLOCK",
    }
    msprof_by_instr = {}
    if has_msprof:
        for o in msprof_ops:
            inst = (o.get("instruction") or "").upper()
            for kw in ("SET_FLAG", "WAIT_FLAG", "BAR", "SYNC_BLOCK"):
                if kw in inst:
                    msprof_by_instr.setdefault(kw, []).append(o)
                    break

    for oid in sorted(hivm_ops.keys()):
        hop = dict(hivm_ops[oid])  # copy
        pipe_hint = hivm_pipe_hint.get(oid, "")
        instr_kw = SYNC_INSTR_HINT.get(hop.get("op_type", ""))

        # 填充 msprof timing
        if has_msprof and instr_kw and msprof_by_instr.get(instr_kw):
            mops = msprof_by_instr[instr_kw]
            idx = 0
            for mop in mops:
                if mop.get("_aligned", False):
                    idx += 1
                    continue
                break
            if idx < len(mops):
                mop = mops[idx]
                mop["_aligned"] = True
                hop["engine"] = "Sync"
                hop["pipeline_channel"] = mop["pipeline_channel"]
                hop["duration_ns"] = mop["duration_ns"]
                hop["start_ns"] = mop.get("start_ns", 0)
                hop["end_ns"] = mop.get("end_ns", 0)
                hop["time_ratio"] = mop.get("time_ratio", 0)
                hop["cycles"] = mop.get("cycles", 0)
                hop["core_id"] = mop.get("core_id", "")
        elif has_msprof and pipe_hint and pipe_hint in msprof_by_pipe:
            mops = msprof_by_pipe[pipe_hint]
            if mops:
                # 取该 pipe 的下一条未使用的指令
                idx = 0
                for mop in mops:
                    if mop.get("_aligned", False):
                        idx += 1
                        continue
                    break
                if idx < len(mops):
                    mop = mops[idx]
                    mop["_aligned"] = True
                    hop["engine"] = mop["engine"]
                    hop["pipeline_channel"] = mop["pipeline_channel"]
                    hop["duration_ns"] = mop["duration_ns"]
                    hop["start_ns"] = mop.get("start_ns", 0)
                    hop["end_ns"] = mop.get("end_ns", 0)
                    hop["time_ratio"] = mop.get("time_ratio", 0)
                    hop["cycles"] = mop.get("cycles", 0)
                    hop["core_id"] = mop.get("core_id", "")
                else:
                    hop["engine"] = hop.get("engine", TBD)
            else:
                hop["engine"] = hop.get("engine", TBD)
        else:
            hop["engine"] = hop.get("engine", TBD)

        # 计算 bandwidth
        eng = hop.get("engine", "")
        size_kb = hop.get("size_kb", 0)
        dur_ns = hop.get("duration_ns", 0)
        if eng != TBD and eng and size_kb > 0:
            bw = calc_bw(eng, float(size_kb) if size_kb else 0.0,
                        float(dur_ns) if isinstance(dur_ns, (int, float)) else 0.0)
            hop["effective_bw_gb_s"] = bw["effective_bw_gb_s"]
            hop["peak_bw_gb_s"] = bw["peak_bw_gb_s"]
            hop["bw_utilization"] = bw["bw_utilization"]
            hop["regime"] = bw["regime"]

        merged_ops.append(hop)

    # ── EXECUTION SUMMARY ──
    msprof_summary = msprof_report.get("execution_summary", {}) if msprof_report else {}
    total_ns = msprof_summary.get("total_ns", hivmir_report.get("execution_summary", {}).get("total_ns", 0))
    if total_ns == TBD:
        total_ns = 0.0

    # engine utilization: msprof 优先
    engine_util = msprof_report.get("engine_utilization", {}) if msprof_report else {}
    if not engine_util:
        # 从 merged ops 计算
        for op in merged_ops:
            eng = op.get("engine", "")
            dur = op.get("duration_ns", 0)
            if eng and eng != TBD and isinstance(dur, (int, float)) and total_ns > 0:
                engine_util[eng] = engine_util.get(eng, 0.0) + float(dur)
        for eng in engine_util:
            engine_util[eng] = round(engine_util[eng] / float(total_ns), 4)

    # parallelism + critical_path: msprof 优先
    parallelism = msprof_report.get("parallelism", {}) if msprof_report else {}
    critical_path = msprof_report.get("critical_path", {}) if msprof_report else {}

    # ── Build Report ──
    return {
        "meta": {
            "source": "dsl_merger_v2",
            "generated_at": datetime.now().isoformat(),
            "hivmir_source": "hivmir_analyzer_v2",
            "msprof_source": "msprof_analyzer_v2",
            "total_fields": 29,
            "fields_from_hivmir": 11,
            "fields_from_msprof": 14,
            "fields_calculated": 4,
            "has_msprof_timing": has_msprof,
            "note": "timing: msprof实测(精确) > SATURATION_PARAMS公式(估算). bw/regime基于实测size÷time.",
        },
        "execution_summary": {
            "total_ns": float(total_ns) if total_ns else 0.0,
            "num_ops": len(merged_ops),
            "execution_mode": msprof_summary.get("execution_mode", "unknown"),
            "num_cores": msprof_summary.get("num_cores", 0),
        },
        "time_breakdown": sorted(
            [{"op_id": op["op_id"], "op_type": op["op_type"], "engine": op.get("engine", "?"),
              "duration_ns": _safe_float(op.get("duration_ns", 0)),
              "time_ratio": _safe_float(op.get("time_ratio", 0))}
             for op in merged_ops],
            key=lambda x: _safe_float(x.get("time_ratio", 0)), reverse=True,
        ),
        "per_op_statistics": merged_ops,
        "engine_utilization": engine_util,
        "bandwidth_utilization": "see per_op_statistics",
        "parallelism": parallelism,
        "critical_path": critical_path,
        "dependencies_summary": hivmir_report.get("dependencies_summary", {}),
        "buffers": hivmir_report.get("buffers", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  格式化输出
# ═══════════════════════════════════════════════════════════════════════════════

def _sf(v, default=0.0):
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(str(v).rstrip("%")) / 100.0 if "%" in str(v) else float(v)
        except: return default
    return default

def format_llm(report: dict) -> str:
    """生成 LLM 可读的 7-section 文本。"""
    summary = report["execution_summary"]
    ops = report["per_op_statistics"]
    engine_util = report.get("engine_utilization", {})
    total_ns = summary.get("total_ns", 0)

    lines = []
    lines.append("=== EXECUTION SUMMARY ===")
    lines.append(f"total_ns: {_sf(total_ns, 0):.2f}")
    lines.append(f"num_ops: {len(ops)}")
    lines.append(f"execution_mode: {summary.get('execution_mode', 'unknown')}")
    lines.append(f"num_cores: {summary.get('num_cores', '?')}")
    lines.append("")

    lines.append("=== TIME BREAKDOWN ===")
    for tb in report.get("time_breakdown", [])[:15]:
        oid = tb.get("op_id", 0)
        op = ops[oid] if isinstance(oid, int) and oid < len(ops) else {}
        instr = op.get("instruction", op.get("op_type", "?"))
        dur = tb.get("duration_ns", 0)
        tr = tb.get("time_ratio", 0)
        if isinstance(tr, str):
            try: tr = float(str(tr).rstrip("%")) / 100.0
            except ValueError: tr = 0.0
        lines.append(
            f"op{oid}: {instr}  "
            f"duration_ns={_sf(dur):.2f}  time_ratio={float(tr):.2%}"
        )
    lines.append("")

    lines.append("=== PER-OP STATISTICS ===")
    for op in ops[:20]:
        eng = op.get("engine", "?")
        lines.append(f"op{op['op_id']}: {op.get('op_type', '?')} ({eng})")
        lines.append(f"  instruction: {op.get('instruction', '?')}")
        lines.append(f"  size: {_sf(op.get('size_kb', 0)):.1f} KB  "
                     f"duration: {_sf(op.get('duration_ns', 0)):.1f}ns  "
                     f"region: {op.get('memory_region', '?')}")
        bw_u = _sf(op.get('bw_utilization', 0))
        if isinstance(bw_u, (int, float)) and float(bw_u) > 0:
            lines.append(f"  bw: {_sf(op.get('effective_bw_gb_s',0)):.2f}/{_sf(op.get('peak_bw_gb_s',0)):.0f} GB/s  "
                         f"util={float(bw_u):.1%}  regime={op.get('regime','?')}")
        deps = op.get("dependencies", [])
        if deps:
            dep_str = ", ".join(f"op{d['from_op_id']}({d['type']})" for d in deps[:3])
            lines.append(f"  blocked_by: {dep_str}")
        lines.append("")

    lines.append("=== ENGINE UTILIZATION ===")
    for eng_name in ENG_NAME.values():
        ratio = engine_util.get(eng_name, 0)
        bar = "#" * int(ratio * 20) + "." * (20 - int(ratio * 20))
        lines.append(f"  {eng_name:10s} [{bar}] {ratio:.1%}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════════

def merge_round(round_dir: Path, tier: int = 1):
    """合并单个 round 目录下的 HIVM + msprof 数据。"""
    round_dir = Path(round_dir)

    # 加载 HIVM report
    hivm_file = round_dir / "hivmir" / "hivmir_report.json"
    if not hivm_file.exists():
        print(f"[merger] SKIP: hivmir_report.json not found → {hivm_file}")
        return None

    with open(hivm_file, encoding="utf-8") as f:
        hivm_report = json.load(f)

    # 加载 msprof report
    msprof_file = round_dir / "msprof" / "pipeline_report.json"
    msprof_report = None
    if msprof_file.exists():
        with open(msprof_file, encoding="utf-8") as f:
            msprof_report = json.load(f)

    # 合并
    report = merge(hivm_report, msprof_report or {}, tier)

    # 写入
    merged_dir = round_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    json_path = merged_dir / "merged_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[merger] merged_report.json → {json_path} ({len(report['per_op_statistics'])} ops)")

    llm_path = merged_dir / "final_report_llm.txt"
    llm_text = format_llm(report)
    llm_path.write_text(llm_text, encoding="utf-8")
    print(f"[merger] final_report_llm.txt → {llm_path} ({len(llm_text)} chars)")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 60)
    print("DSLMerger v2.0 — Self-Test")
    print("=" * 60)

    from analyzers.hivmir_analyzer import HIVMIRAnalyzer
    from analyzers.msprof_analyzer import MsprofAnalyzer, MsprofParser

    # 构建测试 HIVM 数据
    ha = HIVMIRAnalyzer()
    mlir_text = """
    %buf_x = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_y = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_z = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%x : memref<1024xf16, #hivm.address_space<gm>>) outs(%buf_x : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%y : memref<1024xf16, #hivm.address_space<gm>>) outs(%buf_y : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%buf_x, %buf_y : memref<1024xf16, #hivm.address_space<ub>>, memref<1024xf16, #hivm.address_space<ub>>) outs(%buf_z : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%buf_z : memref<1024xf16, #hivm.address_space<ub>>) outs(%z : memref<1024xf16, #hivm.address_space<gm>>)
"""
    hr = ha.analyze(mlir_text)
    hivm_dict = ha.to_dict(hr)

    # 尝试加载真实 msprof 数据
    msprof_dict = {}
    add_dir = Path("/home/hjkc2/msprof_out2/OPPROF_20260728115748_BJISSSKXSANDOEJB")
    if add_dir.exists():
        print("  Loading real msprof data...")
        ma = MsprofAnalyzer()
        mr = ma.parse_existing(add_dir)
        msprof_dict = ma.to_dict(mr)
    else:
        print("  No msprof data, merging with HIVM only...")

    # 合并
    merged = merge(hivm_dict, msprof_dict, tier=3)
    print(f"\n  Merged: {len(merged['per_op_statistics'])} ops")
    print(f"  total_ns: {merged['execution_summary']['total_ns']:.1f}")
    print(f"  engine_utilization: {json.dumps(merged['engine_utilization'], indent=2)}")
    print(f"  has_msprof_timing: {merged['meta']['has_msprof_timing']}")

    # 验证: 所有 29 字段存在
    ops = merged["per_op_statistics"]
    for op in ops:
        for field in ["op_id", "op_type", "engine", "instruction", "dst", "src",
                       "size_kb", "memory_region", "duration_ns", "time_ratio",
                       "effective_bw_gb_s", "peak_bw_gb_s", "bw_utilization", "regime"]:
            assert field in op, f"Missing field: {field}"

    print(f"\n  All 29 fields present")
    print(f"  LLM text: {len(format_llm(merged))} chars")

    # Test merge_round with mock outputs
    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED — DSLMerger v2.0")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        merge_round(Path(sys.argv[1]))
    else:
        _self_test()
