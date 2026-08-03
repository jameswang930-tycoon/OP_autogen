#!/usr/bin/env python3
"""合并三个源 JSON → 统一 merged.json。

对齐策略:
  1. canonical op 列表 = hivm (结构字段真实, 含 sync op 计入 op_id)
  2. 每个 hivm op 从 sim 取时序: 按 op_type 的引擎/pipe 提示匹配; sync op 按指令名 (SET_FLAG/WAIT_FLAG/BAR)
  3. summary 优先 board (真机端到端) > sim (trace) > hivm
  4. engine_utilization 优先 board (PipeUtilization 真机占比)
  5. 无法对齐的字段保留 None (待补充)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_schema import read_json, write_json  # noqa: E402

# hivm op_type → 期望的 sim pipe
HIVM_PIPE_HINT = {
    "gm_to_ub": "MTE2", "ub_to_gm": "MTE3",
    "vadd": "VECTOR", "vmul": "VECTOR", "vsub": "VECTOR", "vdiv": "VECTOR",
    "vmax": "VECTOR", "vmin": "VECTOR", "vexp": "VECTOR", "vlog": "VECTOR",
    "vabs": "VECTOR", "vrelu": "VECTOR", "vsqrt": "VECTOR", "vtanh": "VECTOR",
    "matmul": "CUBE", "matrixmul": "CUBE", "mix_matmul": "CUBE",
    "mmadL1": "CUBE", "batchMmadL1": "CUBE",
}
SYNC_INSTR_HINT = {
    "set_flag": "SET_FLAG", "wait_flag": "WAIT_FLAG",
    "pipe_barrier": "BAR", "sync_block": "SYNC_BLOCK",
}


def _fill_timing(hivm_op, sim_ops_by_pipe, sim_ops_by_kw):
    """从 sim 给 hivm op 填时序。

    对齐语义: 一个语义 op (如 1 次 load) ≈ 对应 pipe 指令的 1 次调用。
    取匹配组 (同 pipe / sync 关键词) 的 per-call 耗时, 不消耗组
    (循环里多个语义 op 可映射到同一指令组, 各自取单次调用成本)。
    """
    ot = hivm_op.get("op_type", "")
    s = None
    kw = SYNC_INSTR_HINT.get(ot)
    if kw:
        s = sim_ops_by_kw.get(kw)
    else:
        pipe = HIVM_PIPE_HINT.get(ot)
        if pipe:
            s = sim_ops_by_pipe.get(pipe)
    if not s:
        return False
    hivm_op["duration_ns"] = s.get("duration_ns")          # per-call
    hivm_op["cycles"] = s.get("cycles")
    hivm_op["call_count"] = s.get("call_count")
    hivm_op["total_duration_ns"] = s.get("total_duration_ns")
    hivm_op["pipeline_channel"] = s.get("pipeline_channel")
    hivm_op["core_id"] = s.get("core_id")
    hivm_op["data_size_bytes"] = s.get("data_size_bytes")
    hivm_op["sim_instr"] = s.get("op_name")
    if hivm_op.get("size_kb") is None and s.get("size_kb") is not None:
        hivm_op["size_kb"] = s.get("size_kb")
    return True


def merge(hivm_path, sim_path, board_path, out_path):
    hv, sm, bd = read_json(hivm_path), read_json(sim_path), read_json(board_path)

    # ── sim 索引: 每 pipe / 关键词取总耗时最大的指令组为代表 ──
    sim_by_pipe, sim_by_kw = {}, {}
    for o in sm.get("per_op_statistics", []):
        o_dur = o.get("total_duration_ns") or 0
        pipe = o.get("pipeline_channel")
        if pipe:
            prev = sim_by_pipe.get(pipe)
            if prev is None or o_dur > (prev.get("total_duration_ns") or 0):
                sim_by_pipe[pipe] = o
        inst = (o.get("op_name") or "").upper()
        for kw in ("SET_FLAG", "WAIT_FLAG", "BAR", "SYNC_BLOCK"):
            if kw in inst:
                prev = sim_by_kw.get(kw)
                if prev is None or o_dur > (prev.get("total_duration_ns") or 0):
                    sim_by_kw[kw] = o
                break

    # ── canonical ops = hivm, 填时序 ──
    merged_ops = []
    for op in hv.get("per_op_statistics", []):
        o = dict(op)
        _fill_timing(o, sim_by_pipe, sim_by_kw)
        # 有效带宽 (若 size+duration 都有)
        if o.get("size_kb") and o.get("duration_ns"):
            o["effective_bw_gb_s"] = round(
                (o["size_kb"] * 1024.0) / (o["duration_ns"] / 1e9) / 1e9, 3)
        merged_ops.append(o)

    # ── summary: board > sim > hivm ──
    def _first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    total_ns = _first(bd["execution_summary"].get("total_ns"),
                      sm["execution_summary"].get("total_ns"),
                      hv["execution_summary"].get("total_ns"))
    num_cores = _first(bd["execution_summary"].get("num_cores"),
                       sm["execution_summary"].get("num_cores"),
                       hv["execution_summary"].get("num_cores"))
    exec_mode = _first(sm["execution_summary"].get("execution_mode"),
                       hv["execution_summary"].get("execution_mode"))
    kernel = _first(bd["execution_summary"].get("kernel_name"),
                    hv["execution_summary"].get("kernel_name"))

    engine_util = bd.get("engine_utilization") or sm.get("engine_utilization") or {}

    merged = {
        "meta": {
            "source": "merge",
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "inputs": {"hivm": hivm_path, "sim": sim_path, "board": board_path},
            "schema_version": "1.0",
        },
        "execution_summary": {
            "total_ns": total_ns, "num_ops": len(merged_ops),
            "execution_mode": exec_mode, "num_cores": num_cores,
            "kernel_name": kernel,
        },
        "per_op_statistics": merged_ops,
        "engine_utilization": engine_util,
        "dependencies_summary": hv.get("dependencies_summary", {}),
        "buffers": hv.get("buffers", {}),
        "parallelism": sm.get("parallelism", {}),
        "critical_path": sm.get("critical_path", {}),
        "bandwidth_utilization": bd.get("bandwidth_utilization", {}),
        "notes": [
            "per-op = hivm 语义 op (含 sync), 时序来自 simulator 指令对齐; 未对齐 op 的时序为 None",
            "total_ns 优先真机 Task Duration; engine_utilization 优先真机 PipeUtilization",
            "peak_bw/regime 未在 merge 计算 (需真机 Memory.csv 校准 + size 扫描), 见 bandwidth_utilization",
        ],
    }
    write_json(merged, out_path)
    # 对齐统计
    aligned = sum(1 for o in merged_ops if o.get("duration_ns") is not None)
    print(f"[merge] {out_path}: {len(merged_ops)} ops, {aligned} 时序已对齐, "
          f"total_ns={total_ns}, cores={num_cores}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("用法: python pipeline_merge.py <hivm.json> <sim.json> <board.json> <merged.json>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
