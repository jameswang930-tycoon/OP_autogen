#!/usr/bin/env python3
"""4 源整合 → diagnosis.json (按优化策略组织, 精准找瓶颈)。

输入: hivm.json (结构) + sim.json (指令时序) + task.json (真机每op指标) + board.json (聚合)
输出: diagnosis.json:
  - summary         端到端/核数/L2/执行模式/引擎占比
  - ops[]           每 op: 结构(hivm) + 指令时序(sim) + 真机带宽/L2(task) + 依赖
  - transfer_paths[] 每通路(引擎): 真实带宽 vs 峰值 → 哪条饱和=瓶颈
  - dependencies[]  RAW/WAR/WAW
  - bottlenecks{}   每 Tier 瓶颈信号 + 优化提示

用法: python integrate.py <hivm.json> <sim.json> <task.json> <board.json> <out.json>
"""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_schema import read_json, write_json  # noqa: E402

# hivm op_type → 通路/引擎
ENGINE_FOR = {
    'gm_to_ub': 'GM→UB', 'ub_to_gm': 'UB→GM',
    'vadd': 'VecUnit', 'vmul': 'VecUnit', 'vsub': 'VecUnit', 'vdiv': 'VecUnit',
    'vmax': 'VecUnit', 'vmin': 'VecUnit', 'vexp': 'VecUnit', 'vlog': 'VecUnit',
    'vabs': 'VecUnit', 'vrelu': 'VecUnit', 'vsqrt': 'VecUnit', 'vtanh': 'VecUnit',
    'vbrc': 'VecUnit', 'vcvt': 'VecUnit', 'vmov': 'VecUnit', 'vsel': 'VecUnit',
    'vcmp': 'VecUnit', 'vdul': 'VecUnit', 'vdup': 'VecUnit', 'vneg': 'VecUnit',
    'gm_to_l1': 'GM→L1', 'l1_to_l0': 'L1→L0',
    'matmul': 'CubeUnit', 'matrixmul': 'CubeUnit', 'mix_matmul': 'CubeUnit',
    'mmadL1': 'CubeUnit', 'batchMmadL1': 'CubeUnit',
    'l0_to_gm': 'L0→GM',
    'set_flag': 'Sync', 'wait_flag': 'Sync', 'pipe_barrier': 'Sync', 'sync_block': 'Sync',
    'fixpipe': 'FixPipe',
}
# 通路 → 从哪搬到哪 (诊断用)
PATH_DESC = {
    'GM→UB': 'GM→UB', 'UB→GM': 'UB→GM', 'GM→L1': 'GM→L1',
    'L1→L0': 'L1→L0', 'L0→GM': 'L0→GM',
    'VecUnit': 'compute(UB)', 'CubeUnit': 'compute(L0C)', 'FixPipe': 'L0C→UB',
    'Sync': 'synchronize',
}
# 通路 → task.json 里的真实带宽字段
PATH_BW_KEY = {
    'GM→UB': 'main_mem_read_bw', 'UB→GM': 'main_mem_write_bw',
    'GM→L1': 'main_mem_read_bw', 'L1→L0': 'l1_read_bw',
    'L0→GM': 'main_mem_write_bw',
    'VecUnit': 'ub_read_bw', 'CubeUnit': 'l0c_read_bw', 'FixPipe': 'l0c_read_bw',
}
# 真机峰值参考 (占位; 用 task 实测最大值校准后替换)
PEAK_GB_S = {'GM→UB': None, 'UB→GM': None, 'GM→L1': None, 'L1→L0': None,
             'L0→GM': None, 'VecUnit': None, 'CubeUnit': None, 'FixPipe': None,
             'Sync': 0.0}


def _bw_gb_s(v):
    """op_summary 带宽值 (可能是 MB/s 或 GB/s) → GB/s (按量级推断)"""
    if v is None:
        return None
    if v >= 1e4:      # 大数值假设是 MB/s
        return round(v / 1000.0, 3)
    return round(v, 3)


def integrate(hivm_p, sim_p, task_p, board_p, out_p):
    hv = read_json(hivm_p)
    sm = read_json(sim_p)
    tk = read_json(task_p)
    bd = read_json(board_p)

    # ── sim 索引 (指令时序 per pipe / sync 关键词) ──
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
    HIVM_PIPE = {'gm_to_ub': 'MTE2', 'ub_to_gm': 'MTE3',
                 'vadd': 'VECTOR', 'vmul': 'VECTOR', 'vbrc': 'VECTOR', 'vcvt': 'VECTOR',
                 'vmov': 'VECTOR', 'vsel': 'VECTOR', 'vcmp': 'VECTOR',
                 'matmul': 'CUBE', 'matrixmul': 'CUBE', 'mix_matmul': 'CUBE',
                 'mmadL1': 'CUBE', 'batchMmadL1': 'CUBE'}
    SYNC_KW = {'set_flag': 'SET_FLAG', 'wait_flag': 'WAIT_FLAG',
               'pipe_barrier': 'BAR', 'sync_block': 'SYNC_BLOCK'}

    # ── per-op: hivm 结构 + sim 时序 + 真机指标 ──
    ops = []
    for m in hv.get("per_op_statistics", []):
        t = m.get("op_type") or ""
        o = {
            "op_id": m.get("op_id"),
            "op_type": t,
            "transfer_path": ENGINE_FOR.get(t, t),
            "path_desc": PATH_DESC.get(ENGINE_FOR.get(t, t), t),
            "dst": m.get("dst"), "src": m.get("src"), "src2": m.get("src2"),
            "dst_region": m.get("memory_region"),
            "size_kb": m.get("size_kb"), "dtype": m.get("dtype"),
            "attrs": m.get("attrs"),
            "duration_ns": m.get("duration_ns"),      # 待 sim 填
            "cycles": m.get("cycles"),
            "pipe": m.get("pipeline_channel"),
            "call_count": m.get("call_count"),
            "real_duration_ns": None,                  # 真机 (task 每 op)
            "real_bw_gb_s": None, "l2_hit": None,
            "dependencies": m.get("dependencies") or [],
        }
        # sim 时序对齐 (引擎/pipe + sync 关键词)
        kw = SYNC_KW.get(t)
        s = sim_by_kw.get(kw) if kw else sim_by_pipe.get(HIVM_PIPE.get(t, ""))
        if s:
            o["duration_ns"] = s.get("duration_ns")
            o["cycles"] = s.get("cycles")
            o["pipe"] = s.get("pipeline_channel")
            o["call_count"] = s.get("call_count")
            o["sim_instr"] = s.get("op_name")
        # 真机指标 (task 主 kernel)
        if tk.get("per_op"):
            top = tk["per_op"][0]
            o["real_duration_ns"] = round((top.get("task_duration_us") or 0) * 1000, 2)
            bw = _bw_gb_s(top.get(PATH_BW_KEY.get(ENGINE_FOR.get(t, ""), "")))
            if bw:
                o["real_bw_gb_s"] = bw
            o["l2_hit"] = top.get("l2_hit_rate")
        ops.append(o)

    # ── transfer_paths: 按引擎聚合真实带宽 ──
    paths = {}
    for o in ops:
        p = o["transfer_path"]
        a = paths.setdefault(p, {
            "path": p, "desc": o["path_desc"], "num_ops": 0,
            "total_size_kb": 0.0, "total_duration_ns": 0.0,
            "real_bw_gb_s": None, "peak_bw_gb_s": None, "bw_utilization": None,
            "regime": None,
        })
        a["num_ops"] += 1
        a["total_size_kb"] += o.get("size_kb") or 0
        a["total_duration_ns"] += o.get("duration_ns") or 0
    # 真机带宽注入 (task 主 kernel 的通路 bw)
    if tk.get("per_op"):
        top = tk["per_op"][0]
        for p, key in PATH_BW_KEY.items():
            if p in paths:
                bw = _bw_gb_s(top.get(key))
                if bw:
                    paths[p]["real_bw_gb_s"] = bw
    # 利用率/regime (peak 校准后才有意义; 无 peak 标 None)
    for a in paths.values():
        peak = PEAK_GB_S.get(a["path"])
        rbw = a["real_bw_gb_s"]
        if peak and rbw:
            a["bw_utilization"] = round(rbw / peak, 3)
            a["regime"] = ('saturated' if a["bw_utilization"] >= 0.95
                           else ('floor' if a["bw_utilization"] <= 0.5 else 'ramp'))
            a["peak_bw_gb_s"] = peak
        if a["total_size_kb"] and a["total_duration_ns"]:
            a["effective_bw_gb_s"] = round(a["total_size_kb"] * 1024 / a["total_duration_ns"], 3)

    # ── dependencies ──
    deps = []
    for o in ops:
        for d in o["dependencies"]:
            deps.append({"from_op": d.get("from_op_id"), "to_op": o["op_id"],
                         "type": d.get("type"), "buffer": d.get("buffer")})

    # ── summary ──
    tk_sum = tk.get("execution_summary", {})
    sm_sum = sm.get("execution_summary", {})
    hv_sum = hv.get("execution_summary", {})
    total_ns = (tk_sum.get("total_ns") or sm_sum.get("total_ns")
                or hv_sum.get("total_ns"))
    num_cores = (tk_sum.get("num_cores") or sm_sum.get("num_cores")
                 or hv_sum.get("num_cores"))
    l2_hit = None
    if tk.get("per_op"):
        l2_hit = tk["per_op"][0].get("l2_hit_rate")
    # 引擎占比: task pipe ratios (真机) 优先
    engine_util = {}
    if tk.get("per_op"):
        top = tk["per_op"][0]
        engine_util = {k: v for k, v in (top.get("pipe_ratios") or {}).items()}
    summary = {
        "total_ns": total_ns, "num_cores": num_cores,
        "kernel_name": tk_sum.get("kernel_name") or hv_sum.get("kernel_name"),
        "execution_mode": sm_sum.get("execution_mode"),
        "l2_hit_rate": l2_hit,
        "engine_utilization": engine_util or bd.get("engine_utilization", {}),
    }

    # ── bottlenecks: 每 Tier 信号 + 提示 ──
    def _path_bottleneck():
        if not paths:
            return None
        # 有真实带宽的路径里, 利用率最高/耗时最大的
        cand = [p for p in paths.values() if p["real_bw_gb_s"]]
        if cand:
            return max(cand, key=lambda p: p.get("bw_utilization") or 0
                       or (p.get("total_duration_ns") or 0))
        return max(paths.values(), key=lambda p: p.get("total_duration_ns") or 0)

    def _top_ops(n=5):
        return sorted(ops, key=lambda o: -(o.get("duration_ns") or 0))[:n]

    pb = _path_bottleneck()
    top = _top_ops()
    war = [d for d in deps if d["type"] == "WAR"]
    raw_vec_chain = sum(1 for o in ops if o["transfer_path"] == "VecUnit" and o.get("duration_ns"))

    bottlenecks = {
        "tier1_algorithm": {
            "execution_mode": summary["execution_mode"],
            "num_ops": len(ops),
            "top_ops": [{"op_id": o["op_id"], "op_type": o["op_type"],
                         "duration_ns": o.get("duration_ns")} for o in top],
            "hint": ("串行执行 + 多 op → 考虑 Persistent Kernel / 减少 launch"
                     if summary["execution_mode"] == "sequential" and len(ops) > 10
                     else "结构检查: 看 top_ops 哪个 op 最耗时"),
        },
        "tier2_fusion": {
            "war_deps": len(war),
            "war_buffers": sorted({d["buffer"] for d in war}),
            "vec_ops": raw_vec_chain,
            "fusion_candidates": [o["op_type"] for o in ops
                                  if o["transfer_path"] == "VecUnit" and o["op_id"] > 0],
            "hint": (f"存在 {len(war)} 条 WAR 依赖 → 分配独立 buffer 可解锁并行"
                     if war else "看 RAW 链上逐元素 op 是否可融合"),
        },
        "tier3_tiling": {
            "path_regimes": {p["path"]: p["regime"] for p in paths.values()},
            "path_effective_bw": {p["path"]: p.get("effective_bw_gb_s")
                                  for p in paths.values()},
            "hint": (f"瓶颈通路 {pb['path']} regime={pb.get('regime')} — "
                     f"{'tile 过小(ramp/floor), 增大 tile 到饱和' if pb.get('regime') in ('ramp','floor') else '已达峰值, 换思路'}"
                     if pb else "需真机带宽校准后判断"),
        },
        "tier4_memory": {
            "l2_hit_rate": l2_hit,
            "small_transfer_ops": [o["op_type"] for o in ops
                                   if (o.get("size_kb") or 0) and (o.get("size_kb") or 0) < 8],
            "hint": (f"L2 命中 {l2_hit} — {'低, 考虑 L2 驻留/改善访问模式' if l2_hit is not None and l2_hit < 0.5 else '正常'}"
                     if l2_hit is not None else "L2 数据需 op_summary 提供"),
        },
        "tier5_compute": {
            "cube_fops": (tk["per_op"][0].get("cube_fops") if tk.get("per_op") else None),
            "vector_fops": (tk["per_op"][0].get("vector_fops") if tk.get("per_op") else None),
            "vec_ops": raw_vec_chain,
            "hint": "看 cube/vec fops 谁大 → 判断算力瓶颈在哪; 结合 sim 看 Vec 是否等 MTE(气泡)",
        },
        "tier6_arch": {
            "block_num": summary["num_cores"],
            "engine_utilization": summary["engine_utilization"],
            "hint": "看哪条 pipe 利用率高/低 → grid 或 pipeline 调整; 20/40 核约束",
        },
    }

    report = {
        "meta": {
            "source": "integrate", "generated_at": datetime.now().isoformat(),
            "inputs": {"hivm": hivm_p, "sim": sim_p, "task": task_p, "board": board_p},
            "schema_version": "2.0",
        },
        "summary": summary,
        "ops": ops,
        "transfer_paths": list(paths.values()),
        "dependencies": deps,
        "bottlenecks": bottlenecks,
        "notes": ["按优化策略组织: 每 Tier 直接看 bottlenecks.<tier> 的 hint",
                  "real_bw 来自真机 op_summary; 峰值校准后 bw_utilization 才有意义",
                  "duration 来自 simulator (per-call); real_duration 来自真机 Task Duration"],
    }
    write_json(report, out_p)
    print(f"[integrate] {out_p}: {len(ops)} ops, {len(paths)} paths, "
          f"{len(deps)} deps, total_ns={summary['total_ns']}")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("用法: python integrate.py <hivm.json> <sim.json> <task.json> <board.json> <out.json>")
        sys.exit(1)
    integrate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
