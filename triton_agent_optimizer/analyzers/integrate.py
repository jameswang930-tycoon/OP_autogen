#!/usr/bin/env python3
"""整合 msprof(骨架) + msprof op(每kernel深层) → 最终 diagnosis.json。

流程 (见 docx/aggregation_rules.md):
  1. task.json = 通用 msprof 骨架 (kernel_slots[]: distinct kernel, task 填满, deep=null)
  2. 每 kernel 一个 board.json = msprof op 深层 (bandwidth/engine/compute/conflict/l2/freq)
  3. 按 kernel 名匹配 → kernels[i].deep 填满, filled_by 标记
  4. roofline 每 kernel 一个 (带宽对1.8TB/s, 算力对294.9TFLOPS)

用法: python integrate.py <task.json> <out.json> [board_*.json...]
"""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_schema import read_json, write_json  # noqa: E402

# ── 910B3 峰值 ──
#   优先用 bench_910b3/hardware_peak.json 实测值 (run_bench.py 生成, 多变体取 max);
#   没有则回退硬编码 (GM 理论 1.8TB/s, cube fp16 294.9 TFLOPS).
def _load_measured_peaks() -> dict:
    import json as _json
    p = Path(__file__).resolve().parent.parent / "bench_910b3" / "hardware_peak.json"
    try:
        if p.exists():
            d = _json.loads(p.read_text(encoding="utf-8"))
            return d.get("peak", {})
    except Exception:
        pass
    return {}

_MEASURED = _load_measured_peaks()
PEAK_MEM_BW_GB_S = float(_MEASURED.get("gm_bw_gb_s", 1800.0))          # 实测 GM 聚合峰值
PEAK_COMPUTE_TFLOPS = float(_MEASURED.get("cube_fp16_tflops", 294.9))  # 实测 cube fp16


def _classify(mem_util, comp_util):
    """roofline 判型: memory/compute/latency/balanced"""
    if mem_util >= 0.8 and comp_util < 0.5:
        return "memory_bound"
    if comp_util >= 0.8 and mem_util < 0.5:
        return "compute_bound"
    if mem_util < 0.3 and comp_util < 0.3:
        return "latency_bound"
    return "balanced"


def build_deep(bd):
    """一个 board.json → deep 子对象 (normalized + 该 kernel 的 roofline)"""
    b_norm = bd.get("normalized", {})
    b_sum = bd.get("execution_summary", {})
    bw = b_norm.get("bandwidth_gb_s", {})
    comp = b_norm.get("compute", {})
    # ★C3: 用 读+写 之和 (原 max 只取单向, 写多读少时访存利用率/算术强度偏低)
    achieved_mem = (bw.get("main_mem_read_gb_s") or 0) + (bw.get("main_mem_write_gb_s") or 0)
    cube_fops = comp.get("cube_fops") or 0
    vec_fops = comp.get("vector_fops") or 0
    achieved_compute = (cube_fops + vec_fops) / 1e12 if (cube_fops or vec_fops) else 0
    mem_util = achieved_mem / PEAK_MEM_BW_GB_S if PEAK_MEM_BW_GB_S else 0
    comp_util = achieved_compute / PEAK_COMPUTE_TFLOPS if PEAK_COMPUTE_TFLOPS else 0
    return {
        "freq_mhz": b_sum.get("freq_mhz"),
        "bandwidth_gb_s": bw,
        "engine_utilization": b_norm.get("engine_utilization", {}),
        "compute": comp,
        "conflict": b_norm.get("conflict", {}),
        "l2_hit_rate": b_norm.get("l2_hit_rate"),
        "roofline": {
            "achieved_memory_bw_gb_s": round(achieved_mem, 1),
            "peak_memory_bw_gb_s": PEAK_MEM_BW_GB_S,
            "memory_utilization": round(mem_util, 3),
            "achieved_compute_tflops": round(achieved_compute, 2),
            "peak_compute_tflops": PEAK_COMPUTE_TFLOPS,
            "compute_utilization": round(comp_util, 3),
            "arithmetic_intensity": round(achieved_compute * 1e12 / achieved_mem / 1e9, 2)
                                 if achieved_mem else None,
            "bottleneck_type": _classify(mem_util, comp_util),
        },
    }


def integrate(task_p, out_p, board_paths):
    tk = read_json(task_p)
    t_norm = tk.get("normalized", {})
    t_sum = tk.get("execution_summary", {})
    slots = t_norm.get("kernel_slots", [])

    # board 按 kernel 名建索引
    deep_map = {}
    for p in board_paths:
        if not Path(p).exists():
            continue
        b = read_json(p)
        name = b.get("execution_summary", {}).get("kernel_name")
        if name:
            deep_map[name] = b

    # 逐个 slot 填 deep (规则 M11: kernel 名匹配)
    filled = 0
    kernels_out = []
    for slot in slots:
        b = deep_map.get(slot.get("kernel_name"))
        if b:
            slot = dict(slot)
            slot["deep"] = build_deep(b)
            slot["filled_by"] = "msprof op"
            filled += 1
        kernels_out.append(slot)

    num_kernels = t_sum.get("num_kernels", len(slots))
    summary = {
        "num_kernels": num_kernels,                      # 优化目标 kernel 数 (非框架)
        "num_kernels_total": t_sum.get("num_kernels_total", num_kernels),  # 含框架 kernel 总数
        "total_ns": t_sum.get("total_ns"),
        "num_cores": t_sum.get("num_cores"),
        "api_overhead_total_us": sum(a.get("total_us") or 0 for a in t_norm.get("api_overhead", [])),
        "l2_hit_rate": t_norm.get("l2_hit_rate"),
        "filled_kernels": filled,
    }

    report = {
        "meta": {"source": "msprof (generic) + msprof op per-kernel",
                 "generated_at": datetime.now().isoformat(),
                 "num_kernels": num_kernels, "filled_kernels": filled,
                 "inputs": {"task": str(task_p), "boards": [str(p) for p in board_paths]},
                 "schema_version": "4.0"},
        "summary": summary,
        "kernels": kernels_out,
        "framework_kernels": t_norm.get("framework_kernels", []),  # torch 框架 kernel (aclnn*), 非优化目标
        "api_overhead": t_norm.get("api_overhead", []),
        "multi_kernel": t_norm.get("multi_kernel", []),
        "notes": ["骨架=通用msprof(task.json); deep=msprof op 按 kernel 名填充 (见 docx/aggregation_rules.md)",
                  "roofline 每 kernel 一个: 带宽对1.8TB/s, 算力对294.9TFLOPS",
                  "filled_by='msprof only' = 该 kernel 没跑到 op (deep=null)",
                  "kernels[].task.transfers 的 bytes 为估算 (每元素每通路搬一次)"],
    }
    write_json(report, out_p)
    print(f"[integrate] {out_p}: kernels={num_kernels} filled={filled}/{num_kernels}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python integrate.py <task.json> <out.json> [board_*.json...]")
        sys.exit(1)
    integrate(sys.argv[1], sys.argv[2], sys.argv[3:])
