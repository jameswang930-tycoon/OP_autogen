#!/usr/bin/env python3
"""整合 board + task → diagnosis.json (roofline 核心, 只用 msprof + msprof op)。

主流程已弃 hivm/sim → 诊断 = kernel + 引擎 + 通路 三级, roofline 判类型。

结构:
  kernel_summary   每 kernel 耗时/核数/多kernel/launch/L2
  roofline         访存 vs 计算 → memory/compute/latency bound (★一针见血)
  engine_util      各引擎利用率 (cube/vec/mte/scalar/fixpipe)
  transfer_paths   每通路真实带宽
  memory_issues    L2 + UB 冲突
  compute          cube/vec fops
  bottlenecks      每 Tier 处方化 hint

用法: python integrate.py <board.json> <task.json> <out.json>
"""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_schema import read_json, write_json  # noqa: E402

# ── 910B3 理论峰值 (官网/实测) ──
PEAK_MEM_BW_GB_S = 1800.0     # GM 理论 ~1.8 TB/s
PEAK_COMPUTE_TFLOPS = 294.9   # cube fp16: 20核 × 16³ FMA × 1.8GHz

# 通路 → 带宽字段 (board normalized.bandwidth_gb_s)
PATH_BW_KEY = {
    "GM读": "main_mem_read_gb_s", "GM写": "main_mem_write_gb_s",
    "L1": "l1_read_gb_s", "L2": "l2_read_gb_s",
    "UB": "ub_read_gb_s",
    "L0A": "l0a_read_gb_s", "L0B": "l0b_read_gb_s", "L0C": "l0c_read_gb_s",
}


def integrate(board_p, task_p, out_p):
    bd = read_json(board_p)
    tk = read_json(task_p)
    b_norm = bd.get("normalized", {})
    t_norm = tk.get("normalized", {})

    # ── kernel_summary ──
    b_sum = bd.get("execution_summary", {})
    t_sum = tk.get("execution_summary", {})
    total_ns = b_sum.get("total_ns") or t_sum.get("total_ns")
    num_cores = b_sum.get("num_cores") or t_sum.get("num_cores")
    kernels = t_norm.get("kernels", [])
    main_kernel = next((k for k in kernels if k.get("task_duration_us")), None)
    kernel_summary = {
        "kernel_name": b_sum.get("kernel_name") or t_sum.get("kernel_name"),
        "total_ns": total_ns,
        "num_cores": num_cores,
        "freq_mhz": b_sum.get("freq_mhz"),
        "num_kernels": t_sum.get("num_kernels", len(kernels)),
        "input_shapes": main_kernel.get("input_shapes") if main_kernel else None,
        "output_shapes": main_kernel.get("output_shapes") if main_kernel else None,
        "api_overhead_total_us": sum(a.get("total_us") or 0 for a in t_norm.get("api_overhead", [])),
        "l2_hit_rate": b_norm.get("l2_hit_rate") or t_norm.get("l2_hit_rate"),
    }

    # ── roofline ──
    bw = b_norm.get("bandwidth_gb_s", {})
    comp = b_norm.get("compute", {})
    achieved_mem = max([bw.get("main_mem_read_gb_s"), bw.get("main_mem_write_gb_s")],
                       key=lambda x: x or 0) or 0
    cube_fops = comp.get("cube_fops") or 0
    vec_fops = comp.get("vector_fops") or 0
    achieved_compute_tflops = (cube_fops + vec_fops) / 1e12 if (cube_fops or vec_fops) else 0
    mem_util = achieved_mem / PEAK_MEM_BW_GB_S if PEAK_MEM_BW_GB_S else 0
    comp_util = achieved_compute_tflops / PEAK_COMPUTE_TFLOPS if PEAK_COMPUTE_TFLOPS else 0
    if mem_util >= 0.8 and comp_util < 0.5:
        btype = "memory_bound"
    elif comp_util >= 0.8 and mem_util < 0.5:
        btype = "compute_bound"
    elif mem_util < 0.3 and comp_util < 0.3:
        btype = "latency_bound"
    else:
        btype = "balanced"
    roofline = {
        "achieved_memory_bw_gb_s": round(achieved_mem, 1),
        "peak_memory_bw_gb_s": PEAK_MEM_BW_GB_S,
        "memory_utilization": round(mem_util, 3),
        "achieved_compute_tflops": round(achieved_compute_tflops, 2),
        "peak_compute_tflops": PEAK_COMPUTE_TFLOPS,
        "compute_utilization": round(comp_util, 3),
        "arithmetic_intensity": round(achieved_compute_tflops * 1e12 / achieved_mem / 1e9, 2)
                                if achieved_mem else None,
        "bottleneck_type": btype,
    }

    # ── engine_utilization ──
    engine_util = b_norm.get("engine_utilization", {})

    # ── transfer_paths (每通路真实带宽) ──
    transfer_paths = []
    for path, key in PATH_BW_KEY.items():
        v = bw.get(key)
        if v is not None:
            transfer_paths.append({"path": path, "real_bw_gb_s": v,
                                   "source": "Memory/MemoryL0.csv (真机)"})

    # ── memory_issues ──
    conflict = b_norm.get("conflict", {})
    memory_issues = {
        "l2_hit_rate": b_norm.get("l2_hit_rate"),
        "ub_bank_conflict_ratio": conflict.get("aiv_vec_bank_cflt_ratio"),
        "ub_bankgroup_conflict_ratio": conflict.get("aiv_vec_bankgroup_cflt_ratio"),
        "vec_resc_conflict_ratio": conflict.get("aiv_vec_resc_cflt_ratio"),
    }

    # ── compute ──
    compute = {
        "cube_fops": cube_fops, "vector_fops": vec_fops,
        "cube_ratio": comp.get("cube_ratio"), "vec_ratio": comp.get("vec_ratio"),
        "aic_total_cycles": comp.get("aic_total_cycles"),
        "aiv_total_cycles": comp.get("aiv_total_cycles"),
    }

    # ── bottlenecks (每 Tier 处方化 hint) ──
    multi = t_norm.get("multi_kernel", [])
    n_kernels = t_sum.get("num_kernels", len(kernels))
    api_over = kernel_summary["api_overhead_total_us"]
    l2 = b_norm.get("l2_hit_rate")
    bank = memory_issues["ub_bank_conflict_ratio"]
    scalar_r = engine_util.get("scalar")
    mte2_r = engine_util.get("mte2")
    cube_r = engine_util.get("cube")
    vec_r = engine_util.get("vec")

    bottlenecks = {
        "tier1_kernel": {
            "num_kernels": n_kernels, "api_overhead_us": api_over,
            "multi_kernel_top": multi[:5],
            "hint": ("launch/API 开销占比高 → 考虑 Persistent Kernel 减少 launch"
                     if api_over and total_ns and api_over * 1000 / total_ns > 0.2
                     else ("多 kernel → 考虑 kernel 融合"
                           if n_kernels and n_kernels > 1 else "单 kernel, 无 kernel 级优化空间")),
        },
        "tier3_tiling": {
            "bottleneck_type": btype,
            "memory_utilization": roofline["memory_utilization"],
            "compute_utilization": roofline["compute_utilization"],
            "hint": ("访存 bound (带宽利用率高, 算力低) → 增大 tile / 双缓冲 / 减少 GM 流量"
                     if btype == "memory_bound"
                     else ("计算 bound → 优化计算指令 / 提高占用率 / 精度取舍"
                           if btype == "compute_bound"
                           else ("latency bound (两者都低) → 增并行 / 减同步"
                                 if btype == "latency_bound" else "计算访存平衡, 可微调"))),
        },
        "tier4_memory": {
            "l2_hit_rate": l2,
            "bank_conflict_ratio": bank,
            "mte2_ratio": mte2_r,
            "hint": ("L2 命中低 ({}<0.8) → L2 驻留: 切工作集 ≤ 192MB, 改善访问模式".format(l2)
                     if l2 is not None and l2 < 0.8
                     else ("UB bank 冲突高 → 数据对齐 / padding".format(bank)
                           if bank and bank > 0.05
                           else "L2/冲突正常")),
        },
        "tier5_compute": {
            "cube_ratio": cube_r, "vec_ratio": vec_r, "scalar_ratio": scalar_r,
            "hint": ("scalar 占用高 ({:.0%}) → 简化索引/避免 i64 比较退化 scalar".format(scalar_r)
                     if scalar_r and scalar_r > 0.3
                     else ("cube 忙 vec 闲 → 计算-搬运重叠 (双缓冲/pipeline)".format(cube_r, vec_r)
                           if cube_r and vec_r and cube_r > vec_r * 2
                           else "cube/vec 相对均衡")),
        },
        "tier6_arch": {
            "engine_utilization": engine_util,
            "l2_hit_rate": l2,
            "num_cores": num_cores,
            "hint": ("核数使用 ({} 核) vs 理论 (20 AIC) → 检查 grid 是否充分利用".format(num_cores)
                     if num_cores and num_cores < 20
                     else "核数正常; 看 engine_utilization 哪条 pipe 饱和 → 对应优化"),
        },
    }

    report = {
        "meta": {"source": "integrate (msprof+op only)", "generated_at": datetime.now().isoformat(),
                 "inputs": {"board": board_p, "task": task_p}, "schema_version": "3.0"},
        "kernel_summary": kernel_summary,
        "roofline": roofline,
        "engine_utilization": engine_util,
        "transfer_paths": transfer_paths,
        "memory_issues": memory_issues,
        "compute": compute,
        "bottlenecks": bottlenecks,
        "notes": ["只用 msprof + msprof op (hivm/sim 已弃用主流程)",
                  "roofline 是核心: 先判 memory/compute/latency bound, 再细化到通路/引擎",
                  "peak_memory=1.8TB/s, peak_compute=294.9TFLOPS (910B3 理论, 可校准)",
                  "缺: Tier2 内部融合 (需 hivm 依赖); 每 op 精准归因"],
    }
    write_json(report, out_p)
    print(f"[integrate] {out_p}: total_ns={total_ns} bottleneck={btype} "
          f"paths={len(transfer_paths)}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python integrate.py <board.json> <task.json> <out.json>")
        sys.exit(1)
    integrate(sys.argv[1], sys.argv[2], sys.argv[3])
