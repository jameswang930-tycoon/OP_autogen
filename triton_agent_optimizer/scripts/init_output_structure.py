#!/usr/bin/env python3
"""初始化 outputs/ 完整目录结构 + 模拟内容。"""

import json, shutil
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


# ═══════════════════════════════════════════════════════════════════════════════
#  模拟数据工厂
# ═══════════════════════════════════════════════════════════════════════════════

def make_msprof_report():
    return {
        "meta": {"source": "msprof_op_simulator", "fields_filled": 16,
                 "fields_pending_hivmir": 13,
                 "note": "msprof提供timing/engine/channel, HIVMIR字段标记为待补充"},
        "execution_summary": {"total_ns": 3655.6, "num_ops": 3,
                              "execution_mode": "sequential", "num_cores": 1},
        "time_breakdown": [
            {"op_id": 2, "op_type": "ub_to_gm", "engine": "UB→GM",
             "duration_ns": 1709.6, "time_ratio": 0.4677},
            {"op_id": 0, "op_type": "gm_to_ub", "engine": "GM→UB",
             "duration_ns": 1621.6, "time_ratio": 0.4436},
            {"op_id": 1, "op_type": "vadd", "engine": "VecUnit",
             "duration_ns": 324.4, "time_ratio": 0.0888},
        ],
        "per_op_statistics": [
            {"op_id": 0, "op_type": "gm_to_ub", "engine": "GM→UB",
             "pipeline_channel": "MTE2", "core_id": "core0.veccore0",
             "instruction": "待补充", "dst": "待补充", "src": "待补充",
             "size_kb": "待补充", "variable_name": "待补充", "memory_region": "待补充",
             "duration_ns": 1621.6, "start_ns": 0.0, "end_ns": 1621.6,
             "time_ratio": 0.4436, "effective_bw_gb_s": "待补充",
             "peak_bw_gb_s": "待补充", "bw_utilization": "待补充",
             "regime": "待补充", "wait_before_start_ns": "待补充", "blocked_by": []},
            {"op_id": 1, "op_type": "vadd", "engine": "VecUnit",
             "pipeline_channel": "VECTOR", "core_id": "core0.veccore0",
             "instruction": "待补充", "dst": "待补充", "src": "待补充",
             "size_kb": "待补充", "variable_name": "待补充", "memory_region": "待补充",
             "duration_ns": 324.4, "start_ns": 1621.6, "end_ns": 1946.0,
             "time_ratio": 0.0888, "effective_bw_gb_s": "待补充",
             "peak_bw_gb_s": "待补充", "bw_utilization": "待补充",
             "regime": "待补充", "wait_before_start_ns": "待补充", "blocked_by": []},
            {"op_id": 2, "op_type": "ub_to_gm", "engine": "UB→GM",
             "pipeline_channel": "MTE3", "core_id": "core0.veccore0",
             "instruction": "待补充", "dst": "待补充", "src": "待补充",
             "size_kb": "待补充", "variable_name": "待补充", "memory_region": "待补充",
             "duration_ns": 1709.6, "start_ns": 1946.0, "end_ns": 3655.6,
             "time_ratio": 0.4677, "effective_bw_gb_s": "待补充",
             "peak_bw_gb_s": "待补充", "bw_utilization": "待补充",
             "regime": "待补充", "wait_before_start_ns": "待补充", "blocked_by": []},
        ],
        "engine_utilization": {"GM→UB": 0.44, "UB→GM": 0.47, "VecUnit": 0.09,
                               "GM→L1": 0, "L1→L0": 0, "CubeUnit": 0, "L0→GM": 0},
        "bandwidth_utilization": "待补充",
        "parallelism": {"parallel_pairs": [], "total_pairs": 0,
                        "root_cause": "RAW chain op0→op1→op2 (待HIVMIR确认)"},
        "critical_path": {"path": [0, 1, 2], "length_ns": 3655.6,
                          "fraction": "100%", "edges": []},
    }


def make_hivmir_report():
    return {
        "meta": {"source": "hivmir_compiler_trace", "fields_filled": 9,
                 "fields_pending_msprof": 16,
                 "note": "HIVMIR提供buffer名/size/依赖, msprof字段标记为待补充"},
        "execution_summary": {"total_ns": "待补充", "num_ops": 3,
                              "execution_mode": "待补充"},
        "per_op_statistics": [
            {"op_id": 0, "op_type": "gm_to_ub", "engine": "GM→UB",
             "instruction": "gm_to_ub(ub_1, gm_1)", "dst": "ub_1", "src": "gm_1",
             "src2": "", "size_kb": 128.0, "memory_region_dst": "UB",
             "memory_region_src": "GM", "variable_name": "ub_1",
             "dependencies": [],
             "duration_ns": "待补充", "start_ns": "待补充", "end_ns": "待补充",
             "time_ratio": "待补充", "bw_utilization": "待补充", "regime": "待补充"},
            {"op_id": 1, "op_type": "vadd", "engine": "VecUnit",
             "instruction": "vadd(ub_2, ub_1, 2.0)", "dst": "ub_2", "src": "ub_1",
             "src2": "", "scalar": 2.0, "size_kb": 128.0,
             "memory_region_dst": "UB", "memory_region_src": "UB",
             "variable_name": "ub_2",
             "dependencies": [{"from_op_id": 0, "type": "RAW"}],
             "duration_ns": "待补充", "start_ns": "待补充", "end_ns": "待补充",
             "time_ratio": "待补充", "bw_utilization": "待补充", "regime": "待补充"},
            {"op_id": 2, "op_type": "ub_to_gm", "engine": "UB→GM",
             "instruction": "ub_to_gm(gm_2, ub_2)", "dst": "gm_2", "src": "ub_2",
             "src2": "", "size_kb": 128.0, "memory_region_dst": "GM",
             "memory_region_src": "UB", "variable_name": "gm_2",
             "dependencies": [{"from_op_id": 1, "type": "RAW"}],
             "duration_ns": "待补充", "start_ns": "待补充", "end_ns": "待补充",
             "time_ratio": "待补充", "bw_utilization": "待补充", "regime": "待补充"},
        ],
        "buffers": {
            "gm_1": {"region": "GM", "size_kb": 128, "producers": [], "consumers": [0]},
            "ub_1": {"region": "UB", "size_kb": 128, "producers": [0], "consumers": [1]},
            "ub_2": {"region": "UB", "size_kb": 128, "producers": [1], "consumers": [2]},
            "gm_2": {"region": "GM", "size_kb": 128, "producers": [2], "consumers": []},
        },
        "dependencies_summary": {
            "total": 2,
            "raw": [{"from_op": 0, "to_op": 1, "buffer": "ub_1"},
                    {"from_op": 1, "to_op": 2, "buffer": "ub_2"}],
            "war": [], "waw": [],
        },
        "engine_utilization": "待补充",
        "bandwidth_utilization": "待补充",
        "critical_path": "待补充",
    }


FINAL_LLM_TXT = """=== EXECUTION SUMMARY ===
total_ns: 3655.57
num_ops: 3
execution_mode: sequential

=== TIME BREAKDOWN ===
(time_ratio = op duration / total_ns, sorted biggest first)
op2: ub_to_gm(gm_2, ub_2)  duration_ns=1709.56  time_ratio=46.77%  (1709.56/3655.57 ns)
op0: gm_to_ub(ub_1, gm_1)  duration_ns=1621.58  time_ratio=44.36%  (1621.58/3655.57 ns)
op1: vadd(ub_2, ub_1, 2.0)  duration_ns=324.44  time_ratio=8.88%  (324.44/3655.57 ns)

=== PER-OP STATISTICS ===
op0: gm_to_ub(ub_1, gm_1)
  engine: GM→UB
  size: 128 KB
  cycles_ns: [0.00..1621.58]  duration_ns=1621.58  time_ratio=44.36%
  bandwidth: effective=80.83 GB/s  peak=80.83 GB/s  utilization=100.00%  regime=saturated
  wait_ns_before_start: 0.00
  blocked_by: none

op1: vadd(ub_2, ub_1, 2.0)
  engine: VecUnit
  size: 128 KB
  cycles_ns: [1621.58..1946.01]  duration_ns=324.44  time_ratio=8.88%
  bandwidth: effective=404 GB/s  peak=404 GB/s  utilization=100.00%  regime=saturated
  wait_ns_before_start: 1621.58
  blocked_by: op0 via RAW on 'ub_1' -- ub_1 written by op0, read by op1

op2: ub_to_gm(gm_2, ub_2)
  engine: UB→GM
  size: 128 KB
  cycles_ns: [1946.01..3655.57]  duration_ns=1709.56  time_ratio=46.77%
  bandwidth: effective=76.67 GB/s  peak=76.67 GB/s  utilization=100.00%  regime=saturated
  wait_ns_before_start: 1946.01
  blocked_by: op1 via RAW on 'ub_2' -- ub_2 written by op1, read by op2

=== ENGINE UTILIZATION ===
GM→UB: busy=1621.58/3655.57 ns  utilization=44.36%
UB→GM: busy=1709.56/3655.57 ns  utilization=46.77%
VecUnit: busy=324.44/3655.57 ns  utilization=8.88%
GM→L1: busy=0.00/3655.57 ns  utilization=0.00%

=== BANDWIDTH UTILIZATION ===
(effective_bw / peak_bw per op; bandwidth ramps with size)
op0 (GM→UB): effective=80.83 GB/s  peak=80.83 GB/s  utilization=100.00%  regime=saturated  k0=6.7KB
op1 (VecUnit): effective=404 GB/s  peak=404 GB/s  utilization=100.00%  regime=saturated  k0=4.5KB
op2 (UB→GM): effective=76.67 GB/s  peak=76.67 GB/s  utilization=100.00%  regime=saturated  k0=10.7KB

=== PARALLELISM ===
parallel_pairs: 0
root_cause_of_sequential_execution:
  op0->op1: RAW on 'ub_1'
  op1->op2: RAW on 'ub_2'

=== CRITICAL PATH ===
algorithm: topo
length_ns: 3655.57
fraction_of_makespan: 100%
path: op0 -> op1 -> op2
edges:
  op0 -> op1: RAW on 'ub_1'
  op1 -> op2: RAW on 'ub_2'
per_op:
  op0 gm_to_ub(ub_1, gm_1)  engine=GM→UB  ns=[0.00..1621.58]  duration_ns=1621.58
  op1 vadd(ub_2, ub_1, 2.0)  engine=VecUnit  ns=[1621.58..1946.01]  duration_ns=324.44
  op2 ub_to_gm(gm_2, ub_2)  engine=UB→GM  ns=[1946.01..3655.57]  duration_ns=1709.56
"""

FINAL_HUMAN_TXT = """  ┌─ Pipeline Execution Graph ───────────────────────────────────────────────┐

  Engine bandwidth (peak, per core):
    GM→UB    81 GB/s (vpeak=121, k0=6.7KB)     UB→GM    77 GB/s (vpeak=190, k0=10.7KB)
    VecUnit 404 GB/s (vpeak=461, k0=4.5KB)     GM→L1    38 GB/s (placeholder)
    L1→L0  100 GB/s (placeholder)               CubeUnit 150 GB/s (placeholder)
    L0→GM   38 GB/s (placeholder)

  Time axis: 90 cols ≈ 3.66 µs makespan (40.6 ns/col)
           ────────────────────────────────────────────────────────────────────────
    GM→UB      │ op0█████████████████████████████████████······························
    UB→GM      │ ················································op2███████████████████
    VecUnit    │ ········································op1█████······················
    GM→L1      │ ······································································
    L1→L0      │ ······································································
    CubeUnit   │ ······································································
    L0→GM      │ ······································································
           ────────────────────────────────────────────────────────────────────────

  Op  Instruction                Engine   Size     Time (ns)          BW util  Waits for
  ─────────────────────────────────────────────────────────────────────────────────────
   0  gm_to_ub(ub_1, gm_1)       GM→UB   128 KB   [0.0..1621.6]      100%    —
   1  vadd(ub_2, ub_1, 2.0)      VecUnit 128 KB   [1621.6..1946.0]   100%    op0(RAW)
   2  ub_to_gm(gm_2, ub_2)       UB→GM   128 KB   [1946.0..3655.6]   100%    op1(RAW)

  Time breakdown (op duration / total_ns):
    op2  ub_to_gm(gm_2, ub_2)     [█████████           ]  46.8%  (1709.6/3655.6 ns)
    op0  gm_to_ub(ub_1, gm_1)     [█████████           ]  44.4%  (1621.6/3655.6 ns)
    op1  vadd(ub_2, ub_1, 2.0)    [██                  ]   8.9%  ( 324.4/3655.6 ns)

  Engine utilization:
    GM→UB       [████████████████████]  44%  (1621.6/3655.6 ns busy)
    UB→GM       [████████████████████]  46%  (1709.6/3655.6 ns busy)
    VecUnit     [████████░░░░░░░░░░░░]   8%  ( 324.4/3655.6 ns busy)

  Execution is fully sequential — no parallel overlap.

  Critical path (topo, length = 3655.6 ns = 100% of makespan):
    op0 ─(RAW on 'ub_1')→ op1 ─(RAW on 'ub_2')→ op2
"""

OPTIMIZATION_SUMMARY_TXT = """# vector_add (fp16, N=65536) — 优化总结

## 初始状态 (round0)
| 指标 | 值 |
|------|-----|
| 绝对延迟 | 0.0183 ms |
| 吞吐量 | 850 GB/s (峰值 52%) |
| 执行模式 | 全串行 (RAW 链) |
| 瓶颈 | op2(ub_to_gm) 占 46.8%, 已饱和 |
| BLOCK_SIZE | 256 |
| Grid | 20 (AI Core transfer) |

## 瓶颈变化历史
| 轮次 | Tier | 瓶颈 op | 瓶颈类型 | 时间占比 | 加速比 |
|------|------|---------|----------|----------|--------|
| 0 | baseline | op2(ub_to_gm) | memory_bandwidth (saturated) | 46.8% | 1.00× |
(后续轮次待优化)

## 成功的优化策略
(待记录)

## 失败的优化策略
(待记录)

## 经验总结与优化建议
- **Tier 1 优先**: 初始 BLOCK_SIZE=256 偏小, 增大 tile 可快速提升带宽利用率
- **ub_to_gm 已饱和**: 传输带宽已达峰值, 需 double buffer 或减少数据量
- **RAW 串行链**: 当前 3 op 全 RAW 依赖, 无可打破的 WAR
- **910B3 注意**: UB=192KB/core, tile 增大时注意不超过 UB 容量
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  构建目录
# ═══════════════════════════════════════════════════════════════════════════════

def create_round0(base_dir: Path):
    r0 = base_dir / "round0"
    r0.mkdir(parents=True, exist_ok=True)

    # 1. kernel.py — 原始 Triton kernel
    (r0 / "kernel.py").write_text("""import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    \"\"\"Vector add: out = x + y (fp16, 128KB tile, grid=20)\"\"\"
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
""", encoding="utf-8")

    # 2. benchmark_result.json — 基准测试结果
    (r0 / "benchmark_result.json").write_text(json.dumps({
        "op_name": "vector_add",
        "dtype": "fp16",
        "N": 65536,
        "BLOCK_SIZE": 256,
        "grid": 20,
        "absolute_latency_ms": 0.0183,
        "speedup_vs_baseline": 1.0,
        "throughput_gb_s": 850.0,
        "utilization_vs_hbm_peak": 0.52,
        "hbm_peak_gb_s": 1616.6,
        "engine_time_ratios": {
            "GM→UB": 0.44, "VecUnit": 0.09, "UB→GM": 0.47,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. msprof/
    msprof = r0 / "msprof"
    msprof.mkdir(exist_ok=True)
    opprof = msprof / "OPPROF_20260723_150000_001"
    (opprof / "simulator").mkdir(parents=True, exist_ok=True)
    (opprof / "simulator" / "trace.json").write_text("[]", encoding="utf-8")
    (opprof / "simulator" / "core0.veccore0").mkdir(exist_ok=True)
    (msprof / "pipeline_report.json").write_text(
        json.dumps(make_msprof_report(), indent=2, ensure_ascii=False),
        encoding="utf-8")

    # 4. hivmir/
    hivmir = r0 / "hivmir"
    hivmir.mkdir(exist_ok=True)
    comp = hivmir / "compiler_output"
    comp.mkdir(exist_ok=True)
    (comp / "hivmir_output.mlir").write_text("""hivm.alloc %gm_1 : memref<128KB>
hivm.alloc %ub_1 : memref<128KB>
hivm.alloc %ub_2 : memref<128KB>
hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
hivm.vadd %ub_2, %ub_1, 2.0
hivm.ub_to_gm %gm_2, %ub_2 : memref<128KB>
""", encoding="utf-8")
    (hivmir / "hivmir_report.json").write_text(
        json.dumps(make_hivmir_report(), indent=2, ensure_ascii=False),
        encoding="utf-8")

    # 5. merged/
    merged = r0 / "merged"
    merged.mkdir(exist_ok=True)
    (merged / "merged_report.json").write_text(json.dumps({
        "meta": {
            "source": "dsl_merger",
            "status": "placeholder — waiting dsl_merger.py implementation",
            "note": "合并 msprof pipeline_report.json + hivmir hivmir_report.json → 29字段完整填充",
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (merged / "final_report_llm.txt").write_text(FINAL_LLM_TXT, encoding="utf-8")
    (merged / "final_report_human.txt").write_text(FINAL_HUMAN_TXT, encoding="utf-8")


def create_roundN_sample(tier_dir: Path, round_num: int):
    """在 Tier 目录下创建一个示例 round (带 optimization_record.json)。"""
    rn = tier_dir / f"round{round_num}"
    rn.mkdir(parents=True, exist_ok=True)

    # kernel.py (模拟优化后)
    (rn / "kernel.py").write_text(f"""# Triton kernel: vector_add — round{round_num} optimized
# BLOCK_SIZE increased from 256 → 8192
import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
""", encoding="utf-8")

    # benchmark_result.json
    (rn / "benchmark_result.json").write_text(json.dumps({
        "op_name": "vector_add", "dtype": "fp16", "N": 65536,
        "BLOCK_SIZE": 8192, "grid": 20,
        "absolute_latency_ms": 0.0163, "speedup_vs_baseline": 1.12,
        "throughput_gb_s": 950.0, "utilization_vs_hbm_peak": 0.59,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # msprof/ (简化占位)
    msprof = rn / "msprof"
    msprof.mkdir(exist_ok=True)
    (msprof / "pipeline_report.json").write_text(
        json.dumps(make_msprof_report(), indent=2), encoding="utf-8")

    # hivmir/ (简化占位)
    hivmir = rn / "hivmir"
    hivmir.mkdir(exist_ok=True)
    (hivmir / "compiler_output").mkdir(exist_ok=True)
    (hivmir / "hivmir_report.json").write_text(
        json.dumps(make_hivmir_report(), indent=2), encoding="utf-8")

    # merged/
    merged = rn / "merged"
    merged.mkdir(exist_ok=True)
    (merged / "merged_report.json").write_text(json.dumps({
        "meta": {"source": "dsl_merger", "status": "placeholder"},
    }, indent=2), encoding="utf-8")
    (merged / "final_report_llm.txt").write_text(FINAL_LLM_TXT, encoding="utf-8")
    (merged / "final_report_human.txt").write_text(FINAL_HUMAN_TXT, encoding="utf-8")

    # ★ optimization_record.json — 本轮独有
    (rn / "optimization_record.json").write_text(json.dumps({
        "round": round_num,
        "phase": "Tier3_tiling",
        "timestamp": f"2026-07-23T{16+round_num:02d}:00:00",

        "plan": {
            "strategy": "increase_tile_size",
            "strategy_tier": 1,
            "description": "增大 BLOCK_SIZE 256→8192, 使传输 tile 从1KB→32KB, 超过 k0 半饱和点",
            "plan_file": f"round{round_num}_plan.md",
        },

        "bottleneck_before": {
            "op_id": 2, "op_type": "ub_to_gm", "engine": "UB→GM",
            "time_ratio": 0.4677, "bw_utilization": 1.0, "regime": "saturated",
            "note": "已饱和 → 不能通过增大 tile 提速, 瓶颈转移到计算",
        },
        "bottleneck_after": {
            "op_id": 1, "op_type": "vadd", "engine": "VecUnit",
            "time_ratio": 0.42, "bw_utilization": 0.88, "regime": "saturated",
        },

        "code_change": {
            "diff_file": f"round{round_num}_diff.patch",
            "lines_changed": 1,
            "summary": "BLOCK_SIZE: 256 → 8192",
        },

        "verification": {
            "stage1_emulator": {"passed": True, "max_abs_error": 4.77e-07},
            "stage2_simulator": {"estimated_speedup": 1.14},
            "stage3_hardware": {"passed": True, "actual_speedup": 1.12},
        },

        "performance": {
            "target_speedup": 1.10,
            "actual_speedup": 1.12,
            "cumulative_speedup": 1.12,
            "latency_ms_before": 0.0183,
            "latency_ms_after": 0.0163,
            "throughput_gb_s_before": 850.0,
            "throughput_gb_s_after": 950.0,
        },

        "decision": "KEEP",
        "decision_reason": "GM→UB+UB→GM 传输已饱和, 瓶颈从传输转移到 VecUnit 计算",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUTS.mkdir(exist_ok=True)
    kernel_dir = OUTPUTS / "vector_add_fp16_N65536"

    # 清理旧目录 (如果需要)
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir()

    # ── 6 个策略文件夹 ──
    tiers = [
        "01_algorithmic_structure",    # Tier 1: Algorithm → 最先
        "02_operator_fusion",          # Tier 2: Fusion → 其次
        "03_tiling_block_config",      # Tier 3: Tiling → 然后
        "04_memory_access",            # Tier 4: Memory → 再
        "05_compute_occupancy",        # Tier 5: Compute → 再
        "06_910b3_architecture",       # Tier 6: Arch → 最后
    ]
    for t in tiers:
        (kernel_dir / t).mkdir()

    # ── round0 ──
    create_round0(kernel_dir)

    # ── 示例 round1 (Tier 1 文件夹) ──
    create_roundN_sample(kernel_dir / "03_tiling_block_config", round_num=1)

    # ── optimization_trajectory.json ──
    (kernel_dir / "optimization_trajectory.json").write_text(json.dumps({
        "op_name": "vector_add", "dtype": "fp16", "N": 65536,
        "initial_latency_ms": 0.0183, "initial_throughput_gb_s": 850.0,
        "total_rounds": 2, "rounds_kept": 1, "rounds_reverted": 0,
        "best_speedup": 1.12, "current_speedup": 1.12,
        "target_speedup": 1.5,
        "trajectory": [
            {"round": 0, "phase": "baseline", "strategy": "-",
             "speedup": 1.0, "bottleneck": "ub_to_gm",
             "bottleneck_type": "memory_bandwidth", "decision": "-"},
            {"round": 1, "phase": "Tier1_block_size", "strategy": "increase_tile_size",
             "speedup": 1.12, "bottleneck": "vadd",
             "bottleneck_type": "compute_vec", "decision": "KEEP"},
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── final_output/ ──
    final = kernel_dir / "final_output"
    final.mkdir(exist_ok=True)
    (final / "optimization_summary.md").write_text(OPTIMIZATION_SUMMARY_TXT, encoding="utf-8")
    (final / "optimized_kernel.py").write_text(
        (kernel_dir / "round0" / "kernel.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    # trajectory_chart.png — 占位 (等 trajectory_chart.py 实现后生成)
    (final / "trajectory_chart.png.placeholder").write_text(
        "# 优化轨迹图 — 待 trajectory_chart.py 生成\n", encoding="utf-8")

    # ── 打印 ──
    print("=" * 60)
    print(f"Output structure: {kernel_dir}")
    print("=" * 60)
    _print_tree(kernel_dir)
    count = sum(1 for _ in kernel_dir.rglob("*") if _.is_file())
    print(f"\nTotal files: {count}")
    print("Done.")


def _print_tree(path: Path, prefix: str = "", depth: int = 0):
    if depth > 5:
        return
    items = sorted(path.iterdir())
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}{item.name}")
        if item.is_dir():
            _print_tree(item, prefix + ("    " if is_last else "│   "), depth + 1)


if __name__ == "__main__":
    main()
