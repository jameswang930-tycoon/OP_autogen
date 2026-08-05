#!/usr/bin/env python3
"""bench_common — msprof 测量工具 (双模式).

模式 A: 通用 msprof → op_summary 目标 kernel Task Duration(us) → 聚合 GB/s / TFLOPS
  (与主优化循环 verify 同一数据源, 保证口径一致)

模式 B: msprof op → OPPROF 8 CSV → 复用 pipeline_parse_board 解析 → 每路径带宽
  (main_mem_r/w, l1, l2, gm_to_ub, ub_to_gm, l0a/l0b/l0c, engine_util, conflict)
  → 这才是填 timing_estimator 微路径占位的真实数据源

科学原则: run_bench.py 对每类跑多个变体, 每个度量取全变体最大值.

用法 (库文件, 由 run_bench.py 调用):
  from bench_common import measure_msprof, measure_msprof_op
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


def _read_kernel_durations(prof_out: Path, kernel_name: str) -> list:
    """读 op_summary, 返回目标 kernel 的**逐次** Task Duration(us) 列表 (按行序 = launch 顺序).
    ★一次 msprof 内 app 循环 launch N 次 → 每行一次 → 可跳过热身、平均稳态."""
    summaries = sorted(prof_out.rglob("op_summary*.csv"))
    if not summaries:
        return []
    try:
        with open(summaries[0], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    durs = []
    for row in rows:
        dur = row.get("Task Duration(us)") or row.get("TaskDuration")
        op = row.get("Op Name") or row.get("OpName") or ""
        if dur and op == kernel_name:
            try:
                durs.append(float(dur))
            except ValueError:
                pass
    return durs


# ═══════════════════════════════════════════════════════════════════════
#  模式 A: 一次 msprof 内循环 (app 自己 launch warmup+measure 次) → 稳态平均
#  ⚠ 不拆成 measure 次独立 msprof — 每次 msprof 有 ~30s 工具开销, 100 次不可行
# ═══════════════════════════════════════════════════════════════════════

def measure_msprof(app_cmd: str, kernel_name: str,
                   warmup: int = 30, measure: int = 100,
                   work_dir: Path = None) -> tuple:
    """模式 A: 一次 msprof 跑整个 app (app 内部 launch warmup+measure 次),
    读 op_summary 目标 kernel 逐次耗时, 跳过前 warmup 个, 平均后 measure 个.
    返回 (avg_us, measured_durations)."""
    if work_dir is None:
        work_dir = Path(os.environ.get("BENCH_OUT", "bench_out"))
    work_dir.mkdir(parents=True, exist_ok=True)

    msprof_out = work_dir / "msprof_0"
    msprof_out.mkdir(parents=True, exist_ok=True)
    # 把 warmup/measure 传给 app (bench_kernels.py 读 env)
    env = dict(os.environ, BENCH_WARMUP_ITERS=str(warmup),
               BENCH_MEASURE_ITERS=str(measure))
    cmd = ["msprof", f"--output={msprof_out}",
           f"--application={app_cmd}", "--ai-core=on"]
    print(f"  msprof 采集 (app 内部 {warmup}+{measure} launch, 1 次 msprof)...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)

    durs = _read_kernel_durations(msprof_out, kernel_name)
    if len(durs) < warmup + 1:
        tail = (r.stderr or "")[-400:] + (r.stdout or "")[-400:]
        raise RuntimeError(f"msprof 只记录 {len(durs)} 次 [{kernel_name}] "
                           f"(需 ≥ warmup+1={warmup+1}): {tail.strip()[:200]}")
    measured = durs[warmup:]                      # 跳过前 warmup 次 (冷cache/编译)
    if len(measured) > measure:
        measured = measured[-measure:]            # 只取最后 measure 次
    avg_us = sum(measured) / len(measured)
    print(f"  记录 {len(durs)} 次, 取后 {len(measured)} 次平均: {avg_us:.1f}us", flush=True)
    return avg_us, measured


# ═══════════════════════════════════════════════════════════════════════
#  模式 B: msprof op → 每路径带宽 (复用 pipeline_parse_board 解析)
# ═══════════════════════════════════════════════════════════════════════

def measure_msprof_op(app_cmd: str, kernel_name: str,
                      work_dir: Path, run_idx: int) -> dict:
    """模式 B: msprof op 单轮 → board.json dict (全路径带宽 + 引擎占比).
    返回 pipeline_parse_board.parse 的结构 (normalized.bandwidth_gb_s / engine_utilization ...)."""
    from analyzers.pipeline_parse_board import parse as parse_board, find_opprof

    out = work_dir / f"op_{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["msprof", "op", f"--kernel-name={kernel_name}", f"--output={out}",
           "--warm-up=10", *app_cmd.split()]
    print(f"    msprof op [{kernel_name}]: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    opprof = find_opprof(out)
    if opprof is None:
        tail = (r.stderr or "")[-400:] + (r.stdout or "")[-400:]
        raise RuntimeError(f"msprof op 无 OPPROF: {tail.strip()[:300]}")
    board = parse_board(opprof)
    return board


def flatten_per_path(board: dict) -> dict:
    """从 board.json 提取扁平 per-path 度量 (跑多个变体后取最大)."""
    n = board.get("normalized", {})
    bw = n.get("bandwidth_gb_s", {}) or {}
    comp = n.get("compute", {}) or {}
    eng = n.get("engine_utilization", {}) or {}
    return {
        "main_mem_read_gb_s": bw.get("main_mem_read_gb_s"),
        "main_mem_write_gb_s": bw.get("main_mem_write_gb_s"),
        "l1_read_gb_s": bw.get("l1_read_gb_s"),
        "l2_read_gb_s": bw.get("l2_read_gb_s"),
        "gm_to_ub_gb_s": bw.get("gm_to_ub_gb_s"),
        "ub_to_gm_gb_s": bw.get("ub_to_gm_gb_s"),
        "l0a_read_gb_s": bw.get("l0a_read_gb_s"),
        "l0a_write_gb_s": bw.get("l0a_write_gb_s"),
        "l0b_read_gb_s": bw.get("l0b_read_gb_s"),
        "l0b_write_gb_s": bw.get("l0b_write_gb_s"),
        "l0c_read_gb_s": bw.get("l0c_read_gb_s"),
        "l0c_write_gb_s": bw.get("l0c_write_gb_s"),
        "ub_vector_read_gb_s": bw.get("ub_vector_read_gb_s"),
        "ub_vector_write_gb_s": bw.get("ub_vector_write_gb_s"),
        "mte1_ratio": eng.get("mte1"),
        "mte2_ratio": eng.get("mte2"),
        "mte3_ratio": eng.get("mte3"),
        "cube_ratio": eng.get("cube"),
        "vec_ratio": eng.get("vec"),
        "scalar_ratio": eng.get("scalar"),
        "cube_fops": comp.get("cube_fops"),
        "vector_fops": comp.get("vector_fops"),
        "l2_hit_rate": n.get("l2_hit_rate"),
    }
