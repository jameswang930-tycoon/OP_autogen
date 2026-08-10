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
import json
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
                   warmup: int = 10, measure: int = 30,
                   work_dir: Path = None) -> tuple:
    """模式 A: 一次 msprof 跑整个 app (app 内部 launch warmup+measure 次),
    读 op_summary 目标 kernel 逐次耗时, 跳过前 warmup 个, 平均后 measure 个.
    ★msprof 测的是设备侧 kernel 时间(确定性高): 热身10(过JIT/冷cache) + 测量30(够稳) 即可,
      100 轮无统计增益, 且要求 msprof 记录 131 行(易触发合并问题). 返回 (avg_us, measured_durations)."""
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


# ═══════════════════════════════════════════════════════════════════════
#  PyTorch 统一 msprof 测量 (pt_msprof.py / bench_industrial.py 用)
#  一次 msprof 同时算 端到端 + 纯kernel:
#    端到端   = Σ全部 kernel 行 (跳过热身) ÷ N      ← 与 triton 侧 verify 同口径 (可比)
#    纯kernel = 同一实测设备时间 (torch 的计算 kernel 全是 aclnn 前缀, 与端到端不可分)
#  ⚠ 热身行要跳过 (app 内部 warmup+measure 次 launch; 每遍 kernel 数 ≈ 总行数/(warmup+measure))
# ═══════════════════════════════════════════════════════════════════════

def _read_op_summary_rows(prof_out: Path) -> list:
    """读全部 op_summary*.csv 合并 → [(op_name, dur_us), ...] 按行序.
    ★msprof 可能按 {device}_{model}_{iter} 拆多份文件 — 只读 summaries[0] 会漏掉大部分 kernel
      (表现为"仅 13 行 < measure(30) → 回退"). 必须合并所有文件."""
    summaries = sorted(prof_out.rglob("op_summary*.csv"))
    if not summaries:
        return []
    out = []
    for p in summaries:
        try:
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        for row in rows:
            dur = row.get("Task Duration(us)") or row.get("TaskDuration")
            op = row.get("Op Name") or row.get("OpName") or ""
            try:
                d = float(dur)
            except (TypeError, ValueError):
                continue
            out.append((op, d))
    return out


def measure_pytorch_msprof(app_cmd: str, out_json: Path, flops: float,
                           measure: int = 30, warmup: int = 5,
                           extras: dict = None) -> dict:
    """把 PyTorch bench 包进**一次** msprof → 从 op_summary 同时算 端到端 + 纯kernel (÷measure),
    连同 flops 写 out_json. 返回 metrics dict; 失败 (无 msprof/无 kernel) 返回 None (调用方回退 Event).
    ★app 内部应 warmup + measure 次 launch (bench 脚本的 --warmup/--measure); 热身行跳过."""
    try:
        import shutil as _sh
        msprof_out = out_json.parent / "msprof_pt"
        _sh.rmtree(msprof_out, ignore_errors=True)
        msprof_out.mkdir(parents=True, exist_ok=True)
        cmd = ["msprof", f"--output={msprof_out}", f"--application={app_cmd}", "--ai-core=on"]
        print(f"  [PT-msprof] 一次 msprof 采集 {app_cmd[:80]}...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        rows = _read_op_summary_rows(msprof_out)
        total = len(rows)
        if total < measure:
            print(f"  ⚠ [PT-msprof] 仅 {total} 行 < measure({measure}) → 回退 Event", flush=True)
            return None
        k_per_iter = total / (warmup + measure)          # 每遍 kernel 数 (含框架)
        skip = int(warmup * k_per_iter)
        take = int(measure * k_per_iter)
        measured = rows[skip:skip + take] if take else rows[skip:]
        if not measured:
            return None
        all_us = sum(d for _, d in measured)
        # ★torch 侧修复: 计算 kernel 全是 aclnn 前缀 (aclnnMatmul 等), "非 aclnn"过滤会得 0.
        #   torch 的"纯 kernel"与端到端不可分 (计算=框架都走 aclnn 下发) → kernel_time_us = 实测设备时间.
        #   (triton 侧 verifier 才用"非 aclnn=目标 kernel"过滤, 两者口径本就不同)
        e2e_us = all_us / measure
        kernel_us = e2e_us
        e2e_s = e2e_us / 1e6
        tflops = (flops / 1e12 / e2e_s) if e2e_s else None
        data = {"tflops": round(tflops, 2) if tflops else None,
                "time_us": round(e2e_us, 1),
                "kernel_time_us": round(kernel_us, 1) if kernel_us else None,
                "method": "msprof", "warmup": warmup, "measure": measure,
                "rows_total": total, "rows_measured": len(measured),
                # ★每遍 kernel 数: ≈1.0 = 已融合成单 kernel (工业级融合成功标志)
                "kernels_per_iter": round(len(measured) / measure, 1) if measured else None}
        if extras:
            data.update(extras)
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [PT-msprof] 端到端 {e2e_us:.1f}us / 纯kernel {kernel_us:.1f}us (÷{measure}) → {out_json.name}",
              flush=True)
        return data
    except Exception as e:
        print(f"  ⚠ [PT-msprof] 失败: {str(e)[:150]} → 回退 Event", flush=True)
        return None


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
