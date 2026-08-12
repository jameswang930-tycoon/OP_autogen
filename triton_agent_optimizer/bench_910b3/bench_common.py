#!/usr/bin/env python3
"""bench_common — 测量工具 (三块).

模式 A: 通用 msprof → op_summary 目标 kernel Task Duration(us) → 聚合 GB/s / TFLOPS
  (与主优化循环 verify 同一数据源, 保证口径一致)

模式 B: msprof op → OPPROF 8 CSV → 复用 pipeline_parse_board 解析 → 每路径带宽
  (main_mem_r/w, l1, l2, gm_to_ub, ub_to_gm, l0a/l0b/l0c, engine_util, conflict)
  → 这才是填 timing_estimator 微路径占位的真实数据源

★模式 C: measure_event — 工业级 Event 设备侧计时 (2026-08-12 新增, 对齐 triton testing.do_bench)
  - 时间预算自适应 (warmup 25ms / rep 100ms 折算次数)
  - 多窗口 median: n_rep 个独立 Event 对 → median (另报 min/mean)
  - ★输入轮换破 L2: 调用方 fn(i) 按 i 轮换输入 buffer (Ascend 无清 L2 API, 轮换等效 clear_cache)

科学原则: run_bench.py 对每类跑多个变体, 每个度量取全变体最大值.

用法 (库文件, 由 run_bench.py / bench_industrial.py / bench_pytorch_*.py 调用):
  from bench_common import measure_msprof, measure_msprof_op, measure_event
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

# ★所有基准产物统一放 bench_910b3/outputs/ (工业级/PT基准 json、actual 标记、msprof 临时)
BENCH_OUT = Path(__file__).resolve().parent / "outputs"


def clean_bench_out():
    """清理 bench_910b3/outputs/ 全部产物 (json/txt/msprof 临时). 返回删除的文件数."""
    import shutil as _sh
    n = 0
    if BENCH_OUT.exists():
        for p in BENCH_OUT.iterdir():
            try:
                if p.is_dir():
                    _sh.rmtree(p)
                else:
                    p.unlink()
                n += 1
            except Exception:
                pass
    return n


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

def _detect_kernels_per_iter(rows: list, tail_n: int = 60) -> int:
    """★从 op_summary 行序列检测每遍 kernel 数 (真实 launch 序列是周期性的, 每遍 kernel 名相同).
    取末尾 tail_n 行 (loop 部分, 60 行≈15 遍, 通常不含开头 setup 行), 找最小重复周期 k.
    返回整数 k; 检测不到返回 None (调用方兜底取整)."""
    tail = rows[-tail_n:]
    if len(tail) < 4:
        return None
    names = [op for op, _ in tail]
    n = len(names)
    for k in range(1, n // 2 + 1):
        m = (n // k) * k                       # 只检查完整周期部分
        if m < k * 2:
            continue
        if all(names[i] == names[i - k] for i in range(k, m)):
            return k
    return None


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
        # ★每遍 kernel 数: 从行序列检测重复周期 (真实 launch 序列周期性, 每遍 kernel 名相同) —
        #   比 total/(warmup+measure) 取整更可靠 (后者会被 setup 行/非均匀撑偏). 检测不到才兜底取整.
        k_per_iter = _detect_kernels_per_iter(rows)
        if not k_per_iter:
            k_per_iter = max(1, int(round(total / (warmup + measure))))
        # ★从末尾取最后 measure 遍: 前面是 setup+warmup, 天然排除 setup 行膨胀.
        #   (app 结构 = setup → warmup 循环 → measure 循环; measure 循环在末尾)
        take = measure * k_per_iter
        measured = rows[-take:] if take else rows
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
                # ★每遍 kernel 数 (整数): 1 = 已融合成单 kernel (工业级融合成功标志)
                "kernels_per_iter": k_per_iter}
        if extras:
            data.update(extras)
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [PT-msprof] 端到端 {e2e_us:.1f}us / 纯kernel {kernel_us:.1f}us (÷{measure}) → {out_json.name}",
              flush=True)
        return data
    except Exception as e:
        print(f"  ⚠ [PT-msprof] 失败: {str(e)[:150]} → 回退 Event", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  ★Event 设备侧统一测量 (对齐 triton testing.do_bench 标准) — 2026-08-12 修复
#  修 3 个方法学缺陷:
#   ① L2 复用虚高: 连续 forward 同一批张量, 工作集<192MB(L2) 时后 N 次全 L2 命中
#      → 测到 L2 带宽. 修复: 每次 rep 轮换输入 buffer (调用方 fn(i) 按 i 取组).
#   ② 单窗口 ÷N 只有 1 个样本: 修复: n_rep 个独立 Event 对 (设备流水连续, 最后 sync),
#      返回 median/min/mean (抗单次抖动).
#   ③ 固定次数对小 kernel 太少: 修复: 时间预算自适应 (warmup_ms/rep_ms → 次数).
# ═══════════════════════════════════════════════════════════════════════

def measure_event(fn, warmup_ms: int = 25, rep_ms: int = 100,
                  max_rep: int = 50, min_rep: int = 5) -> dict:
    """★工业级 Event 设备侧计时 (do_bench 同款):
       fn(i) = 第 i 次 forward — 调用方按 i % n_buf 轮换输入 buffer (破 L2 复用).
       流程: 5 次估时长 → n_warmup/n_rep 按时间预算自适应 → warmup → n_rep 个独立 Event 对
       (每 rep: record → fn(i) → record, 不逐个 sync, 设备流水连续) → 最后 sync.
       返回 {median_us, min_us, mean_us, rep, warmup, est_us, times_us} (us 为单次)."""
    import statistics
    import torch
    # ── 估时长 (buffer 0, 5 次) ──
    for _ in range(5):
        fn(0)
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        fn(0)
    e.record()
    torch.npu.synchronize()
    est_ms = s.elapsed_time(e) / 5.0
    # ── 时间预算 → 次数 (快 kernel 自动加次, 慢 kernel 自动减次) ──
    if est_ms > 0:
        n_warmup = max(3, int(warmup_ms / est_ms))
        n_rep = max(min_rep, min(max_rep, int(rep_ms / est_ms)))
    else:
        n_warmup, n_rep = 25, min_rep
    # ── warmup (破坏性 JIT/图编译/冷 cache 在此消化, 不计时) ──
    for _ in range(n_warmup):
        fn(0)
    torch.npu.synchronize()
    # ── 测量: n_rep 个独立 Event 对 (每 rep 轮换 buffer, 破 L2 复用) ──
    times = []
    for i in range(n_rep):
        s.record()
        fn(i)
        e.record()
        times.append(s.elapsed_time(e))
    torch.npu.synchronize()
    us = [t * 1000.0 for t in times]
    return {"median_us": round(statistics.median(us), 1),
            "min_us": round(min(us), 1),
            "mean_us": round(statistics.mean(us), 1),
            "rep": n_rep, "warmup": n_warmup,
            "est_us": round(est_ms * 1000.0, 1),
            "times_us": [round(x, 1) for x in us]}


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
