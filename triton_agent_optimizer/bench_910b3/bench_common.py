#!/usr/bin/env python3
"""bench_common — msprof 测量工具 (warmup + 多轮取平均)。

═══ 怎么运行 ═══
  这是库文件 (供 run_bench.py 调用), 不直接跑。
  要跑整套基准 → python3 run_bench.py (见 run_bench.py 顶部教程)。
  环境: conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  依赖: msprof (CANN 自带)

核心: 对一个 bench kernel, 先裸跑 warmup 次 (JIT 编译/冷cache 预热),
再跑 rounds 次 msprof, 每次读 op_summary 的目标 kernel Task Duration(us),
取平均 → 配合 bench 的 bytes/flops 算 GB/s 或 TFLOPS。

用法:
  from bench_common import measure_kernel
  avg_us, durations = measure_kernel("python3 bench_kernels.py --bench read_bw")
"""
import csv
import os
import subprocess
from pathlib import Path

try:
    sys = __import__("sys")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _read_target_duration(prof_out: Path):
    """读 op_summary, 取目标 kernel (非 aclnn) 的最大 Task Duration(us)。"""
    summaries = sorted(prof_out.rglob("op_summary*.csv"))
    if not summaries:
        return None
    try:
        with open(summaries[0], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    dur_us = None
    for row in rows:
        dur = row.get("Task Duration(us)") or row.get("TaskDuration")
        op = row.get("Op Name") or row.get("OpName") or ""
        if dur and not op.lower().startswith("aclnn"):
            try:
                dur_us = max(dur_us or 0, float(dur))
            except ValueError:
                pass
    return dur_us


def run_msprof_once(work_dir: Path, app_cmd: str, run_idx: int):
    """跑一次 msprof, 返回目标 kernel 的 Task Duration(us)。"""
    msprof_out = work_dir / f"msprof_{run_idx}"
    msprof_out.mkdir(parents=True, exist_ok=True)
    cmd = ["msprof", f"--output={msprof_out}",
           f"--application={app_cmd}", "--ai-core=on"]
    print(f"    msprof run{run_idx}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    d = _read_target_duration(msprof_out)
    if d is None:
        tail = (r.stderr or "")[-500:] + (r.stdout or "")[-500:]
        raise RuntimeError(f"msprof 无 op_summary: {tail.strip()[:300]}")
    return d


def measure_kernel(app_cmd: str, warmup: int = 1, rounds: int = 3,
                   work_dir: Path = None):
    """对 bench kernel: warmup 裸跑 + rounds 次 msprof → 平均 us。
    返回 (avg_us, durations_list)。"""
    if work_dir is None:
        work_dir = Path(os.environ.get("BENCH_OUT", "bench_out"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # warmup: 裸跑 (不 profile)
    for i in range(warmup):
        subprocess.run(["python3"] + app_cmd.split()[1:], capture_output=True,
                       text=True, timeout=3600)
    print(f"  warmup x{warmup} 完成, 测 {rounds} 轮 msprof...", flush=True)

    durations = []
    for i in range(rounds):
        d = run_msprof_once(work_dir, app_cmd, i)
        durations.append(d)
        print(f"    run{i}: {d:.1f}us", flush=True)
    avg_us = sum(durations) / len(durations)
    return avg_us, durations
