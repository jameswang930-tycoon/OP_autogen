#!/usr/bin/env python3
"""
优化轨迹图生成器 — 双面板图 (参照 AutoKernel progress.png)。

═══════════════════════════════════════════════════════════════════════════════
  图表结构
═══════════════════════════════════════════════════════════════════════════════

  上图 (Top): Cumulative Speedup × Round
    - 绿色 ● = KEEP 轮次
    - 红色 ○ = REVERT 轮次
    - 蓝色 ─ = Running Best (累计最优加速比)
    - 灰色 --- = 目标加速比

  下图 (Bottom): Latency (ms) × Round
    - 绿色 ● = KEEP 轮次
    - 红色 ○ = REVERT 轮次
    - 蓝色 ─ = Running Best (累计最低延迟)
    - 灰色 --- = Baseline 延迟
    - 灰色 --- = 目标延迟

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  python feedback/trajectory_chart.py outputs/vector_add_fp16_N65536
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def generate(kernel_dir: Path, output_path: Optional[Path] = None) -> Path:
    """从 optimization_trajectory.json 生成双面板优化轨迹图。

    Args:
        kernel_dir: outputs/<kernel_name>/ 目录
        output_path: 输出图片路径 (默认 final_output/trajectory_chart.png)

    Returns:
        输出图片路径
    """
    traj_file = kernel_dir / "optimization_trajectory.json"
    if not traj_file.exists():
        raise FileNotFoundError(f"trajectory not found: {traj_file}")

    traj = json.loads(traj_file.read_text(encoding="utf-8"))
    history = traj.get("history", [])
    baseline = traj.get("baseline", {})
    target = traj.get("state", {}).get("best_speedup", 1.5)

    if not history:
        raise ValueError("No history in trajectory")

    # 提取数据
    rounds = [r["round"] for r in history]
    speeds = [r.get("actual_speedup", 1.0) for r in history]
    cum_speeds = [r.get("cumulative_speedup", 1.0) for r in history]
    decisions = [r.get("decision", "?") for r in history]

    baseline_lat_ns = baseline.get("total_ns", 0)
    baseline_lat_ms = baseline_lat_ns / 1e6 if baseline_lat_ns > 0 else 0

    # 计算 running best
    import numpy as np
    speeds_arr = np.array(speeds, dtype=float)
    cum_arr = np.array(cum_speeds, dtype=float)

    # Running best speedup
    running_best = np.maximum.accumulate(cum_arr)

    # Running best latency (从 speedup 反推)
    if baseline_lat_ms > 0:
        latencies = baseline_lat_ms / speeds_arr
        running_best_lat = baseline_lat_ms / running_best
    else:
        latencies = np.ones_like(speeds_arr)
        running_best_lat = np.ones_like(speeds_arr)

    # 分离 KEEP/REVERT
    keep_mask = np.array([d == "KEEP" for d in decisions])
    revert_mask = np.array([d == "REVERT" for d in decisions])

    # ── 绘图 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Optimization Trajectory: {kernel_dir.name}",
                 fontsize=14, fontweight="bold")

    # ═══ 上图: Speedup ═══
    # Running best
    ax1.plot(rounds, running_best, "b-", linewidth=2, alpha=0.7, label="Running Best")
    ax1.fill_between(rounds, 1.0, running_best, alpha=0.05, color="blue")

    # KEEP dots
    if keep_mask.any():
        ax1.scatter(np.array(rounds)[keep_mask], speeds_arr[keep_mask],
                    c="#2ecc71", s=60, zorder=5, label=f"KEEP ({keep_mask.sum()})")
    # REVERT dots
    if revert_mask.any():
        ax1.scatter(np.array(rounds)[revert_mask], speeds_arr[revert_mask],
                    c="#e74c3c", s=40, marker="x", zorder=5,
                    label=f"REVERT ({revert_mask.sum()})")

    # Target line
    ax1.axhline(y=target, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                label=f"Target ({target}x)")
    ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    ax1.set_ylabel("Speedup (x)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))

    # 标注最终值
    final_r = rounds[-1]
    final_s = running_best[-1]
    ax1.annotate(f"{final_s:.2f}x", xy=(final_r, final_s),
                 xytext=(final_r + 1, final_s + 0.02),
                 fontsize=10, fontweight="bold", color="blue")

    # ═══ 下图: Latency ═══
    if baseline_lat_ms > 0:
        # Running best latency
        ax2.plot(rounds, running_best_lat, "b-", linewidth=2, alpha=0.7)
        ax2.fill_between(rounds, running_best_lat.max(), running_best_lat,
                         alpha=0.05, color="blue")

        # KEEP dots
        if keep_mask.any():
            ax2.scatter(np.array(rounds)[keep_mask], latencies[keep_mask],
                        c="#2ecc71", s=60, zorder=5)
        # REVERT dots
        if revert_mask.any():
            ax2.scatter(np.array(rounds)[revert_mask], latencies[revert_mask],
                        c="#e74c3c", s=40, marker="x", zorder=5)

        # Baseline
        ax2.axhline(y=baseline_lat_ms, color="gray", linestyle="--", linewidth=1,
                    alpha=0.6, label=f"Baseline ({baseline_lat_ms:.4f} ms)")
        # Target
        target_lat = baseline_lat_ms / target
        ax2.axhline(y=target_lat, color="gray", linestyle=":", linewidth=0.8,
                    alpha=0.4, label=f"Target ({target_lat:.4f} ms)")

        ax2.set_ylabel("Latency (ms)", fontsize=12)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 标注
        final_lat = running_best_lat[-1]
        ax2.annotate(f"{final_lat:.4f} ms", xy=(final_r, final_lat),
                     xytext=(final_r + 1, final_lat * 1.005),
                     fontsize=10, fontweight="bold", color="blue")

    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_xlim(left=-0.5)

    plt.tight_layout()
    out = output_path or (kernel_dir / "final_output" / "trajectory_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[chart] Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    kd = _PROJECT_DIR / "outputs" / "vector_add_fp16_N65536"
    tj = kd / "optimization_trajectory.json"
    if not tj.exists():
        print("[chart] SKIP: trajectory.json not found")
        return

    try:
        generate(kd)
        print("[chart] OK")
    except Exception as e:
        print(f"[chart] ERROR: {e} (matplotlib/numpy may not be installed)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate(Path(sys.argv[1]))
    else:
        _self_test()
