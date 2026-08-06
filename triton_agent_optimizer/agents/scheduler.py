#!/usr/bin/env python3
"""Scheduler — v4 状态机: 读 diagnosis.json → 提取当前 tier 字段 → 驱动 Planner→Coder→验证→晋升。

v4 流程 (见 README v4):
  每轮:
    ① run_optimize.sh <input_dir> <round_dir>  → 采集+解析 → diagnosis.json
    ② 读 diagnosis.json → summary.num_kernels
    ③ 按当前 tier 提取该策略要看的字段段 (extract_tier_fields)
    ④ Planner: 字段段 + 策略文档 + 单文件 + config → plan.md + 晋升决策
    ⑤ Coder: plan + 教程 + 纠错文档 → 改 kernel_op.py (单文件)
    ⑥ 验证: 只跑 msprof 端到端 → 加速比; 失败报错回传 Coder 同轮重改
    ⑦ 记录 + 晋升/降级/停止 → 下一轮/下一 tier

用法:
  python -m agents.scheduler <op_dir> [--max-rounds N] [--stub]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

TIER_NAMES = {
    1: "01_algorithmic_structure", 2: "02_operator_fusion",
    3: "03_tiling_block_config", 4: "04_memory_access",
    5: "05_compute_occupancy", 6: "06_910b3_architecture",
}
TIER_LABEL = {
    1: "算法结构", 2: "算子融合", 3: "分块配置",
    4: "访存", 5: "计算占用", 6: "架构专属",
}

# ── 每 tier 要提取的字段段 (JSON path → 中文说明) ──
#   deep 在 kernels[i].deep 下 → 前缀 kernels[].deep
#   每 tier 的字段要足以: ①判断瓶颈是否属本层(promote) ②给出具体改法
TIER_FIELDS = {
    1: [  # 算法结构: 算力利用 + 精度 + 强度 (11 字段)
        ("summary.num_kernels", "优化目标kernel数"),
        ("summary.total_ns", "端到端耗时ns"),
        ("kernels[].deep.compute.cube_fops", "cube浮点运算数"),
        ("kernels[].deep.compute.vector_fops", "向量运算数"),
        ("kernels[].deep.compute.cube_ratio", "cube指令占比"),
        ("kernels[].deep.compute.cube_fp16_ratio", "cube fp16占比"),
        ("kernels[].deep.compute.cube_int8_ratio", "cube int8占比"),
        ("kernels[].deep.engine_utilization.vec", "向量指令占比"),
        ("kernels[].deep.roofline.compute_utilization", "算力利用率"),
        ("kernels[].deep.roofline.arithmetic_intensity", "算术强度(计算/访存)"),
        ("kernels[].deep.roofline.bottleneck_type", "瓶颈类型"),
    ],
    2: [  # 算子融合: 多算子/launch/类型 (8 字段)
        ("summary.num_kernels", "优化目标kernel数"),
        ("summary.num_kernels_total", "总kernel数(含框架)"),
        ("summary.api_overhead_total_us", "launch开销us"),
        ("kernels[].task.task_type", "每kernel引擎"),
        ("kernels[].launch_count", "每kernel launch次数"),
        ("api_overhead", "API开销明细"),
        ("multi_kernel", "算子类型分解"),
        ("framework_kernels", "框架kernel(非目标)"),
    ],
    3: [  # 分块配置: 核数 + L0A/B 搬运 (8 字段)
        ("kernels[].task.block_dim", "核数"),
        ("kernels[].deep.engine_utilization.mte1", "MTE1(L1→L0A/B)占比"),
        ("kernels[].deep.engine_utilization.mte2", "MTE2(GM→L1)占比"),
        ("kernels[].deep.engine_utilization.cube", "cube占比"),
        ("kernels[].deep.bandwidth_gb_s.l0a_read_gb_s", "L0A读带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0a_write_gb_s", "L0A写带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0b_read_gb_s", "L0B读带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0b_write_gb_s", "L0B写带宽"),
    ],
    4: [  # 访存: GM带宽/L2/搬运 (9 字段)
        ("kernels[].deep.bandwidth_gb_s.main_mem_read_gb_s", "GM读带宽"),
        ("kernels[].deep.bandwidth_gb_s.main_mem_write_gb_s", "GM写带宽"),
        ("kernels[].deep.bandwidth_gb_s.gm_to_ub_gb_s", "GM→UB带宽"),
        ("kernels[].deep.bandwidth_gb_s.ub_to_gm_gb_s", "UB→GM带宽"),
        ("kernels[].deep.l2_hit_rate", "L2命中率"),
        ("kernels[].task.pipes_us.aic_mte2_time_us", "MTE2(GM读)耗时"),
        ("kernels[].task.pipes_us.aic_mte3_time_us", "MTE3(GM写)耗时"),
        ("kernels[].deep.roofline.memory_utilization", "访存利用率"),
        ("kernels[].deep.roofline.arithmetic_intensity", "算术强度(计算/访存)"),
    ],
    5: [  # 计算占用: cube/标量时间 + 冲突 (8 字段)
        ("kernels[].task.pipes_us.aic_cube_time_us", "cube耗时"),
        ("kernels[].task.pipes_us.aic_scalar_time_us", "标量耗时"),
        ("kernels[].deep.engine_utilization.scalar", "scalar占比"),
        ("kernels[].deep.engine_utilization.fixpipe", "fixpipe占比"),
        ("kernels[].deep.compute.cube_ratio", "cube指令占比"),
        ("kernels[].deep.conflict.bank_cflt_ratio", "bank冲突"),
        ("kernels[].deep.conflict.bankgroup_cflt_ratio", "bankgroup冲突"),
        ("kernels[].deep.conflict.total_cflt_ratio", "vec总冲突"),
    ],
    6: [  # 910B3 架构: 引擎分布/阻塞 (6 字段)
        ("kernels[].deep.engine_utilization", "各引擎利用率"),
        ("kernels[].deep.conflict.mte_cflt_ratio", "mte冲突"),
        ("kernels[].deep.conflict.wait_ratio", "vec被阻塞占比"),
        ("kernels[].task.task_type", "每kernel引擎"),
        ("kernels[].task.block_dim", "核数"),
        ("kernels[].deep.roofline.bottleneck_type", "瓶颈类型"),
    ],
}


def _keep_floor() -> float:
    """★#2 噪声地板: 采纳需 speedup ≥ prev_speedup×floor (默认 1.01, env KEEP_FLOOR 可调).
    防 msprof ±1% 噪声让实际变慢的 kernel 被采纳进链 (v3 轨迹 0.97~1.01 全是噪声的教训)."""
    return float(os.environ.get("KEEP_FLOOR", "1.01"))


def _extract_mnk(code: str):
    """从 kernel_op.py ①config 区提取 M/N/K (默认值) → (M,N,K) 或 None。"""
    import re
    vals = {}
    for var, env in (("M", "MATMUL_M"), ("N", "MATMUL_N"), ("K", "MATMUL_K")):
        m = re.search(rf'os\.environ\.get\("{env}",\s*"?(\d+)"?\)', code)
        if not m:
            m = re.search(rf"^\s*{var}\s*=\s*(\d+)", code, re.M)
        if m:
            vals[var] = int(m.group(1))
    if len(vals) == 3 and all(vals.values()):
        return vals["M"], vals["N"], vals["K"]
    return None


def _clip(s: str, n: int) -> str:
    """截断到 ≤n 字符但保证括号成对 (避免 'grid = (triton.cdiv(' 这种半截表达式).
    左括号多于右括号时向后扫描到括号归零 (★计嵌套: 内层 '(' 也要自己的 ')' 配平);
    超长末尾加 '...'."""
    if len(s) <= n:
        return s
    cut = s[:n]
    balance = cut.count("(") - cut.count(")")
    if balance > 0:
        i = n
        while i < len(s) and balance > 0:
            if s[i] == "(":
                balance += 1
            elif s[i] == ")":
                balance -= 1
            i += 1
        if balance == 0:
            return s[:i] + "..."
    return cut + "..."


def _summarize_changes(plan) -> str:
    """把 plan 的 changes[] 压成一句梗概 (hist 记录用, 让 planner 知道试过什么)。
    用 '=' 提取 LHS(变量名)=RHS(新值), 截断时括号保持成对 (不再出现半截表达式)."""
    changes = _extract_changes_from_plan(getattr(plan, "plan_text", ""))
    if not changes:
        return getattr(plan, "strategy", "?")[:60]
    parts = []
    for ch in changes[:2]:
        old = (ch.get("old_code") or "").strip()
        new = (ch.get("new_code") or "").strip()
        # ★split("=", 1): 只按第一个 "=" 切 — 否则 new 里第二个 "=" (如 input_precision="tf32")
        #   会把 RHS 取错段/截断, 产生"只有左括号"的半截表达式
        lhs = _clip(old.split("=", 1)[0].strip(), 40)
        rhs = _clip(new.split("=", 1)[1].strip(), 40) if "=" in new else ""
        if rhs:
            parts.append(f"{lhs}={rhs}")
        else:
            parts.append(f"{_clip(old, 60)}→{_clip(new, 60)}")
    return _clip("; ".join(parts), 150)


def _extract_changes_from_plan(plan_text: str) -> list:
    """从 plan JSON 提取 changes[] (与 coder._extract_changes 同一实现, 不重复维护)。"""
    from agents.coder import _extract_changes
    return _extract_changes(plan_text)


def _get(d, path: str):
    """按 'a.b.c' 或 'kernels[].x.y' 路径取值 (返回第一个匹配)。
    每级 dict 先精确键, 无则子串匹配兜底 (如 conflict.aiv_vec_bank_cflt_ratio ↔ bank_cflt_ratio)。"""
    parts = path.split(".")
    cur = d
    if parts and parts[0].endswith("[]"):
        key = parts[0][:-2]
        items = cur.get(key, []) if isinstance(cur, dict) else []
        for it in items:
            v = _get(it, ".".join(parts[1:]))
            if v is not None:
                return v
        return None
    for p in parts:
        if isinstance(cur, dict):
            if p in cur:
                cur = cur[p]
            else:
                nxt = next((vv for kk, vv in cur.items() if p.lower() in kk.lower()), None)
                if nxt is None:
                    return None
                cur = nxt
        else:
            return None
    return cur


# ★全局摘要字段: 前层信号, 任何 tier 都喂 — 让 planner 做"前层优先检查"
#   (算法/融合/算力/精度是否还有优化空间, 不能只闷头调本层参数)
#   ★不喂 kernels[].deep.* — 那些必须 per-kernel 展示 (见 TIER_PER_KERNEL), 只取第一个 kernel 会误导
GLOBAL_FIELDS = [
    ("summary.num_kernels", "目标kernel数 (多→融合空间Tier2)"),
    ("summary.num_kernels_total", "总kernel数(含框架)"),
    ("summary.api_overhead_total_us", "launch开销us (大→融合空间Tier2)"),
]

# ★每 tier 的 per-kernel 关键指标 — 多 kernel 算子 (MLP/attention_mlp/conv_bias_relu) 每个 kernel 都列,
#   解决 _get("kernels[].xxx") 只取第一个 kernel 的致命缺陷 (planner 误以为所有 kernel 都一样).
#   值来自 kernel_slots[i].task (pipes_us/block_dim/est_bytes) 和 .deep (board 填充的 roofline/带宽/引擎/冲突)
TIER_PER_KERNEL = {
    1: [("deep.roofline.compute_utilization", "cube_util"),
        ("deep.roofline.bottleneck_type", "bottleneck"),
        ("deep.roofline.arithmetic_intensity", "AI"),
        ("deep.compute.cube_ratio", "cube_r"),
        ("deep.compute.cube_fp16_ratio", "fp16_r"),
        ("deep.compute.vector_fops", "vec_fops"),
        ("task.pipes_us.aiv_vec_time_us", "vec_us")],   # ★vec 耗时 (rms_norm/softmax/bias_gelu 命门)
    2: [("task.task_type", "type"),
        ("deep.roofline.compute_utilization", "cube_util"),
        ("deep.roofline.bottleneck_type", "bottleneck"),
        ("launch_count", "launches")],
    3: [("task.block_dim", "cores"),
        ("deep.engine_utilization.cube", "cube"),
        ("deep.engine_utilization.mte1", "mte1"),
        ("deep.engine_utilization.mte2", "mte2"),
        ("deep.bandwidth_gb_s.l0a_read_gb_s", "l0a_r"),
        ("deep.bandwidth_gb_s.l0b_read_gb_s", "l0b_r"),
        ("task.pipes_us.aic_mte1_time_us", "mte1_us"),  # ★L1→L0A/B 搬运耗时
        ("deep.conflict.bank_cflt_ratio", "bank_cflt")],
    4: [("deep.bandwidth_gb_s.main_mem_read_gb_s", "gm_r"),
        ("deep.bandwidth_gb_s.main_mem_write_gb_s", "gm_w"),
        ("deep.bandwidth_gb_s.gm_to_ub_gb_s", "gm2ub"),
        ("deep.bandwidth_gb_s.ub_to_gm_gb_s", "ub2gm"),
        ("deep.l2_hit_rate", "l2"),
        ("task.est_bytes_in", "in_B"),                  # ★绝对搬运量 (L2 复用/降搬运判断)
        ("task.est_bytes_out", "out_B")],
    5: [("task.pipes_us.aic_cube_time_us", "cube_us"),
        ("task.pipes_us.aic_scalar_time_us", "scalar_us"),
        ("deep.engine_utilization.scalar", "scalar"),
        ("deep.conflict.bank_cflt_ratio", "bank_cflt"),
        ("deep.conflict.wait_ratio", "wait")],
    6: [("deep.engine_utilization.cube", "cube"),
        ("deep.engine_utilization.vec", "vec"),
        ("deep.engine_utilization.mte2", "mte2"),
        ("deep.engine_utilization.mte3", "mte3"),
        ("deep.conflict.wait_ratio", "wait"),
        ("task.task_type", "type"),
        ("task.block_dim", "cores"),
        ("deep.roofline.bottleneck_type", "bottleneck")],
}


def _pk_get(k: dict, path: str):
    """per-kernel 取值: 'task.X' → k['task']['X'], 'deep.Y.Z' → k['deep']['Y']['Z'], 否则 k[path]."""
    cur = k
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        if p in cur:
            cur = cur[p]
        else:                       # 大小写不敏感兜底 (如 column 名小写差异)
            nxt = next((v for kk, v in cur.items() if p.lower() in str(kk).lower()), None)
            if nxt is None:
                return None
            cur = nxt
    return cur if not isinstance(cur, (dict, list)) else None


def _fnum(v):
    """统一数值格式化: 整数直出, 小值(利用率/占比) 3 位小数, 大值(带宽/耗时) 1 位."""
    if v is None:
        return "?"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return str(int(f))
    return f"{f:.3f}" if abs(f) < 10 else f"{f:.1f}"


def extract_tier_fields(diagnosis: dict, tier: int) -> str:
    """喂 Planner: ★全局摘要(前层信号) + Per-Kernel 概览(★每 kernel 该 tier 关键指标) + 本层附加字段。
    ★多 kernel 算子每个 kernel 都列 (不再 _get 只取第一个 kernel 的值误导 planner);
      per-kernel 深层指标 (算力/带宽/引擎/冲突) 必须逐 kernel 看, 才能打中占比最大的瓶颈 kernel。"""

    def _fmt(path: str, desc: str) -> str:
        v = _get(diagnosis, path)
        if v is None:
            return f"- {desc} ({path}): (无数据)"
        if isinstance(v, (dict, list)):
            return f"- {desc} ({path}): {json.dumps(v, ensure_ascii=False)[:600]}"
        return f"- {desc} ({path}): {v}"

    lines = [f"# 当前 Tier {tier} ({TIER_LABEL.get(tier, '')})"]
    lines.append("# ══ 全局摘要 (前层信号: 融合空间/launch开销) ★任何轮都看 ══")
    lines += [_fmt(p, d) for p, d in GLOBAL_FIELDS]

    # ★每 kernel 一行: 耗时占比 + 该 tier 关键指标 (多 kernel 全列, 优化打中占比大的瓶颈 kernel)
    lines.append("# ══ Per-Kernel 概览 (★每 kernel 该 tier 关键指标; 优化打中占比最大的 kernel) ══")
    _total_us = (diagnosis.get("summary") or {}).get("total_ns")
    _total_us = (_total_us / 1000) if _total_us else None
    _pk_fields = TIER_PER_KERNEL.get(tier, [])
    for _k in (diagnosis.get("kernels") or []):
        _name = _k.get("kernel_name", "?")
        _dur = (_k.get("task") or {}).get("task_duration_us")
        # ★bug 修复: 占比必须 × launch_count — total_ns 是所有 launch 之和,
        #   而 task_duration_us 是该 kernel 首次 launch 的单次耗时。
        #   重复调用 kernel (如 attention 的 matmul_kernel 被 QKV 复用 3 次)
        #   若用单次耗时占比会被严重低估 (3×300=900 却显示 300)。
        _launch = _k.get("launch_count") or 1
        _dur_total = (_dur * _launch) if _dur else None
        _pct = f"{_dur_total/_total_us*100:.1f}%" if (_dur_total and _total_us) else "?"
        _parts = []
        for _path, _label in _pk_fields:
            _v = _pk_get(_k, _path)
            if _v is not None:
                if _label.endswith("_B") and isinstance(_v, (int, float)):
                    _parts.append(f"{_label}={_v / 2**20:.1f}MB")   # 字节 → MB 可读
                else:
                    _parts.append(f"{_label}={_fnum(_v)}")
        _pk = " | " + " ".join(_parts) if _parts else ""
        _dur_show = f"{_dur_total:.0f}us" if _dur_total else ("无耗时" if not _dur else f"{_dur}us")
        _lbl = f" x{_launch}" if _launch > 1 else ""
        lines.append(f"- {_name}: {_pct} ({_dur_show}{_lbl}){_pk}")

    # 本层附加字段: 只留非 kernel 级路径 (per-kernel 已在上表; summary.* 已在全局)
    _tier_extra = [(p, d) for p, d in TIER_FIELDS.get(tier, [])
                   if not p.startswith("kernels[") and not p.startswith("summary.")]
    if _tier_extra:
        lines.append("# ══ 本层附加字段 ══")
        lines += [_fmt(p, d) for p, d in _tier_extra]
    return "\n".join(lines)


class Scheduler:
    """v4 状态机调度器。"""

    def __init__(self, op_dir: Path, max_rounds: int = 200,
                 target_speedup: float = 1.5, use_llm: bool = True,
                 stub: bool = False, resume: bool = False):
        self.op_dir = op_dir
        self.max_rounds = max_rounds
        self.target_speedup = target_speedup
        self.use_llm = use_llm and not stub
        self.optimize_timeout = int(os.environ.get("OPTIMIZE_TIMEOUT", "3600"))
        self.outputs = _PROJECT / "outputs"
        self.kernel_name = op_dir.name
        self.kernel_dir = self.outputs / self.kernel_name
        self.traj_path = self.kernel_dir / "optimization_trajectory.json"
        self.traj = self._load_traj()
        self.stage_times = []   # ★每轮各阶段耗时收集 (stats 输出用)
        # ★默认每次初始化 (round1/tier1 重来, 避免读旧路径错位); --resume 才续跑
        if not resume or not self.traj.get("state", {}).get("current_kernel"):
            self._reset_state()
        else:
            ck = Path(self.traj["state"]["current_kernel"])
            self.current_kernel = ck if ck.exists() else (op_dir / "kernel_op.py")
            print(f"  [Scheduler] 续跑: 从 tier{self.traj['state'].get('tier')} "
                  f"round{self.traj['state'].get('round')} 继续, kernel={self.current_kernel}")

    def _reset_state(self):
        """初始化: 从头开始 (round1/tier1 读源文件)。"""
        self.traj = {"v": 4, "state": {"tier": 1, "round": 1, "best_speedup": 1.0,
                                       "baseline_ns": None, "num_kernels": None,
                                       "current_speedup": 1.0,   # 当前已接受 kernel 的加速比 (基线=1.0)
                                       "current_kernel": str(self.op_dir / "kernel_op.py"),
                                       "total_rounds": 0},       # ★D7: 含 promote 轮的总执行数 (防 promote 白耗 max_rounds)
                     "history": []}
        self.current_kernel = self.op_dir / "kernel_op.py"
        # ★D6: 保存原始 kernel 副本 → 环境漂移时重测 baseline 用它 (优化后 current_kernel 已不是原始版)
        try:
            self.kernel_dir.mkdir(parents=True, exist_ok=True)
            (self.kernel_dir / "baseline_kernel.py").write_text(
                self.current_kernel.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    # ── 轨迹 ──
    def _load_traj(self) -> dict:
        if self.traj_path.exists():
            traj = json.loads(self.traj_path.read_text(encoding="utf-8"))
            if traj.get("v") == 4:
                return traj
            # 旧版本 (v3) trajectory → 重置, 避免 tier/round 错位
            print("  [Scheduler] 检测到旧版本 trajectory, 重置为 v4")
        return {"v": 4, "state": {"tier": 1, "round": 1, "best_speedup": 1.0,
                                  "baseline_ns": None, "num_kernels": None,
                                  "current_speedup": 1.0},
                "history": []}

    def _save_traj(self):
        self.traj_path.parent.mkdir(parents=True, exist_ok=True)
        self.traj_path.write_text(json.dumps(self.traj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    def _write_timing_stats(self):
        """★每轮各阶段耗时统计 → outputs/<op>/stats/timing_stats.json (找项目自身瓶颈).
        aggregate: 各阶段 总/平均/占本轮总时%, 标出耗时最大的阶段."""
        if not self.stage_times:
            return
        stats_dir = self.kernel_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stages = ("collect_s", "diag_s", "fusion_s", "planner_s", "coder_s", "verify_s")
        total_round = sum(r["round_s"] for r in self.stage_times) or 1e-9
        agg = {}
        for k in stages:
            vals = [r[k] for r in self.stage_times]
            agg[k] = {"total_s": round(sum(vals), 2),
                      "avg_s": round(sum(vals) / len(vals), 2),
                      "pct_of_round": round(sum(vals) / total_round * 100, 1)}
        bottleneck = max(agg, key=lambda k: agg[k]["total_s"])
        out = {
            "op": self.kernel_name,
            "rounds": self.stage_times,
            "aggregate": agg,
            "bottleneck_stage": bottleneck,   # 耗时最大阶段 (项目自身瓶颈)
            "generated_at": datetime.now().isoformat(),
        }
        (stats_dir / "timing_stats.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        n = len(self.stage_times)
        total_all = sum(r["round_s"] for r in self.stage_times)
        avg_round = total_all / n if n else 0
        print(f"\n  [Stats] 各阶段耗时统计 → {stats_dir}/timing_stats.json")
        print(f"         共 {n} 轮, 总耗时 {total_all:.0f}s, 平均每轮 {avg_round:.0f}s")
        print(f"         瓶颈阶段: {bottleneck} (总 {agg[bottleneck]['total_s']}s, "
              f"占本轮总时 {agg[bottleneck]['pct_of_round']}%)")
        print(f"         各阶段平均: " + ", ".join(f"{k.replace('_s','')}={agg[k]['avg_s']}s" for k in stages))

    # ── 轮目录 ──
    def _round_dir(self, tier: int, rn: int) -> Path:
        return self.kernel_dir / TIER_NAMES.get(tier, "0x") / f"round{rn}"

    # ── ① 采集+解析 ──
    def _run_optimize(self, round_dir: Path, tier: int = 1) -> Optional[dict]:
        """调 run_optimize.sh <op_dir> <round_dir> [M N K] → 读 diagnosis.json。
        ★M/N/K 从当前 kernel 的 config 提取并传给 run_optimize (它设 MATMUL_M/N/K env):
          否则 run_optimize 默认 512 会覆盖 kernel_op.py 里的默认值 → baseline 尺寸与 verify 不一致,
          speedup 严重失真 (512³ 的 FLOPs 只有 2048³ 的 1/64).
        TIER 环境变量传给 run_optimize, 让它解析完自动产出 07_tier<N>_fields。"""
        run_sh = (_PROJECT / "analyzers" / "run_optimize.sh").as_posix()
        input_dir = self.current_kernel.parent    # round1=源目录; 后续=上一轮输出目录
        cmd = ["bash", run_sh, str(input_dir), str(round_dir)]
        mnk = _extract_mnk(self.current_kernel.read_text(encoding="utf-8")
                           if self.current_kernel.exists() else "")
        if mnk:
            cmd += [str(mnk[0]), str(mnk[1]), str(mnk[2])]   # ★传真实尺寸, 避免 512 默认覆盖
        print(f"  [Scheduler] {' '.join(cmd)} (TIER={tier}, kernel={self.current_kernel})")
        print(f"  ⏳ 采集进行中 (msprof 需几分钟, 期间无输出是正常的; 超时 {self.optimize_timeout}s)...")
        env = dict(os.environ)
        env["TIER"] = str(tier)
        # ★流式打印 run_optimize 输出 → 终端 + 运行日志 (outputs/<op>/optimization.log)
        try:
            # ★编码: backslashreplace 代替 replace — 非 UTF-8 字节转成 \xNN 保留信息, 不再变成 � 丢字节
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="backslashreplace", env=env)
            import threading as _th

            def _drain():
                for line in proc.stdout:
                    print(line, end="", flush=True)
            _t = _th.Thread(target=_drain, daemon=True)
            _t.start()
            try:
                proc.wait(timeout=self.optimize_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  ❌ run_optimize 超时 ({self.optimize_timeout}s), 看 {round_dir}/05_task/task_run.txt")
                return None
            _t.join()
        except Exception as e:
            print(f"  ❌ run_optimize 启动失败: {str(e)[:150]}")
            return None
        dgn = round_dir / "06_diagnosis" / "diagnosis.json"
        if dgn.exists():
            return json.loads(dgn.read_text(encoding="utf-8"))
        # 兜底: 产物也可能直接在 round_dir 下
        alt = round_dir / "diagnosis.json"
        if alt.exists():
            return json.loads(alt.read_text(encoding="utf-8"))
        print("  [Scheduler] ❌ diagnosis.json 未生成")
        return None

    def _print_kernel_breakdown(self, diagnosis: dict):
        """★终端打印每个算子的耗时占比 (端到端口径, 与 total_ns 同源, 多 launch 计入).
        total_ns = 所有 target kernel 单次耗时之和×1000; 每 kernel 占比 = 单次耗时×launch_count/端到端.
        附每 kernel 关键瓶颈字段, 优化一眼看打中谁."""
        ks = diagnosis.get("kernels") or []
        total_ns = (diagnosis.get("summary") or {}).get("total_ns")
        total_us = total_ns / 1000 if total_ns else None
        if not ks:
            return
        _tus = f"{total_us:.0f}us" if total_us else "?"
        print(f"  [每算子耗时占比] 端到端 total_ns={total_ns}ns ({_tus}), "
              f"{len(ks)} 个目标 kernel:")
        for _k in ks:
            _name = _k.get("kernel_name", "?")
            _dur = (_k.get("task") or {}).get("task_duration_us")
            _launch = _k.get("launch_count") or 1
            _dur_total = (_dur * _launch) if _dur else None
            _pct = f"{_dur_total / total_us * 100:.1f}%" if (_dur_total and total_us) else "?"
            _d = _k.get("deep") or {}
            _rl = _d.get("roofline") or {}
            _bn = _rl.get("bottleneck_type") or "?"
            _cu = _rl.get("compute_utilization")
            _mu = _rl.get("memory_utilization")
            _cu_s = f"{_cu:.2f}" if isinstance(_cu, (int, float)) else "?"
            _mu_s = f"{_mu:.2f}" if isinstance(_mu, (int, float)) else "?"
            _launch_s = f" x{_launch}" if _launch > 1 else ""
            _dur_s = f"{_dur_total:.0f}us" if _dur_total else "无耗时"
            print(f"    {_name}: {_dur_s} ({_pct}){_launch_s}  "
                  f"bottleneck={_bn} cube_util={_cu_s} mem_util={_mu_s}")

    # ── ③ 诊断: 按当前阶段筛选字段 → 写 07 (planner 只读这个) ──
    def _diagnose(self, diagnosis: dict, tier: int, round_dir: Path) -> str:
        extracted = extract_tier_fields(diagnosis, tier)
        d7 = round_dir / f"07_tier{tier}_fields"
        d7.mkdir(parents=True, exist_ok=True)
        (d7 / f"tier{tier}_fields.txt").write_text(extracted, encoding="utf-8")
        # 结构化 JSON: {字段说明: 值} (含全局摘要 + 当前 tier 字段)
        vals = {}
        for path, desc in GLOBAL_FIELDS + TIER_FIELDS.get(tier, []):
            vals[desc] = _get(diagnosis, path)
        (d7 / f"tier{tier}_fields.json").write_text(
            json.dumps(vals, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"  [诊断] Tier{tier} 筛出 {len(extracted.splitlines())} 行 → {d7}")
        return extracted

    # ── 融合专用: 编译 HIVM MLIR → nga run 依赖分析 → 08_fusion/ ──
    def _run_fusion(self, round_dir: Path):
        """Tier2 独有: 生成 hivm MLIR + nga run 融合分析, 写 round_dir/08_fusion/。"""
        from analyzers.run_hivm_fusion import run_fusion
        print("  [Scheduler] Tier2 融合: 编译 HIVM MLIR → nga run 分析依赖 → 08_fusion/")
        return run_fusion(self.current_kernel, round_dir, use_llm=self.use_llm)

    # ── ④ Planner (只喂当前阶段筛好的字段 + 融合分析 + Tier3 分块实测数据) ──
    def _plan(self, diagnosis: dict, extracted: str, tier: int, rn: int, round_dir: Path,
              fusion_analysis: Optional[dict] = None, tier3_sweep: Optional[dict] = None):
        from agents.planner import PlannerAgent, _extract_config_constants
        kernel_code = self.current_kernel.read_text(encoding="utf-8") if self.current_kernel.exists() else ""
        skill = (_PROJECT / "skills" / "triton-op-planner" / "SKILL.md")
        cfg = _extract_config_constants(kernel_code)
        # ★D1-Tier3: 分块实测数据追加进喂给 planner 的字段 (精简, 只给决策要用的 key 结果)
        if tier3_sweep and tier3_sweep.get("available"):
            sw = tier3_sweep
            lines = ["", "# ══ Tier3 分块实测 (sweep, ★决策依据 — 用数据不是猜) ══",
                     "各 config 实测 (ns 越小越快; 标★=实测最优; 标[当前]=当前块):"]
            for c in sw["configs"]:
                mark = " ★最优" if c is sw["best"] else (" [当前]" if c.get("is_current") else "")
                sp = f" ({c['speedup']}x)" if c.get("speedup") else ""
                lines.append(f"- {c['block']}: {c['ns']:.0f}ns{sp}{mark}")
            lines.append("★分块层决策指引: 若最优比当前明显快 → changes[] 直接采用实测最优块; "
                         "若当前已接近最优/无增益 → 分块已到位, promote 到下一层. "
                         "禁止再猜一个新的 BLOCK 值 (实测数据就是答案).")
            extracted = extracted + "\n" + "\n".join(lines)
        print(f"  [Planner] 输入: 07字段({len(extracted.splitlines())}行) + playbook_tier{tier} "
              f"+ {self.current_kernel}({len(kernel_code)}字符) + config[{cfg.splitlines()[0] if cfg else '?'}] "
              f"+ 历史{len(self.traj.get('history', []))}轮"
              + (" + Tier3分块实测数据" if (tier3_sweep and tier3_sweep.get("available")) else ""))
        print(f"  [Planner] 调 skill: {skill}")
        planner = PlannerAgent(use_llm=self.use_llm)
        plan = planner.generate_v4(
            extracted=extracted, tier=tier,
            history=self.traj.get("history", []),
            kernel_code=kernel_code,
            round_num=rn,
            op_dir=self.op_dir,
            fusion_analysis=fusion_analysis,
            round_dir=round_dir,
            current_kernel=self.current_kernel,
        )
        round_dir.mkdir(parents=True, exist_ok=True)
        plan_md = (f"# Tier{tier} Round{rn} Plan\n\n{plan.plan_text}\n\n"
                   f"## 提取字段\n{extracted}"
                   + (f"\n\n## 融合分析\n{json.dumps(fusion_analysis, ensure_ascii=False, indent=1)}"
                      if fusion_analysis else ""))
        (round_dir / "plan.md").write_text(plan_md, encoding="utf-8")
        n_changes = len(_extract_changes_from_plan(plan.plan_text))
        print(f"  [Planner] → {round_dir}/plan.md ({len(plan_md)}字符, changes[]={n_changes}项) "
              f"promote={plan.promote}")
        return plan

    # ── ⑤ Coder (读 current_kernel, 精确应用 changes[], 输出 round_dir/kernel_op.py) ──
    def _code(self, plan, rn: int, round_dir: Path, prev_err: str = "") -> tuple:
        """应用计划 → (优化后代码, 是否成功应用, 错误文本).
        ★B2: 返回 success/error — coder 应用失败(old_code没匹配/语法错/LLM超时/no-op)时,
        调度器把错误记进 history → 下一轮 planner 能看到, 不再重复提同样的错 old_code."""
        from agents.coder import CoderAgent, CoderResult
        original = self.current_kernel.read_text(encoding="utf-8") if self.current_kernel.exists() else ""
        skill = _PROJECT / "skills" / "triton-op-coder" / "SKILL.md"
        print(f"  [Coder] 读 {self.current_kernel} 的 changes[] (精确替换) + skill {skill}")
        coder = CoderAgent(use_llm=self.use_llm)
        try:
            result = coder.apply(original, plan.plan_text, prev_err, plan.tier,
                                 kernel_path=str(round_dir / "kernel_op.py"))
        except Exception as e:
            # ★任何编码异常(含 LLM 超时) → 不崩整个循环: 沿用原代码, 本轮失败
            print(f"  [Coder] ❌ 编码异常: {str(e)[:200]} → 沿用原代码, 本轮失败")
            result = CoderResult(success=False, optimized_code=original, diff="",
                                 error_message=f"编码异常: {str(e)[:200]}")
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "diff.patch").write_text(result.diff or "(no change)", encoding="utf-8")
        n_changes = len(_extract_changes_from_plan(plan.plan_text))
        if result.success:
            lines = result.lines_changed or 0
            print(f"  [Coder] ✅ 应用 {n_changes} 处 changes → {lines} 行改动 → diff.patch")
        else:
            print(f"  [Coder] ⚠ 未成功: {result.error_message[:200]}")
        return result.optimized_code, result.success, result.error_message

    # ── ⑥ 验证: 只跑 msprof 端到端 (验证本轮新 kernel) ──
    def _verify(self, round_dir: Path, baseline_ns: Optional[float]) -> dict:
        """只跑一次 msprof 端到端 → 端到端耗时 → 加速比。
        验证的是 round_dir/kernel_op.py (本轮 coder 的新输出)。
        传 num_kernels 给 verify 做行数合理性告警 (防漏记/循环丢失静默错数)."""
        kernel = round_dir / "kernel_op.py"
        try:
            from agents.verifier import verify_end_to_end
            nk = self.traj.get("state", {}).get("num_kernels")
            return verify_end_to_end(kernel, round_dir, baseline_ns, num_kernels=nk)
        except Exception as e:
            print(f"  [Scheduler] verify stub: {e}")
            # ★F1对齐: stub 也反推 ns (speedup=1.0 → ns=baseline), 不留 None 脏字段
            return {"ok": True, "ns": round(baseline_ns, 1) if baseline_ns else None,
                    "speedup": 1.0, "note": "stub(无真机)"}

    # ★D5: 场景泛化 sanity — 优化后 kernel 在相邻尺寸下仍正确 (防过拟合单一形状).
    #   默认关 (SANITY_VERIFY=1 开); 只动主维度 (×0.5/×2), 失败仅警告不阻断 (主验证已过).
    _SANITY_ENVS = {
        "matmul": ["MATMUL_M", "MATMUL_N", "MATMUL_K"],
        "attention_mlp": ["MATMUL_M", "MATMUL_N"],
        "rms_norm": ["RMS_M", "RMS_N"],
        "flash_attention": ["FA_SEQ"],
        "conv2d": ["CONV_H", "CONV_W"],
        "conv_bias_relu": ["CONV_H", "CONV_W"],
    }

    def _sanity_verify(self, kernel_path: Path):
        import subprocess
        op = self.kernel_name
        envs = self._SANITY_ENVS.get(op, [])
        if not envs or not kernel_path.exists():
            return
        # 主维度: matmul 用 config 的 M, 其余按 op 默认 (rms/attention/fa=2048, conv=64)
        dim_key = envs[0]
        base = _extract_mnk(kernel_path.read_text(encoding="utf-8"))
        if op == "matmul" and base:
            dim_val = base[0]
        else:
            dim_val = {"conv2d": 64, "conv_bias_relu": 64}.get(op, 2048)
        for scale, label in ((0.5, "半"), (2, "双")):
            new_dim = int(round(dim_val * scale / 16) * 16)   # 保持 16 倍数
            if new_dim <= 0 or new_dim == dim_val:
                continue
            env = dict(os.environ, KERNEL_LOOP="1", MATMUL_VERIFY="1",
                       **{dim_key: str(new_dim)})
            try:
                r = subprocess.run(["python3", str(kernel_path)], capture_output=True,
                                   text=True, encoding="utf-8", errors="backslashreplace",
                                   timeout=1800, env=env)
                out = (r.stdout or "") + (r.stderr or "")
                if "result check: PASS" in out:
                    print(f"    [Sanity] {label}尺寸({dim_key}={new_dim}) 正确性 PASS")
                else:
                    print(f"    ⚠ [Sanity] {label}尺寸({dim_key}={new_dim}) 未通过: {out.strip()[-200:]}")
            except Exception as e:
                print(f"    ⚠ [Sanity] {label}尺寸异常: {str(e)[:120]}")

    # ★D1-Tier3: 自动分块实测 — 到分块层时先跑一遍 L0 合法 BLOCK 候选, 收集各 config 实测数据
    #   (含当前块对比), **喂给 planner 决策** (不是这里写死写回). 确定性 autotune 思想.
    # 兜底: 任何异常/无可扫 → 返回 None 或 {"available": False}, 调用方继续正常走 LLM (不阻断迭代).
    # 返回精简数据 (不含原始 msprof 输出/报错全文, 只给 planner 决策要用的关键结果).
    def _tier3_sweep_data(self, tier: int, rn: int, round_dir: Path) -> Optional[dict]:
        try:
            from sweep_blocks import SWEEP, _read_current_block, _apply_block
        except Exception as e:
            return {"available": False, "error": f"sweep_blocks 不可用: {str(e)[:80]}"}
        cfg = SWEEP.get(self.kernel_name)
        kernel = self.current_kernel
        if cfg is None or not kernel.exists():
            return None                      # rms_norm 行级, 无自由分块参数 → 正常走 LLM
        code = kernel.read_text(encoding="utf-8")
        cur = _read_current_block(code, cfg["vars"])
        if cur is None:
            return {"available": False, "error": "读不到 BLOCK 值"}
        # 候选 = 当前块(作对比) + L0 合法变体; TIER3_SWEEP_CANDS 限数(+1 含当前)
        cands = [cur] + [c for c in cfg["cands"] if c != cur]
        _lim = int(os.environ.get("TIER3_SWEEP_CANDS", "0"))
        if _lim > 0:
            cands = cands[:_lim + 1]
        from agents.verifier import verify_end_to_end
        base = self.traj["state"].get("baseline_ns")
        # ★时间控制: sweep 只求"相对排序", 每 config 轻量计时 (VERIFY_LOOP=TIER3_SWEEP_LOOP 默认5,
        #   warmup=1) — 单 config ~1-2min, 8 config ~10-15min; 防 tiling 卡死拉长总运行.
        #   (最终 block 正确性由主流程 verify 严格查, sweep 阶段从轻)
        sweep_loop = int(os.environ.get("TIER3_SWEEP_LOOP", "5"))
        _ovl, _ovw = os.environ.get("VERIFY_LOOP"), os.environ.get("VERIFY_WARMUP")
        os.environ["VERIFY_LOOP"] = str(sweep_loop)
        os.environ["VERIFY_WARMUP"] = "1"
        # 产物目录: roundN/09_tier3_sweep/ (对齐 07_tier3_fields/08_fusion)
        sweep_dir = round_dir / "09_tier3_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        configs = []
        print(f"  [Tier3] 分块实测: 测 {len(cands)} 个 config (含当前 {cur}, 每 config {sweep_loop} 轮) → {sweep_dir}", flush=True)
        try:
            for vals in cands:
                try:
                    cand_code = _apply_block(code, cfg["vars"], vals)
                    cfg_rd = sweep_dir / ("config_" + "_".join(map(str, vals)))
                    cfg_rd.mkdir(parents=True, exist_ok=True)
                    (cfg_rd / "kernel_op.py").write_text(cand_code, encoding="utf-8")
                    v = verify_end_to_end(cfg_rd / "kernel_op.py", cfg_rd, None, num_kernels=None)
                    if v.get("ok") and v.get("ns"):
                        configs.append({"block": list(vals), "ns": v["ns"],
                                        "speedup": round(base / v["ns"], 3) if base else None,
                                        "is_current": list(vals) == list(cur)})
                        print(f"      {vals}: {v['ns']:.0f}ns", flush=True)
                    else:
                        configs.append({"block": list(vals), "ns": None, "is_current": list(vals) == list(cur),
                                        "error": str(v.get("error", ""))[:80]})
                        print(f"      {vals}: FAIL {str(v.get('error',''))[:80]}", flush=True)
                except Exception as e:
                    configs.append({"block": list(vals), "ns": None, "is_current": list(vals) == list(cur),
                                    "error": str(e)[:80]})
        finally:
            if _ovl is None:
                os.environ.pop("VERIFY_LOOP", None)
            else:
                os.environ["VERIFY_LOOP"] = _ovl
            if _ovw is None:
                os.environ.pop("VERIFY_WARMUP", None)
            else:
                os.environ["VERIFY_WARMUP"] = _ovw
        valid = [c for c in configs if c.get("ns")]
        if not valid:
            (sweep_dir / "sweep_result.json").write_text(
                json.dumps({"available": False, "error": "所有候选实测失败", "configs": configs},
                           ensure_ascii=False, indent=1), encoding="utf-8")
            return {"available": False, "error": "所有候选实测失败"}
        valid_sorted = sorted(valid, key=lambda c: c["ns"])
        # 写 09 报告 (planner 可读, 也留档)
        (sweep_dir / "sweep_result.json").write_text(json.dumps(
            {"available": True, "vars": list(cfg["vars"]), "configs": valid_sorted,
             "best": valid_sorted[0]}, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(sweep_dir / "sweep_result.txt", "w", encoding="utf-8") as f:
            f.write("══ Tier3 分块实测 (ns 越小越快; ★最优; [当前]) ══\n")
            for c in valid_sorted:
                mark = " ★最优" if c is valid_sorted[0] else (" [当前]" if c.get("is_current") else "")
                sp = f" ({c['speedup']}x)" if c.get("speedup") else ""
                f.write(f"  {c['block']}: {c['ns']:.0f}ns{sp}{mark}\n")
        print(f"  [Tier3] 实测报告 → {sweep_dir}")
        return {"available": True, "configs": valid_sorted, "best": valid_sorted[0],
                "vars": list(cfg["vars"])}

    # ★D6: 环境漂移防护 — 每 N 轮重测原始 baseline kernel + 当前 kernel,
    #   校正加速比基数 (长时运行跨时段温度/负载漂移会污染 baseline_ns).
    #   只在校正 (不动 history 旧值; 轨迹图上从本轮起用新基准).
    def _maybe_rebaseline(self, tier: int, rn: int):
        st = self.traj["state"]
        try:
            from agents.verifier import verify_end_to_end
            # 重测原始 baseline kernel (reset 时保存的副本)
            base_kernel = self.kernel_dir / "baseline_kernel.py"
            if not base_kernel.exists():
                return
            rb_dir = self.kernel_dir / "rebaseline"
            rb_dir.mkdir(parents=True, exist_ok=True)
            vb = verify_end_to_end(base_kernel, rb_dir / "base", None,
                                   num_kernels=st.get("num_kernels"))
            if not (vb.get("ok") and vb.get("ns")):
                print(f"  [重基准] R{rn} 重测原始 kernel 失败: {str(vb.get('error',''))[:120]} (跳过)")
                return
            new_base = vb["ns"]
            # 重测当前已接受 kernel → 新累计加速比
            cur = verify_end_to_end(self.current_kernel, rb_dir / "cur", None,
                                    num_kernels=st.get("num_kernels"))
            if not (cur.get("ok") and cur.get("ns")):
                print(f"  [重基准] R{rn} 重测当前 kernel 失败 (跳过, 沿用旧 baseline)")
                return
            new_speedup = new_base / cur["ns"] if cur["ns"] else None
            drift = new_base / st["baseline_ns"] if st.get("baseline_ns") else 1.0
            st["baseline_ns"] = new_base
            st["current_speedup"] = round(new_speedup, 4) if new_speedup else st.get("current_speedup", 1.0)
            st["last_rebase_round"] = rn
            # ★不进 history (避免与正常轮同 round 号, trajectory 图点重叠):
            #   REBASELINE 是测量事件不是优化轮, 只更新 state; 换基后后续轮 speedup 用新基准.
            self._save_traj()
            print(f"  [重基准] R{rn}: baseline {drift:.3f}x 漂移 → 新基准 {new_base:.0f}ns, "
                  f"当前 kernel 累计 {new_speedup:.3f}x (校正后, 不进 history)")
        except Exception as e:
            print(f"  [重基准] R{rn} 异常: {str(e)[:120]} (跳过)")

    # ── 主循环 (无 round0: 首轮采集即基准, 全部轮次直接进 outputs/<op>/<tier>/roundN) ──
    def run(self):
        st = self.traj["state"]
        tier, rn = st.get("tier", 1), st.get("round", 1)
        tgt = f"目标 {self.target_speedup}x" if self.target_speedup > 0 else "无目标(跑满 max_rounds 看最优)"
        print(f"══ Scheduler: {self.kernel_name} {tgt} ══")

        # Warm-up: 首次 nga run 冷启动(模型加载)可能很久, 提前预热
        if self.use_llm:
            print("  [Warm-up] 预热 nga run (首次冷启动)...")
            try:
                from agents.llm_client import LLMClient
                c = LLMClient()
                if c.mode == "cli":
                    c.chat("你是测试", "只回复 OK")
                    print("  [Warm-up] ✅ nga run 预热完成")
                else:
                    print(f"  [Warm-up] 模式={c.mode} (非 cli, 跳过)")
            except Exception as e:
                print(f"  [Warm-up] ⚠ 预热失败: {str(e)[:120]} (继续, 后续调用再试)")

        total_start = time.time()
        coll_fail = 0   # ★H2: 连续采集失败计数 (失败重试/跳过, 不是一停到底)
        total_rounds = st.get("total_rounds", 0)
        # ★D7: 总预算 = max_rounds + 已用 promote 额度 (promote 轮免费, 不挤占有效优化轮)
        while total_rounds < self.max_rounds + int(st.get("promote_budget", 0)):
            total_rounds += 1
            round_dir = self._round_dir(tier, rn)
            round_start = time.time()
            print(f"\n══ Tier{tier}({TIER_LABEL.get(tier)}) Round{rn} ══")

            # ★D6: 环境漂移防护 — 每 N 轮重测 baseline (默认10), 校正加速比基数
            rebase_every = int(os.environ.get("REBASELINE_EVERY", "10"))
            if (rebase_every > 0 and rn > 1 and st.get("baseline_ns")
                    and st.get("last_rebase_round", 0) != rn
                    and rn % rebase_every == 0):
                self._maybe_rebaseline(tier, rn)

            # ① 采集+解析 (run_optimize 自动产出 07_tier{N}_fields)
            _t0 = time.time()
            diagnosis = self._run_optimize(round_dir, tier)
            if not diagnosis:
                # ★H2: 采集失败不再一票否决整个 run — 先重试同轮 1 次,
                #   仍失败则跳过本轮(沿用当前 kernel 进下一轮), 连续 3 次才停
                coll_fail += 1
                if coll_fail <= 1:
                    print(f"  ⚠ 采集失败(第{coll_fail}次), 重试同一轮 {round_dir}...")
                    continue
                if coll_fail >= 3:
                    print(f"  ⚠ 连续 {coll_fail} 次采集失败, 停止")
                    break
                ps = st.get("current_speedup", 1.0)
                self.traj["history"].append({"round": rn, "tier": tier,
                    "strategy": "采集失败跳过", "change": "",
                    "speedup": round(ps, 4), "prev_speedup": round(ps, 4),
                    "ns": None, "decision": "FAIL", "result": "FAIL",
                    "error": f"采集失败 {coll_fail} 次"})
                print(f"  ⚠ 采集失败(第{coll_fail}次), 跳过本轮 → R{rn+1} (沿用当前 kernel)")
                st["round"] = rn + 1
                rn += 1
                self._save_traj()
                continue
            coll_fail = 0
            t_collect = time.time() - _t0
            print(f"  ⏱ ①采集+解析: {t_collect:.1f}s")
            # 诊断摘要 + ★每算子耗时占比 (端到端口径, 与 total_ns 同源, 优化打中占比大的 kernel)
            ks0 = diagnosis.get("summary", {})
            k0 = (diagnosis.get("kernels") or [{}])[0]
            ro = (k0.get("deep") or {}).get("roofline", {})
            print(f"  [诊断] kernels={ks0.get('num_kernels')} total_ns={ks0.get('total_ns')} "
                  f"bottleneck={ro.get('bottleneck_type')} 产物→{round_dir}")
            self._print_kernel_breakdown(diagnosis)
            # 首轮 (原始 kernel_op.py 未改) 采集 = 基准
            if st.get("baseline_ns") is None:
                st["baseline_ns"] = ks0.get("total_ns")
                st["num_kernels"] = ks0.get("num_kernels")
                mnk = _extract_mnk(self.current_kernel.read_text(encoding="utf-8")
                                   if self.current_kernel.exists() else "")
                if mnk:
                    st["baseline_mnk"] = list(mnk)   # 记 baseline 尺寸, 防后续轮跨尺寸失真
                # initial_tflops (供轨迹图):
                #   ★优先用诊断的真实 cube_fops 之和 (多 matmul/多 kernel 正确, MLP=2×2MNK)
                #   兜底用 config 的 2MNK (单 matmul)
                cube_fops = sum((k.get("deep") or {}).get("compute", {}).get("cube_fops") or 0
                                for k in (diagnosis.get("kernels") or []))
                if cube_fops and st["baseline_ns"]:
                    st["initial_tflops"] = round(
                        cube_fops / (st["baseline_ns"] / 1e9) / 1e12, 2)
                elif mnk and st["baseline_ns"]:
                    st["initial_tflops"] = round(
                        2 * mnk[0] * mnk[1] * mnk[2] / (st["baseline_ns"] / 1e9) / 1e12, 2)
                # ★F4: PyTorch 基准 — 按算子显式映射 (bench_910b3/bench_pytorch_*.py 输出),
                #   显式映射缺时回退旧启发式 (多 kernel→MLP, 单→单 matmul); 必须与 op 同尺寸同 dtype (见 H5/E4).
                try:
                    from bench_910b3.bench_config import PT_BENCH_MAP
                except Exception:
                    PT_BENCH_MAP = {}
                pt_file = PT_BENCH_MAP.get(self.kernel_dir.name)
                if not pt_file:
                    nk = st.get("num_kernels") or 0
                    pt_file = "pytorch_mlp_tflops.json" if nk > 1 else "pytorch_tflops.json"
                pt = _PROJECT / "bench_910b3" / pt_file
                if pt.exists():
                    try:
                        st["pytorch_tflops"] = json.loads(pt.read_text(encoding="utf-8"))["tflops"]
                        st["pytorch_baseline"] = pt_file
                    except Exception:
                        pass
                # ★基准复测: 诊断 total_ns 是单次 msprof 采样(噪声大),
                #   用 verify 机制 (warmup + VERIFY_RUNS 轮 msprof 平均) 重测源 kernel,
                #   与后续轮完全同口径, 加速比才可信. 默认开, VERIFY_BASELINE=0 跳过.
                if os.environ.get("VERIFY_BASELINE", "1") == "1":
                    try:
                        from agents.verifier import verify_end_to_end
                        base_rd = self.kernel_dir / "baseline_verify"
                        if base_rd.exists():
                            import shutil as _sh
                            _sh.rmtree(base_rd)   # ★P1: 每次重测基准前清干净 (防跨 run 累积)
                        base_rd.mkdir(parents=True, exist_ok=True)
                        vb = verify_end_to_end(self.current_kernel, base_rd, None,
                                               num_kernels=st.get("num_kernels"))
                        if vb.get("ok") and vb.get("ns"):
                            st["baseline_ns"] = vb["ns"]
                            if cube_fops:
                                st["initial_tflops"] = round(
                                    cube_fops / (vb["ns"] / 1e9) / 1e12, 2)
                            print(f"  [基准] 复测平均 baseline_ns={vb['ns']}ns "
                                  f"(warmup+msprof 平均, 与后续轮同口径)")
                        else:
                            berr = vb.get("error", "")
                            print(f"  [基准] 复测失败: {berr[:200]} → 用诊断 total_ns")
                            if "正确性" in berr:
                                # ★A3: 源 kernel 正确性校验失败 → 源代码本身算错, 后面优化全白跑 → 停
                                print("  ⛔ 源 kernel 正确性校验失败 → 停止 (源代码算错了, 先修 input/<op>)")
                                st["round"] = rn + 1
                                self._save_traj()
                                break
                    except Exception as e:
                        print(f"  [基准] 复测失败({str(e)[:100]}), 用诊断 total_ns")
                if st.get("baseline_ns") is None:
                    # ★基准必须能算加速比: 诊断 total_ns 和验证复测都失败 → 停, 别带着假 baseline 跑
                    print("  ⛔ 基准未设置 (诊断 total_ns 与验证复测都失败) → 停止, 先修采集/验证")
                    st["round"] = rn + 1
                    self._save_traj()
                    break
                self._save_traj()
                print(f"  [基准] total_ns={ks0.get('total_ns')} kernels={ks0.get('num_kernels')} "
                      f"initial_tflops={st.get('initial_tflops')} (加速比基准)")

            # 尺寸一致性 guard: 若本轮 kernel 的 M/N/K 与 baseline 不同 → 加速比跨尺寸失真
            bmnk = st.get("baseline_mnk")
            if bmnk:
                cmnk = _extract_mnk(self.current_kernel.read_text(encoding="utf-8")
                                    if self.current_kernel.exists() else "")
                if cmnk and tuple(cmnk) != tuple(bmnk):
                    print(f"  ⚠ 尺寸变化! baseline M/N/K={bmnk}, 当前={list(cmnk)} "
                          f"→ 加速比跨尺寸失真, 检查优化是否误改 M/N/K")

            # ③ 诊断: 按当前阶段筛选字段 → 写 07
            _t0 = time.time()
            extracted = self._diagnose(diagnosis, tier, round_dir)
            t_diag = time.time() - _t0
            print(f"  ⏱ ②诊断筛字段: {t_diag:.1f}s")

            # ③.5 Tier2 融合: 多走一步 — 编译 HIVM MLIR → nga run 分析依赖 → 08_fusion/
            fusion_analysis = None
            t_fusion = 0.0
            if tier == 2:
                _t0 = time.time()
                fusion_analysis = self._run_fusion(round_dir)
                t_fusion = time.time() - _t0
                print(f"  ⏱ ③融合分析: {t_fusion:.1f}s")

            # ★D1-Tier3: 自动分块实测 — 进分块层先跑候选 BLOCK 收集各 config 实测数据,
            #   喂给 planner 决策 (不是这里猜/写死). 报错/无可扫 → None, 正常走 LLM (兜底).
            #   TIER3_SWEEP=1 默认; 每 op 首次进 Tier3 触发一次.
            # ★计时: t_plan 从 sweep 前开始计 (sweep 是 planner 决策的数据准备, 计入 planner 阶段)
            _t_plan0 = time.time()
            tier3_sweep = None
            if tier == 3 and os.environ.get("TIER3_SWEEP", "1") == "1" and not st.get("tier3_swept"):
                st["tier3_swept"] = True
                tier3_sweep = self._tier3_sweep_data(tier, rn, round_dir)
                if tier3_sweep and tier3_sweep.get("available"):
                    print(f"  ⏱ Tier3 分块实测完成: {len(tier3_sweep['configs'])} config, "
                          f"最优 {tier3_sweep['best']['block']} {tier3_sweep['best']['ns']:.0f}ns "
                          f"({time.time()-_t_plan0:.0f}s) → 数据喂 planner 决策")
                else:
                    print(f"  [Tier3] 分块实测不可用 "
                          f"({(tier3_sweep or {}).get('error','行级/无候选')}) → 走 LLM (兜底)")

            # ④ Planner → plan + 晋升决策 (只喂 07 筛好的字段 + 融合分析 + Tier3 分块实测数据)
            plan = self._plan(diagnosis, extracted, tier, rn, round_dir, fusion_analysis, tier3_sweep)
            t_plan = time.time() - _t_plan0
            print(f"  ⏱ ④Planner: {t_plan:.1f}s")

            # ⑤ Coder → 改代码 / ★晋升轮原样输出, 输出到 round_dir/kernel_op.py (不碰源文件)
            # ★计时修复: 重置 _t0 (之前没重置, ⑤ 把 ④ planner 的时间也算进去了)
            _t0 = time.time()
            t_code = t_verify = 0.0
            prev_err, new_code = "", ""
            round_kernel = round_dir / "kernel_op.py"
            prev_speedup = st.get("current_speedup", 1.0)   # 上一轮已接受 kernel 的加速比 (基线=1.0)
            kept = False                                      # 本轮是否被采纳进 kernel 链
            pre_code = self.current_kernel.read_text(encoding="utf-8") \
                if self.current_kernel.exists() else ""    # NOOP 对比用 (改之前的版本)
            if getattr(plan, "promote", False):
                # ★晋升轮: 不调 LLM 改码, 原样拷贝当前 kernel → roundN/kernel_op.py
                #   (保证每个 round 目录格式一致: 都有 kernel_op.py + diff.patch)
                #   ★下一轮读 current_kernel.parent/kernel_op.py = 本轮 (链连续, 不回 input)
                round_kernel.write_text(pre_code, encoding="utf-8")
                (round_dir / "diff.patch").write_text("(promote, 无代码改动)", encoding="utf-8")
                speedup = prev_speedup                        # 未改码 → 加速比与上一轮相同
                _bns = st.get("baseline_ns")
                v = {"ok": True,                             # ★F1: ns 反推, 不再留 None 脏字段
                     "ns": round(_bns / speedup, 1) if _bns and speedup else None,
                     "speedup": speedup, "note": "promote轮(无改动)"}
                self.current_kernel = round_kernel
                st["current_kernel"] = str(round_kernel)
                st["current_speedup"] = round(speedup, 4)
                kept = True
                self._save_traj()
            else:
                v = None
                for attempt in range(3):
                    _tc = time.time()
                    new_code, code_ok, code_err = self._code(plan, rn, round_dir, prev_err)
                    t_code += time.time() - _tc
                    round_kernel.write_text(new_code, encoding="utf-8")
                    if not code_ok:
                        # ★B2: coder 应用失败(old_code没匹配/语法错/超时/no-op) → 错误进 prev_err → history →
                        #   planner 下轮可见; 不测假加速比, 重试(下次带错误走 LLM 修复)
                        prev_err = code_err
                        print(f"  ⚠ coder 未成功应用(第{attempt+1}次): {code_err[:160]}...")
                        continue
                    # ⑥ 验证 (只 msprof 端到端, 验证本轮新 kernel)
                    _tv = time.time()
                    v = self._verify(round_dir, st.get("baseline_ns"))
                    t_verify += time.time() - _tv
                    if v.get("ok"):
                        # ★保留判定: speedup 始终 = 初始基线/本轮 (累计, 输出的就是这个);
                        #   但"是否采纳进 kernel 链"对比上一轮已接受的加速比 (prev_speedup)
                        speedup = v.get("speedup", 1.0) or 1.0   # None→1.0 防御 (baseline 缺失时 verify 返回 None)
                        floor = _keep_floor()
                        if speedup >= prev_speedup * floor:
                            self.current_kernel = round_kernel
                            st["current_kernel"] = str(round_kernel)
                            st["current_speedup"] = round(speedup, 4)
                            kept = True
                            # ★D5: 场景泛化 sanity — 采纳后对相邻尺寸做正确性检查 (默认关 SANITY_VERIFY=1)
                            if os.environ.get("SANITY_VERIFY", "0") == "1":
                                self._sanity_verify(round_kernel)
                        else:
                            print(f"  ↩ 回退: 本轮 {speedup:.3f}x < 上一轮 {prev_speedup:.3f}x×{floor} (噪声地板), 沿用上一轮 kernel")
                        self._save_traj()
                        break
                    prev_err = v.get("error", "unknown error")
                    print(f"  ⚠ 运行失败(第{attempt+1}次): {prev_err[:200]}... 回传 Coder 同轮改")
                if v is None:
                    # ★3次都是 coder 应用失败 → 本轮 FAIL (不产生假 speedup), 错误已在 prev_err → history
                    v = {"ok": False, "error": f"coder 连续3次未成功应用: {(prev_err or '')[:160]}",
                         "speedup": 1.0, "ns": None}

            # ★兜底: 保证每个 round 目录都有 kernel_op.py (格式一致)
            if not round_kernel.exists():
                round_kernel.write_text(pre_code or "", encoding="utf-8")

            speedup = v.get("speedup", 1.0) or 1.0   # None→1.0 防御 (round1 已保证 baseline 存在, 正常到不了)
            ns = v.get("ns")
            print(f"  加速比: {speedup:.3f}x (vs 初始基线 {st.get('baseline_ns')}ns; 上一轮 {prev_speedup:.3f}x)"
                  + ("  ✅采纳" if kept else "  ↩未采纳"))
            t_round = time.time() - round_start
            print(f"  ⏱ ⑤Coder: {t_code:.1f}s")
            print(f"  ⏱ ⑥Verify: {t_verify:.1f}s")
            print(f"  ⏱ 本轮总时: {t_round:.1f}s  (总用时 {time.time()-total_start:.1f}s)")
            # ★每轮各阶段耗时收集 → stats (找项目自身瓶颈)
            self.stage_times.append({
                "round": rn, "tier": tier,
                "collect_s": round(t_collect, 2), "diag_s": round(t_diag, 2),
                "fusion_s": round(t_fusion, 2), "planner_s": round(t_plan, 2),
                "coder_s": round(t_code, 2), "verify_s": round(t_verify, 2),
                "round_s": round(t_round, 2),
            })

            # ⑦ 记录 + 晋升决策 (hist 记"改了啥+结果"梗概, 让 planner 知道试过什么)
            ok = v.get("ok", False)
            # NOOP 检测: 本轮写出的 kernel 和改之前的 pre_code 一模一样 (coder 没成功应用)
            noop = False
            if round_kernel.exists() and pre_code:
                try:
                    noop = (round_kernel.read_text(encoding="utf-8") == pre_code)
                except Exception:
                    pass
            result = "NOOP" if noop else ("OK" if ok else "FAIL")
            decision = "KEEP" if kept else ("FAIL" if not ok else "REVERT")
            hist = {"round": rn, "tier": tier, "strategy": plan.strategy,
                    # ★D2: expected_impact = planner 建议的预期加速比 (→ 下轮反馈"预期vs实际", 学习闭环)
                    "expected_impact": getattr(plan, "expected_impact", ""),
                    # ★change = 简短梗概 (planner 历史上下文 + 轨迹图标签用, 截断省 token);
                    "change": _summarize_changes(plan),       # 例: "BLOCK_M,BLOCK_N,BLOCK_K=64,64,64"
                    # ★changes_full = 完整 changes[] 数组 (old_code/new_code 全文, 审计/复盘不丢信息)
                    "changes_full": _extract_changes_from_plan(plan.plan_text) or [],
                    "speedup": round(speedup, 4),             # 加速比 = 初始基线/本轮 (累计)
                    "prev_speedup": round(prev_speedup, 4),   # 上一轮已接受 kernel 的加速比 (保留判定用)
                    "ns": ns, "decision": decision, "result": result,
                    "error": (prev_err[:120] if prev_err else "")}
            # ★F3: 每轮真实 tflops (kernel 结构变化后 FLOPs 变, 轨迹图用 hist 值, 不再 initial×speedup 失真)
            _cf = sum((k.get("deep") or {}).get("compute", {}).get("cube_fops") or 0
                      for k in (diagnosis.get("kernels") or []))
            if _cf and ns:
                hist["tflops"] = round(_cf / (ns / 1e9) / 1e12, 2)
            if st.get("best_speedup") is None or speedup > st["best_speedup"]:
                st["best_speedup"] = speedup
            self.traj["history"].append(hist)

            # 晋升决策: planner.promote (读瓶颈判断) + 连续3轮无改进兜底 + 达标/到Tier6停止
            planner_promote = getattr(plan, "promote", False)
            # ★无改进 = 本轮加速比没超过上一轮已接受×噪声地板 (speedup <= prev_speedup×floor);
            #   数"本 tier 连续无改进"轮数 (跨 tier 边界即断) — 与 KEEP 地板同步
            no_improve = 0
            floor = _keep_floor()
            for h in reversed(self.traj["history"]):
                if h.get("tier") != tier:
                    break
                if h.get("speedup", 0) > h.get("prev_speedup", 0) * floor:
                    break
                no_improve += 1
            if self.target_speedup > 0 and speedup >= self.target_speedup:
                # ★D3: 达标不硬停 — 继续探后续层确认无更大空间, 由 no_improve/max_rounds 收尾.
                #   (目标改成 -1 → 不再触发停; 否则一轮达标就过早停, 错过 Tier3+ 更大优化空间)
                print(f"  🎯 已达标 {speedup:.3f}x ≥ target {self.target_speedup}x — 继续探后续层找更大空间")
                self.target_speedup = -1
            if planner_promote:
                target = getattr(plan, "promote_to", 0) or 0
                if target and 1 <= target <= 6 and target != tier:
                    # ★尊重 planner 的目标层: 支持回退前层(算法/融合) 和 晋升后层
                    direction = "回退" if target < tier else "晋升"
                    # ★防死循环: 回退到已充分探索(≥3轮)的层 → 拒绝, 改晋升
                    #   (planner 可能 T3 说"前面有空间"→回 T1 → T1 又跳回 T3 → 无限循环)
                    _t_rounds = {}
                    for _h in self.traj["history"]:
                        _t_rounds[_h.get("tier")] = _t_rounds.get(_h.get("tier"), 0) + 1
                    if target < tier and _t_rounds.get(target, 0) >= 3:
                        print(f"  ⛔ 拒绝回退: Tier{target} 已探索 {_t_rounds[target]} 轮 (防 T{target}↔T{tier} 死循环), 改晋升")
                        if tier >= 6:
                            print(f"  ⛔ 且已到 Tier6, 停止")
                            st["round"] = rn + 1
                            self._save_traj()
                            break
                        print(f"  → 晋升 Tier{tier}→Tier{tier+1} (回退被拒)")
                        tier += 1
                        st["tier"] = tier
                    else:
                        print(f"  → {direction} Tier{tier}→Tier{target} "
                              f"(planner: {getattr(plan,'promote_reason','')})")
                        tier = target
                        st["tier"] = tier
                elif tier >= 6:
                    print(f"  ⛔ planner 判瓶颈已非本tier且到Tier6, 停止 ({getattr(plan,'promote_reason','')})")
                    st["round"] = rn + 1
                    self._save_traj()
                    break
                else:
                    print(f"  → 晋升 Tier{tier}→Tier{tier+1} (planner: {getattr(plan,'promote_reason','瓶颈不属本tier')})")
                    tier += 1
                    st["tier"] = tier
            elif no_improve >= 3 or tier >= 6:
                if tier >= 6:
                    if no_improve >= 3:
                        print("  ⛔ Tier6 连续3轮无改进, 停止")
                        st["round"] = rn + 1
                        self._save_traj()
                        break
                    # ★B1修复: Tier6 有改进 → 继续 (README 写的"连续无改进才停"; 原来一轮即停是 bug)
                    print(f"  → Tier6 有改进(no_improve={no_improve}), 继续")
                else:
                    print(f"  → 晋升 Tier{tier}→Tier{tier+1} (本tier连续{no_improve}轮无改进)")
                    tier += 1
                    st["tier"] = tier
            # ★D7: promote 轮免费 — 不消耗 max_rounds 配额 (用 promote_budget 扩总预算),
            #   但 rn 编号统一推进 (历史/目录编号一致, 不被 promote 打乱)
            if getattr(plan, "promote", False):
                st["promote_budget"] = int(st.get("promote_budget", 0)) + 1
                print(f"  (promote 轮免费, 已用 {st['promote_budget']} 次 promote 额度)")
            st["round"] = rn + 1
            rn += 1
            st["total_rounds"] = total_rounds   # 总执行轮 (含 promote)
            self._save_traj()

        self._write_timing_stats()   # ★各阶段耗时统计 (项目自身瓶颈)
        # ★最后产出汇总: 总轮次/总耗时/平均每轮/最终 kernel 的 ns 与 speedup
        _bs = st.get("baseline_ns")
        _cur_ns = round(_bs / st.get("current_speedup", 1.0), 1) if (_bs and st.get("current_speedup")) else None
        _prom = st.get("promote_budget", 0)
        _n_eff = st.get("total_rounds", 0) - _prom
        print(f"\n══ 完成: best_speedup={st.get('best_speedup')}x ══")
        print(f"  [产出] 总执行 {st.get('total_rounds', 0)} 轮 (含 {_prom} 次 promote, 有效优化 {_n_eff} 轮), "
              f"总耗时 {time.time()-total_start:.0f}s")
        print(f"  [产出] 最终 kernel 累计加速比 {st.get('current_speedup')}x "
              f"(baseline {_bs}ns → 当前 {_cur_ns}ns); best {st.get('best_speedup')}x")
        print(f"  [产出] 最终 kernel 文件 → {st.get('current_kernel')}")
        return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="v4 Scheduler")
    p.add_argument("op_dir", type=str)
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--target", type=float, default=1.5)
    p.add_argument("--stub", action="store_true", help="不调 LLM/真机, 用 stub")
    args = p.parse_args()
    s = Scheduler(Path(args.op_dir), max_rounds=args.max_rounds,
                  target_speedup=args.target, stub=args.stub)
    sys.exit(s.run())
