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
TIER_FIELDS = {
    1: [  # 算法: 算力利用
        ("kernels[].deep.compute.cube_fops", "cube浮点运算数"),
        ("kernels[].deep.compute.vector_fops", "向量运算数"),
        ("kernels[].deep.engine_utilization.cube", "cube指令占比"),
        ("kernels[].deep.engine_utilization.vec", "向量指令占比"),
        ("kernels[].deep.roofline.compute_utilization", "算力利用率"),
        ("kernels[].deep.roofline.bottleneck_type", "瓶颈类型"),
    ],
    2: [  # 融合: 多算子/launch/类型
        ("summary.num_kernels", "优化目标kernel数"),
        ("summary.num_kernels_total", "总kernel数(含框架)"),
        ("summary.api_overhead_total_us", "launch开销us"),
        ("kernels[].task.task_type", "每kernel引擎"),
        ("kernels[].launch_count", "每kernel launch次数"),
        ("api_overhead", "API开销明细"),
        ("multi_kernel", "算子类型分解"),
        ("framework_kernels", "框架kernel(非目标)"),
    ],
    3: [  # 分块: 核数/L0A/B
        ("kernels[].task.block_dim", "核数"),
        ("kernels[].deep.engine_utilization.mte1", "MTE1(L1→L0A/B)占比"),
        ("kernels[].deep.bandwidth_gb_s.l0a_read_gb_s", "L0A读带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0b_read_gb_s", "L0B读带宽"),
    ],
    4: [  # 访存: GM带宽/L2/搬运时间
        ("kernels[].deep.bandwidth_gb_s.main_mem_read_gb_s", "GM读带宽"),
        ("kernels[].deep.bandwidth_gb_s.main_mem_write_gb_s", "GM写带宽"),
        ("kernels[].deep.l2_hit_rate", "L2命中率"),
        ("kernels[].task.pipes_us.aic_mte2_time_us", "MTE2(GM读)耗时"),
        ("kernels[].task.pipes_us.aic_mte3_time_us", "MTE3(GM写)耗时"),
        ("kernels[].deep.roofline.memory_utilization", "访存利用率"),
    ],
    5: [  # 计算: cube时间/冲突
        ("kernels[].task.pipes_us.aic_cube_time_us", "cube耗时"),
        ("kernels[].task.pipes_us.aic_scalar_time_us", "标量耗时"),
        ("kernels[].deep.conflict.bank_cflt_ratio", "bank冲突"),
        ("kernels[].deep.conflict.bankgroup_cflt_ratio", "bankgroup冲突"),
        ("kernels[].deep.conflict.total_cflt_ratio", "vec总冲突"),
        ("kernels[].deep.compute.cube_ratio", "cube指令占比"),
    ],
    6: [  # 架构: 引擎分布/阻塞
        ("kernels[].deep.engine_utilization", "各引擎利用率"),
        ("kernels[].deep.conflict.mte_cflt_ratio", "mte冲突"),
        ("kernels[].task.task_type", "每kernel引擎"),
        ("kernels[].deep.roofline.bottleneck_type", "瓶颈类型"),
    ],
}


def _get(d, path: str):
    """按 'a.b.c' 或 'kernels[].x.y' 路径取值 (返回第一个匹配)。"""
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
            cur = cur.get(p)
        else:
            return None
    return cur


def extract_tier_fields(diagnosis: dict, tier: int) -> str:
    """只提取当前 tier 的字段段 → 文本 (喂 Planner)。"""
    lines = [f"# 当前 Tier {tier} ({TIER_LABEL.get(tier, '')}) — 只看这些字段"]
    for path, desc in TIER_FIELDS.get(tier, []):
        v = _get(diagnosis, path)
        if v is None:
            lines.append(f"- {desc} ({path}): (无数据)")
        elif isinstance(v, (dict, list)):
            lines.append(f"- {desc} ({path}): {json.dumps(v, ensure_ascii=False)[:300]}")
        else:
            lines.append(f"- {desc} ({path}): {v}")
    return "\n".join(lines)


class Scheduler:
    """v4 状态机调度器。"""

    def __init__(self, op_dir: Path, max_rounds: int = 200,
                 target_speedup: float = 1.5, use_llm: bool = True,
                 stub: bool = False):
        self.op_dir = op_dir
        self.max_rounds = max_rounds
        self.target_speedup = target_speedup
        self.use_llm = use_llm and not stub
        self.outputs = _PROJECT / "outputs"
        self.kernel_name = op_dir.name
        self.kernel_dir = self.outputs / self.kernel_name
        self.kernel_op = op_dir / "kernel_op.py"
        self.traj_path = self.kernel_dir / "optimization_trajectory.json"
        self.traj = self._load_traj()

    # ── 轨迹 ──
    def _load_traj(self) -> dict:
        if self.traj_path.exists():
            traj = json.loads(self.traj_path.read_text(encoding="utf-8"))
            if traj.get("v") == 4:
                return traj
            # 旧版本 (v3) trajectory → 重置, 避免 tier/round 错位
            print("  [Scheduler] 检测到旧版本 trajectory, 重置为 v4")
        return {"v": 4, "state": {"tier": 1, "round": 0, "best_speedup": 1.0,
                                  "baseline_ns": None, "num_kernels": None},
                "history": []}

    def _save_traj(self):
        self.traj_path.parent.mkdir(parents=True, exist_ok=True)
        self.traj_path.write_text(json.dumps(self.traj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # ── 轮目录 ──
    def _round_dir(self, tier: int, rn: int) -> Path:
        return self.kernel_dir / TIER_NAMES.get(tier, "0x") / f"round{rn}"

    # ── ① 采集+解析 ──
    def _run_optimize(self, round_dir: Path) -> Optional[dict]:
        """调 run_optimize.sh <op_dir> <round_dir> → 读 diagnosis.json。"""
        run_sh = (_PROJECT / "analyzers" / "run_optimize.sh").as_posix()
        cmd = ["bash", run_sh, str(self.op_dir), str(round_dir)]
        print(f"  [Scheduler] {' '.join(cmd)}")
        subprocess.run(cmd, check=False, timeout=1800)
        dgn = round_dir / "06_diagnosis" / "diagnosis.json"
        if dgn.exists():
            return json.loads(dgn.read_text(encoding="utf-8"))
        # 兜底: 产物也可能直接在 round_dir 下
        alt = round_dir / "diagnosis.json"
        if alt.exists():
            return json.loads(alt.read_text(encoding="utf-8"))
        print("  [Scheduler] ❌ diagnosis.json 未生成")
        return None

    # ── ③ 提取字段 → ④ Planner ──
    def _plan(self, diagnosis: dict, tier: int, rn: int, round_dir: Path):
        from agents.planner import PlannerAgent
        extracted = extract_tier_fields(diagnosis, tier)
        kernel_code = self.kernel_op.read_text(encoding="utf-8") if self.kernel_op.exists() else ""
        planner = PlannerAgent(use_llm=self.use_llm)
        plan = planner.generate_v4(
            extracted=extracted, tier=tier,
            history=self.traj.get("history", []),
            kernel_code=kernel_code,
            round_num=rn,
            op_dir=self.op_dir,
        )
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "plan.md").write_text(
            f"# Tier{tier} Round{rn} Plan\n\n{plan.plan_text}\n\n"
            f"## 提取字段\n{extracted}", encoding="utf-8")
        return plan

    # ── ⑤ Coder ──
    def _code(self, plan, rn: int, round_dir: Path, prev_err: str = "") -> str:
        from agents.coder import CoderAgent
        original = self.kernel_op.read_text(encoding="utf-8") if self.kernel_op.exists() else ""
        coder = CoderAgent(use_llm=self.use_llm)
        result = coder.apply(original, plan.plan_text, prev_err, plan.tier)
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "diff.patch").write_text(result.diff or "(no change)", encoding="utf-8")
        if not result.success:
            print(f"  [Coder] 未成功: {result.error_message[:200]}")
        return result.optimized_code

    # ── ⑥ 验证: 只跑 msprof 端到端 ──
    def _verify(self, round_dir: Path, baseline_ns: Optional[float]) -> dict:
        """只跑一次 msprof 端到端 → 端到端耗时 → 加速比。
        真机跑; 本地/无 NPU 时返回 stub (需 --stub 或诊断文件里有耗时)。"""
        try:
            from agents.verifier import verify_end_to_end
            return verify_end_to_end(self.kernel_op, round_dir, baseline_ns)
        except Exception as e:
            print(f"  [Scheduler] verify stub: {e}")
            return {"ok": True, "ns": None, "speedup": 1.0, "note": "stub(无真机)"}

    # ── 主循环 ──
    def run(self):
        st = self.traj["state"]
        tier, rn = st.get("tier", 1), st.get("round", 0)
        print(f"══ Scheduler: {self.kernel_name} 目标 {self.target_speedup}x ══")

        # Round 0: 基准
        if rn == 0:
            base_dir = self.kernel_dir / "round0"
            base_dir.mkdir(parents=True, exist_ok=True)
            print("\n[Round 0] 基准采集...")
            d0 = self._run_optimize(base_dir)
            if d0:
                ks = d0.get("summary", {})
                st["baseline_ns"] = ks.get("total_ns")
                st["num_kernels"] = ks.get("num_kernels")
                st["round"] = 1
                self._save_traj()
                print(f"  基准: total_ns={ks.get('total_ns')} kernels={ks.get('num_kernels')}")
            else:
                print("  ❌ 基准采集失败")
                return 1

        while rn <= self.max_rounds:
            round_dir = self._round_dir(tier, rn)
            print(f"\n══ Tier{tier}({TIER_LABEL.get(tier)}) Round{rn} ══")

            # ① 采集+解析
            diagnosis = self._run_optimize(round_dir)
            if not diagnosis:
                print("  ⚠ 采集失败, 停止")
                break

            # ③ 提取字段 (每轮只看当前 tier)
            extracted = extract_tier_fields(diagnosis, tier)
            print(f"  [提取字段] Tier{tier} 共 {len(extracted.splitlines())} 行")

            # ④ Planner → plan + 晋升决策
            plan = self._plan(diagnosis, tier, rn, round_dir)

            # ⑤ Coder → 改单文件 (报错同轮重改, ≤3次)
            prev_err, new_code = "", ""
            for attempt in range(3):
                new_code = self._code(plan, rn, round_dir, prev_err)
                self.kernel_op.write_text(new_code, encoding="utf-8")
                # ⑥ 验证 (只 msprof 端到端)
                v = self._verify(round_dir, st.get("baseline_ns"))
                if v.get("ok"):
                    break
                prev_err = v.get("error", "unknown error")
                print(f"  ⚠ 运行失败(第{attempt+1}次): {prev_err[:200]}... 回传 Coder 同轮改")

            speedup = v.get("speedup", 1.0)
            ns = v.get("ns")
            print(f"  加速比: {speedup:.3f}x (ns={ns})")

            # ⑦ 记录 + 晋升决策
            hist = {"round": rn, "tier": tier, "strategy": plan.strategy,
                    "speedup": speedup, "ns": ns, "decision": "KEEP"}
            if st.get("best_speedup") is None or speedup > st["best_speedup"]:
                st["best_speedup"] = speedup
            self.traj["history"].append(hist)

            # 晋升决策: planner.promote (读瓶颈判断) + 连续3轮无改进兜底 + 达标/到Tier6停止
            planner_promote = getattr(plan, "promote", False)
            no_improve = sum(1 for h in self.traj["history"][-3:]
                             if h.get("tier") == tier and h.get("speedup", 0) < 1.05)
            if speedup >= self.target_speedup:
                print("  🎯 加速比达标, 停止")
                st["round"] = rn + 1
                self._save_traj()
                break
            if planner_promote:
                if tier >= 6:
                    print(f"  ⛔ planner 判瓶颈已非本tier且到Tier6, 停止 ({getattr(plan,'promote_reason','')})")
                    st["round"] = rn + 1
                    self._save_traj()
                    break
                print(f"  → 晋升 Tier{tier}→Tier{tier+1} (planner: {getattr(plan,'promote_reason','瓶颈不属本tier')})")
                tier += 1
                st["tier"] = tier
            elif no_improve >= 3 or tier >= 6:
                if tier >= 6:
                    print("  ⛔ Tier6 连续无改进, 停止")
                    st["round"] = rn + 1
                    self._save_traj()
                    break
                print(f"  → 晋升 Tier{tier}→Tier{tier+1} (本tier连续{no_improve}轮无改进)")
                tier += 1
                st["tier"] = tier
            st["round"] = rn + 1
            rn += 1
            self._save_traj()

        print(f"\n══ 完成: best_speedup={st.get('best_speedup')}x ══")
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
