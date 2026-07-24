#!/usr/bin/env python3
"""
反馈与记录层 — 决策引擎。

═══════════════════════════════════════════════════════════════════════════════
  职责
═══════════════════════════════════════════════════════════════════════════════

  1. 接收 VerifyResult → 决定 KEEP/REVERT
  2. 保存 optimization_record.json 到 round_N/
  3. 读写 optimization_trajectory.json (中枢状态文件)
  4. 管理 Tier 晋升/降级
  5. 检查 7 条停止条件
  6. 达标时触发案例生成 + Gantt 图 (按需)

  调度器只管循环调用: analyzers → planner → coder → verifier → record_manager

═══════════════════════════════════════════════════════════════════════════════
  数据流
═══════════════════════════════════════════════════════════════════════════════

  Verifier → VerifyResult
      │
      ▼
  RecordManager.evaluate(result, plan, coder_result, round_dir, diag)
      │
      ├─ decide: speedup>1.01? KEEP:REVERT
      ├─ save optimization_record.json → round_N/
      ├─ update optimization_trajectory.json
      ├─ check stop: plateau? budget? target? all_tiers?
      │
      └─ return (should_stop, reason)
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict


# ═══════════════════════════════════════════════════════════════════════════════
#  停止条件
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StopResult:
    should_stop: bool
    reason: str
    tier_promoted: bool = False
    new_tier: int = 0


class StopChecker:
    def __init__(self, max_rounds=200, target_speedup=1.5,
                 max_consecutive_reverts=5, plateau_rounds=10,
                 plateau_variance=0.02, max_tier=6):
        self.max_rounds = max_rounds
        self.target_speedup = target_speedup
        self.max_consecutive_reverts = max_consecutive_reverts
        self.plateau_rounds = plateau_rounds
        self.plateau_variance = plateau_variance
        self.max_tier = max_tier

    def check(self, state: dict, history: list) -> StopResult:
        # 连续 REVERT → tier+1 或 stop
        if len(history) >= 5:
            recent = history[-5:]
            if all(r.get("decision") == "REVERT" for r in recent):
                if state["tier"] >= self.max_tier:
                    return StopResult(True, "All 6 tiers exhausted")
                new_tier = state["tier"] + 1
                return StopResult(False, f"5 reverts → tier {new_tier}",
                                  tier_promoted=True, new_tier=new_tier)

        # 平台期
        if len(history) >= self.plateau_rounds:
            speeds = [r.get("cumulative_speedup", 1.0)
                      for r in history[-self.plateau_rounds:]]
            if max(speeds) - min(speeds) < self.plateau_variance:
                return StopResult(True, f"Plateau ({self.plateau_rounds} rounds)")

        # 预算
        if state["round"] >= self.max_rounds:
            return StopResult(True, f"Max rounds ({self.max_rounds})")

        # 目标
        if state["best_speedup"] >= self.target_speedup:
            return StopResult(True, f"Target {self.target_speedup}x achieved")

        # Tier6 + 3连败
        if state["tier"] >= self.max_tier and state.get("consecutive_reverts", 0) >= 3:
            return StopResult(True, "Tier 6 + 3 reverts")

        # 连续10轮无改进
        if state.get("consecutive_no_improvement", 0) >= 10:
            return StopResult(True, "10 rounds no improvement")

        return StopResult(False, "Continue")


# ═══════════════════════════════════════════════════════════════════════════════
#  RecordManager
# ═══════════════════════════════════════════════════════════════════════════════

TIER_NAMES = {
    1: "Algorithmic Structure", 2: "Operator Fusion",
    3: "Tiling & Block Config", 4: "Memory Access",
    5: "Compute & Occupancy", 6: "910B3 Architecture",
}


class RecordManager:
    """反馈与记录层 — 决策引擎。

    Usage:
        rm = RecordManager(kernel_dir)
        should_stop, reason = rm.evaluate(
            verify_result, plan, coder_result, round_dir, diagnosis)
    """

    def __init__(self, kernel_dir: Path, target_speedup: float = 1.5,
                 max_rounds: int = 200):
        self.kernel_dir = Path(kernel_dir)
        self.trajectory_path = self.kernel_dir / "optimization_trajectory.json"
        self.stop_checker = StopChecker(
            max_rounds=max_rounds, target_speedup=target_speedup)

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口: 评估本轮结果
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate(
        self,
        verify_result,      # VerifyResult from Verifier
        plan,               # RoundPlan from Planner
        coder_result,       # CoderResult from Coder
        round_dir: Path,
        diagnosis,          # BottleneckDiagnosis
    ) -> Tuple[bool, str]:
        """评估本轮结果，更新轨迹，返回是否停止。

        Returns:
            (should_stop, reason)
        """
        traj = self._load_trajectory()
        s = traj["state"]
        tier = s["tier"]
        rn = s["round"] + 1

        # ── 决策 ──
        if not verify_result.overall_passed:
            decision, reason = "REVERT", (
                f"Verification failed: {verify_result.stage1_error_details[:100]}")
        elif verify_result.speedup <= 1.01:
            decision, reason = "REVERT", f"Speedup {verify_result.speedup:.3f}x <= 1.01"
        else:
            decision, reason = "KEEP", f"Speedup {verify_result.speedup:.3f}x > 1.01"

        cumulative = s["best_speedup"] * verify_result.speedup
        if decision == "KEEP" and verify_result.speedup > s["best_speedup"]:
            s["best_speedup"] = cumulative

        # ── 更新 state ──
        s["round"] = rn
        s["last_updated"] = datetime.now().isoformat()
        if decision == "KEEP":
            s["consecutive_reverts"] = 0
            s["consecutive_no_improvement"] = (
                0 if verify_result.speedup >= 1.01
                else s.get("consecutive_no_improvement", 0) + 1)
        else:
            s["consecutive_reverts"] = s.get("consecutive_reverts", 0) + 1
            s["consecutive_no_improvement"] = (
                s.get("consecutive_no_improvement", 0) + 1)

        # ── 追加 history ──
        diag_before = {
            "op_id": getattr(diagnosis, "bottleneck_op_id", -1),
            "type": getattr(diagnosis, "bottleneck_type", "?"),
        }
        traj["history"].append({
            "round": rn, "tier": tier,
            "tier_name": TIER_NAMES.get(tier, "?"),
            "strategy": plan.strategy,
            "target_speedup": plan.target_speedup,
            "actual_speedup": verify_result.speedup,
            "cumulative_speedup": cumulative,
            "decision": decision, "decision_reason": reason,
            "bottleneck_before": diag_before,
            "code_lines_changed": coder_result.lines_changed,
            "emulator_passed": verify_result.stage1_passed,
            "hardware_tested": verify_result.stage2_tested,
            "timestamp": datetime.now().isoformat(),
        })

        # Tier progress
        tp = traj.setdefault("tier_progress", {})
        tk = f"tier_{tier}"
        if tk not in tp:
            tp[tk] = {"rounds_spent": 0, "best_in_tier": 1.0}
        tp[tk]["rounds_spent"] += 1
        tp[tk]["best_in_tier"] = max(tp[tk]["best_in_tier"], verify_result.speedup)

        # ── 记录经验 (成功/失败) ──
        self._record_experience(tier, diagnosis, plan.strategy,
                                 verify_result.speedup, decision, reason)

        # ── 保存 optimization_record.json ──
        self._save_round_record(round_dir, rn, tier, plan, coder_result,
                                 verify_result, decision, reason, cumulative)

        # ── 检查停止条件 ──
        stop_result = self.stop_checker.check(s, traj["history"])
        if stop_result.tier_promoted:
            s["tier"] = stop_result.new_tier
            s["consecutive_reverts"] = 0
            s["consecutive_no_improvement"] = 0

        self._save_trajectory(traj)

        if stop_result.should_stop:
            return True, stop_result.reason
        if decision == "KEEP" and verify_result.speedup >= self.stop_checker.target_speedup:
            self._trigger_case_generation(traj)
            return True, f"Target {self.stop_checker.target_speedup}x achieved"

        return False, "Continue"

    # ═══════════════════════════════════════════════════════════════════════════
    #  初始化 baseline
    # ═══════════════════════════════════════════════════════════════════════════

    def init_baseline(self, merged_report: dict, diagnosis, kernel_path: str):
        """写入 round0 baseline。"""
        summary = merged_report.get("execution_summary", {})
        now = datetime.now().isoformat()
        traj = {
            "kernel": {"name": self.kernel_dir.name, "dtype": "fp16",
                       "initial_kernel_path": kernel_path},
            "state": {"tier": 1, "round": 0, "best_speedup": 1.0,
                      "consecutive_reverts": 0,
                      "consecutive_no_improvement": 0,
                      "started_at": now, "last_updated": now},
            "baseline": {
                "total_ns": summary.get("total_ns", 0),
                "num_ops": summary.get("num_ops", 0),
                "execution_mode": summary.get("execution_mode", ""),
                "bottleneck_op_id": getattr(diagnosis, "bottleneck_op_id", -1),
                "bottleneck_type": getattr(diagnosis, "bottleneck_type", "?"),
                "engine_utilization": merged_report.get("engine_utilization", {}),
            },
            "tier_progress": {},
            "history": [{
                "round": 0, "tier": 0, "tier_name": "baseline",
                "strategy": "initial_analysis",
                "actual_speedup": 1.0, "cumulative_speedup": 1.0,
                "decision": "BASELINE",
                "timestamp": now,
            }],
        }
        self._save_trajectory(traj)

    # ═══════════════════════════════════════════════════════════════════════════
    #  查询
    # ═══════════════════════════════════════════════════════════════════════════

    def get_state(self) -> dict:
        return self._load_trajectory().get("state", {})

    def get_tier(self) -> int:
        return self.get_state().get("tier", 1)

    def get_recent_history(self, n: int = 5) -> list:
        h = self._load_trajectory().get("history", [])
        return h[-n:]

    # ═══════════════════════════════════════════════════════════════════════════
    #  IO
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_trajectory(self) -> dict:
        if self.trajectory_path.exists():
            return json.loads(self.trajectory_path.read_text(encoding="utf-8"))
        return {"state": {"tier": 1, "round": 0, "best_speedup": 1.0,
                          "consecutive_reverts": 0,
                          "consecutive_no_improvement": 0},
                "history": []}

    def _save_trajectory(self, traj: dict):
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_path.write_text(
            json.dumps(traj, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_round_record(self, rd: Path, rn: int, tier: int,
                           plan, coder_r, vr, decision, reason, cumulative):
        data = {
            "round": rn, "tier": tier,
            "tier_name": TIER_NAMES.get(tier, "?"),
            "strategy": plan.strategy,
            "target_speedup": plan.target_speedup,
            "actual_speedup": vr.speedup,
            "cumulative_speedup": cumulative,
            "decision": decision, "decision_reason": reason,
            "code_lines_changed": coder_r.lines_changed,
            "emulator_passed": vr.stage1_passed,
            "hardware_tested": vr.stage2_tested,
            "verification": {
                "stage1_passed": vr.stage1_passed,
                "stage1_error": vr.stage1_error_details,
                "stage2_actual_speedup": vr.stage2_actual_speedup,
            },
        }
        (rd / "optimization_record.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _record_experience(self, tier: int, diagnosis, strategy: str,
                            speedup: float, decision: str, reason: str):
        """自动记录成功/失败经验到记忆层。"""
        if speedup > 1.05:
            status = "SUCCESS"
        elif speedup < 0.98:
            status = "FAIL"
        else:
            return  # 中性结果不记录

        try:
            from memory.experience_retriever import record
            record(
                tier=tier,
                fingerprint={
                    "op_type": getattr(diagnosis, "bottleneck_op_type", ""),
                    "bottleneck_type": getattr(diagnosis, "bottleneck_type", ""),
                    "engine": getattr(diagnosis, "bottleneck_engine", ""),
                },
                strategy=strategy,
                speedup=speedup,
                status=status,
                description=f"Tier {tier}: {strategy}",
                decision_reason=reason if status == "FAIL" else "",
            )
        except Exception:
            pass  # 经验记录失败不影响主流程

    def _trigger_case_generation(self, traj: dict):
        """达标: 写入 final_output/。"""
        fd = self.kernel_dir / "final_output"
        fd.mkdir(exist_ok=True)
        s = traj["state"]
        h = traj["history"]
        kept = [r for r in h if r.get("decision") == "KEEP"]
        reverted = [r for r in h if r.get("decision") == "REVERT"]

        # 完整总结
        lines = [
            f"# {self.kernel_dir.name} — Optimization Report",
            "",
            f"## Results",
            f"- **Total rounds**: {s['round']}",
            f"- **Kept**: {len(kept)}  **Reverted**: {len(reverted)}",
            f"- **Best speedup**: {s['best_speedup']:.2f}x",
            f"- **Final tier**: {s['tier']}",
            f"- **Started**: {s.get('started_at','?')}",
            f"- **Completed**: {s.get('last_updated','?')}",
            "",
            f"## Baseline",
            f"- total_ns: {traj['baseline'].get('total_ns',0):.1f}",
            f"- ops: {traj['baseline'].get('num_ops',0)}",
            f"- bottleneck: {traj['baseline'].get('bottleneck_type','?')}",
            "",
            f"## Successful Strategies",
        ]
        for r in kept:
            lines.append(
                f"- **R{r['round']}** (Tier {r.get('tier','?')}): "
                f"`{r.get('strategy','?')}` → "
                f"{r.get('actual_speedup',1.0):.2f}x "
                f"(cumulative {r.get('cumulative_speedup',1.0):.2f}x)")
        lines.append("")
        lines.append("## Failed Strategies")
        for r in reverted:
            lines.append(
                f"- R{r['round']}: `{r.get('strategy','?')[:60]}` → "
                f"REVERT ({r.get('decision_reason','?')[:80]})")
        lines.append("")
        lines.append("## Recommendations")
        lines.append("(See optimization_trajectory.json for full data)")
        (fd / "optimization_summary.md").write_text("\n".join(lines), encoding="utf-8")
