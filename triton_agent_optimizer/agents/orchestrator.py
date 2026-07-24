#!/usr/bin/env python3
"""
优化调度器 — Python 状态机, 非 LLM Agent。

═══════════════════════════════════════════════════════════════════════════════
  每轮流程
═══════════════════════════════════════════════════════════════════════════════

  Round 0 (基准分析):
    Analyzers (msprof→hivmir→merger→diagnoser→extractor) → 写 trajectory.json

  Round 1..N (优化循环):
    ① Analyzers (重新分析当前 kernel 的 DSL 流水线)
    ② Planner (LLM)    → 读诊断+extracted+playbook+history → 生成计划
    ③ Coder (LLM)      → 按计划改代码
    ④ Verifier (脚本)  → CPU emulator 验证 (FAIL→重试最多3次)
                       → Simulator 预估
                       → Hardware 实测 (可选)
    ⑤ Decide           → speedup>1.01? KEEP/REVERT
    ⑥ 更新 trajectory.json

═══════════════════════════════════════════════════════════════════════════════
  入口
═══════════════════════════════════════════════════════════════════════════════

  python agents/orchestrator.py                                    # 自测
  python agents/orchestrator.py outputs/vector_add_fp16_N65536     # 真实运行
"""

from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 从 agents/ 子目录运行时也能 import analyzers
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoundPlan:
    round_num: int
    tier: int
    tier_name: str
    strategy: str
    target_speedup: float
    specific_change: str
    expected_impact: str
    verification_method: str
    plan_text: str = ""


@dataclass
class CoderResult:
    success: bool
    optimized_code: str
    diff: str
    lines_changed: int = 0
    error_message: str = ""


@dataclass
class VerifyResult:
    stage1_emulator_passed: bool
    stage1_max_abs_error: float = 0.0
    stage1_error_details: str = ""
    stage2_simulator_passed: bool = True
    stage2_estimated_speedup: float = 1.0
    stage3_hardware_passed: Optional[bool] = None
    stage3_actual_speedup: Optional[float] = None
    stage3_latency_ms: Optional[float] = None
    overall_passed: bool = False

    @property
    def speedup(self) -> float:
        if self.stage3_actual_speedup is not None:
            return self.stage3_actual_speedup
        return self.stage2_estimated_speedup


@dataclass
class RoundRecord:
    round: int; tier: int; tier_name: str; strategy: str
    target_speedup: float; actual_speedup: float; cumulative_speedup: float
    decision: str; decision_reason: str
    bottleneck_before: dict = field(default_factory=dict)
    bottleneck_after: dict = field(default_factory=dict)
    code_lines_changed: int = 0; emulator_passed: bool = False
    hardware_tested: bool = False; coder_retries: int = 0
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  停止条件
# ═══════════════════════════════════════════════════════════════════════════════

class StopChecker:
    def __init__(self, max_rounds=200, target_speedup=1.5,
                 max_consecutive_reverts=5, plateau_rounds=10,
                 plateau_variance=0.02, max_tier=6,
                 emulator_retry_max=3):
        self.max_rounds = max_rounds
        self.target_speedup = target_speedup
        self.max_consecutive_reverts = max_consecutive_reverts
        self.plateau_rounds = plateau_rounds
        self.plateau_variance = plateau_variance
        self.max_tier = max_tier
        self.emulator_retry_max = emulator_retry_max

    def check(self, state: dict, history: list) -> Tuple[bool, str, dict]:
        updates = {}
        # 1. 连续 REVERT → 晋升或停止
        if len(history) >= 5 and all(
                r.get("decision") == "REVERT" for r in history[-5:]):
            if state["tier"] >= self.max_tier:
                return True, "All 6 tiers exhausted", updates
            updates["tier"] = state["tier"] + 1
            updates["consecutive_reverts"] = 0
            updates["consecutive_no_improvement"] = 0
            return False, f"5 reverts → tier {updates['tier']}", updates
        # 2. 平台期
        if len(history) >= self.plateau_rounds:
            speeds = [r.get("cumulative_speedup", 1.0)
                      for r in history[-self.plateau_rounds:]]
            if max(speeds) - min(speeds) < self.plateau_variance:
                return True, f"Plateau detected ({self.plateau_rounds} rounds)", updates
        # 3. 预算
        if state["round"] >= self.max_rounds:
            return True, f"Max rounds ({self.max_rounds})", updates
        # 4. 目标
        if state["best_speedup"] >= self.target_speedup:
            return True, f"Target {self.target_speedup}x achieved", updates
        # 5. Tier6 + 3连败
        if state["tier"] >= self.max_tier and state.get("consecutive_reverts", 0) >= 3:
            return True, "Tier 6 + 3 reverts", updates
        # 6. 连续10轮无改进
        if state.get("consecutive_no_improvement", 0) >= 10:
            return True, "10 rounds no improvement", updates
        return False, "Continue", updates


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    TIER_NAMES = {1: "Algorithmic Structure", 2: "Operator Fusion",
                  3: "Tiling & Block Config", 4: "Memory Access",
                  5: "Compute & Occupancy", 6: "910B3 Architecture"}
    TIER_DIRS = {1: "01_algorithmic_structure", 2: "02_operator_fusion",
                 3: "03_tiling_block_config", 4: "04_memory_access",
                 5: "05_compute_occupancy", 6: "06_910b3_architecture"}

    def __init__(self, kernel_path: Path, kernel_name: str,
                 target_speedup: float = 1.5, max_rounds: int = 200,
                 output_root: Optional[Path] = None):
        self.kernel_path = Path(kernel_path)
        self.kernel_name = kernel_name
        if output_root is None:
            output_root = _PROJECT_DIR / "outputs"
        self.output_root = Path(output_root)
        self.kernel_dir = self.output_root / kernel_name
        self.trajectory_path = self.kernel_dir / "optimization_trajectory.json"
        self.stop_checker = StopChecker(max_rounds=max_rounds,
                                         target_speedup=target_speedup)
        self.trajectory: dict = {}
        self.current_kernel: str = ""
        self.best_kernel: str = ""

    # ═══════════════════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self) -> dict:
        print("=" * 60)
        print(f"Orchestrator — {self.kernel_name}")
        print(f"Target: {self.stop_checker.target_speedup}x  "
              f"Max: {self.stop_checker.max_rounds} rounds")
        print("=" * 60)

        # Round 0: 基准分析
        self._run_round0()

        # Round 1..N: 优化循环
        while True:
            should_stop, reason, updates = self.stop_checker.check(
                self.trajectory["state"], self.trajectory["history"])
            for k, v in updates.items():
                self.trajectory["state"][k] = v
            if updates:
                self._save_trajectory()
            if should_stop:
                print(f"\n[STOP] {reason}")
                break

            record = self._run_one_round()
            self._update_state(record)
            self._append_history(record)
            self._save_trajectory()

            print(f"\n[ROUND {record.round}] {record.decision} | "
                  f"{record.actual_speedup:.2f}x "
                  f"(cum={record.cumulative_speedup:.2f}x) | "
                  f"tier={record.tier}")

        return self._finalize()

    # ═══════════════════════════════════════════════════════════════════════════
    #  Round 0
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_round0(self):
        """基准分析: 跑全部分析层, 写 trajectory baseline。"""
        print("\n[ROUND 0] Baseline analysis...")
        round_dir = self.kernel_dir / "round0"
        round_dir.mkdir(parents=True, exist_ok=True)

        # kernel
        kf = round_dir / "kernel.py"
        if not kf.exists():
            shutil.copy2(self.kernel_path, kf)
        self.current_kernel = kf.read_text(encoding="utf-8")
        self.best_kernel = self.current_kernel

        # 初始化 trajectory (先写 state, analyzers 需要读 tier)
        now = datetime.now().isoformat()
        self.trajectory = {
            "kernel": {"name": self.kernel_name, "dtype": "fp16",
                       "initial_kernel_path": str(kf)},
            "state": {"tier": 1, "round": 0, "best_speedup": 1.0,
                      "consecutive_reverts": 0,
                      "consecutive_no_improvement": 0,
                      "started_at": now, "last_updated": now},
            "baseline": {}, "tier_progress": {}, "history": [],
        }

        # 分析
        merged, diag, _ = self._run_analyzers(round_dir)

        # 回填 baseline + history
        self.trajectory["baseline"] = self._baseline_from(merged, diag)
        self.trajectory["history"] = [self._round0_history_entry(diag,
            self.trajectory["state"]["started_at"])]
        self._save_trajectory()

        bn = self.trajectory["baseline"]
        print(f"[ROUND 0] Baseline: {bn['total_ns']:.1f}ns, "
              f"{bn['num_ops']} ops, "
              f"bottleneck=op{bn['bottleneck_op_id']}({bn['bottleneck_type']})")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Round N
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_one_round(self) -> RoundRecord:
        s = self.trajectory["state"]
        tier, rn = s["tier"], s["round"] + 1
        rd = self._round_dir(tier, rn)
        rd.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"ROUND {rn} | Tier {tier}: {self.TIER_NAMES.get(tier)}")
        print(f"{'─'*60}")

        # ── ① 分析 (每轮重跑) ──
        merged, diag, extracted = self._run_analyzers(rd)
        if diag is None:
            diag = _FallbackDiag()

        # ── ② Planner (LLM) ──
        print(f"  [Planner] generating plan...")
        plan = self._call_planner(diag, extracted, tier, rn)
        (rd / "plan.md").write_text(_format_plan_md(plan), encoding="utf-8")

        # ── ③ Coder (LLM) ──
        print(f"  [Coder] applying: {plan.strategy}...")
        coder_r = self._call_coder(self.current_kernel, plan)
        if not coder_r.success:
            return self._make_revert_record(rn, tier, plan,
                f"Coder failed: {coder_r.error_message}", diag)

        # ── ④ Verifier (脚本, 带 emulator 重试) ──
        vr, retries = self._verify_with_retry(
            coder_r.optimized_code, self.current_kernel, plan, rd)
        if not vr.overall_passed:
            return self._make_revert_record(rn, tier, plan,
                f"Verification failed after {retries} retries: {vr.stage1_error_details}", diag)

        # 保存产出
        (rd / "kernel.py").write_text(coder_r.optimized_code, encoding="utf-8")
        (rd / "diff.patch").write_text(coder_r.diff, encoding="utf-8")

        # ── ⑤ Decide ──
        decision, reason = "KEEP", ""
        if vr.speedup <= 1.01:
            decision, reason = "REVERT", f"Speedup {vr.speedup:.3f}x <= 1.01x"
        else:
            self.current_kernel = coder_r.optimized_code
            reason = f"Speedup {vr.speedup:.3f}x > 1.01x"

        cumulative = s["best_speedup"] * vr.speedup

        record = RoundRecord(
            round=rn, tier=tier, tier_name=self.TIER_NAMES.get(tier, "?"),
            strategy=plan.strategy, target_speedup=plan.target_speedup,
            actual_speedup=vr.speedup, cumulative_speedup=cumulative,
            decision=decision, decision_reason=reason,
            bottleneck_before={"op_id": diag.bottleneck_op_id,
                                "type": diag.bottleneck_type,
                                "time_ratio": diag.bottleneck_time_ratio},
            code_lines_changed=coder_r.lines_changed,
            emulator_passed=vr.stage1_emulator_passed,
            hardware_tested=vr.stage3_hardware_passed is not None,
            coder_retries=retries,
            timestamp=datetime.now().isoformat())

        # 保存 optimization_record.json
        (rd / "optimization_record.json").write_text(_json({
            "round": record.round, "tier": record.tier,
            "tier_name": record.tier_name, "strategy": record.strategy,
            "target_speedup": record.target_speedup,
            "actual_speedup": record.actual_speedup,
            "cumulative_speedup": record.cumulative_speedup,
            "decision": record.decision,
            "decision_reason": record.decision_reason,
            "bottleneck_before": record.bottleneck_before,
            "code_lines_changed": record.code_lines_changed,
            "coder_retries": retries,
            "verification": {
                "stage1_passed": vr.stage1_emulator_passed,
                "stage1_error": vr.stage1_error_details,
                "stage2_estimated_speedup": vr.stage2_estimated_speedup,
                "stage3_actual_speedup": vr.stage3_actual_speedup,
            },
        }), encoding="utf-8")

        return record

    # ═══════════════════════════════════════════════════════════════════════════
    #  Verifier 重试循环
    # ═══════════════════════════════════════════════════════════════════════════

    def _verify_with_retry(self, opt_code: str, orig_code: str,
                           plan: RoundPlan, rd: Path) -> Tuple[VerifyResult, int]:
        """Stage1(CPU emulator) 失败→Coder重试, 最多3次。"""
        max_retry = self.stop_checker.emulator_retry_max
        current_code = opt_code

        for attempt in range(max_retry + 1):
            vr = self._call_verifier(current_code, orig_code, rd)
            if vr.stage1_emulator_passed:
                return vr, attempt

            if attempt < max_retry:
                print(f"  [Verifier] Stage1 FAIL (attempt {attempt+1}/{max_retry})"
                      f" — retrying Coder with error feedback")
                cr = self._call_coder(current_code, plan,
                                       previous_error=vr.stage1_error_details)
                if cr.success:
                    current_code = cr.optimized_code
                else:
                    return vr, attempt

        return vr, max_retry

    # ═══════════════════════════════════════════════════════════════════════════
    #  分析层
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_analyzers(self, round_dir: Path):
        """运行完整分析链。每轮开始前调用。"""
        from analyzers.dsl_merger import merge_round
        from analyzers.bottleneck_diagnoser import diagnose
        from analyzers.data_extractor import extract

        merged_file = round_dir / "merged" / "merged_report.json"

        # 尝试合并 msprof + hivmir
        if not merged_file.exists():
            try:
                merge_round(round_dir)
            except Exception:
                pass

        # 读取 merged 数据
        if merged_file.exists():
            merged = json.loads(merged_file.read_text(encoding="utf-8"))
        else:
            # fallback: round0
            fb = self.kernel_dir / "round0" / "merged" / "merged_report.json"
            if fb.exists():
                print(f"  [WARN] Using round0 merged data as fallback")
                merged = json.loads(fb.read_text(encoding="utf-8"))
            else:
                return {}, None, ""

        tier = self.trajectory["state"]["tier"]
        diag = diagnose(merged, current_tier=tier)
        extracted = extract(merged, _diag_dict(diag) if diag else {}, tier=tier)
        return merged, diag, extracted

    # ═══════════════════════════════════════════════════════════════════════════
    #  Agent stubs (后续替换为真实 LLM)
    # ═══════════════════════════════════════════════════════════════════════════

    def _call_planner(self, diagnosis, extracted_text: str,
                      tier: int, round_num: int) -> RoundPlan:
        history = self.trajectory.get("history", [])
        from agents.planner import PlannerAgent
        planner = PlannerAgent()
        return planner.generate(
            diagnosis=diagnosis,
            extracted_text=extracted_text,
            tier=tier,
            history=history[-5:],
            kernel_code=self.current_kernel,
            round_num=round_num,
        )

    def _call_coder(self, kernel_code: str, plan: RoundPlan,
                    previous_error: str = "") -> CoderResult:
        from agents.coder import CoderAgent
        coder = CoderAgent()
        return coder.apply(
            kernel_code=kernel_code,
            plan_text=plan.plan_text,
            previous_error=previous_error,
        )

    def _call_verifier(self, opt_code: str, orig_code: str,
                       rd: Path) -> VerifyResult:
        from agents.verifier import VerifierAgent
        verifier = VerifierAgent()
        baseline = self.trajectory.get("baseline", {})
        return verifier.verify(
            optimized_code=opt_code,
            original_dsl="",       # TODO: 从当前 kernel 生成 DSL
            optimized_dsl="",      # TODO: 从优化后 kernel 生成 DSL
            baseline_latency_ms=baseline.get("total_ns", 0) / 1e6,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  辅助
    # ═══════════════════════════════════════════════════════════════════════════

    def _round_dir(self, tier: int, rn: int) -> Path:
        td = self.TIER_DIRS.get(tier, f"tier{tier}")
        return self.kernel_dir / td / f"round{rn}"

    def _baseline_from(self, merged: dict, diag) -> dict:
        s = merged.get("execution_summary", {}) if merged else {}
        return {
            "total_ns": s.get("total_ns", 0), "num_ops": s.get("num_ops", 0),
            "execution_mode": s.get("execution_mode", ""),
            "num_cores": s.get("num_cores", 0),
            "bottleneck_op_id": diag.bottleneck_op_id if diag else -1,
            "bottleneck_op_type": diag.bottleneck_op_type if diag else "?",
            "bottleneck_engine": diag.bottleneck_engine if diag else "?",
            "bottleneck_type": diag.bottleneck_type if diag else "?",
            "bottleneck_time_ratio": diag.bottleneck_time_ratio if diag else 0,
            "engine_utilization": merged.get("engine_utilization", {}) if merged else {},
        }

    def _round0_history_entry(self, diag, now: str) -> dict:
        return {
            "round": 0, "tier": 0, "tier_name": "baseline",
            "strategy": "initial_analysis", "target_speedup": 1.0,
            "actual_speedup": 1.0, "cumulative_speedup": 1.0,
            "decision": "BASELINE", "decision_reason": "Initial analysis",
            "bottleneck_before": {},
            "bottleneck_after": {
                "op_id": diag.bottleneck_op_id if diag else -1,
                "type": diag.bottleneck_type if diag else "?",
            },
            "code_lines_changed": 0, "emulator_passed": False,
            "hardware_tested": False, "coder_retries": 0,
            "timestamp": now,
        }

    def _make_revert_record(self, rn, tier, plan, reason, diag) -> RoundRecord:
        s = self.trajectory["state"]
        return RoundRecord(
            round=rn, tier=tier, tier_name=self.TIER_NAMES.get(tier, "?"),
            strategy=plan.strategy, target_speedup=plan.target_speedup,
            actual_speedup=1.0, cumulative_speedup=s["best_speedup"],
            decision="REVERT", decision_reason=reason,
            bottleneck_before={"op_id": diag.bottleneck_op_id,
                                "type": diag.bottleneck_type},
            timestamp=datetime.now().isoformat())

    def _update_state(self, rec: RoundRecord):
        s = self.trajectory["state"]
        s["round"] += 1
        s["last_updated"] = datetime.now().isoformat()
        if rec.decision == "KEEP":
            s["consecutive_reverts"] = 0
            if rec.actual_speedup > s["best_speedup"]:
                s["best_speedup"] = rec.cumulative_speedup
                self.best_kernel = self.current_kernel
            s["consecutive_no_improvement"] = (
                0 if rec.actual_speedup >= 1.01
                else s.get("consecutive_no_improvement", 0) + 1)
        else:
            s["consecutive_reverts"] = s.get("consecutive_reverts", 0) + 1
            s["consecutive_no_improvement"] = s.get("consecutive_no_improvement", 0) + 1

    def _append_history(self, rec: RoundRecord):
        self.trajectory["history"].append({
            "round": rec.round, "tier": rec.tier,
            "tier_name": rec.tier_name, "strategy": rec.strategy,
            "target_speedup": rec.target_speedup,
            "actual_speedup": rec.actual_speedup,
            "cumulative_speedup": rec.cumulative_speedup,
            "decision": rec.decision,
            "decision_reason": rec.decision_reason,
            "bottleneck_before": rec.bottleneck_before,
            "bottleneck_after": rec.bottleneck_after,
            "code_lines_changed": rec.code_lines_changed,
            "emulator_passed": rec.emulator_passed,
            "hardware_tested": rec.hardware_tested,
            "coder_retries": rec.coder_retries,
            "timestamp": rec.timestamp,
        })
        tp = self.trajectory.setdefault("tier_progress", {})
        tk = f"tier_{rec.tier}"
        if tk not in tp:
            tp[tk] = {"rounds_spent": 0, "best_in_tier": 1.0}
        tp[tk]["rounds_spent"] += 1
        tp[tk]["best_in_tier"] = max(tp[tk]["best_in_tier"], rec.actual_speedup)

    def _save_trajectory(self):
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_path.write_text(_json(self.trajectory), encoding="utf-8")

    def _finalize(self) -> dict:
        s = self.trajectory["state"]
        h = self.trajectory["history"]
        kept = [r for r in h if r.get("decision") == "KEEP"]
        fd = self.kernel_dir / "final_output"
        fd.mkdir(exist_ok=True)
        if self.best_kernel:
            (fd / "optimized_kernel.py").write_text(self.best_kernel, encoding="utf-8")
        lines = [f"# {self.kernel_name} — Optimized",
                 f"Rounds: {s['round']} ({len(kept)} kept) | "
                 f"Speedup: {s['best_speedup']:.2f}x | Tier: {s['tier']}"]
        for r in kept:
            lines.append(f"- R{r['round']}: {r['strategy']} → {r['actual_speedup']:.2f}x")
        (fd / "optimization_summary.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n{'='*60}\nDONE | {s['round']} rounds | "
              f"{s['best_speedup']:.2f}x | tier={s['tier']}\n{'='*60}")
        return {"kernel": self.kernel_name, "rounds": s["round"],
                "kept": len(kept), "best_speedup": s["best_speedup"],
                "final_tier": s["tier"], "output": str(fd)}


class _FallbackDiag:
    bottleneck_op_id = -1; bottleneck_op_type = "?"
    bottleneck_engine = "?"; bottleneck_type = "unknown"
    bottleneck_category = "UNKNOWN"; optimization_headroom = "UNCERTAIN"
    bottleneck_time_ratio = 0.0; bottleneck_bw_utilization = 0.0
    bottleneck_regime = "?"; suggested_strategies = []; structural_issues = []


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _format_plan_md(plan) -> str:
    return f"""# Round {plan.round_num} Optimization Plan

**Tier**: {plan.tier} ({plan.tier_name})
**Strategy**: {plan.strategy}
**Target Speedup**: {plan.target_speedup}x

## Specific Change
{plan.specific_change}

## Expected Impact
{plan.expected_impact}

## Verification Method
{plan.verification_method}

## Raw Plan JSON
```json
{plan.plan_text}
```
"""


def _diag_dict(diag) -> dict:
    return {"bottleneck": {"op_id": diag.bottleneck_op_id,
            "op_type": diag.bottleneck_op_type,
            "engine": diag.bottleneck_engine,
            "type": diag.bottleneck_type,
            "category": diag.bottleneck_category,
            "headroom": diag.optimization_headroom,
            "time_ratio": diag.bottleneck_time_ratio,
            "bw_utilization": diag.bottleneck_bw_utilization,
            "regime": diag.bottleneck_regime},
            "strategies": diag.suggested_strategies,
            "structural_issues": diag.structural_issues}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI & 自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    outputs = _PROJECT_DIR / "outputs"
    kd = outputs / "vector_add_fp16_N65536"
    kf = kd / "round0" / "kernel.py"
    if not kf.exists():
        print("[SKIP] round0/kernel.py not found")
        return
    tj = kd / "optimization_trajectory.json"
    if tj.exists(): tj.unlink()
    orch = Orchestrator(kernel_path=kf, kernel_name="vector_add_fp16_N65536",
                        target_speedup=2.0, max_rounds=3)
    orch.run()
    if tj.exists():
        t = json.loads(tj.read_text(encoding="utf-8"))
        print(f"\nTrajectory: state={_json(t['state'])}")
        for h in t["history"]:
            print(f"  R{h['round']}: {h['strategy'][:40]} → {h['decision']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        kd = Path(sys.argv[1])
        Orchestrator(kernel_path=kd / "round0" / "kernel.py",
                     kernel_name=kd.name).run()
    else:
        _self_test()
