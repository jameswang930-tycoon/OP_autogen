#!/usr/bin/env python3
"""
优化调度器 — 薄循环 (Python 状态机)。

═══════════════════════════════════════════════════════════════════════════════
  职责: 调用 Analyzers → Planner → Coder → Verifier → RecordManager
═══════════════════════════════════════════════════════════════════════════════

  Orchestrator 只跑循环:
    for round in range(max_rounds):
      analyzers.run()         # 分析层 (脚本)
      plan = planner.plan()   # 规划 (LLM)
      code = coder.apply()    # 编码 (LLM)
      result = verifier.verify()  # 验证 (脚本)
      should_stop = record_mgr.evaluate(result, ...)  # 决策+记录
      if should_stop: break

  决策/记录/停止/Tier → 全在 feedback/record_manager.py

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  python agents/orchestrator.py
  python agents/orchestrator.py outputs/vector_add_fp16_N65536
"""

from __future__ import annotations

import json, shutil, sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


TIER_NAMES = {1: "Algorithmic Structure", 2: "Operator Fusion",
              3: "Tiling & Block Config", 4: "Memory Access",
              5: "Compute & Occupancy", 6: "910B3 Architecture"}

TIER_DIRS = {1: "01_algorithmic_structure", 2: "02_operator_fusion",
             3: "03_tiling_block_config", 4: "04_memory_access",
             5: "05_compute_occupancy", 6: "06_910b3_architecture"}


class Orchestrator:
    """薄循环调度器。"""

    def __init__(self, kernel_path: Path, kernel_name: str,
                 target_speedup=1.5, max_rounds=200,
                 output_root: Optional[Path] = None):
        self.kernel_path = Path(kernel_path)
        self.kernel_name = kernel_name
        if output_root is None:
            output_root = _PROJECT_DIR / "outputs"
        self.kernel_dir = Path(output_root) / kernel_name
        self.max_rounds = max_rounds
        self.target_speedup = target_speedup

        # 反馈层
        from feedback.record_manager import RecordManager
        self.record_mgr = RecordManager(self.kernel_dir,
                                         target_speedup=target_speedup,
                                         max_rounds=max_rounds)

        self.current_kernel = ""
        self.best_kernel = ""

    # ═══════════════════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self) -> dict:
        print("=" * 60)
        print(f"Orchestrator — {self.kernel_name}")
        print(f"Target: {self.target_speedup}x  Max: {self.max_rounds}")
        print("=" * 60)

        self._run_round0()

        while True:
            # ① Analyzers → ② Planner → ③ Coder → ④ Verifier → ⑤ Evaluate
            stop, reason = self._run_one_round()
            print(f"\n[ROUND {self.record_mgr.get_state()['round']}] {reason}")
            if stop:
                break

        return self._finalize()

    # ═══════════════════════════════════════════════════════════════════════════
    #  Round 0
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_round0(self):
        print("\n[ROUND 0] Baseline...")
        rd = self.kernel_dir / "round0"
        rd.mkdir(parents=True, exist_ok=True)
        kf = rd / "kernel.py"
        if not kf.exists():
            shutil.copy2(self.kernel_path, kf)
        self.current_kernel = kf.read_text(encoding="utf-8")
        self.best_kernel = self.current_kernel

        merged, diag, _ = self._run_analyzers(rd)
        self.record_mgr.init_baseline(merged or {}, diag, str(kf))
        print(f"[ROUND 0] Done — {merged.get('execution_summary',{}).get('num_ops',0) if merged else 0} ops")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Round N
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_one_round(self) -> tuple:
        tier = self.record_mgr.get_tier()
        rn = self.record_mgr.get_state()["round"] + 1
        rd = self._round_dir(tier, rn)
        rd.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"ROUND {rn} | Tier {tier}: {TIER_NAMES.get(tier)}")
        print(f"{'─'*60}")

        # ① Analyzers
        merged, diag, extracted = self._run_analyzers(rd)

        # ② Planner
        print(f"  [Planner] ...")
        plan = self._call_planner(diag, extracted or "", tier, rn)
        (rd / "plan.md").write_text(_plan_md(plan), encoding="utf-8")

        # ③ Coder
        print(f"  [Coder] {plan.strategy}...")
        coder_r = self._call_coder(self.current_kernel, plan)
        if not coder_r.success:
            # Coder 失败 → 直接 REVERT
            from agents.verifier import VerifyResult
            vr = VerifyResult(stage1_passed=False,
                              stage1_error_details=coder_r.error_message,
                              overall_passed=False)
            return self.record_mgr.evaluate(vr, plan, coder_r, rd, diag)

        (rd / "kernel.py").write_text(coder_r.optimized_code, encoding="utf-8")
        (rd / "diff.patch").write_text(coder_r.diff, encoding="utf-8")

        # ④ Verifier (带重试)
        vr, _ = self._verify_with_retry(coder_r.optimized_code, self.current_kernel, plan, rd)

        # ⑤ 决策+记录 → RecordManager
        return self.record_mgr.evaluate(vr, plan, coder_r, rd, diag)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Analyzers
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_analyzers(self, rd: Path):
        from analyzers.dsl_merger import merge_round
        from analyzers.bottleneck_diagnoser import diagnose
        from analyzers.data_extractor import extract

        mf = rd / "merged" / "merged_report.json"
        if not mf.exists():
            try:
                merge_round(rd)
            except Exception:
                pass

        if mf.exists():
            merged = json.loads(mf.read_text(encoding="utf-8"))
        else:
            fb = self.kernel_dir / "round0" / "merged" / "merged_report.json"
            if fb.exists():
                merged = json.loads(fb.read_text(encoding="utf-8"))
            else:
                return {}, None, ""

        tier = self.record_mgr.get_tier()
        diag = diagnose(merged, current_tier=tier) if merged else None
        extracted = extract(merged, _diag_dict(diag), tier=tier)
        return merged, diag, extracted

    # ═══════════════════════════════════════════════════════════════════════════
    #  Agents (Planner + Coder 是 LLM, Verifier 是脚本)
    # ═══════════════════════════════════════════════════════════════════════════

    def _call_planner(self, diag, extracted, tier, rn):
        from agents.planner import PlannerAgent
        h = self.record_mgr.get_recent_history(5)
        return PlannerAgent().generate(diag, extracted, tier, h,
                                        self.current_kernel, rn)

    def _call_coder(self, code, plan, prev_err=""):
        from agents.coder import CoderAgent
        return CoderAgent().apply(code, plan.plan_text, prev_err)

    def _verify_with_retry(self, opt_code, orig_code, plan, rd):
        from agents.verifier import VerifierAgent, VerifyResult
        v = VerifierAgent()
        max_r = 3; cur = opt_code
        for a in range(max_r + 1):
            vr = v.verify(rd / "kernel.py", rd)
            if vr.stage1_passed:
                return vr, a
            if a < max_r:
                cr = self._call_coder(cur, plan, vr.stage1_error_details)
                if cr.success:
                    cur = cr.optimized_code
                    (rd / "kernel.py").write_text(cur, encoding="utf-8")
        return vr, max_r

    def _round_dir(self, tier, rn):
        return self.kernel_dir / TIER_DIRS.get(tier, f"tier{tier}") / f"round{rn}"

    def _finalize(self) -> dict:
        s = self.record_mgr.get_state()
        fd = self.kernel_dir / "final_output"
        fd.mkdir(exist_ok=True)

        # 1. 最优 kernel
        if self.best_kernel:
            (fd / "optimized_kernel.py").write_text(self.best_kernel, encoding="utf-8")

        # 2. 尝试生成 Gantt 图 (最后一轮的分析数据)
        try:
            last_merged = None
            for tier_dir in sorted(self.kernel_dir.glob("0*")):
                rounds = sorted(tier_dir.glob("round*"))
                if rounds:
                    mf = rounds[-1] / "merged" / "merged_report.json"
                    if mf.exists():
                        last_merged = mf
            if last_merged:
                shutil.copy2(last_merged, fd / "final_merged_report.json")
                # 生成 LLM + Human 文本
                from analyzers.dsl_merger import format_llm, format_human
                import json as _j
                report = _j.loads(last_merged.read_text(encoding="utf-8"))
                (fd / "final_report_llm.txt").write_text(format_llm(report), encoding="utf-8")
                try:
                    (fd / "final_report_human.txt").write_text(format_human(report), encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            pass

        # 3. 生成优化轨迹图
        try:
            from feedback.trajectory_chart import generate as gen_chart
            gen_chart(self.kernel_dir, fd / "trajectory_chart.png")
        except Exception:
            pass

        print(f"\n{'='*60}\nDONE | {s['round']} rounds | "
              f"{s['best_speedup']:.2f}x | tier={s['tier']}\n{'='*60}")
        print(f"Output: {fd}")
        return {"kernel": self.kernel_name, "rounds": s["round"],
                "best_speedup": s["best_speedup"], "output": str(fd)}


def _plan_md(p) -> str:
    return f"# Round {p.round_num} Plan\n**Tier {p.tier}**: {p.strategy}\n\n{p.specific_change}\n\n{p.expected_impact}"

def _diag_dict(d) -> dict:
    return {"bottleneck": {"op_id": d.bottleneck_op_id, "op_type": d.bottleneck_op_type,
            "engine": d.bottleneck_engine, "type": d.bottleneck_type,
            "category": d.bottleneck_category, "headroom": d.optimization_headroom,
            "time_ratio": d.bottleneck_time_ratio, "bw_utilization": d.bottleneck_bw_utilization,
            "regime": d.bottleneck_regime},
            "strategies": d.suggested_strategies, "structural_issues": d.structural_issues}


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    kd = _PROJECT_DIR / "outputs" / "vector_add_fp16_N65536"
    kf = kd / "round0" / "kernel.py"
    if not kf.exists():
        print("[SKIP] round0/kernel.py not found")
        return
    tj = kd / "optimization_trajectory.json"
    if tj.exists(): tj.unlink()
    orch = Orchestrator(kernel_path=kf, kernel_name="vector_add_fp16_N65536",
                        target_speedup=2.0, max_rounds=3)
    orch.run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        kd = Path(sys.argv[1])
        Orchestrator(kernel_path=kd / "round0" / "kernel.py",
                     kernel_name=kd.name).run()
    else:
        _self_test()
