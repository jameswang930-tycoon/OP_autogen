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
        self._kernel_fn_name = "add_kernel"  # 可由 main.py 注入
        self._op_type = "element_wise"       # 可由 main.py 注入

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
        """调用 Planner — 优先 Python API, 否则写自包含任务文件。"""
        # 尝试 Python API (有 ANTHROPIC_API_KEY 时)
        if _has_api_key():
            from agents.planner import PlannerAgent
            h = self.record_mgr.get_recent_history(5)
            return PlannerAgent().generate(diag, extracted, tier, h,
                                            self.current_kernel, rn)
        # 回退: 写自包含任务文件
        return self._wait_for_plan_file(diag, extracted, tier, rn)

    def _call_coder(self, code, plan, prev_err=""):
        """调用 Coder — 优先 Python API, 否则写自包含任务文件。"""
        if _has_api_key():
            from agents.coder import CoderAgent
            return CoderAgent().apply(code, plan.plan_text, prev_err)
        # 回退: 写自包含任务文件
        return self._wait_for_code_file(code, plan, prev_err)

    def _write_plan_task(self, diag, extracted, tier, rn, rd: Path):
        """写自包含的 Planner 任务文件。任何 LLM 都能读取执行。"""
        from analyzers.data_extractor import TIER_CONFIGS
        playbook_file = {
            1: "playbook_tier1_algorithm.md", 2: "playbook_tier2_fusion.md",
            3: "playbook_tier3_tiling.md", 4: "playbook_tier4_memory.md",
            5: "playbook_tier5_compute.md", 6: "playbook_tier6_architecture.md",
        }.get(tier, "playbook_tier1_algorithm.md")
        playbook_path = _PROJECT_DIR / "docx" / playbook_file

        task = f"""# AGENT TASK: Generate Optimization Plan

## Your Role
You are an expert Triton kernel optimizer for Huawei Ascend 910B3 NPU.
Generate ONE specific, small optimization plan for Round {rn}, Tier {tier}.

## Instructions
1. Read ALL sections below carefully
2. Read the Playbook file: {playbook_path}
3. Generate ONE specific change — minimal, focused, concrete
4. Write plan.md AND plan.json to this directory: {rd}

## Bottleneck Diagnosis (Tier {tier})
- op_id: {getattr(diag, 'bottleneck_op_id', '?')}
- op_type: {getattr(diag, 'bottleneck_op_type', '?')}
- engine: {getattr(diag, 'bottleneck_engine', '?')}
- bottleneck_type: {getattr(diag, 'bottleneck_type', '?')}
- headroom: {getattr(diag, 'optimization_headroom', '?')}
- time_ratio: {getattr(diag, 'bottleneck_time_ratio', 0):.2%}
- bw_utilization: {getattr(diag, 'bottleneck_bw_utilization', 0):.2%}
- regime: {getattr(diag, 'bottleneck_regime', '?')}
- suggested_strategies: {getattr(diag, 'suggested_strategies', [])}

## Pipeline Data
{extracted if extracted else '(no data)'}

## Recent History
{self._format_history_text()}

## 910B3 Parameters
- UB = 192 KB/core. n_buffers × tile_size ≤ 192 KB
- GM->UB peak 80.83 GB/s/core, k0=6.65KB, saturates >13KB
- UB->GM peak 76.67 GB/s/core, k0=10.72KB, saturates >21KB
- VecUnit peak 404 GB/s/core, k0=4.5KB, saturates >9KB
- 20 AI Cores (transfer) + 40 Vec Cores (compute) @ 1.8 GHz

## Current Kernel Code
```python
{self.current_kernel}
```

## Output (write to {rd}/)

### plan.md:
```markdown
# Round {rn} Optimization Plan
**Tier**: {tier}
**Strategy**: <strategy_name>
## Specific Change
<exact change>
## Expected Impact
<impact with 910B3 reasoning>
## Target Speedup
<number>
## Verification
CPU emulator multi-shape test
```

### plan.json:
```json
{{
  "round": {rn}, "tier": {tier},
  "strategy": "<strategy>", "target_speedup": <number>,
  "specific_change": "<change>", "expected_impact": "<impact>"
}}
```
"""
        (rd / "AGENT_TASK_PLAN.md").write_text(task, encoding="utf-8")
        print(f"  [Planner] Task file written → AGENT_TASK_PLAN.md")
        print(f"  [Planner] Waiting for plan.md...")
        print(f"  [Planner] Run: claude -p \"$(cat {rd / 'AGENT_TASK_PLAN.md'})\"")
        return task

    def _write_code_task(self, code, plan, prev_err, rd):
        """写自包含的 Coder 任务文件。"""
        error_block = ""
        if prev_err:
            error_block = f"""
## Previous Error (MUST FIX)
The last code change caused:
```
{prev_err[:1000]}
```
Fix this error while implementing the plan.
"""

        task = f"""# AGENT TASK: Apply Code Change

## Your Role
You are a precise Triton kernel code modifier. Apply EXACTLY the change from plan.md.
Modify ONLY kernel.py — no other files. Output the COMPLETE modified file.

## Instructions
1. Read plan.md and plan.json from this directory: {rd}
2. Read the current kernel code below
3. Apply EXACTLY the change specified in the plan — nothing more
4. Verify Python syntax: compile(modified_code, "kernel.py", "exec")
5. Write the COMPLETE modified kernel to: {rd}/kernel.py
6. Write unified diff to: {rd}/diff.patch
{error_block}

## Optimization Plan
{plan.plan_text}

## Current Kernel Code
```python
{code}
```

## 910B3 Constraints
- UB = 192 KB/core. n_buffers × tile_size_kb ≤ 192 KB
- fp16 = 2 bytes/elem
- Check: new_tile_kb × n_buffers ≤ 192 KB

## Output Files (write to {rd}/)
- kernel.py — COMPLETE modified kernel code
- diff.patch — unified diff
"""
        (rd / "AGENT_TASK_CODE.md").write_text(task, encoding="utf-8")
        print(f"  [Coder] Task file written → AGENT_TASK_CODE.md")
        print(f"  [Coder] Waiting for kernel.py update...")
        return task

    def _wait_for_plan_file(self, diag, extracted, tier, rn):
        """写任务文件 + 等待 or stub fallback。"""
        from agents.planner import RoundPlan
        rd = self._round_dir(tier, rn)
        self._write_plan_task(diag, extracted, tier, rn, rd)
        # Stub fallback (无 LLM 环境时)
        return RoundPlan(round_num=rn, tier=tier,
                         tier_name=TIER_NAMES.get(tier, "?"),
                         strategy="[PENDING] See AGENT_TASK_PLAN.md",
                         target_speedup=1.0,
                         specific_change="Task file written — LLM execution pending",
                         expected_impact="—",
                         verification_method="CPU emulator")

    def _wait_for_code_file(self, code, plan, prev_err):
        """写任务文件 + 等待 or stub fallback。"""
        from agents.coder import CoderResult
        rd = self._round_dir(plan.tier, plan.round_num)
        self._write_code_task(code, plan, prev_err, rd)
        # Stub fallback
        return CoderResult(success=True, optimized_code=code,
                           diff="# [PENDING] See AGENT_TASK_CODE.md\n",
                           lines_changed=0)

    def _format_history_text(self):
        h = self.record_mgr.get_recent_history(5)
        if not h:
            return "(no history)"
        lines = []
        for r in h:
            lines.append(
                f"- Round {r.get('round','?')}: {r.get('strategy','?')} → "
                f"{r.get('decision','?')} ({r.get('actual_speedup',1.0):.2f}x)")
        return "\n".join(lines)

    def _verify_with_retry(self, opt_code, orig_code, plan, rd):
        from agents.verifier import VerifierAgent, VerifyResult
        v = VerifierAgent()
        max_r = 3; cur = opt_code
        for a in range(max_r + 1):
            vr = v.verify(rd / "kernel.py", rd,
                          kernel_fn_name=self._kernel_fn_name,
                          op_type=self._op_type)
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


def _has_api_key() -> bool:
    import os
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


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
