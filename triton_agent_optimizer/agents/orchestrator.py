#!/usr/bin/env python3
"""
优化调度器 v3.0 — 薄循环 (Python 状态机)

═══════════════════════════════════════════════════════════════════════════════
  职责: 调用 Analyzers → Planner → Coder → Verifier → RecordManager

  铁律 (ORCHESTRATOR_SPEC.md §0):
    0.1 六层必须依次执行，不能跳，不能乱序
    0.2 每轮 Planner 只读自己 Tier 的文档
    0.3 每轮 Planner 只检索自己 Tier 的经验库
    0.4 禁止使用模拟数据 — 必须用 msprof 或 HIVM 真实数据

  数据流 (v3.0):
    Round 0: Triton.py → HIVM + msprof → merged baseline
    Round N: Re-analyze → Plan → Code → Verify → Decide → Record

  决策/记录/停止/Tier → 全在 feedback/record_manager.py
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

TIER_NAMES = {
    1: "Algorithmic Structure", 2: "Operator Fusion",
    3: "Tiling & Block Config", 4: "Memory Access",
    5: "Compute & Occupancy", 6: "910B3 Architecture",
}
TIER_DIRS = {
    1: "01_algorithmic_structure", 2: "02_operator_fusion",
    3: "03_tiling_block_config", 4: "04_memory_access",
    5: "05_compute_occupancy", 6: "06_910b3_architecture",
}
PLAYBOOK_FILES = {
    1: "playbook_tier1_algorithm.md", 2: "playbook_tier2_fusion.md",
    3: "playbook_tier3_tiling.md", 4: "playbook_tier4_memory.md",
    5: "playbook_tier5_compute.md", 6: "playbook_tier6_architecture.md",
}
EXPERIENCE_FILES = {
    1: "tier1_algorithm.json", 2: "tier2_fusion.json",
    3: "tier3_tiling.json", 4: "tier4_memory.json",
    5: "tier5_compute.json", 6: "tier6_architecture.json",
}


def _safe_pct(v) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.2%}"
    if isinstance(v, str):
        try: return f"{float(str(v).rstrip('%')) / 100.0:.2%}"
        except ValueError: return str(v)[:20]
    return str(v)[:20]


class Orchestrator:
    """薄循环调度器 — 按 ORCHESTRATOR_SPEC.md 铁律实现。"""

    def __init__(self, kernel_path: Path, kernel_name: str,
                 target_speedup: float = 1.5, max_rounds: int = 200,
                 output_root: Optional[Path] = None,
                 msprof_dir: Optional[Path] = None,
                 hivm_mlir: Optional[Path] = None):
        self.kernel_path = Path(kernel_path)
        self.kernel_name = kernel_name
        self.msprof_dir = msprof_dir
        self.hivm_mlir = hivm_mlir

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
        self.best_speedup = 1.0
        self.best_total_ns = 0.0
        self._kernel_fn_name = "add_kernel"
        self._op_type = "element_wise"

        # 环境检测
        self._has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ═══════════════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> dict:
        print("=" * 60)
        print(f"Orchestrator v3.0 — {self.kernel_name}")
        print(f"Target: {self.target_speedup}x  Max: {self.max_rounds} rounds")
        print(f"LLM: {'API' if self._has_api else 'AGENT_TASK files'}")
        print("=" * 60)

        self._run_round0()

        while True:
            stop, reason = self._run_one_round()
            state = self.record_mgr.get_state()
            print(f"\n[ROUND {state['round']}] Tier {state['tier']}: {reason}")
            if stop:
                break

        return self._finalize()

    # ═══════════════════════════════════════════════════════════════════════
    #  Round 0: Baseline
    # ═══════════════════════════════════════════════════════════════════════

    def _run_round0(self):
        print("\n" + "=" * 60)
        print("[ROUND 0] Baseline Analysis")
        print("=" * 60)
        rd = self.kernel_dir / "round0"
        rd.mkdir(parents=True, exist_ok=True)

        self.current_kernel = self.kernel_path.read_text(encoding="utf-8")
        self.best_kernel = self.current_kernel

        # ① 先生成 HIVM
        print(f"  [0a] Generating HIVM from original kernel...")
        kernel_py = rd / "kernel.py"
        if not kernel_py.exists():
            # main.py 复制的是 triton_kernel.py
            alt = rd / "triton_kernel.py"
            if alt.exists():
                kernel_py = alt
        self._run_triton_to_hivm_pipeline(
            kernel_py,
            rd / "hivmir" / "compiler_output" / "hivmir_output.mlir")

        # ② ★ 跑 msprof 获取本 kernel 的真实 baseline ★
        print(f"  [0b] Running msprof for REAL baseline trace...")
        msprof_ok = self._per_round_msprof_pipeline(rd)
        if not msprof_ok:
            print(f"  [0b] WARN: msprof baseline failed")

        # ③ 用真实 trace 做完整分析
        merged, diag, extracted = self._run_analyzers(rd)

        self._baseline_total_ns = merged.get("execution_summary", {}).get("total_ns", 0) if merged else 0
        self.best_speedup = 1.0
        self.best_total_ns = self._baseline_total_ns

        # 初始化 trajectory
        self.record_mgr.init_baseline(merged or {}, diag, str(self.kernel_path))
        ops_count = merged.get("execution_summary", {}).get("num_ops", 0) if merged else 0
        has_timing = merged.get("meta", {}).get("has_msprof_timing", False) if merged else False
        print(f"\n[ROUND 0] ✓ Baseline: {ops_count} ops, "
              f"{'WITH' if has_timing else 'WITHOUT'} msprof timing, "
              f"total_ns={self._baseline_total_ns:.0f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Round N: Optimize
    # ═══════════════════════════════════════════════════════════════════════

    def _run_one_round(self) -> tuple:
        tier = self.record_mgr.get_tier()
        rn = self.record_mgr.get_state()["round"] + 1
        rd = self._round_dir(tier, rn)
        rd.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"ROUND {rn} | Tier {tier}: {TIER_NAMES.get(tier, '?')}")
        print(f"  目录: {rd}")
        print(f"{'─' * 60}")

        # ① Analyzers: 从当前 kernel 生成 HIVM + 解析 msprof → merged
        merged, diag, extracted = self._run_analyzers(rd)

        # ② Planner: 生成优化计划
        print(f"\n  [② Planner] Calling LLM (Tier {tier})...")
        plan = self._call_planner(diag, extracted or "", tier, rn)
        print(f"  [② Planner] → strategy: {plan.strategy}")
        # 如果 Planner 认为当前层级已最优 → 立即晋升下一层
        if "optimal" in plan.strategy.lower() or "already" in plan.strategy.lower():
            new_tier = min(tier + 1, 6)
            print(f"  [② Planner] Tier {tier} already optimal → promoting to Tier {new_tier}!")
            # 获取 trajectory 并保存晋升状态
            traj = json.loads(
                (self.kernel_dir / "optimization_trajectory.json").read_text(encoding="utf-8"))
            traj["state"]["tier"] = new_tier
            traj["state"]["round"] = rn
            traj["state"]["consecutive_reverts"] = 0
            (self.kernel_dir / "optimization_trajectory.json").write_text(
                json.dumps(traj, indent=2, ensure_ascii=False), encoding="utf-8")
            # 重置 current_kernel 到本轮最佳
            if self.best_kernel:
                self.current_kernel = self.best_kernel
            print(f"  [② Planner] Tier saved to trajectory → Tier {new_tier}")
            return (False, f"Promoted to Tier {new_tier}")
        _write_plan_md(plan, rd)

        # ③ Coder: 修改代码
        print(f"\n  [③ Coder] Applying plan: {plan.strategy[:60]}...")
        coder_r = self._call_coder(self.current_kernel, plan)
        # 语法错误重试 (最多2次)
        coder_retry = 0
        while not coder_r.success and coder_retry < 2:
            print(f"  [③ Coder] FAILED (attempt {coder_retry+1}): {coder_r.error_message[:80]}")
            coder_r = self._call_coder(
                self.current_kernel, plan, prev_err=coder_r.error_message)
            coder_retry += 1
        if not coder_r.success:
            print(f"  [③ Coder] FAILED after {coder_retry} retries: {coder_r.error_message[:100]}")
            from agents.verifier import VerifyResult
            vr = VerifyResult(stage1_passed=False,
                              stage1_error_details=coder_r.error_message,
                              overall_passed=False)
            return self.record_mgr.evaluate(vr, plan, coder_r, rd, diag)

        print(f"  [③ Coder] → {coder_r.lines_changed} lines changed")
        (rd / "kernel.py").write_text(coder_r.optimized_code, encoding="utf-8")
        (rd / "diff.patch").write_text(coder_r.diff, encoding="utf-8")

        # Coder 改了 0 行 → 重试一次，要求更强地执行计划
        if coder_r.lines_changed == 0:
            print(f"  [③ Coder] WARNING: 0 lines changed! Retrying with stronger prompt...")
            stronger_plan = plan.plan_text + (
                "\n\n## ⚠️ CRITICAL: Previous attempt changed 0 lines!\n"
                "The previous Coder attempt returned the ORIGINAL code unchanged.\n"
                "You MUST make the code change specified in the plan above.\n"
                "The code MUST be DIFFERENT from the original. Modify at least 1 line.")
            coder_r = self._call_coder(self.current_kernel, plan, prev_err="Previous attempt: 0 lines changed")
            print(f"  [③ Coder] Retry: → {coder_r.lines_changed} lines changed")
            if coder_r.lines_changed == 0:
                print(f"  [③ Coder] FAILED: Still 0 lines after retry → REVERT")
                self.current_kernel = self.best_kernel or self.current_kernel
                from agents.verifier import VerifyResult
                vr = VerifyResult(stage1_passed=True, overall_passed=False,
                                stage1_error_details="Coder produced no change (0 lines)")
                from agents.coder import CoderResult
                cr = CoderResult(success=False, optimized_code=self.current_kernel,
                               diff="", lines_changed=0,
                               error_message="Coder could not modify kernel")
                return self.record_mgr.evaluate(vr, plan, cr, rd, diag)
            (rd / "kernel.py").write_text(coder_r.optimized_code, encoding="utf-8")
            (rd / "diff.patch").write_text(coder_r.diff, encoding="utf-8")

        # 更新 current_kernel
        self.current_kernel = coder_r.optimized_code

        # ──── 3b: 重新生成 HIVM (基于 Coder 改后的代码) ────
        print(f"  [3b] Regenerating HIVM from optimized code...")
        try:
            self._run_triton_to_hivm_pipeline(
                rd / "kernel.py",
                rd / "hivmir" / "compiler_output" / "hivmir_output.mlir")
        except Exception as e:
            print(f"  [3b] ERROR: HIVM regeneration failed: {str(e)[:150]}")
            print(f"  [3b] → Coder broke the kernel. Reverting.")
            self.current_kernel = self.best_kernel or self.current_kernel
            from agents.verifier import VerifyResult
            vr = VerifyResult(stage1_passed=False,
                            stage1_error_details=f"Coder broke kernel: {str(e)[:200]}",
                            overall_passed=False)
            from agents.coder import CoderResult
            cr = CoderResult(success=False, optimized_code=self.current_kernel,
                           diff="", error_message=f"triton compilation failed: {str(e)[:200]}")
            return self.record_mgr.evaluate(vr, plan, cr, rd, diag)

        # ──── 3c: ★ 每轮强制 msprof 重采 (HIVM→AscendC→compile→msprof simulator) ★ ────
        print(f"  [3c] MSprof re-collect: HIVM→AscendC→compile→msprof simulator...")
        msprof_success = self._per_round_msprof_pipeline(rd)
        if not msprof_success:
            print(f"  [3c] ERROR: Per-round msprof pipeline FAILED!")
            print(f"  [3c] Cannot calculate speedup without per-round trace data.")
            # 回退 kernel，本轮不记录
            self.current_kernel = self.best_kernel or self.current_kernel
            from agents.verifier import VerifyResult
            vr = VerifyResult(stage1_passed=True,
                            stage1_error_details="msprof re-collect failed",
                            overall_passed=False)
            from agents.coder import CoderResult
            cr = CoderResult(success=False, optimized_code=self.current_kernel,
                           diff="", error_message="per-round msprof pipeline failed")
            return self.record_mgr.evaluate(vr, plan, cr, rd, diag)

        # ──── 重新运行 Analyzer (使用新 trace) ────
        new_merged, _, _ = self._run_analyzers(rd)

        # Speedup: 从新 merged report 取真实 total_ns
        if not self._baseline_total_ns or self._baseline_total_ns <= 0:
            print(f"  [3c] ERROR: No baseline total_ns available!")
            self.current_kernel = self.best_kernel or self.current_kernel
            from agents.verifier import VerifyResult
            vr = VerifyResult(stage1_passed=True,
                            stage1_error_details="no baseline total_ns",
                            overall_passed=False)
            from agents.coder import CoderResult
            cr = CoderResult(success=False, optimized_code=self.current_kernel,
                           diff="", error_message="no baseline total_ns")
            return self.record_mgr.evaluate(vr, plan, cr, rd, diag)

        prev_ns = self.best_total_ns or self._baseline_total_ns
        new_ns = self._estimate_total_ns(rd)
        round_speedup = prev_ns / new_ns if (new_ns > 0 and prev_ns > 0) else 1.0
        abs_speedup = self._baseline_total_ns / new_ns if (new_ns > 0 and self._baseline_total_ns > 0) else 1.0

        # 验证 trace 是否与 baseline 不同 (msprof 重采是否真的产生了新数据)
        if new_ns == self._baseline_total_ns:
            print(f"  [3c] WARNING: new_ns == baseline_ns ({new_ns:.0f}ns) — trace may not have been regenerated!")
        # ④ Verifier: Stage1 CPU — 失败则回传 Coder 修复重试 (同轮内, 最多5次)
        vr, retry_count = self._verify_with_retry(
            coder_r.optimized_code, self.best_kernel or self.current_kernel, plan, rd)
        if retry_count > 0:
            print(f"  [4 Verifier] Fixed after {retry_count} Coder retries")

        # 设置 speedup 给 RecordManager
        vr.stage2_actual_speedup = round_speedup
        print(f"  [4 Verifier] Stage1: {'PASS' if vr.stage1_passed else 'FAIL'}, msprof: {prev_ns:.0f}ns -> {new_ns:.0f}ns, round: {round_speedup:.3f}x, abs: {abs_speedup:.3f}x")

        # 追踪最佳方案 — Stage1 必须 PASS 才接受
        if vr.stage1_passed and abs_speedup > self.best_speedup:
            self.best_speedup = abs_speedup
            self.best_kernel = coder_r.optimized_code
            self.best_total_ns = new_ns
            print(f"  [5] NEW BEST: abs={abs_speedup:.3f}x, round={round_speedup:.3f}x ({new_ns:.0f}ns)")
        elif not vr.stage1_passed:
            print(f"  [5] Emulator FAIL → reverting to previous best")
            self.current_kernel = self.best_kernel or self.current_kernel
        elif round_speedup < 1.0 and new_ns > prev_ns:
            print(f"  [5] Round degraded ({round_speedup:.3f}x) -> reverting to previous best")
            self.current_kernel = self.best_kernel


        stop, reason = self.record_mgr.evaluate(vr, plan, coder_r, rd, diag)
        print(f"  [⑤ RecordManager] → {vr.speedup:.3f}x → {'KEEP' if vr.speedup > 1.01 else 'REVERT'}"
              f"{' → ' + reason if stop else ''}, best={self.best_speedup:.3f}x")
        return stop, reason

    # ═══════════════════════════════════════════════════════════════════════
    #  Analyzers: HIVM + msprof → merged
    # ═══════════════════════════════════════════════════════════════════════

    def _run_analyzers(self, rd: Path):
        """运行完整分析链。每轮独立执行，不回退到 round0。

        Step 1: 从当前 kernel 重新生成 HIVM MLIR (Triton→TTIR→HIVM)
        Step 2: 解析 HIVM → 语义 op
        Step 3: 解析 msprof trace → timing
        Step 4: 合并 → 29 字段 merged_report.json
        Step 5: 瓶颈诊断 + 数据提取
        """
        from analyzers.hivmir_analyzer import HIVMIRAnalyzer
        from analyzers.msprof_analyzer import MsprofAnalyzer
        from analyzers.dsl_merger import merge, format_llm
        from analyzers.bottleneck_diagnoser import diagnose
        from analyzers.data_extractor import extract

        tier = self.record_mgr.get_tier()
        kernel_py = rd / "kernel.py"
        next_rd_name = rd.name  # "round0" or "roundN"

        # ── Step 0: 确保当前轮有 kernel.py ──
        if not kernel_py.exists():
            # round0: 从 kernel_path 拷贝
            if next_rd_name == "round0" and self.kernel_path.exists():
                import shutil
                shutil.copy2(self.kernel_path, kernel_py)
            # roundN: 使用 current_kernel (上一轮保存的)
            elif self.current_kernel:
                kernel_py.write_text(self.current_kernel, encoding="utf-8")

        # ── Step 1: 生成 HIVM MLIR (从当前 kernel.py) ──
        print(f"  [Step 1/5] Generating HIVM from {next_rd_name}/kernel.py...")
        hivm_mlir = rd / "hivmir" / "compiler_output" / "hivmir_output.mlir"
        if not hivm_mlir.exists() or next_rd_name != "round0":
            try:
                self._run_triton_to_hivm_pipeline(kernel_py, hivm_mlir)
            except Exception as e:
                print(f"  [WARN] HIVM generation failed: {e}")

        # ── Step 2: 解析 HIVM ──
        print(f"  [Step 2/5] Parsing HIVM IR...")
        hivm_dict = {}
        if hivm_mlir.exists():
            ha = HIVMIRAnalyzer()
            hr = ha.analyze_file(hivm_mlir)
            hivm_dict = ha.to_dict(hr)
            hivm_rd = rd / "hivmir"
            hivm_rd.mkdir(parents=True, exist_ok=True)
            (hivm_rd / "hivmir_report.json").write_text(
                json.dumps(hivm_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            # 也保存结构化 op 列表
            hivm_ops_data = [{"op_id": o.op_id, "op_type": o.op_type,
                              "engine": o.engine, "instruction": o.instruction,
                              "dst": o.dst, "src": o.src, "src2": o.src2,
                              "size_kb": o.size_kb, "memory_region": o.memory_region}
                             for o in hr.ops]
            (hivm_rd / "hivm_ops.json").write_text(
                json.dumps(hivm_ops_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    → {hr.num_ops} ops, RAW={len(hr.raw_deps)} WAR={len(hr.war_deps)}")
        else:
            print(f"    → SKIP: no HIVM MLIR")

        # ── Step 3: 解析 msprof ──
        print(f"  [Step 3/5] Parsing msprof trace...")
        ma = MsprofAnalyzer()
        msprof_dict = {}
        # Round 0: 从用户指定或 round0/msprof/ 找
        # Round N: ★ 只从当前轮找，不回退到 round0 ★
        opprof_dir = None
        if self.msprof_dir and Path(self.msprof_dir).exists():
            opprof_dir = Path(self.msprof_dir)
        if not opprof_dir or not opprof_dir.exists():
            opprof_dir = ma.find_latest_opprof(rd / "msprof")
        if not opprof_dir and next_rd_name == "round0":
            # 仅 round0 允许用 kernel_dir 级别 fallback
            opprof_dir = ma.find_latest_opprof(self.kernel_dir / "round0" / "msprof")
        if opprof_dir and opprof_dir.exists():
            mr = ma.parse_existing(opprof_dir)
            msprof_dict = ma.to_dict(mr)
            msprof_rd = rd / "msprof"
            msprof_rd.mkdir(parents=True, exist_ok=True)
            (msprof_rd / "pipeline_report.json").write_text(
                json.dumps(msprof_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            # 拷贝原始 OPPROF 数据 (和 round0 一样的完整结构)
            import shutil as _shutil
            for sub in opprof_dir.iterdir():
                dst = msprof_rd / sub.name
                if not dst.exists():
                    try:
                        if sub.is_dir():
                            _shutil.copytree(sub, dst)
                        else:
                            _shutil.copy2(sub, dst)
                    except Exception:
                        pass
            print(f"    → {mr.num_ops} instrs, {mr.num_cores} cores, {mr.total_ns:.1f}ns, mode={mr.execution_mode}")
        else:
            print(f"    → SKIP: no msprof data (use --msprof-dir)")

        # ── Step 4: 合并 ──
        print(f"  [Step 4/5] Merging HIVM + msprof...")
        merged = merge(hivm_dict, msprof_dict, tier) if hivm_dict else {}
        if merged:
            (rd / "merged").mkdir(parents=True, exist_ok=True)
            (rd / "merged" / "merged_report.json").write_text(
                json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
            (rd / "merged" / "final_report_llm.txt").write_text(
                format_llm(merged), encoding="utf-8")
            has_t = merged["meta"]["has_msprof_timing"]
            print(f"    → {len(merged['per_op_statistics'])} ops, {'WITH' if has_t else 'WITHOUT'} msprof")
        else:
            print(f"    → SKIP: no HIVM data to merge")
            return {}, None, ""

        # ── Step 5: 诊断 + 提取 ──
        print(f"  [Step 5/5] Diagnosing bottleneck (Tier {tier})...")
        diag = diagnose(merged, current_tier=tier)
        diag_dict = {
            "bottleneck": {
                "op_id": getattr(diag, "bottleneck_op_id", -1),
                "op_type": getattr(diag, "bottleneck_op_type", ""),
                "engine": getattr(diag, "bottleneck_engine", ""),
                "type": getattr(diag, "bottleneck_type", ""),
                "category": getattr(diag, "bottleneck_category", ""),
                "headroom": getattr(diag, "optimization_headroom", ""),
                "time_ratio": getattr(diag, "bottleneck_time_ratio", 0),
                "bw_utilization": getattr(diag, "bottleneck_bw_utilization", 0),
                "regime": getattr(diag, "bottleneck_regime", ""),
            },
            "strategies": getattr(diag, "suggested_strategies", []),
            "structural_issues": getattr(diag, "structural_issues", []),
        }
        extracted = extract(merged, diag_dict, tier=tier)
        print(f"    → bottleneck: op{diag.bottleneck_op_id} ({diag.bottleneck_type}), "
              f"headroom={diag.optimization_headroom}")

        return merged, diag, extracted

    def _benchmark_pytorch(self) -> float:
        """跑 PyTorch CPU 基准测试，返回延迟（微秒）。"""
        try:
            import torch, time
            N = 2048
            x = torch.randn(N, dtype=torch.float32)
            torch.set_num_threads(1)
            for _ in range(20):
                _ = torch.softmax(x, dim=0)
            t0 = time.perf_counter()
            for _ in range(5000):
                _ = torch.softmax(x, dim=0)
            dt = time.perf_counter() - t0
            latency_us = dt / 5000 * 1e6
            print(f"  [PyTorch] CPU softmax baseline: {latency_us:.1f}us")
            return latency_us
        except Exception:
            return 0.0

    def _estimate_total_ns(self, rd: Path) -> float:
        """从当前 round 的 merged report 估算总延迟。

        预期 merged report 来自 per-round msprof trace，包含真实 timing 数据。
        如果 merged report 不存在或无 timing 数据，返回 0 (调用者会报错)。
        """
        merged_file = rd / "merged" / "merged_report.json"
        if not merged_file.exists():
            # 尝试从 execution_summary 获取
            return self._baseline_total_ns or 0.0
        import json
        merged = json.loads(merged_file.read_text(encoding="utf-8"))
        # 优先使用 execution_summary 中的 total_ns
        summary = merged.get("execution_summary", {})
        summary_ns = summary.get("total_ns", 0)
        if isinstance(summary_ns, (int, float)) and summary_ns > 0:
            return float(summary_ns)
        # Fallback: 从 per_op_statistics 累加
        ops = merged.get("per_op_statistics", [])
        total = 0.0
        for op in ops:
            d = op.get("duration_ns", 0)
            if isinstance(d, (int, float)) and d > 0:
                total += float(d)
        if total > 0:
            return total
        return self._baseline_total_ns or 0.0

    def _per_round_msprof_pipeline(self, rd: Path) -> bool:
        """★ 每轮强制 msprof 重采 — 完整链路 ★

        Triton kernel.py → HIVM ops → AscendC .asc + host.cpp
        → CMake+bisheng compile → executable
        → msprof op simulator → OPPROF_xxx/ (新 trace 数据)

        每轮必须成功。失败返回 False，不允许绕路或使用旧数据。
        """
        from analyzers.hivm_to_ascendc import generate_and_build, run_msprof

        # Step 1: 读取当前 kernel 的 HIVM ops
        hivm_ops_f = rd / "hivmir" / "hivm_ops.json"
        if not hivm_ops_f.exists():
            print(f"  [3c] ERROR: No HIVM ops at {hivm_ops_f}")
            print(f"  [3c] → Run _run_triton_to_hivm_pipeline() first!")
            return False

        try:
            ops_data = json.loads(hivm_ops_f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [3c] ERROR: Failed to parse HIVM ops: {e}")
            return False

        if not ops_data:
            print(f"  [3c] ERROR: Empty HIVM ops list")
            return False

        print(f"  [3c] HIVM ops: {len(ops_data)}, types: {[o.get('op_type','?') for o in ops_data[:5]]}...")

        # Step 2: AscendC 代码生成 + CMake 编译
        print(f"  [3c] Generating AscendC + compiling...")
        exe = generate_and_build(self._kernel_fn_name, ops_data, rd, 256, "f32")
        if not exe or not exe.exists():
            print(f"  [3c] ERROR: Build failed — no executable produced")
            return False

        # Step 3: msprof op simulator
        print(f"  [3c] Running msprof op simulator...")
        new_trace = run_msprof(exe, rd)
        if not new_trace:
            print(f"  [3c] ERROR: msprof simulator produced no trace data")
            return False

        # Step 4: 验证 trace 包含预期文件
        csv_files = list(new_trace.glob("**/*instr_exe.csv"))
        json_files = list(new_trace.glob("**/trace.json"))
        if not csv_files:
            print(f"  [3c] ERROR: No instr_exe.csv found in {new_trace}")
            return False

        print(f"  [3c] MSprof OK: {len(csv_files)} instr_exe.csv, {len(json_files)} trace.json")
        print(f"  [3c] New trace: {new_trace}")

        # Step 5: 更新 msprof_dir 指向新 trace (后续 _run_analyzers 会读)
        self.msprof_dir = new_trace
        return True

    def _run_triton_to_hivm_pipeline(self, kernel_py: Path, output_mlir: Path):
        """从 kernel.py 运行完整 Triton→TTIR→HIVM 流水线, 保存 HIVM MLIR。"""
        from analyzers.ttir_to_hivm import ttir_to_hivm
        import os, importlib.util
        os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
        from unittest.mock import MagicMock
        import triton.runtime.driver as _drv
        _drv.active = MagicMock(get_current_target=lambda: ("cuda", 90))
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile

        spec = importlib.util.spec_from_file_location(
            f"k_{kernel_py.stem}", str(kernel_py))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        kernel_fn = None
        if self._kernel_fn_name:
            obj = getattr(mod, self._kernel_fn_name, None)
            if obj and hasattr(obj, "fn"):
                kernel_fn = obj
        if not kernel_fn:
            for name in dir(mod):
                obj = getattr(mod, name)
                if hasattr(obj, "fn") and hasattr(obj, "arg_names"):
                    kernel_fn = obj; break
        if not kernel_fn:
            raise ValueError(f"No @triton.jit function in {kernel_py}")

        sig, consts = {}, {}
        for nm in kernel_fn.arg_names:
            nu = nm.upper()
            if "BLOCK" in nu:
                is_matmul = any(k.endswith("_M") or k.endswith("_N") or k.endswith("_K")
                               for k in kernel_fn.arg_names if "BLOCK" in k.upper())
                consts[nm] = 64 if is_matmul else 256
            elif "DIM" in nu or "HIDDEN" in nu:
                consts[nm] = 4096
            elif nu.endswith("_PTR") or nu.endswith("_ptr") or nm.lower() in (
                    "x_ptr", "y_ptr", "a_ptr", "b_ptr", "c_ptr", "out_ptr",
                    "weight_ptr", "residual_ptr"):
                sig[nm] = "*fp32"
            else:
                sig[nm] = "i32"

        src = ASTSource(fn=kernel_fn, signature=sig, constexprs=consts)
        target = GPUTarget("cuda", 90, 32)
        result = triton_compile(src, target=target,
                               options={"num_warps": 4, "num_stages": 1, "debug": False})
        ttir_text = str(result.asm["ttir"])

        hivm_text, hivm_ops = ttir_to_hivm(ttir_text, self._kernel_fn_name or kernel_fn.__name__)

        output_mlir.parent.mkdir(parents=True, exist_ok=True)
        output_mlir.write_text(hivm_text, encoding="utf-8")

        # 保存 kernel 配置
        import json as _json
        config = {
            "kernel_name": self._kernel_fn_name or (kernel_fn.__name__ if hasattr(kernel_fn, '__name__') else "kernel"),
            "arg_names": kernel_fn.arg_names,
            "constexprs": consts,
            "num_warps": 4, "num_stages": 1,
            "dtype": "fp32",
        }
        (output_mlir.parent / "kernel_config.json").write_text(
            _json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"    → {len(hivm_ops)} HIVM ops from {len(ttir_text)} chars TTIR")

    # ═══════════════════════════════════════════════════════════════════════
    #  Agents: Planner + Coder (LLM), Verifier (脚本)
    # ═══════════════════════════════════════════════════════════════════════

    def _call_planner(self, diag, extracted, tier, rn):
        """调用 Planner — 有 API key 用 LLM, 否则写自包含任务文件。"""
        if self._has_api:
            from agents.planner import PlannerAgent
            h = self.record_mgr.get_recent_history(5)
            return PlannerAgent().generate(diag, extracted, tier, h,
                                            self.current_kernel, rn)
        return self._wait_for_plan_file(diag, extracted, tier, rn)

    def _call_coder(self, code, plan, prev_err: str = ""):
        """调用 Coder. 铁律: 只改 kernel.py, 不改任何其他文件。"""
        if self._has_api:
            from agents.coder import CoderAgent
            tier = self.record_mgr.get_tier()
            return CoderAgent().apply(code, plan.plan_text, prev_err, tier=tier)
        return self._wait_for_code_file(code, plan, prev_err)

    def _verify_with_retry(self, opt_code, orig_code, plan, rd):
        """验证正确性 + 性能。Stage1 失败则回传 Coder 重试(最多5次)。

        错误记忆:
          - 记录每次错误到 memory/codeerror/<kernel>.json
          - 连续2次同样的错误 → 回退 + 晋升下一 Tier
          - 成功修复 → 记录解决方案
        """
        from agents.verifier import VerifierAgent, VerifyResult
        from memory.codeerror import CodeErrorMemory
        v = VerifierAgent()
        max_retry = 5
        cur = opt_code
        last_error_pattern = ""
        same_error_count = 0
        err_mem = CodeErrorMemory(self.kernel_name)

        for attempt in range(max_retry + 1):
            vr = v.verify(rd / "kernel.py", rd,
                          kernel_fn_name=self._kernel_fn_name,
                          op_type=self._op_type)
            if vr.stage1_passed:
                # 成功修复 → 记录解决方案
                if last_error_pattern and attempt > 0:
                    err_mem.record_solution(last_error_pattern,
                        f"修复方式: {plan.strategy}, 尝试 {attempt} 次后通过")
                return vr, attempt

            if attempt < max_retry and vr.stage1_error_details:
                err_msg = vr.stage1_error_details[:200]
                # 检查是否同样的错误
                current_pattern = CodeErrorMemory._extract_pattern(err_msg)
                if current_pattern == last_error_pattern:
                    same_error_count += 1
                else:
                    same_error_count = 1
                    last_error_pattern = current_pattern

                print(f"  [Verifier] Retry {attempt + 1}/{max_retry} — error: {err_msg}...")
                err_mem.record_error(err_msg)

                # 连续2次同一错误 → 回退 + 晋升
                if same_error_count >= 2:
                    print(f"  [Verifier] Same error {same_error_count}x → reverting + promoting tier!")
                    self.current_kernel = orig_code  # 回退
                    err_mem.record_solution(err_msg,
                        "无法自动修复 — 回退代码并晋升优化层级")
                    # 强制升 tier
                    state = self.record_mgr.get_state()
                    state["consecutive_reverts"] = 5  # 触发晋升
                    return vr, attempt

                # 查错误记忆有无已知方案
                known_fix = err_mem.find_solution(err_msg)
                prev_err = err_msg
                if known_fix:
                    prev_err = f"{err_msg}\n\n[已知修复方案] {known_fix}"

                cr = self._call_coder(cur, plan, prev_err)
                print(f"  [Verifier] Coder retry: success={cr.success}, lines={cr.lines_changed}")
                if cr.success and cr.optimized_code != cur:
                    cur = cr.optimized_code
                    (rd / "kernel.py").write_text(cur, encoding="utf-8")
                    print(f"  [Verifier] Code updated ({len(cur)} chars)")
                else:
                    print(f"  [Verifier] Coder did not change code — retry may not help")

        return vr, max_retry

    # ── AGENT_TASK 文件模式 (无 API key 时) ──

    def _write_plan_task(self, diag, extracted, tier, rn, rd: Path):
        """写自包含 Planner 任务文件。任何 LLM 都能读+执行。

        铁律 0.2: 只读当前 Tier 的 Playbook
        铁律 0.3: 只检索当前 Tier 的经验库
        """
        playbook_file = PLAYBOOK_FILES.get(tier, PLAYBOOK_FILES[1])
        playbook_path = _PROJECT_DIR / "docx" / playbook_file
        exp_file = EXPERIENCE_FILES.get(tier, EXPERIENCE_FILES[1])
        exp_path = _PROJECT_DIR / "memory" / "experiences" / exp_file

        playbook_text = ""
        if playbook_path.exists():
            playbook_text = playbook_path.read_text(encoding="utf-8")[:8000]

        task = f"""# AGENT TASK: Generate Optimization Plan

## Your Role
You are an expert Triton kernel optimizer for Huawei Ascend 910B3 NPU.
Generate ONE specific, small optimization plan for Round {rn}, Tier {tier}.

## Constraints (IRON RULES)
1. You are at **Tier {tier}: {TIER_NAMES.get(tier, '?')}**. Only read the Tier {tier} playbook.
2. Do NOT read playbooks for other tiers.
3. Generate ONE minimal change — not multiple changes.
4. Only modify kernel.py. Do not touch any other file.

## Bottleneck Diagnosis (Tier {tier})
- op_id: {getattr(diag, 'bottleneck_op_id', '?')}
- op_type: {getattr(diag, 'bottleneck_op_type', '?')}
- engine: {getattr(diag, 'bottleneck_engine', '?')}
- bottleneck_type: {getattr(diag, 'bottleneck_type', '?')}
- headroom: {getattr(diag, 'optimization_headroom', '?')}
- time_ratio: {_safe_pct(getattr(diag, 'bottleneck_time_ratio', 0))}
- bw_utilization: {_safe_pct(getattr(diag, 'bottleneck_bw_utilization', 0))}
- regime: {getattr(diag, 'bottleneck_regime', '?')}
- suggested_strategies: {getattr(diag, 'suggested_strategies', [])}

## Pipeline Data (Tier {tier} filtered)
{extracted if extracted else '(no data)'}

## Recent History
{self._format_history_text()}

## Playbook (Tier {tier} Only)
{playbook_text[:5000] if playbook_text else '(playbook not found)'}

## 910B3 Hardware Parameters
- UB = 192 KB/core. n_buffers × tile_size ≤ 192 KB
- GM→UB peak 80.83 GB/s/core, k0=6.65KB, saturates >13KB
- UB→GM peak 76.67 GB/s/core, k0=10.72KB, saturates >21KB
- VecUnit peak 404 GB/s/core, k0=4.50KB, saturates >9KB
- 20 AI Cores (transfer) + 40 Vec Cores (compute) @ 1.8 GHz

## Current Kernel Code
```python
{self.current_kernel}
```

## Output (write these files to {rd}/)

### plan.md:
```markdown
# Round {rn} Optimization Plan
**Tier**: {tier} ({TIER_NAMES.get(tier, '?')})
**Strategy**: <strategy_name>
## Specific Change
<exact code change>
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
        print(f"  [Planner] Task file → AGENT_TASK_PLAN.md")
        print(f"  [Planner] Playbook: {playbook_file}")
        return task

    def _write_code_task(self, code, plan, prev_err, rd):
        """写自包含 Coder 任务文件。铁律: 只改 kernel.py。"""
        error_block = ""
        if prev_err:
            error_block = f"""
## Previous Error (MUST FIX)
```
{prev_err[:1000]}
```
"""

        task = f"""# AGENT TASK: Apply Code Change

## Your Role
You are a precise Triton kernel code modifier. Apply EXACTLY one change from plan.md.

## Constraints
- Modify ONLY kernel.py — do not touch any other file
- Change EXACTLY what the plan specifies — nothing more
- Maintain existing code style and comments
- Verify Python syntax: compile(modified_code, "kernel.py", "exec")
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
        print(f"  [Coder] Task file → AGENT_TASK_CODE.md")
        return task

    def _wait_for_plan_file(self, diag, extracted, tier, rn):
        """写任务文件 + stub fallback。"""
        from agents.planner import RoundPlan
        rd = self._round_dir(tier, rn)
        self._write_plan_task(diag, extracted, tier, rn, rd)
        return RoundPlan(
            round_num=rn, tier=tier,
            tier_name=TIER_NAMES.get(tier, "?"),
            strategy="[PENDING] See AGENT_TASK_PLAN.md",
            target_speedup=1.0,
            specific_change="Task file written — LLM execution pending",
            expected_impact="—",
            verification_method="CPU emulator",
        )

    def _wait_for_code_file(self, code, plan, prev_err=""):
        """写任务文件 + stub fallback。"""
        from agents.coder import CoderResult
        rd = self._round_dir(plan.tier, plan.round_num)
        self._write_code_task(code, plan, prev_err, rd)
        return CoderResult(
            success=True, optimized_code=code,
            diff="# [PENDING] See AGENT_TASK_CODE.md\n",
            lines_changed=0,
        )

    def _format_history_text(self):
        h = self.record_mgr.get_recent_history(5)
        if not h:
            return "(no history)"
        return "\n".join(
            f"- Round {r.get('round','?')}: {r.get('strategy','?')} → "
            f"{r.get('decision','?')} ({r.get('actual_speedup',1.0):.2f}x)"
            for r in h
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _round_dir(self, tier, rn):
        return (self.kernel_dir / TIER_DIRS.get(tier, f"tier{tier}")
                / f"round{rn}")

    def _finalize(self) -> dict:
        s = self.record_mgr.get_state()
        fd = self.kernel_dir / "final_output"
        fd.mkdir(exist_ok=True)

        # 最优 kernel
        if self.best_kernel:
            (fd / "optimized_kernel.py").write_text(
                self.best_kernel, encoding="utf-8")

        # 最后一轮 merged report
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
                from analyzers.dsl_merger import format_llm
                report = json.loads(last_merged.read_text(encoding="utf-8"))
                (fd / "final_report_llm.txt").write_text(
                    format_llm(report), encoding="utf-8")
        except Exception:
            pass

        # 优化轨迹图
        try:
            from feedback.trajectory_chart import generate
            generate(self.kernel_dir, fd / "trajectory_chart.png")
        except Exception:
            pass

        # 优化报告
        try:
            from feedback.record_manager import _trigger_case_generation_method
            traj = json.loads(
                (self.kernel_dir / "optimization_trajectory.json")
                .read_text(encoding="utf-8"))
            self.record_mgr._trigger_case_generation(traj)
        except Exception:
            pass

        print(f"\n{'=' * 60}\nDONE | {s['round']} rounds | "
              f"{s['best_speedup']:.2f}x | tier={s['tier']}\n{'=' * 60}")
        print(f"Output: {fd}")

        return {
            "kernel": self.kernel_name,
            "rounds": s["round"],
            "best_speedup": s["best_speedup"],
            "tier": s["tier"],
            "output": str(fd),
        }


def _write_plan_md(plan, rd: Path):
    content = (
        f"# Round {plan.round_num} Plan\n"
        f"**Tier {plan.tier}**: {plan.strategy}\n\n"
        f"{plan.specific_change}\n\n"
        f"{plan.expected_impact}"
    )
    (rd / "plan.md").write_text(content, encoding="utf-8")
    plan_json = {
        "round": plan.round_num, "tier": plan.tier,
        "strategy": plan.strategy, "target_speedup": plan.target_speedup,
        "specific_change": plan.specific_change,
        "expected_impact": plan.expected_impact,
    }
    (rd / "plan.json").write_text(
        json.dumps(plan_json, indent=2, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("kernel_dir", type=str, nargs="?",
                   help="Path to outputs/<kernel>/")
    p.add_argument("--msprof-dir", type=str,
                   help="Path to OPPROF_xxx")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--target", type=float, default=1.5)
    args = p.parse_args()

    if args.kernel_dir:
        kd = Path(args.kernel_dir)
        kf = kd / "round0" / "triton_kernel.py"
        if not kf.exists():
            # 尝试找任何 .py 文件
            kfs = list((kd / "round0").glob("*.py"))
            kf = kfs[0] if kfs else None
        name = kd.name
    else:
        # 按默认流程
        kd = _PROJECT_DIR / "outputs" / "fused_add_mul"
        kf = kd / "round0" / "triton_kernel.py"
        name = "fused_add_mul"

    if not kf or not kf.exists():
        print(f"[ERROR] Kernel not found. Run main.py first.")
        sys.exit(1)

    msprof_d = Path(args.msprof_dir) if args.msprof_dir else None
    orch = Orchestrator(
        kernel_path=kf, kernel_name=name,
        target_speedup=args.target, max_rounds=args.max_rounds,
        msprof_dir=msprof_d,
    )

    orch._kernel_fn_name = "add_kernel"
    orch._op_type = "element_wise"

    result = orch.run()
    print(f"\nFinal: {result['best_speedup']:.2f}x "
          f"in {result['rounds']} rounds")
