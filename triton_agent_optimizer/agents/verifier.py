#!/usr/bin/env python3
"""
Verifier Agent — 两阶段验证 (脚本, 非 LLM)。

═══════════════════════════════════════════════════════════════════════════════
  两阶段验证
═══════════════════════════════════════════════════════════════════════════════

  Stage 1: CPU Emulator (秒级, 每轮必跑)
    用 emulators/common 模拟执行 kernel, 验证数值正确性。
    → PASS → 进入 Stage 2
    → FAIL → 错误返回给 Orchestrator, 由 Coder 基于错误重试 (最多3次)

  Stage 2: 910B3 Hardware (分钟级, Stage1 PASS 后跑)
    编译 + benchmark + msprof 采集, 拿到真实性能。
    → 返回真实加速比

  注意: Cost Simulator 不在验证环节 — 它已经在分析层用过了 (瓶颈诊断)。

═══════════════════════════════════════════════════════════════════════════════
  验证 = 跑命令 + 读结果。不需要 LLM。
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


@dataclass
class EmulatorStageResult:
    """Stage 1 结果。"""
    passed: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    error_details: str = ""


@dataclass
class HardwareStageResult:
    """Stage 2 结果。"""
    tested: bool = False          # 是否实际运行了 (本地环境可能跳过)
    passed: bool = False
    latency_ms: float = 0.0
    throughput_gb_s: float = 0.0
    speedup_vs_baseline: float = 1.0


@dataclass
class VerifyResult:
    """Verifier 完整验证结果。"""
    # Stage 1
    stage1_passed: bool
    stage1_max_abs_error: float = 0.0
    stage1_error_details: str = ""

    # Stage 2
    stage2_tested: bool = False
    stage2_passed: bool = False
    stage2_actual_speedup: Optional[float] = None

    # 综合
    overall_passed: bool = False

    @property
    def speedup(self) -> float:
        """拿最高精度加速比: hardware > 1.0 (默认)"""
        if self.stage2_actual_speedup is not None:
            return self.stage2_actual_speedup
        return 1.0   # 未上板时默认 1.0 (Orchestrator 会 REVERT)


class VerifierAgent:
    """两阶段验证协调器。

    Usage:
        v = VerifierAgent()
        result = v.verify(
            kernel_code=optimized_code,
            round_dir=Path("outputs/.../round1"),
            baseline_latency_ms=0.0183,
        )
    """

    def __init__(self, skip_hardware_on_local: bool = True):
        self.skip_hardware = skip_hardware_on_local

        from execution.emulator_runner import EmulatorRunner
        from execution.hardware_runner import HardwareRunner

        self.emulator = EmulatorRunner()
        self.hardware = HardwareRunner()

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════════════════════

    def verify(
        self,
        kernel_code: str,
        round_dir: Optional[Path] = None,
        baseline_latency_ms: float = 0.0,
    ) -> VerifyResult:
        """完整两阶段验证。

        Args:
            kernel_code: Coder 产出的优化后 kernel 代码
            round_dir: 本轮输出目录 (用于保存 benchmark 结果)
            baseline_latency_ms: round0 的基准延迟

        Returns:
            VerifyResult
        """

        # ── Stage 1: CPU Emulator ──
        emu = self._run_stage1(kernel_code)
        if not emu.passed:
            return VerifyResult(
                stage1_passed=False,
                stage1_error_details=emu.error_details,
                overall_passed=False,
            )

        # ── Stage 2: 910B3 Hardware ──
        hw = self._run_stage2(kernel_code, round_dir, baseline_latency_ms)

        return VerifyResult(
            stage1_passed=True,
            stage1_max_abs_error=emu.max_abs_error,
            stage2_tested=hw.tested,
            stage2_passed=hw.passed,
            stage2_actual_speedup=hw.speedup_vs_baseline,
            overall_passed=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 1: CPU Emulator
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage1(self, kernel_code: str) -> EmulatorStageResult:
        """用 emulators/common 模拟执行, 验证正确性。"""
        r = self.emulator.verify(kernel_code)
        return EmulatorStageResult(
            passed=r.passed,
            max_abs_error=r.max_abs_error,
            max_rel_error=r.max_rel_error,
            error_details=r.error_details,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 2: 910B3 Hardware
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage2(
        self, kernel_code: str, round_dir: Optional[Path],
        baseline_latency_ms: float,
    ) -> HardwareStageResult:
        """在 910B3 上编译 + benchmark。本地环境跳过。"""
        if not self.hardware.available:
            return HardwareStageResult(tested=False)

        if self.skip_hardware:
            return HardwareStageResult(tested=False)

        try:
            # 编译
            from execution.compiler import CompilerInterface
            compiler = CompilerInterface()
            compile_r = compiler.compile(kernel_code,
                round_dir or Path("."))
            if not compile_r.success:
                return HardwareStageResult(tested=True, passed=False)

            # benchmark
            hw_r = self.hardware.benchmark(
                Path(compile_r.binary_path), baseline_latency_ms)
            return HardwareStageResult(
                tested=True, passed=hw_r.success,
                latency_ms=hw_r.latency_ms,
                throughput_gb_s=hw_r.throughput_gb_s,
                speedup_vs_baseline=hw_r.speedup_vs_baseline,
            )

        except Exception:
            return HardwareStageResult(tested=True, passed=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    v = VerifierAgent()
    code = "import triton; import triton.language as tl\n@triton.jit\ndef k(): pass"
    r = v.verify(code)
    print(f"Stage1: passed={r.stage1_passed}")
    print(f"Stage2: tested={r.stage2_tested} (skipped on local)")
    print(f"Overall: passed={r.overall_passed}")
    assert r.overall_passed
    print("[Verifier] OK")


if __name__ == "__main__":
    _self_test()
