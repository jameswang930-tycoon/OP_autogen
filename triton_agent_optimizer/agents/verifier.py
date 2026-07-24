#!/usr/bin/env python3
"""
Verifier Agent — 两阶段验证 (脚本, 非 LLM)。

═══════════════════════════════════════════════════════════════════════════════
  数据流
═══════════════════════════════════════════════════════════════════════════════

  Coder → round_N/kernel.py
    │
    ▼
  Verifier.verify(kernel_path, round_dir, baseline_latency_ms)
    │
    ├─ Stage 1: CPU Emulator
    │   emulator_runner.verify(kernel_path) → EmulatorResult
    │   → PASS: 继续
    │   → FAIL: 返回 error_details → Orchestrator → Coder 重试
    │
    ├─ Stage 2: 910B3 Hardware (本地跳过)
    │   hardware_runner.benchmark(binary) → HardwareResult
    │
    ├─ 保存 verification.json 到 round_N/
    │
    └─ 返回 VerifyResult 给 Orchestrator
         Orchestrator 读 speedup → KEEP/REVERT
         Orchestrator 读 error_details → Coder 重试
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmulatorStageResult:
    passed: bool
    max_abs_error: float = 0.0; max_rel_error: float = 0.0
    error_details: str = ""
    shapes_tested: int = 0; shapes_failed: int = 0


@dataclass
class HardwareStageResult:
    tested: bool = False; passed: bool = False
    latency_ms: float = 0.0; throughput_gb_s: float = 0.0
    speedup_vs_baseline: float = 1.0


@dataclass
class VerifyResult:
    """Verifier 返回给 Orchestrator 的完整结果。"""
    stage1_passed: bool
    stage1_max_abs_error: float = 0.0
    stage1_error_details: str = ""

    stage2_tested: bool = False
    stage2_passed: bool = False
    stage2_actual_speedup: Optional[float] = None

    overall_passed: bool = False

    @property
    def speedup(self) -> float:
        """Orchestrator 用这个值决定 KEEP/REVERT。"""
        if self.stage2_actual_speedup is not None:
            return self.stage2_actual_speedup
        return 1.0   # 未上板 → 默认1.0 → REVERT


# ═══════════════════════════════════════════════════════════════════════════════
#  Verifier
# ═══════════════════════════════════════════════════════════════════════════════

class VerifierAgent:
    """两阶段验证。"""

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
        kernel_path: Path,
        round_dir: Optional[Path] = None,
        kernel_fn_name: str = "add_kernel",
        baseline_latency_ms: float = 0.0,
    ) -> VerifyResult:
        """完整两阶段验证。

        Args:
            kernel_path: round_N/kernel.py 文件路径 (Coder 已写入)
            round_dir: 本轮输出目录
            kernel_fn_name: kernel 函数名
            baseline_latency_ms: round0 基准延迟

        Returns:
            VerifyResult → Orchestrator 用于决定 KEEP/REVERT
        """
        kernel_path = Path(kernel_path)

        # ── Stage 1: CPU Emulator ──
        emu = self._run_stage1(kernel_path, kernel_fn_name)
        if not emu.passed:
            return VerifyResult(
                stage1_passed=False,
                stage1_error_details=emu.error_details,
                overall_passed=False,
            )

        # ── Stage 2: 910B3 Hardware ──
        hw = self._run_stage2(kernel_path, round_dir, baseline_latency_ms)

        result = VerifyResult(
            stage1_passed=True,
            stage1_max_abs_error=emu.max_abs_error,
            stage2_tested=hw.tested,
            stage2_passed=hw.passed,
            stage2_actual_speedup=hw.speedup_vs_baseline if hw.tested else None,
            overall_passed=True,
        )

        # ── 保存 verification.json 到 round_N/ ──
        if round_dir:
            self._save_result(result, round_dir, emu, hw)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 1: CPU Emulator
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage1(self, kernel_path: Path, fn_name: str) -> EmulatorStageResult:
        """从 round_N/kernel.py 导入 kernel, 用 emulators/common 验证正确性。"""
        r = self.emulator.verify(kernel_path, kernel_fn_name=fn_name)
        shapes_tested = len(self.emulator.DEFAULT_SHAPES)
        return EmulatorStageResult(
            passed=r.passed,
            max_abs_error=r.max_abs_error,
            max_rel_error=r.max_rel_error,
            error_details=r.error_details,
            shapes_tested=shapes_tested,
            shapes_failed=len(r.failed_shapes),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 2: 910B3 Hardware
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage2(
        self, kernel_path: Path, round_dir: Optional[Path],
        baseline_latency_ms: float,
    ) -> HardwareStageResult:
        """在 910B3 上编译 + benchmark。"""
        if not self.hardware.available:
            return HardwareStageResult(tested=False)

        if self.skip_hardware:
            return HardwareStageResult(tested=False)

        try:
            # 编译
            from execution.compiler import CompilerInterface
            compiler = CompilerInterface()
            code = kernel_path.read_text(encoding="utf-8")
            compile_r = compiler.compile(code, round_dir or kernel_path.parent)
            if not compile_r.success:
                return HardwareStageResult(tested=True, passed=False)

            # benchmark
            hw_r = self.hardware.benchmark(Path(compile_r.binary_path), baseline_latency_ms)
            return HardwareStageResult(
                tested=True, passed=hw_r.success,
                latency_ms=hw_r.latency_ms,
                throughput_gb_s=hw_r.throughput_gb_s,
                speedup_vs_baseline=hw_r.speedup_vs_baseline,
            )
        except Exception:
            return HardwareStageResult(tested=True, passed=False)

    # ═══════════════════════════════════════════════════════════════════════════
    #  结果保存
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_result(self, vr: VerifyResult, round_dir: Path,
                     emu: EmulatorStageResult, hw: HardwareStageResult):
        """保存 verification.json 到 round_N/。"""
        data = {
            "stage1": {
                "passed": emu.passed,
                "max_abs_error": emu.max_abs_error,
                "max_rel_error": emu.max_rel_error,
                "shapes_tested": emu.shapes_tested,
                "shapes_failed": emu.shapes_failed,
            },
            "stage2": {
                "tested": hw.tested,
                "passed": hw.passed,
                "latency_ms": hw.latency_ms,
                "throughput_gb_s": hw.throughput_gb_s,
                "speedup_vs_baseline": hw.speedup_vs_baseline,
            },
            "overall_passed": vr.overall_passed,
            "effective_speedup": vr.speedup,
        }
        (round_dir / "verification.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    v = VerifierAgent()
    kf = _PROJECT_DIR / "outputs" / "vector_add_fp16_N65536" / "round0" / "kernel.py"
    if kf.exists():
        r = v.verify(kf, kernel_fn_name="add_kernel")
        print(f"Stage1: passed={r.stage1_passed}")
        print(f"Stage2: tested={r.stage2_tested} (skipped on local)")
        print(f"Overall: passed={r.overall_passed}")
    else:
        print(f"kernel not found: {kf}")
    print("[Verifier] OK")


if __name__ == "__main__":
    _self_test()
