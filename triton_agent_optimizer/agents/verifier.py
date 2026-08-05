#!/usr/bin/env python3
"""
Verifier Agent v2.0 — 两阶段验证 (脚本, 非 LLM)。

═══════════════════════════════════════════════════════════════════════════════
  数据流
═══════════════════════════════════════════════════════════════════════════════

  Coder → round_N/kernel.py (Triton kernel 源码)
    │
    ▼
  Verifier.verify(kernel_path, round_dir, op_type, baseline_ns)
    │
    ├─ Stage 1: CPU Emulator (正确性验证, 必跑)
    │   emulator_runner.auto_verify(kernel_path) → EmulatorResult
    │   → PASS: 继续 Stage 2
    │   → FAIL: 返回 error_details → Orchestrator → Coder 重试 (最多3次)
    │
    ├─ Stage 2: 性能验证 (根据环境自动选择)
    │   ┌─ msprof_mode="hardware"  → 真机 benchmark (需 910B3 NPU)
    │   ├─ msprof_mode="simulator" → msprof op simulator (CPU仿真, 无需NPU)
    │   └─ msprof_mode="none"      → 跳过, speedup=1.0
    │
    ├─ 保存 verification.json 到 round_N/
    │
    └─ 返回 VerifyResult 给 Orchestrator
         Orchestrator 读 speedup → KEEP/REVERT
         Orchestrator 读 error_details → Coder 重试

  环境切换:
    - 自动: config.py 检测 npu-smi / msprof 工具
    - 手动: export TRITON_AGENT_MSPROF_MODE=simulator|hardware
═══════════════════════════════════════════════════════════════════════════════
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
    """两阶段验证。V2.0: 支持 msprof simulator/真机灵活切换。

    Stage 1: CPU Emulator — 正确性 (必跑, 无NPU也能跑)
    Stage 2: 性能验证 — 根据 msprof_mode 自动选择
      - "simulator": msprof op simulator (CPU仿真, cycle-accurate)
      - "hardware": 真机 msprof benchmark (需 910B3 NPU)
      - "none": 跳过性能验证 (speedup=1.0)
    """

    def __init__(self, skip_hardware_on_local: bool = True, msprof_mode: str = ""):
        self.skip_hardware = skip_hardware_on_local

        from execution.emulator_runner import EmulatorRunner
        from execution.hardware_runner import HardwareRunner

        self.emulator = EmulatorRunner()
        self.hardware = HardwareRunner()

        # msprof mode: "simulator" | "hardware" | "none"
        if msprof_mode:
            self.msprof_mode = msprof_mode
        else:
            try:
                from config import config
                self.msprof_mode = config.msprof_mode
            except Exception:
                self.msprof_mode = "none"

        # 是否需要在 Stage 2 运行 msprof
        self._run_msprof_stage2 = self.msprof_mode in ("simulator", "hardware")

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════════════════════

    def verify(
        self,
        kernel_path: Path,
        round_dir: Optional[Path] = None,
        kernel_fn_name: str = "add_kernel",
        op_type: str = "element_wise",
        baseline_latency_ns: float = 0.0,
    ) -> VerifyResult:
        """完整两阶段验证。

        Args:
            kernel_path: round_N/kernel.py 文件路径 (Coder 已写入)
            round_dir: 本轮输出目录
            kernel_fn_name: kernel 函数名
            op_type: 算子类型
            baseline_latency_ns: round0 基准延迟 (ns)

        Returns:
            VerifyResult → Orchestrator 用于决定 KEEP/REVERT
        """
        kernel_path = Path(kernel_path)

        # ── Stage 1: CPU Emulator ──
        emu = self._run_stage1(kernel_path, kernel_fn_name, op_type)
        if not emu.passed:
            return VerifyResult(
                stage1_passed=False,
                stage1_error_details=emu.error_details,
                overall_passed=False,
            )

        # ── Stage 2: 性能验证 ──
        speedup = 1.0
        hw_tested = False
        stage2_details = ""

        if self._run_msprof_stage2:
            if self.msprof_mode == "simulator":
                speedup, hw_tested, stage2_details = self._run_stage2_simulator(
                    kernel_path, round_dir, baseline_latency_ns)
            elif self.msprof_mode == "hardware":
                hw = self._run_stage2_hardware(kernel_path, round_dir, baseline_latency_ns)
                speedup = hw.speedup_vs_baseline
                hw_tested = hw.tested
        else:
            stage2_details = "msprof mode=none (no msprof/npu-smi found)"

        result = VerifyResult(
            stage1_passed=True,
            stage1_max_abs_error=emu.max_abs_error,
            stage2_tested=hw_tested,
            stage2_passed=(speedup >= 1.0),
            stage2_actual_speedup=speedup if hw_tested else None,
            overall_passed=True,
        )

        # ── 保存 verification.json ──
        if round_dir:
            self._save_result_v2(result, round_dir, emu, hw_tested, speedup, stage2_details)

        return result

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 1: CPU Emulator
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage1(self, kernel_path: Path, fn_name: str,
                    op_type: str = "element_wise") -> EmulatorStageResult:
        """从 round_N/kernel.py 导入 kernel, 用 emulators/common 验证正确性。"""
        r = self.emulator.auto_verify(kernel_path, fn_name, op_type=op_type)
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
    #  Stage 2: msprof simulator (CPU 仿真, 无需 NPU)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage2_simulator(
        self, kernel_path: Path, round_dir: Optional[Path],
        baseline_ns: float,
    ) -> tuple:
        """使用 msprof op simulator 采集性能数据。

        前置条件: CANN 9.0+ 环境 + bisheng 编译器 + AscendC kernel .asc 文件
        注意: 这一步需要用户预先在 WSL/Linux 中使用 CMake 编译 AscendC kernel。
              对于 Triton kernel, 需要先转成 AscendC 或 HIVM MLIR。
        """
        if not round_dir:
            return 1.0, False, "no round_dir"

        # 检查是否有预编译的 msprof 可执行文件
        msprof_app = round_dir / "msprof_app"
        if not msprof_app.exists():
            # 尝试从 msprof_simulator_test 目录找
            test_dir = _PROJECT_DIR.parent / "msprof_simulator_test" / "build"
            candidates = list(test_dir.glob("demo*")) if test_dir.exists() else []
            if candidates:
                msprof_app = candidates[0]

        if not msprof_app.exists():
            return 1.0, False, "msprof: no executable found (compile AscendC kernel first)"

        # 获取 msprof 工具
        import shutil
        msprof_bin = shutil.which("msprof")
        if not msprof_bin:
            return 1.0, False, "msprof: msprof binary not found in PATH"

        # 运行 msprof op simulator
        import subprocess
        msprof_out = round_dir / "msprof"
        msprof_out.mkdir(parents=True, exist_ok=True)

        try:
            r = subprocess.run(
                [msprof_bin, "op", "simulator",
                 "--soc-version=Ascend910B3",
                 f"--output={msprof_out}",
                 f"--timeout=5",
                 str(msprof_app)],
                capture_output=True, text=True, timeout=300,
            )

            # 查找生成的 OPPROF 目录
            opprof_dirs = sorted(msprof_out.glob("OPPROF_*"),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            if opprof_dirs:
                # 解析 trace 数据获取 timing
                from analyzers.msprof_analyzer import MsprofAnalyzer
                ma = MsprofAnalyzer()
                report = ma.parse_existing(opprof_dirs[0])
                if report.total_ns > 0 and baseline_ns > 0:
                    speedup = baseline_ns / report.total_ns
                    return speedup, True, f"simulator: {report.total_ns:.1f}ns, {speedup:.3f}x"
                return 1.0, True, f"simulator: {report.total_ns:.1f}ns (no baseline)"
        except Exception as e:
            return 1.0, False, f"msprof simulator failed: {e}"

        return 1.0, False, "msprof: OPPROF not generated"

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 2: 真机 msprof (需要 910B3 NPU 硬件)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_stage2_hardware(
        self, kernel_path: Path, round_dir: Optional[Path],
        baseline_ns: float,
    ) -> HardwareStageResult:
        """在 910B3 上: 编译 → 提取 HIVMIR → benchmark。

        仅当有 NPU 硬件 + CANN 环境时可用。
        msprof_mode="hardware" 时自动调用。
        """
        if not self.hardware.available:
            return HardwareStageResult(tested=False)
        if self.skip_hardware:
            return HardwareStageResult(tested=False)

        try:
            from execution.compiler import CompilerInterface
            compiler = CompilerInterface()
            if not compiler.available:
                return HardwareStageResult(tested=False)

            code = kernel_path.read_text(encoding="utf-8")
            rd = round_dir or kernel_path.parent

            # 编译 + HIVMIR 提取
            compile_r = compiler.compile(code, rd / "compiler_output")
            if not compile_r.success:
                return HardwareStageResult(tested=True, passed=False)

            # 将 HIVMIR 复制到 hivmir/ 目录
            hivmir_dir = rd / "hivmir"
            hivmir_dir.mkdir(exist_ok=True)
            hivmir_compiler_dir = hivmir_dir / "compiler_output"
            hivmir_compiler_dir.mkdir(exist_ok=True)
            if compile_r.hivmir_path:
                import shutil
                shutil.copy2(compile_r.hivmir_path,
                             hivmir_compiler_dir / "hivmir_output.mlir")

            # benchmark
            hw_r = self.hardware.benchmark(
                Path(compile_r.binary_path), baseline_ns / 1e6)  # ns → ms
            return HardwareStageResult(
                tested=True, passed=hw_r.success,
                latency_ms=hw_r.latency_ms,
                throughput_gb_s=hw_r.throughput_gb_s,
                speedup_vs_baseline=hw_r.speedup_vs_baseline,
            )
        except Exception:
            return HardwareStageResult(tested=True, passed=False)

    # ═══════════════════════════════════════════════════════════════════════════
    #  保存结果 (v2)
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_result_v2(self, vr: VerifyResult, round_dir: Path,
                        emu: EmulatorStageResult,
                        hw_tested: bool, speedup: float, stage2_details: str):
        """保存 verification.json 到 round_N/ (v2 格式)。"""
        data = {
            "stage1": {
                "passed": emu.passed,
                "max_abs_error": emu.max_abs_error,
                "max_rel_error": emu.max_rel_error,
                "shapes_tested": emu.shapes_tested,
                "shapes_failed": emu.shapes_failed,
            },
            "stage2": {
                "tested": hw_tested,
                "mode": self.msprof_mode,
                "speedup": speedup,
                "details": stage2_details,
            },
            "overall_passed": vr.overall_passed,
            "effective_speedup": speedup if hw_tested else 1.0,
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


# ═══════════════════════════════════════════════════════════════════════════════
#  v4: 只跑 msprof 端到端 (整文件) → 端到端耗时 → 加速比
# ═══════════════════════════════════════════════════════════════════════════════

def _read_target_duration(prof_out: Path) -> Optional[float]:
    """读 op_summary, 目标 kernel (非 aclnn) 的 Task Duration(us) 之和 = 端到端.
    ★多 kernel (如 MLP: fc1+bias_gelu+fc2) 求和才是总耗时; 单 kernel 即本身.
      排除 aclnn 框架 kernel — 与 task.json total_ns (baseline) 口径一致."""
    import csv as _csv
    summaries = sorted(prof_out.rglob("op_summary*.csv"))
    if not summaries:
        return None
    try:
        with open(summaries[0], encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        return None
    total_us = None
    for row in rows:
        dur = row.get("Task Duration(us)") or row.get("TaskDuration")
        op = row.get("Op Name") or row.get("OpName") or ""
        if dur and not op.lower().startswith("aclnn"):
            try:
                total_us = (total_us or 0) + float(dur)
            except ValueError:
                pass
    return total_us


def verify_end_to_end(kernel_op: Path, round_dir: Path,
                      baseline_ns: Optional[float] = None) -> dict:
    """v4 验证: warmup + 多轮 msprof 端到端取平均 (整文件), 不提取字段、不跑 msprof op。

    策略:
      warmup = VERIFY_WARMUP (默认1): 先裸跑 kernel_op.py N 次 (JIT 编译/冷cache 预热)
      runs   = VERIFY_RUNS (默认3):   跑 N 次 msprof, 每次读目标 kernel Task Duration(us),
                                       取各轮 max 的平均 (去抖动)

    返回:
      ok=True  → {"ok": True, "ns": 平均端到端ns, "speedup": baseline/ns}
      ok=False → {"ok": False, "error": 报错文本}  (回传 Coder 同轮改)
    """
    import os as _os
    import subprocess
    warmup = int(_os.environ.get("VERIFY_WARMUP", "3"))   # 裸跑热身 (JIT/冷cache), 便宜
    runs = int(_os.environ.get("VERIFY_RUNS", "3"))       # msprof 轮数 (每次~1-2min, 别调太大)
    py = "python3"

    # warmup: 裸跑预热 (不 profile)
    for i in range(warmup):
        subprocess.run([py, str(kernel_op)], capture_output=True, text=True, timeout=1800)
    print(f"  [Verify] warmup x{warmup} done, 测 {runs} 轮 msprof...")

    # measure: runs 次 msprof, 每次独立输出目录
    durations_us = []
    for i in range(runs):
        msprof_out = round_dir / f"msprof_run{i}"
        msprof_out.mkdir(parents=True, exist_ok=True)
        cmd = ["msprof", f"--output={msprof_out}",
               f"--application={py} {kernel_op}", "--ai-core=on"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except Exception as e:
            return {"ok": False, "error": f"msprof run failed: {e}"}
        d = _read_target_duration(msprof_out)
        if d is None:
            tail = (r.stderr or "")[-800:] + (r.stdout or "")[-800:]
            return {"ok": False, "error": tail.strip() or f"msprof_run{i} 无 op_summary/目标kernel"}
        durations_us.append(d)
        print(f"    run{i}: {d:.1f}us")

    ns = sum(durations_us) / len(durations_us) * 1000   # 平均 (us→ns)
    speedup = (baseline_ns / ns) if baseline_ns else 1.0
    return {"ok": True, "ns": round(ns, 1), "speedup": round(speedup, 4),
            "runs": runs, "durations_us": durations_us}
