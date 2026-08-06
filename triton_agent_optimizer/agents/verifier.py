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

import json, re, sys
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

def _read_target_duration(prof_out: Path) -> tuple:
    """读 op_summary, 返回 (目标 kernel 非 aclnn 的 Task Duration(us) 之和, 行数).
    ★求和 = 一次执行的端到端 (多 kernel 如 MLP: fc1+bias_gelu+fc2 求和);
      KERNEL_LOOP=N 时行数=N×kernel数, 和=N×端到端 → 除 N 得单次.
      msprof 合并同名连续 kernel 也不影响总和 (时间不丢), 故求和法稳健.
      排除 aclnn 框架 kernel — 与 task.json total_ns (baseline) 口径一致."""
    import csv as _csv
    summaries = sorted(prof_out.rglob("op_summary*.csv"))
    if not summaries:
        return None, 0
    try:
        with open(summaries[0], encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        return None, 0
    total_us, count = None, 0
    for row in rows:
        dur = row.get("Task Duration(us)") or row.get("TaskDuration")
        op = row.get("Op Name") or row.get("OpName") or ""
        if dur and not op.lower().startswith("aclnn"):
            try:
                total_us = (total_us or 0) + float(dur)
                count += 1
            except ValueError:
                pass
    return total_us, count


def verify_end_to_end(kernel_op: Path, round_dir: Path,
                      baseline_ns: Optional[float] = None,
                      num_kernels: Optional[int] = None) -> dict:
    """v4 验证: warmup + 一次 msprof 内循环 KERNEL_LOOP 次取平均 (整文件).

    策略 (与 bench_910b3 同技术, 取代旧的 3 次独立 msprof — 每次 msprof 有 ~1-2min 启动开销):
      warmup = VERIFY_WARMUP (默认3): 先裸跑 kernel_op.py (KERNEL_LOOP 次) 预热 JIT/cache
      loop   = VERIFY_LOOP (默认30):  **一次 msprof** 内 kernel_op.py 内部循环 loop 次,
                                       读 op_summary 目标 kernel 耗时之和 → ÷loop = 单次端到端
      (msprof 合并同名 kernel 不影响总和; 求和法稳健)

    ★合理性告警 (防"漏记/循环丢失"导致静默错数):
      - 期望行数 ≈ loop × num_kernels; 若实际远少 → 可能 msprof 漏记 或 coder 弄丢了 KERNEL_LOOP 循环
      - 这种情况 sum/loop 会算错 → 打警告, 不静默

    返回:
      ok=True  → {"ok": True, "ns": 单次端到端ns, "speedup": baseline/ns}
      ok=False → {"ok": False, "error": 报错文本}  (回传 Coder 同轮改)
    """
    import os as _os
    import subprocess
    warmup = int(_os.environ.get("VERIFY_WARMUP", "3"))
    loop = int(_os.environ.get("VERIFY_LOOP", "30"))
    py = "python3"
    env = dict(_os.environ, KERNEL_LOOP=str(loop))   # kernel_op.py main() 内部循环 loop 次

    # warmup: 裸跑预热 (KERNEL_LOOP=loop, JIT 编译/冷cache 预热; 便宜)
    for i in range(warmup):
        subprocess.run([py, str(kernel_op)], capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace", timeout=1800, env=env)
    print(f"  [Verify] warmup x{warmup} (每轮内部 {loop} 次) done, 1 次 msprof 测 {loop} 次平均...")

    # ★正确性验证 (v4 曾只测性能不测数值): 单独跑一次 MATMUL_VERIFY=1, 结果必须 PASS.
    #   kernel_op.py main() 里 MATMUL_VERIFY=1 时对 torch 参考算 diff, 打印 "result check: PASS/CHECK".
    #   不 PASS(数值错/无校验) → 本轮 FAIL, 防"优化把结果改错还通过".
    chk_env = dict(_os.environ, KERNEL_LOOP="1", MATMUL_VERIFY="1")
    try:
        rc = subprocess.run([py, str(kernel_op)], capture_output=True, text=True,
                            encoding="utf-8", errors="backslashreplace", timeout=1800, env=chk_env)
    except Exception as e:
        return {"ok": False, "error": f"正确性校验运行失败: {e}"}
    _chk_out = (rc.stdout or "") + (rc.stderr or "")
    if "result check: PASS" not in _chk_out:
        return {"ok": False, "error": f"正确性未通过 (MATMUL_VERIFY 需输出 result check: PASS): {_chk_out.strip()[-400:]}"}
    print("    [Verify] ✅ 正确性 PASS (MATMUL_VERIFY)")

    # measure: 一次 msprof, app 内部循环 loop 次 → 和 ÷loop = 单次端到端
    msprof_out = round_dir / "msprof_0"
    msprof_out.mkdir(parents=True, exist_ok=True)
    cmd = ["msprof", f"--output={msprof_out}",
           f"--application={py} {kernel_op}", "--ai-core=on"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace", timeout=7200, env=env)
    except Exception as e:
        return {"ok": False, "error": f"msprof run failed: {e}"}
    total_us, n_rows = _read_target_duration(msprof_out)
    if total_us is None or n_rows < 1:
        tail = (r.stderr or "")[-800:] + (r.stdout or "")[-800:]
        return {"ok": False, "error": tail.strip() or "msprof 无目标 kernel"}
    # ★H1 自校正循环丢失: 从 kernel 源码判断 KERNEL_LOOP 循环是否还在.
    #   coder 弄丢/改坏 main() 的 for-range 循环时, 只跑 1 遍 → op_summary 行数 ≈ num_kernels,
    #   若仍 ÷loop 会虚高 30 倍. 源码里找不到循环 → 改用实测遍数 = 行数/每遍kernel数, 不再假设 ÷loop.
    loop_ok = True
    try:
        src = kernel_op.read_text(encoding="utf-8")
        loop_ok = bool(re.search(r"for\s+\w+\s+in\s+range\(\s*LOOP\s*\)\s*:", src))
    except Exception:
        loop_ok = True   # 读不到源码 → 按 loop 假设, 靠下方行数告警兜底
    if loop_ok:
        divisor = loop
    else:
        nk = num_kernels or 1
        divisor = max(1, int(round(n_rows / nk)))   # 实测有效遍数 (循环丢失时 = 1)
        print(f"    ⚠ 警告! kernel 源码无 KERNEL_LOOP 循环 (coder 弄丢?) "
              f"→ 用实测 {divisor} 遍平均, 不再 ÷{loop}")
    per_pass_us = total_us / divisor
    ns = per_pass_us * 1000
    # ★显式处理 baseline 缺失: 基准测量调用(baseline_ns=None)不算加速比 → None;
    #   轮次验证若 baseline 缺失 → None → 调度器告警, 不再静默当 1.0 (曾导致永远 1.000x)
    speedup = (baseline_ns / ns) if baseline_ns else None
    print(f"    msprof 记录 {n_rows} 行目标 kernel (期望 ~{loop}×{num_kernels or '?'}), "
          f"有效遍数={divisor}, 单次端到端={per_pass_us:.1f}us")
    # ★合理性告警: 行数远少于期望 → 循环丢失 或 msprof 严重漏记 (暴露, 不静默)
    if n_rows < loop:
        print(f"    ⚠ 警告! 目标 kernel 行数 {n_rows} < loop({loop}) "
              f"(coder 丢掉了 KERNEL_LOOP 循环? 或 msprof 漏记) → 单次端到端可能不准, 加速比存疑!")
    return {"ok": True, "ns": round(ns, 1),
            "speedup": round(speedup, 4) if speedup is not None else None,
            "loop": loop, "rows": n_rows, "duration_us": round(per_pass_us, 1)}
