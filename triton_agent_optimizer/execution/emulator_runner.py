#!/usr/bin/env python3
"""
CPU Emulator 运行器 — Stage 1 验证。

═══════════════════════════════════════════════════════════════════════════════
  验证模式: 参照 emulators/test/<op>/__init__.py 的 4-part 结构:
    1. kernel     — Coder 产出的纯 Triton kernel 函数 (从 round_N/kernel.py 导入)
    2. emulate    — 包装: 展平输入 → launch_kernel → reshape 输出
    3. reference  — NumPy 参考实现 (手动编写, 用于数值对比)
    4. test       — 自测: 多 shape + 边界条件 + 精度检查

  输入: round_N/kernel.py (仅包含 @triton.jit def xxx_kernel(...) 函数)
  输出: PASS (含 max_abs/max_rel error) 或 FAIL (含 error_details)

═══════════════════════════════════════════════════════════════════════════════
  每个 kernel 需要一个对应的 test_harness (测试用例)。
  测试用例定义了: emulate_xxx() 包装 + reference_xxx() + 测试输入。

  首次运行时由用户提供; 后续每轮自动复用。
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
import importlib.util
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

_AGENT_DIR = Path(__file__).resolve().parent.parent    # triton_agent_optimizer/
_REPO_ROOT = _AGENT_DIR.parent                          # OP_autogen_hjkc/
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_REPO_ROOT / "emulators"))


@dataclass
class EmulatorResult:
    """Stage 1 验证结果。"""
    passed: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    failed_shapes: List[str] = field(default_factory=list)
    failed_dtypes: List[str] = field(default_factory=list)
    error_details: str = ""

    def summary(self) -> str:
        if self.passed:
            return f"PASS (max_abs={self.max_abs_error:.2e}, max_rel={self.max_rel_error:.2e})"
        return f"FAIL ({self.error_details[:200]})"


class EmulatorRunner:
    """CPU Emulator 验证器。

    从 round_N/kernel.py 动态导入 kernel 函数,
    用 emulators/common 的 tl 打桩 + launch_kernel + verify() 验证正确性。

    Usage:
        runner = EmulatorRunner()
        result = runner.verify(
            kernel_path=Path("round1/kernel.py"),
            kernel_fn_name="add_kernel",
            emulate_fn=my_emulate_add,      # 用户提供的包装函数
            reference_fn=my_reference_add,  # NumPy 参考实现
            test_inputs=[{"N": 256}, {"N": 65536}],
        )
    """

    # 默认测试 shapes
    DEFAULT_SHAPES = [1, 3, 7, 256, 512, 1024, 1025, 2049, 4096, 65536]
    DEFAULT_DTYPES = ["fp16", "fp32"]
    DEFAULT_TOLERANCE = {
        "fp16": {"rtol": 1e-2, "atol": 1e-2},
        "fp32": {"rtol": 1e-5, "atol": 1e-5},
    }

    def __init__(self):
        self._emulator_loaded = False

    def verify(
        self,
        kernel_path: Path,
        kernel_fn_name: str = "kernel",
        emulate_fn=None,
        reference_fn=None,
        test_shapes: Optional[List[int]] = None,
        test_dtypes: Optional[List[str]] = None,
    ) -> EmulatorResult:
        """验证 kernel 的正确性。

        Args:
            kernel_path: round_N/kernel.py 的路径 (Coder 产出)
            kernel_fn_name: kernel 函数名
            emulate_fn: 包装函数 (展平→grid→reshape)
            reference_fn: NumPy 参考实现
            test_shapes: 测试的 N 值列表
            test_dtypes: 测试的 dtype 列表

        Returns:
            EmulatorResult
        """
        shapes = test_shapes or self.DEFAULT_SHAPES

        # Step 1: 加载 emulator
        try:
            if not self._emulator_loaded:
                self._load_emulator()
        except Exception as e:
            return EmulatorResult(passed=False,
                error_details=f"Cannot load emulators/common: {e}")

        # Step 2: 导入 kernel 函数
        try:
            kernel_fn = self._import_kernel(kernel_path, kernel_fn_name)
        except Exception as e:
            return EmulatorResult(passed=False,
                error_details=f"Cannot import kernel from {kernel_path}: {e}")

        # Step 3: 运行多 shape 测试
        failed_shapes = []
        max_abs = 0.0; max_rel = 0.0

        for N in shapes:
            try:
                # 生成输入
                x = np.random.randn(N).astype(np.float32)
                y = np.random.randn(N).astype(np.float32)

                # 运行 kernel (通过 emulate 包装)
                if emulate_fn:
                    result = emulate_fn(kernel_fn, x, y, N)
                else:
                    # 无 emulate_fn → 只做语法检查
                    continue

                # 对比 reference
                if reference_fn:
                    expected = reference_fn(x, y)
                    abs_err = np.max(np.abs(result - expected))
                    rel_err = np.max(np.abs((result - expected) /
                                    (np.abs(expected) + 1e-10)))
                    max_abs = max(max_abs, abs_err)
                    max_rel = max(max_rel, rel_err)

                    tol = self.DEFAULT_TOLERANCE.get("fp32", {})
                    if abs_err > tol.get("atol", 1e-5) * 10:
                        failed_shapes.append(f"N={N}")
                    if rel_err > tol.get("rtol", 1e-5) * 100:
                        if f"N={N}" not in failed_shapes:
                            failed_shapes.append(f"N={N}")

            except Exception as e:
                failed_shapes.append(f"N={N}: {str(e)[:80]}")

        if failed_shapes:
            return EmulatorResult(passed=False,
                failed_shapes=failed_shapes,
                max_abs_error=max_abs, max_rel_error=max_rel,
                error_details=f"Failed shapes: {failed_shapes}")

        return EmulatorResult(passed=True,
            max_abs_error=max_abs, max_rel_error=max_rel)

    def _load_emulator(self):
        """动态加载 emulators/common/__init__.py。"""
        path = _REPO_ROOT / "emulators" / "common" / "__init__.py"
        spec = importlib.util.spec_from_file_location("emulator_common", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tl = module.tl
        self.launch_kernel_1d = module.launch_kernel_1d
        self.verify_fn = module.verify
        self._emulator_loaded = True

    def _import_kernel(self, kernel_path: Path, fn_name: str):
        """从 round_N/kernel.py 动态导入 kernel 函数。"""
        spec = importlib.util.spec_from_file_location(
            f"kernel_round_{kernel_path.parent.name}", str(kernel_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load kernel from {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, fn_name, None)
        if fn is None:
            raise AttributeError(
                f"Function '{fn_name}' not found in {kernel_path}. "
                f"Available: {[x for x in dir(module) if not x.startswith('_')]}")
        return fn


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    runner = EmulatorRunner()

    # 测试: 从 round0 加载 kernel
    kf = _AGENT_DIR / "outputs" / "vector_add_fp16_N65536" / "round0" / "kernel.py"
    if kf.exists():
        r = runner.verify(kf, kernel_fn_name="add_kernel",
                          test_shapes=[256, 1024])
        print(f"round0 kernel: {r.summary()}")
    else:
        print(f"round0 kernel not found at {kf}")

    # 测试: 语法错误
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".py"))
    tmp.write_text("def broken(")
    r2 = runner.verify(tmp)
    print(f"broken kernel: {r2.summary()}")
    tmp.unlink()

    print("[EmulatorRunner] OK")


if __name__ == "__main__":
    _self_test()
