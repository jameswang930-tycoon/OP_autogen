#!/usr/bin/env python3
"""
CPU Emulator 运行器 — Stage 1 验证。

═══════════════════════════════════════════════════════════════════════════════
  原理: Triton kernel 代码在 CPU 模拟器和真实 NPU 上完全一样—只换 import。

  真实 Triton:                          CPU Emulator:
    import triton                         from common import tl, xarray
    import triton.language as tl          from common import launch_kernel_1d, verify

  kernel 函数体不变 → 直接导入 → 模拟执行 → verify() 数值对比
═══════════════════════════════════════════════════════════════════════════════

  测试策略 (参照 emulators/test/<op>/__init__.py 的 test() 函数):
    - 多 shape 测试 (边界/典型/大/非整除)
    - 与 NumPy reference 对比
    - 验证 max_abs_error, max_rel_error
"""

from __future__ import annotations

import importlib.util
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable

_AGENT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _AGENT_DIR.parent
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
            return (f"PASS (max_abs={self.max_abs_error:.2e}, "
                    f"max_rel={self.max_rel_error:.2e})")
        return f"FAIL ({self.error_details[:200]})"


class EmulatorRunner:
    """CPU Emulator 验证器。

    从 round_N/kernel.py 导入 kernel, 用 emulators/common 验证正确性。

    Usage:
        runner = EmulatorRunner()
        result = runner.verify(
            kernel_path=Path("round1/kernel.py"),
            kernel_fn_name="add_kernel",
            emulate_fn=my_emulate_add,      # 包装函数 (展平→grid→reshape)
            reference_fn=my_reference_add,  # NumPy 参考实现
        )
    """

    DEFAULT_SHAPES = [1, 3, 7, 256, 512, 1024, 1025, 2049, 4096, 65536]
    DEFAULT_TOLERANCE = {
        "fp16": {"rtol": 1e-2, "atol": 1e-2},
        "fp32": {"rtol": 1e-5, "atol": 1e-5},
    }

    def __init__(self):
        self._emulator_loaded = False
        self.tl = None
        self.launch_kernel_1d = None
        self.verify_fn = None

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════════════════════

    def verify(
        self,
        kernel_path: Path,
        kernel_fn_name: str = "add_kernel",
        emulate_fn: Optional[Callable] = None,
        reference_fn: Optional[Callable] = None,
        test_shapes: Optional[List[int]] = None,
        test_dtypes: Optional[List[str]] = None,
    ) -> EmulatorResult:
        """验证 kernel 正确性。

        Args:
            kernel_path: round_N/kernel.py 路径
            kernel_fn_name: kernel 函数名
            emulate_fn: 包装函数 (kernel_fn, x, y, N) → output
            reference_fn: NumPy 参考 (x, y) → expected
            test_shapes: 测试的 N 值
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

        # Step 2: 导入 kernel
        try:
            kernel_fn = self._import_kernel(kernel_path, kernel_fn_name)
        except Exception as e:
            return EmulatorResult(passed=False,
                error_details=f"Cannot import kernel: {e}")

        # Step 3: 多 shape 测试
        failed = []
        max_abs = 0.0; max_rel = 0.0

        for N in shapes:
            try:
                x = np.random.randn(N).astype(np.float32)
                y = np.random.randn(N).astype(np.float32)

                if emulate_fn:
                    result = emulate_fn(kernel_fn, x, y, N)
                else:
                    result = self._emulate_vector_add(kernel_fn, x, y, N)

                if reference_fn:
                    expected = reference_fn(x, y)
                else:
                    expected = (x + y).astype(np.float32)

                abs_err = float(np.max(np.abs(result - expected)))
                rel_err = float(np.max(
                    np.abs((result - expected) / (np.abs(expected) + 1e-10))))
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

                if abs_err > 1e-3:
                    failed.append(f"N={N}(abs={abs_err:.2e})")
            except Exception as e:
                failed.append(f"N={N}: {str(e)[:80]}")

        if failed:
            return EmulatorResult(passed=False, failed_shapes=failed,
                max_abs_error=max_abs, max_rel_error=max_rel,
                error_details=f"Failed shapes: {failed}")

        return EmulatorResult(passed=True, max_abs_error=max_abs,
                               max_rel_error=max_rel)

    # ═══════════════════════════════════════════════════════════════════════════
    #  智能 emulation — 根据算子类型自动选择验证方式
    # ═══════════════════════════════════════════════════════════════════════════

    # 已知算子类型 → 默认 emulate + reference
    KNOWN_KERNEL_TYPES = {
        "add": {
            "emulate": "_emulate_element_wise",
            "reference": "_reference_add",
            "inputs": 2,  # x, y
        },
        "mul": {
            "emulate": "_emulate_element_wise",
            "reference": "_reference_mul",
            "inputs": 2,
        },
        "matmul": {
            "emulate": "_emulate_matmul",
            "reference": "_reference_matmul",
            "inputs": 2,  # A, B
        },
        "element_wise": {
            "emulate": "_emulate_element_wise",
            "reference": "_reference_add",
            "inputs": 2,
        },
        "unknown": {
            "emulate": "_emulate_generic",
            "reference": None,  # 无 reference → 只做语法检查
            "inputs": 2,
        },
    }

    def auto_verify(self, kernel_path: Path, kernel_fn_name: str,
                    op_type: str = "element_wise",
                    test_shapes=None) -> EmulatorResult:
        """自动选择合适的 emulate/reference 函数进行验证。

        Args:
            kernel_path: kernel.py 路径
            kernel_fn_name: kernel 函数名
            op_type: 算子类型 (add/mul/matmul/element_wise/reduction/unknown)
            test_shapes: 测试 shapes (默认使用 DEFAULT_SHAPES)
        """
        config = self.KNOWN_KERNEL_TYPES.get(op_type,
                                               self.KNOWN_KERNEL_TYPES["unknown"])
        emulate_fn_name = config["emulate"]
        reference_fn_name = config["reference"]

        # 获取 emulate 方法
        emulate_fn = getattr(self, emulate_fn_name, None)
        reference_fn = getattr(self, reference_fn_name, None) if reference_fn_name else None

        return self.verify(
            kernel_path=kernel_path,
            kernel_fn_name=kernel_fn_name,
            emulate_fn=emulate_fn,
            reference_fn=reference_fn,
            test_shapes=test_shapes,
        )

    def _emulate_element_wise(self, kernel_fn, x, y, N):
        """通用逐元素操作包装: 展平→grid→reshape。BLOCK_SIZE 自动适配 N。"""
        bs = 1
        while bs * 2 <= min(N, 4096):
            bs *= 2
        if bs < 1:
            bs = 1
        x_f = x.ravel().astype(np.float32)
        y_f = y.ravel().astype(np.float32)
        out_f = np.zeros(N, dtype=np.float32)
        grid = self.tl.cdiv(N, bs)
        self.launch_kernel_1d(kernel_fn, x_f, y_f, out_f, N, bs, grid_size=grid)
        return out_f.reshape(x.shape)

    def _emulate_matmul(self, kernel_fn, A, B, M, N, K):
        """矩阵乘法包装 (简化: M×K @ K×N)。"""
        A_f = A.ravel().astype(np.float32)
        B_f = B.ravel().astype(np.float32)
        out_f = np.zeros(M * N, dtype=np.float32)
        self.launch_kernel_1d(kernel_fn, A_f, B_f, out_f, M, N, K,
                               grid_size=M * N)
        return out_f.reshape(M, N)

    def _emulate_generic(self, kernel_fn, x, y, N):
        """通用包装 (不假设 kernel 输入格式，只做语法检查)。"""
        return self._emulate_element_wise(kernel_fn, x, y, N)

    @staticmethod
    def _reference_add(x, y):
        return (x + y).astype(np.float32)

    @staticmethod
    def _reference_mul(x, y):
        return (x * y).astype(np.float32)

    @staticmethod
    def _reference_matmul(A, B):
        return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════════════
    #  加载 emulators/common
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_emulator(self):
        path = _REPO_ROOT / "emulators" / "common" / "__init__.py"
        spec = importlib.util.spec_from_file_location("emulator_common", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tl = module.tl
        self.launch_kernel_1d = module.launch_kernel_1d
        self.launch_kernel_2d = getattr(module, "launch_kernel_2d", None)
        self.verify_fn = module.verify
        self._emulator_loaded = True

    # ═══════════════════════════════════════════════════════════════════════════
    #  导入 kernel
    # ═══════════════════════════════════════════════════════════════════════════

    def _import_kernel(self, kernel_path: Path, fn_name: str):
        """从 round_N/kernel.py 动态导入 kernel 函数。

        kernel 中的 triton import 被移除, emulator 对象注入 namespace。
        Triton 指针风格 (ptr + offs) 被转换为 emulator 风格 (ptr, offs)。
        """
        code = kernel_path.read_text(encoding="utf-8")

        # 移除 triton 相关行和 @triton.jit 装饰器
        lines = []
        for line in code.split("\n"):
            s = line.strip()
            if s.startswith("import triton") or s.startswith("from triton"):
                continue
            if s.startswith("@triton.jit") or s.startswith("@triton.autotune"):
                continue
            lines.append(line)
        clean_code = "\n".join(lines)

        # 转换调用约定: Triton ptr+offset → emulator ptr,offset
        # tl.load(x_ptr + offs, mask=mask) → tl.load(x_ptr, offs, mask=mask)
        import re
        clean_code = re.sub(
            r'tl\.load\((\w+)\s*\+\s*(\w+)\s*,',
            r'tl.load(\1, \2,',
            clean_code)
        # tl.store(out_ptr + offs, val, mask=mask) → tl.store(out_ptr, offs, val, mask=mask)
        clean_code = re.sub(
            r'tl\.store\((\w+)\s*\+\s*(\w+)\s*,',
            r'tl.store(\1, \2,',
            clean_code)

        # 注入 emulator 对象到 exec namespace
        namespace: dict = {
            "np": np,
            "tl": self.tl,
            "xarray": getattr(self, "xarray", None),
        }
        exec(compile(clean_code, str(kernel_path), "exec"), namespace)

        fn = namespace.get(fn_name)
        if fn is None:
            available = [k for k in namespace
                         if callable(namespace[k]) and not k.startswith("_")]
            raise AttributeError(
                f"'{fn_name}' not found in {kernel_path.name}. "
                f"Found: {available}")
        return fn


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    runner = EmulatorRunner()

    kf = _AGENT_DIR / "outputs" / "vector_add_fp16_N65536" / "round0" / "kernel.py"
    if kf.exists():
        r = runner.verify(kf, kernel_fn_name="add_kernel",
                          test_shapes=[256, 1024, 1025])
        print(f"round0 kernel: {r.summary()}")
    else:
        print(f"kernel not found: {kf}")

    # 语法错误测试
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".py"))
    tmp.write_text("def broken(")
    r2 = runner.verify(tmp)
    print(f"broken: {r2.summary()}")
    tmp.unlink()

    print("[EmulatorRunner] OK")


if __name__ == "__main__":
    _self_test()
