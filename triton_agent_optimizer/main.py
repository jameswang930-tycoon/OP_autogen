#!/usr/bin/env python3
"""
Triton Agent Optimizer — 主入口。

Usage:
  python main.py <kernel.py>                     # 单 kernel 优化
  python main.py <kernel.py> --target 2.0        # 目标加速比 2x
  python main.py <kernel.py> --max-rounds 100    # 最大 100 轮
  python main.py <kernel_dir/>                    # 批量优化目录下所有 kernel
"""

from __future__ import annotations
import ast
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def detect_kernels(kernel_path: Path) -> List[Tuple[str, str]]:
    """从 Triton kernel 文件中自动检测所有 kernel 函数。

    Returns:
        [(函数名, 算子类型), ...]
        算子类型: "element_wise" | "reduction" | "matmul" | "attention" | "unknown"
    """
    code = kernel_path.read_text(encoding="utf-8")
    tree = ast.parse(code)

    kernels = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 检查是否有 @triton.jit 装饰器
            has_triton_jit = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if (isinstance(decorator.value, ast.Name)
                            and decorator.value.id == "triton"
                            and decorator.attr == "jit"):
                        has_triton_jit = True
                        break
                elif isinstance(decorator, ast.Call):
                    if (isinstance(decorator.func, ast.Attribute)
                            and isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == "triton"
                            and decorator.func.attr in ("jit", "autotune")):
                        has_triton_jit = True
                        break

            if not has_triton_jit:
                continue

            # 分析函数体判断算子类型
            op_type = _classify_kernel_type(node)
            kernels.append((node.name, op_type))

    return kernels


def _classify_kernel_type(func_node: ast.FunctionDef) -> str:
    """通过 AST 分析推断算子类型。"""
    source = ast.unparse(func_node)

    # MatMul: tl.dot 调用
    if "tl.dot" in source:
        return "matmul"

    # Reduction: tl.sum / tl.max / tl.min over axis
    if any(kw in source for kw in ["tl.sum", "tl.max", "tl.min"]):
        if any(kw in source for kw in ["softmax", "norm", "rms", "layer"]):
            return "reduction"

    # Attention: Q/K/V pattern
    if all(kw in source.lower() for kw in ["q_", "k_", "v_"]):
        return "attention"

    # Element-wise (default): tl.load → compute → tl.store, no reduction
    if "tl.load" in source and "tl.store" in source:
        return "element_wise"

    return "unknown"


def get_kernel_name(kernel_path: Path) -> str:
    """从文件路径推断 kernel 名称 (用于 outputs/ 目录)。"""
    stem = kernel_path.stem
    # 去掉 _kernel 后缀
    if stem.endswith("_kernel"):
        stem = stem[:-7]
    return stem


def init_output_dir(kernel_path: Path, kernel_name: str) -> Path:
    """初始化输出目录 (round0 + tier 文件夹)。"""
    outputs = _PROJECT / "outputs"
    kernel_dir = outputs / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)

    round0 = kernel_dir / "round0"
    round0.mkdir(exist_ok=True)

    # 拷贝 kernel
    shutil.copy2(kernel_path, round0 / "kernel.py")

    # 创建 tier 文件夹
    tiers = [
        "01_algorithmic_structure",
        "02_operator_fusion",
        "03_tiling_block_config",
        "04_memory_access",
        "05_compute_occupancy",
        "06_910b3_architecture",
    ]
    for t in tiers:
        (kernel_dir / t).mkdir(exist_ok=True)

    return kernel_dir


def run_optimization(
    kernel_path: Path,
    target_speedup: float = 1.5,
    max_rounds: int = 200,
):
    """运行完整优化流程。"""
    # 1. 检测 kernel
    kernels = detect_kernels(kernel_path)
    if not kernels:
        print(f"[ERROR] No @triton.jit kernel found in {kernel_path}")
        print(f"  Expected: @triton.jit decorated function")
        return 1

    kernel_fn_name, op_type = kernels[0]
    kernel_name = get_kernel_name(kernel_path)

    print(f"Kernel: {kernel_name}")
    print(f"  Function: {kernel_fn_name}()")
    print(f"  Type: {op_type}")
    if len(kernels) > 1:
        print(f"  Also found: {[k[0] for k in kernels[1:]]}")
        print(f"  (optimizing primary kernel '{kernel_fn_name}' — "
              f"use multi-kernel mode for fusion)")

    # 2. 初始化输出目录
    kernel_dir = init_output_dir(kernel_path, kernel_name)
    print(f"Output: {kernel_dir}")

    # 3. 运行优化
    from agents.orchestrator import Orchestrator
    orch = Orchestrator(
        kernel_path=kernel_dir / "round0" / "kernel.py",
        kernel_name=kernel_name,
        target_speedup=target_speedup,
        max_rounds=max_rounds,
    )
    # 注入检测到的信息
    orch._kernel_fn_name = kernel_fn_name
    orch._op_type = op_type

    result = orch.run()

    print(f"\nFinal: {result['best_speedup']:.2f}x in {result['rounds']} rounds")
    return 0


def main():
    import argparse

    p = argparse.ArgumentParser(description="Triton Agent Optimizer")
    p.add_argument("input", type=str, help="Triton kernel file (.py) or directory")
    p.add_argument("--target", type=float, default=1.5, help="Target speedup")
    p.add_argument("--max-rounds", type=int, default=200, help="Max optimization rounds")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Not found: {input_path}")
        return 1

    if input_path.is_dir():
        # 批量: 目录下所有 .py 文件
        py_files = sorted(input_path.glob("*.py"))
        if not py_files:
            print(f"[ERROR] No .py files in {input_path}")
            return 1
        for pf in py_files:
            print(f"\n{'='*60}")
            run_optimization(pf, args.target, args.max_rounds)
    else:
        return run_optimization(input_path, args.target, args.max_rounds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
