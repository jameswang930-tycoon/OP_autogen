#!/usr/bin/env python3
"""
Triton Agent Optimizer v3.0 — 主入口

完整执行链路:
  1. Triton .py → triton 2.3.1 → TTIR MLIR
  2. TTIR → ttir_to_hivm.py → HIVM MLIR
  3. HIVM → hivmir_analyzer → 11 语义字段
  4. msprof trace → msprof_analyzer → 14 timing 字段
  5. dsl_merger → 29 字段全填充
  6. Orchestrator → 优化循环 (Planner→Coder→Verifier→RecordManager)

Usage:
  # 完整流程 (需要 triton 2.3.1 + CANN 9.0)
  python main.py input/kernel.py --max-rounds 5

  # 跳过 Triton→HIVM (已有 .mlir 文件)
  python main.py input/kernel.py --skip-triton --hivm-mlir path/to/mlir

  # 使用已有 msprof 数据
  python main.py input/kernel.py --msprof-dir path/to/OPPROF_xxx

  # 只跑分析, 不优化
  python main.py input/kernel.py --analyze-only
"""

from __future__ import annotations
import ast, json, os, shutil, sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# 加载 .env (确保 ANTHROPIC_API_KEY 在 os.environ 中)
from config import config  # noqa: E402 — triggers _load_dotenv() at import


def detect_kernels(kernel_path: Path) -> List[Tuple[str, str]]:
    """从 Triton kernel 文件中自动检测所有 kernel 函数。"""
    code = kernel_path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    kernels = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            has_triton_jit = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if (isinstance(decorator.value, ast.Name)
                            and decorator.value.id == "triton"
                            and decorator.attr == "jit"):
                        has_triton_jit = True
                        break
            if not has_triton_jit:
                continue
            op_type = _classify_kernel_type(node)
            kernels.append((node.name, op_type))
    return kernels


def _classify_kernel_type(func_node: ast.FunctionDef) -> str:
    source = ast.unparse(func_node)
    if "tl.dot" in source:
        return "matmul"
    if any(kw in source for kw in ["tl.sum", "tl.max", "tl.min"]):
        if any(kw in source for kw in ["softmax", "norm", "rms", "layer"]):
            return "reduction"
    if "tl.load" in source and "tl.store" in source:
        return "element_wise"
    return "unknown"


def get_kernel_name(kernel_path: Path) -> str:
    if "input" in kernel_path.parts:
        idx = list(kernel_path.parts).index("input")
        if idx + 1 < len(kernel_path.parts):
            return kernel_path.parts[idx + 1]
    stem = kernel_path.stem
    if stem.endswith("_kernel"):
        stem = stem[:-7]
    return stem


def init_output_dir(kernel_path: Path, kernel_name: str) -> Path:
    outputs = _PROJECT / "outputs"
    kernel_dir = outputs / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    round0 = kernel_dir / "round0"
    round0.mkdir(exist_ok=True)
    shutil.copy2(kernel_path, round0 / kernel_path.name)
    tiers = [
        "01_algorithmic_structure", "02_operator_fusion",
        "03_tiling_block_config", "04_memory_access",
        "05_compute_occupancy", "06_910b3_architecture",
    ]
    for t in tiers:
        (kernel_dir / t).mkdir(exist_ok=True)
    (kernel_dir / "final_output").mkdir(exist_ok=True)
    return kernel_dir


def run_triton_to_hivm(kernel_path: Path, round_dir: Path,
                       kernel_fn_name: str) -> Optional[str]:
    """Triton .py → TTIR → HIVM MLIR。

    需要: triton 2.3.1 + LD_PRELOAD stub (WSL2)
    """
    try:
        from unittest.mock import MagicMock
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        import triton.runtime.driver as _drv
        _drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
        import triton.compiler.compiler as _comp
        _comp.CompiledKernel = MagicMock()
        import triton
        from triton.compiler import ASTSource
        from types import SimpleNamespace

        # 加载 kernel
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            kernel_path.stem, str(kernel_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        kernel_fn = getattr(mod, kernel_fn_name)
        if not hasattr(kernel_fn, "fn"):
            print(f"  [WARN] {kernel_fn_name} is not a @triton.jit function")
            return None

        # 生成 TTIR — 智能推断签名
        n_args = len(kernel_fn.arg_names)
        sig = {}
        consts = {}
        ptr_count = 0
        for i, name in enumerate(kernel_fn.arg_names):
            name_upper = name.upper()
            if "BLOCK_SIZE" in name_upper or name.startswith("BLOCK"):
                consts[name] = 256
            elif "EPS" in name_upper or "eps" == name:
                consts[name] = 1e-5
            elif name_upper in ("N", "M", "K", "DIM", "LEN", "SIZE", "STRIDE",
                               "HIDDEN_DIM", "NUM_ELEMENTS"):
                sig[i] = "i32"
            elif name_upper.endswith("_PTR") or name_upper.endswith("_ptr") or ptr_count < n_args - 2:
                sig[i] = "*fp32"
                ptr_count += 1
            else:
                sig[i] = "i32"  # 默认标量

        # 确保至少一个 i32 参数或者全是 ptr
        if not any(v == "i32" for v in sig.values()) and len(consts) == 0:
            # 最后一个非 const 参数设为 i32
            last_idx = n_args - 1
            while last_idx >= 0 and last_idx in consts:
                last_idx -= 1
            if last_idx >= 0:
                sig[last_idx] = "i32"

        src = ASTSource(fn=kernel_fn, signature=sig, constants=consts)
        opts = SimpleNamespace(num_warps=4, num_stages=1, debug=False)
        ttir_module = src.make_ir(opts)
        ttir_text = str(ttir_module)
        print(f"  TTIR: {len(ttir_text)} chars")

        # TTIR → HIVM
        from analyzers.ttir_to_hivm import ttir_to_hivm
        hivm_text, hivm_ops = ttir_to_hivm(ttir_text, kernel_fn_name)
        print(f"  HIVM: {len(hivm_ops)} ops")

        # 保存
        hivm_dir = round_dir / "hivmir" / "compiler_output"
        hivm_dir.mkdir(parents=True, exist_ok=True)
        (hivm_dir / "hivmir_output.mlir").write_text(hivm_text, encoding="utf-8")

        # 保存结构化 ops
        (hivm_dir.parent / "hivm_ops.json").write_text(
            json.dumps(hivm_ops, indent=2, ensure_ascii=False), encoding="utf-8")

        return hivm_text
    except Exception as e:
        print(f"  [WARN] Triton→HIVM failed: {e}")
        print(f"  (fallback: provide --hivm-mlir)")
        return None


def run_analyzers(round_dir: Path, tier: int = 1) -> Optional[dict]:
    """运行完整分析链: HIVM + msprof → merged report。"""
    from analyzers.hivmir_analyzer import HIVMIRAnalyzer
    from analyzers.msprof_analyzer import MsprofAnalyzer
    from analyzers.dsl_merger import merge, format_llm

    # Step 1: HIVM
    hivm_file = round_dir / "hivmir" / "compiler_output" / "hivmir_output.mlir"
    if not hivm_file.exists():
        print(f"  [SKIP] No HIVM MLIR at {hivm_file}")
        return None

    ha = HIVMIRAnalyzer()
    hr = ha.analyze_file(hivm_file)
    hivm_dict = ha.to_dict(hr)

    hivm_report_file = round_dir / "hivmir" / "hivmir_report.json"
    hivm_report_file.parent.mkdir(parents=True, exist_ok=True)
    hivm_report_file.write_text(json.dumps(hivm_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  HIVM: {hr.num_ops} ops, RAW={len(hr.raw_deps)} WAR={len(hr.war_deps)}")

    # Step 2: msprof
    ma = MsprofAnalyzer()
    msprof_dict = {}
    opprof_dir = ma.find_latest_opprof(round_dir / "msprof")
    if not opprof_dir:
        opprof_dir = ma.find_latest_opprof(Path("/home/hjkc2/msprof_out2"))
    if opprof_dir and opprof_dir.exists():
        mr = ma.parse_existing(opprof_dir)
        msprof_dict = ma.to_dict(mr)
        msprof_file = round_dir / "msprof" / "pipeline_report.json"
        msprof_file.parent.mkdir(parents=True, exist_ok=True)
        msprof_file.write_text(json.dumps(msprof_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  msprof: {mr.num_ops} instrs, {mr.num_cores} cores, {mr.total_ns:.1f}ns")
    else:
        print(f"  msprof: no trace data (SKIP)")

    # Step 3: Merge
    merged = merge(hivm_dict, msprof_dict, tier)
    merged_dir = round_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "merged_report.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    (merged_dir / "final_report_llm.txt").write_text(
        format_llm(merged), encoding="utf-8")
    print(f"  Merged: {len(merged['per_op_statistics'])} ops, "
          f"has_timing={merged['meta']['has_msprof_timing']}")

    return merged


def run_optimization(
    kernel_path: Path,
    target_speedup: float = 1.5,
    max_rounds: int = 200,
    skip_triton: bool = False,
    hivm_mlir: Optional[Path] = None,
    msprof_dir: Optional[Path] = None,
    analyze_only: bool = False,
):
    """运行完整优化流程。"""

    # 1. 检测 kernel
    kernels = detect_kernels(kernel_path)
    if not kernels:
        print(f"[ERROR] No @triton.jit kernel found in {kernel_path}")
        return 1

    kernel_fn_name, op_type = kernels[0]
    kernel_name = get_kernel_name(kernel_path)

    print(f"Kernel: {kernel_name}")
    print(f"  Function: {kernel_fn_name}()")
    print(f"  Type: {op_type}")

    # 2. 初始化输出目录
    kernel_dir = init_output_dir(kernel_path, kernel_name)
    print(f"Output: {kernel_dir}")

    round0 = kernel_dir / "round0"

    # 3. Triton .py → HIVM MLIR (跳过则用已有文件)
    if not skip_triton:
        print("\n[Step 1] Triton .py → HIVM MLIR...")
        hivm_text = run_triton_to_hivm(kernel_path, round0, kernel_fn_name)
        if hivm_text is None and hivm_mlir:
            shutil.copy2(hivm_mlir,
                         round0 / "hivmir" / "compiler_output" / "hivmir_output.mlir")
            print(f"  Using provided HIVM MLIR: {hivm_mlir}")
    elif hivm_mlir:
        hivm_dir = round0 / "hivmir" / "compiler_output"
        hivm_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hivm_mlir, hivm_dir / "hivmir_output.mlir")

    # 4. 复制 msprof 数据
    if msprof_dir and msprof_dir.exists():
        msprof_dst = round0 / "msprof"
        msprof_dst.mkdir(parents=True, exist_ok=True)
        for item in msprof_dir.iterdir():
            dst = msprof_dst / item.name
            if not dst.exists():
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
        print(f"\n[Step 2] msprof data: {msprof_dir} → {msprof_dst}")

    # 5. 运行分析
    print(f"\n[Step 3] Analyzers...")
    merged = run_analyzers(round0)

    if analyze_only:
        print(f"\n[ANALYZE-ONLY] Done. Output: {kernel_dir}")
        return 0

    # 6. 优化循环
    print(f"\n[Step 4] Optimization Loop...")
    from agents.orchestrator import Orchestrator
    orch = Orchestrator(
        kernel_path=round0 / kernel_path.name,
        kernel_name=kernel_name,
        target_speedup=target_speedup,
        max_rounds=max_rounds,
        msprof_dir=msprof_dir,
    )
    orch._kernel_fn_name = kernel_fn_name
    orch._op_type = op_type
    orch.current_kernel = (round0 / kernel_path.name).read_text(encoding="utf-8")

    result = orch.run()
    print(f"\nFinal: {result['best_speedup']:.2f}x in {result['rounds']} rounds")
    print(f"Output: {result['output']}")
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description="Triton Agent Optimizer v3.0")
    p.add_argument("input", type=str, help="Triton kernel file (.py)")
    p.add_argument("--target", type=float, default=1.5)
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--skip-triton", action="store_true",
                   help="Skip Triton→HIVM (use existing .mlir)")
    p.add_argument("--hivm-mlir", type=str,
                   help="Path to existing HIVM MLIR file")
    p.add_argument("--msprof-dir", type=str,
                   help="Path to OPPROF_xxx directory with msprof trace data")
    p.add_argument("--analyze-only", action="store_true",
                   help="Only run analyzers, skip optimization loop")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Not found: {input_path}")
        return 1

    return run_optimization(
        input_path,
        args.target,
        args.max_rounds,
        skip_triton=args.skip_triton,
        hivm_mlir=Path(args.hivm_mlir) if args.hivm_mlir else None,
        msprof_dir=Path(args.msprof_dir) if args.msprof_dir else None,
        analyze_only=args.analyze_only,
    )


if __name__ == "__main__":
    sys.exit(main())
