#!/usr/bin/env python3
"""单文件合并 — 算子(triton_kernel.py) + 场景(config.json) + 测试(test_*.py) → kernel_op.py。

v4 架构核心: 后续所有输入/输出只读写这一个文件, 杜绝多文件错位。
  - kernel 源码原样内联 (不解析不重写, 保真)
  - test 驱动去掉 `from triton_kernel import ...` 和 sys.path 注入, 其余保留
  - config.json 注入为 os.environ.setdefault + block 常量, 调度器每轮可改 config.json 后重合并

用法:
  python merge_single_file.py <op_dir> [--out <kernel_op.py>]
  默认: <op_dir>/kernel_op.py
"""
import argparse
import json
import re
import sys
from pathlib import Path


def _find_test_file(op_dir: Path) -> Path:
    """找测试驱动: 优先 test*.py (避免误选 real_report.py 等工具脚本)。"""
    candidates = [p for p in op_dir.glob("*.py")
                  if p.name not in ("triton_kernel.py", "kernel_op.py")
                  and "__pycache__" not in p.parts]
    # 优先 test_*.py / test*.py
    for p in sorted(candidates):
        if p.name.startswith("test"):
            return p
    # 兜底: 任取一个非 kernel 的 .py
    return candidates[0] if candidates else (op_dir / "test.py")


# 头部已 import 的模块, 从 kernel/test 里去掉重复 (os/sys/torch/torch_npu/triton/tl, 允许尾注释)
_DUP_IMPORTS = re.compile(
    r"^\s*import\s+(os|sys|torch|torch_npu|triton(?:\.language)?)\b.*$|"
    r"^\s*from\s+(triton(?:\.language)?|torch_npu)\s+import\s+.*$",
    re.M)


def _strip_imports(src: str) -> str:
    """去掉 `from triton_kernel import X` / sys.path / BLOCK_*/DTYPE 赋值 / 重复 import。
    BLOCK_*/DTYPE 由 config.json 注入为准, 避免 test 硬编码覆盖。"""
    src = re.sub(r"^\s*from\s+triton_kernel\s+import\s+.*$", "", src, flags=re.M)
    src = re.sub(r"^\s*sys\.path\.insert\(.*$", "", src, flags=re.M)
    src = re.sub(r"^\s*BLOCK_\w+(?:\s*,\s*BLOCK_\w+)*\s*=.*$", "", src, flags=re.M)
    src = re.sub(r"^\s*DTYPE\s*=.*$", "", src, flags=re.M)
    src = _DUP_IMPORTS.sub("", src)
    return src


def _config_injection(config: dict) -> str:
    """config.json → env setdefault + block/dtype 常量 (注入到 test 顶部)。"""
    s = config.get("scenario", {})
    kc = config.get("kernel_config", {})
    bp = kc.get("block_params", {})
    lines = []
    lines.append("# ═══ 场景 config (config.json) 注入 ═══")
    for var, val in (("M", s.get("M")), ("N", s.get("N")), ("K", s.get("K"))):
        if val:
            lines.append(f'os.environ.setdefault("MATMUL_{var}", "{val}")')
    lines.append(f"DTYPE = torch.{s.get('dtype', 'float32')}")
    for k, v in bp.items():
        if v:
            lines.append(f"{k} = {v}")
    lines.append("")
    return "\n".join(lines)


def merge(op_dir: Path, out_file: Path) -> Path:
    kernel_py = op_dir / "triton_kernel.py"
    if not kernel_py.exists():
        raise SystemExit(f"[merge] 找不到 {kernel_py}")
    test_py = _find_test_file(op_dir)
    config_py = op_dir / "config.json"
    config = json.loads(config_py.read_text(encoding="utf-8")) if config_py.exists() else {}

    kernel_src = _strip_imports(kernel_py.read_text(encoding="utf-8"))
    test_src = _strip_imports(test_py.read_text(encoding="utf-8"))
    injection = _config_injection(config)

    merged = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "# ═══════════════════════════════════════════════════════════════════\n"
        "#  单文件 kernel_op.py — 算子 + 场景 config + 测试 main 合并 (v4)\n"
        "#  由 analyzers/merge_single_file.py 生成; 后续输入/输出只读写这一个文件\n"
        "# ═══════════════════════════════════════════════════════════════════\n"
        "import os, sys\n"
        "import torch\n"
        "import torch_npu\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        + injection
        + "\n"
        + "# ══════ 算子 (triton_kernel.py 内联) ══════\n"
        + kernel_src
        + "\n"
        + "# ══════ 测试驱动 (test_*.py, 去掉 import) ══════\n"
        + test_src
        + "\n"
    )

    out_file.write_text(merged, encoding="utf-8")
    print(f"[merge] {op_dir} → {out_file}")
    return out_file


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="合并算子+场景+测试为单文件")
    p.add_argument("op_dir", type=str, help="input/<op> 目录")
    p.add_argument("--out", type=str, default=None, help="输出 kernel_op.py 路径")
    args = p.parse_args()
    op_dir = Path(args.op_dir)
    out = Path(args.out) if args.out else (op_dir / "kernel_op.py")
    merge(op_dir, out)
