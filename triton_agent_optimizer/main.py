#!/usr/bin/env python3
"""
Triton Agent Optimizer v4 — 主入口（单文件驱动 + 真机双源 + 6 层轮次化）

v4 流程:
  1. 合并 input/<op> 的算子(triton_kernel.py) + 场景(config.json) + 测试(test_*.py)
     → kernel_op.py（单文件，后续输入/输出只读写这一个文件）
  2. 初始化 outputs/<op>/<tier>/roundN 结构
  3. Scheduler 循环（每轮）:
     run_optimize.sh <op_dir> <round_dir> → diagnosis.json
       → 提取当前 tier 字段段 → Planner(LLM, 走 nga run) → plan.md
       → Coder(LLM, 走 nga run) → 改 kernel_op.py
       → 只跑 msprof 端到端 → 加速比 → 报错就地回传 Coder
       → 晋升/降级/停止

LLM 调用（服务器无 Claude API）:
  export LLM_CLI_COMMAND="nga run"   # echo "<prompt>" | nga run
  Planner/Coder 自动引用 skills/triton-op-planner、skills/triton-op-coder

Usage:
  python main.py input/matmul [--max-rounds N] [--target 1.5] [--stub]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from analyzers.merge_single_file import merge  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Triton Agent Optimizer v4")
    p.add_argument("op_dir", type=str, help="input/<op> 目录 (含 triton_kernel.py + config.json + test)")
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--target", type=float, default=1.5)
    p.add_argument("--stub", action="store_true", help="不调 LLM/真机, 用 stub (本地测试)")
    p.add_argument("--remerge", action="store_true", help="强制重新合并单文件 (会覆盖 coder 改动)")
    args = p.parse_args()

    op_dir = Path(args.op_dir)
    if not op_dir.exists():
        print(f"[ERROR] Not found: {op_dir}")
        return 1

    # ① 单文件 kernel_op.py (★源文件, coder 直接改它, 不覆盖)
    #    若缺 (旧式三文件 op) → 用 merge_single_file.py 生成一次
    kernel_op = op_dir / "kernel_op.py"
    if not kernel_op.exists():
        merge(op_dir, kernel_op)
    if not kernel_op.exists():
        print(f"[ERROR] 单文件不存在: {kernel_op}")
        return 1
    print(f"[main] 单文件: {kernel_op}")

    # ② Scheduler 循环
    from agents.scheduler import Scheduler
    s = Scheduler(op_dir, max_rounds=args.max_rounds,
                  target_speedup=args.target, stub=args.stub)
    return s.run()


if __name__ == "__main__":
    sys.exit(main())
