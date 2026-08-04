#!/usr/bin/env python3
"""
Triton Agent Optimizer v4 — 主入口（单文件驱动 + 真机双源 + 6 层轮次化）

v4 流程:
  1. 单文件 kernel_op.py（①config + ②kernel + ③test 一体, coder 只改它）
  2. Scheduler 循环（每轮）:
     run_optimize.sh <op_dir> <round_dir> → diagnosis.json
       → 07_tier{N}_fields (筛选字段) → Planner(nga run) → plan.md (含 changes[])
       → Coder(nga run / 确定性替换 changes[]) → 改 kernel_op.py
       → (Tier2 多走一步) 08_fusion: 编译 HIVM MLIR → nga run fusion skill → 依赖分析
       → 只跑 msprof 端到端 → 加速比 → 报错就地回传 Coder → 晋升/降级/停止

LLM 调用（服务器无 Claude API, 用本地 codeagent）:
  export LLM_CLI_COMMAND="nga run"   # echo "<prompt>" | nga run
  三个 skill: skills/triton-op-planner / triton-op-coder / triton-op-fusion

════════════════════ 服务器运行步骤 ════════════════════
  0. 环境:
     conda activate triton-npu
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     cd triton_agent_optimizer
     echo "测试, 调用 skill triton-op-planner" | nga run   # 确认 nga 通
  1. 单文件能跑:
     python3 input/matmul/kernel_op.py
     # 预期: [info] kernel launched & synced OK
  2. 完整优化循环 (一键):
     LLM_CLI_COMMAND="nga run" python3 main.py input/matmul --max-rounds 2
     # 每轮: 采集→07字段→planner→coder→msprof端到端→加速比
  3. 只采集+解析 (不跑优化):
     bash analyzers/run_optimize.sh input/matmul input/matmul/e2e_run
  4. 只看各 tier 筛字段:
     python3 analyzers/test_tier_extract.py input/matmul/e2e_run/06_diagnosis/diagnosis.json
  5. 只测 Tier2 融合分析:
     python3 analyzers/run_hivm_fusion.py input/matmul/kernel_op.py /tmp/fusion_test
  6. 看诊断报告:
     python3 input/matmul/real_report.py <round_dir>/06_diagnosis/diagnosis.json
  ══════════════════════════════════════════════════════════

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
