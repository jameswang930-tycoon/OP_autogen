#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
  算子耗时诊断工具 — 手动指定 kernel_op.py, 跑 msprof, 打印每个算子耗时
═══════════════════════════════════════════════════════════════════════════════
  ★怎么用:
    1) 改下面 KERNEL 为你要测的 kernel_op.py 的【绝对路径】(服务器上)
    2) 运行:  python3 diagnose_kernel_time.py
       或命令行指定: python3 diagnose_kernel_time.py /abs/path/kernel_op.py
    3) 输出: 每个 kernel 的 启动次数/总耗时us/平均us/占比%
              + 目标 kernel 总计 + ÷LOOP 的单遍端到端 + aclnn 框架开销

  用途: 判断某个版本(基线/某轮)是否真的有优化 —
        ⚠ 对比两个版本: 看【占比最大的算子】有没有降。
        如果大头算子(如 99% 的 attention_scores_kernel)没降 → 优化没打中瓶颈。
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════ 手动配置区 ═══════════════════════
# ★改这里: 要测的 kernel_op.py 绝对路径 (服务器上)
KERNEL = "/home/user/triton_agent_optimizer/outputs/attention_mlp/baseline_verify/kernel_op.py"
# 一次 msprof 内 kernel 内部循环次数 (求平均; 和 verify 一致默认 30)
LOOP = 30
# 预热次数 (JIT 编译/冷cache)
WARMUP = 3
# ═══════════════════════════════════════════════════════════

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="算子耗时诊断: msprof 跑单个 kernel_op.py, 打印每算子耗时")
    p.add_argument("path", nargs="?", default=KERNEL, help="kernel_op.py 绝对路径 (默认用顶部 KERNEL)")
    p.add_argument("--loop", type=int, default=LOOP)
    p.add_argument("--warmup", type=int, default=WARMUP)
    args = p.parse_args()

    kernel = Path(args.path)
    if not kernel.exists():
        print(f"❌ 找不到 kernel_op.py: {kernel}")
        print("   → 改顶部 KERNEL 的绝对路径, 或命令行传路径")
        sys.exit(1)

    print(f"═══ 测: {kernel} ═══")
    print(f"    内部循环 {args.loop} 次/遍, 预热 {args.warmup} 次")

    env = dict(os.environ, KERNEL_LOOP=str(args.loop))
    # 预热 (JIT 编译/冷cache)
    for i in range(args.warmup):
        r = subprocess.run(["python3", str(kernel)], capture_output=True, text=True, timeout=1800, env=env)
    print(f"    预热 x{args.warmup} done")

    # 一次 msprof 内循环 loop 次
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "msprof_0"
        cmd = ["msprof", f"--output={out}", f"--application=python3 {kernel}", "--ai-core=on"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
        if r.returncode != 0:
            print(f"❌ msprof 失败: {(r.stderr or r.stdout)[-600:]}")
            sys.exit(1)
        per, aclnn = _parse_op_summary(out)

    if not per:
        print("❌ op_summary 无目标 kernel 行")
        sys.exit(1)

    total_target = sum(e["total_us"] for e in per.values())
    total_aclnn = sum(e["total_us"] for e in aclnn.values())

    print("\n═══ 各 kernel 耗时 (1 次 msprof, 内部循环 %d 次) ═══" % args.loop)
    rows = sorted(per.items(), key=lambda kv: -kv[1]["total_us"])
    for name, e in rows:
        avg = e["total_us"] / e["count"]
        pct = e["total_us"] / total_target * 100 if total_target else 0
        print(f"  {name:<28} 启动{e['count']:>5}次  总{e['total_us']:>12.1f}us  平均{avg:>9.2f}us  {pct:>5.1f}%")
    print("─" * 70)
    print(f"  目标 kernel 总计      : {total_target:>12.1f}us")
    print(f"  ÷{args.loop} 单遍端到端      : {total_target/args.loop:>12.1f}us   (≈ {total_target/args.loop*1000:.0f} ns)")
    if aclnn:
        print(f"  aclnn 框架总计        : {total_aclnn:>12.1f}us   (数据准备, 不在目标内)")
    print(f"\n  ★占比最大的算子 = 瓶颈: {rows[0][0]} ({rows[0][1]['total_us']/total_target*100:.1f}%)")
    print("  ⚠ 优化要打中它才有效; 若它没降 → 优化没打到瓶颈, 加速比才会 ~1")
    print(f"\n  (baseline_ns 对比: 拿这个 ÷{args.loop} 的值 ×1000 当 baseline_ns 去对比)")


def _parse_op_summary(prof_dir):
    """读 op_summary: 返回 ({目标kernel名: {count,total_us}}, {aclnn名: {...}})."""
    summaries = sorted(Path(prof_dir).rglob("op_summary*.csv"))
    if not summaries:
        return {}, {}
    with open(summaries[0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per, aclnn = {}, {}
    for row in rows:
        dur = row.get("Task Duration(us)") or row.get("TaskDuration")
        op = row.get("Op Name") or row.get("OpName") or "?"
        if not dur:
            continue
        try:
            d = float(dur)
        except ValueError:
            continue
        bucket = aclnn if op.lower().startswith("aclnn") else per
        e = bucket.setdefault(op, {"count": 0, "total_us": 0.0})
        e["count"] += 1
        e["total_us"] += d
    return per, aclnn


if __name__ == "__main__":
    main()
