"""最小闭环演示。

用桩函数(stub)代替真实的规划/生成/校验技能,只为展示回路如何接线,
以及经验如何被检索、被写回、越用越靠前。真实接入见
docs/project_knowledge/memory_integration.md。

运行(从项目根): python memory/demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 本脚本在 memory/ 包内,需把项目根加到 path,使 from memory import 生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import (
    ExperienceStore,
    RunLog,
    compute_fingerprint,
    retrieve,
    format_context,
    record_attempt,
    add_experience,
)


# ---- 桩:真实系统里这两处替换为你现有的技能 --------------------------------

def generate_kernel(op_spec: dict, context: str) -> str:
    """替换为:triton-plan + triton-gen(把 context 拼进提示词)。"""
    return f"// kernel for {op_spec['op_kind']}\n// context used:\n{context}"


def verify(kernel_ref: str) -> bool:
    """替换为:triton-verify(CPU 模拟器正确性校验)。此处恒为通过。"""
    return True


# ---- 一次尝试:检索 → 生成 → 校验 → 写回 ------------------------------------

def run_attempt(store, log, op_spec, bottleneck):
    fp = compute_fingerprint(op_spec, bottleneck=bottleneck)

    # 注入点:检索历史经验,拼进上下文
    hits = retrieve(store, fp, n=3)
    context = format_context(hits)

    # 现有流水线(桩)
    kernel = generate_kernel(op_spec, context)
    passed = verify(kernel)

    # 写回点:记日志 + 更新经验统计
    rec = record_attempt(
        log, store, fp,
        retrieved_ids=[e.id for e in hits],
        passed=passed,
        kernel_ref="artifact://demo",
    )
    print(f"  特征={fp.key()}  注入经验={[e.id for e in hits]}  通过={passed}  run={rec.run_id}")
    return fp, passed


def main():
    workdir = Path(tempfile.mkdtemp(prefix="mem_demo_"))
    store = ExperienceStore(workdir / "experience.json")
    log = RunLog(workdir / "runlog.jsonl")
    op_spec = {"op_kind": "matmul", "shapes": {"M": 4096, "N": 4096, "K": 4096}}
    bottleneck = "compute_bound"

    print("第 1 次尝试(经验库为空):")
    fp, _ = run_attempt(store, log, op_spec, bottleneck)

    # 成功后手工沉淀一条经验(最小版;将来由自动蒸馏替换)
    add_experience(store, fp, text="K 维分块取 64,配合双缓冲可提升计算受限型 matmul 的吞吐。")
    add_experience(store, fp, text="占用率不足时优先增大每线程负载,而非盲目加 block。")
    print("  -> 已沉淀 2 条经验")

    print("第 2 次尝试(同类问题,应检索到上面的经验):")
    run_attempt(store, log, op_spec, bottleneck)

    print("第 3 次尝试:")
    run_attempt(store, log, op_spec, bottleneck)

    print("\n经验库当前状态(用过/帮上忙):")
    for e in store.all():
        print(f"  [{e.id}] used={e.used} helped={e.helped} score={e.score():.3f}  {e.text[:20]}...")

    print(f"\n日志共 {len(log.read_all())} 条,文件位于: {workdir}")


if __name__ == "__main__":
    main()
