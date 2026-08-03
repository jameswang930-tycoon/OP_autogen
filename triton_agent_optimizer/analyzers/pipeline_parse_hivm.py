#!/usr/bin/env python3
"""源1: 解析真实 HIVM (hivm_try.txt) → 统一格式 JSON。

结构字段真实 (op_type/dst/src/size_kb/region/attrs/依赖);
时序/带宽字段 = None, 由 sim/board 源补充。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hivmir_analyzer import HIVMIRAnalyzer  # noqa: E402
from pipeline_schema import empty_op, make_report, write_json  # noqa: E402


def parse(hivm_path):
    ha = HIVMIRAnalyzer()
    report = ha.analyze_file(Path(hivm_path))
    d = ha.to_dict(report)

    ops = []
    for op in d["per_op_statistics"]:
        o = empty_op()
        o.update({
            "op_id": op["op_id"],
            "op_name": op["op_type"],           # 可对齐名 = op_type (gm_to_ub/mmadL1/set_flag...)
            "op_type": op["op_type"],
            "engine": op["engine"],
            "instruction": op["instruction"],
            "dst": op["dst"], "src": op["src"], "src2": op["src2"],
            "size_kb": op["size_kb"],
            "memory_region": op["memory_region"],
            "dtype": op.get("dtype"),
            "attrs": op.get("attrs"),
            "dependencies": op.get("dependencies"),
        })
        ops.append(o)

    summary = {
        "total_ns": None, "num_ops": len(ops),
        "execution_mode": None, "num_cores": None,
        "kernel_name": report.kernel_name or Path(hivm_path).stem,
    }
    return make_report(
        "hivm", [hivm_path], summary, ops,
        deps=d.get("dependencies_summary"),
        buffers=d.get("buffers"),
        notes=["结构字段来自真实 HIVM (hivm_try.txt); 时序/带宽待 sim/board 补充",
               "sync op (set_flag/wait_flag/pipe_barrier) 已计入 op_id"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_hivm.py <hivm_try.txt> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[hivm] {sys.argv[1]} -> {sys.argv[2]}")
