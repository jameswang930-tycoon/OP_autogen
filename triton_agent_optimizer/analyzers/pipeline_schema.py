#!/usr/bin/env python3
"""Pipeline 公共 schema — 三个源脚本产出统一格式 JSON, 方便 merge。

约定:
  - 每个 op 都是同一组字段 (OP_FIELDS), 该源没有的字段 = None (待补充)。
  - merge 阶段用其他源填充 None。
  - 字段命名与 dsl_merger/29 字段对齐。
"""
import json
from datetime import datetime
from pathlib import Path

# 每个 op 的统一字段 (三源同形)
OP_FIELDS = [
    "op_id", "op_name", "op_type", "engine", "instruction",
    "dst", "src", "src2", "size_kb", "memory_region", "dtype", "attrs",
    "dependencies",
    "duration_ns", "start_ns", "end_ns", "time_ratio", "cycles",
    "pipeline_channel", "core_id", "data_size_bytes",
    "effective_bw_gb_s", "peak_bw_gb_s", "bw_utilization", "regime",
]


def empty_op() -> dict:
    return {k: None for k in OP_FIELDS}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_report(source: str, input_files: list, summary: dict, ops: list,
                engine_util=None, deps=None, parallelism=None,
                critical_path=None, buffers=None, notes=None) -> dict:
    return {
        "meta": {
            "source": source,
            "generated_at": datetime.now().isoformat(),
            "input_files": [str(x) for x in input_files],
            "schema_version": "1.0",
        },
        "execution_summary": {
            "total_ns": summary.get("total_ns"),
            "num_ops": summary.get("num_ops", len(ops)),
            "execution_mode": summary.get("execution_mode"),
            "num_cores": summary.get("num_cores"),
            "kernel_name": summary.get("kernel_name"),
        },
        "per_op_statistics": ops,
        "engine_utilization": engine_util or {},
        "dependencies_summary": deps or {},
        "parallelism": parallelism or {},
        "critical_path": critical_path or {},
        "buffers": buffers or {},
        "notes": notes or [],
    }
