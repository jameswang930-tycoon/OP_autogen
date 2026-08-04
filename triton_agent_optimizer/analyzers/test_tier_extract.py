#!/usr/bin/env python3
"""逐 tier 提取核对 — 对 diagnosis.json 按 6 层策略各筛各的字段, 打印每层有数据/无数据。

用途: 验证"当前阶段→筛当前策略字段"的规则是否正确 (见 docx/msprof_fields_reference.md 第四节)。
用法: python3 analyzers/test_tier_extract.py [diagnosis.json]
  默认: input/matmul/e2e_run/06_diagnosis/diagnosis.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.scheduler import extract_tier_fields, TIER_LABEL  # noqa: E402


if __name__ == "__main__":
    dgn = sys.argv[1] if len(sys.argv) > 1 \
        else "input/matmul/e2e_run/06_diagnosis/diagnosis.json"
    d = json.loads(Path(dgn).read_text(encoding="utf-8"))
    for t in range(1, 7):
        txt = extract_tier_fields(d, t)
        filled = sum(1 for l in txt.splitlines() if "(无数据)" not in l and l.startswith("-"))
        total = sum(1 for l in txt.splitlines() if l.startswith("-"))
        print(f"══════ Tier{t} ({TIER_LABEL.get(t, '')}) — {filled}/{total} 字段有数据 ══════")
        print(txt)
        print()
