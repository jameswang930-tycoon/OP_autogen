"""从 api_inventory.txt 生成 extension 签名表（任务 B）。

inventory 行格式已知：``模块路径 | 名称 | signature | doc首行``。本脚本把每行的
signature 解析成位置参数个数，按名称合并重载，产出签名表 yaml。

**逻辑不涉密**（行格式已知）。保密环境的 GLM 4.7 只需跑一次本脚本、把真实 inventory
喂进去，把产出的 yaml 放到 ``PRESIM_SIGNATURE_TABLE`` 配置路径——不写任何检查逻辑。

用法：
  .venv/bin/python -m control.build_signature_table <api_inventory.txt> <signature_table.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _split_top_level(s: str) -> list[str]:
    """按顶层逗号（深度 0）切分，忽略括号内的逗号。"""
    out: list[str] = []
    depth = 0
    cur = ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


def count_positional_params(signature: str) -> int:
    """从形如 ``foo(a, b, c=1, *args)`` 的 signature 数出必填位置参数个数。"""
    if "(" not in signature or ")" not in signature:
        return 0
    inside = signature[signature.index("(") + 1: signature.rindex(")")]
    if not inside.strip():
        return 0
    n = 0
    for raw in _split_top_level(inside):
        p = raw.strip()
        if not p or p.startswith("*"):
            continue
        if "=" in p:  # 带默认值 -> 非必填位置参数
            continue
        n += 1
    return n


def parse_inventory(text: str) -> list[dict]:
    """解析 inventory 文本 -> [{name, param_counts}]，同名重载合并。"""
    counts: dict[str, set[int]] = {}
    order: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        name, signature = parts[1], parts[2]
        if name not in counts:
            counts[name] = set()
            order.append(name)
        counts[name].add(count_positional_params(signature))
    return [{"name": name, "param_counts": sorted(counts[name])} for name in order]


def main(inventory_path: str, out_path: str) -> int:
    text = Path(inventory_path).read_text(encoding="utf-8")
    entries = parse_inventory(text)
    Path(out_path).write_text(
        yaml.safe_dump(entries, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(entries)} signatures to {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="build extension signature table from api_inventory.txt")
    ap.add_argument("inventory")
    ap.add_argument("out")
    args = ap.parse_args()
    sys.exit(main(args.inventory, args.out))
