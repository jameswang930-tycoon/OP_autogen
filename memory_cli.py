#!/usr/bin/env python3
"""memory_cli —— 记忆模块的独立驱动入口(CLI)。

在 triton-plan 之后、triton-gen 之前手动调 inject;在 triton-verify 之后手动调 record。
plan/gen/verify 仍是现有 LLM skill,本工具只封装 memory 包的纯函数部分。

用法(从项目根,用 .venv 的 python):
  .venv/bin/python memory_cli.py inject <op> [--plan PATH] [--bottleneck BN]
  .venv/bin/python memory_cli.py record <op> --passed|--failed [--plan PATH] [--kernel-ref REF]
  .venv/bin/python memory_cli.py add <op> --text "..." [--plan PATH] [--bottleneck BN]
  .venv/bin/python memory_cli.py stats [--op OP]

经验库/日志默认落 项目根 memory/experience/、memory/runlog/;用环境变量
MEMORY_STORE_DIR 覆盖(AB 实验可分 on/off 两套)。不依赖 cwd。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from memory import (
    ExperienceStore,
    RunLog,
    fingerprint_from_plan_json,
    retrieve,
    format_context,
    record_attempt,
    add_experience,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MEMORY_STORE_DIR", ROOT / "memory"))
STORE_PATH = DATA_DIR / "experience" / "experience.json"
LOG_PATH = DATA_DIR / "runlog" / "runlog.jsonl"


def _plan_json_path(op: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return ROOT / "emulators" / "test" / op / ".plan.json"


def _load_plan(op: str, override: str | None, bottleneck: str | None):
    p = _plan_json_path(op, override)
    plan_json = json.loads(p.read_text(encoding="utf-8"))
    fp = fingerprint_from_plan_json(plan_json, bottleneck=bottleneck)
    return p, plan_json, fp


def cmd_inject(args):
    store = ExperienceStore(STORE_PATH)
    p, plan_json, fp = _load_plan(args.op, args.plan, args.bottleneck)
    hits = retrieve(store, fp, n=3)
    ids = [e.id for e in hits]
    plan_json["retrieved_experience"] = format_context(hits)   # 给 triton-gen 读
    plan_json["retrieved_ids"] = ids                            # 给 record 读(非 gen 契约字段)
    p.write_text(json.dumps(plan_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fp={fp.key()}  注入 {len(ids)} 条: {ids}")
    print(f"已写回 {p.name}(retrieved_experience + retrieved_ids)")


def cmd_record(args):
    store = ExperienceStore(STORE_PATH)
    log = RunLog(LOG_PATH)
    _, plan_json, fp = _load_plan(args.op, args.plan, args.bottleneck)
    ids = plan_json.get("retrieved_ids", [])
    rec = record_attempt(
        log, store, fp,
        retrieved_ids=ids,
        passed=args.passed,
        kernel_ref=args.kernel_ref or f"emulators/test/{args.op}",
    )
    print(f"已记录 run={rec.run_id} fp={fp.key()} passed={rec.passed} retrieved={ids}")


def cmd_add(args):
    store = ExperienceStore(STORE_PATH)
    _, _, fp = _load_plan(args.op, args.plan, args.bottleneck)
    eid = add_experience(store, fp, text=args.text)
    print(f"已新增经验 [{eid}] applies_to={fp.key()}: {args.text}")


def cmd_stats(args):
    store = ExperienceStore(STORE_PATH)
    exps = store.all()
    if args.op:
        exps = [e for e in exps if e.applies_to.split("|", 1)[0] == args.op]
    if not exps:
        print("(经验库为空)")
        return
    print(f"共 {len(exps)} 条经验:")
    for e in exps:
        print(f"  [{e.id}] {e.applies_to}  used={e.used} helped={e.helped} "
              f"failed={e.failed} harmed={e.harmed} score={e.score():.3f}  {e.text[:30]}")


def main():
    ap = argparse.ArgumentParser(description="memory 模块独立驱动 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inject", help="plan 后:读 .plan.json + 检索 + 追加 retrieved_experience")
    pi.add_argument("op"); pi.add_argument("--plan"); pi.add_argument("--bottleneck")
    pi.set_defaults(func=cmd_inject)

    pr = sub.add_parser("record", help="verify 后:写 runlog + 更新经验统计")
    pr.add_argument("op"); pr.add_argument("--plan"); pr.add_argument("--bottleneck")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--passed", action="store_true")
    g.add_argument("--failed", action="store_true")
    pr.add_argument("--kernel-ref")
    pr.set_defaults(func=cmd_record)

    pa = sub.add_parser("add", help="手工新增一条经验")
    pa.add_argument("op"); pa.add_argument("--plan"); pa.add_argument("--bottleneck")
    pa.add_argument("--text", required=True)
    pa.set_defaults(func=cmd_add)

    ps = sub.add_parser("stats", help="看经验库统计")
    ps.add_argument("--op")
    ps.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
