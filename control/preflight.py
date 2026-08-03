"""preflight —— 配置自检（E6.1）。

一条命令扫描所有需要保密环境配置的点，报告每项状态（OK/STUB/MISSING/EXAMPLE）。
**纯本地静态检查，不执行任何远程调用，秒级返回。** 拉代码进环境后第一件事就跑它，
一眼看到还差哪几项没配，而不是跑编排器撞一堆报错。

  OK       已正确配置
  STUB     仍是占位（槽位未实现 / 未配置）
  MISSING  配了路径但文件不存在
  EXAMPLE  仍是示例内容，未定稿
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

EXAMPLE_VOCAB = {"compute_bound_at_peak", "memory_underfilled", "stall_dependency"}


def _check_llm(env):
    g = env.get("AGENT_GENERATE_MODEL")
    c = env.get("AGENT_CHOOSE_LEVER_MODEL")
    if g and c:
        return ("llm", "OK", f"generate={g} choose_lever={c}")
    missing = [k for k, v in (("generate", g), ("choose_lever", c)) if not v]
    return ("llm", "STUB", f"models not set: {missing} (set AGENT_GENERATE_MODEL / AGENT_CHOOSE_LEVER_MODEL)")


def _check_agent(env):
    # agent 命令前缀首 token（AGENT_CMD 首词，缺省 "agent"）——不写死具体后端名
    cmd = env.get("AGENT_CMD")
    binname = cmd.split()[0] if cmd else "agent"
    if shutil.which(binname):
        return ("agent", "OK", f"{binname} available")
    return ("agent", "STUB", f"{binname} not on PATH (confidential-env only)")


def _check_signature_table(env):
    path = env.get("PRESIM_SIGNATURE_TABLE")
    if path:
        if not Path(path).exists():
            return ("signature_table", "MISSING", f"not found: {path}")
        try:
            from .presim_gate import load_signature_table
            load_signature_table(path)
        except Exception as exc:  # noqa: BLE001
            return ("signature_table", "MISSING", f"unreadable: {exc}")
        return ("signature_table", "OK", path)
    return ("signature_table", "EXAMPLE", "default example table; set PRESIM_SIGNATURE_TABLE to the real one")


def _check_template(env):
    import re
    from .launch_template import load_launchable_template, LAUNCHABLE_PLACEHOLDERS
    path = env.get("LAUNCHABLE_TEMPLATE_PATH")
    if path:
        if not Path(path).exists():
            return ("launchable_template", "MISSING", f"not found: {path}")
        tmpl = load_launchable_template(path)
        ph = set(re.findall(r"{{([A-Z][A-Z0-9_]*)}}", tmpl))
        if ph == set(LAUNCHABLE_PLACEHOLDERS):
            return ("launchable_template", "OK", path)
        return ("launchable_template", "MISSING", f"placeholder mismatch: {ph}")
    return ("launchable_template", "EXAMPLE", "default example template; set LAUNCHABLE_TEMPLATE_PATH")


def _check_vocab():
    try:
        from . import vocabulary
        ids = vocabulary.all_ids()
    except Exception as exc:  # noqa: BLE001
        return ("vocabulary", "MISSING", str(exc))
    if ids == EXAMPLE_VOCAB:
        return ("vocabulary", "EXAMPLE", "still the 3 example entries; replace with real stall types")
    return ("vocabulary", "OK", f"{len(ids)} categories")


def _check_launch_dirs(env):
    states = []
    for var in ("LAUNCH_INPUT_DIR", "LAUNCH_OUTPUT_DIR", "LAUNCH_SCRIPT"):
        v = env.get(var)
        states.append("OK" if (v and Path(v).exists()) else ("MISSING" if v else "STUB"))
    if "MISSING" in states:
        return ("launch_dirs", "MISSING", "some configured launch paths do not exist")
    if all(s == "STUB" for s in states):
        return ("launch_dirs", "STUB", "launch dirs/script not configured (slot 4)")
    return ("launch_dirs", "OK", "launch dirs + script configured")


def _check_parse_raw():
    from .feedback_adapter import parse_raw
    try:
        parse_raw({"correct": True})
        return ("parse_raw", "OK", "implemented")
    except NotImplementedError:
        return ("parse_raw", "STUB", "slot still NotImplementedError (slot 2 not ready)")
    except Exception as exc:  # noqa: BLE001 - any other error means it IS implemented
        return ("parse_raw", "OK", f"implemented (raised {type(exc).__name__} on dummy input)")


def _check_cheatsheet():
    import yaml
    from .check_extension_cheatsheet import DEFAULT_REFS
    names = []
    for f in sorted([*DEFAULT_REFS.glob("*.yaml"), *DEFAULT_REFS.glob("*.yml")]):
        try:
            d = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("category") is not None:
                names.append(d.get("name", ""))
        except Exception:  # noqa: BLE001
            continue
    real = [n for n in names if n and not n.startswith("sample")]
    if not real:
        return ("cheatsheet", "EXAMPLE", "only the sample entry; fill references/ with real primitives")
    return ("cheatsheet", "OK", f"{len(real)} real primitive(s)")


def run_checks(env=None) -> list[tuple[str, str, str]]:
    """运行全部本地检查，返回 [(name, state, detail), ...]。"""
    env = os.environ if env is None else env
    return [
        _check_llm(env),
        _check_agent(env),
        _check_signature_table(env),
        _check_template(env),
        _check_vocab(),
        _check_launch_dirs(env),
        _check_parse_raw(),
        _check_cheatsheet(),
    ]


def main(argv=None) -> int:
    rows = run_checks()
    for name, state, detail in rows:
        print(f"[{state:<8}] {name:<20} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
