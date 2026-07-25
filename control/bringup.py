"""bringup —— 逐点联调命令（E6.2）。

把端到端联调拆成一串可单独执行的单点验证：每个子命令只碰一个接缝、给明确 PASS/FAIL。
联调顺序：配一项 → `bringup <该项>` 验一项 → 通过再下一项。任一环炸，范围立刻锁定。

真实远程/nga 调用只在保密环境有效；开发机测试用注入的假 llm/launcher/parse_raw。
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def bringup_template(path=None) -> tuple[bool, str]:
    """验证可发射模板路径配对、占位符集合匹配、组装产物语法合法。"""
    try:
        from .launch_template import load_launchable_template, assemble_launchable, LAUNCHABLE_PLACEHOLDERS
        tmpl = load_launchable_template(path)
        ph = set(re.findall(r"{{([A-Z][A-Z0-9_]*)}}", tmpl))
        if ph != set(LAUNCHABLE_PLACEHOLDERS):
            return False, f"placeholder mismatch: {ph} != {set(LAUNCHABLE_PLACEHOLDERS)}"
        values = {
            "OP": "matmul", "SHAPES": [1], "DTYPE": "fp32",
            "KERNEL_BODY": "def kernel(a, b, c):\n    return a @ b",
            "REFERENCE": "def reference(a, b):\n    return a @ b",
        }
        assembled = assemble_launchable(tmpl, values)
        compile(assembled, "<bringup-template>", "exec")
        return True, "template loads, placeholders match, assembly compiles"
    except Exception as exc:  # noqa: BLE001
        return False, f"template check failed: {exc}"


def bringup_llm(llm) -> tuple[bool, str]:
    """验证 nga 能调通、返回内容能过解析闸门（恰好 1 python + 1 json 块）。"""
    try:
        from .orchestrator import parse_generate_response
        resp = llm.generate(
            'Output one python block printing hello, then one json block {"ok": true}. '
            'No explanation.')
        parse_generate_response(resp, allowed_extensions=set())
        return True, "llm returned 1 python + 1 json block"
    except Exception as exc:  # noqa: BLE001
        return False, f"llm check failed (need 1 python + 1 json block): {exc}"


def bringup_launch(launcher, kernel_file=None, save_dir=None) -> tuple[bool, str]:
    """验证远程仿真通 + 返回的 raw_sim_output 键齐全；把 raw 存到 save_dir/last_raw.json。"""
    try:
        required = ("correct", "max_abs_err", "cycles", "pipeline", "compiled", "compile_log")
        raw = launcher(kernel_file or "(kernel)")
        missing = [k for k in required if k not in raw]
        if missing:
            return False, f"raw_sim_output missing keys: {missing}"
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "last_raw.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, "launch returned complete raw_sim_output"
    except Exception as exc:  # noqa: BLE001
        return False, f"launch check failed: {exc}"


def bringup_parse(raw_file, parse_raw_fn=None) -> tuple[bool, str]:
    """验证黑盒解析：raw -> parse_raw -> adapt，产出合法 Event + Verdict（bottleneck 在词表内）。"""
    try:
        from .feedback_adapter import adapt, parse_raw as default_parse_raw
        pr = parse_raw_fn or default_parse_raw
        raw = json.loads(Path(raw_file).read_text(encoding="utf-8"))
        events = pr(raw)
        out = adapt(events)
        return True, f"parse+adapt OK; bottleneck={out.verdict.bottleneck}"
    except Exception as exc:  # noqa: BLE001
        return False, f"parse check failed (which field maps wrong?): {exc}"


def bringup_extcheck(kernel_file) -> tuple[bool, str]:
    """验证签名检查：对一个含 extension 调用的 kernel 跑静态检查。"""
    try:
        from .presim_gate import check_extension_calls
        src = Path(kernel_file).read_text(encoding="utf-8")
        probs = check_extension_calls(src)
        if probs:
            return False, f"extension problems: {probs}"
        return True, "extension check passed (no problems)"
    except Exception as exc:  # noqa: BLE001
        return False, f"extcheck failed: {exc}"


def bringup_all(launcher, parse_raw_fn=None, kernel_file=None, save_dir=None) -> list[tuple[str, bool, str]]:
    """把各单点串起来跑一遍。不替代单点验证——任一环 FAIL 即停在那一环。"""
    results = []
    for name, passed, msg in [
        ("template", *bringup_template()),
        ("launch", *bringup_launch(launcher, kernel_file=kernel_file, save_dir=save_dir)),
        ("parse", *bringup_parse(save_dir / "last_raw.json" if save_dir else kernel_file,
                                 parse_raw_fn=parse_raw_fn)),
    ]:
        results.append((name, passed, msg))
        if not passed:
            break
    return results


def main(argv=None) -> int:
    import argparse
    from .llm_backend import NgaBackend
    from .launch_template import launch as _launch
    ap = argparse.ArgumentParser(prog="control.bringup")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template")
    sub.add_parser("llm")
    bp = sub.add_parser("launch"); bp.add_argument("--kernel")
    pp = sub.add_parser("parse"); pp.add_argument("--raw", required=True)
    ep = sub.add_parser("extcheck"); ep.add_argument("--kernel", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "template":
        ok, msg = bringup_template()
    elif args.cmd == "llm":
        ok, msg = bringup_llm(NgaBackend())
    elif args.cmd == "launch":
        ok, msg = bringup_launch(_launch, kernel_file=args.kernel, save_dir=Path("bringup"))
    elif args.cmd == "parse":
        ok, msg = bringup_parse(args.raw)
    elif args.cmd == "extcheck":
        ok, msg = bringup_extcheck(args.kernel)
    print(("PASS" if ok else "FAIL") + " " + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
