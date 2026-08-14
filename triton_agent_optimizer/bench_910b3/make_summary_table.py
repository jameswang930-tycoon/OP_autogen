#!/usr/bin/env python3
"""读 bench_all 产物 → 生成"工业级基准 vs 我们优化结果"对比表 (.md + 终端).

用法 (服务器, bench_all.py 跑完后):
  ① 提前填我们结果到 outputs/our_results.txt (每行 "op=us", # 注释), 然后:
  python3 make_summary_table.py                                   # 自动读 our_results.txt
  ② 或命令行/其他文件:
  python3 make_summary_table.py --our matmul=620.5 conv2d=310.2   # 命令行覆盖 (可多次)
  python3 make_summary_table.py --our-file /path/our.txt          # 指定其他文件
输出:
  bench_910b3/outputs/industrial_summary_table.md   (UTF-8 markdown 表格)

表格 7 列: 算子 | eager (CANN厂商kernel) | compile (TorchAir融合) | fa (CANN FlashAttention) | 最短耗时 | 我们结果 | 对比效果
  数值 = time_us (Event 设备侧端到端 median)
  最短耗时 = 各方法中真正执行的最小值 (fa 仅 flash_attention 有, 它是唯一基准)
  回退的 (compile→eager 等) 标 ⚠, 不参与最短; 无该方法的格子 "—"
  对比效果 = 最短耗时 / 我们结果 (>1 = 我们更快; <1 = 慢于工业级最优)
  ★占位/无效 json (缺 method 字段) 一律跳过, 不冒充数据
"""
import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BENCH_DIR = Path(__file__).resolve().parent
_OUT = _BENCH_DIR / "outputs"

# 与 bench_all.py OP_MODES 对齐 (算子 → 测量模式)
OP_MODES = {
    "matmul": ["eager", "compile"],
    "attention_mlp": ["eager", "compile"],
    "matmul_relu": ["eager", "compile"],
    "matmul_transpose": ["eager", "compile"],
    "rms_norm": ["eager", "compile"],
    "rms_norm_residual": ["eager", "compile"],
    "layernorm": ["eager", "compile"],
    "sigmoid": ["eager", "compile"],
    "softmax": ["eager", "compile"],
    "vector_add": ["eager", "compile"],
    "fused_add_mul": ["eager", "compile"],
    "flash_attention": ["fa"],
    "conv2d": ["eager", "compile"],
    "conv_bias_relu": ["eager", "compile"],
    "batchnorm2d": ["eager", "compile"],
    "maxpool2d": ["eager", "compile"],
    "conv1d": ["eager", "compile"],
}
MODES = ["eager", "compile", "fa"]   # 表格可显示的方法 (fa 仅 flash_attention)


def _read(op, mode):
    p = _OUT / f"industrial_{op}_{mode}_tflops.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not j.get("method"):          # ★占位/无效 json (只有 time_us+actual_mode) → 不算数
        return None
    return j


def _parse_our(args_our, file_our):
    """解析我们结果 (单位 us): 优先级 = 命令行 --our > --our-file 指定文件 > 默认 outputs/our_results.txt.
    文件格式: 每行 op=数值 (如 matmul=620.5), # 开头为注释. 非法项忽略."""
    out = {}
    for kv in (args_our or []):
        if "=" not in kv:
            continue
        op, _, v = kv.partition("=")
        try:
            out[op.strip()] = float(v.strip())
        except ValueError:
            continue
    files = [p for p in (file_our, str(_OUT / "our_results.txt")) if p]
    for f in files:
        try:
            for ln in Path(f).read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                op, _, v = ln.partition("=")
                try:
                    out.setdefault(op.strip(), float(v.strip()))
                except ValueError:
                    continue
        except OSError:
            continue
    return out


def main():
    p = argparse.ArgumentParser(description="工业级基准 vs 我们结果对比表")
    p.add_argument("--our", action="append", default=None,
                   help='我们结果, 可多次: --our matmul=620.5 --our conv2d=310.2 (单位 us)')
    p.add_argument("--our-file", type=str, default=None,
                   help="从文件读我们结果 (每行 op=us, # 开头为注释)")
    args = p.parse_args()
    our = _parse_our(args.our, args.our_file)
    lines, out_md = build_table(our)
    print(f"→ {out_md}")
    print("\n".join(lines))


def _fmt(c, is_best):
    if c is None:
        return "—"
    s = f"{c['t']:g}"
    if not c["ok"]:
        s += " ⚠回退"
    return f"**{s}**" if is_best else s


def build_table(our: dict) -> tuple:
    """读 bench_all 产物 json → 生成对比表. 返回 (lines: list[str], out_md: Path).
    our: {算子名: 我们耗时 us}; 未提供的算子留空 (手动填)."""
    rows = []
    for op, modes in OP_MODES.items():
        cell = {}
        for mode in modes:
            j = _read(op, mode)
            if j is None or j.get("time_us") is None:
                cell[mode] = None
                continue
            actual = j.get("actual_mode", mode)
            cell[mode] = {"t": round(j["time_us"], 1), "ok": actual == mode, "actual": actual}
        # 每算子"最短耗时" (仅真正执行的方法)
        best = None
        for m in modes:
            c = cell.get(m)
            if c and c["ok"] and (best is None or c["t"] < best["t"]):
                best = {"m": m, "t": c["t"]}
        rows.append((op, cell, best))

    # ── markdown 表格 (7 列) ──
    lines = [
        "# 工业级基准对比表 (910B3, Event 设备侧端到端 median, 单位 us)",
        "",
        "| 算子 | eager (CANN厂商kernel) | compile (TorchAir融合) | fa (CANN FlashAttention) | 最短耗时 | 我们结果 | 对比效果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for op, cell, best in rows:
        c0 = best is not None and best["m"] == "eager"
        c1 = best is not None and best["m"] == "compile"
        c3 = best is not None and best["m"] == "fa"
        best_t = f"{best['t']:g}" if best else "—"
        o = our.get(op)
        our_cell = "" if o is None else f"{o:g}"
        if o is not None and best is not None and o > 0:
            ratio = best["t"] / o
            effect = f"{ratio:.2f}x" if ratio >= 1 else f"{ratio:.2f}x(慢)"
        else:
            effect = "—"
        lines.append(f"| {op} | {_fmt(cell.get('eager'), c0)} | {_fmt(cell.get('compile'), c1)} "
                     f"| {_fmt(cell.get('fa'), c3)} "
                     f"| {best_t} | {our_cell} | {effect} |")
    lines += [
        "",
        "说明:",
        "- **加粗** = 该算子工业级最短; 最短耗时 = 该值 (fa 仅 flash_attention, 为该算子唯一工业级基准)",
        "- ⚠回退 = 该方法未真正执行 (如 torchair 不可用时 compile→eager), 数值是该回退实现的重复测量, 不可当该方法的成绩",
        "- 对比效果 = 最短耗时 ÷ 我们结果: >1 = 我们比工业级最优快 (1.36x = 快 36%), <1 = 慢",
        "- \"我们结果\"列: 填法一 bench_all.py 顶部 OUR_RESULTS_US 提前填好自动带; "
        "填法二 outputs/our_results.txt (每行 op=us) / `--our matmul=620.5`",
        "- 口径对齐: 我们 verify 的 e2e_event_ns (Event 设备侧, 同尺对比)",
    ]
    out_md = _OUT / "industrial_summary_table.md"
    _OUT.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return lines, out_md


def main():
    p = argparse.ArgumentParser(description="工业级基准 vs 我们结果对比表")
    p.add_argument("--our", action="append", default=None,
                   help='我们结果, 可多次: --our matmul=620.5 --our conv2d=310.2 (单位 us)')
    p.add_argument("--our-file", type=str, default=None,
                   help="从文件读我们结果 (每行 op=us, # 开头为注释)")
    args = p.parse_args()
    our = _parse_our(args.our, args.our_file)
    lines, out_md = build_table(our)
    print(f"→ {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
