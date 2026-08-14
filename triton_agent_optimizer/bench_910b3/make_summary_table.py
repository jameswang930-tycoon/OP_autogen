#!/usr/bin/env python3
"""读 bench_all 产物 → 生成"工业级 4 方法 vs 我们优化结果"对比表 (.md + 终端).

用法 (服务器, bench_all.py 跑完后):
  python3 make_summary_table.py
输出:
  bench_910b3/outputs/industrial_summary_table.md   (UTF-8 markdown 表格)
  "我们优化结果"列留空 — 你手动填 (脚本永不写这一列)

表格: 列 = eager / compile / cann-fused / fa / 我们优化结果; 行 = 各算子
  数值 = time_us (Event 设备侧端到端 median); 最优加粗 **x**
  回退的 (compile→eager 等) 标 ⚠, 不参与最优; 无该方法的格子 "—"
  ★占位/无效 json (缺 method 字段) 一律跳过, 不冒充数据
"""
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
    "matmul": ["eager", "compile", "cann-fused"],
    "attention_mlp": ["eager", "compile"],
    "matmul_relu": ["eager", "compile", "cann-fused"],
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
    "conv_bias_relu": ["eager", "compile", "cann-fused"],
    "batchnorm2d": ["eager", "compile"],
    "maxpool2d": ["eager", "compile"],
    "conv1d": ["eager", "compile"],
}
MODES = ["eager", "compile", "cann-fused", "fa"]   # 列顺序


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


def main():
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
        # 每算子"工业级最优" (仅真正执行的方法)
        best = None
        for m in modes:
            c = cell.get(m)
            if c and c["ok"] and (best is None or c["t"] < best["t"]):
                best = {"m": m, "t": c["t"]}
        rows.append((op, cell, best))

    def _fmt(c, is_best):
        if c is None:
            return "—"
        s = f"{c['t']:g}"
        if not c["ok"]:
            s += " ⚠回退"
        return f"**{s}**" if is_best else s

    # ── markdown 表格 ──
    lines = [
        "# 工业级基准对比表 (910B3, Event 设备侧端到端 median, 单位 us)",
        "",
        "| 算子 | eager (CANN厂商kernel) | compile (TorchAir融合) | cann-fused (厂商融合) | fa (CANN FlashAttention) | 我们优化结果 |",
        "|---|---|---|---|---|---|",
    ]
    for op, cell, best in rows:
        c0 = best is not None and best["m"] == "eager"
        c1 = best is not None and best["m"] == "compile"
        c2 = best is not None and best["m"] == "cann-fused"
        c3 = best is not None and best["m"] == "fa"
        lines.append(f"| {op} | {_fmt(cell.get('eager'), c0)} | {_fmt(cell.get('compile'), c1)} "
                     f"| {_fmt(cell.get('cann-fused'), c2)} | {_fmt(cell.get('fa'), c3)} |  |")
    lines += [
        "",
        "说明:",
        "- **加粗** = 该算子工业级最优 (仅从真正执行的方法里选; 回退的不参与)",
        "- ⚠回退 = 该方法未真正执行 (如 torchair 不可用时 compile→eager), 数值是该回退实现的重复测量, 不可当该方法的成绩",
        "- \"我们优化结果\"列手动填写; 口径对齐: 我们 verify 的 e2e_event_ns (Event 设备侧, 同尺对比)",
    ]
    out_md = _OUT / "industrial_summary_table.md"
    _OUT.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
