#!/usr/bin/env python3
"""前置 BLOCK 扫描 (D1) — 对 input/<op>/kernel_op.py 扫 BLOCK 候选, msprof 取最快, 写回 config 区.

★为什么: 分块是"乘性地基"。block 差 → MTE 搬运效率低 → 算力/访存利用率字段全失真 →
  planner 在错误的判断上做所有优化。先固定一个经过实测的好块, 后续每层诊断才可信。
  (LLM 一次猜的块 ≠ 最优; 实测扫描才接近最优 — 解决"分块差几十倍"的根因)

用法 (910B3, 优化主流程前先跑一次):
  conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 sweep_blocks.py input/matmul            # 扫 matmul 的 BLOCK 候选, 最快写回
  python3 sweep_blocks.py input/conv2d --quick    # 只扫前 4 个候选 (省时间)
  # 或直接: python3 main.py input/matmul --sweep-blocks --max-rounds 15  (main 内置)

候选: 每 op 预定义 **L0 合法** 组合 (fp32: L0A/B=BM·BK·4≤64KB, L0C=BM·BN·4≤128KB; 全 16 倍数).
正确性: 每个候选都必须 MATMUL_VERIFY PASS 才纳入比较 (防"更快但算错").
输出: outputs/<op>/block_sweep/ 候选对比 + 最优 BLOCK 写回 kernel_op.py config 区.
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ── per-op 扫描策略: 变量名 + L0 合法候选 (fp32) ──
#   matmul 型 (BLOCK_M/N/K): 覆盖 大M/大N/大K/平衡, 全部通过 L0A/B≤64KB、L0C≤128KB 校验
_MNK_CANDS = [
    (64, 64, 32), (128, 128, 64), (256, 64, 64), (64, 256, 64),
    (128, 64, 128), (256, 128, 64), (128, 128, 128), (512, 64, 32),
]
SWEEP = {
    "matmul": {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"), "cands": _MNK_CANDS},
    "attention_mlp": {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"), "cands": _MNK_CANDS},
    "flash_attention": {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"), "cands": _MNK_CANDS},
    "conv2d": {"vars": ("BLOCK_K", "BLOCK_OW"), "cands": [(32, 64), (64, 64), (32, 128), (64, 128), (128, 64)]},
    "conv_bias_relu": {"vars": ("BLOCK_K", "BLOCK_OW"), "cands": [(32, 64), (64, 64), (32, 128), (64, 128), (128, 64)]},
    "rms_norm": None,   # 行级 kernel, BLOCK_N 由 N 推导 (2 幂), 无自由分块参数 → 不扫
}


def _read_current_block(code: str, varnames):
    """读 config 区当前 BLOCK 值 (tuple). 找不到返回 None."""
    if varnames == ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        m = re.search(r"BLOCK_M\s*,\s*BLOCK_N\s*,\s*BLOCK_K\s*=\s*([\d, ]+)", code)
        if m:
            vals = [int(x) for x in re.split(r"[,，\s]+", m.group(1).strip()) if x.strip().isdigit()]
            if len(vals) == 3:
                return tuple(vals)
    else:
        vals = []
        for var in varnames:
            m = re.search(rf"{var}\s*=\s*(\d+)", code)
            if not m:
                return None
            vals.append(int(m.group(1)))
        return tuple(vals)
    return None


def _apply_block(code: str, varnames, vals) -> str:
    """把 config 区 BLOCK 赋值替换成 vals (其余不动)."""
    if varnames == ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        m = re.search(r"BLOCK_M\s*,\s*BLOCK_N\s*,\s*BLOCK_K\s*=\s*[\d, ]+", code)
        if not m:
            raise ValueError("源码缺 BLOCK_M, BLOCK_N, BLOCK_K 赋值 (sweep 无法替换)")
        return code[:m.start()] + f"BLOCK_M, BLOCK_N, BLOCK_K = {vals[0]}, {vals[1]}, {vals[2]}" + code[m.end():]
    for var, val in zip(varnames, vals):
        m = re.search(rf"{var}\s*=\s*\d+", code)
        if not m:
            raise ValueError(f"源码缺 {var} 赋值 (sweep 无法替换)")
        code = code[:m.start()] + f"{var} = {val}" + code[m.end():]
    return code


def sweep(op_dir: Path, quick: bool = False) -> dict:
    """对 input/<op> 扫 BLOCK → 返回 {best_block, results:[...]} 并写回最优."""
    op = op_dir.name
    cfg = SWEEP.get(op)
    kernel = op_dir / "kernel_op.py"
    if cfg is None:
        print(f"[sweep] {op}: 无自由分块参数 (rms_norm 行级), 跳过")
        return {"skipped": True}
    if not kernel.exists():
        print(f"[sweep] ❌ 缺 {kernel}")
        return {"error": "no kernel_op.py"}

    code = kernel.read_text(encoding="utf-8")
    cur = _read_current_block(code, cfg["vars"])
    if cur is None:
        print(f"[sweep] ❌ 读不到 {cfg['vars']} 当前值, 跳过")
        return {"error": "cannot read current block"}
    print(f"[sweep] {op}: 当前 BLOCK {cfg['vars']}={cur}, 扫 {len(cfg['cands'])} 个候选")

    # 候选去重 (跳过与当前相同的) + quick 截断
    cands = [c for c in cfg["cands"] if c != cur]
    if quick:
        cands = cands[:4]

    from agents.verifier import verify_end_to_end
    out_root = _PROJECT / "outputs" / op / "block_sweep"
    results = []
    best = None
    for vals in cands:
        # 候选代码写到临时 round dir (不动源文件, 测完统一写回最优)
        rd = out_root / "_".join(str(v) for v in vals)
        rd.mkdir(parents=True, exist_ok=True)
        cand_code = _apply_block(code, cfg["vars"], vals)
        (rd / "kernel_op.py").write_text(cand_code, encoding="utf-8")
        print(f"  ─ 试 {cfg['vars']}={vals} ...", flush=True)
        try:
            v = verify_end_to_end(rd / "kernel_op.py", rd, None, num_kernels=None)
            if v.get("ok") and v.get("ns"):
                results.append({"block": list(vals), "ns": v["ns"], "ok": True,
                                "speedup_vs_cur": None})
                print(f"      ✅ {vals}: {v['ns']:.0f}ns", flush=True)
            else:
                results.append({"block": list(vals), "ns": None, "ok": False,
                                "error": str(v.get("error", ""))[:120]})
                print(f"      ❌ {vals}: {str(v.get('error',''))[:100]}", flush=True)
        except Exception as e:
            results.append({"block": list(vals), "ns": None, "ok": False, "error": str(e)[:120]})
            print(f"      ❌ {vals}: {str(e)[:100]}", flush=True)

    valid = [r for r in results if r.get("ok") and r.get("ns")]
    if not valid:
        print(f"[sweep] 无有效候选 (当前 {cur} 保留)")
        _write_report(out_root, cur, results)
        return {"best_block": cur, "results": results, "unchanged": True}

    # 最优 = ns 最小
    best = min(valid, key=lambda r: r["ns"])
    cur_ns = valid[0]["ns"] if cur in [tuple(r["block"]) for r in valid] else None
    if cur_ns is None:
        # 当前块没测过 → 单独测一次作对比 (保证"写回最优"有依据)
        rd = out_root / "current"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "kernel_op.py").write_text(code, encoding="utf-8")
        v = verify_end_to_end(rd / "kernel_op.py", rd, None, num_kernels=None)
        if v.get("ok") and v.get("ns"):
            cur_ns = v["ns"]
            results.insert(0, {"block": list(cur), "ns": cur_ns, "ok": True,
                               "speedup_vs_cur": 1.0})
    for r in results:
        if r.get("ns") and cur_ns:
            r["speedup_vs_cur"] = round(cur_ns / r["ns"], 3)

    # 写回最优 (若比当前快)
    if best["ns"] and cur_ns and best["ns"] < cur_ns:
        new_code = _apply_block(kernel.read_text(encoding="utf-8"), cfg["vars"], best["block"])
        kernel.write_text(new_code, encoding="utf-8")
        print(f"\n[sweep] ✅ 最优 {cfg['vars']}={best['block']} ({best['ns']:.0f}ns, "
              f"vs 当前 {cur_ns:.0f}ns = {cur_ns/best['ns']:.2f}x) → 已写回 {kernel}")
        _write_report(out_root, best["block"], results)
        return {"best_block": best["block"], "results": results,
                "improvement_x": round(cur_ns / best["ns"], 3)}
    print(f"\n[sweep] 当前 {cur} ({cur_ns:.0f}ns) 已是最优, 保留")
    _write_report(out_root, cur, results)
    return {"best_block": cur, "results": results, "unchanged": True}


def _write_report(out_root: Path, best_block, results):
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep_result.json").write_text(json.dumps({
        "measured_at": datetime.now().isoformat(),
        "best_block": list(best_block),
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_root / "sweep_result.txt", "w", encoding="utf-8") as f:
        f.write("══ BLOCK 扫描结果 (ns 越小越快) ══\n")
        for r in sorted(results, key=lambda x: x.get("ns") or 9e18):
            blk = r.get("block")
            if r.get("ok") and r.get("ns"):
                f.write(f"  {blk}: {r['ns']:.0f}ns"
                        + (f"  ({r.get('speedup_vs_cur', 1):.2f}x vs 当前)" if r.get("speedup_vs_cur") else "")
                        + "\n")
            else:
                f.write(f"  {blk}: FAIL {r.get('error', '')[:60]}\n")
    print(f"[sweep] 报告 → {out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="前置 BLOCK 扫描")
    p.add_argument("op_dir", type=str)
    p.add_argument("--quick", action="store_true", help="只扫前 4 个候选")
    args = p.parse_args()
    sys.exit(0 if sweep(Path(args.op_dir), args.quick) else 1)
