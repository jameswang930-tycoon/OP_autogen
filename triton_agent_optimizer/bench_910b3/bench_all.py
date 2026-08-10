#!/usr/bin/env python3
"""跑全部算子的工业级基准 → 自动取每算子最优 (min 端到端) + 最优来源 → 终端打印汇总表.

═══ 用法 (910B 服务器) ═══
  python3 bench_all.py                 # 跑全部算子 (每个: eager+compile, flash: fa) — 需几分钟~几十分钟
  python3 bench_all.py --op matmul     # 只跑一个算子
  python3 bench_all.py --skip-existing # 已有 json 的不重跑 (缺的才跑)
  python3 bench_all.py --measure 30    # 每个候选测 30 次 (默认 30)

═══ 输出 ═══
  终端表格: 算子 | 最优模式 | 端到端us | 纯kernelus | 来源json
  并写 bench_910b3/industrial_summary.json (供轨迹图/报告参考)

═══ 口径 ═══
  每个候选 = bench_industrial.py (一次 msprof, 端到端=Σ全部含框架, 纯kernel=Σ非aclnn, ÷measure).
  每算子最优 = 各 mode 的 time_us 最小值 (eager=CANN厂商kernel, compile=TorchAir融合, fa=CANN-FA).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BENCH_DIR = Path(__file__).resolve().parent
_PROJ = _BENCH_DIR.parent

# 与 input/ 算子对齐; flash_attention 只跑 fa (CANN FA);
# matmul/matmul_relu/conv_bias_relu 有 CANN 融合算子 (aclnnFusedMatmul/FusedConvBiasRelu) → 加 cann-fused
# 其余 eager+compile (compile 的 GE 图融合也会生成 CANN 融合 kernel)
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
}


def _read_json(op, mode):
    p = _BENCH_DIR / f"industrial_{op}_{mode}_tflops.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_one(op, mode, measure):
    """跑 bench_industrial.py <op> --mode <mode> → 返回结果 dict 或 None."""
    script = _BENCH_DIR / "bench_industrial.py"
    cmd = [sys.executable or "python3", str(script), op, "--mode", mode,
           "--measure", str(measure)]
    print(f"\n══ {op} [{mode}] ══", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                       encoding="utf-8", errors="backslashreplace")
    # bench_industrial 外层已写 json; 打印尾部输出
    tail = (r.stdout or "").strip().splitlines()
    for line in tail[-3:]:
        print(f"  {line}", flush=True)
    if r.returncode != 0 and (r.stderr or ""):
        print(f"  ⚠ stderr: {(r.stderr or '').strip()[-300:]}", flush=True)
    return _read_json(op, mode)


def main():
    p = argparse.ArgumentParser(description="全部算子工业级基准 + 自动取最优")
    p.add_argument("--op", type=str, default=None, help="只跑指定算子 (缺省=全部)")
    p.add_argument("--skip-existing", action="store_true",
                   help="已有 json 的模式不重跑 (缺的才跑)")
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--list", action="store_true", help="只列出算子×模式, 不跑")
    args = p.parse_args()

    ops = [args.op] if args.op else list(OP_MODES)
    for op in ops:
        if op not in OP_MODES:
            print(f"⚠ 未知算子 {op} (可用: {list(OP_MODES)})")
            sys.exit(1)

    if args.list:
        for op in ops:
            print(f"  {op:20s} → {', '.join(OP_MODES[op])}")
        return

    # ── 跑 / 收集 ──
    results = []
    for op in ops:
        best_t = best_k = None
        best_mode = best_file = None
        for mode in OP_MODES[op]:
            j = _read_json(op, mode)
            if j is None or not args.skip_existing:
                if j is None or not j.get("time_us"):
                    j = _run_one(op, mode, args.measure)
                elif args.skip_existing:
                    print(f"  ⏭ {op}[{mode}] 已有 json, 跳过 (--skip-existing)")
            if j and j.get("time_us"):
                t = j["time_us"]
                if best_t is None or t < best_t:
                    best_t, best_k = t, j.get("kernel_time_us")
                    best_mode, best_file = mode, f"industrial_{op}_{mode}_tflops.json"
        results.append({"op": op, "mode": best_mode, "e2e_us": best_t,
                        "kernel_us": best_k, "source": best_file})

    # ── 终端汇总表 ──
    print("\n" + "═" * 78)
    print("  工业级基准汇总 (每算子取最优 = 端到端最小者)   [单位: us]")
    print("═" * 78)
    print(f"  {'算子':<20}{'最优模式':<12}{'端到端us':>12}{'纯kernelus':>14}   来源")
    print("  " + "-" * 76)
    for r in results:
        e = f"{r['e2e_us']:.1f}" if r["e2e_us"] is not None else "—"
        k = f"{r['kernel_us']:.1f}" if r["kernel_us"] is not None else "—"
        src = r["source"] or "(无)"
        print(f"  {r['op']:<20}{r['mode'] or '—':<12}{e:>12}{k:>14}   {src}")
    print("═" * 78)
    ok = [r for r in results if r["e2e_us"] is not None]
    print(f"  成功 {len(ok)}/{len(results)} 个算子有工业级最优端到端.")
    if len(ok) != len(results):
        miss = [r["op"] for r in results if r["e2e_us"] is None]
        print(f"  ⚠ 缺: {miss} (看上面的 stderr / 确认 TorchAir/CANN-FA 可用)")

    # 写汇总 json
    (_BENCH_DIR / "industrial_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  汇总 → {_BENCH_DIR}/industrial_summary.json")


if __name__ == "__main__":
    main()
