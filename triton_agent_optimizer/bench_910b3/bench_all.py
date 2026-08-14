#!/usr/bin/env python3
"""跑全部算子的工业级基准 → 自动取每算子最优 + 自动出对比表 (★主入口).

═══ 运行命令 (910B 服务器) ═══
  # ── 完整流程 (推荐): 我们侧纯 kernel → 工业级纯 kernel → 出表 ──
  cd triton_agent_optimizer/bench_910b3
  python3 ../test/measure_final_event.py --msprof --force   # ① 我们结果 (纯 kernel) → 自动写回 OUR_RESULTS_US
  python3 bench_all.py --msprof                             # ② 工业级 (纯 kernel 口径) + 自动出对比表
  cat outputs/industrial_summary_table.md                   # ③ 看结果
  # ── 全部算子 (默认 Event 流水化 ÷10 口径) ──
  python3 bench_all.py
  # ── 测量模式切换 ──
  python3 bench_all.py --msprof              # msprof 纯 kernel (Task Duration 求和, 与 verify ns 同源)
  python3 bench_all.py --msprof --measure 200   # 调 msprof 循环次数 (默认 100)
  python3 bench_all.py --pipelined 0         # Event 单次含 host 开销 (默认流水化 ÷10)
  # ── 范围控制 ──
  python3 bench_all.py --op matmul           # 只跑一个算子
  python3 bench_all.py --op transformer_decoder_block --msprof   # 只跑复杂多算子链
  python3 bench_all.py --list                # 列出算子×模式, 不跑
  python3 bench_all.py --clean               # 清理 outputs/ 全部产物
  # ── 单独出对比表 (不重跑测量) ──
  python3 make_summary_table.py              # 读 outputs/*.json + OUR_RESULTS_US → industrial_summary_table.md
  # ── 单算子单模式底层命令 ──
  python3 bench_industrial.py <op> --mode eager|compile|fa [--msprof] [--pipelined N] [--measure N]

═══ 算子清单 (17 基础 + 4 复杂多算子链) ═══
  基础: matmul(MLP链) attention_mlp matmul_relu matmul_transpose rms_norm rms_norm_residual
        layernorm sigmoid softmax vector_add fused_add_mul flash_attention(fa)
        conv2d conv_bias_relu batchnorm2d maxpool2d conv1d
  复杂 (KernelBench L2/L3 风格, 2026-08-14 新增): transformer_decoder_block(LLaMA decoder layer)
        swiglu_mlp(LLaMA FFN) resnet_block(ResNet 残差块) batched_matmul(BMM)
  ★我们结果填写区: 本文件顶部 OUR_RESULTS_US (或 measure_final_event 自动写回)

═══ 输出 ═══
  终端表格: 算子 | 最优模式 | 端到端us | 纯kernelus | 来源json
  bench_910b3/outputs/industrial_summary.json (供轨迹图/报告参考)
  bench_910b3/outputs/industrial_summary_table.md (★对比表: eager/compile/fa/最短耗时/我们结果/对比效果)

═══ 口径 ═══
  每个候选 = bench_industrial.py:
    - Event 设备侧 (默认, 流水化 ÷10): 多窗口 median + 输入轮换破 L2 + 时间预算自适应 (do_bench 同款)
    - msprof 纯 kernel (--msprof): Task Duration 求和 ÷次数, 不含 host launch (与 verify ns 同源)
  每算子最优 = 各 mode 的 time_us(median) 最小值 (eager=CANN厂商kernel, compile=TorchAir融合, fa=CANN-FA),
  且只从"真正执行"的方法里选 (actual_mode==mode; 回退的是别的方法的重复测量, 不顶替).
  ★表格口径自动标注: 标题按 json.method 显示 "msprof 纯 kernel" 或 "Event 设备侧"
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
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))   # ★供 from bench_910b3.bench_common import ...

# 与 input/ 算子对齐; flash_attention 只跑 fa (CANN FA, torch_npu 自带, 无需桥库);
# 其余 eager+compile (compile 的 GE 图融合生成 CANN 融合 kernel)
# ★cann-fused 已移除: 依赖 cann_ops_transformer/npu_ops_transformer 桥库, 服务器装不上 → 回退
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
    # ★复杂多算子链 (KernelBench L2/L3 风格, 2026-08-14 新增)
    "transformer_decoder_block": ["eager", "compile"],
    "swiglu_mlp": ["eager", "compile"],
    "resnet_block": ["eager", "compile"],
    "batched_matmul": ["eager", "compile"],
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ★我们优化结果填写区 — 单位 us (Event 设备侧端到端 median, 与工业级同尺)
#    每个算子跑完优化循环后, 把最后一轮 e2e_event 填到对应行 (None 或 0 = 未填, 表格留空)
#    跑完 bench_all 自动出对比表: outputs/industrial_summary_table.md
# ═══════════════════════════════════════════════════════════════════════════════
OUR_RESULTS_US = {
    "matmul": None,             # 例如 620.5
    "attention_mlp": None,
    "matmul_relu": None,
    "matmul_transpose": None,
    "rms_norm": None,
    "rms_norm_residual": None,
    "layernorm": None,
    "sigmoid": None,
    "softmax": None,
    "vector_add": None,
    "fused_add_mul": None,
    "flash_attention": None,
    "conv2d": None,
    "conv_bias_relu": None,
    "batchnorm2d": None,
    "maxpool2d": None,
    "conv1d": None,
}


def _read_json(op, mode):
    p = _BENCH_DIR / "outputs" / f"industrial_{op}_{mode}_tflops.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_one(op, mode, rep_ms, pipelined, msprof, measure):
    """跑 bench_industrial.py <op> --mode <mode> → 返回结果 dict 或 None.
    ★2026-08-12 同步: bench_industrial 已改 Event 时间预算测量 (--warmup-ms/--rep-ms),
      不再有 --measure 次数参数 — 传 --rep-ms 时间预算 (do_bench 同款自适应次数).
    ★pipelined>1: 流水化 /N (隐藏 host 开销, 近似纯设备时间).
    ★msprof=True: msprof 纯 kernel 求和口径 (与 verify ns 同源)."""
    script = _BENCH_DIR / "bench_industrial.py"
    cmd = [sys.executable or "python3", str(script), op, "--mode", str(mode),
           "--rep-ms", str(rep_ms)]
    if pipelined and pipelined > 1:
        cmd += ["--pipelined", str(pipelined)]
    if msprof:
        cmd += ["--msprof", "--measure", str(measure)]
    # ★防御: 全部转 str — OP_MODES 被写回误伤成数字时, 遍历会拿到 float,
    #   subprocess._fork_exec 报 "expected str... not float"
    cmd = [str(c) for c in cmd]
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
    p.add_argument("--rep-ms", type=int, default=100,
                   help="每个候选测量时间预算 ms (传给 bench_industrial, do_bench 同款按估时长折算次数)")
    p.add_argument("--pipelined", type=int, default=10, metavar="N",
                   help="流水化模式 (传给 bench_industrial): 每窗口连续调用 N 次 /N, "
                        "隐藏 host 下发开销 ~纯设备时间 (与 verify/measure_final_event 同口径); "
                        "0=单次含 host")
    p.add_argument("--msprof", action="store_true",
                   help="★msprof 纯 kernel 口径 (传给 bench_industrial): op_summary Task Duration "
                        "求和 /次数, 不含 host launch — 与我们 verify 的 ns 同源, 小算子可比")
    p.add_argument("--measure", type=int, default=100,
                   help="msprof 模式: app 内部 forward 循环次数 (默认 100)")
    p.add_argument("--list", action="store_true", help="只列出算子x模式, 不跑")
    p.add_argument("--clean", action="store_true",
                   help="清理 bench_910b3/outputs/ 全部产物 (json/txt/msprof临时) 后退出")
    args = p.parse_args()

    # ★清理产物 (bench_910b3/outputs/): 干不干净, 不用手动删垃圾
    if args.clean:
        try:
            from bench_910b3.bench_common import clean_bench_out
            n = clean_bench_out()
            print(f"  ✅ 已清理 bench_910b3/outputs/ 共 {n} 个文件/目录")
        except Exception as e:
            print(f"  ⚠ 清理失败: {e}")
        return

    ops = [args.op] if args.op else list(OP_MODES)
    modes_of = {}
    for op in ops:
        if op not in OP_MODES:
            print(f"⚠ 未知算子 {op} (可用: {list(OP_MODES)})")
            sys.exit(1)
        # ★防御: OP_MODES 值被误伤成非字符串时**真正过滤** (不只警告) —
        #   否则 float mode 会传进 bench_industrial 报 invalid choice
        _clean = [m for m in OP_MODES[op] if isinstance(m, str)]
        _bad = [m for m in OP_MODES[op] if not isinstance(m, str)]
        if _bad:
            print(f"⚠ OP_MODES[{op}] 含非字符串 {_bad} — 已过滤, 只跑 {_clean} "
                  f"(OP_MODES 可能被写回误伤, 请检查文件)")
        modes_of[op] = _clean

    if args.list:
        for op in ops:
            print(f"  {op:20s} → {', '.join(modes_of[op])}")
        return

    # ── 跑 / 收集 (每候选记录 kernels_per_iter + actual_mode, 供融合判定) ──
    results = []
    for op in ops:
        best_t = best_k = None
        best_mode = best_file = None
        cands = []
        for mode in modes_of[op]:
            j = _read_json(op, mode)
            # ★默认重跑(覆盖旧结果); --skip-existing 才对已有**有效** json 跳过(只补缺的)
            #   有效 = 有 method 字段 (无 method 的旧占位/假数据不顶替, 会重跑)
            if args.skip_existing and j and j.get("time_us") and j.get("method"):
                print(f"  ⏭ {op}[{mode}] 已有 json, 跳过 (--skip-existing)")
            else:
                j = _run_one(op, mode, args.rep_ms, args.pipelined, args.msprof, args.measure)
            if j and j.get("time_us"):
                _actual = j.get("actual_mode", mode)   # 实际执行模式 (compile 是否回退)
                cands.append({"mode": mode, "time_us": j["time_us"],
                              "kernel_us": j.get("kernel_time_us"),
                              "kpi": j.get("kernels_per_iter"),          # 每遍 kernel 数 (~1=融合)
                              "actual": _actual})
                t = j["time_us"]
                # ★最优只从"真正执行"的方法里选 (回退的是别的方法的重复测量, 不该顶替成最优)
                if _actual == mode and (best_t is None or t < best_t):
                    best_t, best_k = t, j.get("kernel_time_us")
                    best_mode, best_file = mode, f"industrial_{op}_{mode}_tflops.json"
        results.append({"op": op, "mode": best_mode, "e2e_us": best_t,
                        "kernel_us": best_k, "source": best_file, "cands": cands})

    # ── 终端明细表 (含"是否真正执行" + 融合判定) ──
    def _exec_status(mode, actual):
        """判断该候选是否真正执行了对应方法, 还是回退成了别的."""
        if actual == mode:
            return "✅ 真正执行"
        _fb = actual.split("→")[-1]
        return f"⚠ 未真正执行 (回退 {_fb})"
    print("\n" + "═" * 96)
    print("  工业级基准明细 (kernels/遍: ~1=已融合; 执行状态: 是否真正跑了该方法)   [单位: us]")
    print("═" * 96)
    print(f"  {'算子':<18}{'模式':<13}{'端到端':>10}{'纯kernel':>12}{'kernels/遍':>11}   {'执行状态':<18}   来源")
    print("  " + "-" * 94)
    for r in results:
        n_real = n_fb = 0
        for c in r["cands"]:
            e = f"{c['time_us']:.1f}" if c["time_us"] is not None else "—"
            k = f"{c['kernel_us']:.1f}" if c["kernel_us"] is not None else "—"
            kpi = f"{c['kpi']}" if c["kpi"] is not None else "—"
            st = _exec_status(c["mode"], c["actual"])
            n_real += 1 if st.startswith("✅") else 0
            n_fb += 1 if st.startswith("⚠") else 0
            src = f"industrial_{r['op']}_{c['mode']}_tflops.json"
            print(f"  {r['op']:<18}{c['mode']:<13}{e:>10}{k:>12}{kpi:>11}   {st:<18}   {src}")
        # ★融合判定: eager 的 kernels/遍 vs compile/cann-fused 的 → 是否变小
        _e = next((c for c in r["cands"] if c["mode"] == "eager"), None)
        _cf = next((c for c in r["cands"] if c["mode"] in ("compile", "cann-fused")), None)
        if _cf and _cf.get("actual", _cf["mode"]) == "eager":
            print(f"  {'':<3}└─ ⚠ {_cf['mode']} 未真正执行 (回退 eager, torchair 不可用) → 无融合")
        elif _e and _cf and _e.get("kpi") and _cf.get("kpi") and _cf["kpi"] < _e["kpi"]:
            print(f"  {'':<3}└─ 融合判定: eager {_e['kpi']} → {_cf['mode']} {_cf['kpi']} → 融合✓")
        elif _e and _cf and _e.get("kpi") and _cf.get("kpi"):
            print(f"  {'':<3}└─ 融合判定: eager {_e['kpi']} = {_cf['mode']} {_cf['kpi']} → 未融合✗ (GE 未融合该模式)")
        if n_fb:
            print(f"  {'':<3}└─ 方法执行: {n_real} 个真正执行, {n_fb} 个回退 → 回退的方法结果不可信, 别当最优")
        print("  " + "-" * 94)
    ok = [r for r in results if r["e2e_us"] is not None]
    print(f"  成功 {len(ok)}/{len(results)} 个算子有工业级最优端到端.")
    if len(ok) != len(results):
        miss = [r["op"] for r in results if r["e2e_us"] is None]
        print(f"  ⚠ 缺: {miss} (看上面的 stderr / 确认 TorchAir/CANN-FA 可用)")

    # 写汇总 json
    _out_dir = _BENCH_DIR / "outputs"
    _out_dir.mkdir(parents=True, exist_ok=True)
    (_out_dir / "industrial_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  汇总 → {_out_dir}/industrial_summary.json")

    # ★自动出对比表 (OUR_RESULTS_US 填了就有我们结果+对比效果; 没填则留空)
    try:
        from bench_910b3.make_summary_table import build_table
        our = {k: v for k, v in OUR_RESULTS_US.items() if v}
        lines, out_md = build_table(our)
        print(f"  对比表 → {out_md}")
        print("\n".join(lines))
    except Exception as e:
        print(f"  ⚠ 对比表生成失败: {str(e)[:100]} (可单独运行 make_summary_table.py)")


if __name__ == "__main__":
    main()
