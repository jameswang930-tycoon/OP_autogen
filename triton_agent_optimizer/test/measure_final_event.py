#!/usr/bin/env python3
"""测量"最终优化完算子"的 Event 端到端耗时 — 严格对齐工业级测量方法.

══════ 用法 (910B 服务器, 有 NPU) ══════
  1. 把每个算子最终优化完的 kernel_op.py 路径填到下面 OP_PATHS (字典: 算子名 → 路径)
     ★路径: 绝对路径 或 仓库相对路径 (相对 triton_agent_optimizer/, 如 input/matmul/kernel_op.py)
  2. python3 measure_final_event.py              # 全部测一遍 → 终端表格 + 自动写回
     python3 measure_final_event.py --rep 50     # 调测量次数 (默认 30)
     python3 measure_final_event.py --warmup 5   # 调预热次数 (默认 10)
     python3 measure_final_event.py --no-write   # 只测不写回

══════ 自动写回 ══════
  测量成功的算子 → 结果自动写入 bench_910b3/bench_all.py 顶部的 OUR_RESULTS_US
  ("matmul": None → "matmul": 620.5), 之后跑 bench_all.py 出对比表直接带上我们结果

══════ 测量方法 (与 bench_common.measure_event / 工业级 bench_all 同构) ══════
  - Event 设备侧计时: 每窗口 ev_s.record() → 一次完整 kernel launch 链 → ev_e.record()
    ★每窗口只测 1 次完整调用 (不 ÷LOOP) — 与工业级每次 fn(i) 完全同构, host 下发开销全额计入
  - 多窗口 median: N 个独立 Event 对, 最后 sync, 取 median (抗单次抖动)
  - 破 L2: 每窗口前重建输入张量 (新地址; Ascend 无清 L2 API, 重建 = n_buf 轮换同效)
  - warmup: W 次完整链路 (JIT 编译/冷 cache 消化, 不计时)
  ★口径声明: 与工业级同尺 (Event 设备侧端到端), 可直接与 bench_all 结果对比
  ★注意: conv2d/conv_bias_relu 的 unfold 展开张量是派生分配, 不在重建范围 (工作集小, 影响有限)

══════ 输出 ══════
  FINAL_EVENT_E2E_US (单次完整链路 median, us) → 终端汇总表 → 写回 OUR_RESULTS_US
"""
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ★项目根: test/ 的上一级 (相对路径从这里解析)
_ROOT = Path(__file__).resolve().parent.parent
# ★bench_all.py 位置 (写回 OUR_RESULTS_US 的目标)
_BENCH_ALL = _ROOT / "bench_910b3" / "bench_all.py"

# ═══════════════════════════════════════════════════════════════════════════════
#  ★填写区: 最终优化完算子的 kernel_op.py 路径 (绝对路径 或 相对 triton_agent_optimizer/)
# ═══════════════════════════════════════════════════════════════════════════════
OP_PATHS = {
    # "matmul":            "input/matmul/kernel_op.py",
    # "attention_mlp":     "input/attention_mlp/kernel_op.py",
    # "matmul_relu":       "input/matmul_relu/kernel_op.py",
    # "matmul_transpose":  "input/matmul_transpose/kernel_op.py",
    # "rms_norm":          "input/rms_norm/kernel_op.py",
    # "rms_norm_residual": "input/rms_norm_residual/kernel_op.py",
    # "layernorm":         "input/layernorm/kernel_op.py",
    # "sigmoid":           "input/sigmoid/kernel_op.py",
    # "softmax":           "input/softmax/kernel_op.py",
    # "vector_add":        "input/vector_add/kernel_op.py",
    # "fused_add_mul":     "input/fused_add_mul/kernel_op.py",
    # "flash_attention":   "input/flash_attention/kernel_op.py",
    # "conv2d":            "input/conv2d/kernel_op.py",
    # "conv_bias_relu":    "input/conv_bias_relu/kernel_op.py",
    # "batchnorm2d":       "input/batchnorm2d/kernel_op.py",
    # "maxpool2d":         "input/maxpool2d/kernel_op.py",
    # "conv1d":            "input/conv1d/kernel_op.py",
}

_WORK = Path(os.environ.get("FINAL_EVENT_WORK", Path(__file__).resolve().parent / "event_work"))
_TIMEOUT = int(os.environ.get("FINAL_EVENT_TIMEOUT", "1800"))


def _inject(src: str, warmup: int, reps: int) -> str:
    """在 kernel_op.py 的 for LOOP 循环处注入"严格单次" Event 计时块 (窗口包 1 次完整链路).
    失败返回 "" (找不到标准循环 → 调用方跳过该算子)."""
    lines = src.splitlines(keepends=True)
    for_idx = None
    base_indent = 0
    for i, ln in enumerate(lines):
        m = re.match(r"(\s*)for\s+\w+\s+in\s+range\(\s*LOOP\s*\)\s*:", ln)
        if m:
            for_idx, base_indent = i, len(m.group(1))
            break
    if for_idx is None:
        return ""
    # 提取循环体 (缩进 > base_indent)
    body_start = for_idx + 1
    body_end = body_start
    while body_end < len(lines):
        ln = lines[body_end]
        if ln.strip() == "":
            body_end += 1
            continue
        if len(re.match(r"\s*", ln).group()) > base_indent:
            body_end += 1
        else:
            break
    body = "".join(lines[body_start:body_end])
    if not body.strip():
        return ""
    body_deep = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(keepends=True))
    ind = " " * base_indent
    # 循环前的 torch 直接分配行 (重建破 L2)
    alloc_lines = []
    for _ln in lines[:for_idx]:
        m = re.match(r"^\s{4,}(\w+)\s*=\s*torch\.(randn?|rand|empty|zeros|ones)\(", _ln)
        if m:
            alloc_lines.append((_ln.rstrip("\n"), m.group(1)))
    # ★缩进: 注入块的 for _r 在 ind+4, 体内语句在 ind+8 (12 空格, 与 _ev_s 同级)
    alloc_block = "".join((ind + "        " + a.lstrip() + "\n") for a, _ in alloc_lines)
    names = ", ".join(n for _, n in alloc_lines)
    keep_block = f"{ind}        _keep.append(({names}))\n" if names else ""
    ind = " " * base_indent
    inject = (
        f"{ind}# ★严格单次 Event 计时 (注入; FINAL_EVENT_TIME 触发)\n"
        f"{ind}if os.environ.get('FINAL_EVENT_TIME'):\n"
        f"{ind}    _W = int(os.environ.get('FINAL_EVENT_WARMUP', '{warmup}'))\n"
        f"{ind}    for _ in range(_W):\n{body_deep}"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _REPS = int(os.environ.get('FINAL_EVENT_REPS', '{reps}'))\n"
        f"{ind}    _ts = []\n"
        f"{ind}    _keep = []\n"
        f"{ind}    for _r in range(_REPS):\n"
        f"{alloc_block}"
        f"{keep_block}"
        f"{ind}        _ev_s = torch.npu.Event(enable_timing=True)\n"
        f"{ind}        _ev_e = torch.npu.Event(enable_timing=True)\n"
        f"{ind}        _ev_s.record()\n"
        f"{body_deep}"
        f"{ind}        _ev_e.record()\n"
        f"{ind}        _ts.append(_ev_s.elapsed_time(_ev_e))\n"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _ts.sort()\n"
        f"{ind}    print('FINAL_EVENT_E2E_US:%.2f' % _ts[len(_ts) // 2])\n"
        f"{ind}    raise SystemExit(0)\n"
    )
    out = "".join(lines[:for_idx]) + inject + "".join(lines[for_idx:])
    if not re.search(r"^\s*import\s+os\b", out, re.M):
        out = "import os\n" + out
    return out


def _measure(op: str, path: Path, warmup: int, reps: int):
    """对单个算子跑 Event 计时 → 返回 (median_us, err) 或 (None, err)."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"读不了: {e}"
    injected = _inject(src, warmup, reps)
    if not injected:
        return None, "找不到标准 for LOOP 循环"
    evt_kernel = _WORK / f"{op}_event.py"
    _WORK.mkdir(parents=True, exist_ok=True)
    evt_kernel.write_text(injected, encoding="utf-8")
    env = dict(os.environ, FINAL_EVENT_TIME="1", FINAL_EVENT_WARMUP=str(warmup),
               FINAL_EVENT_REPS=str(reps), KERNEL_LOOP="1")
    r = subprocess.run([sys.executable or "python3", str(evt_kernel)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="backslashreplace", timeout=_TIMEOUT, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"FINAL_EVENT_E2E_US:([\d.]+)", out)
    if not m:
        return None, out.strip().splitlines()[-1][:150] if out.strip() else "无输出"
    return float(m.group(1)), None


def _resolve(path_str: str) -> Path:
    """绝对路径直接用; 相对路径按仓库根 (triton_agent_optimizer/) 解析."""
    p = Path(path_str)
    return p if p.is_absolute() else _ROOT / p


def _write_back(rows) -> int:
    """测量成功的算子 → 写回 bench_all.py 的 OUR_RESULTS_US ("op": None → "op": 值).
    返回写回个数."""
    if not _BENCH_ALL.exists():
        print(f"  ⚠ 找不到 {_BENCH_ALL}, 跳过写回")
        return 0
    src = _BENCH_ALL.read_text(encoding="utf-8")
    updated = []
    for op, us, _err in rows:
        if us is None:
            continue
        pat = re.compile(rf'("{re.escape(op)}"\s*:\s*)None')
        new, n = pat.subn(rf"\g<1>{us:g}", src, count=1)
        if n:
            src = new
            updated.append(op)
    if updated:
        _BENCH_ALL.write_text(src, encoding="utf-8")
        print(f"  ✅ 已写回 {_BENCH_ALL} OUR_RESULTS_US: {', '.join(updated)}")
    return len(updated)


def main():
    import argparse
    p = argparse.ArgumentParser(description="最终算子 Event 端到端耗时 (工业级同构方法)")
    p.add_argument("--rep", type=int, default=30, help="测量 Event 对个数 (默认 30)")
    p.add_argument("--warmup", type=int, default=10, help="预热完整链路次数 (默认 10)")
    p.add_argument("--no-write", action="store_true", help="只测不写回 bench_all.py")
    args = p.parse_args()

    if not OP_PATHS:
        print("❌ 先在脚本顶部 OP_PATHS 填最终算子的 kernel_op.py 路径 "
              "(绝对路径 或 相对 triton_agent_optimizer/ 的相对路径)")
        sys.exit(1)

    print(f"══ 最终算子 Event 测量 (严格单次链路, warmup={args.warmup}, reps={args.rep}) ══\n")
    rows = []
    for op, p_str in OP_PATHS.items():
        path = _resolve(p_str)
        print(f"⏱  {op}: {path}")
        us, err = _measure(op, path, args.warmup, args.rep)
        if us is not None:
            print(f"   ✅ FINAL_EVENT_E2E_US = {us:.2f} (median of {args.rep})")
        else:
            print(f"   ❌ {err}")
        rows.append((op, us, err))

    print("\n" + "═" * 72)
    print(f"  最终算子对比 (单位 us, Event 设备侧单次完整链路 median)   [工业级同尺]")
    print("═" * 72)
    print(f"  {'算子':<20}{'e2e_event(us)':>14}   状态")
    print("  " + "-" * 70)
    for op, us, err in rows:
        if us is not None:
            print(f"  {op:<20}{us:>12.2f}   ✅")
        else:
            print(f"  {op:<20}{'—':>12}   ❌ {err}")
    ok = sum(1 for _, us, _ in rows if us is not None)
    print(f"\n  成功 {ok}/{len(rows)} 个算子.")

    if not args.no_write and ok:
        n = _write_back(rows)
        if n:
            print("  下次跑 bench_all.py 出的对比表将自动带上我们结果列.")


if __name__ == "__main__":
    main()
