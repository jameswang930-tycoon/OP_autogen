#!/usr/bin/env python3
"""测量"最终优化完算子"的耗时 — 与工业级同尺 (Event 流水化 或 msprof 纯 kernel).

══════ 用法 (910B 服务器, 有 NPU) ══════
  1. 把每个算子最终优化完的 kernel_op.py 路径填到下面 OP_PATHS (字典: 算子名 → 路径)
     ★路径: 绝对路径 或 仓库相对路径 (相对 triton_agent_optimizer/, 如 input/matmul/kernel_op.py)
  2. 执行命令:
     # ── 纯 kernel 口径 (推荐: 不含 host launch, 与工业级 --msprof 同源可比) ──
     python3 measure_final_event.py --msprof --force
     #   (msprof 包原文件, 目标 kernel Task Duration 求和 /loop; --loop N 调循环数 默认100)
     # ── Event 口径 (默认流水化 P=10, 与 verify/bench_all 同口径) ──
     python3 measure_final_event.py
     python3 measure_final_event.py --pipelined 10     # 流水化 P=N (默认 10)
     python3 measure_final_event.py --pipelined 0      # 单次含 host 开销
     python3 measure_final_event.py --rep 50           # 调测量次数 (默认 30)
     python3 measure_final_event.py --warmup 5         # 调预热次数 (默认 10)
     # ── 写回控制 ──
     python3 measure_final_event.py --no-write         # 只测不写回
     python3 measure_final_event.py --force            # 写回时覆盖已有值
     #   ★默认只替换 None 占位 (防覆盖手动/旧值); 换测量口径重测时必加 --force,
     #     否则表格里还是旧口径的值!

══════ 自动写回 ══════
  测量成功的算子 → 结果自动写入 bench_910b3/bench_all.py 顶部的 OUR_RESULTS_US
  ("matmul": None → "matmul": 620.5), 之后跑 bench_all.py 出对比表直接带上我们结果
  ★不覆盖已有值 (防误覆盖手动/旧值); --force 总是覆盖为最新值

══════ 测量方法 (与 bench_common.measure_event / 工业级 bench_all 同构) ══════
  - Event 设备侧计时: 每窗口 ev_s.record() → 一次完整 kernel launch 链 → ev_e.record()
    ★流水化 /N (默认 10): 窗口内连续 N 次 /N — host 下发开销被隐藏 ~纯设备时间
      (与 verify /LOOP / bench_all --pipelined 同口径); --pipelined 0 = 单次含 host
  - 多窗口 median: N 个独立 Event 对, 最后 sync, 取 median (抗单次抖动)
  - 破 L2: 每窗口前重建输入张量 (新地址; Ascend 无清 L2 API, 重建 = n_buf 轮换同效)
  - warmup: W 次完整链路 (JIT 编译/冷 cache 消化, 不计时)
  - --msprof 模式: 直接 msprof 包原文件 → 目标 kernel (非 aclnn) Task Duration 求和 /loop
    = 纯 kernel 时间 (不含 host launch, 与 verify 的 ns 口径同源; 小算子不受 launch 开销污染)
  ★口径声明: 与工业级同尺 (Event 设备侧端到端), 可直接与 bench_all 结果对比
  ★注意: conv2d/conv_bias_relu 的 unfold 展开张量是派生分配, 不在重建范围 (工作集小, 影响有限)

══════ 输出 ══════
  FINAL_EVENT_E2E_US (单次 median, us) 或 msprof 纯 kernel us → 终端汇总表 → 写回 OUR_RESULTS_US
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


def _inject(src: str, warmup: int, reps: int, pipelined: int = 1) -> str:
    """在 kernel_op.py 的 for LOOP 循环处注入 Event 计时块.
    pipelined=1: 每窗口 1 次完整链路 (含 host 下发开销, 与工业级单次同构);
    pipelined>1: 每窗口连续 P 次 /P (host 开销被流水隐藏 ~纯设备时间, 与 bench --pipelined 同构).
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
    # 流水化: 窗口内再套一层 for _p in range(_PIPE) (深 4 格)
    body_pipe = "".join(("    " + ln if ln.strip() else ln) for ln in body_deep.splitlines(keepends=True))
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
    # 窗口体: 单次 or 流水化 (P 次 /P)
    if pipelined and pipelined > 1:
        window_body = (f"{ind}        for _p in range(_PIPE):\n{body_pipe}"
                       f"{ind}        _ev_e.record()\n"
                       f"{ind}        _ts.append(_ev_s.elapsed_time(_ev_e) / _PIPE * 1000.0)\n")
        pipe_env = f"{ind}    _PIPE = int(os.environ.get('FINAL_EVENT_PIPE', '{pipelined}'))\n"
    else:
        window_body = (f"{body_deep}"
                       f"{ind}        _ev_e.record()\n"
                       f"{ind}        _ts.append(_ev_s.elapsed_time(_ev_e) * 1000.0)\n")
        pipe_env = ""
    inject = (
        f"{ind}# ★Event 计时 (注入; FINAL_EVENT_TIME 触发)\n"
        f"{ind}if os.environ.get('FINAL_EVENT_TIME'):\n"
        f"{ind}    _W = int(os.environ.get('FINAL_EVENT_WARMUP', '{warmup}'))\n"
        f"{ind}    for _ in range(_W):\n{body_deep}"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _REPS = int(os.environ.get('FINAL_EVENT_REPS', '{reps}'))\n"
        f"{pipe_env}"
        f"{ind}    _ts = []\n"
        f"{ind}    _keep = []\n"
        f"{ind}    for _r in range(_REPS):\n"
        f"{alloc_block}"
        f"{keep_block}"
        f"{ind}        _ev_s = torch.npu.Event(enable_timing=True)\n"
        f"{ind}        _ev_e = torch.npu.Event(enable_timing=True)\n"
        f"{ind}        _ev_s.record()\n"
        f"{window_body}"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _ts.sort()\n"
        f"{ind}    print('FINAL_EVENT_E2E_US:%.2f' % _ts[len(_ts) // 2])\n"
        f"{ind}    raise SystemExit(0)\n"
    )
    out = "".join(lines[:for_idx]) + inject + "".join(lines[for_idx:])
    if not re.search(r"^\s*import\s+os\b", out, re.M):
        out = "import os\n" + out
    return out


def _measure(op: str, path: Path, warmup: int, reps: int, pipelined: int = 1):
    """对单个算子跑 Event 计时 → 返回 (median_us, err) 或 (None, err)."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"读不了: {e}"
    injected = _inject(src, warmup, reps, pipelined)
    if not injected:
        return None, "找不到标准 for LOOP 循环"
    evt_kernel = _WORK / f"{op}_event.py"
    _WORK.mkdir(parents=True, exist_ok=True)
    evt_kernel.write_text(injected, encoding="utf-8")
    env = dict(os.environ, FINAL_EVENT_TIME="1", FINAL_EVENT_WARMUP=str(warmup),
               FINAL_EVENT_REPS=str(reps), FINAL_EVENT_PIPE=str(pipelined), KERNEL_LOOP="1")
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


def _write_back(rows, force: bool = False) -> int:
    """测量成功的算子 → 写回 bench_all.py 的 OUR_RESULTS_US.
    默认只替换 None 占位 (防覆盖手动/旧值); force=True → 总是覆盖为最新值.
    返回写回个数."""
    if not _BENCH_ALL.exists():
        print(f"  ⚠ 找不到 {_BENCH_ALL}, 跳过写回")
        return 0
    src = _BENCH_ALL.read_text(encoding="utf-8")
    updated = []
    for op, us, _err in rows:
        if us is None:
            continue
        # ★值模式限定 None 或数字: 绝不匹配列表 (如 OP_MODES 的 ["eager",...]),
        #   OP_MODES 在 OUR_RESULTS_US 之前, 否则 --force 会破坏它 (expected ':' after dictionary key)
        if force:
            pat = re.compile(rf'("{re.escape(op)}"\s*:\s*)(None|\d+(?:\.\d+)?)')
        else:
            pat = re.compile(rf'("{re.escape(op)}"\s*:\s*)None')
        new, n = pat.subn(rf"\g<1>{us:g}", src, count=1)
        if n:
            src = new
            updated.append(op)
    if updated:
        _BENCH_ALL.write_text(src, encoding="utf-8")
        print(f"  ✅ 已写回 {_BENCH_ALL} OUR_RESULTS_US: {', '.join(updated)}"
              + (" (--force 覆盖)" if force else ""))
    return len(updated)


def _measure_msprof(op: str, path: Path, loop: int = 100):
    """msprof 纯 kernel 模式: 包原 kernel_op.py (KERNEL_LOOP=loop) → op_summary
    目标 kernel (非 aclnn) Task Duration 求和 /loop = 单次纯 kernel 时间 (us).
    ★与 verify 的 ns 口径同源 (不含 host launch) — 与工业级 --msprof 直接可比."""
    import csv
    try:
        _msprof_out = _WORK / f"{op}_msprof"
        import shutil as _sh
        _sh.rmtree(_msprof_out, ignore_errors=True)
        _msprof_out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, KERNEL_LOOP=str(loop))
        cmd = ["msprof", f"--output={_msprof_out}",
               f"--application={sys.executable or 'python3'} {path}", "--ai-core=on"]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace",
                           timeout=_TIMEOUT, env=env)
        summaries = sorted(_msprof_out.rglob("op_summary*.csv"))
        rows = []
        for p in summaries:
            try:
                with open(p, encoding="utf-8") as f:
                    rows += list(csv.DictReader(f))
            except Exception:
                continue
        target_us = 0.0
        n_rows = 0
        for row in rows:
            dur = row.get("Task Duration(us)") or row.get("TaskDuration")
            opn = row.get("Op Name") or row.get("OpName") or ""
            if dur is None or opn.startswith("aclnn"):
                continue
            try:
                target_us += float(dur)
                n_rows += 1
            except ValueError:
                continue
        if n_rows < loop:
            print(f"    ⚠ msprof 只记 {n_rows} 行 < loop({loop}) — 欠采, 结果可能偏大")
        return round(target_us / loop, 2), None
    except Exception as e:
        return None, f"msprof 失败: {str(e)[:120]}"


def main():
    import argparse
    p = argparse.ArgumentParser(description="最终算子 Event 端到端耗时 (工业级同构方法)")
    p.add_argument("--rep", type=int, default=30, help="测量 Event 对个数 (默认 30)")
    p.add_argument("--warmup", type=int, default=10, help="预热完整链路次数 (默认 10)")
    p.add_argument("--pipelined", type=int, default=10, metavar="N",
                   help="流水化模式: 每窗口连续调用 N 次 /N (隐藏 host 开销 ~纯设备时间, "
                        "与 verify/bench_all 同口径; 默认 10); 0=单次含 host 开销")
    p.add_argument("--msprof", action="store_true",
                   help="★msprof 纯 kernel 模式: 包原文件 msprof → 目标 kernel Task Duration 求和 "
                        "/N (不含 host launch, 与工业级 --msprof 同源可比); 替代 Event 注入计时")
    p.add_argument("--loop", type=int, default=100, help="msprof 模式: KERNEL_LOOP 循环次数 (默认 100)")
    p.add_argument("--no-write", action="store_true", help="只测不写回 bench_all.py")
    p.add_argument("--force", action="store_true",
                   help="写回时覆盖已有值 (默认只替换 None 占位, 防覆盖手动/旧值; "
                        "换测量口径重测时用 --force 更新)")
    args = p.parse_args()

    if not OP_PATHS:
        print("❌ 先在脚本顶部 OP_PATHS 填最终算子的 kernel_op.py 路径 "
              "(绝对路径 或 相对 triton_agent_optimizer/ 的相对路径)")
        sys.exit(1)

    pipe = args.pipelined if args.pipelined and args.pipelined > 1 else 1
    if args.msprof:
        mode_s = "msprof 纯 kernel"
        param_s = "loop=%d" % args.loop
    else:
        mode_s = "流水化 P=%d (近似纯设备)" % pipe if pipe > 1 else "严格单次 (含 host)"
        param_s = "warmup=%d, reps=%d" % (args.warmup, args.rep)
    print(f"══ 最终算子测量 ({mode_s}, {param_s}) ══\n")
    rows = []
    for op, p_str in OP_PATHS.items():
        path = _resolve(p_str)
        print(f"⏱  {op}: {path}")
        if args.msprof:
            us, err = _measure_msprof(op, path, args.loop)
        else:
            us, err = _measure(op, path, args.warmup, args.rep, pipe)
        if us is not None:
            print(f"   ✅ FINAL_EVENT_E2E_US = {us:.2f} ({'msprof 纯 kernel' if args.msprof else 'median of ' + str(args.rep)})")
        else:
            print(f"   ❌ {err}")
        rows.append((op, us, err))

    if args.msprof:
        tail_s = "msprof 纯 kernel"
    else:
        tail_s = "Event median" + (" 流水化P=%d" % pipe if pipe > 1 else " 单次完整链路")
    print("\n" + "═" * 72)
    print(f"  最终算子对比 (单位 us, {tail_s})")
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
        n = _write_back(rows, force=args.force)
        if n:
            print("  下次跑 bench_all.py 出的对比表将自动带上我们结果列.")


if __name__ == "__main__":
    main()
