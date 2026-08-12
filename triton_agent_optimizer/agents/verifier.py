#!/usr/bin/env python3
"""Verifier Agent (v4) — 端到端验证: 正确性校验 + msprof 耗时 → 加速比.

流程:
  ① warmup × VERIFY_WARMUP (默认3): 裸跑 kernel 预热 JIT/cache
  ② 正确性校验: MATMUL_VERIFY=1 → kernel 对 torch 参考算 diff → 必须 "result check: PASS"
     (不 PASS → 本轮 FAIL, 回传 coder 修; 在 msprof 之前, 防"改错结果还通过")
  ③ msprof 测时: 一次 msprof 内 KERNEL_LOOP=30 遍 → op_summary 求和 ÷ loop = 单次端到端 ns
  ④ H1 循环检测: 源码找 `for _ in range(LOOP):` → 找不到则用实测遍数 (防 coder 弄丢循环)
  ⑤ msprof_0 目录每次先 rmtree (防重试残留旧 CSV)
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  v4: 只跑 msprof 端到端 (整文件) → 端到端耗时 → 加速比
# ═══════════════════════════════════════════════════════════════════════════════

def _read_durations(prof_out: Path) -> tuple:
    """读 op_summary, 一次同时返回两种口径的 Task Duration(us) 之和与行数:
      (目标kernel(非aclnn)之和, 全部kernel(含aclnn框架)之和, 目标行数, 总行数).
    ★纯 kernel 耗时 = 目标(非aclnn)之和; 端到端耗时 = 全部之和 (含 torch 框架 kernel).
      KERNEL_LOOP=N 时 和=N×单次 → 除 N 得单次.
      msprof 合并同名连续 kernel 也不影响总和 (时间不丢), 故求和法稳健.
      目标口径与 task.json total_ns (baseline) 一致; 端到端口径两端(我们/PyTorch)统一.
      注: 端到端是 device 端到端 (所有 kernel 执行时间之和), 不含 host 侧 launch/gap;
          两端同法, 故可比."""
    import csv as _csv
    # ★读全部 op_summary*.csv 合并: msprof 可能按 {device}_{model}_{iter} 拆多份文件,
    #   只读 summaries[0] 会漏掉大部分 kernel (表现为行数 < loop, 加速比失真)
    summaries = sorted(prof_out.rglob("op_summary*.csv"))
    if not summaries:
        return None, None, 0, 0
    rows = []
    for _s in summaries:
        try:
            with open(_s, encoding="utf-8") as f:
                rows.extend(list(_csv.DictReader(f)))
        except Exception:
            continue
    if not rows:
        return None, None, 0, 0
    target_us, all_us = None, None
    target_n = all_n = 0
    for row in rows:
        dur = row.get("Task Duration(us)") or row.get("TaskDuration")
        op = row.get("Op Name") or row.get("OpName") or ""
        if not dur:
            continue
        try:
            d = float(dur)
        except ValueError:
            continue
        if not op.lower().startswith("aclnn"):
            target_us = (target_us or 0) + d
            target_n += 1
        all_us = (all_us or 0) + d
        all_n += 1
    return target_us, all_us, target_n, all_n


def verify_end_to_end(kernel_op: Path, round_dir: Path,
                      baseline_ns: Optional[float] = None,
                      num_kernels: Optional[int] = None,
                      num_launches: Optional[int] = None) -> dict:
    """v4 验证: warmup + 一次 msprof 内循环 KERNEL_LOOP 次取平均 (整文件).

    策略 (与 bench_910b3 同技术, 取代旧的 3 次独立 msprof — 每次 msprof 有 ~1-2min 启动开销):
      warmup = VERIFY_WARMUP (默认3): 先裸跑 kernel_op.py (KERNEL_LOOP 次) 预热 JIT/cache
      loop   = VERIFY_LOOP (默认30):  **一次 msprof** 内 kernel_op.py 内部循环 loop 次,
                                       读 op_summary 目标 kernel 耗时之和 → ÷loop = 单次端到端
      (msprof 合并同名 kernel 不影响总和; 求和法稳健)

    ★合理性告警 (防"漏记/循环丢失"导致静默错数):
      - 期望行数 ≈ loop × num_kernels; 若实际远少 → 可能 msprof 漏记 或 coder 弄丢了 KERNEL_LOOP 循环
      - 这种情况 sum/loop 会算错 → 打警告, 不静默

    返回:
      ok=True  → {"ok": True, "ns": 单次端到端ns, "speedup": baseline/ns}
      ok=False → {"ok": False, "error": 报错文本}  (回传 Coder 同轮改)
    """
    import os as _os
    import subprocess
    warmup = int(_os.environ.get("VERIFY_WARMUP", "3"))
    loop = int(_os.environ.get("VERIFY_LOOP", "30"))
    py = "python3"
    env = dict(_os.environ, KERNEL_LOOP=str(loop))   # kernel_op.py main() 内部循环 loop 次

    # warmup: 裸跑预热 (KERNEL_LOOP=loop, JIT 编译/冷cache 预热; 便宜)
    for i in range(warmup):
        subprocess.run([py, str(kernel_op)], capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace", timeout=1800, env=env)
    print(f"  [Verify] warmup x{warmup} (每轮内部 {loop} 次) done, 1 次 msprof 测 {loop} 次平均...")

    # ★正确性验证 (v4 曾只测性能不测数值): 单独跑一次 MATMUL_VERIFY=1, 结果必须 PASS.
    #   kernel_op.py main() 里 MATMUL_VERIFY=1 时对 torch 参考算 diff, 打印 "result check: PASS/CHECK".
    #   不 PASS(数值错/无校验) → 本轮 FAIL, 防"优化把结果改错还通过".
    chk_env = dict(_os.environ, KERNEL_LOOP="1", MATMUL_VERIFY="1")
    try:
        rc = subprocess.run([py, str(kernel_op)], capture_output=True, text=True,
                            encoding="utf-8", errors="backslashreplace", timeout=1800, env=chk_env)
    except Exception as e:
        return {"ok": False, "error": f"正确性校验运行失败: {e}"}
    _chk_out = (rc.stdout or "") + (rc.stderr or "")
    if "result check: PASS" not in _chk_out:
        return {"ok": False, "error": f"正确性未通过 (MATMUL_VERIFY 需输出 result check: PASS): {_chk_out.strip()[-400:]}"}
    print("    [Verify] ✅ 正确性 PASS (MATMUL_VERIFY)")

    # measure: 一次 msprof, app 内部循环 loop 次 → 和 ÷loop = 单次端到端
    # ★bug 修复: 同一 round_dir 可能重试多次 (scheduler 3次尝试), msprof 会把新 CSV 写进同一目录,
    #   旧 CSV 残留 → _read_target_duration 读 sorted()[0] 会拿到旧数据. 每次先清目录.
    import shutil as _shutil
    msprof_out = round_dir / "msprof_0"
    _shutil.rmtree(msprof_out, ignore_errors=True)
    msprof_out.mkdir(parents=True, exist_ok=True)
    cmd = ["msprof", f"--output={msprof_out}",
           f"--application={py} {kernel_op}", "--ai-core=on"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace", timeout=7200, env=env)
    except Exception as e:
        return {"ok": False, "error": f"msprof run failed: {e}"}
    target_us, all_us, n_rows, n_all = _read_durations(msprof_out)
    if target_us is None or n_rows < 1:
        tail = (r.stderr or "")[-800:] + (r.stdout or "")[-800:]
        return {"ok": False, "error": tail.strip() or "msprof 无目标 kernel"}
    # ★H1 自校正循环丢失: 从 kernel 源码判断 KERNEL_LOOP 循环是否还在.
    #   coder 弄丢/改坏 main() 的 for-range 循环时, 只跑 1 遍 → op_summary 行数 ≈ num_kernels,
    #   若仍 ÷loop 会虚高 30 倍. 源码里找不到循环 → 改用实测遍数 = 行数/每遍kernel数, 不再假设 ÷loop.
    loop_ok = True
    try:
        src = kernel_op.read_text(encoding="utf-8")
        loop_ok = bool(re.search(r"for\s+\w+\s+in\s+range\(\s*LOOP\s*\)\s*:", src))
    except Exception:
        loop_ok = True   # 读不到源码 → 按 loop 假设, 靠下方行数告警兜底
    if loop_ok:
        divisor = loop
    else:
        # ★修复: 除数用"每遍实际 launch 总数"而非去重 kernel 数.
        #   op_summary 一行=一次 launch; 多 kernel 多 launch (如 attention_mlp 9次/遍, 去重只 5)
        #   用去重数会高估遍数 → 单遍耗时被除多 → 加速比虚高 (如 ÷2). num_launches = Σ launch_count.
        _lp = num_launches or num_kernels or 1
        divisor = max(1, int(round(n_rows / _lp)))   # 实测有效遍数 (循环丢失时 = 1)
        print(f"    ⚠ 警告! kernel 源码无 KERNEL_LOOP 循环 (coder 弄丢?) "
              f"→ 用实测 {divisor} 遍平均, 不再 ÷{loop}")
    # ★两种口径同一次 op_summary 算出 (÷同一 divisor = 单次):
    #   纯 kernel 耗时 = 目标(非aclnn)之和; 端到端耗时 = 全部(含框架)之和
    per_pass_us = target_us / divisor          # 纯 kernel 单次
    e2e_per_pass_us = (all_us or target_us) / divisor   # 端到端单次 (无框架行时退化为纯kernel)
    ns = per_pass_us * 1000
    e2e_ns = e2e_per_pass_us * 1000
    # ★显式处理 baseline 缺失: 基准测量调用(baseline_ns=None)不算加速比 → None;
    #   轮次验证若 baseline 缺失 → None → 调度器告警, 不再静默当 1.0 (曾导致永远 1.000x)
    speedup = (baseline_ns / ns) if baseline_ns else None
    print(f"    msprof: 目标kernel {n_rows}行/全部 {n_all}行 (期望目标 ~{loop}×{(num_launches or num_kernels) or '?'}), "
          f"有效遍数={divisor}, 纯kernel={per_pass_us:.1f}us 端到端={e2e_per_pass_us:.1f}us")
    # ★合理性告警: 行数远少于期望 → 循环丢失 或 msprof 严重漏记 (暴露, 不静默)
    if n_rows < loop:
        print(f"    ⚠ 警告! 目标 kernel 行数 {n_rows} < loop({loop}) "
              f"(coder 丢掉了 KERNEL_LOOP 循环? 或 msprof 漏记) → 单次耗时可能不准, 加速比存疑!")
    return {"ok": True, "ns": round(ns, 1), "e2e_ns": round(e2e_ns, 1),
            "speedup": round(speedup, 4) if speedup is not None else None,
            "loop": loop, "rows": n_rows, "duration_us": round(per_pass_us, 1),
            "e2e_event_ns": _event_e2e_ns(kernel_op, round_dir, loop)}


# ═══════════════════════════════════════════════════════════════════════════════
#  ★工业级 Event 计时 (设备侧, torch.npu.Event) — 每 round 成功后补测一次
#    与 msprof 并列: msprof 给诊断/纯kernel拆解; Event 给权威绝对端到端 (无 profiler 扰动).
#    法: 改写 kernel_op.py 的 main(), 在 KERNEL_LOOP 循环前注入 warmup + Event 窗口,
#        跑一遍 → 解析 stdout 的 EVENT_E2E_US → ns. 失败返回 None (不阻断主流程).
# ═══════════════════════════════════════════════════════════════════════════════

def _inject_event_timing(src: str) -> str:
    """把 kernel_op.py 的 `for X in range(LOOP): <body>` 循环前注入 Event 计时分支.
    KERNEL_EVENT_TIME=1 时: warmup W 次 + Event 窗口包 LOOP 次 → 打印 EVENT_E2E_US, return.
    否则原样跑 (不影响正常/msprof 路径). 14 个算子 main() 都是同一个标准模式 → 通用."""
    import re as _re
    lines = src.splitlines(keepends=True)
    for_idx = None
    base_indent = 0
    for i, ln in enumerate(lines):
        m = _re.match(r'(\s*)for\s+\w+\s+in\s+range\(\s*LOOP\s*\)\s*:', ln)
        if m:
            for_idx = i
            base_indent = len(m.group(1))
            break
    if for_idx is None:
        return ""   # 找不到标准循环 → 调用方放弃
    # 收集循环体: for_idx 之后缩进 > base_indent 的行 (含空行)
    body_start = for_idx + 1
    body_end = body_start
    while body_end < len(lines):
        ln = lines[body_end]
        if ln.strip() == "":
            body_end += 1
            continue
        cur_indent = len(_re.match(r'\s*', ln).group())
        if cur_indent > base_indent:
            body_end += 1
        else:
            break
    body = "".join(lines[body_start:body_end])
    if not body.strip():
        return ""
    # 循环体再缩进 4 空格 (放进新 for 的内部)
    body_deep = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(keepends=True))
    ind = " " * base_indent
    # 注入块: 在原 for 之前插入 (原 for+body 保留在后, 仅非 Event 模式走)
    inject = (
        f"{ind}# ★工业级 Event 计时 (注入; KERNEL_EVENT_TIME 触发, 不影响正常/msprof 路径)\n"
        f"{ind}if os.environ.get('KERNEL_EVENT_TIME'):\n"
        f"{ind}    _W = int(os.environ.get('KERNEL_EVENT_WARMUP', '5'))\n"
        f"{ind}    for _ in range(_W):\n{body_deep}"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _ev_s = torch.npu.Event(enable_timing=True)\n"
        f"{ind}    _ev_e = torch.npu.Event(enable_timing=True)\n"
        f"{ind}    _ev_s.record()\n"
        f"{ind}    for _ in range(LOOP):\n{body_deep}"
        f"{ind}    _ev_e.record()\n"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    print('EVENT_E2E_US:%.2f' % (_ev_s.elapsed_time(_ev_e) / LOOP * 1000.0))\n"
        f"{ind}    raise SystemExit(0)\n"
    )
    out = "".join(lines[:for_idx]) + inject + "".join(lines[for_idx:])
    # ★保险: 注入块引用 os.environ — 若源文件顶部没有 import os (未来新算子), 补上防 NameError
    #   (Event 测不到 → 方案A 永不采纳 → 误 REVERT)
    if not re.search(r"^\s*import\s+os\b", out, re.M):
        out = "import os\n" + out
    return out


def _event_e2e_ns(kernel_op: Path, round_dir: Path, loop: int) -> Optional[float]:
    """对 kernel_op.py 跑一次 Event 计时 → 返回单次端到端 ns (设备侧), 失败返回 None.
    ★工业级: torch.npu.Event 紧凑循环, 无 msprof profiler 扰动."""
    import os as _os
    import subprocess
    import shutil as _sh
    try:
        src = kernel_op.read_text(encoding="utf-8")
        injected = _inject_event_timing(src)
        if not injected:
            return None
        evt_kernel = round_dir / "event_kernel.py"
        evt_kernel.write_text(injected, encoding="utf-8")
        env = dict(_os.environ, KERNEL_EVENT_TIME="1",
                   KERNEL_LOOP=str(loop), KERNEL_EVENT_WARMUP="5",
                   MATMUL_VERIFY="")   # Event 计时不算正确性 (verify 已先校验过)
        r = subprocess.run(["python3", str(evt_kernel)], capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace",
                           timeout=1800, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"EVENT_E2E_US:([\d.]+)", out)
        if not m:
            return None
        return round(float(m.group(1)) * 1000.0, 1)   # us → ns
    except Exception:
        return None
