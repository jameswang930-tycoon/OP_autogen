#!/usr/bin/env python3
"""Verifier Agent (v4.5) — 端到端验证: 正确性校验 + msprof 耗时 → 加速比.

★两段验证:
  段1 verify_fast_gate (秒级, 无 msprof): _correctness_check + _event_e2e_ns
    → scheduler 用它做快速门: Event 不快于 best 直接 REVERT (省 msprof 分钟级/轮)
  段2 verify_end_to_end (全量): 过门才跑 — warmup + msprof 双口径 + Event 确认

verify_end_to_end 流程:
  ① warmup × VERIFY_WARMUP (默认3): 裸跑 kernel 预热 JIT/cache
  ② 正确性校验 (_correctness_check): MATMUL_VERIFY=1 → 必须 "result check: PASS"
     (不 PASS → 本轮 FAIL, 回传 coder 修; 在 msprof 之前, 防"改错结果还通过")
  ③ msprof 测时: 一次 msprof 内 KERNEL_LOOP=30 遍 → op_summary 求和 ÷ loop = 单次端到端 ns
  ④ H1 循环检测: 源码找 `for _ in range(LOOP):` → 找不到则用实测遍数 (防 coder 弄丢循环)
  ⑤ msprof_0 目录每次先 rmtree (防重试残留旧 CSV)
  ⑥ ★Event 设备侧 (_event_e2e_ns): 注入 warmup + 多窗口 (KERNEL_EVENT_REPS=5) median,
     每窗口输入重建破 L2 — 与工业级基准同口径; 循环异常 (行数<loop) 不测 Event (None)
"""
from __future__ import annotations

import ast, json, math, re, sys, textwrap
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
        if not math.isfinite(d) or d < 0:
            continue
        if not op.lower().startswith("aclnn"):
            target_us = (target_us or 0) + d
            target_n += 1
        all_us = (all_us or 0) + d
        all_n += 1
    return target_us, all_us, target_n, all_n


def _correctness_check(kernel_op: Path) -> tuple:
    """正确性校验 (MATMUL_VERIFY=1): 一次子进程, 秒级. 返回 (ok, error)."""
    import os as _os
    import subprocess
    py = "python3"
    chk_env = dict(_os.environ, KERNEL_LOOP="1", MATMUL_VERIFY="1")
    try:
        rc = subprocess.run([py, str(kernel_op)], capture_output=True, text=True,
                            encoding="utf-8", errors="backslashreplace", timeout=1800, env=chk_env)
    except Exception as e:
        return False, f"正确性校验运行失败: {e}"
    _chk_out = (rc.stdout or "") + (rc.stderr or "")
    if rc.returncode != 0:
        return False, f"正确性校验进程失败 (exit={rc.returncode}): {_chk_out.strip()[-400:]}"
    if "result check: PASS" not in _chk_out:
        # ★2026-08-12 报错分类: HIVM 编译错 (vsel/root alloc) / Python 语法错 / Traceback
        #   ≠ 数值错误 — 分类后 planner/coder 拿到准确报错 (vsel → 改连续仿射寻址; 数值错 → 改计算逻辑)
        if re.search(r"error:|Traceback|unsupported op|SyntaxError|not supported|MLIR", _chk_out):
            return False, f"kernel 编译/运行失败 (非数值错): {_chk_out.strip()[-400:]}"
        return False, f"正确性未通过 (MATMUL_VERIFY 需输出 result check: PASS): {_chk_out.strip()[-400:]}"
    return True, ""


def verify_fast_gate(kernel_op: Path, round_dir: Path, loop: int = 30) -> dict:
    """★两段验证·段1: 正确性 + Event 快速门 (无 msprof, ~秒级).
    AutoKernel 式快速实验循环: 大多数轮次是"不快于 best"的 REVERT 轮,
    段1 用 Event 绝对延迟直接判死, 省掉 warmup×3 + msprof (几分钟/轮).

    返回:
      {"ok": False, "error": ...}           → 编译/正确性失败 (回传 coder)
      {"ok": True, "e2e_event_ns": None}    → Event 不可用 (注入失败/找不到循环)
                                                → 调用方退化跑全量 msprof
      {"ok": True, "e2e_event_ns": value}   → 门值: 调用方与 best 比较
    """
    import os as _os
    ok, err = _correctness_check(kernel_op)
    if not ok:
        return {"ok": False, "error": err}
    _evt = _event_e2e_ns(kernel_op, round_dir, loop)
    return {"ok": True, "e2e_event_ns": _evt}


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
    if loop <= 0 or warmup < 0:
        return {"ok": False, "error": "VERIFY_LOOP 必须大于零，VERIFY_WARMUP 不得为负"}
    py = "python3"
    env = dict(_os.environ, KERNEL_LOOP=str(loop))   # kernel_op.py main() 内部循环 loop 次

    # warmup: 裸跑预热 (KERNEL_LOOP=loop, JIT 编译/冷cache 预热; 便宜)
    for i in range(warmup):
        try:
            warm = subprocess.run([py, str(kernel_op)], capture_output=True, text=True,
                                  encoding="utf-8", errors="backslashreplace", timeout=1800, env=env)
        except Exception as exc:
            return {"ok": False, "error": f"warmup failed: {exc}"}
        if warm.returncode != 0:
            return {"ok": False, "error": f"warmup exit={warm.returncode}: {(warm.stderr or '')[-400:]}"}
    print(f"  [Verify] warmup x{warmup} (每轮内部 {loop} 次) done, 1 次 msprof 测 {loop} 次平均...")

    # ★正确性验证 (v4 曾只测性能不测数值): 单独跑一次 MATMUL_VERIFY=1, 结果必须 PASS.
    #   kernel_op.py main() 里 MATMUL_VERIFY=1 时对 torch 参考算 diff, 打印 "result check: PASS/CHECK".
    #   不 PASS(数值错/无校验) → 本轮 FAIL, 防"优化把结果改错还通过".
    _ok_chk, _err_chk = _correctness_check(kernel_op)
    if not _ok_chk:
        return {"ok": False, "error": _err_chk}
    print("    [Verify] ✅ 正确性 PASS (MATMUL_VERIFY)")

    # measure: 一次 msprof, app 内部循环 loop 次 → 和 ÷loop = 单次端到端
    # ★bug 修复: 同一 round_dir 可能重试多次 (scheduler 3次尝试), msprof 会把新 CSV 写进同一目录,
    #   旧 CSV 残留 → _read_target_duration 读 sorted()[0] 会拿到旧数据. 每次先清目录.
    import shutil as _shutil
    msprof_out = round_dir / "msprof_0"
    _shutil.rmtree(msprof_out, ignore_errors=True)
    msprof_out.mkdir(parents=True, exist_ok=True)
    # 保留既有输入重建协议；随机初始化可能产生设备任务，全部任务和含这些成本。
    # 重建不保证冷 L2，也不等同工业基准的预分配轮换协议。
    _app = kernel_op
    count_injected = False
    try:
        original = kernel_op.read_text(encoding="utf-8")
        _inj = _inject_l2_rebuild(original) or original
        counted = _inject_iteration_counter(_inj)
        count_injected = bool(counted)
        _inj = counted or _inj
        if _inj:
            _app = round_dir / "msprof_app.py"
            _app.write_text(_inj, encoding="utf-8")
    except Exception:
        _app = kernel_op
    cmd = ["msprof", f"--output={msprof_out}",
           f"--application={py} {_app}", "--ai-core=on"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace", timeout=7200, env=env)
    except Exception as e:
        return {"ok": False, "error": f"msprof run failed: {e}"}
    if r.returncode != 0:
        return {"ok": False, "error": f"msprof exit={r.returncode}: {(r.stderr or r.stdout or '')[-800:]}"}
    if count_injected:
        completed = re.findall(r"VERIFY_ITERATIONS:(\d+)", (r.stdout or '') + (r.stderr or ''))
        if len(completed) != 1 or int(completed[0]) != loop:
            return {"ok": False, "error": f"迭代完成次数不符: expected={loop}, actual={completed}"}
    target_us, all_us, n_rows, n_all = _read_durations(msprof_out)
    if target_us is None or target_us <= 0 or n_rows < 1:
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
    e2e_per_pass_us = (all_us or target_us) / divisor   # 历史字段 e2e_ns 实为全部任务和
    ns = per_pass_us * 1000
    e2e_ns = e2e_per_pass_us * 1000
    # ★显式处理 baseline 缺失: 基准测量调用(baseline_ns=None)不算加速比 → None;
    #   轮次验证若 baseline 缺失 → None → 调度器告警, 不再静默当 1.0 (曾导致永远 1.000x)
    speedup = (baseline_ns / ns) if baseline_ns else None
    print(f"    msprof: 目标kernel {n_rows}行/全部 {n_all}行 (期望目标 ~{loop}×{(num_launches or num_kernels) or '?'}), "
          f"有效遍数={divisor}, 目标任务和={per_pass_us:.1f}us 全部任务和={e2e_per_pass_us:.1f}us")
    # ★合理性告警: 行数远少于期望 → 循环丢失 或 msprof 严重漏记 (暴露, 不静默)
    if n_rows < loop:
        print(f"    ⚠ 警告! 目标 kernel 行数 {n_rows} < loop({loop}) "
              f"(coder 丢掉了 KERNEL_LOOP 循环? 或 msprof 漏记) → 单次耗时可能不准, 加速比存疑!")
    # ★2026-08-12 假小防护简化: Event 真实性的唯一保证 = KERNEL_LOOP 循环完整.
    #   循环真被改没 (loop_ok=False) → Event 注入也找不到循环 → None → "Event 缺失 → 方案A 不采纳".
    #   循环完整 → Event 是真实设备侧计时, **即使 >10x 也是真优化** (naive→tl.dot 单轮可 >10x),
    #   照常采纳/记录 (不再用 EVENT_MIN_RATIO 比值误伤).
    # ★bug 修复 (2026-08-18): 测 Event 的前提用 loop_ok (源码循环在) 而非 n_rows>=loop —
    #   msprof 漏记 (n_rows<loop 但循环完整) 时 Event 独立注入测量、不依赖 op_summary 行数,
    #   原条件把"漏记"株连成"Event 不可信" → 真实改进轮 Event=None → 方案A 误 REVERT.
    _evt_ns = _event_e2e_ns(kernel_op, round_dir, loop) if loop_ok else None
    return {"ok": True, "ns": round(ns, 1), "e2e_ns": round(e2e_ns, 1),
            "metric": "target_task_duration_sum_per_iteration",
            "e2e_metric": "all_task_duration_sum_per_iteration",
            "iteration_count_source": "completed_loop_marker" if count_injected else "estimated_from_launches",
            "speedup": round(speedup, 4) if speedup is not None else None,
            "loop": loop, "rows": n_rows, "duration_us": round(per_pass_us, 1),
            "e2e_event_ns": _evt_ns}


# ═══════════════════════════════════════════════════════════════════════════════
#  ★工业级 Event 计时 (设备侧, torch.npu.Event) — 每 round 成功后补测一次
#    与 msprof 并列: msprof 给诊断/纯kernel拆解; Event 给权威绝对端到端 (无 profiler 扰动).
#    法: 改写 kernel_op.py 的 main(), 在 KERNEL_LOOP 循环前注入 warmup + Event 窗口,
#        跑一遍 → 解析 stdout 的 EVENT_E2E_US → ns. 失败返回 None (不阻断主流程).
# ═══════════════════════════════════════════════════════════════════════════════

def _inject_iteration_counter(src: str) -> str:
    """在标准测量循环末记录实际完成次数；不依赖 kernel 名称或 launch 数。"""
    src = src.rstrip('\n') + '\n'
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
             and n.iter.func.id == 'range' and len(n.iter.args) == 1
             and isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == 'LOOP']
    if len(loops) != 1 or loops[0].orelse:
        return ''
    node = loops[0]
    lines = src.splitlines(keepends=True)
    ind = ' ' * node.col_offset
    lines.insert(node.end_lineno, ind + "print('VERIFY_ITERATIONS:%d' % _verify_iterations)\n")
    lines.insert(node.end_lineno, ind + '    _verify_iterations += 1\n')
    lines.insert(node.lineno - 1, ind + '_verify_iterations = 0\n')
    result = ''.join(lines)
    ast.parse(result)
    return result


def _input_allocations(src: str, for_idx: int):
    """提取同一函数内、循环前完整的直接张量分配语句（支持多行）。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    line = for_idx + 1
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.lineno < line <= n.end_lineno]
    if not functions:
        return []
    function = max(functions, key=lambda n: n.lineno)
    result = []
    for node in function.body:
        if node.lineno >= line:
            break
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        allocations = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                       and n.func.value.id == 'torch' and n.func.attr in ('rand', 'randn', 'empty', 'zeros', 'ones')]
        if allocations:
            lines = src.splitlines()[node.lineno - 1:node.end_lineno]
            result.append((textwrap.dedent('\n'.join(lines)), node.targets[0].id))
        elif any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and n.id in {name for _, name in result} for n in ast.walk(node.value)):
            # 如 y = x.view(...)：只重建 x 会让 y 指向旧输入，保留原输入更安全。
            return []
    return result


def _with_os_import(src: str) -> str:
    """保留模块 docstring / __future__ 的合法位置。"""
    if re.search(r"^\s*import\s+os\b", src, re.M):
        return src
    tree = ast.parse(src)
    after = 0
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module == '__future__') or (
                node is tree.body[0] and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            after = node.end_lineno
        else:
            break
    lines = src.splitlines(keepends=True)
    lines.insert(after, 'import os\n')
    return ''.join(lines)


def _inject_event_timing(src: str, rebuild_inputs: bool = True) -> str:
    """把 kernel_op.py 的 `for X in range(LOOP): <body>` 循环前注入 Event 计时分支.
    KERNEL_EVENT_TIME=1 时: warmup W 次 + ★KERNEL_EVENT_REPS 个独立 Event 窗口 (每窗口包 LOOP 次)
    → 取 median 打印 EVENT_E2E_US, return. 否则原样跑 (不影响正常/msprof 路径).
    ★2026-08-12: 单窗口 ÷LOOP 只有 1 个样本 → 改多窗口 median (与 bench measure_event 同款,
    抗单次抖动, KEEP 决策更稳). 14 个算子 main() 都是同一个标准模式 → 通用.
    ★rebuild_inputs=True (默认): 每窗口前重放输入分配 (新地址) 破 L2 — 与工业级基准同口径;
    False → 热 L2 (量化 L2 复用虚高用, feedback/remeasure_best.py)."""
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
    # ★2026-08-12 输入轮换 (破 L2, 与 bench_common.measure_event / triton do_bench 同口径):
    #   注入的 Event 窗口若一直用同一批输入, 工作集 < 192MB(L2) 时后 N 次全 L2 命中 → 端到端虚高
    #   (假快) → vs_industrial 比值偏乐观. Ascend 无清 L2 API → do_bench 的 cache 清空用
    #   "每窗口前重放输入分配 (新地址)" 替代: 扫描 for LOOP 之前的 torch 张量分配行,
    #   重放进每个 rep 窗口内 → 每窗口全新输入, 与工业级基准 (fn(i) 轮换 32 组) 同口径.
    #   ★2026-08-13 修复: 重建的张量必须被 _keep 列表持有 (防 GC) — 否则 PyTorch caching
    #   allocator 会复用刚释放的地址 → 窗口2 起 L2 又命中 → 破 L2 失效 (实测虚高比值全 ≈1.0).
    #   分配行都在 main 的循环前 (14 个算子统一模式, 单行); 提取不到 → 不重建 (退化为热 L2, 不报错).
    _alloc_lines = _input_allocations(src, for_idx)
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
    # ★多窗口 rep 循环内再深 4 格 (两层 for: rep → LOOP → body)
    body_deep2 = "".join(("    " + ln if ln.strip() else ln) for ln in body_deep.splitlines(keepends=True))
    ind = " " * base_indent
    # ★输入重建块 (缩进到 rep 内层 12 空格, 与 _ev_s 同级): 每窗口前重新分配输入 → 新地址 → 破 L2 复用
    #   + _keep.append: 持有本窗口张量防 GC — 否则 caching allocator 复用地址, 破 L2 失效
    #   rebuild_inputs=False → 不重建 (热 L2, 量化虚高用)
    _alloc_block = ""
    _keep_block = ""
    if rebuild_inputs and _alloc_lines:
        _alloc_block = "".join(textwrap.indent(_ln + "\n", ind + "        ") for _ln, _ in _alloc_lines)
        _names = ", ".join(_n for _, _n in _alloc_lines)
        _keep_block = f"{ind}        _keep.append(({_names}))\n"
    # 注入块: 在原 for 之前插入 (原 for+body 保留在后, 仅非 Event 模式走)
    inject = (
        f"{ind}# ★工业级 Event 计时 (注入; KERNEL_EVENT_TIME 触发, 不影响正常/msprof 路径)\n"
        f"{ind}if os.environ.get('KERNEL_EVENT_TIME'):\n"
        f"{ind}    _W = int(os.environ.get('KERNEL_EVENT_WARMUP', '5'))\n"
        f"{ind}    for _ in range(_W):\n{body_deep}"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _REPS = int(os.environ.get('KERNEL_EVENT_REPS', '5'))\n"
        f"{ind}    _ts = []\n"
        f"{ind}    _keep = []\n"                                  # ★持有每窗口张量 (防地址复用)
        f"{ind}    for _r in range(_REPS):\n"
        f"{_alloc_block}"                                         # ★每窗口前重建输入 (破 L2)
        f"{_keep_block}"                                          # ★防 GC 释放 → 地址不复用
        f"{ind}        _ev_s = torch.npu.Event(enable_timing=True)\n"        f"{ind}        _ev_e = torch.npu.Event(enable_timing=True)\n"
        f"{ind}        _ev_s.record()\n"
        f"{ind}        for _ in range(LOOP):\n{body_deep2}"
        f"{ind}        _ev_e.record()\n"
        f"{ind}        _ts.append(_ev_s.elapsed_time(_ev_e))\n"
        f"{ind}    torch.npu.synchronize()\n"
        f"{ind}    _ts.sort()\n"
        f"{ind}    print('EVENT_E2E_US:%.2f' % (_ts[len(_ts) // 2] / LOOP * 1000.0))\n"
        f"{ind}    raise SystemExit(0)\n"
    )
    out = "".join(lines[:for_idx]) + inject + "".join(lines[for_idx:])
    # ★保险: 注入块引用 os.environ — 若源文件顶部没有 import os (未来新算子), 补上防 NameError
    #   (Event 测不到 → 方案A 永不采纳 → 误 REVERT)
    out = _with_os_import(out)
    ast.parse(out)
    return out


def _inject_l2_rebuild(src: str) -> str:
    """把 for LOOP 循环内注入"每轮重建输入"(新地址破 L2) — msprof 冷 L2 对齐 (2026-08-17).
    与 _inject_event_timing 同法: 扫描循环前 torch 直接分配行, 在循环体顶部逐轮重建,
    _msprof_keep 持有防 GC 地址复用 (否则 caching allocator 复用刚释放地址 → L2 又命中).
    rand/zeros 等初始化会产生设备工作；新地址不保证冷 L2。
    与工业基准预分配轮换协议不等价，全部任务和不能作完整 forward 耗时。
    找不到标准循环/分配行 → 返回 "" (调用方回退原文件)."""
    import re as _re
    lines = src.splitlines(keepends=True)
    for_idx = None
    base_indent = 0
    for i, ln in enumerate(lines):
        m = _re.match(r'(\s*)for\s+\w+\s+in\s+range\(\s*LOOP\s*\)\s*:', ln)
        if m:
            for_idx, base_indent = i, len(m.group(1))
            break
    if for_idx is None:
        return ""
    alloc_lines = _input_allocations(src, for_idx)
    if not alloc_lines:
        return ""
    ind = " " * base_indent
    rebuild = "".join(textwrap.indent(_a + "\n", ind + "    ") for _a, _ in alloc_lines)
    _names = ", ".join(_n for _, _n in alloc_lines)
    pre = (f"{ind}_msprof_keep = []\n"
           f"{ind}# 每轮重建输入；初始化可能产生设备任务，不保证冷 L2。\n")
    body_inject = (f"{rebuild}"
                   f"{ind}    _msprof_keep.append(({_names}))\n")
    out = ("".join(lines[:for_idx]) + pre + "".join(lines[for_idx])
           + body_inject + "".join(lines[for_idx + 1:]))
    out = _with_os_import(out)
    ast.parse(out)
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
        if r.returncode != 0:
            return None
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"EVENT_E2E_US:([\d.]+)", out)
        if not m:
            return None
        value = float(m.group(1))
        return round(value * 1000.0, 1) if math.isfinite(value) and value > 0 else None
    except Exception:
        return None
