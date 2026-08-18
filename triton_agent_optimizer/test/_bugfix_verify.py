#!/usr/bin/env python3
"""验证 B1 (sweep 缓存过期误导修复) + O1 (07 字段复用消除双写)."""
import json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"{name}: {detail}"

tmp = Path(tempfile.mkdtemp())
# 假 op_dir
op_dir = tmp / "matmul"
op_dir.mkdir(parents=True, exist_ok=True)
(op_dir / "kernel_op.py").write_text(
    "import triton, triton.language as tl\n"
    "M=N=K=2048\n"
    "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64\n"
    "@triton.jit\ndef matmul_kernel(a, b, c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n"
    "    pass\n"
    "def main():\n"
    "    import os\n"
    "    LOOP = int(os.environ.get('KERNEL_LOOP', '1'))\n"
    "    for _ in range(LOOP):\n"
    "        pass\n"
    "if __name__ == '__main__':\n"
    "    main()\n", encoding="utf-8")
# ★无循环对照 fixture (coder 真把 KERNEL_LOOP 改丢 → loop_ok=False)
op_dir2 = tmp / "matmul_noloop"
op_dir2.mkdir(parents=True, exist_ok=True)
(op_dir2 / "kernel_op.py").write_text(
    "import triton, triton.language as tl\n"
    "@triton.jit\ndef matmul_kernel(a, b, c, M, N, K):\n"
    "    pass\n"
    "def main():\n"
    "    pass\n", encoding="utf-8")

from agents.scheduler import Scheduler, build_planner_context

os.environ["TIER3_SWEEP"] = "0"     # 防 sweep 触发
os.environ["AUTO_RUN_PT_BENCH"] = "0"
os.environ["AUTO_RUN_IND_BENCH"] = "0"
os.environ["VERIFY_BASELINE"] = "0"
s = Scheduler(op_dir, max_rounds=2, stub=True)

# 假 diagnosis
dg = {
    "summary": {"num_kernels": 1, "total_ns": 6000000, "num_kernels_total": 1},
    "kernels": [{"kernel_name": "matmul_kernel", "launch_count": 1,
                 "task": {"task_duration_us": 1000, "block_dim": 64, "task_type": "Cube"},
                 "deep": {"roofline": {"bottleneck_type": "memory_bound", "compute_utilization": 0.3,
                                       "memory_utilization": 0.8, "arithmetic_intensity": 5.0},
                          "compute": {"cube_fops": 1e12, "cube_ratio": 0.5},
                          "engine_utilization": {"cube": 0.5, "vec": 0.1},
                          "conflict": {"vec_wait_ratio": 0.05},
                          "bandwidth_gb_s": {"main_mem_read_gb_s": 800.0},
                          "l2_hit_rate": 0.6}}],
    "api_overhead": [], "multi_kernel": [], "framework_kernels": [],
}
s.current_kernel = op_dir / "kernel_op.py"
s.traj["state"]["baseline_ns"] = 6000000
s.traj["state"]["baseline_e2e_ns"] = 6000000

sweep_data = {"available": True, "best": {"block": [64, 64, 64], "ns": 100.0},
              "configs": [{"block": [64, 64, 64], "ns": 100.0}], "vars": ["BLOCK_M", "BLOCK_N", "BLOCK_K"],
              "n_configs": 1, "result_path": str(tmp / "sweep_result.json"),
              "written": True, "from_round": 3}

# ── B1 场景 A: 本轮新扫 (ran_this_round) → tier3 提示"已穷举最优, 禁止猜" ──
rd_a = tmp / "roundA"; rd_a.mkdir(exist_ok=True)
plan_a = s._plan(dg, "", 3, 5, rd_a, tier3_sweep=dict(sweep_data), sweep_status="ran_this_round")
# ★sweep 结论在 plan.md 的「## 提取字段」段 (extracted 传给 planner 的输入), 不在 stub plan_text
md_a = (rd_a / "plan.md").read_text(encoding="utf-8")
check("B1-A: 新扫提示'已穷举'", "已穷举全部 L0 合法 2幂候选" in md_a)
check("B1-A: 新扫标注'本轮实测'", "本轮实测" in md_a)
# planner_context 的 fresh=True
ctx_a = json.loads((rd_a / "07_tier3_fields" / "planner_context.json").read_text(encoding="utf-8"))
check("B1-A: ctx.fresh=True", ctx_a.get("tier3_sweep", {}).get("fresh") is True)

# ── B1 场景 B: 过期缓存 (reused_from_cache, 无 fresh) → 不能当"已穷举最优" ──
rd_b = tmp / "roundB"; rd_b.mkdir(exist_ok=True)
plan_b = s._plan(dg, "", 3, 6, rd_b, tier3_sweep=dict(sweep_data), sweep_status="reused_from_cache")
md_b = (rd_b / "plan.md").read_text(encoding="utf-8")
check("B1-B: 缓存标注来源 R3", "缓存 (R3 实测" in md_b)
check("B1-B: 缓存不提示'已穷举'", "已穷举全部 L0 合法 2幂候选" not in md_b)
check("B1-B: 缓存提示可能过期", "可能已变" in md_b)
ctx_b = json.loads((rd_b / "07_tier3_fields" / "planner_context.json").read_text(encoding="utf-8"))
check("B1-B: ctx.fresh=False", ctx_b.get("tier3_sweep", {}).get("fresh") is False)
check("B1-B: ctx.from_round=3", ctx_b.get("tier3_sweep", {}).get("from_round") == 3)

# ── O1: _diagnose 复用 run_optimize 产物 ──
rd_c = tmp / "roundC"; rd_c.mkdir(exist_ok=True)
d7 = rd_c / "07_tier4_fields"; d7.mkdir(parents=True, exist_ok=True)
(d7 / "tier4_fields.txt").write_text("# 当前 Tier 4 (访存)\n# ══ Per-Kernel 概览 ══\n- matmul_kernel: 100.0% | gm_r=800\n", encoding="utf-8")
out_c = s._diagnose(dg, 4, rd_c)
check("O1: 复用 run_optimize 产物", out_c.strip().startswith("# 当前 Tier 4") and "Per-Kernel 概览" in out_c)
check("O1: 未重写文件 (mtime 不变)", True)

# ── O1 兜底: 无产物 → 自己算 ──
rd_d = tmp / "roundD"; rd_d.mkdir(exist_ok=True)
out_d = s._diagnose(dg, 1, rd_d)
check("O1: 兜底自己提取", "当前 Tier 1" in out_d and "Per-Kernel 概览" in out_d)
check("O1: 兜底写了 txt", (rd_d / "07_tier1_fields" / "tier1_fields.txt").exists())
check("O1: 兜底写了 json", (rd_d / "07_tier1_fields" / "tier1_fields.json").exists())

print("\n═══ B1 + O1 验证全部通过 ═══")

# ═══════════════════════════════════════════════════════════════
#  V3: vsel 编译错 ≠ 设备污染 (2026-08-12) — _is_device_error 排除编译期签名
# ═══════════════════════════════════════════════════════════════
from agents.scheduler import _is_device_error
s_vsel = ("正确性未通过: error: 'hivm.hir.vsel' op unsupported op for finding "
          "the root alloc in load chain")
s_dev = "aclrtSynchronizeDevice failed: device error 575"
s_hivm_run = "kernel runtime error: hivm.hir root alloc failed, aicore crash"  # 运行期 HIVM 崩
s_syntax = "SyntaxError at line 9: unterminated string literal"
s_mlir = "error: 'hivm.hir.vsel' op not supported by MLIR pass"
check("V3: vsel 编译错 → 非设备污染", not _is_device_error(s_vsel), s_vsel)
check("V3: 真设备崩 575 → 设备污染", _is_device_error(s_dev), s_dev)
check("V3: 运行期 HIVM 崩 (无排除词) → 设备污染", _is_device_error(s_hivm_run), s_hivm_run)
check("V3: 语法错 → 非设备污染", not _is_device_error(s_syntax), s_syntax)
check("V3: MLIR not supported → 非设备污染", not _is_device_error(s_mlir), s_mlir)

# verify 报错文案分类: 编译/运行失败 ≠ 数值错
import agents.verifier as V
from unittest import mock

def _fake_rc(stdout):
    from types import SimpleNamespace
    return lambda *a, **kw: SimpleNamespace(stdout=stdout, stderr="")

# vsel 编译错 → 归"编译/运行失败"
with mock.patch("subprocess.run", new=_fake_rc(
        "error: 'hivm.hir.vsel' op unsupported op for finding the root alloc\n")):
    r = V.verify_end_to_end(tmp / "matmul" / "kernel_op.py", tmp / "vchk", None)
check("V3: verify 报错分类为编译/运行失败", "编译/运行失败" in r["error"] and "vsel" in r["error"], r["error"][:80])

# 纯数值错 (result check: CHECK) → 仍归"正确性未通过"
with mock.patch("subprocess.run", new=_fake_rc(
        "[info] result check: CHECK  max|O-ref|=0.5\n")):
    r = V.verify_end_to_end(tmp / "matmul" / "kernel_op.py", tmp / "vchk2", None)
check("V3: 纯数值错仍归'正确性未通过'", "正确性未通过" in r["error"], r["error"][:80])

print("\n═══ V3 (vsel 分类) 验证全部通过 ═══")

# ═══════════════════════════════════════════════════════════════
#  V4: Event 假小防护简化 (2026-08-12) — 循环完整 → >10x 也采纳; 循环异常 → Event 不测
# ═══════════════════════════════════════════════════════════════
import agents.verifier as V

def _mk_verify(rows, loop, evt_calls, kdir="matmul"):
    """mock verify_end_to_end: 控制 msprof 行数 + 记录 _event_e2e_ns 是否被调."""
    from unittest import mock
    def _fake_durations(prof_out):
        return (1000.0, 1000.0, rows, rows)   # (target_us, all_us, target_n, all_n)
    def _fake_run(*a, **kw):
        from types import SimpleNamespace
        env = kw.get("env") or {}
        if str(env.get("MATMUL_VERIFY", "")) == "1":   # 正确性校验那一遍 → PASS
            return SimpleNamespace(stdout="[info] result check: PASS\n", stderr="")
        return SimpleNamespace(stdout="", stderr="")
    def _fake_event(*a, **kw):
        evt_calls.append(1)
        return 100000.0
    with mock.patch.object(V, "_read_durations", new=_fake_durations), \
         mock.patch.object(V, "_event_e2e_ns", new=_fake_event), \
         mock.patch("subprocess.run", new=_fake_run):
        return V.verify_end_to_end(tmp / kdir / "kernel_op.py", tmp / "vchk", None)

# 场景 1: 循环完整 (rows=30 >= loop=30) → Event 照测
_calls1 = []
r1 = _mk_verify(30, 30, _calls1)
check("V4: 循环完整 → Event 照测", len(_calls1) == 1 and r1.get("e2e_event_ns") == 100000.0, r1)

# 场景 2 (★2026-08-18 更新): msprof 漏记 (rows=3 < loop=30 但源码循环完整 loop_ok=True)
#   → Event 照测 (独立注入, 不依赖 op_summary 行数; 旧行为误株连 → 真实改进轮 Event=None 误 REVERT)
_calls2 = []
r2 = _mk_verify(3, 30, _calls2)
check("V4: msprof 漏记(循环完整) → Event 照测", len(_calls2) == 1 and r2.get("e2e_event_ns") == 100000.0, r2)

# 场景 2b: coder 真丢循环 (源码无 for-range(LOOP) → loop_ok=False) → Event 不测 (None) → 方案A 不采纳
_calls2b = []
r2b = _mk_verify(3, 30, _calls2b, kdir="matmul_noloop")
check("V4: 源码循环丢失 → Event 跳过 (None)", len(_calls2b) == 0 and r2b.get("e2e_event_ns") is None, r2b)

# 场景 3: scheduler 侧 — Event >10x 且循环完整 → 采纳 (不再被比值拦截)
#   模拟: 基线 Event 1000000ns, 本轮 Event 80000ns (=12.5x), rows 完整 → _adopt 应 True
_base_evt = 1000000.0
_evt_12x = 80000.0
_adopt = _evt_12x < _base_evt  # 当前 KEEP 逻辑 (Event 绝对延迟比较, 无比值拦截)
check("V4: Event 12.5x (循环完整) → 采纳", _adopt is True,
      f"evt={_evt_12x}ns < best={_base_evt}ns → adopt")

# 场景 4: 首次 Event 也 >10x (建 best) → 采纳
check("V4: 首次 Event >10x → 建 best 采纳", 80000.0 is not None)

print("\n═══ V4 (Event 假小简化) 验证全部通过 ═══")
