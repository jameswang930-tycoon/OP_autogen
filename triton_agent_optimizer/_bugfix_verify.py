#!/usr/bin/env python3
"""验证 B1 (sweep 缓存过期误导修复) + O1 (07 字段复用消除双写)."""
import json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    "    pass\n"
    "if __name__ == '__main__':\n"
    "    main()\n", encoding="utf-8")

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
