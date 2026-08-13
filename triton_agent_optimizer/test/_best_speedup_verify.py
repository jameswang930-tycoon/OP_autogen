# -*- coding: utf-8 -*-
"""V8: best_speedup 派生 bug 修复验证 — Event 基线缺失时兜底用 msprof 口径, 不再停留 1.0."""
import json, os, re, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tmp = Path(tempfile.mkdtemp())
op_dir = tmp / "matmul"
op_dir.mkdir(parents=True)
(op_dir / "kernel_op.py").write_text(
    "import os, torch, triton, triton.language as tl\n"
    "M = N = K = 2048\n"
    "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64\n"
    "@triton.jit\n"
    "def matmul_kernel(a, b, c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n"
    "    pass\n"
    "def main():\n"
    "    a = torch.randn(M, K, device='npu')\n"
    "    b = torch.randn(K, N, device='npu')\n"
    "    c = torch.empty(M, N, device='npu')\n"
    "    for _ in range(LOOP):\n"
    "        matmul_kernel[(1,)](a, b, c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)\n"
    "    torch.npu.synchronize()\n"
    "if __name__ == '__main__':\n"
    "    main()\n", encoding="utf-8")

from agents.scheduler import Scheduler

os.environ["TIER3_SWEEP"] = "0"
os.environ["AUTO_RUN_PT_BENCH"] = "0"
os.environ["AUTO_RUN_IND_BENCH"] = "0"
os.environ["REBASELINE_EVERY"] = "0"
os.environ["VERIFY_BASELINE"] = "1"

def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"{name}: {detail}"

# stub planner: 每轮给一个 change (BLOCK 值+64), 保证轮次推进
def _stub_plan(tier, code):
    return type("P", (), {
        "strategy": "增大BLOCK", "plan_text": json.dumps({
            "strategy": "增大BLOCK", "changes": [{
                "old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128",
                "reason": "t", "section": "① config", "tier": tier}],
            "expected_impact": "1.1", "promote": False, "promote_to": 0,
            "promote_reason": "", "promote_evidence": "", "handoff": {}}),
        "promote": False, "promote_to": 0, "promote_reason": "",
        "promote_evidence": "", "expected_impact": "1.1", "tier": tier,
        "handoff": {}})()

import agents.scheduler as S

# ★关键: baseline_verify 的 Event 是 None (Event 基线缺失的场景), 后续轮 Event 正常
def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
    rd_n = str(round_dir).replace("\\", "/")
    if "baseline_verify" in rd_n:
        return {"ok": True, "ns": 6_000_000.0, "e2e_ns": 6_000_000.0,
                "e2e_event_ns": None,        # ★Event 基线缺失 (复现 bug 场景)
                "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 6000.0}
    m = re.search(r"round(\d+)", str(round_dir))
    rn = int(m.group(1)) if m else 0
    # 每轮 msprof 快一点, Event 也正常 (有值)
    ns = 6_000_000.0 / (1.0 + 0.2 * rn)
    return {"ok": True, "ns": ns, "e2e_ns": ns, "e2e_event_ns": ns * 0.9,
            "speedup": 6_000_000.0 / ns, "loop": 30, "rows": 90, "duration_us": ns / 1000}

# stub _run_optimize (返回 diagnosis), 避免跑 run_optimize.sh
def _stub_run_optimize(self, round_dir, tier=1):
    d7 = round_dir / f"07_tier{tier}_fields"
    d7.mkdir(parents=True, exist_ok=True)
    (d7 / f"tier{tier}_fields.txt").write_text("# 当前 Tier 1\n# ══ Per-Kernel 概览 ══\n- matmul_kernel\n",
                                               encoding="utf-8")
    return {"summary": {"num_kernels": 1, "total_ns": 6_000_000, "num_kernels_total": 1},
            "kernels": [{"kernel_name": "matmul_kernel", "launch_count": 1,
                         "task": {"task_duration_us": 1000, "block_dim": 64, "task_type": "Cube"},
                         "deep": {"roofline": {"bottleneck_type": "compute_bound",
                                               "compute_utilization": 0.5,
                                               "memory_utilization": 0.2,
                                               "arithmetic_intensity": 10.0},
                                  "compute": {"cube_fops": 1e12, "cube_ratio": 0.5},
                                  "engine_utilization": {"cube": 0.5, "vec": 0.1},
                                  "conflict": {"vec_wait_ratio": 0.05},
                                  "bandwidth_gb_s": {"main_mem_read_gb_s": 800.0},
                                  "l2_hit_rate": 0.6}}],
            "api_overhead": [], "multi_kernel": [], "framework_kernels": []}

import agents.verifier as V
import unittest.mock as mock

with mock.patch("agents.planner.PlannerAgent.generate_v4",
                new=lambda self, *a, **kw: _stub_plan(kw["tier"], "")), \
     mock.patch.object(S, "_PROJECT", tmp), \
     mock.patch.object(S.Scheduler, "_run_optimize", new=_stub_run_optimize), \
     mock.patch.object(V, "verify_end_to_end", new=verify):
    s = Scheduler(op_dir, max_rounds=4, stub=True)
    s.run()

st = json.loads((tmp / "outputs" / "matmul" / "optimization_trajectory.json")
                .read_text(encoding="utf-8"))["state"]
hist = json.loads((tmp / "outputs" / "matmul" / "optimization_trajectory.json")
                  .read_text(encoding="utf-8"))["history"]

check("V8: baseline_e2e_event_ns 缺失 (场景成立)", st.get("baseline_e2e_event_ns") is None)
check("V8: best_e2e_event_ns 有值 (Event 正常更新)",
      st.get("best_e2e_event_ns") is not None, st.get("best_e2e_event_ns"))
check("V8: best_speedup 不再停留 1.0", st.get("best_speedup", 0) > 1.05,
      f"best_speedup={st.get('best_speedup')}")
check("V8: best_speedup 与 history 实际加速比一致 (msprof 兜底口径)",
      abs(st["best_speedup"] - max(h.get("speedup", 0) for h in hist)) < 0.05,
      f"best={st['best_speedup']} hist_max={max(h.get('speedup', 0) for h in hist)}")
check("V8: history 轮次有实际 speedup", any(h.get("speedup", 0) > 1.1 for h in hist))
print(f"  best_speedup={st.get('best_speedup')}  best_round={st.get('best_round')} "
      f"best_e2e_event_ns={st.get('best_e2e_event_ns')}")

print("\n═══ V8 (best_speedup 派生修复) 全部通过 ═══")
