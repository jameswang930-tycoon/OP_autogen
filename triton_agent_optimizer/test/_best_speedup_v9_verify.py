# -*- coding: utf-8 -*-
"""V9: 首次 Event 必须对比基线 — msprof 欠采假快轮 (14x) 在 Event 口径下未变快/更慢时不得进链,
   best_speedup 不得被毒成 <1 (复现用户场景: history 14.多 vs best_speedup 0.953)."""
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

BASE_MS = 6_000_000.0      # msprof 端到端基线 (ns)
BASE_EVT = 100_000.0       # Event 基线 (ns) — 破 L2 口径, 存在

def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
    rd = str(round_dir).replace("\\", "/")
    if "baseline_verify" in rd:
        return {"ok": True, "ns": BASE_MS, "e2e_ns": BASE_MS,
                "e2e_event_ns": BASE_EVT,      # ★Event 基线存在
                "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 6000.0}
    m = re.search(r"round(\d+)", rd)
    rn = int(m.group(1)) if m else 0
    if rn == 1:
        # ★复现用户场景: msprof 欠采 → 假快 14x; Event 真实口径反而慢 5% (105us vs 100us)
        return {"ok": True, "ns": BASE_MS / 14.0, "e2e_ns": BASE_MS / 14.0,
                "e2e_event_ns": BASE_EVT * 1.05,
                "speedup": 14.0, "loop": 30, "rows": 90, "duration_us": 428.6}
    # round2: Event 真快 10%
    return {"ok": True, "ns": BASE_MS / 15.0, "e2e_ns": BASE_MS / 15.0,
            "e2e_event_ns": BASE_EVT * 0.9,
            "speedup": 15.0, "loop": 30, "rows": 90, "duration_us": 400.0}

def _stub_run_optimize(self, round_dir, tier=1):
    d7 = round_dir / f"07_tier{tier}_fields"
    d7.mkdir(parents=True, exist_ok=True)
    (d7 / f"tier{tier}_fields.txt").write_text("# 当前 Tier 1\n- matmul_kernel\n", encoding="utf-8")
    return {"summary": {"num_kernels": 1, "total_ns": BASE_MS, "num_kernels_total": 1},
            "kernels": [{"kernel_name": "matmul_kernel", "launch_count": 1,
                         "task": {"task_duration_us": 1000, "block_dim": 64, "task_type": "Cube"},
                         "deep": {"roofline": {"bottleneck_type": "compute_bound"},
                                  "compute": {"cube_fops": 1e12},
                                  "engine_utilization": {"cube": 0.5}}}]}

import agents.verifier as V
import unittest.mock as mock

with mock.patch("agents.planner.PlannerAgent.generate_v4",
                new=lambda self, *a, **kw: _stub_plan(kw["tier"], "")), \
     mock.patch.object(S, "_PROJECT", tmp), \
     mock.patch.object(S.Scheduler, "_run_optimize", new=_stub_run_optimize), \
     mock.patch.object(V, "verify_end_to_end", new=verify):
    s = Scheduler(op_dir, max_rounds=3, stub=True)
    s.run()

traj = json.loads((tmp / "outputs" / "matmul" / "optimization_trajectory.json")
                  .read_text(encoding="utf-8"))
st, hist = traj["state"], traj["history"]

check("V9: round1 (假14x但Event慢5%) 未被采纳",
      not any(h.get("decision") == "KEEP" and abs(h.get("speedup", 0) - 14.0) < 0.01
              for h in hist),
      [ (h.get("round"), h.get("speedup"), h.get("decision")) for h in hist ])
check("V9: round2 (Event真快10%) 被采纳",
      any(h.get("decision") == "KEEP" and abs(h.get("speedup", 0) - 15.0) < 0.01
          for h in hist))
check("V9: best_speedup 不被毒成 <1 (Event 口径真实值 1.11)",
      abs(st["best_speedup"] - BASE_EVT / (BASE_EVT * 0.9)) < 0.01,
      f"best_speedup={st.get('best_speedup')}")
check("V9: best_speedup >= 1.0", st.get("best_speedup", 0) >= 1.0)
check("V9: best_e2e_event_ns = round2 的 90us",
      abs(st["best_e2e_event_ns"] - BASE_EVT * 0.9) < 1.0,
      st.get("best_e2e_event_ns"))
print(f"  best_speedup={st.get('best_speedup')} best_round={st.get('best_round')} "
      f"best_e2e_event_ns={st.get('best_e2e_event_ns')}")

print("\n═══ V9 (首次 Event 对比基线) 全部通过 ═══")
