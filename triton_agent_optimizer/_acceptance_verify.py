# -*- coding: utf-8 -*-
"""V7: feedback/acceptance_report.py — mock 假 outputs 数据验证提取/除法/判定."""
import json, sys, tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

tmp = Path(tempfile.mkdtemp())
out_root = tmp / "outputs"

def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"{name}: {detail}"

# 假数据: 3 个算子 (快于工业级/打平/有空间)
def _mk(op, our_ns, ind_us, speedup, rounds, tier, with_final_summary=True):
    kd = out_root / op
    kd.mkdir(parents=True)
    st = {"best_speedup": speedup, "best_e2e_event_ns": our_ns,
          "industrial_time_us": ind_us, "total_rounds": rounds, "tier": tier}
    (kd / "optimization_trajectory.json").write_text(
        json.dumps({"state": st, "history": []}), encoding="utf-8")
    if with_final_summary:
        (kd / "final_output").mkdir(exist_ok=True)
        (kd / "final_output" / "final_summary.json").write_text(json.dumps({
            "op": op, "our_best_e2e_event_ns": our_ns, "industrial_time_us": ind_us,
            "industrial_baseline": f"industrial_{op}_compile_tflops.json",
            "best_speedup": speedup, "total_rounds": rounds, "final_tier": tier,
        }), encoding="utf-8")

_mk("matmul", 4_000_000.0, 3_000.0, 1.5, 30, 3)          # 我们 4000us 工业级 3000us → 0.75x (有空间)
_mk("vector_add", 800_000.0, 900.0, 2.0, 12, 4)           # 我们 800us 工业级 900us → 1.125x (快于)
_mk("rms_norm", 1_000_000.0, 1_050.0, 1.2, 8, 2)          # 我们 1000us 工业级 1050us → 1.05x (快于)
_mk("flash_attention", 2_000_000.0, 1_900.0, 1.3, 20, 5, with_final_summary=False)  # 无 final_summary → 回退 trajectory

import feedback.acceptance_report as AR

with mock.patch.object(AR, "_PROJECT_DIR", tmp), \
     mock.patch.object(sys, "argv", ["acceptance_report.py"]):
    rc = AR.main()
check("V7: 退出码 0", rc == 0)

rows = json.loads((out_root / "acceptance_summary.json").read_text(encoding="utf-8"))["ops"]
d = {r["op"]: r for r in rows}

# matmul: 4000us / 3000us = 0.75 → 有空间
check("V7: matmul 验收 0.75", d["matmul"]["acceptance_x"] == 0.75, d["matmul"]["acceptance_x"])
check("V7: matmul 判定有空间", "有空间" in d["matmul"]["verdict"])
# vector_add: 800/900 → 1.125 快于
check("V7: vector_add 验收 1.125", abs(d["vector_add"]["acceptance_x"] - 1.125) < 0.001)
check("V7: vector_add 判定快于", "快于" in d["vector_add"]["verdict"])
# rms_norm: 1000/1050 → 1.05
check("V7: rms_norm 验收 1.05", abs(d["rms_norm"]["acceptance_x"] - 1.05) < 0.001)
# flash_attention: 无 final_summary → 回退 trajectory state 也能算
check("V7: FA 回退 trajectory 算验收", d["flash_attention"]["acceptance_x"] is not None
      and abs(d["flash_attention"]["acceptance_x"] - 0.95) < 0.001,
      d["flash_attention"]["acceptance_x"])
check("V7: FA industrial_baseline 缺省", d["flash_attention"].get("industrial_baseline") is None)

# --md 模式
with mock.patch.object(AR, "_PROJECT_DIR", tmp), \
     mock.patch.object(sys, "argv", ["acceptance_report.py", "--md"]):
    AR.main()
check("V7: --md 写 acceptance_report.md", (out_root / "acceptance_report.md").exists())

# --op 过滤
with mock.patch.object(AR, "_PROJECT_DIR", tmp), \
     mock.patch.object(sys, "argv", ["acceptance_report.py", "--op", "matmul"]):
    rc2 = AR.main()
check("V7: --op 过滤退出码 0", rc2 == 0)

print("\n═══ V7 (acceptance_report) 全部通过 ═══")
