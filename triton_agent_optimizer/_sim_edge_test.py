#!/usr/bin/env python3
"""边界情况测试 #2 — 各种失败/回退/晋升/目标/基准场景, 全部真实执行 scheduler/verifier/coder 代码.
桩: 真机 msprof/HIVM/LLM (与 _sim_flow_test.py 同法).
"""
import json, os, re, shutil, subprocess, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sim_flow_test import canned_diagnosis, FAKE_KERNEL, setup, teardown, OP, OUT
from agents.scheduler import Scheduler, TIER_NAMES
from agents.planner import RoundPlan, PlannerAgent
import agents.verifier as V_mod
import agents.scheduler as S_mod

def make_sched(optimize_fn, verify_fn, max_rounds=15, target=0.0):
    shutil.rmtree(OUT, ignore_errors=True)
    s = Scheduler(OP, max_rounds=max_rounds, target_speedup=target, stub=True)
    s._run_optimize = types.MethodType(optimize_fn, s)
    s._run_fusion = types.MethodType(lambda self, rd: {"op_count": 5, "fusion_candidates": []}, s)
    s._sim_calls = []
    V_mod.verify_end_to_end = verify_fn
    return s

def read_hist():
    t = json.loads((OUT / "optimization_trajectory.json").read_text(encoding="utf-8"))
    return t["state"], t["history"]

def flip_gen(promote_rule=None):
    """返回 generate_v4: 翻转 BLOCK, promote 由 promote_rule(tier,round_num)->(promote,promote_to) 决定."""
    def gen(self, extracted, tier, history, kernel_code, round_num, **kw):
        m = re.search(r"(BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, (\d+))", kernel_code)
        old = m.group(1) if m else "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32"
        val = m.group(2) if m else "32"
        new = f"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, {'64' if val == '32' else '32'}"
        promote, promote_to = (0, 0)
        if promote_rule:
            promote, promote_to = promote_rule(tier, round_num)
        pt = json.dumps({"strategy": "s",
                         "changes": [{"old_code": old, "new_code": new, "reason": "t",
                                      "section": "① config", "tier": tier}],
                         "promote": promote, "promote_to": promote_to, "promote_reason": "sim" if promote else ""})
        return RoundPlan(round_num=round_num, tier=tier, tier_name="", strategy="s",
                         target_speedup=1.1, specific_change="", expected_impact="",
                         verification_method="msprof", plan_text=pt,
                         promote=promote, promote_to=promote_to, promote_reason="sim" if promote else "")
    return gen

def ok_verify(base=5_000_000.0):
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        rd = str(round_dir)
        if "baseline" in rd:
            return {"ok": True, "ns": base, "e2e_ns": base, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": base/1000}
        # ★strict-best: 优化轮须严格 > baseline(1.0) 才 KEEP → sp=1.05 (改进)
        return {"ok": True, "ns": base/1.05, "e2e_ns": base/1.05, "speedup": 1.05, "loop": 30, "rows": 90, "duration_us": base/1.05/1000}
    return verify

def optimize_ok(self, round_dir, tier):
    return canned_diagnosis()

# ═══ 1. 真实 verify_end_to_end (假 msprof 写 op_summary) ═══
def test_verify_e2e():
    from agents.verifier import verify_end_to_end
    V = V_mod
    results = {"csv": ""}
    def fake_run(cmd, *a, **kw):
        if cmd[0] == "msprof":
            outdir = Path(cmd[1].split("=")[1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "op_summary_0.csv").write_text(results["csv"], encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        # 非 msprof (warmup / MATMUL_VERIFY 正确性校验): 返回 PASS 让正确性检查通过
        return subprocess.CompletedProcess(cmd, 0, "[info] result check: PASS", "")
    subprocess.run = fake_run   # verify_end_to_end 函数内 import subprocess → 补模块级 run

    kernel = OP / "kernel_op.py"
    rd = OUT / "t_verify" / "round1"
    # 1a 循环在 + 90 行 → ÷30
    kernel.write_text(FAKE_KERNEL, encoding="utf-8")
    rows = [r for _ in range(30) for r in
            ("10,matmul_kernel", "5,bias_gelu_kernel", "10,matmul_kernel2")]
    results["csv"] = "Task Duration(us),Op Name\n" + "\n".join(rows)
    r = verify_end_to_end(kernel, rd, 5_000_000, num_kernels=3)
    assert r["ok"] and r["ns"] == 25 * 1000, r
    print(f"[1a] ✅ 循环在+90行 → ÷30, ns={r['ns']} (期望25000)")

    # 1b 循环丢 + 3 行 → ÷实测遍数(1), 不虚高
    broken = FAKE_KERNEL.replace("for _ in range(LOOP):", "for _ in range(3):")
    kernel.write_text(broken, encoding="utf-8")
    results["csv"] = "Task Duration(us),Op Name\n10,matmul_kernel\n5,bias_gelu_kernel\n10,matmul_kernel2\n"
    r = verify_end_to_end(kernel, rd, 5_000_000, num_kernels=3)
    assert r["ok"] and r["ns"] == 25 * 1000, r
    print(f"[1b] ✅ 循环丢+3行 → ÷1, ns={r['ns']} (期望25000, 不虚高30x)")

    # 1c aclnn 行排除
    results["csv"] = ("Task Duration(us),Op Name\n10,matmul_kernel\n5,bias_gelu_kernel\n"
                      "10,matmul_kernel2\n99999,aclnnMatmul\n")
    r = verify_end_to_end(kernel, rd, 5_000_000, num_kernels=3)
    assert r["ns"] == 25 * 1000, r
    print(f"[1c] ✅ aclnn 排除, ns={r['ns']} (期望25000)")

    # 1d 无 op_summary → ok=False 不崩
    rd2 = OUT / "t_verify" / "round2"
    def fake_run_none(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")
    subprocess.run = fake_run_none
    r = verify_end_to_end(kernel, rd2, 5_000_000, num_kernels=3)
    assert not r["ok"] and "error" in r, r
    print(f"[1d] ✅ 无 op_summary → ok=False (不崩)")

# ═══ 2. coder old_code 不匹配 → NOOP/FAIL 不崩 ═══
def test_coder_missing():
    PlannerAgent.generate_v4 = flip_gen()
    def gen(self, *a, **kw):
        pt = json.dumps({"strategy": "bad",
                         "changes": [{"old_code": "NO_SUCH_LINE_XYZ", "new_code": "x=1",
                                      "reason": "t", "section": "① config", "tier": kw["tier"]}],
                         "promote": False, "promote_to": 0})
        return RoundPlan(round_num=kw["round_num"], tier=kw["tier"], tier_name="", strategy="bad",
                         target_speedup=1.1, specific_change="", expected_impact="",
                         verification_method="msprof", plan_text=pt, promote=False, promote_to=0, promote_reason="")
    PlannerAgent.generate_v4 = gen
    s = make_sched(optimize_ok, ok_verify(), max_rounds=4)
    s.run()
    _, hist = read_hist()
    assert len(hist) == 4, len(hist)
    assert all(h["result"] in ("NOOP", "FAIL") for h in hist), hist
    print(f"[2] ✅ coder old_code 不匹配: {len(hist)}轮全记录(NOOP/FAIL), 循环不崩")

# ═══ 3. verify 3 次全失败 → FAIL 不崩, 继续 ═══
def test_verify_3fail():
    PlannerAgent.generate_v4 = flip_gen()
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        if "baseline" in str(round_dir):
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        return {"ok": False, "error": "sim kernel run error", "speedup": 0.5, "ns": None}
    s = make_sched(optimize_ok, verify, max_rounds=4)
    s.run()
    _, hist = read_hist()
    assert len(hist) == 4, len(hist)
    # 重试3次都失败 → coder 带错误重试(stub返原码)→ NOOP, 记 FAIL, 循环继续(no_improve 兜底晋升到 tier2)
    assert all(h["decision"] == "FAIL" for h in hist), hist
    assert all(h["result"] in ("NOOP", "FAIL") for h in hist), hist
    assert hist[3]["tier"] == 2, hist[3]
    print(f"[3] ✅ verify 3次失败: 每轮重试3次→记FAIL(NOOP), 不崩, no_improve兜底晋升({len(hist)}轮)")

# ═══ 4. promote_to 晋升 1→2→3 再回退 3→1 ═══
def test_promote_backoff():
    PlannerAgent.generate_v4 = flip_gen(promote_rule=lambda t, rn: (True, {1: 2, 2: 3, 3: 1}.get(t, 0)))
    s = make_sched(optimize_ok, ok_verify(), max_rounds=10)
    s.run()
    _, hist = read_hist()
    tiers = [h["tier"] for h in hist]
    print(f"[4] tier 序列: {tiers}")
    assert tiers[0] == 1 and tiers[1] == 2 and tiers[2] == 3 and tiers[3] == 1, tiers
    print("[4] ✅ promote_to 晋升(1→2→3) + 回退(3→1) 正确")

# ═══ 5. ★D3: 目标加速比达标 → 不硬停, 继续探 (防过早停) ═══
def test_target_stop():
    PlannerAgent.generate_v4 = flip_gen()
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        if "baseline" in str(round_dir):
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        return {"ok": True, "ns": 5_000_000.0/1.3, "e2e_ns": 5_000_000.0/1.3, "speedup": 1.3, "loop": 30, "rows": 90, "duration_us": 5000.0}
    s = make_sched(optimize_ok, verify, max_rounds=15, target=1.2)
    s.run()
    st, hist = read_hist()
    # ★D3: 第1轮达 1.3x ≥ 1.2 → 达标不硬停, 继续探 (由 no_improve/max_rounds 收尾), 不止 1 轮
    assert len(hist) >= 2, f"D3 达标应继续探, got {len(hist)} 轮"
    assert st["best_speedup"] == 1.3
    assert hist[0]["speedup"] == 1.3
    print(f"[5] ✅ D3: target=1.2, round1 达 1.3x → 不硬停, 继续探 ({len(hist)} 轮), best={st['best_speedup']}x")

# ═══ 6. 连续 3 轮无改进 → 兜底晋升 ═══
def test_no_improve_promote():
    PlannerAgent.generate_v4 = flip_gen()
    speedups = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1}
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        rd = str(round_dir)
        if "baseline" in rd:
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        rn = int(re.search(r"round(\d+)", rd).group(1))
        sp = speedups.get(rn, 1.0)
        return {"ok": True, "ns": 5_000_000.0/sp, "e2e_ns": 5_000_000.0/sp, "speedup": sp, "loop": 30, "rows": 90, "duration_us": 5000.0}
    s = make_sched(optimize_ok, verify, max_rounds=8)
    s.run()
    _, hist = read_hist()
    tiers = [h["tier"] for h in hist]
    print(f"[6] tier 序列: {tiers[:5]}")
    assert tiers[0] == 1 and tiers[1] == 1 and tiers[2] == 1 and tiers[3] == 2, tiers[:4]
    print("[6] ✅ 连续3轮无改进 → 兜底晋升 tier1→tier2")

# ═══ 7. round1 采集失败跳过 → round2 设基准 ═══
def test_baseline_after_fail():
    PlannerAgent.generate_v4 = flip_gen()
    def optimize(self, round_dir, tier):
        rn = int(re.search(r"round(\d+)", str(round_dir)).group(1))
        if rn == 1:
            return None          # r1 两次都失败 → 跳过
        return canned_diagnosis()
    s = make_sched(optimize, ok_verify(), max_rounds=5)
    s.run()
    st, hist = read_hist()
    assert st["baseline_ns"] == 5_000_000.0, st
    assert hist[0]["round"] == 1 and hist[0]["decision"] == "FAIL", hist[0]   # r1 跳过
    assert hist[1]["round"] == 2 and hist[1]["decision"] == "KEEP", hist[1]   # r2 正常
    print("[7] ✅ r1 采集失败跳过 → r2 设基准并正常跑; baseline_ns 来自验证复测")

# ═══ 8. _verify 异常 → stub 兜底 ns 不 None (F1 对齐) ═══
def test_verify_exception():
    PlannerAgent.generate_v4 = flip_gen()
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        raise RuntimeError("sim msprof missing")
    s = make_sched(optimize_ok, verify, max_rounds=3)
    s.run()
    st, hist = read_hist()
    assert len(hist) == 3, len(hist)
    # stub 兜底 speedup=1.0 → prev=1.0 → KEEP; ns = baseline 反推 (非 None)
    for h in hist:
        assert h["ns"] is not None, h
    print(f"[8] ✅ _verify 异常 → stub 兜底 ns 反推非None ({hist[0]['ns']}), 不崩")

if __name__ == "__main__":
    setup()
    try:
        test_verify_e2e(); print()
        test_coder_missing(); print()
        test_verify_3fail(); print()
        test_promote_backoff(); print()
        test_target_stop(); print()
        test_no_improve_promote(); print()
        test_baseline_after_fail(); print()
        test_verify_exception(); print()
        print("═══ 边界测试全部通过 ═══")
    finally:
        teardown()
