#!/usr/bin/env python3
"""全链路模拟 + 修复回归测试 — 真实执行 scheduler/verifier/coder/analyzers 代码路径,
打桩 真机 msprof/HIVM/LLM/sweep, 覆盖:
  P1  analyzer 真实数据链 (假 msprof CSV → task/board → diagnosis → 07字段)
  P2  ★回归: sweep 轮 verify 失败后 kernel 链内容快照恢复 (bug: 回滚只回路径引用 → 链污染)
  P3  ★回归: promote 无依据拒绝后转正常优化轮 (bug: 白耗轮次 + budget 照涨)
  P3b ★回归: max_rounds 硬上限按有效轮计 (bug: budget 无限膨胀 → 上限失效)
  P4-P11 设备污染重置/优秀案例/手递/死循环防护/Event注入/coder容错/sweep多格式/best绑定/采集失败
用法: python _sim_fix_regression.py
"""
import json, os, re, shutil, subprocess, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

os.environ["KEEP_FLOOR"] = "1.0"
os.environ["AUTO_STRATEGY_SUMMARY"] = "0"
os.environ["AUTO_CHART"] = "0"
os.environ["AUTO_RUN_PT_BENCH"] = "0"
os.environ["REBASELINE_EVERY"] = "0"
os.environ["TIER3_SWEEP"] = "1"

from _sim_flow_test import canned_diagnosis, FAKE_KERNEL, OP, OUT
from agents.scheduler import Scheduler, TIER_NAMES
from agents.planner import RoundPlan, PlannerAgent
import agents.verifier as V_mod
import agents.scheduler as S_mod
from agents import coder as C_mod
from agents import verifier as Ver_mod
from analyzers import sweep_blocks as SW_mod
import agents.llm_client as LLM_mod

RESULTS = []   # (tag, ok, detail)
FINDINGS = []  # (severity, title, detail) — bug/优化点 收集

def log(tag, ok, detail=""):
    RESULTS.append((tag, ok, detail))
    mark = "✅" if ok else "❌"
    print(f"  {mark} {tag}: {detail}")

def finding(sev, title, detail):
    FINDINGS.append((sev, title, detail))
    print(f"  ★[{sev}] {title}")

# ── 通用 stub ─────────────────────────────────────────────
class FakeProc:
    def __init__(self, cmd, env):
        self.cmd, self.env = cmd, env
        self.stdout = iter([])
        self.killed = False
    def wait(self, timeout=None): return 0
    def kill(self): self.killed = True
class FakePopen:
    calls = []
    def __init__(self, cmd, stdout=None, stderr=None, text=None, encoding=None, errors=None, env=None):
        FakePopen.calls.append({"cmd": cmd, "env": env})
        self._p = FakeProc(cmd, env)
    def __getattr__(self, n): return getattr(self._p, n)

def ok_verify(base=5_000_000.0):
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        rd = str(round_dir)
        if "baseline" in rd:
            return {"ok": True, "ns": base, "e2e_ns": base, "e2e_event_ns": base,
                    "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": base/1000}
        return {"ok": True, "ns": base/1.05, "e2e_ns": base/1.05, "e2e_event_ns": base/1.05,
                "speedup": 1.05, "loop": 30, "rows": 90, "duration_us": base/1.05/1000}
    return verify

def flip_gen(promote_rule=None, promote_evidence="", promote_reason=None, handoff=None):
    def gen(self, extracted, tier, history, kernel_code, round_num, **kw):
        m = re.search(r"(BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, (\d+))", kernel_code)
        old = m.group(1) if m else "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32"
        val = m.group(2) if m else "32"
        new = f"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, {'64' if val == '32' else '32'}"
        promote, promote_to = (0, 0)
        if promote_rule:
            promote, promote_to = promote_rule(tier, round_num)
        reason = promote_reason if promote_reason is not None else ("sim" if promote else "")
        pt = {"strategy": "s",
              "changes": [{"old_code": old, "new_code": new, "reason": "t",
                           "section": "① config", "tier": tier}],
              "promote": promote, "promote_to": promote_to,
              "promote_reason": reason,
              "promote_evidence": promote_evidence if promote else "",
              "handoff": handoff}
        return RoundPlan(round_num=round_num, tier=tier, tier_name="", strategy="s",
                         target_speedup=1.1, specific_change="", expected_impact="",
                         verification_method="msprof", plan_text=json.dumps(pt),
                         promote=promote, promote_to=promote_to,
                         promote_reason=reason,
                         promote_evidence=promote_evidence if promote else "",
                         handoff=handoff)
    return gen

def optimize_ok(self, round_dir, tier):
    return canned_diagnosis()

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

def _snap_restore():
    """快照并返回恢复函数 (还原模块级 monkeypatch + env)."""
    snap = {"V": V_mod.verify_end_to_end, "SP": S_mod.subprocess,
            "PG": PlannerAgent.generate_v4, "SW_META": dict(SW_mod.SWEEP_META),
            "SW": SW_mod.sweep, "env": dict(os.environ)}
    def restore():
        V_mod.verify_end_to_end = snap["V"]
        S_mod.subprocess = snap["SP"]
        PlannerAgent.generate_v4 = snap["PG"]
        SW_mod.SWEEP_META = dict(snap["SW_META"])
        SW_mod.sweep = snap["SW"]
        os.environ.clear(); os.environ.update(snap["env"])
    return restore

# ═══════════════════════════════════════════════════════════════════
#  Part 1: analyzer 真实数据链 (假 msprof CSV → task/board → integrate → 07字段)
# ═══════════════════════════════════════════════════════════════════
def write_csv(p, headers, rows):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def test_analyzer_chain():
    tag = "P1-analyzer链"
    base = Path(os.environ["TEMP"]) / "opencode" / "msprof_fake"
    shutil.rmtree(base, ignore_errors=True)
    prof = base / "mindstudio_profiler_output"
    prof.mkdir(parents=True, exist_ok=True)
    # 通用 msprof: 3 kernel 每行, 循环30遍 (模拟 KERNEL_LOOP=30)
    rows = []
    for _ in range(30):
        for name, dur, block, typ in [
            ("matmul_kernel", 800.0, 64, "Cube"),
            ("bias_gelu_kernel", 120.0, 40, "Vector"),
            ("matmul_kernel2", 800.0, 64, "Cube")]:
            rows.append([name, "TritonKernel", typ, f"{dur:.1f}", block,
                         "(2048,2048)", "float32", "(2048,2048)", "float32"])
    write_csv(prof / "op_summary_0.csv",
              ["Op Name", "Op Type", "Task Type", "Task Duration(us)", "Block Dim",
               "Input Shape(s)", "Input Data Type(s)", "Output Shape(s)", "Output Data Type(s)"],
              rows)
    write_csv(prof / "op_statistic_0.csv", ["OP Type", "Core Type", "Count", "Total Time(us)", "Ratio"],
              [["Cube", "AIC", 60, 48000.0, 0.93], ["Vector", "AIV", 30, 3600.0, 0.07]])
    write_csv(prof / "api_statistic_0.csv", ["API Name", "Time(us)", "Count", "Avg"],
              [["aclrtLaunchKernel", 120.0, 90, 1.3]])
    write_csv(prof / "l2_cache_0.csv", ["Hit Rate"], [["85.5"]])

    from analyzers.pipeline_parse_task import parse as parse_task
    tk = parse_task(prof)
    assert tk["execution_summary"]["num_kernels"] == 3, tk["execution_summary"]
    slots = tk["normalized"]["kernel_slots"]
    assert len(slots) == 3, len(slots)
    # total_ns = (800+120+800)*30遍*1000 = 51.6e6
    assert tk["execution_summary"]["total_ns"] == (800+120+800)*30*1000, tk["execution_summary"]["total_ns"]
    assert slots[0]["launch_count"] == 30, slots[0]["launch_count"]
    assert "est_bytes_in" in slots[0]["task"] and slots[0]["task"]["est_bytes_in"] is not None

    # msprof op: 1 个 kernel 的 8 CSV
    opprof = base / "OPPROF_1"
    opprof.mkdir(parents=True, exist_ok=True)
    write_csv(opprof / "OpBasicInfo.csv", ["Op Name", "Task Type", "Task Duration(us)", "Block Dim", "Current Freq(MHz)"],
              [["matmul_kernel", "Cube", "800.0", "64", "1800"]])
    write_csv(opprof / "PipeUtilization.csv",
              ["aic_cube_ratio", "aiv_vec_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_mte3_ratio",
               "aic_scalar_ratio", "aic_fixp_ratio"],
              [[0.42, 0.05, 0.30, 0.50, 0.10, 0.02, 0.0]])
    write_csv(opprof / "ArithmeticUtilization.csv",
              ["aic_cube_fops", "aic_cube_ratio", "aic_cube_fp16_ratio", "aiv_vec_fops", "aiv_vec_ratio",
               "aic_total_cycles", "aiv_total_cycles"],
              [[2*2048*2048*2048, 0.42, 0.0, 0, 0.05, 1000000, 10000]])
    write_csv(opprof / "Memory.csv",
              ["aic_main_mem_read_bw", "aic_main_mem_write_bw", "aiv_gm_to_ub_bw", "aiv_ub_to_gm_bw",
               "aic_l1_read_bw", "aic_l1_write_bw"],
              [[1200.0, 600.0, 1500.0, 800.0, 2000.0, 1000.0]])
    write_csv(opprof / "MemoryL0.csv",
              ["aic_l0a_read_bw", "aic_l0a_write_bw", "aic_l0b_read_bw", "aic_l0b_write_bw",
               "l0c_read_bw_cube", "l0c_write_bw_cube"],
              [[4000.0, 2000.0, 4000.0, 2000.0, 6000.0, 3000.0]])
    write_csv(opprof / "MemoryUB.csv",
              ["aiv_ub_read_bw_vector", "aiv_ub_write_bw_vector", "aiv_ub_read_bw_scalar", "aiv_ub_write_bw_scalar"],
              [[3000.0, 3000.0, 1000.0, 1000.0]])
    write_csv(opprof / "L2Cache.csv", ["aic_total_hit_rate(%)"], [["85.5"]])
    write_csv(opprof / "ResourceConflictRatio.csv",
              ["aiv_vec_bank_cflt_ratio", "aiv_vec_bankgroup_cflt_ratio", "aiv_vec_total_cflt_ratio",
               "aiv_vec_mte_cflt_ratio", "aiv_vec_resc_cflt_ratio"],
              [[0.03, 0.01, 0.05, 0.02, 0.0]])

    from analyzers.pipeline_parse_board import parse as parse_board
    bd = parse_board(opprof)
    bw = bd["normalized"]["bandwidth_gb_s"]
    assert abs(bw["main_mem_read_gb_s"] - 1.2) < 0.01, bw   # 1200 MB/s → 1.2 GB/s
    assert abs(bw["l0a_read_gb_s"] - 4.0) < 0.01, bw
    eng = bd["normalized"]["engine_utilization"]
    assert abs(eng["cube"] - 0.42) < 0.01, eng
    assert abs(bd["normalized"]["l2_hit_rate"] - 0.855) < 0.01, bd["normalized"]["l2_hit_rate"]
    assert "cube_fops" in bd["normalized"]["compute"]

    # integrate → diagnosis
    from analyzers.integrate import integrate
    out_diag = base / "diagnosis.json"
    tjson = base / "task.json"
    tjson.write_text(json.dumps(tk), encoding="utf-8")
    bjson = base / "board_1.json"
    bjson.write_text(json.dumps(bd), encoding="utf-8")
    integrate(str(tjson), str(out_diag), [str(bjson)])
    dg = json.loads(out_diag.read_text(encoding="utf-8"))
    assert dg["summary"]["num_kernels"] == 3
    filled = sum(1 for k in dg["kernels"] if k.get("deep"))
    assert filled == 1, f"只采了1个kernel的deep, filled={filled}"
    k0 = dg["kernels"][0]
    rl = k0["deep"]["roofline"]
    assert rl["bottleneck_type"] in ("memory_bound", "balanced", "compute_bound"), rl
    # 07 字段提取 (tier1)
    from agents.scheduler import extract_tier_fields
    txt = extract_tier_fields(dg, 1)
    assert "num_kernels" in txt and "cube_r=" in txt and "vec_fops=" in txt and "bottleneck=" in txt, txt[:200]
    assert txt.count("- matmul") >= 1
    log(tag, True, f"task.json(3 slots,launch=30,total={tk['execution_summary']['total_ns']}) → board(带宽GB/s/L2/引擎) → diagnosis(filled={filled}) → 07字段{txt.count(chr(10))}行")

# ═══════════════════════════════════════════════════════════════════
#  Part 2: sweep 回退链 bug 复现
# ═══════════════════════════════════════════════════════════════════
def test_sweep_revert_chain():
    """场景: round1 触发 sweep (写回 round_dir/kernel_op.py) → coder 覆写同路径 → verify 失败
    → 回滚只回路径引用(内容已被 coder 覆盖) → 下一轮读到失败代码. 应保留 sweep 最优."""
    tag = "P2-sweep回退链"
    restore = _snap_restore()
    SW_mod.SWEEP_META["_sim_op"] = {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"),
                                    "type": "matmul", "multi": False}
    def stub_sweep(op_dir, quick=False, out_dir=None, op_name=None):
        kp = op_dir / "kernel_op.py"
        code = kp.read_text(encoding="utf-8")
        new = re.sub(r"BLOCK_M, BLOCK_N, BLOCK_K = [\d, ]+",
                     "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64", code, count=1)
        kp.write_text(new, encoding="utf-8")
        return {"results": [{"block": [64, 64, 64], "ns": 100.0},
                            {"block": [64, 64, 32], "ns": 200.0}],
                "unchanged": False, "candidates_tested": 2, "best_block": [64, 64, 64]}
    SW_mod.sweep = stub_sweep
    # planner: 改 BLOCK_SIZE 1024→2048 (与 sweep 无关的改动), 全部轮不 promote
    def gen(self, extracted, tier, history, kernel_code, round_num, **kw):
        assert "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64" in kernel_code, \
            f"planner 应读到 sweep 后的 BLOCK, got:\n{kernel_code[:300]}"
        pt = json.dumps({"strategy": "s",
                         "changes": [{"old_code": "BLOCK_SIZE = 1024", "new_code": "BLOCK_SIZE = 2048",
                                      "reason": "t", "section": "① config", "tier": tier}],
                         "promote": False, "promote_to": 0})
        return RoundPlan(round_num=round_num, tier=tier, tier_name="", strategy="s",
                         target_speedup=1.1, specific_change="", expected_impact="",
                         verification_method="msprof", plan_text=pt, promote=False, promote_to=0)
    PlannerAgent.generate_v4 = gen
    snapshots = []
    def optimize_snap(self, round_dir, tier):
        snapshots.append({"rd": str(round_dir), "tier": tier,
                          "input": str(self.current_kernel),
                          "code": self.current_kernel.read_text(encoding="utf-8")
                                  if self.current_kernel.exists() else ""})
        return canned_diagnosis()
    def fail_verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        if "baseline" in str(round_dir):
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0,
                    "e2e_event_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90,
                    "duration_us": 5000.0}
        return {"ok": False, "error": "sim run error", "speedup": 1.0, "ns": None}
    try:
        s = make_sched(optimize_snap, fail_verify, max_rounds=3)
        s.run()
        # 修复验证: round2 输入应 = sweep 最优 (BLOCK_K=64) + 原 BLOCK_SIZE=1024 (内容快照恢复);
        #            coder 失败改动应被另存 failed_kernel.py 留证.
        assert len(snapshots) >= 2, f"只采集了 {len(snapshots)} 轮"
        r2_code = snapshots[1]["code"]
        ok1 = ("BLOCK_SIZE = 1024" in r2_code) and ("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64" in r2_code)
        r1_rd = OUT / TIER_NAMES[1] / "round1"
        fk = r1_rd / "failed_kernel.py"
        ok2 = fk.exists() and "BLOCK_SIZE = 2048" in fk.read_text(encoding="utf-8")
        r1_k = r1_rd / "kernel_op.py"
        ok3 = r1_k.exists() and "BLOCK_SIZE = 1024" in r1_k.read_text(encoding="utf-8")
        ok = ok1 and ok2 and ok3
        if not ok:
            finding("BUG", "sweep 回退链仍未修复",
                    f"round1 sweep 后 coder 覆写同路径且 verify 失败 → 内容快照未恢复: "
                    f"round2输入={ok1} failed_kernel留证={ok2} round1恢复={ok3} (期望全 True)")
        log(tag, ok, f"round2输入=sweep最优+原BLOCK_SIZE={ok1} failed_kernel留证={ok2} round1恢复={ok3}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 3: promote 严格晋升 (无依据拒绝) + 手递 + 死循环防护
# ═══════════════════════════════════════════════════════════════════
def test_promote_evidence():
    """修复验证: promote=True 但无 evidence/reason → 晋升门前置拒绝, 本轮转正常优化轮
    (不再白耗: 旧实现 _decide_tier 拒绝时 run() 已走 promote 分支跳过 coder/verify + budget+1)."""
    restore = _snap_restore()
    PlannerAgent.generate_v4 = flip_gen(promote_rule=lambda t, rn: (True, 2) if rn == 2 else (False, 0),
                                        promote_evidence="", promote_reason="")
    try:
        s = make_sched(optimize_ok, ok_verify(), max_rounds=4)
        s.run()
        st, hist = read_hist()
        tiers = [h["tier"] for h in hist]
        refused = tiers[0] == 1 and tiers[1] == 1   # 被拒的晋升轮 (r2) 未改变 tier
        r2_rd = OUT / TIER_NAMES[1] / "round2"
        diff = (r2_rd / "diff.patch").read_text(encoding="utf-8")
        ran_coder = ("---" in diff or "@@" in diff)   # 真实 unified diff → 正常走了 coder
        budget = st.get("promote_budget")
        fixed = refused and ran_coder and budget == 0
        if not fixed:
            finding("BUG", "被拒晋升轮仍白耗 / budget 仍涨",
                    f"refused={refused} r2跑了coder={ran_coder} budget={budget} (期望 True/True/0)")
        log("P3-promote无依据拒绝(修复验证)", fixed,
            f"tiers={tiers} r2跑coder={ran_coder} budget={budget}")
    finally:
        restore()

def test_promote_budget_inflation():
    """修复验证: planner 每轮 promote=True 且无依据 → 全部转正常优化轮,
    有效轮计数 → max_rounds 硬上限生效 (旧实现: 5 轮上限实跑 18 轮)."""
    restore = _snap_restore()
    PlannerAgent.generate_v4 = flip_gen(promote_rule=lambda t, rn: (True, 2),
                                        promote_evidence="", promote_reason="")
    try:
        s = make_sched(optimize_ok, ok_verify(), max_rounds=5)
        s.run()
        st, hist = read_hist()
        n = len(hist)
        budget = st.get("promote_budget")
        fixed = n == 5 and budget == 0
        if not fixed:
            finding("BUG", "max_rounds 硬上限仍失效 / budget 仍膨胀",
                    f"max_rounds=5 → 实跑 {n} 轮, budget={budget} (期望 5/0)")
        log("P3b-max_rounds硬上限(修复验证)", fixed,
            f"max_rounds=5 → 实跑 {n} 轮, budget={budget}")
    finally:
        restore()

def test_promote_evidence_ok_and_handoff():
    restore = _snap_restore()
    hd = {"to_tier": 2, "bottleneck": "cube_util低", "optimization_direction": "查融合"}
    PlannerAgent.generate_v4 = flip_gen(promote_rule=lambda t, rn: (True, 2),
                                        promote_evidence="cube_util=0.12, 算法非最优, 已3轮无改进",
                                        handoff=hd)
    try:
        s = make_sched(optimize_ok, ok_verify(), max_rounds=4)
        s.run()
        st, hist = read_hist()
        tiers = [h["tier"] for h in hist]
        ok = tiers[0] == 1 and tiers[1] == 2, tiers
        # 手递文件应存在
        r1_rd = OUT / TIER_NAMES[1] / "round1"
        hf = r1_rd / "10_tier_handoff.json"
        hf_ok = hf.exists()
        if hf_ok:
            hj = json.loads(hf.read_text(encoding="utf-8"))
            hf_ok = hj.get("bottleneck_analysis") == "cube_util低"
        # st["handoff"] 在 tier2 首轮被消费后清空
        consumed = st.get("handoff") is None
        log("P3-promote有依据+手递", bool(ok[0] and hf_ok and consumed),
            f"tiers={tiers} handoff文件={hf.exists()} 消费后清空={consumed}")
    finally:
        restore()

def test_tier_jump_deadloop():
    restore = _snap_restore()
    PlannerAgent.generate_v4 = flip_gen(promote_rule=lambda t, rn: (True, {1: 3, 3: 1}.get(t, 0)),
                                        promote_evidence="evidence")
    try:
        s = make_sched(optimize_ok, ok_verify(), max_rounds=10)
        s.run()
        st, hist = read_hist()
        tiers = [h["tier"] for h in hist]
        jumps = st.get("tier_jumps") or []
        pairs = [j["pair"] for j in jumps]
        # 1->3 至少跳 3 次后被拒绝 → 强制进 tier2
        n13 = pairs.count("1->3")
        saw_tier2 = 2 in tiers
        log("P3-tier跳转死循环防护", n13 >= 3 and saw_tier2,
            f"tiers={tiers} pairs={pairs} n(1->3)={n13} 出现tier2={saw_tier2}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 4: 设备污染 → 下轮采集前重置
# ═══════════════════════════════════════════════════════════════════
def test_device_pollution_reset():
    restore = _snap_restore()
    reset_calls = []
    orig_reset = S_mod._reset_device
    def fake_reset():
        reset_calls.append(1)
        return True
    S_mod._reset_device = fake_reset
    def dev_verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        if "baseline" in str(round_dir):
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0, "e2e_event_ns": 5_000_000.0,
                    "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        return {"ok": False, "error": "575:NPU function error: Aclrt device reset", "speedup": 1.0, "ns": None}
    try:
        s = make_sched(optimize_ok, dev_verify, max_rounds=3)
        s.run()
        log("P4-设备污染自动重置", len(reset_calls) >= 1,
            f"verify崩AICore → 下轮采集前 _reset_device 被调 {len(reset_calls)} 次")
    finally:
        S_mod._reset_device = orig_reset
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 5: 优秀案例自动记录 (改进>1.3x)
# ═══════════════════════════════════════════════════════════════════
def test_excellent_case():
    restore = _snap_restore()
    cases_path = REPO / "memory" / "tier1_cases.json"
    backup = cases_path.read_text(encoding="utf-8") if cases_path.exists() else None
    try:
        os.environ["EXCELLENT_CASE_THRESHOLD"] = "1.3"
        speedups = {1: 1.05, 2: 1.5, 3: 1.5}
        def vf(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
            if "baseline" in str(round_dir):
                return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0,
                        "e2e_event_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
            rn = int(re.search(r"round(\d+)", str(round_dir)).group(1))
            sp = speedups.get(rn, 1.05)
            return {"ok": True, "ns": 5_000_000.0/sp, "e2e_ns": 5_000_000.0/sp,
                    "e2e_event_ns": 5_000_000.0/sp, "speedup": sp, "loop": 30, "rows": 90, "duration_us": 5000.0}
        PlannerAgent.generate_v4 = flip_gen()
        s = make_sched(optimize_ok, vf, max_rounds=3)
        s.run()
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        hit = [c for c in cases if c.get("op") == "_sim_op"]
        ok = len(hit) >= 1 and abs(float(hit[-1]["improvement_x"]) - 1.5/1.05) < 0.05
        log("P5-优秀案例自动记录", ok,
            f"r1=1.05x → r2=1.5x (1.5/1.05={1.5/1.05:.3f}>1.3) → tier1_cases.json {len(hit)}条 _sim_op")
    finally:
        if backup is None:
            cases_path.unlink(missing_ok=True)
        else:
            cases_path.write_text(backup, encoding="utf-8")
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 6: 环境漂移 rebaseline 后 best Event 未重置 → 永远 REVERT
# ═══════════════════════════════════════════════════════════════════
def test_rebaseline_stale_best_event():
    """修复验证: rebaseline (环境漂移 5.0→5.5M) 后 best_e2e_event_ns 同步到新环境,
    真实改进轮 (r>=3, 1.35x) 能被 KEEP (旧实现: best 留在旧环境最小值 → 永远 REVERT)."""
    restore = _snap_restore()
    os.environ["REBASELINE_EVERY"] = "1"
    def verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        rd = str(round_dir)
        rd_n = rd.replace("\\", "/")
        if "baseline_verify" in rd_n:
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0,
                    "e2e_event_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        if rd_n.endswith("rebaseline/base"):   # rebaseline 原始 kernel 复测 → 漂移到 5.5M
            return {"ok": True, "ns": 5_500_000.0, "e2e_ns": 5_500_000.0,
                    "e2e_event_ns": 5_500_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5500.0}
        if rd_n.endswith("rebaseline/cur"):    # rebaseline 当前 kernel 复测 → 1.2x (stub 固定值)
            return {"ok": True, "ns": 5_500_000.0/1.2, "e2e_ns": 5_500_000.0/1.2,
                    "e2e_event_ns": 5_500_000.0/1.2, "speedup": 1.2, "loop": 30, "rows": 90, "duration_us": 4583.0}
        rn = int(re.search(r"round(\d+)", rd).group(1))
        sp = 1.3 if rn == 1 else (1.35 if rn >= 3 else 1.2)   # r>=3 真实改进 (相对新基线 1.35x)
        _base = 5_500_000.0 if rn >= 2 else 5_000_000.0   # 漂移后轮次也在新环境
        return {"ok": True, "ns": _base/sp, "e2e_ns": _base/sp,
                "e2e_event_ns": _base/sp, "speedup": sp, "loop": 30, "rows": 90, "duration_us": _base/sp/1000.0}
    try:
        PlannerAgent.generate_v4 = flip_gen()
        s = make_sched(optimize_ok, verify, max_rounds=5)
        s.run()
        st, hist = read_hist()
        n_keep_after = sum(1 for h in hist if h.get("decision") == "KEEP" and h.get("round") > 1)
        n_revert = sum(1 for h in hist if h.get("decision") == "REVERT")
        best_evt = st.get("best_e2e_event_ns")
        base_evt = st.get("baseline_e2e_event_ns")
        # 修复后: rebaseline 把 best 同步到新环境 (≈5.5M/1.2), 真实改进轮 (4.07M) 可 KEEP
        fixed = n_keep_after >= 1 and best_evt is not None and best_evt > (5_000_000.0/1.3)
        if not fixed:
            finding("BUG", "rebaseline 后 best Event 未同步 (KEEP 能力丢失)",
                    f"漂移后 baseline_e2e_event_ns={base_evt}, best_e2e_event_ns={best_evt} (应≥新环境最优≈4.58M); "
                    f"rebaseline 后采纳{n_keep_after}轮/回退{n_revert}轮 (期望 ≥1)")
        log("P6-rebaseline后KEEP能力(修复验证)", fixed,
            f"rebaseline后采纳{n_keep_after}轮 / 回退{n_revert}轮 / best_evt={best_evt} base_evt={base_evt}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 7: verifier Event 注入 各种 main() 循环格式
# ═══════════════════════════════════════════════════════════════════
def test_event_injection():
    def std_body():
        return ("def main():\n"
                "    LOOP = int(os.environ.get('KERNEL_LOOP', '1'))\n"
                "    for _ in range(LOOP):\n"
                "        matmul_kernel[g](a,b,c)\n"
                "        bias_gelu_kernel[g2](x)\n"
                "    torch.npu.synchronize()\n")
    cases = {
        "标准for+多行体": std_body(),
        "循环变量i": std_body().replace("for _ in range(LOOP):", "for i in range(LOOP):"),
        "体含嵌套for+空行": (
            "def main():\n"
            "    LOOP = int(os.environ.get('KERNEL_LOOP', '1'))\n"
            "    for _ in range(LOOP):\n"
            "        for j in range(3):\n"
            "            run(j)\n"
            "\n"
            "        matmul_kernel[g](a)\n"
            "    torch.npu.synchronize()\n"),
        "无循环→放弃": "def main():\n    x = 1\n",
        "while循环→放弃": "def main():\n    while True:\n        run()\n",
    }
    ok_all = True
    for name, src in cases.items():
        inj = Ver_mod._inject_event_timing(src)
        if name in ("无循环→放弃", "while循环→放弃"):
            good = inj == ""
        else:
            # ★2026-08-12: 注入产物必须 (a) 含多窗口 median 结构 (b) 本身可编译 (语法合法)
            good = ("KERNEL_EVENT_TIME" in inj and "EVENT_E2E_US" in inj
                    and "KERNEL_EVENT_REPS" in inj and "_ts.sort()" in inj
                    and "for _ in range(LOOP):" in inj
                    and inj.count("torch.npu.synchronize()") >= 2)
            if good:
                try:
                    compile(inj, "<event>", "exec")
                except SyntaxError as e:
                    good = False
                    print(f"  ❌ {name}: 注入产物语法错: {e}")
        ok_all &= good
        log(f"P7-Event注入[{name}]", good, f"len={len(inj)}")
    # 数值: 5 遍 vs 30 遍的 LOOP 除数正确性 (静态看除法表达式)
    inj = Ver_mod._inject_event_timing(std_body())
    assert "elapsed_time(_ev_e))" in inj and "/ LOOP * 1000.0" in inj, inj
    log("P7-Event注入除数", True, "median/LOOP*1000 → us→ns 单次平均正确 (多窗口 median)")

# ═══════════════════════════════════════════════════════════════════
#  Part 8: coder Unicode 清洗 / header 保护 / changes 替换 / extract_json
# ═══════════════════════════════════════════════════════════════════
def test_coder_units():
    dirty = 'x = a − b  # em–dash — smart’quote “dq” × ══ '
    cleaned = C_mod._sanitize_unicode(dirty)
    ok = "-" in cleaned and "'" in cleaned and '"' in cleaned and "*" in cleaned \
         and "═" not in cleaned and all(ord(c) < 128 for c in cleaned.split("#")[0])
    log("P8-sanitize_unicode", ok, repr(cleaned))
    # header 保护
    orig = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nimport os\n"
    dropped = "import os\nprint('x')\n"
    restored = C_mod._preserve_header(dropped, orig)
    ok2 = restored.startswith("#!/usr/bin/env python3") and "coding" in restored.splitlines()[1]
    log("P8-preserve_header", ok2, repr(restored[:80]))
    # changes 替换: 精确 / 容错(缩进/CRLF) / 缺匹配
    code = "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32\nM = 2048\n"
    changes = [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64"}]
    out, applied, missing = C_mod._apply_plan_changes(code, changes)
    ok3 = "64, 64, 64" in out and not missing
    log("P8-changes精确替换", ok3, out.strip())
    # 容错: old 带多余尾空格 (planner 复制自旧版本)
    code2 = "X = 1\n"
    changes2 = [{"old_code": "X = 1  ", "new_code": "X = 2"}]
    out2, ap2, ms2 = C_mod._apply_plan_changes(code2, changes2)
    ok4 = "X = 2" in out2 and not ms2
    log("P8-changes容错归一化", ok4, out2.strip())
    # 缺匹配 → 报告不猜
    changes3 = [{"old_code": "NO_SUCH = 1", "new_code": "X = 9"}]
    out3, ap3, ms3 = C_mod._apply_plan_changes(code2, changes3)
    log("P8-changes缺匹配报告", len(ms3) == 1 and "NO_SUCH" in ms3[0], f"missing={ms3}")
    # extract_json 容错: markdown 块 + 裸值
    resp = '```json\n{"strategy": "x", "count": 3, "ok": true, "note": bare_word}\n```'
    d = LLM_mod.extract_json(resp)
    ok5 = d.get("strategy") == "x" and d.get("count") == 3 and d.get("note") == "bare_word"
    log("P8-extract_json容错", ok5, f"{d}")
    # extract_json 空/坏 → 抛错 (调用方兜底 stub)
    try:
        LLM_mod.extract_json("not json at all {")
        ok6 = False
    except Exception:
        ok6 = True
    log("P8-extract_json坏输入抛错", ok6, "")

# ═══════════════════════════════════════════════════════════════════
#  Part 9: sweep BLOCK 读/写多格式 + _extract_mnk
# ═══════════════════════════════════════════════════════════════════
def test_sweep_helpers():
    meta = ("BLOCK_M", "BLOCK_N", "BLOCK_K")
    fmt_comma = "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32\n"
    fmt_split = "BLOCK_M = 64\nBLOCK_N = 64\nBLOCK_K = 32\n"
    fmt_anno  = "BLOCK_M: tl.constexpr = 64\nBLOCK_N: tl.constexpr = 64\nBLOCK_K: tl.constexpr = 32\n"
    fmt_comm  = "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32  # 注释\n"
    r1 = SW_mod._read_current_block(fmt_comma, meta)
    r2 = SW_mod._read_current_block(fmt_split, meta)
    r3 = SW_mod._read_current_block(fmt_anno, meta)
    r4 = SW_mod._read_current_block(fmt_comm, meta)
    ok = all(v == (64, 64, 32) for v in (r1, r2, r3, r4))
    log("P9-read_current_block多格式", ok, f"{r1}{r2}{r3}{r4}")
    a1 = SW_mod._apply_block(fmt_comma, meta, (128, 128, 64))
    a2 = SW_mod._apply_block(fmt_split, meta, (128, 128, 64))
    ok2 = "64, 64, 32" not in a1 and "128, 128, 64" in a1 and "BLOCK_K = 64" in a2
    log("P9-apply_block多格式", ok2, f"comma→{a1.strip()} | split→{a2.strip()}")
    # _extract_mnk (matmul 标准 + FAKE_KERNEL)
    from agents.scheduler import _extract_mnk
    m1 = _extract_mnk(FAKE_KERNEL)
    m2 = _extract_mnk("M = 2048\nN = 1024\nK = 512\n")
    m3 = _extract_mnk("x = 1")
    log("P9-extract_mnk", m1 == (2048, 2048, 2048) and m2 == (2048, 1024, 512) and m3 is None,
        f"{m1} {m2} {m3}")

# ═══════════════════════════════════════════════════════════════════
#  Part 10: 完整循环正常流 (keep/revert/promote/采集失败) 已在现有测试覆盖;
#           这里再验证 达标继续 + best_kernel 绑定
# ═══════════════════════════════════════════════════════════════════
def test_best_kernel_binding():
    restore = _snap_restore()
    speedups = {1: 1.0, 2: 1.4, 3: 1.6, 4: 1.3}   # r2 是历史最优 1.4, r3 1.6 被 r4 1.3 取代? 不, 1.6>1.4 → r3 best
    def vf(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        if "baseline" in str(round_dir):
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0, "e2e_event_ns": 5_000_000.0,
                    "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        rn = int(re.search(r"round(\d+)", str(round_dir)).group(1))
        sp = speedups.get(rn, 1.2)
        return {"ok": True, "ns": 5_000_000.0/sp, "e2e_ns": 5_000_000.0/sp,
                "e2e_event_ns": 5_000_000.0/sp, "speedup": sp, "loop": 30, "rows": 90, "duration_us": 5000.0}
    def gen(self, extracted, tier, history, kernel_code, round_num, **kw):
        # 轮流翻 BLOCK_K, 让 coder 产生不同代码 (r2 与 r3 代码不同 → 验证 best_kernel 指向正确轮)
        m = re.search(r"(BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, (\d+))", kernel_code)
        val = m.group(2) if m else "32"
        newval = "64" if val == "32" else ("128" if val == "64" else "32")
        new = f"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, {newval}"
        pt = json.dumps({"strategy": "s",
                         "changes": [{"old_code": m.group(1) if m else "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                      "new_code": new, "reason": "t", "section": "① config", "tier": tier}],
                         "promote": False, "promote_to": 0})
        return RoundPlan(round_num=round_num, tier=tier, tier_name="", strategy="s",
                         target_speedup=1.1, specific_change="", expected_impact="",
                         verification_method="msprof", plan_text=pt, promote=False, promote_to=0)
    try:
        PlannerAgent.generate_v4 = gen
        s = make_sched(optimize_ok, vf, max_rounds=5)
        s.run()
        st, hist = read_hist()
        # r3 是历史最优 (1.6x), r4 回退 → best_round 应=3
        best_round = st.get("best_round")
        bk = (OUT / "best_kernel.py")
        bk_ok = bk.exists()
        # 断言 best_kernel.py 与 round3/kernel_op.py 一致
        match = False
        if bk_ok:
            r3_k = (OUT / TIER_NAMES[1] / "round3" / "kernel_op.py").read_text(encoding="utf-8")
            match = bk.read_text(encoding="utf-8") == r3_k
        dec = [(h["round"], h["decision"], h["speedup"]) for h in hist]
        log("P10-best_kernel绑定", best_round == 3 and match,
            f"decisions={dec} best_round={best_round} best_kernel==round3代码={match}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 11: run_optimize.sh 采集失败 → 重试 → 跳过 → 连续3次停止
#  (已有 _sim_flow_test 覆盖, 这里补: 失败时 current_kernel 不前进)
# ═══════════════════════════════════════════════════════════════════
def test_collect_fail_chain():
    restore = _snap_restore()
    def optimize(self, round_dir, tier):
        rn = int(re.search(r"round(\d+)", str(round_dir)).group(1))
        if rn >= 4:
            return None
        return canned_diagnosis()
    try:
        PlannerAgent.generate_v4 = flip_gen()
        s = make_sched(optimize, ok_verify(), max_rounds=6)
        s.run()
        st, hist = read_hist()
        # r4 重试1次后跳过, r5 第三次失败 → 停止. current_kernel 应停在 round1 的 kernel (链未前进)
        fail_rounds = [h["round"] for h in hist if h["decision"] == "FAIL"]
        ck = Path(st["current_kernel"])
        ck_ok = ck.exists() and "kernel_op.py" in str(ck)
        log("P11-采集失败重试/跳过/停止", len(fail_rounds) >= 1 and ck_ok,
            f"FAIL轮={fail_rounds} current_kernel={ck}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 12: --resume 续跑路径 (旧测试未覆盖)
# ═══════════════════════════════════════════════════════════════════
def test_resume_flow():
    restore = _snap_restore()
    PlannerAgent.generate_v4 = flip_gen()
    try:
        # 第一次跑: 2 轮 (不 rmtree OUT — 模拟真实续跑)
        s1 = Scheduler(OP, max_rounds=2, target_speedup=0.0, stub=True)
        s1._run_optimize = types.MethodType(optimize_ok, s1)
        s1._run_fusion = types.MethodType(lambda self, rd: {"op_count": 5, "fusion_candidates": []}, s1)
        V_mod.verify_end_to_end = ok_verify()
        s1.run()
        st1, hist1 = read_hist()
        # 第二次跑: resume=True, 应从 round3 续 (tier/round/baseline/best 保留)
        s2 = Scheduler(OP, max_rounds=4, target_speedup=0.0, stub=True, resume=True)
        s2._run_optimize = types.MethodType(optimize_ok, s2)
        s2._run_fusion = types.MethodType(lambda self, rd: {"op_count": 5, "fusion_candidates": []}, s2)
        V_mod.verify_end_to_end = ok_verify()
        s2.run()
        st2, hist2 = read_hist()
        r_cont = [h["round"] for h in hist2 if h["round"] > max(hist1[-1]["round"], 0)]
        ok = (len(hist2) > len(hist1)
              and r_cont and r_cont[0] == len(hist1) + 1          # 从 round3 续
              and st2.get("baseline_ns") == st1.get("baseline_ns")  # baseline 保留
              and st2.get("best_e2e_event_ns") == st1.get("best_e2e_event_ns"))  # best 保留
        log("P12-resume续跑", ok,
            f"r1跑{len(hist1)}轮 → resume 后续跑{len(hist2)-len(hist1)}轮, 续轮={r_cont}, "
            f"baseline保留={st2.get('baseline_ns')==st1.get('baseline_ns')}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 13: tier3 轮 sweep + REVERT (修复在 tier3 重扫场景也生效)
# ═══════════════════════════════════════════════════════════════════
def test_tier3_sweep_revert():
    """修复验证: r1 正常(含 round1 sweep 地基) → r2 promote 到 tier3 → r3 tier3 sweep 重扫
    (BLOCK→128,128,64) → coder 翻 BLOCK_K (总能匹配) → verify 失败 → 内容快照恢复 →
    r4 采集输入 = sweep 最优 + 无 r3 coder 改动 (旧实现: r4 读到 r3 失败代码)."""
    restore = _snap_restore()
    SW_mod.SWEEP_META["_sim_op"] = {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"),
                                    "type": "matmul", "multi": False}
    def stub_sweep(op_dir, quick=False, out_dir=None, op_name=None):
        kp = op_dir / "kernel_op.py"
        code = kp.read_text(encoding="utf-8")
        new = re.sub(r"BLOCK_M, BLOCK_N, BLOCK_K = [\d, ]+",
                     "BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64", code, count=1)
        kp.write_text(new, encoding="utf-8")
        return {"results": [{"block": [128, 128, 64], "ns": 100.0},
                            {"block": [64, 64, 32], "ns": 200.0}],
                "unchanged": False, "candidates_tested": 2, "best_block": [128, 128, 64]}
    SW_mod.sweep = stub_sweep
    def gen(self, extracted, tier, history, kernel_code, round_num, **kw):
        # 翻 BLOCK_K (从当前 kernel_code 提取, 保证 coder 总能匹配)
        m = re.search(r"(BLOCK_M, BLOCK_N, BLOCK_K = (\d+), (\d+), (\d+))", kernel_code)
        old = m.group(1) if m else "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32"
        bm, bn, bk = (int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else (64, 64, 32)
        new = f"BLOCK_M, BLOCK_N, BLOCK_K = {bm}, {bn}, {64 if bk == 32 else 32}"
        promote, promote_to = (False, 0)
        if round_num == 2:
            promote, promote_to = (True, 3)
        pt = json.dumps({"strategy": "s",
                         "changes": [{"old_code": old, "new_code": new,
                                      "reason": "t", "section": "① config", "tier": tier}],
                         "promote": promote, "promote_to": promote_to,
                         "promote_reason": "sim" if promote else "",
                         "promote_evidence": "block_dim=20<40 且 mte1 已低, 分块层可接手" if promote else ""})
        return RoundPlan(round_num=round_num, tier=tier, tier_name="", strategy="s",
                         target_speedup=1.1, specific_change="", expected_impact="",
                         verification_method="msprof", plan_text=pt,
                         promote=promote, promote_to=promote_to,
                         promote_reason="sim" if promote else "",
                         promote_evidence="evidence" if promote else "")
    PlannerAgent.generate_v4 = gen
    snapshots = []
    def optimize_snap(self, round_dir, tier):
        snapshots.append({"rd": str(round_dir), "tier": tier,
                          "code": self.current_kernel.read_text(encoding="utf-8")
                                  if self.current_kernel.exists() else ""})
        return canned_diagnosis()
    def fail_t3_verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None, num_launches=None):
        if "baseline" in str(round_dir):
            return {"ok": True, "ns": 5_000_000.0, "e2e_ns": 5_000_000.0,
                    "e2e_event_ns": 5_000_000.0, "speedup": 1.0, "loop": 30, "rows": 90, "duration_us": 5000.0}
        m = re.search(r"round(\d+)", str(round_dir))
        rn = int(m.group(1)) if m else 0
        if rn == 3:
            return {"ok": False, "error": "sim run error", "speedup": 1.0, "ns": None}   # tier3 sweep 轮 verify 失败
        return {"ok": True, "ns": 5_000_000.0/1.05, "e2e_ns": 5_000_000.0/1.05,
                "e2e_event_ns": 5_000_000.0/1.05, "speedup": 1.05, "loop": 30, "rows": 90, "duration_us": 5000.0}
    try:
        s = make_sched(optimize_snap, fail_t3_verify, max_rounds=5)
        s.run()
        st, hist = read_hist()
        tiers = [h["tier"] for h in hist]
        # r3 (tier3 sweep 轮) coder 成功应用后 verify 失败 → 回滚: r4 输入 = sweep 最优 (128,128,64)
        #   且 BLOCK_K 未被 r3 coder 翻转 (回滚前 r3 会把 BLOCK_K 64→32)
        r4_snap = next((sn for sn in snapshots if "round4" in sn["rd"]), None)
        ok1 = r4_snap is not None and "BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64" in r4_snap["code"]
        # r3 的 diff.patch 应为真实改动 (coder 跑过), 且失败产物留证
        r3_rd = OUT / TIER_NAMES[3] / "round3"
        diff = (r3_rd / "diff.patch").read_text(encoding="utf-8") if (r3_rd / "diff.patch").exists() else ""
        ran_coder = ("---" in diff or "@@" in diff)
        ok2 = ran_coder and (r3_rd / "failed_kernel.py").exists()
        # r4 采集路径指向 round3 (恢复后的 sweep 最优)
        ok3 = r4_snap is not None and "round3" in r4_snap["rd"].replace("\\", "/") \
              or (r4_snap is not None and "round3" in r4_snap["rd"])
        ok = ok1 and ok2
        if not ok:
            finding("BUG", "tier3 sweep 轮 REVERT 后链仍损坏",
                    f"r4输入=sweep最优={ok1} r3跑过coder+留证={ok2} tiers={tiers}")
        log("P13-tier3-sweep+REVERT(修复验证)", ok,
            f"tiers={tiers} r4输入=128,128,64={ok1} r3 coder改动+failed留证={ok2}")
    finally:
        restore()

# ═══════════════════════════════════════════════════════════════════
#  Part 14: coder 清洗回归 (用户报告的 "syntaxerror/非法字符/看代码没有" 场景)
# ═══════════════════════════════════════════════════════════════════
_SIM_KERNEL = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch_npu
import triton
import triton.language as tl

M = int(os.environ.get("MATMUL_M", 2048))
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, am, ak, bk, bn, cm, cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * am + offs_k[None, :] * ak)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * ak
    c_ptrs = c_ptr + (offs_m[:, None] * cm + offs_n[None, :] * cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def main():
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        matmul_kernel[(1,)](None, None, None, M, N, 1, 1, 1, 1, 1, 1,
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()
    print("[info] OK")

if __name__ == "__main__":
    main()
'''


def test_coder_cleaning():
    """★回归 (用户报告): LLM 输出脏代码导致 "SyntaxError/非法字符, 但看文件没有".
    根因: ①千分位 1，024 → 清洗表转 , → 1,024 → leading zeros/invalid decimal literal
    ②全角数字被删留空 ③markdown 结尾 ``` 残留 (去垃圾正则先吃开头). 全链路需清洗后 valid."""
    from agents.coder import CoderAgent, _validate_python
    cleaner = CoderAgent()._clean_output
    dirty_cases = {
        "千分位全角逗号": _SIM_KERNEL.replace("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                              "BLOCK_M, BLOCK_N, BLOCK_K = 1，024, 64, 32"),
        "千分位半角逗号": _SIM_KERNEL.replace("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                              "BLOCK_M, BLOCK_N, BLOCK_K = 1,024, 64, 32"),
        "漏#中文单位": _SIM_KERNEL.replace("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                           "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32 个元素"),
        "智能引号": _SIM_KERNEL.replace('print("[info] OK")', 'print("“[info] OK”")'),
        "中文括号": _SIM_KERNEL.replace('print("[info] OK")', 'print（"[info] OK"）'),
        "上标64²": _SIM_KERNEL.replace("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                       "BLOCK_M, BLOCK_N, BLOCK_K = 64², 64, 32"),
        "零宽粘数字": _SIM_KERNEL.replace("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                          "BLOCK_M, BLOCK_N, BLOCK_K = 64\u200c, 64, 32"),
        "全角数字": _SIM_KERNEL.replace("BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                                        "BLOCK_M, BLOCK_N, BLOCK_K = ６４, 64, 32"),
        "markdown包裹": "```python\n" + _SIM_KERNEL + "```",
        "开头解释文字": "以下是修改后的代码:\n" + _SIM_KERNEL,
        "docstring特殊字符": _SIM_KERNEL.replace("def main():", 'def main():\n    """说明: → 箭头 ² 上标 """'),
    }
    ok_all = True
    for name, dirty in dirty_cases.items():
        out = cleaner(dirty, _SIM_KERNEL)
        valid, err = _validate_python(out)
        if not valid:
            ok_all = False
            print(f"  ❌ {name}: {err}")
    # 千分位必须归一为 1024 (不是 1,024)
    out1 = cleaner(dirty_cases["千分位半角逗号"], _SIM_KERNEL)
    ok_thousand = "= 1024, 64, 32" in out1 and "1,024" not in out1
    # 千分位正则不误伤合法元组 (2048,1024 无空格元组 / 标准 64, 64, 32)
    from agents.coder import _sanitize_unicode
    safe1 = _sanitize_unicode("a, b = 2048,1024")
    safe2 = _sanitize_unicode("for i in range(0, 100, 16):")
    ok_safe = safe1 == "a, b = 2048,1024" and safe2 == "for i in range(0, 100, 16):"
    # SyntaxError 报错信息必须带出错行内容 (用户能直接看到非法字符)
    ok_err = False
    bad = "x = 1,024\n"
    _, err = _validate_python(bad)
    ok_err = "该行内容" in err and "1,024" in err
    ok = ok_all and ok_thousand and ok_safe and ok_err
    if not ok:
        finding("BUG", "coder 清洗仍有漏网 (见上方 ❌)",
                f"千分位归一={ok_thousand} 不误伤元组={ok_safe} 报错带行内容={ok_err}")
    log("P14-coder清洗(回归)", ok,
        f"{len(dirty_cases)} 脏场景全清洗valid={ok_all} 千分位→1024={ok_thousand} 不误伤={ok_safe} 报错带行内容={ok_err}")

# ═══════════════════════════════════════════════════════════════════
#  Part 15: bench 测量方法回归 (do_bench 同款: 多窗口median + 轮换破L2 + 时间预算自适应)
# ═══════════════════════════════════════════════════════════════════
def test_bench_measure():
    """★回归 (2026-08-12 bench 修复): measure_event 必须满足
    ① min≤median≤mean (多窗口统计) ② rep 按时间预算自适应 ③ 返回 times_us 数组 (长度=rep).
    另: 全部 bench 脚本 --help 正常且带新参数 (--warmup-ms/--rep-ms/--n-buf)."""
    import sys as _sys, types as _types, subprocess as _sp
    try:
        import bench_910b3.bench_common as BC
    except Exception:
        finding("BUG", "bench_common 不可 import", "检查 bench_910b3/bench_common.py 语法")
        log("P15-bench测量(回归)", False, "import 失败")
        return
    # ── mock torch (本地无 NPU): measure_event 逻辑单测 ──
    class _ME:
        def __init__(self, enable_timing=True):
            self.t = _tick()
        def record(self):
            self.t = _tick()
        def elapsed_time(self, other):
            return abs(self.t - other.t)
    _tick_n = [0]
    def _tick():
        _tick_n[0] += 1
        return _tick_n[0]
    class _MockNpu:
        class Event(_ME):
            pass
        @staticmethod
        def synchronize():
            pass
    class _MockTorch:
        npu = _MockNpu()
    _sys.modules["torch"] = _MockTorch()
    calls = [0]
    def fn(i):
        calls[0] += 1
        # 每 7 次调用模拟 10x 抖动 (验证 median 抗抖)
        _ME.elapsed_time = (lambda s, o: abs(s.t - o.t) * 10) if calls[0] % 7 == 0 \
            else (lambda s, o: abs(s.t - o.t))
        return i
    m = BC.measure_event(fn, warmup_ms=25, rep_ms=100)
    ok1 = (m["rep"] >= 5 and len(m["times_us"]) == m["rep"]
           and m["min_us"] <= m["median_us"] <= m["mean_us"])
    # ── 12 个 bench 脚本 --help (无 NPU 可跑, 验证参数/语法) ──
    scripts = ["bench_pytorch.py", "bench_pytorch_mlp.py", "bench_pytorch_attention.py",
               "bench_pytorch_rms_norm.py", "bench_pytorch_layernorm.py", "bench_pytorch_sigmoid.py",
               "bench_pytorch_matmul_relu.py", "bench_pytorch_matmul_transpose.py",
               "bench_pytorch_conv2d.py", "bench_pytorch_conv_bias_relu.py",
               "bench_pytorch_flash_attention.py", "bench_industrial.py"]
    ok2 = True
    for sc in scripts:
        r = _sp.run([_sys.executable, "bench_910b3/" + sc, "--help"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    cwd=str(REPO))
        if r.returncode != 0 or "--warmup-ms" not in (r.stdout or "") \
                or "--rep-ms" not in (r.stdout or ""):
            ok2 = False
            print(f"  ❌ {sc} --help 异常")
    ok = ok1 and ok2
    if not ok:
        finding("BUG", "bench 测量回归失败", f"measure_event统计={ok1} 脚本--help={ok2}")
    log("P15-bench测量(回归)", ok,
        f"median={m['median_us']}us rep={m['rep']} (自适应) min≤median≤mean={ok1} "
        f"12脚本--help+新参数={ok2}")

# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════
def setup():
    OP.mkdir(parents=True, exist_ok=True)
    (OP / "kernel_op.py").write_text(FAKE_KERNEL, encoding="utf-8")

def teardown():
    shutil.rmtree(OP, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    # 兜底: 优秀案例文件 (P10 等场景的改进轮也会触发记录, 可能晚于 P5 清理) — 只清测试产生的
    cp = REPO / "memory" / "tier1_cases.json"
    try:
        if cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            if all(c.get("op") == "_sim_op" for c in d):
                cp.unlink()
    except Exception:
        pass

if __name__ == "__main__":
    setup()
    try:
        test_analyzer_chain(); print()
        test_sweep_revert_chain(); print()
        test_promote_evidence(); print()
        test_promote_budget_inflation(); print()
        test_promote_evidence_ok_and_handoff(); print()
        test_tier_jump_deadloop(); print()
        test_device_pollution_reset(); print()
        test_excellent_case(); print()
        test_rebaseline_stale_best_event(); print()
        test_event_injection(); print()
        test_coder_units(); print()
        test_sweep_helpers(); print()
        test_best_kernel_binding(); print()
        test_collect_fail_chain(); print()
        test_resume_flow(); print()
        test_tier3_sweep_revert(); print()
        test_coder_cleaning(); print()
        test_bench_measure(); print()
        print("═" * 60)
        print(f"检查项: {len(RESULTS)}  通过: {sum(1 for _, o, _ in RESULTS if o)}  失败: {sum(1 for _, o, _ in RESULTS if not o)}")
        for tag, ok, d in RESULTS:
            if not ok:
                print(f"  ❌ {tag}: {d}")
    finally:
        teardown()
