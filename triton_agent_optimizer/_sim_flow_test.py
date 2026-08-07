#!/usr/bin/env python3
"""端到端链路模拟 — 打桩真机 (msprof/HIVM/LLM), 走真实 scheduler 代码逐轮执行.
验证: 路径/M/N/K/TIER 传递、round 目录产物、trajectory 字段、keep/revert/promote、
      H2 采集失败重试/跳过/停止、F4 pytorch 基准选择、P1 baseline_verify 清理.
"""
import json, os, re, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.scheduler import Scheduler, TIER_NAMES
from agents.planner import RoundPlan, PlannerAgent
import agents.scheduler as S_mod
import agents.verifier as V_mod

PROJ = Path(__file__).resolve().parent
OP = PROJ / "input" / "_sim_op"
OUT = PROJ / "outputs" / "_sim_op"
N_FAIL_START = 8          # round>=8 模拟采集失败
BASELINE_NS = 5_000_000.0
SIM_SPEED = {1:1.0, 2:1.12, 3:0.95, 4:1.2, 6:1.25, 7:1.3}

def canned_diagnosis():
    k = {
        "kernel_name": "matmul_kernel",
        "framework": False, "launch_count": 1, "filled_by": "msprof op",
        "task": {"task_type": "Cube", "task_duration_us": 1000.0, "block_dim": 20,
                 "pipes_us": {"aic_cube_time_us": 800.0, "aic_mte2_time_us": 200.0}},
        "deep": {
            "compute": {"cube_fops": 2*2048*2048*2048, "vector_fops": 0,
                        "cube_ratio": 0.9, "cube_fp16_ratio": 0.0, "cube_int8_ratio": 0.0},
            "engine_utilization": {"mte1": 0.3, "mte2": 0.4, "cube": 0.5, "vec": 0.1,
                                   "scalar": 0.05, "fixpipe": 0.0},
            "bandwidth_gb_s": {"main_mem_read_gb_s": 500.0, "main_mem_write_gb_s": 400.0},
            "conflict": {"bank_cflt_ratio": 0.01, "wait_ratio": 0.2, "mte_cflt_ratio": 0.1},
            "l2_hit_rate": 0.5,
            "roofline": {"compute_utilization": 0.5, "memory_utilization": 0.3,
                         "arithmetic_intensity": 8.0, "bottleneck_type": "compute_bound"},
        },
    }
    return {
        "summary": {"num_kernels": 3, "num_kernels_total": 5, "total_ns": 6_000_000.0,
                    "num_cores": 20, "api_overhead_total_us": 120.0, "l2_hit_rate": 0.5,
                    "filled_kernels": 3},
        "kernels": [k,
                    {**k, "kernel_name": "bias_gelu_kernel"},
                    {**k, "kernel_name": "matmul_kernel2"}],
        "framework_kernels": [], "api_overhead": [], "multi_kernel": [],
    }

FAKE_KERNEL = '''import os, torch, triton, triton.language as tl

M  = int(os.environ.get("MATMUL_M", 2048))
N  = int(os.environ.get("MATMUL_N", 2048))
K  = int(os.environ.get("MATMUL_K", 2048))
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
BLOCK_SIZE = 1024

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, am, ak, bk, bn, cm, cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * am + offs_k[None, :] * ak)
    b_ptrs = b_ptr + (offs_k[:, None] * bk + offs_n[None, :] * bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * ak
        b_ptrs += BLOCK_K * bk
    c_ptrs = c_ptr + (offs_m[:, None] * cm + offs_n[None, :] * cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def main():
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        matmul_kernel[(1,)](None, None, None, M, N, K, 1, 1, 1, 1, 1, 1,
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

if __name__ == "__main__":
    main()
'''

# ═══════════ 桩: Popen (Part A 真实 _run_optimize 用) ═══════════
class FakeProc:
    def __init__(self, cmd, env):
        self.cmd, self.env = cmd, env
        self.stdout = iter([])
        self.killed = False
        rd = Path(cmd[3])                    # round_dir (cmd: [bash, script, input_dir, round_dir, M, N, K])
        d6 = rd / "06_diagnosis"; d6.mkdir(parents=True, exist_ok=True)
        (d6 / "diagnosis.json").write_text(json.dumps(canned_diagnosis()), encoding="utf-8")
    def wait(self, timeout=None):
        return 0
    def kill(self):
        self.killed = True
class FakePopen:
    calls = []
    def __init__(self, cmd, stdout=None, stderr=None, text=None, encoding=None, errors=None, env=None):
        FakePopen.calls.append({"cmd": cmd, "env": env})
        self._p = FakeProc(cmd, env)
    def __getattr__(self, n):
        return getattr(self._p, n)

def fake_verify(kernel_op, round_dir, baseline_ns=None, num_kernels=None):
    rd = str(round_dir)
    if "baseline" in rd:
        return {"ok": True, "ns": BASELINE_NS, "speedup": 1.0, "loop": 30, "rows": 90,
                "duration_us": BASELINE_NS/1000}
    m = re.search(r"round(\d+)", rd)
    rn = int(m.group(1)) if m else 1
    sp = SIM_SPEED.get(rn, 1.2)
    base = baseline_ns or BASELINE_NS
    return {"ok": True, "ns": round(base / sp, 1), "speedup": sp,
            "loop": 30, "rows": 90, "duration_us": round(base / sp / 1000, 1)}

def fake_generate_v4(self, extracted, tier, history, kernel_code, round_num,
                     op_dir=None, fusion_analysis=None, round_dir=None, current_kernel=None,
                     context_path=None, trajectory_path=None, handoff=None):
    m = re.search(r"(BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, (\d+))", kernel_code)
    old = m.group(1) if m else "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32"
    val = m.group(2) if m else "32"
    new = f"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, {'64' if val == '32' else '32'}"
    promote, promote_to = False, 0
    if round_num == 5 and tier == 1:
        promote, promote_to = True, 2
    plan_text = json.dumps({"strategy": "sim",
                            "changes": [{"old_code": old, "new_code": new,
                                         "reason": "sim", "section": "① config", "tier": tier}],
                            "promote": promote, "promote_to": promote_to,
                            "promote_reason": "sim promote" if promote else ""},
                           ensure_ascii=False, indent=2)
    return RoundPlan(round_num=round_num, tier=tier, tier_name="", strategy="sim",
                     target_speedup=1.1, specific_change="", expected_impact="",
                     verification_method="msprof", plan_text=plan_text,
                     promote=promote, promote_to=promote_to, promote_reason="sim promote" if promote else "")

def fake_run_optimize(self, round_dir, tier):
    self._sim_calls.append({"round_dir": str(round_dir), "tier": tier,
                            "input_dir": str(self.current_kernel.parent)})
    rn = int(re.search(r"round(\d+)", str(round_dir)).group(1))
    if rn >= N_FAIL_START:
        return None
    return canned_diagnosis()

def setup():
    OP.mkdir(parents=True, exist_ok=True)
    (OP / "kernel_op.py").write_text(FAKE_KERNEL, encoding="utf-8")
    os.environ["KEEP_FLOOR"] = "1.0"   # 隔离噪声地板, 测试核心 keep/revert 机制

def teardown():
    shutil.rmtree(OP, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    for f in ("pytorch_tflops.json", "pytorch_mlp_tflops.json"):
        p = PROJ / "bench_910b3" / f
        if p.exists(): p.unlink()
    if Path(__file__).name == "_sim_flow_test.py":
        pass

# ═══════════ Part A: 真实 _run_optimize 的 cmd/M/N/K/TIER 构造 ═══════════
def part_a():
    S_mod.subprocess.Popen = FakePopen
    s = Scheduler(OP, max_rounds=15, stub=True)
    rd = OUT / "01_algorithmic_structure" / "round1"
    d = s._run_optimize(rd, 3)          # 真实方法, Popen 打桩
    call = FakePopen.calls[-1]
    cmd = call["cmd"]
    assert cmd[0] == "bash", cmd
    assert cmd[1].endswith("analyzers/run_optimize.sh"), cmd[1]
    assert cmd[2] == str(OP), cmd[2]               # input_dir = 源 kernel 目录
    assert cmd[4:7] == ["2048", "2048", "2048"], cmd[4:7]   # ★真实 M/N/K 传参
    assert call["env"]["TIER"] == "3", call["env"]["TIER"]  # ★TIER 传参
    assert d["summary"]["num_kernels"] == 3, "diagnosis 回读"
    print("[A] ✅ 真实 _run_optimize: cmd=%s M/N/K=%s TIER=%s diagnosis回读OK"
          % (cmd[1].split("/")[-1], cmd[4:7], call["env"]["TIER"]))

# ═══════════ Part B: 全链路循环 ═══════════
def part_b():
    # F4: 临时写两个 pytorch 基准 (多 kernel 应选 MLP)
    (PROJ / "bench_910b3" / "pytorch_tflops.json").write_text(
        json.dumps({"tflops": 17.0}), encoding="utf-8")
    (PROJ / "bench_910b3" / "pytorch_mlp_tflops.json").write_text(
        json.dumps({"tflops": 42.0}), encoding="utf-8")

    PlannerAgent.generate_v4 = fake_generate_v4   # 打桩 planner (自控 promote)
    V_mod.verify_end_to_end = fake_verify          # 打桩 verify
    s = Scheduler(OP, max_rounds=15, stub=True)
    import types
    s._run_optimize = types.MethodType(fake_run_optimize, s)   # 打桩采集 (rn>=8 返回 None)
    s._run_fusion = types.MethodType(lambda self, rd: {"op_count": 5, "fusion_candidates": [],
                                                       "stub": True}, s)   # 打桩 HIVM 融合 (独立组件, 真机才跑)
    s._sim_calls = []
    s.run()

    # ── 产物检查 ──
    assert s._sim_calls, "应多次采集"
    t = json.loads((OUT / "optimization_trajectory.json").read_text(encoding="utf-8"))
    st = t["state"]; hist = t["history"]
    print(f"[B] 完成 {len(hist)} 轮; best={st['best_speedup']} current={st['current_speedup']}")

    # state 字段
    assert st["baseline_ns"] == BASELINE_NS, st
    assert st["num_kernels"] == 3
    assert st["baseline_mnk"] == [2048, 2048, 2048], "尺寸记录"
    assert st["pytorch_tflops"] == 42.0, f"F4 多kernel应选 MLP基准, got {st['pytorch_tflops']}"
    assert st["pytorch_baseline"] == "pytorch_mlp_tflops.json", st
    assert isinstance(st.get("initial_tflops"), (int, float))
    assert "current_kernel" in st
    print(f"[B] ✅ state: baseline_ns={st['baseline_ns']} mk={st['num_kernels']} "
          f"initial_tflops={st.get('initial_tflops')} pytorch_tflops={st['pytorch_tflops']}(MLP基准)")

    # history 决策序列 (r1..r7 + r8跳过)
    dec = [(h["round"], h["tier"], h["decision"], h["result"]) for h in hist]
    print(f"[B] decisions: {dec}")
    assert dec[0][2:] == ("KEEP", "OK"), dec[0]
    assert dec[1][2:] == ("KEEP", "OK"), dec[1]
    assert dec[2][2:] == ("REVERT", "OK"), dec[2]     # 0.95 < 1.12
    assert dec[3][2:] == ("KEEP", "OK"), dec[3]
    assert dec[4][0] == 5 and dec[4][1] == 1 and dec[4][2] == "KEEP"  # promote轮 tier仍1
    assert dec[5][1] == 2, "r6 应已在 tier2"          # promote 生效
    assert dec[6][2:] == ("KEEP", "OK")
    assert dec[7][2:] == ("FAIL", "FAIL") and dec[7][3] == "FAIL"  # r8 采集失败跳过
    print("[B] ✅ keep/revert/promote/采集失败跳过 决策序列正确")

    # hist 字段完整性 (ns 非 None; F1 promote 轮 ns 反推; F3 tflops)
    for h in hist[:-1]:    # 最后是失败跳过
        assert h["ns"] is not None, f"ns 应为值, {h}"
    r5 = [h for h in hist if h["round"] == 5][0]
    assert abs(r5["ns"] - BASELINE_NS / 1.2) < 1, f"F1 promote轮ns反推: {r5['ns']}"
    assert all("tflops" in h for h in hist[:7]), "F3 hist每轮tflops"
    print(f"[B] ✅ F1 promote轮 ns={r5['ns']} 非None; F3 hist每轮tflops 存在")

    # round 目录产物: 每轮 kernel_op.py + diff.patch + plan.md + 07 字段
    for rn in range(1, 8):
        tier = [h for h in hist if h["round"] == rn][0]["tier"]
        rd = OUT / TIER_NAMES[tier] / f"round{rn}"
        assert (rd / "kernel_op.py").exists(), f"{rd}/kernel_op.py 缺"
        assert (rd / "diff.patch").exists(), f"{rd}/diff.patch 缺"
        assert (rd / "plan.md").exists(), f"{rd}/plan.md 缺"
        d7 = rd / f"07_tier{tier}_fields" / f"tier{tier}_fields.txt"
        assert d7.exists(), f"{d7} 缺"
    # 晋升轮 r5 是原样拷贝 (与上一轮 kernel 相同)
    r4_k = (OUT / TIER_NAMES[1] / "round4" / "kernel_op.py").read_text(encoding="utf-8")
    r5_k = (OUT / TIER_NAMES[1] / "round5" / "kernel_op.py").read_text(encoding="utf-8")
    assert r4_k == r5_k, "晋升轮应原样拷贝当前 kernel"
    print("[B] ✅ 每轮产物 (kernel_op.py/diff.patch/plan.md/07字段) 齐备; 晋升轮原样拷贝")

    # 采集 input_dir 演进: r1=源目录, 之后=上一轮输出目录
    calls = s._sim_calls
    assert calls[0]["input_dir"] == str(OP), calls[0]
    rd0 = calls[0]["round_dir"].replace("\\", "/")
    assert "01_algorithmic_structure/round1" in rd0, rd0
    assert calls[1]["input_dir"].replace("\\", "/").endswith("round1"), calls[1]["input_dir"]
    print(f"[B] ✅ 采集链: r1 input={calls[0]['input_dir']} → r2 input={calls[1]['input_dir']} (链连续)")

    # H2: 失败重试 + 跳过 + 停止
    fail_rounds = [c["round_dir"] for c in calls if "round8" in c["round_dir"] or "round9" in c["round_dir"]]
    print(f"[B] 失败轮采集次数: r8 x{sum('round8' in c['round_dir'] for c in calls)}, "
          f"r9 x{sum('round9' in c['round_dir'] for c in calls)}")
    assert sum("round8" in c["round_dir"] for c in calls) == 2, "r8 应重试1次后跳过"
    assert sum("round9" in c["round_dir"] for c in calls) == 1, "r9 触发连续3次停止"
    print("[B] ✅ H2: 失败重试1次→跳过→连续3次停止, 前面轮次不白跑")

# ═══════════ Part C: P1 baseline_verify 清理 ═══════════
def part_c():
    import types
    s = Scheduler(OP, max_rounds=2, stub=True)
    s._run_optimize = types.MethodType(fake_run_optimize, s)
    s._sim_calls = []
    s.run()
    base = OUT / "baseline_verify"
    assert base.exists(), "baseline_verify 应存在"
    junk = base / "junk_old_run.txt"
    junk.write_text("old", encoding="utf-8")     # 模拟上一次 run 残留
    s2 = Scheduler(OP, max_rounds=2, stub=True)
    s2._run_optimize = types.MethodType(fake_run_optimize, s2)
    s2._sim_calls = []
    s2.run()
    assert not junk.exists(), "P1 第二次 run 应 rmtree 掉旧 baseline_verify"
    print("[C] ✅ P1: baseline_verify 每次重测基准前清理 (旧残留被删)")

if __name__ == "__main__":
    setup()
    try:
        part_a()
        print()
        part_b()
        print()
        part_c()
        print()
        print("═══ 全链路模拟通过 ═══")
    finally:
        teardown()
