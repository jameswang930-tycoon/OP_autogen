import json, os, glob

base = "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/matmul"

# Trajectory
t = json.load(open(f"{base}/optimization_trajectory.json"))
s = t["state"]
print(f"Rounds: {s['round']}, Best: {s['best_speedup']:.3f}x, Tier: {s['tier']}")
print()
print("History:")
for h in t["history"]:
    sp = h.get("speedup", 1.0)     # ★v4 键: speedup (非 actual_speedup)
    d = h.get("decision", "?")
    st = h.get("strategy", "?")[:60]
    r = h.get("error", "")[:80] or h.get("change", "")[:80]
    print(f"  R{h['round']:2d} T{h['tier']}: {st:60s} sp={sp:.3f} {d:8s} {r}")

print()
print("Per-round msprof:")
for d in sorted(glob.glob(f"{base}/*/round*")):
    rd = os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d)
    mr_file = os.path.join(d, "merged", "merged_report.json")
    if os.path.exists(mr_file):
        mr = json.load(open(mr_file))
        es = mr.get("execution_summary", {})
        ns = es.get("total_ns", "?")
        ops = es.get("num_kernels", "?")
        print(f"  {rd:40s}: {ns:>8s} ns, {ops} kernels")
    else:
        print(f"  {rd:40s}: no merged_report")

print()
print("Code diffs (Round 0 vs KEPT):")
for h in t["history"]:
    if h.get("decision") == "KEEP" or h.get("speedup", 1) > 1.0:
        rn = h["round"]
        tier_name = {1:"01_algorithmic_structure", 2:"02_operator_fusion", 3:"03_tiling_block_config", 4:"04_memory_access", 5:"05_compute_occupancy", 6:"06_910b3_architecture"}.get(h["tier"], f"0{h['tier']}")
        rd_dir = os.path.join(base, tier_name, f"round{rn}")
        kf = os.path.join(rd_dir, "kernel_op.py")   # ★v4 文件名 kernel_op.py (非 kernel.py)
        if os.path.exists(kf):
            with open(kf) as f:
                code = f.read()
            # Show lines that differ from round0
            lines = code.split("\n")
            r0_code = open(f"{base}/round0/kernel.py").read().split("\n")
            diffs = 0
            for i, (l1, l2) in enumerate(zip(r0_code, lines)):
                if l1 != l2:
                    diffs += 1
                    if diffs <= 10:
                        print(f"  R{rn} line {i+1}: {l1[:80]} -> {l2[:80]}")
