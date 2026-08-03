"""Inspect total_ns calculation chain"""
import json, sys
base = "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/fused_add_mul"

for label, rd in [("Round 0", "round0"), ("Round 4", "03_tiling_block_config/round4")]:
    print(f"=== {label} ===")

    # 1. msprof report
    msprof_file = f"{base}/{rd}/msprof/pipeline_report.json"
    try:
        mr = json.load(open(msprof_file))
        es = mr.get("execution_summary", {})
        print(f"  msprof total_ns: {es.get('total_ns', 'NOT_FOUND')}")
        print(f"  msprof num_ops: {es.get('num_ops', '?')}")
        print(f"  msprof mode: {es.get('execution_mode', '?')}")
    except:
        print(f"  msprof: NOT FOUND")

    # 2. merged report
    merged_file = f"{base}/{rd}/merged/merged_report.json"
    try:
        mr = json.load(open(merged_file))
        es = mr.get("execution_summary", {})
        print(f"  merged total_ns: {es.get('total_ns', 'NOT_FOUND')}")
        print(f"  merged keys: {list(es.keys())[:10]}")
        meta = mr.get("meta", {})
        print(f"  has_msprof_timing: {meta.get('has_msprof_timing', '?')}")
        # per-op stats
        ops = mr.get("per_op_statistics", [])
        if ops:
            print(f"  per_op count: {len(ops)}")
            for o in ops[:3]:
                print(f"    op: {o.get('op_type','?')} dur={o.get('duration_ns','?')}ns pipe={o.get('pipeline_channel','?')}")
    except:
        print(f"  merged: NOT FOUND")

    # 3. msprof trace
    import glob
    csvs = glob.glob(f"{base}/{rd}/msprof/**/*instr_exe.csv", recursive=True)
    if csvs:
        with open(csvs[0]) as f:
            lines = f.readlines()
            print(f"  instr_exe.csv: {len(lines)} lines")
            # total running_time
            total_us = 0
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    try:
                        total_us += float(parts[4])
                    except: pass
            print(f"  sum(running_time): {total_us:.1f}us = {total_us*1000:.0f}ns")
            # last instruction end time
            print(f"  first line: {lines[1].strip()[:100]}")
            print(f"  last line: {lines[-1].strip()[:100]}")
    else:
        print(f"  instr_exe.csv: NOT FOUND")

    print()
