#!/usr/bin/env python3
"""验证 P1-P3 新字段链路: 官方格式假 CSV (含全部新列) → board → integrate → 07 字段 → fusion 耗时标注."""
import json, os, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def write_csv(p, headers, rows):
    import csv
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([str(x) for x in r])

base = Path(tempfile.gettempdir()) / "opencode" / "field_v2_check"
shutil.rmtree(base, ignore_errors=True)
opprof = base / "OPPROF_1"
opprof.mkdir(parents=True, exist_ok=True)

# 官方格式 (850alpha002, A2 训练系列): 含全部新列
write_csv(opprof / "OpBasicInfo.csv",
          ["Op Name", "Op Type", "Task Duration(us)", "Block Dim", "Mix Block Dim",
           "Current Freq", "Rated Freq"],
          [["matmul_kernel", "TritonKernel", "800.0", "64", "N/A", "1800", "1910"]])
write_csv(opprof / "PipeUtilization.csv",
          ["aic_cube_time(us)", "aiv_vec_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)",
           "aic_mte3_time(us)", "aic_scalar_time(us)", "aic_fixpipe_time(us)",
           "aic_icache_miss_rate", "aiv_icache_miss_rate",
           "aiv_mte2_active_bw(GB/s)", "aic_mte3_active_bw(GB/s)", "aiv_mte3_active_bw(GB/s)",
           "aic_fixpipe_active_bw(GB/s)"],
          [[400.0, 50.0, 300.0, 500.0, 100.0, 20.0, 0.0,
            0.01, 0.02, 350.0, 300.0, 280.0, 200.0]])
write_csv(opprof / "ArithmeticUtilization.csv",
          ["aic_cube_fops", "aic_cube_ratio", "aic_cube_fp16_ratio", "aic_cube_int8_ratio",
           "aic_cube_total_instr_number", "aic_cube_fp_instr_number", "aic_cube_int_instr_number",
           "aiv_vec_fops", "aiv_vec_ratio", "aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio",
           "aiv_vec_int32_ratio", "aiv_vec_int16_ratio", "aiv_vec_misc_ratio",
           "aic_total_cycles", "aiv_total_cycles"],
          [[2*2048*2048*2048, 0.42, 0.0, 0.0, 100, 90, 10,
            1e9, 0.05, 0.04, 0.01, 0.0, 0.0, 0.0, 1000000, 10000]])
write_csv(opprof / "Memory.csv",
          ["aic_main_mem_read_bw", "aic_main_mem_write_bw", "aic_l1_read_bw", "aic_l1_write_bw",
           "aiv_gm_to_ub_bw", "aiv_ub_to_gm_bw",
           "read_main_memory_datas(KB)", "write_main_memory_datas(KB)",
           "GM_to_L1_datas(KB)", "L1_to_GM_datas(KB)(estimate)",
           "L0C_to_L1_datas(KB)", "L0C_to_GM_datas(KB)",
           "GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)",
           "GM_to_L1_bw_usage_rate(%)", "L1_to_GM_bw_usage_rate(%)(estimate)",
           "L0C_to_L1_bw_usage_rate(%)", "L0C_to_GM_bw_usage_rate(%)",
           "GM_to_UB_bw_usage_rate(%)", "UB_to_GM_bw_usage_rate(%)",
           "aic_mte1_instructions", "aic_mte2_instructions", "aic_mte3_instructions"],
          [[1200.0, 600.0, 2000.0, 1000.0, 1500.0, 800.0,
            33792.0, 16384.0, 33792.0, 16384.0, 16384.0, 16384.0, 16384.0, 16384.0,
            82.0, 40.0, 50.0, 30.0, 75.0, 60.0, 10, 20, 10]])
write_csv(opprof / "MemoryL0.csv",
          ["aic_l0a_read_bw", "aic_l0a_write_bw", "aic_l0b_read_bw", "aic_l0b_write_bw",
           "aic_l0c_read_bw_cube", "aic_l0c_write_bw_cube"],
          [[4000.0, 2000.0, 4000.0, 2000.0, 6000.0, 3000.0]])
write_csv(opprof / "MemoryUB.csv",
          ["aiv_ub_read_bw_vector", "aiv_ub_write_bw_vector", "aiv_ub_read_bw_scalar", "aiv_ub_write_bw_scalar"],
          [[3000.0, 3000.0, 1000.0, 1000.0]])
write_csv(opprof / "L2Cache.csv", ["aic_total_hit_rate(%)"], [["85.5"]])
write_csv(opprof / "ResourceConflictRatio.csv",
          ["aic_cube_wait_ratio", "aiv_vec_wait_ratio", "aiv_vec_total_cflt_ratio",
           "aiv_vec_bank_cflt_ratio", "aiv_vec_bankgroup_cflt_ratio", "aiv_vec_resc_cflt_ratio",
           "aiv_vec_mte_cflt_ratio",
           "aic_mte1_wait_ratio", "aic_mte2_wait_ratio", "aic_mte3_wait_ratio",
           "aiv_mte2_wait_ratio", "aiv_mte3_wait_ratio"],
          [[0.05, 0.10, 0.05, 0.03, 0.01, 0.0, 0.02,
            0.06, 0.15, 0.08, 0.12, 0.07]])

from analyzers.pipeline_parse_board import parse as parse_board
bd = parse_board(opprof)
norm = bd["normalized"]

def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"{name}: {detail}"

# P2: traffic_kb (官方实际搬运量)
tr = norm["traffic_kb"]
check("traffic_kb.main_mem_read_kb", tr.get("main_mem_read_kb") == 33792.0, tr)
check("traffic_kb.l1_to_gm_kb (estimate后缀)", tr.get("l1_to_gm_kb") == 16384.0, tr)
# P2: bw_usage_rate 归一化 0~1
bu = norm["bw_usage_rate"]
check("bw_usage_rate.gm_to_l1=0.82", abs(bu.get("gm_to_l1", 0) - 0.82) < 0.001, bu)
check("bw_usage_rate.ub_to_gm=0.60", abs(bu.get("ub_to_gm", 0) - 0.60) < 0.001, bu)
# P2: active_bw (GB/s 原值, 不除 1000)
ab = norm["active_bw_gb_s"]
check("active_bw.mte2_aiv=350", abs(ab.get("mte2_aiv_gb_s", 0) - 350.0) < 0.001, ab)
# P2: icache
ic = norm["icache_miss_rate"]
check("icache.cube=0.01", abs(ic.get("cube", 0) - 0.01) < 0.001, ic)
check("icache.vec=0.02", abs(ic.get("vec", 0) - 0.02) < 0.001, ic)
# P4: vec 精度细分
comp = norm["compute"]
check("compute.vec_fp16_ratio=0.01", abs(comp.get("vec_fp16_ratio", 0) - 0.01) < 0.001, comp)
check("compute.cube_fp_instr_number=90", comp.get("cube_fp_instr_number") == 90.0, comp)
# P1: conflict 规范短名
cf = norm["conflict"]
check("conflict.vec_wait_ratio=0.10", abs(cf.get("vec_wait_ratio", 0) - 0.10) < 0.001, cf)
check("conflict.cube_wait_ratio=0.05", abs(cf.get("cube_wait_ratio", 0) - 0.05) < 0.001, cf)
check("conflict.mte2_wait_ratio=0.15 (aic优先)", abs(cf.get("mte2_wait_ratio", 0) - 0.15) < 0.001, cf)
# OpBasicInfo 新字段
es = bd["execution_summary"]
check("rated_freq_mhz=1910", es.get("rated_freq_mhz") == 1910.0, es)
check("mix_block_dim=None(N/A)", es.get("mix_block_dim") is None, es)
check("num_cores=64 (未被 Mix 污染)", es.get("num_cores") == 64.0, es)

# P2: integrate → deep 新字段 + roofline 冗余倍数
from analyzers.pipeline_parse_task import parse as parse_task
prof = base / "mindstudio_profiler_output"
prof.mkdir(parents=True, exist_ok=True)
rows = []
for _ in range(10):
    rows.append(["matmul_kernel", "TritonKernel", "Cube", "800.0", "64",
                 "(2048,2048),(2048,2048)", "float32,float32", "(2048,2048)", "float32"])
write_csv(prof / "op_summary_0.csv",
          ["Op Name", "Op Type", "Task Type", "Task Duration(us)", "Block Dim",
           "Input Shape(s)", "Input Data Type(s)", "Output Shape(s)", "Output Data Type(s)"], rows)
write_csv(prof / "op_statistic_0.csv", ["OP Type", "Count", "Total Time(us)", "Ratio"],
          [["Cube", 10, 8000.0, 1.0]])
write_csv(prof / "api_statistic_0.csv", ["API Name", "Time(us)", "Count"], [["aclrtLaunchKernel", 10.0, 10]])
write_csv(prof / "l2_cache_0.csv", ["Hit Rate"], [["85.5"]])
tk = parse_task(prof)
# est_bytes_in = (2048*2048 + 2048*2048)*4 = 33.5MB; 实际读 33792KB=33.0MB → 冗余 ≈ 0.99
from analyzers.integrate import integrate
tjson = base / "task.json"; tjson.write_text(json.dumps(tk), encoding="utf-8")
bjson = base / "board_1.json"; bjson.write_text(json.dumps(bd), encoding="utf-8")
diag_p = base / "diagnosis.json"
integrate(str(tjson), str(diag_p), [str(bjson)])
dg = json.loads(diag_p.read_text(encoding="utf-8"))
deep = dg["kernels"][0]["deep"]
check("deep.traffic_kb", deep.get("traffic_kb", {}).get("main_mem_read_kb") == 33792.0)
check("deep.bw_usage_rate.gm_to_l1", abs(deep.get("bw_usage_rate", {}).get("gm_to_l1", 0) - 0.82) < 0.001)
check("deep.active_bw_gb_s", deep.get("active_bw_gb_s", {}).get("mte2_aiv_gb_s") == 350.0)
check("deep.icache_miss_rate.cube", abs(deep.get("icache_miss_rate", {}).get("cube", 0) - 0.01) < 0.001)
rd = deep.get("roofline", {}).get("traffic_redundancy_read")
check("roofline.traffic_redundancy_read≈0.99", rd is not None and abs(rd - 0.99) < 0.05, f"redundancy={rd}")

# P1: 07 字段提取用精确规范键 (Tier6: vec_wait_ratio 必须命中 vec 的 0.10, 不是 cube 的 0.05)
from agents.scheduler import extract_tier_fields, _get
txt6 = extract_tier_fields(dg, 6)
check("Tier6 07字段 wait=0.100 (vec 而非 cube 0.050, P1 歧义修复)", "wait=0.100" in txt6, txt6[:500])
v = _get(dg, "kernels[].deep.conflict.vec_wait_ratio")
check("_get vec_wait_ratio=0.10 (无歧义)", abs(v - 0.10) < 0.001, f"got {v}")
v2 = _get(dg, "kernels[].deep.conflict.cube_wait_ratio")
check("_get cube_wait_ratio=0.05", abs(v2 - 0.05) < 0.001, f"got {v2}")
txt4 = extract_tier_fields(dg, 4)
check("Tier4 07字段 redun=1.030 + gm_r_kb=33792", "redun=1.030" in txt4 and "gm_r_kb=33792" in txt4, txt4[:400])

# P3: fusion view 耗时标注
mlir = base / "hivm_try.txt"
mlir.write_text("""
func.func @mm(%A: memref<256x256xf16, #hivm.address_space<gm>>,
              %B: memref<256x256xf16, #hivm.address_space<gm>>,
              %C: memref<256x256xf32, #hivm.address_space<gm>>) {
    %l1_a = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>
    %l1_b = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>
    %l0c  = memref.alloc() : memref<128x128xf32, #hivm.address_space<cc>>
    hivm.hir.load ins(%A : memref<256x256xf16, #hivm.address_space<gm>>) outs(%l1_a : memref<128x128xf16, #hivm.address_space<cbuf>>)
    hivm.hir.load ins(%B : memref<256x256xf16, #hivm.address_space<gm>>) outs(%l1_b : memref<128x128xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag [set_pipe = #hivm.pipe<mte2>, wait_pipe = #hivm.pipe<cube>, flag_id = 0 : i32]
    hivm.hir.mmadL1 ins(%l1_a, %l1_b, %c0_i1, %m, %k, %n : memref<128x128xf16, #hivm.address_space<cbuf>>, memref<128x128xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%l0c : memref<128x128xf32, #hivm.address_space<cc>>) {lhs_m = 128 : i32, rhs_n = 128 : i32, l0b_k = 128 : i32}
    hivm.hir.wait_flag [set_pipe = #hivm.pipe<cube>, wait_pipe = #hivm.pipe<mte3>, flag_id = 0 : i32]
    hivm.hir.store ins(%l0c : memref<128x128xf32, #hivm.address_space<cc>>) outs(%C : memref<256x256xf32, #hivm.address_space<gm>>)
    hivm.hir.pipe_barrier [pipe = #hivm.pipe<mte3>]
    return
}
""")
from analyzers.filter_hivm_for_fusion import main as fusion_main
view = base / "fusion_view.txt"
fusion_main(str(mlir), str(view), [str(bjson)])
vt = view.read_text(encoding="utf-8")
check("fusion view 含每op耗时占比标题", "每 op 估算耗时占比" in vt)
check("fusion view 含 mmadL1 耗时≈400us", "mmadL1" in vt and "400.0us" in vt, vt[-900:])
check("fusion view 含 pipe 实测耗时", "aic_cube_time_us=400" in vt, vt[-900:])
check("fusion view 含融合优先级指引", "融合优先级" in vt)

# 不带 --boards: 不崩, 提示无耗时
view2 = base / "fusion_view2.txt"
fusion_main(str(mlir), str(view2), [])
vt2 = view2.read_text(encoding="utf-8")
check("无 boards 时优雅降级", "无法标注耗时" in vt2)

print("\n═══ 新字段链路验证全部通过 ═══")
