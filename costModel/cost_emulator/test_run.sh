


# test for for-loop
python3 simulator.py --llm "alloc(gm_1, 10KB)  alloc(ub_1, 1KB) alloc(ub_2, 1KB)  for m in range(0, 10, 1) { gm_to_ub(ub_1, gm_1 + m * 1KB) vadd(ub_2, ub_1, 3)}"   # floor:     100 GB/s,  10.24 ns

# test for loading
python3 simulator.py --llm "alloc(gm_1, 1KB)  alloc(ub_1, 1KB)  gm_to_ub(ub_1, gm_1)"   # floor:     100 GB/s,  10.24 ns
python3 simulator.py --llm "alloc(gm_1, 6.5KB) alloc(ub_1, 6.5KB) gm_to_ub(ub_1, gm_1)"  # ramp mid:  800 GB/s,   8.32 ns
python3 simulator.py --llm "alloc(gm_1, 256KB) alloc(ub_1, 256KB) gm_to_ub(ub_1, gm_1)"  # saturated: 1500 GB/s, 174.76 ns



python3 simulator.py --critical-path "alloc(gm_a1, 256KB) alloc(gm_b1, 256KB) alloc(gm_a2, 256KB) alloc(gm_b2, 256KB)
  alloc(l1_a1, 128KB) alloc(l1_b1, 128KB) alloc(l1_a2, 128KB) alloc(l1_b2, 128KB)
  alloc(l0_a1, 256KB) alloc(l0_b1, 256KB) alloc(l0_a2, 256KB) alloc(l0_b2, 256KB)
  alloc(l0_c1, 256KB) alloc(l0_c2, 256KB)
  gm_to_l1(l1_a1, gm_a1) gm_to_l1(l1_b1, gm_b1) l1_to_l0(l0_a1, l1_a1) l1_to_l0(l0_b1, l1_b1) matrixmul(l0_c1, l0_a1, l0_b1) l0_to_gm(gm_c1, l0_c1)
  gm_to_l1(l1_a2, gm_a2) gm_to_l1(l1_b2, gm_b2) l1_to_l0(l0_a2, l1_a2) l1_to_l0(l0_b2, l1_b2) matrixmul(l0_c2, l0_a2, l0_b2) l0_to_gm(gm_c2, l0_c2)
"
#python3 simulator.py --critical-path "alloc(gm_1, 256KB) alloc(gm_2, 256KB) alloc(gm_3, 256KB) alloc(l1_1, 128KB) alloc(l0_1, 128KB) alloc(l0_2, 128KB) alloc(l1_2, 128KB) alloc(l0_3, 128KB) gm_to_l1(l1_1, gm_1) gm_to_l1(l1_2, gm_2) l1_to_l0(l0_1, l1_1) l1_to_l0(l0_2, l1_2) matrixmul(l0_3, l0_1, l0_2) l0_to_gm(gm_3, l0_3)"
#python3 emulator.py "alloc(gm_1, 256KB) alloc(gm_1x, 256KB) alloc(ub_1, 128KB) alloc(ub_1x, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2) gm_to_ub(ub_1x, gm_1x) vadd(ub_2x, ub_1x, 2.0) ub_to_gm(gm_1x, ub_2x)"
#python3 emulator.py "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)" --llm




# -------------------------------- DEMO -------------------------------------- #
# inside Claude:
# find bottleneck of this program: ""alloc(gm_a1, 256KB) alloc(gm_b1, 256KB) alloc(gm_a2, 256KB) alloc(gm_b2, 256KB)
#    alloc(l1_a1, 128KB) alloc(l1_b1, 128KB) alloc(l1_a2, 128KB) alloc(l1_b2, 128KB)
#    alloc(l0_a1, 256KB) alloc(l0_b1, 256KB) alloc(l0_a2, 256KB) alloc(l0_b2, 256KB)
#    alloc(l0_c1, 256KB) alloc(l0_c2, 256KB)
#    gm_to_l1(l1_a1, gm_a1) gm_to_l1(l1_b1, gm_b1) l1_to_l0(l0_a1, l1_a1) l1_to_l0(l0_b1, l1_b1) matrixmul(l0_c1, l0_a1, l0_b1) l0_to_gm(gm_c1, l0_c1)
#    gm_to_l1(l1_a2, gm_a2) gm_to_l1(l1_b2, gm_b2) l1_to_l0(l0_a2, l1_a2) l1_to_l0(l0_b2, l1_b2) matrixmul(l0_c2, l0_a2, l0_b2) l0_to_gm(gm_c2, l0_c2)
#  ""
 

# vectorAdd
# find bottleneck of this program: "alloc(gm_1, 128KB) alloc(ub_1, 128KB) alloc(ub_2, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"
