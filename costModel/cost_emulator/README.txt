
#0 About this software

  GM/UB/L1/L0 Pipeline Simulator -- a tool that simulates execution of data-transfer and
  compute operations across a multi-level memory hierarchy on a seven-engine hardware model.

  Engines:
    Engine 0 (GM->UB):    gm_to_ub  -- transfers Global Memory -> Unified Buffer
    Engine 1 (UB->GM):    ub_to_gm  -- transfers Unified Buffer -> Global Memory
    Engine 2 (VecUnit):   vadd      -- SIMD vector+scalar add
    Engine 3 (GM->L1):    gm_to_l1  -- transfers Global Memory -> L1 (large on-chip SRAM)
    Engine 4 (L1->L0):    l1_to_l0  -- transfers L1 -> L0 (register file / accumulator buffer)
    Engine 5 (CubeUnit):  matrixmul -- matrix multiply (MAC array)
    Engine 6 (L0->GM):    l0_to_gm  -- write-back L0 -> Global Memory

  All engines run in parallel; data hazards (RAW, WAW, WAR) force sequential execution.
  ASAP scheduling assigns operations to the earliest possible cycle given engine availability
  and dependencies.

  Programs may use for-loops (for m in range(...) { ... }) with address-offset operands
  (gm_1 + m*1KB). An emulator front-end executes the loops and emits the concrete instruction
  stream the hardware runs; the simulator then schedules that flat stream — see Iter 12.

  Output: ASCII Gantt chart, operation table with nanosecond time ranges and wait reasons, engine
  utilization bars, per-op bandwidth utilization (effective vs peak), and parallel overlap
  summary.




#1: how this software is built:

This is the prompt how this software is built:


## Iter 1:

Developer:
I want to build a abstract simulator, when I input program like: "ub_to_gm(gm_1, ub_1) gm_to_ub(ub_1, gm_2)", it is a two-line program, and the abstract simulator simulate the execution, and
  output me a pipeline graph to show the execution, it needs to track if they can run in parallel or sequential, sometimes they can run in parallel but due to data dependency
  they have to execute in sequential, our abstract simulator should be able to capture that and reflect this correctly in the final pipeline graph, my hardware have two data tranfer
  engine, so that gm to unified buffer (ub), and ub to gm can happen in parallel if no dependency, could you write me such an abstract simulator

Claude Code:
I'll build this in Python -- parser, dependency analysis (RAW/WAW/WAR hazards), ASAP scheduler respecting both engine constraints and data deps, then an ASCII Gantt chart
  renderer.


## Iter 2:

Developer:
I want to add a feature like this: I want to make the duration more accurate, I will give the bandwidth data like 128KB/s etc, you can make up some numbers, and then let's
  make our program include data allocation, so that we know the size, based on the size and bandwidth data, you can calculate the time needed for transfer, then visualization
  will reflect the more accurate duration in the Gantt chart

Claude Code:
Let me plan this before coding.

  The changes needed:
  1. Add alloc(name, size) syntax to declare buffer sizes (e.g. alloc(gm_1, 256KB))
  2. Add bandwidth constants per engine (made-up plausible numbers)
  3. Compute per-op duration as ceil(size / bandwidth) instead of the hardcoded DURATION = 1
  4. Update Op dataclass, parse(), schedule(), render(), and the banner
  5. Update README


## Iter 3:

Developer:
 I want to add a second output mode, the current mode is for humans, now I want a mode that is tailored for LLM to read, it should only output simple statistics, like the time
  ratio for each task, a summary of executing, like sequential or parallel etc, but keep it general, probably you need to record more information during the execution, like who
  is blocking who because of what reason, like data hazard, so the summary should be about sequential or parallel, if sequential, what is the root cause, etc

Claude Code:
  - hazard_details(pred, succ) -- returns (type, buffer, reason) tuples with a full English explanation per hazard, reusing the existing hazards() logic
  - render_llm(ops, deps) -- three sections: execution summary, per-op stats (duration, time ratio, wait cycles, blocking chain with buffer names and reasons), engine utilization,
   and parallelism analysis with root-cause explanation when sequential
  - --llm flag -- works both on the CLI (python3 simulator.py --llm "...") and in interactive mode (--llm alloc(...))


## Iter 4:

Developer:

I want to add a third output mode: networkx, what it does is it will encode the current pipeline graph into a networkx format, so that later if we want to retrieve information
   that is not in other output modes, we can retrive it by query this networkx format, please encode all key information in the graph, so that we can query the bottleneck which
  means which task takes the highest time ratio, who block it, by which cause, and if we can improve it by some number, what would be the speedup number etc


Claude Code:

Good, networkx is available. Let me plan what to encode before coding:

  Nodes -- one per op, attributes: idx, name, engine, engine_name, dst, src, size_kb, duration, start, end, time_ratio, wait_cycles

  Edges -- one per dependency, attributes: hazard_types (list), buffer (which buffer caused it), reason (English string)

  Graph-level attributes -- total_cycles, execution_mode, per-engine utilization, parallel pairs

  Query helpers -- a query(G, question) function that answers the specific questions you listed: bottleneck op, who blocks it, speedup estimate if a given op is reduced by N
  cycles. This gets serialized as a JSON blob in G.graph['queries'] so it's pre-computed and retrievable without re-running the simulator.

## Iter 5:

Developer:
Add a vector unit (engine 2) for SIMD operations. The vector unit has a peak compute throughput of 512 KB/cycle. The full pipeline for a vector op is GM->UB load -> SIMD compute -> UB->GM store, all three stages can be parallelized if no dependency. Use the Roofline model to reason about bottlenecks: memory roof = GM->UB bandwidth (64 KB/cycle), compute roof = peak / compute_intensity. The lower roof is the bottleneck. Add vadd(ub_dst, ub_src, scalar) as the first vector op with compute intensity = 1. Roofline analysis should appear in all three output modes.

Claude Code:
- Added engine 2 (VecUnit, 512 KB/cycle peak) and COMPUTE_INTENSITY dict
- New vadd(ub_dst, ub_src, scalar) syntax parsed via _VEC_RE; ops sorted by source position
- Op dataclass extended with scalar, compute_intensity, roofline_bound fields
- assign_sizes: vector ops use min(memory_roof=64, compute_peak/intensity) as effective throughput
- schedule: e_free extended to 3 slots
- render: VecUnit Gantt row + Roofline Analysis section (bound, achieved vs peak, headroom)
- render_llm: VecUnit utilization + ROOFLINE ANALYSIS section with optimization hint
- build_graph: Roofline node attributes + G.graph['roofline_summary']

## Iter 6:

Developer:
Add a cube unit (engine 5) for matrix multiplication, with a two-level memory hierarchy:
  L1 (large on-chip SRAM) and L0 (register file / accumulator buffer).
  New engines: GM->L1 (engine 3, 32 KB/cycle), L1->L0 (engine 4, 128 KB/cycle),
  CubeUnit/matrixmul (engine 5, 1024 KB/cycle peak, compute_intensity=4),
  L0->GM write-back (engine 6, 32 KB/cycle).
  matrixmul takes three operands: matrixmul(l0_dst, l0_src_a, l0_src_b).
  Hazard detection extended to cover the second source operand (src2).

Claude Code:
- Added engines 3-6 to ENGINE_FOR, ENG_NAME, BANDWIDTH_KB_PER_CYCLE
- Added MEMORY_ROOF_FOR dict so each compute op uses its own memory feed rate for Roofline
  (VecUnit uses GM->UB=64 KB/cycle; CubeUnit uses L1->L0=128 KB/cycle)
- Op dataclass extended with src2 field for matrixmul's second source
- New _MAT_RE regex; parse() handles three-operand matrixmul alongside existing _VEC_RE
- hazards() and hazard_details() extended to check src2 for RAW and WAR
- schedule: e_free now [0]*len(ENG_NAME) (7 slots)
- _op_sig() helper centralises instruction string formatting across all render modes
- render, render_llm, build_graph: engine loops use range(len(ENG_NAME)); Roofline uses MEMORY_ROOF_FOR

## Iter 7:

Developer:
Remove all Roofline analysis to prepare for a redesign.

Claude Code:
- Removed COMPUTE_INTENSITY and MEMORY_ROOF_FOR dicts
- Removed compute_intensity and roofline_bound fields from Op dataclass
- assign_sizes: all ops (including vadd and matrixmul) now use raw engine bandwidth for duration
- _op_sig: identifies vadd by op.name instead of compute_intensity
- render: removed Roofline Analysis section
- render_llm: removed ROOFLINE ANALYSIS section
- build_graph: removed roofline node attributes and G.graph['roofline_summary']
- Banner and bandwidth legend updated to remove Roofline references


## Iter 8:

Developer:
The bandwidth for transferring data is not a fixed function, instead it is a function of
  how much data it transferred. For each transfer engine there is a size; below that size it
  takes a fixed amount of time, then size/fixed_time is the effective bandwidth; above that
  size, it runs at peak bandwidth. Use this bandwidth to compute the time of each transfer,
  and also give me the bandwidth utilization for each transfer (effective ÷ peak, a
  percentage). Update wherever you see fit.

Claude Code:
- Added TRANSFER_FIXED_LATENCY (cycles) per transfer engine (engines 0,1,3,4,6); the
  "knee" size = fixed_latency × peak_bandwidth
- transfer_profile(engine, size) returns (duration, effective_bw, utilization, regime):
    size ≤ knee → latency-bound: duration = fixed_latency, effective_bw = size/fixed_latency
    size >  knee → bandwidth-bound: duration = ceil(size/peak), effective_bw = peak
    compute engines (VecUnit, CubeUnit) have no startup latency → always peak
- Op dataclass extended with effective_bw, bw_utilization, regime fields
- assign_sizes now calls transfer_profile instead of the flat ceil(size/bandwidth)
- render: legend shows fixed latency + knee per engine; op table gained a "BW util"
  column; new "Bandwidth utilization" section with a bar per op
- render_llm: per-op "bandwidth:" line + new "=== BANDWIDTH UTILIZATION ===" section
- build_graph: nodes gained peak_bw/effective_bw/bw_utilization/bw_regime; engine_bandwidth
  graph attr now records fixed_latency + knee_kb; new G.graph['bandwidth_summary'] with
  per-op data and the lowest-utilization op + hint
- Banner and README updated


## Iter 9:

Developer:
For the first two output modes (default and --llm), also output the critical path of
  the DAG resulted from scheduling, where the cost is the execution time. Let me select
  which algorithm to use to compute the critical path.

Claude Code:
- The schedule DAG is more than data hazards: the ASAP scheduler also serializes ops on
  the same engine. The critical path is computed over the FULL schedule DAG = data-hazard
  edges (deps) + same-engine consecutive-op edges, so its length equals total_cycles.
- critical_path_preds(ops, deps): builds that predecessor map (reuses deps, adds a
  same-engine "last op on this engine" edge); all edges run low-idx → high-idx so op-index
  order is a valid topological order.
- critical_path_topo(ops, deps): longest weighted path via topological DP (single forward
  pass, op weight = duration), returns (path_of_op_indices, length).
- CRITICAL_PATH_ALGOS registry + DEFAULT_CP_ALGO='topo' so more algorithms can be added and
  selected by name; compute_critical_path() dispatches through it.
- _cp_edge_reason(pred, succ): reuses hazards()/hazard_details() to label each critical-path
  link as a data hazard (RAW/WAW/WAR on a buffer) or "engine serialization".
- render() and render_llm() take cp_algo; when set, print a Critical-path / === CRITICAL
  PATH === section (chain, per-link reason, per-op cycles, fraction of makespan). --nx
  output is unchanged.
- CLI: --critical-path[=algo] parsed by parse_cp_flag() (bare flag → 'topo'; unknown algo
  name exits with the valid list); threaded through simulate() and the interactive loop.
- Banner and README updated


## Iter 10:

Developer:
Add a feature for verifying correctness of a program: each on-chip memory (UB, L1, L0)
  has a fixed physical capacity (make up reasonable numbers). Simulate the execution and,
  at every key time step (op start/end), compute the live memory usage of each region;
  make sure we never use more than is physically available. On a violation, trace back to
  the root-cause code with the line number and reason.

Claude Code:
- Added MEMORY_CAPACITY_KB (UB 512 KB, L1 2 MB, L0 1 MB; GM unbounded) and REGION_FOR_PREFIX
  so a buffer's region is derived from its name prefix (gm_/ub_/l1_/l0_); region_of() helper.
- Op dataclass gained a `line` field; parse() computes each op's 1-based source line from the
  char position it already tracks in pos_map (before the program-order re-sort).
- compute_liveness(ops, sizes): every buffer's live interval [first-touch start .. last-use
  end) plus its producer/consumer op indices. A buffer is one static allocation spanning its
  whole use (conservative — the model has no explicit free()).
- verify_memory(ops, sizes): walks every event cycle (op starts), sums live buffers per region,
  records peak usage, and flags any cycle where a bounded region's footprint exceeds capacity.
  Each violation captures the live buffers (largest first) and the trigger op(s) that committed
  memory at that cycle; violations are de-duped per (region, cycle).
- render_verify(): human report — per-region capacity vs peak usage, PASS/FAIL, and for each
  violation the offending region/cycle, live buffers with their producers, the root-cause op
  with its source line + reason, and concrete fix suggestions.
- New --verify output mode threaded through simulate(), the CLI flag parsing, and the
  interactive loop (mutually exclusive with the other render modes). Banner updated.


## Iter 11:

Developer:
The bandwidth for transferring data should follow a realistic latency model based
  on real measured numbers, not the fixed-latency "knee" model. For Global Memory →
  UB: at 1KB transfer granularity the bandwidth is 100 GB/s; at and above 12KB it
  saturates and stabilizes at 1500 GB/s; between 1KB and 12KB it increases linearly.
  Use these numbers and the linear relation to simulate data transfer.

Claude Code:
- Replaced the KB/cycle + TRANSFER_FIXED_LATENCY "knee" model with a piecewise-LINEAR
  bandwidth-vs-size curve per engine (BANDWIDTH_CURVE_GB_S), a list of
  (size_kb, bandwidth_GB_per_s) breakpoints. Bandwidth ramps with size:
    size ≤ first breakpoint → floor (clamped low); between → linear ramp;
    size ≥ last breakpoint  → saturated (peak).
  GM→UB uses the MEASURED numbers [(1KB,100), (12KB,1500)]; the other transfer
  engines use clearly-marked PLACEHOLDER curves (TODO: measure) so the whole
  schedule stays in one coherent time unit. Compute engines are size-independent.
- Durations are now REAL NANOSECONDS (float), not abstract cycles:
    duration_ns = size_kb × 1024 / bandwidth(size)   (GB = 1e9 B, KB = 1024 B).
  Validated: 1KB → 10.24 ns @ 100 GB/s; 12KB → 8.19 ns @ 1500 GB/s.
- bandwidth_at_size() interpolates the curve; bandwidth_profile() returns
  (duration_ns, effective_bw_GB_s, utilization, regime) where regime ∈
  floor | ramp | saturated | flat. Removed transfer_knee_kb / transfer_profile.
- Op.duration/start/end and the scheduler/critical-path are now floats (ns).
- Gantt chart auto-scales to the makespan (chars-per-ns) since ns durations no
  longer fit a fixed chars-per-cycle grid; every op still occupies ≥1 column.
- All output modes re-labelled cycles → ns and KB/cycle → GB/s. The --nx graph keys
  were renamed for honesty (total_ns, duration_ns, start_ns, end_ns, wait_ns,
  delay_ns, busy_ns; peak_bw_gb_s, effective_bw_gb_s; engine_bandwidth now carries
  the curve). Banner, README, and the bottleneck-analysis skill updated.


## Iter 12:

Developer:
I want to add a for-loop. The emulator should execute the loop and emit the real
  instructions each iteration, then send that instruction stream to the simulator.
  A loop body may hold several statements. Address operands like gm_1 + m*1KB should
  index into a buffer.

Claude Code:
- Split the front-end (control flow) from the back-end (hardware). A new EMULATOR
  walks `for VAR in range(START, STOP[, STEP]) { BODY }`, evaluates address
  expressions, and emits the concrete Op stream; the simulator schedules it unchanged.
  Generating the full list up front is fine — ASAP scheduling and hazard analysis are
  global passes that need every instruction anyway, so the result equals the unrolled
  program exactly (not an estimate).
- Tokenizer + recursive-descent parser: statements are `for` (nestable, multi-statement
  bodies) or `call` (alloc / any op). Expressions are sums of products over NUM (size
  literals → KB), loop vars, and buffer names. range(n)/range(a,b)/range(a,b,c) all work.
- Op dataclass gained per-operand offset + sliced flags (dst_off/src_off/src2_off,
  *_sliced). dst/src/src2 still hold the BASE buffer name, so region_of/--verify/render
  are unaffected; offsets are byte tile positions into the base.
- emulate(prog) is the new entry point. A program with no `for` and no `+` offset takes
  the original regex parse() path untouched — verified BYTE-IDENTICAL across all five
  output modes for the existing vadd/matmul/basic programs (zero diff).
- HAZARDS are now offset-aware via _aliases(): same base AND overlapping [off,off+tile)
  ranges. A whole-buffer (non-sliced) operand aliases everything; two tiles alias only on
  overlap. So disjoint tiles parallelize; reusing one destination still serializes (the
  test loop: 100×1KB GM→UB on one ub_1 = WAW chain ≈ 1024 ns, sequential).
- assign_sizes: a sliced operand transfers a tile, so its size comes from the non-sliced
  operand (the destination); for flat programs this collapses to the old src-first rule.
- --verify still charges each buffer's WHOLE allocation once (a buffer is one physical
  allocation regardless of how many tiles a loop touches); loop ops carry their source
  line so root-cause reporting still names a real line number.
- Soft warning when expansion emits > 256 ops (output stays correct, just verbose).
- Banner, module docstring, and README updated.


#2: how to run this software

basic (no alloc -- uses 64 KB default per buffer):
python3 simulator.py "gm_to_ub(gm_1, ub_1) ub_to_gm(ub_1, gm_2)"

with alloc declarations for accurate transfer durations:
python3 simulator.py "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(gm_1, ub_1) ub_to_gm(ub_1, gm_2)"

an output mode that compact the result, so that LLM can read it more naturally
python3 simulator.py --llm "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(gm_1, ub_1) ub_to_gm(ub_1, gm_2)"

an output mode that output the graph into a json, which later can be queried by networkx code
python3 simulator.py --nx "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(gm_1, ub_1) ub_to_gm(ub_1, gm_2)"

show the scheduling critical path (works with default and --llm modes; algo defaults to 'topo'):
python3 simulator.py --critical-path "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"
python3 simulator.py --llm --critical-path=topo "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"

verify memory-capacity correctness (UB/L1/L0 live footprint vs physical capacity):
PASS example (fits):
python3 simulator.py --verify "alloc(gm_1, 256KB) alloc(ub_1, 128KB) alloc(ub_2, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"
FAIL example (three 512KB L0 buffers live at once = 1.5 MB > 1 MB L0 capacity; reports line + root cause):
python3 simulator.py --verify "alloc(gm_a1, 512KB) alloc(gm_b1, 512KB) alloc(l1_a1, 512KB) alloc(l1_b1, 512KB) alloc(l0_a1, 512KB) alloc(l0_b1, 512KB) alloc(l0_c1, 512KB) gm_to_l1(l1_a1, gm_a1) gm_to_l1(l1_b1, gm_b1) l1_to_l0(l0_a1, l1_a1) l1_to_l0(l0_b1, l1_b1) matrixmul(l0_c1, l0_a1, l0_b1) l0_to_gm(gm_c1, l0_c1)"

full pipeline with vector unit (all three output modes):
python3 simulator.py "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"
python3 simulator.py --llm "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"
python3 simulator.py --nx  "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"

full matmul pipeline with CubeUnit (all three output modes):
python3 simulator.py "alloc(gm_1, 256KB) alloc(gm_2, 256KB) gm_to_l1(l1_1, gm_1) gm_to_l1(l1_2, gm_2) l1_to_l0(l0_1, l1_1) l1_to_l0(l0_2, l1_2) matrixmul(l0_3, l0_1, l0_2) l0_to_gm(gm_3, l0_3)"
python3 simulator.py --llm "alloc(gm_1, 256KB) alloc(gm_2, 256KB) gm_to_l1(l1_1, gm_1) gm_to_l1(l1_2, gm_2) l1_to_l0(l0_1, l1_1) l1_to_l0(l0_2, l1_2) matrixmul(l0_3, l0_1, l0_2) l0_to_gm(gm_3, l0_3)"
python3 simulator.py --nx  "alloc(gm_1, 256KB) alloc(gm_2, 256KB) gm_to_l1(l1_1, gm_1) gm_to_l1(l1_2, gm_2) l1_to_l0(l0_1, l1_1) l1_to_l0(l0_2, l1_2) matrixmul(l0_3, l0_1, l0_2) l0_to_gm(gm_3, l0_3)"

for-loop (emulator expands it into the real instruction stream, then simulates):
streaming load of 100 1KB tiles into one UB buffer (WAW-serialized, ~1024 ns):
python3 simulator.py --llm "for m in range(0, 100, 1) { alloc(gm_1, 100KB) alloc(ub_1, 1KB) gm_to_ub(ub_1, gm_1 + m * 1KB) }"
nested loop, multi-statement body (tiled, two index vars):
python3 simulator.py "for i in range(0, 2, 1) { for j in range(0, 2, 1) { alloc(gm_1, 64KB) alloc(ub_1, 16KB) gm_to_ub(ub_1, gm_1 + i*32KB + j*16KB) } }"

#3: bandwidth configuration (in simulator.py)

Bandwidth follows a REALISTIC SIZE-DEPENDENT LATENCY MODEL. Each engine has a
piecewise-LINEAR bandwidth-vs-size curve in BANDWIDTH_CURVE_GB_S, a list of
(size_kb, bandwidth_GB_per_s) breakpoints. Achieved bandwidth ramps up with
transfer size and saturates at peak:

  size <= first breakpoint  -> floor:      clamped to the lowest bandwidth
  between two breakpoints    -> ramp:       linear interpolation of bandwidth
  size >= last breakpoint    -> saturated:  clamped to peak (the last breakpoint)

Duration is reported in REAL NANOSECONDS:

  duration_ns = size_kb * 1024 / bandwidth(size)      (GB = 1e9 bytes, KB = 1024 bytes)

GM->UB uses MEASURED real-world numbers:
  1 KB    -> 100 GB/s            (floor)
  1-12 KB -> linear 100..1500 GB/s (ramp)
  >=12 KB -> 1500 GB/s           (saturated)
  e.g. 1 KB = 10.24 ns, 12 KB = 8.19 ns, 256 KB = 174.76 ns.

VecUnit also uses MEASURED numbers. The measurement was taken in TFLOPS (compute
throughput): 1 KB -> 1 TFLOPS, ramping linearly to 8 TFLOPS at 24 KB, saturated
above. For an fp16 vadd (1 FLOP per element, 2 bytes per element) the arithmetic
intensity is a CONSTANT 2 bytes/FLOP, so TFLOPS converts to an effective bandwidth
by a constant factor: GB/s = TFLOPS * 2000. Because that factor is constant, a curve
linear in TFLOPS-vs-size is identical to one linear in GB/s-vs-size, so VecUnit is
stored and reported as an ordinary size-dependent GB/s curve and needs no special
compute path. The measured 1->8 TFLOPS over 1->24 KB becomes 2000->16000 GB/s:
  1 KB  -> 2000 GB/s   (floor,     1 TFLOPS) = 0.512 ns
  12 KB -> 8696 GB/s   (ramp,  4.348 TFLOPS) = 1.413 ns
  24 KB -> 16000 GB/s  (saturated, 8 TFLOPS) = 1.536 ns
  >=24 KB stays at 16000 GB/s; e.g. 128 KB = 8.192 ns.

The other engines use PLACEHOLDER curves (TODO: replace with measured numbers);
their shapes are scaled from the engines' former relative peaks so the schedule
stays in one coherent time unit. Only GM->UB and VecUnit are calibrated:
  GM->UB   : [(1KB, 100),  (12KB, 1500)]    MEASURED
  UB->GM   : [(1KB,  50),  (12KB,  750)]    placeholder
  VecUnit  : [(1KB, 2000), (24KB, 16000)]   MEASURED (= 1..8 TFLOPS, fp16, x2000 GB/s)
  GM->L1   : [(1KB,  50),  (12KB,  750)]    placeholder
  L1->L0   : [(1KB, 100),  (12KB, 2000)]    placeholder
  CubeUnit : [(1KB, 3000)]                  placeholder, flat (compute, size-independent)
  L0->GM   : [(1KB,  50),  (12KB,  750)]    placeholder

An engine is flat (size-independent, regime "flat") iff its curve is a single
breakpoint. Only CubeUnit (matrixmul) is flat now -- it has no measured size-dependent
numbers yet. Every other engine, including VecUnit, ramps with transfer size.

bandwidth_utilization = effective_bw / peak (a percentage), reported per op in all
three output modes. Supported alloc size units: B, KB, MB, GB. Buffers without an
alloc declaration default to 64 KB.

#4: memory-capacity configuration (in simulator.py)

Each on-chip memory region has a fixed physical capacity, set in MEMORY_CAPACITY_KB.
The --verify mode checks that the buffers simultaneously LIVE in a region never exceed
its capacity at any cycle. Defaults (made-up but plausible):

  UB (Unified Buffer):   512 KB   -- vector pipeline working set
  L1 (L1 SRAM):          2 MB     -- matrix pipeline staging
  L0 (register file):    1 MB     -- feeds the MAC array
  GM (Global Memory):    unbounded (reported, never enforced)

A buffer's region is taken from its name prefix (gm_/ub_/l1_/l0_), via REGION_FOR_PREFIX.
A buffer is treated as a single static allocation that is live from the first cycle any op
touches it (read or write) until the last op touching it finishes -- the conservative
assumption for a model with no explicit free(). Overlapping live ranges in the same region
accumulate.

--verify reports, per region, the capacity and the peak live usage (with the cycle it
occurs). If a bounded region overflows at any cycle it prints FAIL and, for each violation:
the region/cycle, the live buffers (largest first) with their producer ops, the root-cause
op(s) that committed memory at that cycle with their SOURCE LINE NUMBER and reason, and
concrete fix suggestions. Otherwise it prints PASS and the tightest region's headroom.

