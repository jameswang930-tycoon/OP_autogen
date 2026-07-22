---
name: bottleneck-analysis
description: given a program using some data transfer and compute APIs, try to analyze and find its bottleneck for later optimization, the main tool is a simulator, which given the information of some compute and bandwidth data of the hardware we are interested in, and simulate the running of the program, give timeline information, compute and bandwidth utilization, critical path etc. use when user asks for "optimize this program", "find optimization opporutnities" etc
---

# Bottleneck analysis guide

## Overview

The purpose is to find bottleneck of a program on my hardware, I got the hardware simulator as the main tool for analysis.

**Input:** a program written in the data-transfer and compute APIs (`alloc`, `gm_to_ub`,
`ub_to_gm`, `vadd`, `gm_to_l1`, `l1_to_l0`, `matrixmul`, `l0_to_gm`).

**Output:** the program's **bottleneck** — the operation (or resource) that dominates the
makespan — together with the evidence behind it (critical path, engine utilization,
bandwidth utilization) so it can guide a later optimization.

The simulator (`simulator.py`) models a **seven-engine accelerator** with a multi-level
memory hierarchy. All engines run in parallel; data hazards (RAW/WAW/WAR) and same-engine
contention force serialization. Use this hardware summary to reason about bottlenecks.

### Memory hierarchy

- **GM**  — Global Memory (large off-chip; source/sink of all data)
- **UB**  — Unified Buffer (on-chip working buffer for the vector pipeline)
- **L1**  — large on-chip SRAM (matrix pipeline staging)
- **L0**  — register file / accumulator buffer (feeds the matrix MAC array)

### Engines

| # | Engine    | Operation                            | Peak BW (GB/s) | Bandwidth ramp (size_kb → GB/s) |
|---|-----------|--------------------------------------|----------------|----------------------------------|
| 0 | GM→UB     | `gm_to_ub(ub_dst, gm_src)`           | 1500           | 1KB→100, 12KB→1500 (MEASURED)    |
| 1 | UB→GM     | `ub_to_gm(gm_dst, ub_src)`           | 750            | 1KB→50, 12KB→750 (placeholder)   |
| 2 | VecUnit   | `vadd(ub_dst, ub_src, scalar)`       | 16000          | 1KB→2000, 24KB→16000 (MEASURED, = 1→8 TFLOPS fp16) |
| 3 | GM→L1     | `gm_to_l1(l1_dst, gm_src)`           | 750            | 1KB→50, 12KB→750 (placeholder)   |
| 4 | L1→L0     | `l1_to_l0(l0_dst, l1_src)`           | 2000           | 1KB→100, 12KB→2000 (placeholder) |
| 5 | CubeUnit  | `matrixmul(l0_dst, l0_src_a, l0_src_b)` | 3000        | flat (compute, size-independent) |
| 6 | L0→GM     | `l0_to_gm(gm_dst, l0_src)`           | 750            | 1KB→50, 12KB→750 (placeholder)   |

Only GM→UB and VecUnit are calibrated from measured numbers; the other transfer
engines use placeholder curves (TODO: measure). CubeUnit has no measured numbers yet,
so it stays flat (size-independent). Durations are reported in nanoseconds.

### Two compute pipelines

- **Vector pipeline:**  GM→UB load → VecUnit (`vadd`) → UB→GM store
- **Matrix pipeline:**  GM→L1 load → L1→L0 load → CubeUnit (`matrixmul`) → L0→GM write-back

All three (or four) stages of a pipeline can overlap across independent data tiles when no
hazard exists.

### Size-dependent bandwidth model

Bandwidth follows a realistic **latency model**: a transfer engine's achieved
bandwidth **ramps up with transfer size** and saturates at peak. Each engine has a
piecewise-**linear** curve of `(size_kb → GB/s)` breakpoints (`BANDWIDTH_CURVE_GB_S`):

- `size ≤ first breakpoint` → **floor:**      bandwidth clamped to the lowest value
- between two breakpoints     → **ramp:**       bandwidth linearly interpolated
- `size ≥ last breakpoint`    → **saturated:**  bandwidth clamped to peak

Duration is reported in **real nanoseconds**:

```
duration_ns = size_kb × 1024 / bandwidth(size)      (GB = 1e9 bytes, KB = 1024 bytes)
```

**GM→UB** is calibrated from measured numbers: `1 KB → 100 GB/s`, ramping linearly to
`1500 GB/s` at `12 KB`, saturated above. So 1 KB costs 10.24 ns, 12 KB costs 8.19 ns,
256 KB costs 174.76 ns. The other transfer engines use **placeholder** curves
(TODO: measure) of the same shape.

**VecUnit** is also calibrated from measured numbers, taken in TFLOPS: an fp16 `vadd`
ramps `1 TFLOPS` at `1 KB` to `8 TFLOPS` at `24 KB`, saturated above. With a constant
2 bytes/FLOP that is `GB/s = TFLOPS × 2000`, so it's stored as an ordinary
size-dependent curve `1 KB → 2000 GB/s`, `24 KB → 16000 GB/s` (1 KB costs 0.512 ns,
24 KB costs 1.536 ns, 128 KB costs 8.192 ns).

**CubeUnit** (`matrixmul`) is the only **size-independent** engine — a single flat
breakpoint at peak (no measured size-dependent numbers yet).

`bandwidth_utilization = effective_bw / peak` is reported per op. A low utilization
means the op is on the **ramp / floor** (size below saturation) — a candidate
for batching into a larger transfer/tile to reach peak. This now applies to VecUnit
too: a small `vadd` sits below its 24 KB saturation point. The regime is reported
per op as `floor` / `ramp` / `saturated` (size-dependent engines, incl. VecUnit) or
`flat` (CubeUnit, the only size-independent engine). Buffers
without an `alloc(name, size)` declaration default to 64 KB.

### Scheduling & critical path

Ops are placed As-Soon-As-Possible given engine availability and dependencies. The
**critical path** is the longest weighted chain through the full schedule DAG (data-hazard
edges + same-engine serialization edges); its length equals the makespan (`total_ns`).

### Memory-capacity correctness

Each on-chip region has a fixed physical capacity, so a fast program is only useful if it
also **fits**. The `--verify` mode checks that the buffers simultaneously live in a region
never exceed its capacity at any moment (a buffer is live from first touch to last use — the
model has no explicit `free()`). Default capacities:

| Region | Capacity | Role                                  |
|--------|----------|---------------------------------------|
| UB     | 512 KB   | Unified Buffer — vector working set   |
| L1     | 2 MB     | L1 SRAM — matrix pipeline staging     |
| L0     | 1 MB     | register file — feeds the MAC array   |
| GM     | unbounded| off-chip global memory (not enforced) |

A buffer's region comes from its name prefix (`gm_`/`ub_`/`l1_`/`l0_`). On a violation,
`--verify` reports the region and time, the live buffers (largest first), and the root-cause op
with its source line. This matters for optimization because the two main levers below
**increase** the simultaneous footprint:

- **Batching** small transfers into one larger transfer pushes a single buffer's size up.
- **Allocating fresh buffers** to break a WAR hazard adds a buffer whose live range now
  overlaps the one it used to reuse.

Both can tip a region over capacity, so an optimization must be checked for correctness, not
just speed.

## Instructions

**Input:** a program written in the data-transfer and compute APIs (`alloc`, `gm_to_ub`,
`ub_to_gm`, `vadd`, `gm_to_l1`, `l1_to_l0`, `matrixmul`, `l0_to_gm`).

**Output:** the program's **bottleneck** — the operation (or resource) that dominates the
makespan — together with the evidence behind it (critical path, engine utilization,
bandwidth utilization) so it can guide a later optimization.

**Main tool:** run the program through the simulator (`simulator.py`) and analyze its
output. The simulator schedules the program, computes the critical path, and reports
per-op timing, engine utilization, and bandwidth utilization — everything needed to locate
the bottleneck.

### Step 1: Verify correctness before analyzing

Before any bottleneck analysis, confirm the input program is **correct** on this hardware —
a fast schedule is worthless if it overflows on-chip memory, and the input may already
violate a capacity constraint before you change anything. Run the `--verify` mode first:

```bash
python3 simulator.py --verify "<program>"
```

- **PASS** → the program fits in every bounded region (UB/L1/L0). Note the tightest
  region's headroom — it tells you how much slack later optimizations have to grow the
  footprint — and proceed to Step 2.
- **FAIL** → the program is incorrect as-is, so analyzing its performance is premature. The
  report gives you, per violation, the overflowing region and time, the live buffers (largest
  first) with their producers, and the **root-cause op with its source line**. Fix the
  program first, using the lever that matches the violation:
  - **Shrink the buffers** (smaller tiles) so the simultaneous footprint fits the region.
  - **Serialize independent tiles** whose live ranges overlap so they aren't all resident at
    once (trades some parallelism for footprint — acceptable here, since correctness comes
    first and the later steps recover the performance).
  - **Reduce the working set** in the region: reuse buffers sooner instead of allocating
    fresh ones that stay live.

  Apply a fix, then **re-run `--verify` and repeat until it reports PASS.** Only a passing
  program is worth taking into bottleneck analysis.

Carry the now-correct program (and the tightest region's headroom) into the next step.

### Step 2: Run the simulator

Run the program through `simulator.py`. For bottleneck analysis, use the **`--llm`** output
mode (compact, structured text meant for LLM consumption) together with **`--critical-path`**
so the critical chain is included:

```bash
python3 simulator.py --llm --critical-path "<program>"
```

Example:

```bash
python3 simulator.py --llm --critical-path "alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"
```

Pass the user's program verbatim. If buffer sizes are known, make sure each buffer has an
`alloc(name, size)` declaration — otherwise it defaults to 64 KB and the timing will be
inaccurate. Supported size units: B, KB, MB, GB.

The `--llm` run reports the sections you'll analyze next: `EXECUTION SUMMARY`,
`PER-OP STATISTICS`, `ENGINE UTILIZATION`, `BANDWIDTH UTILIZATION`, `PARALLELISM`, and
`CRITICAL PATH`. (For deeper queries — e.g. exact speedup-if-reduced estimates — re-run with
`--nx` to get the queryable NetworkX JSON graph.)

### Step 3: Analyze the output to identify the bottleneck

The bottleneck lives **on the critical path** — nothing off the critical path can be reduced
to shrink the makespan. Work through the output in this order:

1. **Start from the `CRITICAL PATH` section.** It lists the chain of ops whose end-to-end
   time equals `total_ns`, with each op's `duration` and each link's reason. Only these
   ops are worth optimizing first. Note its `fraction_of_makespan` — if it's near 100%, the
   program is essentially serial along this chain.

2. **Find the dominant op on the chain** — the one with the largest `duration` /
   `time_ratio` in `PER-OP STATISTICS`. This single op is the primary bottleneck candidate.

3. **Classify *why* it dominates** so the fix is clear:
   - **Compute-bound** — op is on VecUnit/CubeUnit. CubeUnit is size-independent (always
     at peak), so the data volume is the cost; reducing it (smaller tiles, fewer ops) is
     the lever. VecUnit is now size-dependent: a small `vadd` can sit below its 24 KB
     saturation point (`regime=floor`/`ramp`), in which case batching into a larger tile
     raises its throughput — treat that like a below-peak transfer.
   - **Saturated (bandwidth-bound) transfer** — a transfer engine at ~100% `bw_utilization`
     (`regime=saturated`, size at/above the curve's saturation point). It's running at peak;
     the only lever is moving less data or using a faster engine/level.
   - **Below-peak transfer** — low `bw_utilization` (`regime=floor` or `ramp`, size below the
     engine's saturation point — see the hardware table). The transfer is too small to reach
     peak bandwidth; **batching several small transfers into one larger transfer** raises
     utilization and cuts total time.

4. **Check the critical-path link reasons.** Each edge is either a **data hazard**
   (RAW/WAW/WAR on a named buffer) or **engine serialization** (two ops queued on the same
   engine):
   - A **WAR** hazard is often *avoidable* — the `--llm` output flags these with a `fix:`
     line (allocate a fresh destination buffer instead of reusing one to break the
     anti-dependency and unlock parallelism).
   - **Engine serialization** means one engine is oversubscribed — look for the same engine
     appearing repeatedly on the chain; spreading that work or overlapping independent tiles
     helps.

5. **Corroborate with `ENGINE UTILIZATION` and `PARALLELISM`.** A single engine near 100%
   busy while others idle confirms an engine bottleneck. Few/no parallel pairs confirms the
   program is serialized — the critical-path link reasons tell you by what.

Conclude with the **single most impactful bottleneck**: which op (and engine), how much of
the makespan it accounts for, why it dominates (compute / bandwidth / latency / hazard /
serialization), and the lever that would most reduce the makespan. This is the output that
guides the optimization step.

### Step 4: Propose and validate the optimization

Turn the diagnosis into a concrete change, then prove it helps by re-running the simulator.

1. **Pick the lever from the classification** in Step 3:
   - *Below-peak transfer (floor/ramp)* → batch small transfers into one larger transfer
     (push the size up to the engine's saturation point).
   - *Avoidable WAR hazard* → allocate a fresh destination buffer to break the
     anti-dependency and unlock parallelism.
   - *Engine serialization* → overlap independent tiles, or move work off the oversubscribed
     engine.
   - *Compute- or bandwidth-bound at peak* → reduce data volume (smaller/fewer tiles) or use
     a faster engine/memory level; the engine itself can't go faster.

2. **(Optional) Predict the payoff before editing.** Re-run with `--nx` and read the
   `speedup_if_reduced` field on the bottleneck node (pre-computed estimates for making that
   op 10% / 25% / 50% faster) and the graph-level `bottleneck` summary. This tells you
   whether the fix is worth it before you change the program.

3. **Apply the change to the program** — edit the `alloc` declarations and/or op list to
   implement the lever (e.g. merge two `gm_to_ub` loads into one larger transfer, or rename
   a reused destination buffer).

4. **Re-run the simulator** the same way as Step 2 and compare:

   ```bash
   python3 simulator.py --llm --critical-path "<optimized program>"
   ```

   - Did `total_ns` drop? By how much (the realized speedup)?
   - Did the bottleneck **move** to a different op/engine? If so, the chain has a new
     dominant cost — repeat Steps 3–4 on the new bottleneck.

5. **Verify the optimization is still correct.** A faster program is only valid if it still
   fits in on-chip memory. The two main levers both **grow** the live footprint (batching
   enlarges a buffer; a fresh WAR-breaking buffer adds an overlapping live range), so an
   optimization can introduce a memory-capacity violation that the timing modes won't show.
   Re-run the optimized program through `--verify`:

   ```bash
   python3 simulator.py --verify "<optimized program>"
   ```

   - **PASS** → the speedup is real and the program fits; keep the change.
   - **FAIL** → the optimization overflowed a region (read the region and time, the live
     buffers, and the root-cause source line in the report). The fix is invalid as-is —
     reconcile speed and correctness by, e.g.:
     - batching fewer tiles per transfer (a smaller merged buffer that still reaches the
       engine's saturation point but fits the region),
     - serializing some independent tiles so their live ranges no longer overlap (trades a
       little parallelism back for footprint), or
     - choosing a different lever that doesn't grow the region that overflowed.

     Then re-run both `--critical-path` and `--verify` until the program is both faster and
     correct.

6. **Stop when** the makespan no longer meaningfully improves, the critical path is balanced
   across engines (no single op dominates), or remaining costs are compute-/bandwidth-bound
   at peak (irreducible without changing the algorithm or data volume) — **and** `--verify`
   still reports PASS.

Report the final result: the change made, the before/after `total_ns` and speedup, the
`--verify` status (PASS, with the tightest region's headroom), and the new bottleneck (if
any) for the next round.
