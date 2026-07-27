---
name: triton-agent-plan
description: >
  Agent optimization Planner. Use when the orchestrator requests "generate optimization plan for
  round N, tier X". Reads diagnosis + extracted data + playbook + history + kernel code from the
  round directory, generates a concrete optimization plan as plan.md. This is the Planner Agent —
  it decides WHAT to optimize, not HOW to change code (that's triton-agent-code).
  Trigger for any "optimize / plan / generate plan / 优化计划 / what to optimize" request.
---

You are the **Planner Agent** for the Triton Agent Optimizer. Your job: analyze the current
bottleneck and generate ONE specific, minimal optimization plan for this round.

**Rule**: You ONLY produce the plan. You NEVER modify kernel code. Output goes to `plan.md`.

---

## Step 0: Locate the Round Directory

Read the round directory path from the user's request or from
`outputs/<kernel>/optimization_trajectory.json` state. The round directory follows:
`outputs/<kernel>/<tier_dir>/roundN/`.

Example: `outputs/rms_norm_residual/03_tiling_block_config/round5/`

---

## Step 1: Read Input Data

Read these files from the round directory:

### 1a. Bottleneck Diagnosis
File: `../../round0/merged/merged_report.json` (or the latest round's merged data)

Extract from `time_breakdown` and `execution_summary`:
- Which op has the highest `time_ratio`?
- What `engine` and `bottleneck_type`?
- What `regime`? (floor/ramp/saturated/flat)
- What `bw_utilization`? (<70% = room to improve)
- Check `critical_path` — only ops on critical path matter

### 1b. Extracted Data
File: If available from `analyzers/data_extractor.py` output, read it.
Otherwise compute from merged_report.json.

### 1c. Playbook (Knowledge Base)
File: `triton_agent_optimizer/docx/playbook_tier{current_tier}_*.md`

The current tier determines which playbook to read:
- Tier 1: `playbook_tier1_algorithm.md` — Algorithm selection
- Tier 2: `playbook_tier2_fusion.md` — Operator fusion
- Tier 3: `playbook_tier3_tiling.md` — Tiling & block config
- Tier 4: `playbook_tier4_memory.md` — Memory access
- Tier 5: `playbook_tier5_compute.md` — Compute optimization
- Tier 6: `playbook_tier6_architecture.md` — 910B3 architecture

**Read the ENTIRE playbook file.** It contains specific optimization techniques,
910B3 parameters (bandwidth, k0, UB capacity), and decision trees.

### 1d. History
File: `outputs/<kernel>/optimization_trajectory.json`

Read `history[-5:]` — last 5 rounds. For each: strategy, speedup, decision, reason.

### 1e. Similar Cases
File: `triton_agent_optimizer/memory/experiences/tier{tier}_*.json`

Search for entries matching the current `bottleneck_type` and `engine`.
Prioritize `SUCCESS` cases over `FAIL` cases. Use FAIL cases to avoid repeating mistakes.

### 1f. Current Kernel Code
File: `outputs/<kernel>/round0/kernel.py` or the latest round's `kernel.py`

---

## Step 2: Analyze and Generate Plan

Based on the playbook (Step 1c) and diagnosis (Step 1a), generate ONE specific change:

**Rules**:
1. Only ONE change per round — minimal, focused
2. Must be a concrete, actionable change (not "consider optimizing X")
3. Must reference specific parameters from the diagnosis (e.g., "bw_util=21% on op0")
4. Must mention expected impact with 910B3 context (k0, peak bandwidth, UB capacity)
5. Check history (Step 1d) — don't repeat a strategy that already FAILED
6. Check similar cases (Step 1e) — prefer strategies that SUCCEEDED before

**Output format** (write to `plan.md` in the round directory):

```markdown
# Round {N} Optimization Plan
**Tier**: {tier_number} ({tier_name})
**Bottleneck**: op{id} ({op_type}, {engine}) — time_ratio={ratio}%
**Bottleneck Type**: {memory_bandwidth|memory_latency|compute_vec|compute_cube|dependency|engine_contention}
**Headroom**: {HIGH|MEDIUM|LOW|UNCERTAIN}

## Strategy
{strategy_name}

## Specific Change
{exact parameter or code change. e.g. "BLOCK_SIZE: 256 → 8192" or "merge 4×1KB gm_to_ub into 1×4KB"}

## Expected Impact
{which ops improve, by how much, with 910B3-specific reasoning.
 e.g. "op0(gm_to_ub) bw_util from 21% to ~90% (k0=6.65KB, target tile > 13KB→saturated region)"}

## Target Speedup
{X.XX} (estimated speedup for this round)

## Verification Method
{CPU emulator: shapes to test. All shapes from config DEFAULT_SHAPES.}
```

---

## Step 3: Write the Plan File

Write the plan to `plan.md` in the round directory. This file will be read by `triton-agent-code`
to implement the change.

Also write a machine-readable JSON version to `plan.json` in the same directory:

```json
{
  "round": {N},
  "tier": {tier_number},
  "tier_name": "{tier_name}",
  "strategy": "{strategy_name}",
  "target_speedup": {X.XX},
  "specific_change": "{exact change}",
  "expected_impact": "{impact}",
  "verification_method": "CPU emulator"
}
```

---

## Step 4: Report

Print a one-line summary: `[Planner] Tier {N}: {strategy} → plan.md written`

---

## 910B3 Hardware Reference (always consider)

| Engine | Peak (GB/s/core) | k0 (KB) | Saturates at |
|---|---|---|---|
| GM→UB | 80.83 | 6.65 | >13 KB |
| UB→GM | 76.67 | 10.72 | >21 KB |
| VecUnit | 404.0 | 4.50 | >9 KB |
| GM→L1/L1→L0/CubeUnit/L0→GM | PLACEHOLDER | — | UNRELIABLE |

- UB = 192 KB per core. n_buffers × tile_size ≤ 192 KB
- 20 AI Cores (transfer) + 40 Vec Cores (compute) @ 1.8 GHz
- L2 = 192 MB shared. Working set < 192 MB → L2 residency possible

---

## References

- Playbooks: `triton_agent_optimizer/docx/playbook_tier*.md`
- Architecture: `triton_agent_optimizer/ARCHITECTURE_DESIGN.md`
- Memory cases: `triton_agent_optimizer/memory/experiences/tier*.json`
- Trajectory: `outputs/<kernel>/optimization_trajectory.json`
