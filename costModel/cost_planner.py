#!/usr/bin/env python3
"""
Cost Model bridge: feeds (op, shape) into cost_emulator and returns a plan code.

Pipeline position:
  input(op+shape) -> [cost_planner.plan] -> plan code -> triton-gen Step 3

plan() does three things:
  1. (op, shape) -> cost_emulator DSL   (matmul/vadd mapped; others stubbed)
  2. subprocess call to cost_emulator/simulator.py --llm --critical-path
  3. parse --llm output -> plan code (bottleneck/parallel/bandwidth/tiling hints)
     + pass through raw_llm verbatim

Design: only key metrics are extracted (total_ns / bottleneck op / parallel-pair
count); the cost_emulator --llm full text is passed through unchanged for the
triton-gen LLM to read in depth -- this keeps the parser simple and robust
without needing the real output locally to tune regexes. Complex tiling /
parallel planning is left to the LLM.

Note: cost_emulator/simulator.py uses Python 3.10+ syntax (str | None). This
module uses __future__ annotations for 3.9 compatibility, but run_simulator
invokes COST_SIM_PYTHON which must be a 3.10+ interpreter (local miniforge 3.9
cannot run simulator; set COST_SIM_PYTHON to a 3.10+ python, or run in the
NPU / collaborator environment).

Usage:
  python cost_planner.py matmul 1024 1024 1024
  python cost_planner.py vadd 4096
  or: from cost_planner import plan; plan("matmul", {"M":1024,"N":1024,"K":1024})
"""
from __future__ import annotations
import os
import re
import sys
import json
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PATH = os.path.join(_HERE, "cost_emulator", "simulator.py")
# requires 3.10+; default python3, override via env COST_SIM_PYTHON
SIM_PYTHON = os.environ.get("COST_SIM_PYTHON", "python3")

ELEM = 2  # fp16 bytes per element


# ── DSL builders (op_kind, shape -> cost_emulator program string) ────────────
# cost_emulator 7 engines: gm_to_ub / ub_to_gm / vadd(VecUnit) /
#                          gm_to_l1 / l1_to_l0 / matrixmul(CubeUnit) / l0_to_gm

def _kb(elems: float) -> float:
    """element count -> KB (fp16)."""
    return elems * ELEM / 1024.0


def dsl_matmul(M: int, N: int, K: int, tile: int = 128) -> str:
    """C[M,N] = A[M,K] @ B[K,N] -- cube path GM->L1->L0->Cube->L0->GM.

    tile is the block on L1/L0 (BM=BN=tile, BK=tile representative). One tile's
    transfer is what cost_emulator times; the full tiling strategy is decided
    later (by the LLM) from the critical path.
    """
    a, b, c = _kb(M * K), _kb(K * N), _kb(M * N)
    t = _kb(tile * tile)
    return (
        f"alloc(gm_a,{a:.1f}KB) alloc(gm_b,{b:.1f}KB) alloc(gm_c,{c:.1f}KB) "
        f"alloc(l1_a,{t:.1f}KB) alloc(l1_b,{t:.1f}KB) "
        f"alloc(l0_a,{t:.1f}KB) alloc(l0_b,{t:.1f}KB) alloc(l0_c,{t:.1f}KB) "
        f"gm_to_l1(l1_a,gm_a) gm_to_l1(l1_b,gm_b) "
        f"l1_to_l0(l0_a,l1_a) l1_to_l0(l0_b,l1_b) "
        f"matrixmul(l0_c,l0_a,l0_b) l0_to_gm(gm_c,l0_c)"
    )


def dsl_vadd(N: int, tile: int = 8192) -> str:  # 8192 elem = 16KB UB block
    """C[N] = A[N] + 1.0 -- vec path GM->UB->VecUnit->UB->GM (elementwise proxy).

    cost_emulator's vadd is vector+scalar SIMD (compute_intensity=1). vec+vec
    elementwise is equivalent to one vec compute in this model, approximated
    by vadd.
    """
    full, t = _kb(N), _kb(tile)
    return (
        f"alloc(gm_a,{full:.1f}KB) alloc(gm_c,{full:.1f}KB) "
        f"alloc(ub_a,{t:.1f}KB) alloc(ub_c,{t:.1f}KB) "
        f"gm_to_ub(ub_a,gm_a) vadd(ub_c,ub_a,1.0) ub_to_gm(gm_c,ub_c)"
    )


DSL_BUILDERS = {
    "matmul": lambda s: dsl_matmul(s["M"], s["N"], s["K"]),
    "vadd":   lambda s: dsl_vadd(s["N"]),
}


# ── Invoke cost_emulator ─────────────────────────────────────────────────────

def run_simulator(dsl: str, critical_path: bool = True) -> str:
    """subprocess call to simulator.py --llm, returns stdout. COST_SIM_PYTHON must be 3.10+."""
    cmd = [SIM_PYTHON, SIM_PATH, "--llm"]
    if critical_path:
        cmd.append("--critical-path")
    cmd.append(dsl)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(
            f"cost_emulator exit {r.returncode} (ensure {SIM_PYTHON} is 3.10+):\n"
            f"{r.stderr.strip()[:500]}"
        )
    return r.stdout


# ── Parse --llm output (key metrics only; deep analysis via raw_llm) ─────────

def _f(m) -> float | None:
    return float(m.group(1)) if m else None


def parse_llm(out: str) -> dict:
    """Extract total_ns / execution mode / bottleneck op / parallel-pair count
    from simulator --llm stdout.

    Uses the TIME BREAKDOWN `op{i}: sig  duration_ns=d  time_ratio=r%` lines
    (single line, stable format) rather than the multi-line PER-OP block. On
    parse failure no exception is raised -- missing fields are None and the
    full text is available in raw_llm for the LLM.
    """
    total = _f(re.search(r"total_ns:\s*([\d.]+)", out))
    mode_m = re.search(r"execution_mode:\s*(\w+)", out)
    mode = mode_m.group(1) if mode_m else None

    ops = []
    for m in re.finditer(
        r"^op(\d+):\s+(.+?)\s{2,}duration_ns=([\d.]+)\s+time_ratio=([\d.]+)%",
        out, re.M,
    ):
        ops.append({
            "op": f"op{m.group(1)}",
            "sig": m.group(2).strip(),
            "duration_ns": float(m.group(3)),
            "time_ratio": float(m.group(4)) / 100.0,
        })

    bottleneck = max(ops, key=lambda o: o["time_ratio"]) if ops else None
    parallel_pairs = len(re.findall(r"op\d+\s*\|\|\s*op\d+", out))

    return {
        "total_ns": total,
        "execution_mode": mode,
        "ops": ops,
        "bottleneck": bottleneck,
        "parallel_pairs_count": parallel_pairs,
    }


def _suggestions(parsed: dict) -> list:
    """Readable planning hints derived from the key metrics (for the LLM)."""
    s = []
    bn = parsed["bottleneck"]
    if bn:
        s.append(f"bottleneck: {bn['op']} ({bn['sig']}, time_ratio={bn['time_ratio']:.0%})")
    if parsed["parallel_pairs_count"]:
        s.append(f"{parsed['parallel_pairs_count']} parallel op-pair(s) -- "
                 f"consider double-buffering / pipelining to hide transfers")
    elif parsed["execution_mode"] == "sequential":
        s.append("fully sequential: check RAW/WAW/WAR, use independent dest buffers to enable parallelism")
    return s


# ── Entry: plan ──────────────────────────────────────────────────────────────

def plan(op_kind: str, shapes: dict) -> dict:
    """op + shape -> plan code (dict).

    Return shape:
      supported=False  -> op not mapped, returns a default plan (stub)
      supported=True, error -> DSL built but simulator call failed (check 3.10+)
      supported=True   -> includes dsl / plan(key metrics + hints) / raw_llm (passthrough)
    """
    builder = DSL_BUILDERS.get(op_kind)
    if builder is None:
        return {
            "supported": False,
            "op": op_kind,
            "suggestions": [
                f"'{op_kind}' not mapped to cost_emulator DSL -- using default plan. "
                f"supported: {sorted(DSL_BUILDERS)}. "
                f"you can hand-write DSL (matmul cube path / vadd vec path) and call run_simulator."
            ],
        }

    dsl = builder(shapes)
    try:
        out = run_simulator(dsl)
    except Exception as e:
        return {
            "supported": True,
            "op": op_kind,
            "shapes": shapes,
            "dsl": dsl,
            "error": str(e),
            "suggestions": [
                "cost_emulator call failed -- set COST_SIM_PYTHON to a 3.10+ python.",
                f"DSL generated, you can run it manually: {SIM_PYTHON} {SIM_PATH} --llm \"{dsl}\"",
            ],
        }

    parsed = parse_llm(out)
    return {
        "supported": True,
        "op": op_kind,
        "shapes": shapes,
        "dsl": dsl,
        "plan": {
            "total_ns": parsed["total_ns"],
            "execution_mode": parsed["execution_mode"],
            "bottleneck": parsed["bottleneck"],
            "parallel_pairs_count": parsed["parallel_pairs_count"],
            "suggestions": _suggestions(parsed),
        },
        # pass through cost_emulator --llm full text for the triton-gen LLM
        "raw_llm": out,
    }


def _cli():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print("usage: cost_planner.py <matmul|vadd> <shapes...>")
        print("  matmul M N K    e.g.  cost_planner.py matmul 1024 1024 1024")
        print("  vadd   N        e.g.  cost_planner.py vadd 4096")
        print(f"  env COST_SIM_PYTHON=<3.10+ python>  (default: {SIM_PYTHON})")
        return
    op = sys.argv[1]
    nums = [int(x) for x in sys.argv[2:]]
    if op == "matmul":
        if len(nums) != 3:
            print("matmul needs M N K"); return
        shapes = {"M": nums[0], "N": nums[1], "K": nums[2]}
    elif op == "vadd":
        if len(nums) != 1:
            print("vadd needs N"); return
        shapes = {"N": nums[0]}
    else:
        print(f"unknown op '{op}'; supported: {sorted(DSL_BUILDERS)}"); return
    print(json.dumps(plan(op, shapes), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
