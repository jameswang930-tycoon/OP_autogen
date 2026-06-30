#!/usr/bin/env python3
"""
Pipeline simulator for GM/UB/L1/L0 data-transfer hardware.

  Engine 0 (GM→UB):   gm_to_ub(ub_dst, gm_src)
  Engine 1 (UB→GM):   ub_to_gm(gm_dst, ub_src)
  Engine 2 (VecUnit):  vadd(ub_dst, ub_src, scalar)
  Engine 3 (GM→L1):   gm_to_l1(l1_dst, gm_src)
  Engine 4 (L1→L0):   l1_to_l0(l0_dst, l1_src)
  Engine 5 (CubeUnit): matrixmul(l0_dst, l0_src_a, l0_src_b)
  Engine 6 (L0→GM):   l0_to_gm(gm_dst, l0_src)

Different engines can run in parallel; data hazards force serialization.
Hazard types detected:
  RAW  – Read After Write  (true dependency)
  WAW  – Write After Write (output dependency)
  WAR  – Write After Read  (anti-dependency)

Buffer sizes are declared with alloc(name, size) and combined with a realistic
size-dependent bandwidth model (bandwidth ramps with transfer size and saturates
at peak) to compute transfer durations in nanoseconds.

Programs may use `for VAR in range(...) { BODY }` loops with address-offset operands
(e.g. gm_1 + m*1KB). An emulator front-end executes the loops and emits the concrete
instruction stream the hardware runs; the simulator schedules that flat stream.
"""

import json
import re
import sys
from dataclasses import dataclass

import networkx as nx

# ── Configuration ─────────────────────────────────────────────────────────────

ENGINE_FOR = {
    'gm_to_ub':  0,
    'ub_to_gm':  1,
    'vadd':      2,
    'gm_to_l1':  3,
    'l1_to_l0':  4,
    'matrixmul': 5,
    'l0_to_gm':  6,
}
ENG_NAME = {
    0: 'GM→UB',
    1: 'UB→GM',
    2: 'VecUnit',
    3: 'GM→L1',
    4: 'L1→L0',
    5: 'CubeUnit',
    6: 'L0→GM',
}

# Realistic size-dependent bandwidth model.
#
# A real transfer engine does not move data at a single constant rate: tiny
# transfers can't fill the pipe, so the achieved bandwidth RAMPS UP with transfer
# size until it saturates. We model bandwidth as a piecewise-LINEAR function of
# transfer size, given as a list of (size_kb, bandwidth_GB_per_s) breakpoints:
#
#   size <= first breakpoint        → clamped to the first bandwidth   (floor)
#   between two breakpoints         → linear interpolation             (ramp)
#   size >= last breakpoint         → clamped to the last bandwidth    (saturated)
#
# Duration is then  size / bandwidth(size), reported in real NANOSECONDS.
#
# UNIT CONVENTION: bandwidth is GB/s with GB = 1e9 bytes (standard for bandwidth
# specs); sizes are KB with KB = 1024 bytes (matches parse_size). So the achieved
# rate in KB/ns is  bw_GB_s * 1e9 / 1024 / 1e9 = bw_GB_s / 1024, and
#   duration_ns = size_kb / (bw_GB_s / 1024) = size_kb * 1024 / bw_GB_s.
#
# Engine 0 (GM→UB) uses MEASURED real-world numbers:
#   1 KB → 100 GB/s, ramping linearly to 1500 GB/s at 12 KB, saturated above.
#
# Engine 2 (VecUnit) also uses MEASURED numbers, but the measurement was taken in
# TFLOPS (compute throughput): 1 KB → 1 TFLOPS, ramping linearly to 8 TFLOPS at 24 KB,
# saturated above. For an fp16 vadd (1 FLOP per element, 2 bytes per element) the
# arithmetic intensity is a constant 2 bytes/FLOP, so TFLOPS converts to an effective
# bandwidth by a CONSTANT factor:  GB/s = TFLOPS × 2000  (1 FLOP / 2 bytes = 0.5 FLOP/B
# ⇒ B/s = FLOP/s / 0.5 = 2 × FLOP/s; in GB/s vs TFLOPS that is ×2000). Because the
# factor is constant, a curve that is linear in TFLOPS-vs-size is identical to one
# linear in GB/s-vs-size, so VecUnit is just an ordinary size-dependent GB/s curve here
# and needs no special compute path. The measured 1→8 TFLOPS over 1→24 KB becomes
# 2000→16000 GB/s over 1→24 KB.
#
# The remaining transfer engines use PLACEHOLDER curves (TODO: replace with measured
# numbers). Their shapes are scaled from the engines' former relative peak bandwidths so
# the whole schedule stays in one coherent time unit, but only GM→UB and VecUnit are
# calibrated. CubeUnit has no measured numbers yet, so it keeps a single flat breakpoint
# (size-independent) until its throughput is measured.
BANDWIDTH_CURVE_GB_S = {
    0: [(1.0, 5.0),    (12.0, 75.0)],      # GM→UB  — MEASURED ÷20 (单核, 20 AI Core)
    1: [(1.0, 15.15),  (8.0, 72.25), (32.0, 73.05)],  # UB→GM  — MEASURED ÷20 (单核; 实测303/1445/1461 ÷20)
    2: [(1.0, 50.0),   (24.0, 400.0)],     # VecUnit — MEASURED ÷40 (单核, 40 Vec Core)
    3: [(1.0, 50.0),   (12.0, 750.0)],     # GM→L1  — PLACEHOLDER (TODO: measure)
    4: [(1.0, 100.0),  (12.0, 2000.0)],    # L1→L0  — PLACEHOLDER (TODO: measure)
    5: [(1.0, 3000.0)],                    # CubeUnit — PLACEHOLDER, flat (TODO: measure)
    6: [(1.0, 50.0),   (12.0, 750.0)],     # L0→GM  — PLACEHOLDER (TODO: measure)
}

# An engine is "flat" (size-independent throughput) iff its curve is a single breakpoint.
# Only CubeUnit (engine 5) is flat now — it has no measured size-dependent numbers yet.
# All other engines, including VecUnit, ramp with transfer size.

DEFAULT_SIZE_KB = 64.0   # fallback when a buffer has no alloc declaration


def peak_bandwidth_gb_s(engine: int) -> float:
    """The saturated (maximum) bandwidth of an engine — the last curve breakpoint."""
    return BANDWIDTH_CURVE_GB_S[engine][-1][1]


def bandwidth_at_size(engine: int, size_kb: float) -> tuple[float, str]:
    """Achieved bandwidth (GB/s) for a transfer of `size_kb` on `engine`, plus the
    regime label. Piecewise-linear interpolation between the engine's breakpoints;
    clamped to the first/last bandwidth outside the breakpoint range."""
    curve = BANDWIDTH_CURVE_GB_S[engine]
    if len(curve) == 1:
        return curve[-1][1], 'flat'
    first_size, first_bw = curve[0]
    last_size,  last_bw  = curve[-1]
    if size_kb <= first_size:
        return first_bw, 'floor'
    if size_kb >= last_size:
        return last_bw, 'saturated'
    # Find the segment [lo, hi] containing size_kb and linearly interpolate.
    for (lo_s, lo_bw), (hi_s, hi_bw) in zip(curve, curve[1:]):
        if lo_s <= size_kb <= hi_s:
            frac = (size_kb - lo_s) / (hi_s - lo_s)
            return lo_bw + frac * (hi_bw - lo_bw), 'ramp'
    return last_bw, 'saturated'   # unreachable; defensive


# Physical capacity of each on-chip memory region (KB). A program is only correct if,
# at every cycle, the total size of buffers simultaneously live in a region never
# exceeds its capacity. These are made-up but plausible: small fast register file (L0),
# larger staging SRAM (L1), a working buffer for the vector pipeline (UB). GM (off-chip
# global memory) is treated as effectively unbounded and only reported, never enforced.
#
# A buffer's region is identified by its name prefix (gm_/ub_/l1_/l0_).
MEMORY_CAPACITY_KB = {
    'UB': 512.0,    # Unified Buffer   — vector pipeline working set
    'L1': 2048.0,   # L1 SRAM          — matrix pipeline staging (2 MB)
    'L0': 1024.0,   # L0 register file — feeds the MAC array (1 MB)
    'GM': None,     # Global Memory    — unbounded (reported, not enforced)
}

# Map a buffer-name prefix to its memory region. Unknown prefixes are ignored
# (not capacity-checked) so the verifier never flags buffers it can't place.
REGION_FOR_PREFIX = {
    'gm': 'GM',
    'ub': 'UB',
    'l1': 'L1',
    'l0': 'L0',
}


def region_of(buffer: str) -> str | None:
    """Return the memory region ('UB'/'L1'/'L0'/'GM') a buffer lives in, or None."""
    prefix = buffer.split('_', 1)[0].lower()
    return REGION_FOR_PREFIX.get(prefix)


def bandwidth_profile(engine: int, size_kb: float) -> tuple[float, float, float, str]:
    """
    Map a transfer/compute size to (duration_ns, effective_bw_GB_s, utilization, regime).

    Bandwidth ramps with transfer size (see BANDWIDTH_CURVE_GB_S): small transfers
    achieve below-peak bandwidth, so they cost proportionally more time per byte.
    Utilization = effective_bw / peak_bw (1.0 == saturated).

      duration_ns = size_kb * 1024 / effective_bw_GB_s
      (GB = 1e9 bytes, KB = 1024 bytes → KB/ns = GB_s / 1024)
    """
    eff_bw, regime = bandwidth_at_size(engine, size_kb)
    peak = peak_bandwidth_gb_s(engine)
    duration_ns = (size_kb * 1024.0) / eff_bw if eff_bw else 0.0
    util = eff_bw / peak if peak else 0.0
    return duration_ns, eff_bw, util, regime


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Op:
    idx:               int
    name:              str
    dst:               str
    src:               str
    engine:            int
    size_kb:           float = 0.0
    duration:          float = 1.0   # nanoseconds (size-dependent transfer time)
    start:             float = 0.0   # nanoseconds
    end:               float = 0.0   # nanoseconds
    line:              int   = 0     # 1-based source line of this op (for verification reports)
    scalar:            float = 0.0   # for vadd; unused for transfer/matmul ops
    src2:              str   = ''    # second source for matrixmul(dst, src, src2)
    effective_bw:      float = 0.0   # achieved GB/s (size-dependent via the ramp curve)
    bw_utilization:    float = 1.0   # effective_bw / peak_bw (1.0 == saturated)
    regime:            str   = ''    # 'floor' | 'ramp' | 'saturated' | 'flat'
    # Address-offset metadata for operands emitted by the emulator (loops). dst/src/src2
    # above hold the BASE buffer name; these record the per-operand tile offset (KB into
    # the base buffer) and whether the operand is a slice (gm_1 + m*1KB) vs the whole
    # buffer (gm_1). Plain (non-loop) programs leave these at their defaults, so hazard
    # detection collapses to whole-buffer base-name comparison exactly as before.
    dst_off:           float = 0.0
    src_off:           float = 0.0
    src2_off:          float = 0.0
    dst_sliced:        bool  = False
    src_sliced:        bool  = False
    src2_sliced:       bool  = False

Deps = dict[int, set[int]]   # deps[i] = set of j that op-i must wait for

# ── Parsing ───────────────────────────────────────────────────────────────────

_OP_RE    = re.compile(r'(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)')
_VEC_RE   = re.compile(r'(vadd)\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*([\d.]+)\s*\)')
_MAT_RE   = re.compile(r'(matrixmul)\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)')
_ALLOC_RE = re.compile(r'\balloc\s*\(\s*(\w+)\s*,\s*([\d.]+\s*[A-Za-z]*)\s*\)')

def parse_size(s: str) -> float:
    """Convert a human-readable size string → KB (float)."""
    s = s.strip().upper().replace(' ', '')
    if s.endswith('GB'):
        return float(s[:-2]) * 1024 * 1024
    if s.endswith('MB'):
        return float(s[:-2]) * 1024
    if s.endswith('KB'):
        return float(s[:-2])
    if s.endswith('B'):
        return float(s[:-1]) / 1024
    return float(s)   # bare number → assume KB

def parse(prog: str) -> tuple[list[Op], dict[str, float]]:
    sizes: dict[str, float] = {}
    for m in _ALLOC_RE.finditer(prog):
        sizes[m.group(1)] = parse_size(m.group(2))

    # Track positions already consumed by multi-arg ops so _OP_RE doesn't re-match them
    special_spans: list[tuple[int, int]] = []
    ops: list[Op] = []

    for m in _VEC_RE.finditer(prog):
        special_spans.append((m.start(), m.end()))
        name, dst, src, scalar = m.group(1), m.group(2), m.group(3), float(m.group(4))
        if not dst.startswith('ub_') or not src.startswith('ub_'):
            print(f"  warning: vadd operands should be UB buffers (got dst={dst}, src={src})",
                  file=sys.stderr)
        ops.append(Op(len(ops), name, dst, src, ENGINE_FOR[name],
                      scalar=scalar))

    for m in _MAT_RE.finditer(prog):
        special_spans.append((m.start(), m.end()))
        name, dst, src, src2 = m.group(1), m.group(2), m.group(3), m.group(4)
        if not dst.startswith('l0_') or not src.startswith('l0_') or not src2.startswith('l0_'):
            print(f"  warning: matrixmul operands should be L0 buffers "
                  f"(got dst={dst}, src={src}, src2={src2})", file=sys.stderr)
        ops.append(Op(len(ops), name, dst, src, ENGINE_FOR[name],
                      src2=src2))

    for m in _OP_RE.finditer(prog):
        # Skip positions already matched by special regexes
        if any(s <= m.start() < e for s, e in special_spans):
            continue
        name, dst, src = m.group(1), m.group(2), m.group(3)
        if name == 'alloc':
            continue
        if name not in ENGINE_FOR:
            sys.exit(f"  error: unknown operation '{name}'\n"
                     f"  valid ops: alloc, gm_to_ub, ub_to_gm, vadd, "
                     f"gm_to_l1, l1_to_l0, matrixmul, l0_to_gm")
        ops.append(Op(len(ops), name, dst, src, ENGINE_FOR[name]))

    # Re-sort by position in source so ops are in program order
    pos_map: dict[int, int] = {}  # op.idx -> start position in prog
    for m in _VEC_RE.finditer(prog):
        for op in ops:
            if (op.name == m.group(1) and op.dst == m.group(2)
                    and op.src == m.group(3) and op.idx not in pos_map):
                pos_map[op.idx] = m.start()
                break
    for m in _MAT_RE.finditer(prog):
        for op in ops:
            if (op.name == m.group(1) and op.dst == m.group(2)
                    and op.src == m.group(3) and op.src2 == m.group(4)
                    and op.idx not in pos_map):
                pos_map[op.idx] = m.start()
                break
    for m in _OP_RE.finditer(prog):
        if any(s <= m.start() < e for s, e in special_spans):
            continue
        name, dst, src = m.group(1), m.group(2), m.group(3)
        if name == 'alloc' or name not in ENGINE_FOR:
            continue
        for op in ops:
            if (op.name == name and op.dst == dst
                    and op.src == src and op.idx not in pos_map):
                pos_map[op.idx] = m.start()
                break

    ops.sort(key=lambda o: pos_map.get(o.idx, 0))
    for new_idx, op in enumerate(ops):
        # Source line = 1 + number of newlines before this op's start position
        op.line = prog.count('\n', 0, pos_map.get(op.idx, 0)) + 1
        op.idx  = new_idx

    return ops, sizes

# ── Emulator front-end (loops + address expressions) ───────────────────────────
#
# The simulator's hardware model has no notion of a loop: seven engines pull ops off
# a flat list. A `for` loop is therefore just shorthand for "the body, N times". The
# emulator is a small front-end that *executes the program's control flow* — it walks
# `for VAR in range(...) { BODY }` (bodies may hold several statements and nest other
# loops), evaluates address expressions like `gm_1 + m * 1KB`, and EMITS the concrete
# instruction stream that the hardware actually runs. The simulator back-end then
# schedules that flat stream unchanged.
#
# Emitting the full list up front is deliberate: ASAP scheduling and hazard analysis
# are global passes that need every instruction in hand, so a lazy stream would just be
# drained at the scheduling boundary anyway. The result is identical to a hand-typed
# program — exact timing, no approximation of the loop.
#
# A program with no `for` and no `+` address-offset expression takes the original
# regex parse() path untouched, so every existing program behaves byte-for-byte as
# before; only loop/offset programs go through this expression-aware path.

_TOKEN_RE = re.compile(r"""
    (?P<NUM>     \d+\.?\d*\s*(?:GB|MB|KB|B)? ) |   # 100, 2.0, 1KB, 512B …
    (?P<ID>      [A-Za-z_]\w* )                |   # identifiers / op & keyword names
    (?P<PUNC>    [(){}+\-*,] )                 |   # structural punctuation
    (?P<WS>      \s+ )                             # whitespace (skipped)
""", re.VERBOSE)

_SIZE_SUFFIX = re.compile(r'(GB|MB|KB|B)$', re.IGNORECASE)


def _tokenize(prog: str) -> list[tuple[str, str, int]]:
    """Lex a program into (kind, text, line) tokens. NUM keeps its size suffix verbatim;
    line is the 1-based source line, carried through so emitted ops can report it."""
    toks: list[tuple[str, str, int]] = []
    pos = 0
    while pos < len(prog):
        m = _TOKEN_RE.match(prog, pos)
        if not m:
            sys.exit(f"  error: cannot tokenize near: {prog[pos:pos+20]!r}")
        line = prog.count('\n', 0, pos) + 1
        pos = m.end()
        kind = m.lastgroup
        if kind == 'WS':
            continue
        toks.append((kind, m.group().strip(), line))
    return toks


class _Stream:
    """Tiny cursor over the token list with the few helpers the parser needs."""
    def __init__(self, toks: list[tuple[str, str, int]]):
        self.toks = toks
        self.i = 0

    def peek(self) -> tuple[str, str, int] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> tuple[str, str, int]:
        if self.i >= len(self.toks):
            sys.exit("  error: unexpected end of program")
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> None:
        t = self.next()
        if t[1] != text:
            sys.exit(f"  error: expected '{text}', got '{t[1]}'")


# An expression is a sum of terms; a term is a product of factors. A factor is a NUM
# (possibly size-suffixed → KB), an identifier (a loop variable or a buffer base name),
# or a parenthesised expression. We keep the parse tree as nested tuples and evaluate
# it against the loop-variable environment when the statement runs.
#   ('num',  kb)              numeric literal already in KB
#   ('var',  name)            identifier — resolved at eval time (loop var or buffer)
#   ('add',  [factors...])    sum
#   ('mul',  [factors...])    product

def _parse_expr(s: _Stream) -> tuple:
    terms = [_parse_term(s)]
    while s.peek() and s.peek()[1] in ('+', '-'):
        op = s.next()[1]
        rhs = _parse_term(s)
        terms.append(('neg', rhs) if op == '-' else rhs)
    return ('add', terms) if len(terms) > 1 else terms[0]


def _parse_term(s: _Stream) -> tuple:
    factors = [_parse_factor(s)]
    while s.peek() and s.peek()[1] == '*':
        s.next()
        factors.append(_parse_factor(s))
    return ('mul', factors) if len(factors) > 1 else factors[0]


def _parse_factor(s: _Stream) -> tuple:
    kind, text, _line = s.next()
    if text == '(':
        e = _parse_expr(s)
        s.expect(')')
        return e
    if kind == 'NUM':
        return ('num', parse_size(text))
    if kind == 'ID':
        return ('var', text)
    sys.exit(f"  error: unexpected token '{text}' in expression")


# Statement AST:
#   ('for',  var, start_expr, stop_expr, step_expr, body_statements)
#   ('call', name, [arg_exprs], line)  — alloc / gm_to_ub / vadd / matrixmul / …

def _parse_statements(s: _Stream, stop: str | None) -> list[tuple]:
    stmts: list[tuple] = []
    while True:
        t = s.peek()
        if t is None or (stop is not None and t[1] == stop):
            return stmts
        if t[1] == 'for':
            stmts.append(_parse_for(s))
        else:
            stmts.append(_parse_call(s))


def _parse_for(s: _Stream) -> tuple:
    s.expect('for')
    var = s.next()[1]
    s.expect('in')
    s.expect('range')
    s.expect('(')
    args = [_parse_expr(s)]
    while s.peek() and s.peek()[1] == ',':
        s.next()
        args.append(_parse_expr(s))
    s.expect(')')
    if not 1 <= len(args) <= 3:
        sys.exit("  error: range() takes 1 to 3 arguments")
    s.expect('{')
    body = _parse_statements(s, stop='}')
    s.expect('}')
    # Normalise to (start, stop, step) exprs; range(n) → start 0, step 1.
    if len(args) == 1:
        start, stop, step = ('num', 0.0), args[0], ('num', 1.0)
    elif len(args) == 2:
        start, stop, step = args[0], args[1], ('num', 1.0)
    else:
        start, stop, step = args
    return ('for', var, start, stop, step, body)


def _parse_call(s: _Stream) -> tuple:
    _, name, line = s.next()
    s.expect('(')
    args: list[tuple] = []
    if s.peek() and s.peek()[1] != ')':
        args.append(_parse_expr(s))
        while s.peek() and s.peek()[1] == ',':
            s.next()
            args.append(_parse_expr(s))
    s.expect(')')
    return ('call', name, args, line)


# ── Expression evaluation ──────────────────────────────────────────────────────

def _eval_num(expr: tuple, env: dict[str, float]) -> float:
    """Evaluate a purely-numeric expression (loop bounds, scalars, offset terms).
    Identifiers must resolve to a loop variable; a bare buffer name here is an error."""
    tag = expr[0]
    if tag == 'num':
        return expr[1]
    if tag == 'var':
        if expr[1] not in env:
            sys.exit(f"  error: '{expr[1]}' is not a loop variable in a numeric context")
        return env[expr[1]]
    if tag == 'neg':
        return -_eval_num(expr[1], env)
    if tag == 'add':
        return sum(_eval_num(e, env) for e in expr[1])
    if tag == 'mul':
        out = 1.0
        for e in expr[1]:
            out *= _eval_num(e, env)
        return out
    sys.exit("  error: malformed numeric expression")


def _eval_operand(expr: tuple, env: dict[str, float]) -> tuple[str, float, bool]:
    """Evaluate a buffer operand to (base_name, offset_kb, sliced).

    An operand is one base buffer name optionally combined with numeric offset terms,
    e.g. `gm_1` → ('gm_1', 0, False); `gm_1 + m * 1KB` → ('gm_1', m*1, True). Exactly
    one identifier must be a buffer (not a loop var); everything else is numeric."""
    bases: list[str] = []
    offset = 0.0
    sliced = False

    def walk(e: tuple, sign: float) -> None:
        nonlocal offset, sliced
        tag = e[0]
        if tag == 'add':
            for sub in e[1]:
                walk(sub, sign)
        elif tag == 'neg':
            walk(e[1], -sign)
        elif tag == 'var' and e[1] not in env:
            bases.append(e[1])          # a buffer base name
        else:
            offset += sign * _eval_num(e, env)
            sliced = True               # any explicit offset term ⇒ this is a tile

    walk(expr, 1.0)
    if len(bases) != 1:
        sys.exit(f"  error: operand must reference exactly one buffer "
                 f"(found {bases or 'none'})")
    return bases[0], offset, sliced


def _operand_name(expr: tuple) -> str:
    """Extract the bare buffer name from an alloc's first argument."""
    if expr[0] == 'var':
        return expr[1]
    sys.exit("  error: alloc's first argument must be a buffer name")


# ── Program execution → instruction stream ─────────────────────────────────────

def _run_statements(stmts: list[tuple], env: dict[str, float],
                    ops: list[Op], sizes: dict[str, float]) -> None:
    """Execute statements in order, expanding loops and emitting concrete Ops."""
    for st in stmts:
        if st[0] == 'for':
            _, var, e_start, e_stop, e_step, body = st
            start = int(round(_eval_num(e_start, env)))
            stop  = int(round(_eval_num(e_stop,  env)))
            step  = int(round(_eval_num(e_step,  env)))
            if step == 0:
                sys.exit("  error: range() step must be non-zero")
            for v in range(start, stop, step):
                env[var] = float(v)
                _run_statements(body, env, ops, sizes)
            env.pop(var, None)
        else:
            _emit_call(st, env, ops, sizes)


def _emit_call(st: tuple, env: dict[str, float],
               ops: list[Op], sizes: dict[str, float]) -> None:
    _, name, args, line = st

    if name == 'alloc':
        if len(args) != 2:
            sys.exit("  error: alloc(name, size) takes 2 arguments")
        buf  = _operand_name(args[0])
        size = _eval_num(args[1], env)
        sizes[buf] = size
        return

    if name not in ENGINE_FOR:
        sys.exit(f"  error: unknown operation '{name}'\n"
                 f"  valid ops: alloc, gm_to_ub, ub_to_gm, vadd, "
                 f"gm_to_l1, l1_to_l0, matrixmul, l0_to_gm")

    if name == 'vadd':
        if len(args) != 3:
            sys.exit("  error: vadd(ub_dst, ub_src, scalar) takes 3 arguments")
        (dst, doff, dsl) = _eval_operand(args[0], env)
        (src, soff, ssl) = _eval_operand(args[1], env)
        scalar = _eval_num(args[2], env)
        if not dst.startswith('ub_') or not src.startswith('ub_'):
            print(f"  warning: vadd operands should be UB buffers (got dst={dst}, src={src})",
                  file=sys.stderr)
        ops.append(Op(len(ops), name, dst, src, ENGINE_FOR[name], scalar=scalar, line=line,
                      dst_off=doff, src_off=soff, dst_sliced=dsl, src_sliced=ssl))
        return

    if name == 'matrixmul':
        if len(args) != 3:
            sys.exit("  error: matrixmul(l0_dst, l0_src_a, l0_src_b) takes 3 arguments")
        (dst,  doff,  dsl)  = _eval_operand(args[0], env)
        (src,  soff,  ssl)  = _eval_operand(args[1], env)
        (src2, s2off, s2sl) = _eval_operand(args[2], env)
        if not dst.startswith('l0_') or not src.startswith('l0_') or not src2.startswith('l0_'):
            print(f"  warning: matrixmul operands should be L0 buffers "
                  f"(got dst={dst}, src={src}, src2={src2})", file=sys.stderr)
        ops.append(Op(len(ops), name, dst, src, ENGINE_FOR[name], src2=src2, line=line,
                      dst_off=doff, src_off=soff, src2_off=s2off,
                      dst_sliced=dsl, src_sliced=ssl, src2_sliced=s2sl))
        return

    # Two-operand transfer ops (gm_to_ub, ub_to_gm, gm_to_l1, l1_to_l0, l0_to_gm).
    if len(args) != 2:
        sys.exit(f"  error: {name}(dst, src) takes 2 arguments")
    (dst, doff, dsl) = _eval_operand(args[0], env)
    (src, soff, ssl) = _eval_operand(args[1], env)
    ops.append(Op(len(ops), name, dst, src, ENGINE_FOR[name], line=line,
                  dst_off=doff, src_off=soff, dst_sliced=dsl, src_sliced=ssl))


# Threshold above which we warn that an unrolled loop produced many ops (output stays
# correct, just verbose — Gantt/tables get one row per emitted op).
_UNROLL_WARN_OPS = 256


def emulate(prog: str) -> tuple[list[Op], dict[str, float]]:
    """Front-end entry point. Flat programs (no `for`, no `+` offset) take the original
    regex parse() path so existing behaviour is identical; loop/offset programs are
    tokenized, parsed into a statement AST, and executed into a flat instruction stream."""
    if not re.search(r'\bfor\b', prog) and '+' not in prog:
        return parse(prog)

    stmts = _parse_statements(_Stream(_tokenize(prog)), stop=None)
    ops: list[Op] = []
    sizes: dict[str, float] = {}
    _run_statements(stmts, {}, ops, sizes)

    if len(ops) > _UNROLL_WARN_OPS:
        print(f"  note: loop expansion produced {len(ops)} operations — output "
              f"(Gantt/tables) will be large.\n", file=sys.stderr)
    return ops, sizes


# ── Size / duration assignment ─────────────────────────────────────────────────

def assign_sizes(ops: list[Op], sizes: dict[str, float]) -> None:
    """Compute per-op transfer size (from alloc declarations) and cycle duration.

    A SLICED operand (e.g. `gm_1 + m*1KB`, emitted by a loop) moves a single tile, not
    the whole `gm_1` allocation — so its size is taken from the non-sliced operand (the
    tile actually written/read, e.g. the `ub_1` destination). For plain programs nothing
    is sliced and this collapses to the original src-first / dst-fallback rule."""
    missing: set[str] = set()
    for op in ops:
        # For matrixmul, use the average of both source sizes as the representative size
        if op.src2:
            size_src  = sizes.get(op.src,  sizes.get(op.dst, DEFAULT_SIZE_KB))
            size_src2 = sizes.get(op.src2, sizes.get(op.dst, DEFAULT_SIZE_KB))
            if op.src not in sizes and op.src2 not in sizes and op.dst not in sizes:
                missing.add(op.src)
            size = (size_src + size_src2) / 2.0
        else:
            # Prefer a non-sliced operand's full allocation (src first, then dst); a
            # sliced operand only spans a tile, so it never dictates the transfer size.
            candidates = [
                (op.src, op.src_sliced),
                (op.dst, op.dst_sliced),
            ]
            size = None
            for buf, sliced in candidates:
                if not sliced and buf in sizes:
                    size = sizes[buf]
                    break
            if size is None:
                # Fall back: any allocated operand (even sliced), else default.
                for buf, _ in candidates:
                    if buf in sizes:
                        size = sizes[buf]
                        break
            if size is None:
                missing.add(op.src)
                size = DEFAULT_SIZE_KB
        op.size_kb = size
        duration, eff_bw, util, regime = bandwidth_profile(op.engine, size)
        op.duration       = duration
        op.effective_bw   = eff_bw
        op.bw_utilization = util
        op.regime         = regime

    if missing:
        print(f"  warning: no alloc for {', '.join(sorted(missing))}"
              f" — using default {DEFAULT_SIZE_KB:.0f} KB\n", file=sys.stderr)

# ── Dependency / hazard analysis ──────────────────────────────────────────────
#
# Two operands ALIAS (touch overlapping memory) when they name the same base buffer
# AND their byte ranges overlap. A plain operand (`gm_1`) conservatively spans the whole
# buffer and so aliases every access to that base; two slices (`gm_1 + m*1KB`) alias only
# when their [offset, offset+tile) ranges overlap. For plain (non-loop) programs every
# operand is whole-buffer, so this reduces to base-name equality exactly as before.

def _operand(op: Op, role: str) -> tuple[str, float, bool, float]:
    """(base, offset_kb, sliced, span_kb) for op's 'dst'/'src'/'src2' operand."""
    if role == 'dst':
        return op.dst, op.dst_off, op.dst_sliced, op.size_kb
    if role == 'src':
        return op.src, op.src_off, op.src_sliced, op.size_kb
    return op.src2, op.src2_off, op.src2_sliced, op.size_kb


def _aliases(a: tuple[str, float, bool, float], b: tuple[str, float, bool, float]) -> bool:
    """Do two operands touch overlapping memory?"""
    (base_a, off_a, sliced_a, span_a) = a
    (base_b, off_b, sliced_b, span_b) = b
    if not base_a or not base_b or base_a != base_b:
        return False
    if not sliced_a or not sliced_b:
        return True                       # a whole-buffer access aliases anything
    # Both are tiles: overlap iff [off, off+span) ranges intersect.
    return off_a < off_b + span_b and off_b < off_a + span_a


def hazards(pred: Op, succ: Op) -> list[str]:
    """Return hazard types that force succ to wait for pred."""
    h: list[str] = []
    pred_dst = _operand(pred, 'dst')
    succ_dst = _operand(succ, 'dst')
    succ_reads = [_operand(succ, 'src')] + ([_operand(succ, 'src2')] if succ.src2 else [])
    pred_reads = [_operand(pred, 'src')] + ([_operand(pred, 'src2')] if pred.src2 else [])
    # succ reads what pred wrote (RAW): check both src and src2 of succ
    if any(_aliases(r, pred_dst) for r in succ_reads):
        h.append('RAW')
    # both write the same memory (WAW)
    if _aliases(succ_dst, pred_dst):
        h.append('WAW')
    # succ overwrites memory pred still reads (WAR): check pred's src and src2
    if any(_aliases(succ_dst, r) for r in pred_reads):
        h.append('WAR')
    return h

_HAZARD_DESC = {
    'RAW': lambda pred, succ, buf: f"{buf} written by op{pred.idx}, read by op{succ.idx}",
    'WAW': lambda pred, succ, buf: f"{buf} written by both op{pred.idx} and op{succ.idx}",
    'WAR': lambda pred, succ, buf: f"{buf} read by op{pred.idx}, overwritten by op{succ.idx}",
}

def hazard_details(pred: Op, succ: Op) -> list[tuple[str, str, str]]:
    """Return (hazard_type, buffer_name, human_reason) for each hazard between pred→succ."""
    details: list[tuple[str, str, str]] = []
    pred_dst = _operand(pred, 'dst')
    succ_dst = _operand(succ, 'dst')
    # RAW: succ reads what pred wrote
    for role in (['src', 'src2'] if succ.src2 else ['src']):
        if _aliases(_operand(succ, role), pred_dst):
            details.append(('RAW', pred.dst, _HAZARD_DESC['RAW'](pred, succ, pred.dst)))
    # WAW: both write the same memory
    if _aliases(succ_dst, pred_dst):
        details.append(('WAW', pred.dst, _HAZARD_DESC['WAW'](pred, succ, pred.dst)))
    # WAR: succ overwrites memory pred reads
    for role in (['src', 'src2'] if pred.src2 else ['src']):
        pr = _operand(pred, role)
        if _aliases(succ_dst, pr):
            details.append(('WAR', pr[0], _HAZARD_DESC['WAR'](pred, succ, pr[0])))
    return details

def build_deps(ops: list[Op]) -> Deps:
    deps: Deps = {i: set() for i in range(len(ops))}
    for i in range(len(ops)):
        for j in range(i):
            if hazards(ops[j], ops[i]):
                deps[i].add(j)
    return deps

# ── ASAP scheduling ───────────────────────────────────────────────────────────

def schedule(ops: list[Op], deps: Deps) -> None:
    """Assign start/end cycles using As-Soon-As-Possible scheduling."""
    done:   dict[int, float] = {}
    e_free: list[float]      = [0.0] * len(ENG_NAME)  # one slot per engine (ns)

    for i, op in enumerate(ops):
        earliest = e_free[op.engine]
        for j in deps[i]:
            earliest = max(earliest, done[j])
        op.start          = earliest
        op.end            = earliest + op.duration
        done[i]           = op.end
        e_free[op.engine] = op.end

# ── Critical-path analysis ────────────────────────────────────────────────────
#
# The ASAP scheduler serializes ops by BOTH data hazards (deps) AND engine
# availability (two ops on the same engine run back-to-back even with no data
# dependency). The "critical path" — the chain of ops whose end-to-end time equals
# the makespan — must therefore be computed over the FULL schedule DAG = data-hazard
# edges + same-engine consecutive-op edges. With both edge kinds, the longest
# weighted path length equals total_cycles exactly.

def critical_path_preds(ops: list[Op], deps: Deps) -> dict[int, set[int]]:
    """Predecessors in the full schedule DAG: data hazards + same-engine serialization.
    All edges run low-idx → high-idx, so op-index order is a valid topological order."""
    preds: dict[int, set[int]] = {i: set(deps[i]) for i in range(len(ops))}
    last_on_engine: dict[int, int] = {}
    for op in ops:                       # ops are in schedule order; ops[i].idx == i
        prev = last_on_engine.get(op.engine)
        if prev is not None:
            preds[op.idx].add(prev)
        last_on_engine[op.engine] = op.idx
    return preds


def critical_path_topo(ops: list[Op], deps: Deps) -> tuple[list[int], float]:
    """Longest weighted path via topological DP (op weight = duration).

    dist[i] reproduces ops[i].end exactly, so max(dist) == total_ns. Index order
    is a valid topological order, so a single forward pass suffices.
    Returns (path_as_op_indices, length_in_ns)."""
    preds  = critical_path_preds(ops, deps)
    n      = len(ops)
    dist   = [0.0] * n
    binder: list[int | None] = [None] * n
    for i in range(n):
        best, back = 0.0, None
        for p in preds[i]:
            if dist[p] > best:
                best, back = dist[p], p
        dist[i]   = best + ops[i].duration
        binder[i] = back
    sink = max(range(n), key=lambda i: dist[i])
    path: list[int] = []
    cur: int | None = sink
    while cur is not None:
        path.append(cur)
        cur = binder[cur]
    path.reverse()
    return path, dist[sink]


# Registry of selectable critical-path algorithms (extend as more are added).
CRITICAL_PATH_ALGOS = {
    'topo': critical_path_topo,
}
DEFAULT_CP_ALGO = 'topo'


def compute_critical_path(ops: list[Op], deps: Deps, algo: str) -> tuple[list[int], float]:
    """Run the named critical-path algorithm. Returns (path_of_op_indices, length_ns)."""
    return CRITICAL_PATH_ALGOS[algo](ops, deps)


def _cp_edge_reason(pred: Op, succ: Op) -> str:
    """Why succ is forced after pred on the critical path: data hazard(s) if any,
    otherwise same-engine serialization."""
    h = hazards(pred, succ)              # reuse existing hazard detection
    if h:
        buf = hazard_details(pred, succ)[0][1]
        return f"{'+'.join(h)} on '{buf}'"
    return f"engine serialization ({ENG_NAME[succ.engine]} reused)"


# ── ASCII rendering ───────────────────────────────────────────────────────────

BLOCK = '█'
IDLE  = '·'

GANTT_WIDTH = 90   # target chars for the Gantt time axis (auto-scaled to makespan)


def _gantt_scale(horizon_ns: float, width: int = GANTT_WIDTH) -> float:
    """chars-per-ns so the whole makespan fits in `width` columns (≥ a tiny floor
    so a zero/near-zero makespan still renders)."""
    return width / horizon_ns if horizon_ns > 0 else 1.0


def _col(t_ns: float, scale: float) -> int:
    return int(round(t_ns * scale))


def _gantt_row(eng_ops: list[Op], horizon_ns: float, scale: float) -> str:
    width = max(1, _col(horizon_ns, scale))
    row = [IDLE] * width
    for op in eng_ops:
        c0 = _col(op.start, scale)
        c1 = max(c0 + 1, _col(op.end, scale))   # every op occupies ≥1 column
        for t in range(c0, min(c1, width)):
            row[t] = BLOCK
        label = f"op{op.idx}"
        for k, c in enumerate(label):
            pos = c0 + k
            if c0 <= pos < min(c1, width):
                row[pos] = c
    return ''.join(row)

def _fmt_size(kb: float) -> str:
    if kb >= 1024:
        return f"{kb / 1024:.3g} MB"
    return f"{kb:.3g} KB"

def _fmt_ns(ns: float) -> str:
    """Format a nanosecond duration/time compactly."""
    if ns >= 1000:
        return f"{ns / 1000:.3g} µs"
    return f"{ns:.3g} ns"

def _fmt_operand(base: str, off: float, sliced: bool) -> str:
    """Render an operand, appending its tile offset when it's a slice (gm_1+37KB)."""
    if sliced and off:
        return f"{base}+{_fmt_size(off)}".replace(' ', '')
    if sliced:
        return f"{base}+0KB"
    return base

def _op_sig(op: Op) -> str:
    """Return the canonical instruction string for an op."""
    dst = _fmt_operand(op.dst, op.dst_off, op.dst_sliced)
    src = _fmt_operand(op.src, op.src_off, op.src_sliced)
    if op.src2:
        src2 = _fmt_operand(op.src2, op.src2_off, op.src2_sliced)
        return f"{op.name}({dst}, {src}, {src2})"
    if op.name == 'vadd':
        return f"{op.name}({dst}, {src}, {op.scalar:g})"
    return f"{op.name}({dst}, {src})"


def render(ops: list[Op], deps: Deps, cp_algo: str | None = None) -> None:
    H     = max(op.end for op in ops)
    scale = _gantt_scale(H)
    W     = max(1, _col(H, scale)) + 1
    PAD   = '  '

    print()
    print(PAD + '┌─ Pipeline Execution Graph ' + '─' * max(0, W - 20) + '┐')
    print()

    # Bandwidth legend
    print(PAD + 'Engine bandwidth ramps with transfer size (small transfers run below peak);')
    print(PAD + 'duration = size ÷ bandwidth(size), reported in nanoseconds:')
    for eng in range(len(ENG_NAME)):
        curve = BANDWIDTH_CURVE_GB_S[eng]
        peak  = peak_bandwidth_gb_s(eng)
        label = f"{ENG_NAME[eng]:10}"
        if len(curve) == 1:
            print(PAD + f"  {label}  {peak:6.0f} GB/s  (compute, size-independent)")
        else:
            pts = ', '.join(f"{_fmt_size(s)}→{bw:.0f} GB/s" for s, bw in curve)
            print(PAD + f"  {label}  {peak:6.0f} GB/s peak  (ramp: {pts})")
    print()

    # Time ruler (nanoseconds). Tick every ~10 columns.
    print(PAD + f'Time axis: {W - 1} cols ≈ {_fmt_ns(H)} makespan '
                f'({1.0 / scale:.3g} ns/col)')
    print(PAD + '         ' + '─' * W)

    for eng in range(len(ENG_NAME)):
        row = _gantt_row([op for op in ops if op.engine == eng], H, scale)
        print(f"{PAD}  {ENG_NAME[eng]:10} │ {row}")

    print(PAD + '         ' + '─' * W)
    print()

    # Operation table
    cw = [4, 30, 8, 10, 16, 10]
    print(PAD + f"{'Op':<{cw[0]}} {'Instruction':<{cw[1]}} {'Engine':<{cw[2]}} {'Size':<{cw[3]}} {'Time (ns)':<{cw[4]}} {'BW util':<{cw[5]}} Waits for (hazard)")
    print(PAD + '─' * 110)

    for op in ops:
        sig  = _op_sig(op)
        size = _fmt_size(op.size_kb)
        cyc  = f"[{op.start:.1f}..{op.end:.1f}]"
        util = f"{op.bw_utilization:.0%}"
        if deps[op.idx]:
            ws_parts = [
                f"op{j}({'+'.join(hazards(ops[j], op))})"
                for j in sorted(deps[op.idx])
            ]
            ws = ', '.join(ws_parts)
        else:
            ws = '—'
        print(PAD + f"{op.idx:<{cw[0]}} {sig:<{cw[1]}} {ENG_NAME[op.engine]:<{cw[2]}} {size:<{cw[3]}} {cyc:<{cw[4]}} {util:<{cw[5]}} {ws}")

    print()

    # Per-op time breakdown (duration / total_ns, sorted biggest first)
    print(PAD + 'Time breakdown (op duration ÷ total time, sorted; ops overlap so the')
    print(PAD + 'percentages need not sum to 100%):')
    for op in sorted(ops, key=lambda o: o.duration, reverse=True):
        pct = op.duration / H if H else 0.0
        bar = ('█' * int(round(pct * 20))).ljust(20)
        print(PAD + f"  op{op.idx:<2} {_op_sig(op):<30} [{bar}] {pct:6.1%}  "
                    f"({op.duration:.1f}/{H:.1f} ns)")
    print()

    # Engine utilization
    print(PAD + 'Engine utilization:')
    for eng in range(len(ENG_NAME)):
        busy   = sum(op.end - op.start for op in ops if op.engine == eng)
        pct    = int(100 * busy / H) if H else 0
        bar    = '█' * pct + '░' * (100 - pct)
        bar_sm = bar[:20]
        print(PAD + f"  {ENG_NAME[eng]:10}  [{bar_sm}] {pct:3}%  ({busy:.1f}/{H:.1f} ns busy)")
    print()

    # Bandwidth utilization (per transfer/compute op: effective / peak)
    print(PAD + 'Bandwidth utilization (effective ÷ peak):')
    for op in ops:
        peak = peak_bandwidth_gb_s(op.engine)
        pct  = int(round(100 * op.bw_utilization))
        bar  = ('█' * (pct // 5)).ljust(20)
        print(PAD + f"  op{op.idx:<2} {ENG_NAME[op.engine]:8} [{bar}] {pct:3}%  "
                    f"({op.effective_bw:.4g}/{peak:.0f} GB/s, {op.regime})")
    print()

    # Parallelism summary
    pairs = [
        (i, j)
        for i in range(len(ops))
        for j in range(i + 1, len(ops))
        if ops[i].start < ops[j].end and ops[j].start < ops[i].end
    ]
    if pairs:
        print(PAD + f'Parallel overlap ({len(pairs)} pair(s)):')
        for i, j in pairs:
            s = max(ops[i].start, ops[j].start)
            e = min(ops[i].end,   ops[j].end)
            print(PAD + f"  op{i} ∥ op{j}  (overlap {s:.1f}..{e:.1f} ns)")
    else:
        print(PAD + 'Execution is fully sequential — no parallel overlap.')

    # Critical path (optional; longest weighted chain through the full schedule DAG)
    if cp_algo:
        path, length = compute_critical_path(ops, deps, cp_algo)
        frac = length / H if H else 0.0
        print()
        print(PAD + f"Critical path  (algorithm: {cp_algo}, length = {length:.1f} ns "
                    f"= {frac:.0%} of makespan):")
        chain = '  '.join(
            (f"op{path[k]}" if k == 0
             else f"─({_cp_edge_reason(ops[path[k-1]], ops[path[k]])})→  op{path[k]}")
            for k in range(len(path))
        )
        print(PAD + '    ' + chain)
        for idx in path:
            op = ops[idx]
            print(PAD + f"    op{op.idx:<2} {_op_sig(op):<30} {ENG_NAME[op.engine]:<8} "
                        f"[{op.start:.1f}..{op.end:.1f}]  dur={op.duration:.1f} ns")
        print(PAD + f"    covers {len(path)}/{len(ops)} ops; "
                    f"{length:.1f}/{H:.1f} ns on the critical chain.")

    print()
    print(PAD + '└' + '─' * (W + 10) + '┘')

    print()

# ── LLM output mode ───────────────────────────────────────────────────────────

def render_llm(ops: list[Op], deps: Deps, cp_algo: str | None = None) -> None:
    total_ns = max(op.end for op in ops)

    # Parallel pairs (same definition as human renderer)
    parallel_pairs = [
        (i, j)
        for i in range(len(ops))
        for j in range(i + 1, len(ops))
        if ops[i].start < ops[j].end and ops[j].start < ops[i].end
    ]

    print("=== EXECUTION SUMMARY ===")
    print(f"total_ns: {total_ns:.2f}")
    print(f"num_ops: {len(ops)}")
    print(f"execution_mode: {'parallel' if parallel_pairs else 'sequential'}")
    print()

    print("=== TIME BREAKDOWN ===")
    print("(time_ratio = op duration / total_ns, sorted biggest first; "
          "ops overlap so ratios need not sum to 100%)")
    for op in sorted(ops, key=lambda o: o.duration, reverse=True):
        time_ratio = op.duration / total_ns if total_ns else 0.0
        print(f"op{op.idx}: {_op_sig(op)}  duration_ns={op.duration:.2f}  "
              f"time_ratio={time_ratio:.2%}  ({op.duration:.2f}/{total_ns:.2f} ns)")
    print()

    print("=== PER-OP STATISTICS ===")
    for op in ops:
        sig        = _op_sig(op)
        duration   = op.end - op.start
        time_ratio = duration / total_ns if total_ns else 0.0
        print(f"op{op.idx}: {sig}")
        print(f"  engine: {ENG_NAME[op.engine]}")
        print(f"  size: {_fmt_size(op.size_kb)}")
        print(f"  cycles_ns: [{op.start:.2f}..{op.end:.2f}]  duration_ns={duration:.2f}  time_ratio={time_ratio:.2%}")
        print(f"  bandwidth: effective={op.effective_bw:.4g} GB/s  "
              f"peak={peak_bandwidth_gb_s(op.engine):.0f} GB/s  "
              f"utilization={op.bw_utilization:.2%}  regime={op.regime}")
        print(f"  wait_ns_before_start: {op.start:.2f}")
        if deps[op.idx]:
            for j in sorted(deps[op.idx]):
                for htype, buf, reason in hazard_details(ops[j], op):
                    avoidable = (htype == 'WAR' and
                                 all(h == 'WAR' for h in hazards(ops[j], op)))
                    print(f"  blocked_by: op{j} via {htype} on '{buf}' — {reason}")
                    if avoidable:
                        print(f"  fix: allocate a new destination buffer instead of "
                              f"reusing '{buf}' to remove this WAR dependency")
        else:
            print(f"  blocked_by: none")
        print()

    print("=== ENGINE UTILIZATION ===")
    for eng in range(len(ENG_NAME)):
        busy = sum(op.end - op.start for op in ops if op.engine == eng)
        pct  = busy / total_ns if total_ns else 0
        print(f"{ENG_NAME[eng]}: busy={busy:.2f}/{total_ns:.2f} ns  utilization={pct:.2%}")
    print()

    print("=== BANDWIDTH UTILIZATION ===")
    print("(effective_bw / peak_bw per op; bandwidth ramps with size, so small "
          "transfers run below peak)")
    for op in ops:
        peak  = peak_bandwidth_gb_s(op.engine)
        curve = BANDWIDTH_CURVE_GB_S[op.engine]
        if len(curve) == 1:
            ramp_s = ""
        else:
            sat_size = curve[-1][0]
            ramp_s = f"  saturates_at={_fmt_size(sat_size)}"
        print(f"op{op.idx} ({ENG_NAME[op.engine]}): "
              f"effective={op.effective_bw:.4g} GB/s  peak={peak:.0f} GB/s  "
              f"utilization={op.bw_utilization:.2%}  regime={op.regime}{ramp_s}")
    print()

    print("=== PARALLELISM ===")
    if parallel_pairs:
        print(f"parallel_pairs: {len(parallel_pairs)}")
        for i, j in parallel_pairs:
            s = max(ops[i].start, ops[j].start)
            e = min(ops[i].end,   ops[j].end)
            print(f"  op{i} || op{j}: overlap=[{s:.2f}..{e:.2f}]  overlap_ns={e - s:.2f}")
    else:
        print("parallel_pairs: 0")
        print("root_cause_of_sequential_execution:")
        for i, op in enumerate(ops):
            if deps[i]:
                for j in sorted(deps[i]):
                    for htype, buf, reason in hazard_details(ops[j], op):
                        print(f"  op{j}->op{i}: {htype} on '{buf}' — {reason}")
    print()

    if cp_algo:
        path, length = compute_critical_path(ops, deps, cp_algo)
        frac = length / total_ns if total_ns else 0.0
        print("=== CRITICAL PATH ===")
        print(f"algorithm: {cp_algo}")
        print(f"length_ns: {length:.2f}")
        print(f"fraction_of_makespan: {frac:.0%}")
        print(f"path: {' -> '.join(f'op{i}' for i in path)}")
        print("edges:")
        for k in range(1, len(path)):
            pred, succ = ops[path[k - 1]], ops[path[k]]
            print(f"  op{path[k-1]} -> op{path[k]}: {_cp_edge_reason(pred, succ)}")
        print("per_op:")
        for idx in path:
            op = ops[idx]
            print(f"  op{op.idx} {_op_sig(op)}  engine={ENG_NAME[op.engine]}  "
                  f"ns=[{op.start:.2f}..{op.end:.2f}]  duration_ns={op.duration:.2f}")
        print()

# ── Memory-capacity verification ──────────────────────────────────────────────
#
# Correctness check: at every cycle, the buffers simultaneously LIVE in a memory
# region must fit in that region's physical capacity (MEMORY_CAPACITY_KB). A buffer
# is live from the first cycle any op touches it (read or write) until the last op
# that touches it finishes — a single static allocation spanning its whole use, which
# is the conservative assumption for a model with no explicit free(). Overlapping live
# ranges in the same region accumulate; if the sum ever exceeds capacity, the program
# is incorrect on this hardware and we trace the overflow back to its source ops.

@dataclass
class BufferLife:
    name:      str
    region:    str
    size_kb:   float
    start:     float        # first ns the buffer is live
    end:       float        # ns the buffer is freed (exclusive)
    producers: list[int]    # op indices that write this buffer
    consumers: list[int]    # op indices that read this buffer


def compute_liveness(ops: list[Op], sizes: dict[str, float]) -> dict[str, BufferLife]:
    """Live interval [start, end) and producer/consumer ops for every buffer used."""
    lives: dict[str, BufferLife] = {}

    def touch(buf: str, op: Op, is_write: bool) -> None:
        if not buf:
            return
        region = region_of(buf)
        if region is None:
            return
        size = sizes.get(buf, op.size_kb if op.size_kb else DEFAULT_SIZE_KB)
        bl = lives.get(buf)
        if bl is None:
            bl = BufferLife(buf, region, size, op.start, op.end, [], [])
            lives[buf] = bl
        bl.start = min(bl.start, op.start)
        bl.end   = max(bl.end,   op.end)
        if is_write:
            bl.producers.append(op.idx)
        else:
            bl.consumers.append(op.idx)

    for op in ops:
        touch(op.dst, op, is_write=True)
        touch(op.src, op, is_write=False)
        touch(op.src2, op, is_write=False)
    return lives


def verify_memory(ops: list[Op], sizes: dict[str, float]) -> dict:
    """
    Walk the schedule and, at every event cycle, check each region's live footprint
    against its capacity. Returns a structured report:
      {
        'regions':    {region: {capacity, peak_kb, peak_cycle, peak_buffers}},
        'violations': [ {region, cycle, used_kb, capacity_kb, over_kb,
                         live_buffers, trigger_ops} ],
        'ok':         bool,
      }
    """
    lives = compute_liveness(ops, sizes)

    # Candidate cycles to inspect: every op start (a region's footprint can only grow
    # at a buffer's first-touch, which always coincides with some op's start cycle).
    event_cycles = sorted({op.start for op in ops})

    def live_at(region: str, cycle: float) -> list[BufferLife]:
        return [bl for bl in lives.values()
                if bl.region == region and bl.start <= cycle < bl.end]

    regions_report: dict[str, dict] = {}
    violations: list[dict] = []

    for region, capacity in MEMORY_CAPACITY_KB.items():
        peak_kb, peak_cycle, peak_bufs = 0.0, 0.0, []
        for cycle in event_cycles:
            live = live_at(region, cycle)
            used = sum(bl.size_kb for bl in live)
            if used > peak_kb:
                peak_kb, peak_cycle, peak_bufs = used, cycle, live
            if capacity is not None and used > capacity:
                # Ops that became live exactly at this cycle pushed us over.
                trigger = sorted(
                    {p for bl in live if bl.start == cycle for p in bl.producers},
                )
                violations.append({
                    'region':       region,
                    'cycle':        cycle,
                    'used_kb':      used,
                    'capacity_kb':  capacity,
                    'over_kb':      used - capacity,
                    'live_buffers': sorted(live, key=lambda b: b.size_kb, reverse=True),
                    'trigger_ops':  trigger,
                })
        regions_report[region] = {
            'capacity_kb':  capacity,
            'peak_kb':      peak_kb,
            'peak_cycle':   peak_cycle,
            'peak_buffers': peak_bufs,
        }

    # Keep only the first violation per (region, cycle) so a sustained overflow isn't
    # reported once per inspected cycle.
    seen: set[tuple[str, int]] = set()
    deduped: list[dict] = []
    for v in violations:
        key = (v['region'], v['cycle'])
        if key not in seen:
            seen.add(key)
            deduped.append(v)

    return {
        'regions':    regions_report,
        'violations': deduped,
        'ok':         not deduped,
    }


def render_verify(ops: list[Op], sizes: dict[str, float]) -> None:
    report = verify_memory(ops, sizes)
    PAD = '  '

    print()
    print(PAD + '┌─ Memory-Capacity Verification ' + '─' * 30 + '┐')
    print()
    print(PAD + 'Region capacities (a buffer is live from first touch to last use; live')
    print(PAD + 'buffers in a region must fit its capacity at every moment):')
    for region, cap in MEMORY_CAPACITY_KB.items():
        cap_s = 'unbounded' if cap is None else _fmt_size(cap)
        rep   = report['regions'][region]
        peak  = _fmt_size(rep['peak_kb'])
        print(PAD + f"  {region:3}  capacity={cap_s:>10}   "
                    f"peak_usage={peak:>10} @ {rep['peak_cycle']:.1f} ns")
    print()

    if report['ok']:
        print(PAD + '✔ PASS — every region stays within capacity at all times.')
        print()
        # Show the tightest region (highest peak/capacity ratio) for context.
        bounded = [(r, d) for r, d in report['regions'].items()
                   if d['capacity_kb']]
        if bounded:
            tight_r, tight = max(
                bounded, key=lambda kv: kv[1]['peak_kb'] / kv[1]['capacity_kb'])
            frac = tight['peak_kb'] / tight['capacity_kb']
            print(PAD + f"  tightest region: {tight_r} at {frac:.0%} of capacity "
                        f"({_fmt_size(tight['peak_kb'])} / "
                        f"{_fmt_size(tight['capacity_kb'])}).")
        print()
        print(PAD + '└' + '─' * 61 + '┘')
        print()
        return

    print(PAD + f"✗ FAIL — {len(report['violations'])} capacity violation(s) detected.")
    print()

    for n, v in enumerate(report['violations'], 1):
        print(PAD + f"Violation #{n}: region {v['region']} overflows at {v['cycle']:.1f} ns")
        print(PAD + f"  used {_fmt_size(v['used_kb'])} of {_fmt_size(v['capacity_kb'])} "
                    f"capacity  (over by {_fmt_size(v['over_kb'])})")
        print()
        print(PAD + f"  Live {v['region']} buffers at {v['cycle']:.1f} ns "
                    f"(largest first):")
        for bl in v['live_buffers']:
            prod = (', '.join(f"op{p}" for p in bl.producers)
                    if bl.producers else 'input (no producer)')
            print(PAD + f"    {bl.name:10} {_fmt_size(bl.size_kb):>10}  "
                        f"live [{bl.start:.1f}..{bl.end:.1f}]  written by {prod}")
        print()

        # Root cause: the op(s) whose buffer went live at this moment tipped the region
        # over. Report each with its source line and a concrete reason.
        print(PAD + "  Root cause (op(s) that committed memory at this moment):")
        if v['trigger_ops']:
            for p in v['trigger_ops']:
                op = ops[p]
                print(PAD + f"    line {op.line}: op{op.idx}  {_op_sig(op)}")
                print(PAD + f"      writes '{op.dst}' ({_fmt_size(op.size_kb)}) into "
                            f"{v['region']}, but the region already holds other live "
                            f"buffers — total exceeds capacity.")
        else:
            # Overflow sustained from earlier cycles by inputs with no producer op.
            print(PAD + "    (overflow caused by input buffers with no producer op; "
                        "reduce their declared sizes or stage them in smaller tiles)")
        print()

        # Fix suggestions.
        print(PAD + "  How to fix:")
        print(PAD + f"    • Reduce the working set in {v['region']}: free/reuse buffers "
                    f"sooner, or serialize")
        print(PAD + "      independent tiles so their live ranges don't overlap.")
        print(PAD + f"    • Shrink the buffers (smaller tiles) so the simultaneous "
                    f"footprint fits {_fmt_size(v['capacity_kb'])}.")
        print(PAD + f"    • Or raise {v['region']} capacity in MEMORY_CAPACITY_KB if the "
                    f"hardware actually has more.")
        print()

    print(PAD + '└' + '─' * 61 + '┘')
    print()


# ── NetworkX graph builder ────────────────────────────────────────────────────
def _simulate_speedup(ops: list[Op], deps: Deps, target_idx: int, saved_ns: float) -> float:
    """
    Re-run ASAP scheduling with op[target_idx].duration reduced by saved_ns nanoseconds.
    Returns the new total_ns.
    """
    import copy
    patched = copy.deepcopy(ops)
    patched[target_idx].duration = max(0.0, patched[target_idx].duration - saved_ns)
    schedule(patched, deps)
    return float(max(op.end for op in patched))


def build_graph(ops: list[Op], deps: Deps) -> nx.DiGraph:
    total_ns = max(op.end for op in ops)

    parallel_pairs = [
        (i, j)
        for i in range(len(ops))
        for j in range(i + 1, len(ops))
        if ops[i].start < ops[j].end and ops[j].start < ops[i].end
    ]

    G = nx.DiGraph()

    # ── Graph-level metadata ──────────────────────────────────────────────────
    G.graph['total_ns']        = total_ns
    G.graph['num_ops']         = len(ops)
    G.graph['execution_mode']  = 'parallel' if parallel_pairs else 'sequential'
    G.graph['parallel_pairs']  = [(i, j) for i, j in parallel_pairs]
    G.graph['engine_bandwidth'] = {
        ENG_NAME[e]: {
            'peak_gb_s':        peak_bandwidth_gb_s(e),
            'curve_gb_s':       [list(pt) for pt in BANDWIDTH_CURVE_GB_S[e]],
            'size_dependent':   len(BANDWIDTH_CURVE_GB_S[e]) > 1,
            'saturates_at_kb':  BANDWIDTH_CURVE_GB_S[e][-1][0],
        }
        for e in range(len(ENG_NAME))
    }

    eng_util = {}
    for eng in range(len(ENG_NAME)):
        busy = sum(op.end - op.start for op in ops if op.engine == eng)
        eng_util[ENG_NAME[eng]] = {
            'busy_ns':       round(busy, 4),
            'total_ns':      round(total_ns, 4),
            'utilization':   round(busy / total_ns, 4) if total_ns else 0.0,
        }
    G.graph['engine_utilization'] = eng_util

    # ── Op nodes ─────────────────────────────────────────────────────────────
    for op in ops:
        duration   = op.end - op.start
        time_ratio = round(duration / total_ns, 4) if total_ns else 0.0
        # wait = gap between when all blockers finished and when this op started
        # (0 if no deps or op started immediately after last blocker)
        if deps[op.idx]:
            blocker_done = max(ops[j].end for j in deps[op.idx])
            wait_ns      = op.start - blocker_done
        else:
            wait_ns      = op.start   # idle time before first op on this engine

        # Pre-compute speedup table: what if this op were 10%/25%/50% faster?
        speedup_table = {}
        for pct in (10, 25, 50):
            saved   = duration * pct / 100.0
            new_tot = _simulate_speedup(ops, deps, op.idx, saved)
            speedup = round(total_ns / new_tot, 4) if new_tot else 1.0
            speedup_table[f'reduce_{pct}pct'] = {
                'saved_ns':      round(saved, 4),
                'new_total_ns':  round(new_tot, 4),
                'speedup_ratio': speedup,
            }

        G.add_node(
            f'op{op.idx}',
            idx          = op.idx,
            instruction  = _op_sig(op),
            op_name      = op.name,
            dst          = op.dst,
            src          = op.src,
            engine       = op.engine,
            engine_name  = ENG_NAME[op.engine],
            size_kb      = op.size_kb,
            size_human   = _fmt_size(op.size_kb),
            duration_ns  = round(duration, 4),
            start_ns     = round(op.start, 4),
            end_ns       = round(op.end, 4),
            time_ratio   = time_ratio,
            wait_ns      = round(wait_ns, 4),
            peak_bw_gb_s     = peak_bandwidth_gb_s(op.engine),
            effective_bw_gb_s = round(op.effective_bw, 4),
            bw_utilization   = round(op.bw_utilization, 4),
            bw_regime        = op.regime,
            speedup_if_reduced = speedup_table,
        )

    # ── Dependency edges ──────────────────────────────────────────────────────
    for i, op in enumerate(ops):
        for j in sorted(deps[i]):
            details   = hazard_details(ops[j], op)
            htypes    = [d[0] for d in details]
            buf       = details[0][1] if details else None
            avoidable = 'WAR' in htypes and 'RAW' not in htypes and 'WAW' not in htypes
            fix       = (f"allocate a new destination buffer instead of reusing '{buf}' "
                         f"in op{i} to remove the WAR dependency and allow parallel execution"
                         ) if avoidable else None
            G.add_edge(
                f'op{j}', f'op{i}',
                hazard_types  = htypes,
                buffer        = buf,
                reasons       = [d[2] for d in details],
                delay_ns      = round(op.start - ops[j].end, 4),
                avoidable     = avoidable,
                fix_suggestion= fix,
            )

    # ── Pre-computed query answers ────────────────────────────────────────────
    bottleneck_node = max(G.nodes, key=lambda n: G.nodes[n]['time_ratio'])
    bn              = G.nodes[bottleneck_node]
    blockers        = [
        {
            'blocker_op':     u,
            'hazard_types':   G[u][bottleneck_node]['hazard_types'],
            'buffer':         G[u][bottleneck_node]['buffer'],
            'reasons':        G[u][bottleneck_node]['reasons'],
            'avoidable':      G[u][bottleneck_node]['avoidable'],
            'fix_suggestion': G[u][bottleneck_node]['fix_suggestion'],
        }
        for u in G.predecessors(bottleneck_node)
    ]

    G.graph['bottleneck'] = {
        'op':              bottleneck_node,
        'instruction':     bn['instruction'],
        'duration_ns':     bn['duration_ns'],
        'time_ratio':      bn['time_ratio'],
        'blocked_by':      blockers,
        'speedup_if_reduced': bn['speedup_if_reduced'],
    }

    # ── Bandwidth-utilization summary ─────────────────────────────────────────
    # Per-op effective/peak, plus the op leaving the most bandwidth on the table.
    bw_per_op = {
        f'op{op.idx}': {
            'effective_bw_gb_s': round(op.effective_bw, 4),
            'peak_bw_gb_s':      peak_bandwidth_gb_s(op.engine),
            'bw_utilization':    round(op.bw_utilization, 4),
            'regime':            op.regime,
        }
        for op in ops
    }
    worst = min(ops, key=lambda o: o.bw_utilization)
    below_peak = worst.regime in ('floor', 'ramp')
    G.graph['bandwidth_summary'] = {
        'per_op': bw_per_op,
        'lowest_utilization_op': {
            'op':             f'op{worst.idx}',
            'instruction':    _op_sig(worst),
            'bw_utilization': round(worst.bw_utilization, 4),
            'regime':         worst.regime,
            'hint': ('this transfer runs below peak — its size is on the bandwidth ramp '
                     '(below saturation), so a larger transfer would use bandwidth more '
                     'efficiently'
                     if below_peak else
                     'already at saturated (peak) bandwidth'),
        },
    }

    return G


def render_nx(ops: list[Op], deps: Deps) -> None:
    G = build_graph(ops, deps)

    # Serialize to node-link JSON (standard networkx interchange format)
    data = nx.node_link_data(G, edges='links')
    print(json.dumps(data, indent=2))

# ── Top-level ─────────────────────────────────────────────────────────────────

def simulate(program: str, llm_mode: bool = False, nx_mode: bool = False,
             cp_algo: str | None = None, verify_mode: bool = False) -> None:
    ops, sizes = emulate(program)
    if not ops:
        print("  (no operations found in input)")
        return
    assign_sizes(ops, sizes)
    deps = build_deps(ops)
    schedule(ops, deps)
    if verify_mode:
        render_verify(ops, sizes)
    elif nx_mode:
        render_nx(ops, deps)
    elif llm_mode:
        render_llm(ops, deps, cp_algo)
    else:
        render(ops, deps, cp_algo)

# ── Entry point ───────────────────────────────────────────────────────────────

_BANNER = """\

  GM/UB/L1/L0 Pipeline Simulator
  ─────────────────────────────────
  Declare buffer sizes anywhere in the program:
    alloc(name, size)          e.g.  alloc(gm_1, 256KB)  alloc(l1_1, 512KB)
    Supported units: B, KB, MB, GB  (bare number → KB)

  Operations (argument order is always dst, src):
    gm_to_ub(ub_dst, gm_src)              transfer GM → UB      [GM→UB engine,   1500 GB/s peak]
    ub_to_gm(gm_dst, ub_src)              transfer UB → GM      [UB→GM engine,    750 GB/s peak]
    vadd(ub_dst, ub_src, scalar)           SIMD vector+scalar    [VecUnit,    2000–16000 GB/s (size-dependent)]
    gm_to_l1(l1_dst, gm_src)              transfer GM → L1      [GM→L1 engine,    750 GB/s peak]
    l1_to_l0(l0_dst, l1_src)              transfer L1 → L0      [L1→L0 engine,   2000 GB/s peak]
    matrixmul(l0_dst, l0_src_a, l0_src_b) matrix multiply        [CubeUnit,       3000 GB/s]
    l0_to_gm(gm_dst, l0_src)              write-back L0 → GM    [L0→GM engine,    750 GB/s peak]

  Loops (an emulator front-end expands these into the concrete instruction stream the
  hardware runs; the schedule is identical to the unrolled program, not an estimate):
    for VAR in range(STOP) { BODY }              range(0, STOP, 1)
    for VAR in range(START, STOP) { BODY }       step defaults to 1
    for VAR in range(START, STOP, STEP) { BODY }
    BODY may hold several statements (alloc/ops) and nest further for-loops.
    Operands may carry an address offset built from the loop var(s) and size literals:
      gm_to_ub(ub_1, gm_1 + m * 1KB)    a 1KB tile of gm_1 at offset m·1KB
    Tile-aware hazards: two slices of the same buffer only conflict when their byte
    ranges overlap, so disjoint tiles run in parallel; reusing one destination still
    serializes. Capacity (--verify) still charges each buffer's whole allocation.

  Size-dependent bandwidth (realistic latency model): a transfer engine's achieved
  bandwidth RAMPS UP with transfer size and saturates at peak. It's given as
  piecewise-linear (size_kb → GB/s) breakpoints; duration is reported in nanoseconds.
    size ≤ first breakpoint  → floor:      clamped to the lowest bandwidth
    between breakpoints       → ramp:       linear interpolation of bandwidth
    size ≥ last breakpoint    → saturated:  clamped to peak bandwidth
    duration_ns = size_kb × 1024 / bandwidth(size)   (GB = 1e9 B, KB = 1024 B)
    GM→UB (measured): 1 KB → 100 GB/s, ramping to 1500 GB/s at 12 KB, saturated above.
    VecUnit (measured): an fp16 vadd ramps 1 → 8 TFLOPS over 1 → 24 KB; with a constant
      2 bytes/FLOP that is 2000 → 16000 GB/s (TFLOPS × 2000), so it's an ordinary
      size-dependent curve. CubeUnit (matrixmul) is still flat/size-independent (no
      measured numbers yet).
    bandwidth_utilization = effective_bw / peak  (reported per op in all output modes).
    Buffers without alloc default to 64 KB.

  Hazards detected: RAW (read-after-write), WAW (write-after-write),
                    WAR (write-after-read)

  Output modes:
    (default)   human-readable ASCII Gantt chart + table
    --llm       compact structured text for LLM consumption
    --nx        NetworkX node-link JSON (queryable graph with speedup)
    --verify    memory-capacity correctness check (UB/L1/L0 footprint vs capacity;
                reports violations with source line + root cause)

  Critical path (augments the default and --llm modes; not added to --nx):
    --critical-path[=algo]   longest weighted op-chain through the full schedule
                             DAG (data hazards + same-engine serialization); its
                             length equals the makespan. algo defaults to 'topo'.

  Examples:
    alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) ub_to_gm(gm_2, ub_1)
    alloc(gm_1, 256KB) alloc(ub_1, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)
    alloc(gm_1, 256KB) alloc(gm_2, 256KB) gm_to_l1(l1_1, gm_1) gm_to_l1(l1_2, gm_2) l1_to_l0(l0_1, l1_1) l1_to_l0(l0_2, l1_2) matrixmul(l0_3, l0_1, l0_2) l0_to_gm(gm_3, l0_3)
    --llm alloc(ub_1, 128KB) vadd(ub_2, ub_1, 2.0)
    --nx  alloc(l0_1, 64KB) alloc(l0_2, 64KB) matrixmul(l0_3, l0_1, l0_2)
    for m in range(0, 8, 1) { alloc(gm_1, 64KB) alloc(ub_1, 8KB) gm_to_ub(ub_1, gm_1 + m*8KB) }

  Type Ctrl-D to quit.
"""

if __name__ == '__main__':
    def parse_cp_flag(tokens: list[str]) -> tuple[str | None, list[str]]:
        """Pull --critical-path / --critical-path=NAME out of tokens.
        Returns (cp_algo, remaining_tokens). cp_algo is None if the flag is absent,
        else the chosen algorithm name (DEFAULT_CP_ALGO for the bare flag)."""
        cp_algo: str | None = None
        rest: list[str] = []
        for t in tokens:
            if t == '--critical-path':
                cp_algo = DEFAULT_CP_ALGO
            elif t.startswith('--critical-path='):
                name = t.split('=', 1)[1]
                if name not in CRITICAL_PATH_ALGOS:
                    sys.exit(f"  error: unknown critical-path algorithm '{name}'\n"
                             f"  valid algorithms: {', '.join(sorted(CRITICAL_PATH_ALGOS))}")
                cp_algo = name
            else:
                rest.append(t)
        return cp_algo, rest

    args     = sys.argv[1:]
    llm_mode = '--llm'    in args
    nx_mode  = '--nx'     in args
    verify_mode = '--verify' in args
    cp_algo, args = parse_cp_flag(
        [a for a in args if a not in ('--llm', '--nx', '--verify')])

    if args:
        simulate(' '.join(args), llm_mode, nx_mode, cp_algo, verify_mode)
    else:
        print(_BANNER)
        while True:
            try:
                line = input('  >> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line:
                toks = line.split()
                llm = '--llm'    in toks
                nxm = '--nx'     in toks
                ver = '--verify' in toks
                cp_algo, prog_toks = parse_cp_flag(
                    [t for t in toks if t not in ('--llm', '--nx', '--verify')])
                simulate(' '.join(prog_toks), llm, nxm, cp_algo, ver)
