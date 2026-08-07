#!/usr/bin/env python3
"""前置 BLOCK 扫描 (v2) — 全面覆盖: 生成**所有** L0 合法候选, 单进程 torch.npu.Event 实测, 每 config 5 次取平均.

★v2 改进:
  ① 候选: 程序化枚举所有 L0 合法 (BM,BN,BK) 组合 (非手写 8 个), 全面覆盖
  ② 单进程: 利用 triton JIT 同进程不同 constexpr 自动编译, 像 @triton.autotune 一样运行
  ③ 一次 msprof(可选): msprof 包裹整个 runner → 一次分析捕获所有 config 的 kernel 执行
  ④ 每 config: 预热 2 次 + 计时 5 次 (torch.npu.Event 设备时间), 取平均
  ⑤ 精简输出: 只保留 ns/speedup 排序结果 → 喂 planner 决策

用法 (910B3):
  python3 sweep_blocks.py input/matmul            # 全量扫 matmul 的 BLOCK
  python3 sweep_blocks.py input/matmul --quick    # 只扫 top-48 候选
  python3 main.py input/matmul --sweep-blocks     # main 内置

候选规模 (M=N=K=2048, fp32):
  L0 约束后 ~300-500 候选, 总 kernel call ~3500 次, 单进程 ~5-10min (含编译)
"""
import argparse, json, os, re, shutil, sys, math, time, textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ── 910B3 硬件约束 ──
L0A_BYTES = 64 * 1024   # 64KB
L0B_BYTES = 64 * 1024   # 64KB
L0C_BYTES = 128 * 1024  # 128KB
UB_BYTES  = 192 * 1024  # 192KB

# ── Sweep 参数 ──
DEFAULT_WARMUP = 2       # 每 config 预热次数
DEFAULT_LOOP    = 5       # 每 config 计时次数
DEFAULT_MIN_GRID = 16     # 最小 grid (至少覆盖 20 核的大部分)
DEFAULT_MAX_GRID = 3000   # 最大 grid (防调度开销淹没收益)

# ── 算子元数据 ──
#   每个算子: vars=要扫的参数, type=候选生成方式, multi_kernel=是否多 kernel 管线
SWEEP_META = {
    "matmul":          {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"), "type": "matmul", "multi": False},
    "attention_mlp":   {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"), "type": "matmul", "multi": True},
    "flash_attention": {"vars": ("BLOCK_M", "BLOCK_N", "BLOCK_K"), "type": "matmul", "multi": False},
    "conv2d":          {"vars": ("BLOCK_K", "BLOCK_OW"),           "type": "conv2d", "multi": False},
    "conv_bias_relu":  {"vars": ("BLOCK_K", "BLOCK_OW"),           "type": "conv2d", "multi": True},
    "rms_norm":        None,   # 行级 kernel, BLOCK_N 由 N 推导 (2 幂), 无自由分块参数
}


# ═══════════════════════════════════════════════════════════════════════════════
# 候选生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_matmul_candidates(M: int = 2048, N: int = 2048, K: int = 2048,
                               dtype_size: int = 4,
                               min_grid: int = DEFAULT_MIN_GRID,
                               max_grid: int = DEFAULT_MAX_GRID,
                               verbose: bool = True) -> List[Tuple[int, int, int]]:
    """生成 fp32 matmul 型算子的所有 L0 合法 (BM, BN, BK) 候选.

    约束:
      L0A: BM×BK×4 ≤ 64KB  →  BM×BK ≤ 16384
      L0B: BN×BK×4 ≤ 64KB  →  BN×BK ≤ 16384
      L0C: BM×BN×4 ≤ 128KB →  BM×BN ≤ 32768
      UB:  BM×BK + BN×BK + BM×BN ≤ 49152  (3 缓冲粗略估算)
      全部 16 倍数 (Cube MMA 基础粒度)
      grid = ceil(M/BM)×ceil(N/BN) ∈ [min_grid, max_grid]  (合理并行度)
    """
    L0A_MAX = L0A_BYTES // dtype_size  # 16384
    L0B_MAX = L0B_BYTES // dtype_size  # 16384
    L0C_MAX = L0C_BYTES // dtype_size  # 32768
    UB_MAX  = UB_BYTES  // dtype_size  # 49152

    # BM 扫描范围: [16, min(1024, L0C_MAX//16)]
    bm_max = min(1024, L0C_MAX // 16)
    # BN 扫描范围: [16, min(1024, L0C_MAX//16)]
    bn_max = min(1024, L0C_MAX // 16)
    # BK 扫描范围: [16, min(512, L0A_MAX//16, L0B_MAX//16)]
    bk_max = min(512, L0A_MAX // 16, L0B_MAX // 16)

    candidates = []
    total = 0
    for bm in range(16, bm_max + 1, 16):
        # BN 上限受 L0C 约束: bm×bn ≤ L0C_MAX
        bn_limit = min(bn_max, L0C_MAX // bm)
        for bn in range(16, bn_limit + 1, 16):
            bm_bn = bm * bn
            # BK 上限受 L0A 和 L0B 约束
            bk_limit = min(bk_max, L0A_MAX // bm, L0B_MAX // bn)
            for bk in range(16, bk_limit + 1, 16):
                total += 1
                # UB 约束 (3 缓冲: A tile + B tile + C tile)
                if bm * bk + bn * bk + bm_bn > UB_MAX:
                    continue
                # Grid 约束 (与问题尺寸挂钩, 保证核利用率)
                if M and N:
                    grid = math.ceil(M / bm) * math.ceil(N / bn)
                    if grid < min_grid or grid > max_grid:
                        continue
                candidates.append((bm, bn, bk))

    if verbose:
        print(f"  [候选生成] 检查 {total} 组合 → {len(candidates)} 有效 "
              f"(L0A≤{L0A_MAX}, L0B≤{L0B_MAX}, L0C≤{L0C_MAX}, "
              f"UB≤{UB_MAX}, grid∈[{min_grid},{max_grid}], 步长16)")
        # 统计各维度覆盖
        bms = sorted(set(c[0] for c in candidates))
        bns = sorted(set(c[1] for c in candidates))
        bks = sorted(set(c[2] for c in candidates))
        print(f"    覆盖: BM∈{{{bms[0]}..{bms[-1]}}} ({len(bms)}值)  "
              f"BN∈{{{bns[0]}..{bns[-1]}}} ({len(bns)}值)  "
              f"BK∈{{{bks[0]}..{bks[-1]}}} ({len(bks)}值)")
    return candidates


def generate_conv2d_candidates(OH: int = 64, OW: int = 64, K_OUT: int = 32,
                               dtype_size: int = 4,
                               verbose: bool = True) -> List[Tuple[int, int]]:
    """生成 conv2d 型算子的所有 L0 合法 (BLOCK_K, BLOCK_OW) 候选.

    约束:
      acc = tl.zeros((BLOCK_K, BLOCK_OW)) → BLOCK_K×BLOCK_OW×dtype ≤ 128KB (L0C)
      BLOCK_OW ≤ OW (超出无意义)
      全部 16 倍数
    """
    L0C_MAX = L0C_BYTES // dtype_size  # 32768

    candidates = []
    total = 0
    for bk in range(16, max(16, K_OUT) + 1, 16):
        for bow in range(16, OW + 1, 16):
            total += 1
            if bk * bow > L0C_MAX:
                continue
            candidates.append((bk, bow))

    if verbose:
        bks = sorted(set(c[0] for c in candidates))
        bows = sorted(set(c[1] for c in candidates))
        print(f"  [候选生成] 检查 {total} 组合 → {len(candidates)} 有效 "
              f"(L0C≤{L0C_MAX}, BLOCK_OW≤{OW})")
        print(f"    覆盖: BLOCK_K∈{{{bks[0]}..{bks[-1]}}} ({len(bks)}值)  "
              f"BLOCK_OW∈{{{bows[0]}..{bows[-1]}}} ({len(bows)}值)")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep Runner 生成 (算子特化)
# ═══════════════════════════════════════════════════════════════════════════════

def _read_op_params(code: str) -> dict:
    """从 kernel_op.py 源码提取关键参数 (M, N, K, 等)."""
    params = {}
    # 通用: 提取所有 int(os.environ.get("KEY", default)) 形式的参数
    for m in re.finditer(r'(\w+)\s*=\s*int\(os\.environ\.get\("(\w+)",\s*(\d+)\)\)', code):
        params[m.group(1)] = int(m.group(3))
    # 派生参数: OH, OW
    for m in re.finditer(r'(\w+)\s*=\s*\(.*?\)\s*//?\s*\d+\s*\+\s*\d+', code):
        pass  # 派生公式, 用默认值
    # 默认值: 从纯赋值提取 (如 M = 2048)
    for m in re.finditer(r'^\s*(\w+)\s*=\s*(\d+)\s*$', code, re.M):
        if m.group(1) not in params and m.group(1).isupper():
            params[m.group(1)] = int(m.group(2))
    return params


def _generate_matmul_runner(op_dir: Path, candidates: List[Tuple],
                            op_meta: dict) -> str:
    """生成 matmul 型算子的 sweep runner 脚本."""
    kernel_src = (op_dir / "kernel_op.py").read_text(encoding="utf-8")
    # 取 module-level 代码 (imports + config + kernel defs), 去掉 __main__ 块
    module_code = kernel_src.split('if __name__ == "__main__":')[0].rstrip()

    params = _read_op_params(kernel_src)
    M  = params.get("M", params.get("SEQ", 2048))
    N  = params.get("N", params.get("DIM", 2048))
    K  = params.get("K", 2048)
    vars_ = op_meta["vars"]
    multi = op_meta["multi"]

    cands_json = json.dumps(candidates)

    if op_dir.name == "matmul":
        # ── 单 matmul: 一 kernel, 简单 setup ──
        setup = textwrap.dedent(f"""\
        # ── Tensor setup (matmul) ──
        M, N, K = {M}, {N}, {K}
        DTYPE = torch.float32
        device = torch.device("npu")
        x = (torch.rand(M, K, dtype=DTYPE, device=device) - 0.5) * 0.1
        w = (torch.rand(K, N, dtype=DTYPE, device=device) - 0.5) * 0.1
        z = torch.empty(M, N, dtype=DTYPE, device=device)

        def run_one(bm, bn, bk):
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            matmul_kernel[grid](x, w, z, M, N, K,
                x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                z.stride(0), z.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        """)

    elif op_dir.name == "attention_mlp":
        # ── 多 kernel 管线: 所有 matmul 相关 kernel 一起跑 ──
        seq = M; dim = N
        setup = textwrap.dedent(f"""\
        # ── Tensor setup (attention_mlp) ──
        seq, dim = {seq}, {dim}
        scale = 1.0 / (dim ** 0.5)
        DTYPE = torch.float32
        device = torch.device("npu")
        x  = (torch.rand(seq, dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        wq = (torch.rand(dim, dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        wk = (torch.rand(dim, dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        wv = (torch.rand(dim, dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        w1 = (torch.rand(dim, dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        b1 = (torch.rand(dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        w2 = (torch.rand(dim, dim, dtype=DTYPE, device=device) - 0.5) * 0.1
        q = torch.empty(seq, dim, dtype=DTYPE, device=device)
        k = torch.empty(seq, dim, dtype=DTYPE, device=device)
        v = torch.empty(seq, dim, dtype=DTYPE, device=device)
        s = torch.empty(seq, seq, dtype=DTYPE, device=device)
        p = torch.empty(seq, seq, dtype=DTYPE, device=device)
        o = torch.empty(seq, dim, dtype=DTYPE, device=device)
        y = torch.empty(seq, dim, dtype=DTYPE, device=device)
        z = torch.empty(seq, dim, dtype=DTYPE, device=device)
        out = torch.empty(seq, dim, dtype=DTYPE, device=device)

        def run_one(bm, bn, bk):
            g_hidden  = (triton.cdiv(seq, bm) * triton.cdiv(dim, bn),)
            g_scores  = (triton.cdiv(seq, bm) * triton.cdiv(seq, bn),)
            g_softmax = (seq,)
            g_add     = (triton.cdiv(seq * dim, BLOCK_S),)
            # Q,K,V = X @ Wq/Wk/Wv
            matmul_kernel[g_hidden](x, wq, q, seq, dim, dim,
                x.stride(0), x.stride(1), wq.stride(0), wq.stride(1),
                q.stride(0), q.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            matmul_kernel[g_hidden](x, wk, k, seq, dim, dim,
                x.stride(0), x.stride(1), wk.stride(0), wk.stride(1),
                k.stride(0), k.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            matmul_kernel[g_hidden](x, wv, v, seq, dim, dim,
                x.stride(0), x.stride(1), wv.stride(0), wv.stride(1),
                v.stride(0), v.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            # S = Q@K^T · scale
            attention_scores_kernel[g_scores](q, k, s, seq, dim, scale,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            # P = softmax(S)
            softmax_kernel[g_softmax](s, p, seq, seq, BLOCK=BLOCK_S)
            # O = P @ V
            matmul_kernel[g_hidden](p, v, o, seq, dim, seq,
                p.stride(0), p.stride(1), v.stride(0), v.stride(1),
                o.stride(0), o.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            # Y = GELU(O @ W1 + b1)
            mlp_gelu_kernel[g_hidden](o, w1, b1, y, seq, dim, dim,
                o.stride(0), o.stride(1), w1.stride(0), w1.stride(1),
                y.stride(0), y.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            # Z = Y @ W2
            matmul_kernel[g_hidden](y, w2, z, seq, dim, dim,
                y.stride(0), y.stride(1), w2.stride(0), w2.stride(1),
                z.stride(0), z.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            # Out = Z + O (残差)
            add_kernel[g_add](z, o, out, seq * dim, BLOCK=BLOCK_S)
        """)

    elif op_dir.name == "flash_attention":
        seq = params.get("SEQ", 2048)
        nh = params.get("NHEADS", 8)
        dim = params.get("DIM", 64)
        setup = textwrap.dedent(f"""\
        # ── Tensor setup (flash_attention) ──
        seq, nh, dim = {seq}, {nh}, {dim}
        scale = 1.0 / (dim ** 0.5)
        DTYPE = torch.float32
        device = torch.device("npu")
        q = (torch.randn(seq, nh, dim, dtype=DTYPE, device=device)) * 0.1
        k = (torch.randn(seq, nh, dim, dtype=DTYPE, device=device)) * 0.1
        v = (torch.randn(seq, nh, dim, dtype=DTYPE, device=device)) * 0.1
        o = torch.empty(seq, nh, dim, dtype=DTYPE, device=device)

        def run_one(bm, bn, bk):
            grid = (triton.cdiv(seq, bm) * nh,)
            flash_attn_mha_kernel[grid](q, k, v, o, seq, nh, dim, scale,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        """)

    else:
        raise ValueError(f"未知 matmul 型算子: {op_dir.name}")

    runner = textwrap.dedent(f"""\
    #!/usr/bin/env python3
    """ + '"""Auto-generated BLOCK sweep runner — DO NOT EDIT."""' + f"""
    import os, sys, json, math, time
    import torch
    import torch_npu
    import triton
    import triton.language as tl

    # ── Module-level code from kernel_op.py (imports, config, kernel defs) ──
    {module_code}

    {setup}

    # ── Sweep ──
    CANDIDATES = {cands_json}
    WARMUP = int(os.environ.get("SWEEP_WARMUP", "{DEFAULT_WARMUP}"))
    LOOP   = int(os.environ.get("SWEEP_LOOP", "{DEFAULT_LOOP}"))
    results = []
    n_total = len(CANDIDATES)

    print(f"[sweep] {{n_total}} candidates, warmup={{WARMUP}}, loop={{LOOP}}", flush=True)

    for idx, cfg in enumerate(CANDIDATES):
        {', '.join(f'{v}=cfg[{i}]' for i, v in enumerate(vars_))}
        try:
            # Warmup
            for _ in range(WARMUP):
                run_one({', '.join(vars_)})
            torch.npu.synchronize()

            # Timing
            times = []
            for _ in range(LOOP):
                ev_s = torch.npu.Event(enable_timing=True)
                ev_e = torch.npu.Event(enable_timing=True)
                ev_s.record()
                run_one({', '.join(vars_)})
                ev_e.record()
                torch.npu.synchronize()
                times.append(ev_s.elapsed_time(ev_e))

            avg_ns = sum(times) / len(times) * 1e6  # ms → ns
            results.append({{"block": list(cfg), "ns": round(avg_ns, 1)}})
            print(f"  [{{idx+1}}/{{n_total}}] {{cfg}}: {{avg_ns:.0f}}ns", flush=True)
        except Exception as e:
            results.append({{"block": list(cfg), "ns": None, "error": str(e)[:120]}})
            print(f"  [{{idx+1}}/{{n_total}}] {{cfg}}: ERROR {{str(e)[:100]}}", flush=True)

    # Sort & save
    valid = [r for r in results if r.get("ns")]
    errs  = [r for r in results if not r.get("ns")]
    valid.sort(key=lambda r: r["ns"])
    out = {{"measured_at": __import__("datetime").datetime.now().isoformat(),
           "total_candidates": n_total, "valid": len(valid), "errors": len(errs),
           "warmup": WARMUP, "loop": LOOP,
           "results": valid + errs}}
    out_path = os.environ.get("SWEEP_OUTPUT", "sweep_result.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"\\n[sweep] Done: {{len(valid)}} valid + {{len(errs)}} errors → {{out_path}}", flush=True)
    if valid:
        print(f"[sweep] Best: {{valid[0]['block']}} = {{valid[0]['ns']:.0f}}ns", flush=True)
    """)
    return runner


def _generate_conv2d_runner(op_dir: Path, candidates: List[Tuple],
                            op_meta: dict) -> str:
    """生成 conv2d 型算子的 sweep runner 脚本."""
    kernel_src = (op_dir / "kernel_op.py").read_text(encoding="utf-8")
    module_code = kernel_src.split('if __name__ == "__main__":')[0].rstrip()
    params = _read_op_params(kernel_src)
    multi = op_meta["multi"]
    cands_json = json.dumps(candidates)
    vars_ = op_meta["vars"]

    if op_dir.name == "conv2d":
        setup = textwrap.dedent("""\
        # ── Tensor setup (conv2d) ──
        DTYPE = torch.float32
        device = torch.device("npu")
        x = (torch.randn(N_B, C_IN, H, W, dtype=DTYPE, device=device)) * 0.1
        w = (torch.randn(K_OUT, C_IN, R, S, dtype=DTYPE, device=device)) * 0.1
        y = torch.empty(N_B, K_OUT, OH, OW, dtype=DTYPE, device=device)

        def run_one(bk, bow):
            grid = (N_B * OH * triton.cdiv(OW, bow),)
            conv2d_kernel[grid](x, w, y, N_B, H, W, K_OUT, OH, OW,
                BLOCK_K=bk, BLOCK_OW=bow,
                C=C_IN, R=R, S=S, PAD=PAD)
        """)

    elif op_dir.name == "conv_bias_relu":
        setup = textwrap.dedent("""\
        # ── Tensor setup (conv_bias_relu) ──
        DTYPE = torch.float32
        device = torch.device("npu")
        x = (torch.randn(N_B, C_IN, H, W, dtype=DTYPE, device=device)) * 0.1
        w = (torch.randn(K_OUT, C_IN, R, S, dtype=DTYPE, device=device)) * 0.1
        bias = (torch.randn(K_OUT, dtype=DTYPE, device=device)) * 0.1
        yc = torch.empty(N_B, K_OUT, OH, OW, dtype=DTYPE, device=device)
        yb = torch.empty(N_B, K_OUT, OH, OW, dtype=DTYPE, device=device)
        y  = torch.empty(N_B, K_OUT, OH, OW, dtype=DTYPE, device=device)
        n_el = N_B * K_OUT * OH * OW

        def run_one(bk, bow):
            grid_conv = (N_B * OH * triton.cdiv(OW, bow),)
            grid_el   = (triton.cdiv(n_el, BLOCK_EL),)
            conv2d_kernel[grid_conv](x, w, yc, N_B, H, W, K_OUT, OH, OW,
                BLOCK_K=bk, BLOCK_OW=bow, C=C_IN, R=R, S=S, PAD=PAD)
            bias_kernel[grid_el](yc, bias, yb, n_el, K_OUT, OH, OW, BLOCK=BLOCK_EL)
            relu_kernel[grid_el](yb, y, n_el, BLOCK=BLOCK_EL)
        """)
    else:
        raise ValueError(f"未知 conv2d 型算子: {op_dir.name}")

    runner = textwrap.dedent(f"""\
    #!/usr/bin/env python3
    """ + '"""Auto-generated BLOCK sweep runner — DO NOT EDIT."""' + f"""
    import os, sys, json, math, time
    import torch
    import torch_npu
    import triton
    import triton.language as tl

    # ── Module-level code from kernel_op.py ──
    {module_code}

    {setup}

    # ── Sweep ──
    CANDIDATES = {cands_json}
    WARMUP = int(os.environ.get("SWEEP_WARMUP", "{DEFAULT_WARMUP}"))
    LOOP   = int(os.environ.get("SWEEP_LOOP", "{DEFAULT_LOOP}"))
    results = []
    n_total = len(CANDIDATES)

    print(f"[sweep] {{n_total}} candidates, warmup={{WARMUP}}, loop={{LOOP}}", flush=True)

    for idx, cfg in enumerate(CANDIDATES):
        {', '.join(f'{v}=cfg[{i}]' for i, v in enumerate(vars_))}
        try:
            for _ in range(WARMUP):
                run_one({', '.join(vars_)})
            torch.npu.synchronize()

            times = []
            for _ in range(LOOP):
                ev_s = torch.npu.Event(enable_timing=True)
                ev_e = torch.npu.Event(enable_timing=True)
                ev_s.record()
                run_one({', '.join(vars_)})
                ev_e.record()
                torch.npu.synchronize()
                times.append(ev_s.elapsed_time(ev_e))

            avg_ns = sum(times) / len(times) * 1e6
            results.append({{"block": list(cfg), "ns": round(avg_ns, 1)}})
            print(f"  [{{idx+1}}/{{n_total}}] {{cfg}}: {{avg_ns:.0f}}ns", flush=True)
        except Exception as e:
            results.append({{"block": list(cfg), "ns": None, "error": str(e)[:120]}})
            print(f"  [{{idx+1}}/{{n_total}}] {{cfg}}: ERROR {{str(e)[:100]}}", flush=True)

    valid = [r for r in results if r.get("ns")]
    errs  = [r for r in results if not r.get("ns")]
    valid.sort(key=lambda r: r["ns"])
    out = {{"measured_at": __import__("datetime").datetime.now().isoformat(),
           "total_candidates": n_total, "valid": len(valid), "errors": len(errs),
           "warmup": WARMUP, "loop": LOOP,
           "results": valid + errs}}
    out_path = os.environ.get("SWEEP_OUTPUT", "sweep_result.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"\\n[sweep] Done: {{len(valid)}} valid + {{len(errs)}} errors → {{out_path}}", flush=True)
    if valid:
        print(f"[sweep] Best: {{valid[0]['block']}} = {{valid[0]['ns']:.0f}}ns", flush=True)
    """)
    return runner


def _generate_runner(op_dir: Path, candidates: List[Tuple],
                     op_meta: dict) -> str:
    """根据算子类型生成对应的 sweep runner 脚本."""
    if op_meta["type"] == "matmul":
        return _generate_matmul_runner(op_dir, candidates, op_meta)
    elif op_meta["type"] == "conv2d":
        return _generate_conv2d_runner(op_dir, candidates, op_meta)
    raise ValueError(f"不支持候选生成: {op_meta['type']}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主 sweep 入口
# ═══════════════════════════════════════════════════════════════════════════════

def _read_current_block(code: str, varnames: Tuple[str, ...]) -> Optional[tuple]:
    """读 config 区当前 BLOCK 值."""
    if varnames == ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        m = re.search(r"BLOCK_M\s*,\s*BLOCK_N\s*,\s*BLOCK_K\s*=\s*([\d, ]+)", code)
        if m:
            vals = [int(x) for x in re.split(r"[,，\s]+", m.group(1).strip()) if x.strip().isdigit()]
            if len(vals) == 3:
                return tuple(vals)
    else:
        vals = []
        for var in varnames:
            m = re.search(rf"{var}\s*=\s*(\d+)", code)
            if not m:
                return None
            vals.append(int(m.group(1)))
        return tuple(vals)
    return None


def _apply_block(code: str, varnames: Tuple[str, ...], vals: tuple) -> str:
    """把 config 区 BLOCK 赋值替换成 vals."""
    if varnames == ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        m = re.search(r"BLOCK_M\s*,\s*BLOCK_N\s*,\s*BLOCK_K\s*=\s*[\d, ]+", code)
        if not m:
            raise ValueError("源码缺 BLOCK_M, BLOCK_N, BLOCK_K 赋值")
        return (code[:m.start()]
                + f"BLOCK_M, BLOCK_N, BLOCK_K = {vals[0]}, {vals[1]}, {vals[2]}"
                + code[m.end():])
    # 通用: 逐变量替换
    for var, val in zip(varnames, vals):
        m = re.search(rf"{var}\s*=\s*\d+", code)
        if not m:
            raise ValueError(f"源码缺 {var} 赋值")
        code = code[:m.start()] + f"{var} = {val}" + code[m.end():]
    return code


def sweep(op_dir: Path, quick: bool = False, out_dir: Optional[Path] = None) -> dict:
    """对 input/<op> 扫描最优 BLOCK → 返回 {{best_block, results:[...]}} 并写回。

    流程:
      1. 生成全部 L0 合法候选
      2. 生成 sweep runner 脚本
      3. **单进程**运行 runner → torch.npu.Event 计时 (无 msprof 开销)
      4. 解析结果, 最优写回 kernel_op.py
      5. 可选: 一次 msprof 包裹 runner 获取深度 profiling

    Args:
      op_dir: kernel_op.py 所在目录
      quick: True → 只测 ~48 候选 (均匀采样)
      out_dir: 产物目录 (默认 outputs/<op>/block_sweep/)
    """
    op = op_dir.name
    meta = SWEEP_META.get(op)
    kernel = op_dir / "kernel_op.py"

    if meta is None:
        print(f"[sweep] {op}: 无自由分块参数 (rms_norm 行级), 跳过")
        return {"skipped": True}
    if not kernel.exists():
        print(f"[sweep] ❌ 缺 {kernel}")
        return {"error": "no kernel_op.py"}

    code = kernel.read_text(encoding="utf-8")
    cur = _read_current_block(code, meta["vars"])
    if cur is None:
        print(f"[sweep] ❌ 读不到 {meta['vars']} 当前值")
        return {"error": "cannot read current block"}
    print(f"[sweep] {op}: 当前 BLOCK {meta['vars']}={cur}")

    # 1. 生成候选
    params = _read_op_params(code)
    if meta["type"] == "matmul":
        M = params.get("M", params.get("SEQ", 2048))
        N = params.get("N", params.get("DIM", 2048))
        K = params.get("K", 2048)
        candidates = generate_matmul_candidates(M=M, N=N, K=K)
    elif meta["type"] == "conv2d":
        OH = params.get("OH", 64)
        OW = params.get("OW", 64)
        K_OUT = params.get("K_OUT", 32)
        candidates = generate_conv2d_candidates(OH=OH, OW=OW, K_OUT=K_OUT)

    if quick:
        # Quick 模式: 均匀采样 ~48 个候选
        step = max(1, len(candidates) // 48)
        candidates = candidates[::step][:48]
        print(f"  [quick] 采样 {len(candidates)} 候选")

    # 确保当前块在候选列表里
    cur_tuple = tuple(cur)
    if cur_tuple not in candidates:
        candidates.insert(0, cur_tuple)

    # 2. 生成 runner 脚本
    out_root = (Path(out_dir) if out_dir
                else _PROJECT / "outputs" / op / "block_sweep")
    out_root.mkdir(parents=True, exist_ok=True)
    runner_code = _generate_runner(op_dir, candidates, meta)
    runner_path = out_root / "sweep_runner.py"
    runner_path.write_text(runner_code, encoding="utf-8")
    result_path = out_root / "sweep_result.json"

    print(f"[sweep] Runner → {runner_path} ({len(candidates)} candidates)")

    # 3. 运行 runner (可选 msprof 包裹)
    import subprocess
    py = sys.executable or "python3"
    env = dict(os.environ,
               SWEEP_WARMUP=str(DEFAULT_WARMUP),
               SWEEP_LOOP=str(DEFAULT_LOOP),
               SWEEP_OUTPUT=str(result_path))

    use_msprof = os.environ.get("SWEEP_MSPROF", "0") == "1"
    t0 = time.time()
    if use_msprof:
        msprof_dir = out_root / "msprof_sweep"
        msprof_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["msprof", f"--output={msprof_dir}",
               f"--application={py} {runner_path}", "--ai-core=on"]
        print(f"  [sweep] msprof 模式: {' '.join(cmd)}")
    else:
        cmd = [py, str(runner_path)]
        print(f"  [sweep] 直接运行 (torch.npu.Event 计时, 无 msprof 开销)")

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="backslashreplace",
                          timeout=7200, env=env, cwd=str(out_root))
    except subprocess.TimeoutExpired:
        print(f"  [sweep] ❌ 超时 (2h)")
        return {"error": "sweep timeout"}
    except Exception as e:
        print(f"  [sweep] ❌ 运行失败: {e}")
        return {"error": str(e)[:200]}

    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.stderr and "Error" in r.stderr:
        print(f"  [sweep] stderr: {r.stderr[-500:]}")

    elapsed = time.time() - t0
    print(f"  [sweep] 运行 {elapsed:.0f}s")

    # 4. 解析结果
    if not result_path.exists():
        print(f"  [sweep] ❌ 无结果文件 {result_path}")
        return {"error": "no result file"}

    data = json.loads(result_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    valid = [r for r in results if r.get("ns")]
    if not valid:
        print(f"[sweep] 无有效候选, 当前 {cur} 保留")
        return {"best_block": cur, "results": results, "unchanged": True}

    best = valid[0]
    cur_result = next((r for r in valid if tuple(r["block"]) == cur_tuple), None)
    cur_ns = cur_result["ns"] if cur_result else None

    # 计算 speedup (相对于当前块)
    for r in results:
        if r.get("ns") and cur_ns:
            r["speedup_vs_cur"] = round(cur_ns / r["ns"], 3)

    # 5. 最优写回 (若比当前快)
    best_block = tuple(best["block"])
    if best_block != cur_tuple and cur_ns and best["ns"] < cur_ns:
        new_code = _apply_block(code, meta["vars"], best_block)
        kernel.write_text(new_code, encoding="utf-8")
        improvement = cur_ns / best["ns"]
        print(f"\n[sweep] ✅ 最优 {meta['vars']}={best_block} ({best['ns']:.0f}ns, "
              f"vs {cur_ns:.0f}ns = {improvement:.2f}x) → 已写回 {kernel}")
        _write_report(out_root, best_block, results, meta["vars"])
        return {"best_block": best_block, "results": results,
                "improvement_x": round(improvement, 3),
                "candidates_tested": len(candidates)}

    if best_block == cur_tuple:
        print(f"\n[sweep] 当前 {cur} ({cur_ns:.0f}ns) 已是 {len(valid)} 候选中最优, 保留")
    else:
        print(f"\n[sweep] 最优 {best_block} ({best['ns']:.0f}ns) 但 ≤ 当前 {cur_ns:.0f}ns, 保留当前")
    _write_report(out_root, cur, results, meta["vars"])
    return {"best_block": cur, "results": results, "unchanged": True,
            "candidates_tested": len(candidates)}


def _write_report(out_root: Path, best_block, results, varnames):
    """写人类可读 + JSON 报告."""
    out_root.mkdir(parents=True, exist_ok=True)
    # JSON
    (out_root / "sweep_result.json").write_text(json.dumps({
        "measured_at": datetime.now().isoformat(),
        "best_block": list(best_block),
        "vars": list(varnames),
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    # TXT
    with open(out_root / "sweep_result.txt", "w", encoding="utf-8") as f:
        f.write("══ BLOCK 扫描结果 (ns 越小越快) ══\n")
        f.write(f"候选总数: {len(results)}\n\n")
        for r in sorted(results, key=lambda x: x.get("ns") or 9e18):
            blk = r.get("block")
            sp = f"  ({r.get('speedup_vs_cur', 1):.2f}x vs 当前)" if r.get("speedup_vs_cur") else ""
            if r.get("ns"):
                f.write(f"  {blk}: {r['ns']:.0f}ns{sp}\n")
            else:
                f.write(f"  {blk}: FAIL {r.get('error', '')[:60]}\n")
    print(f"[sweep] 报告 → {out_root}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="全面 BLOCK 扫描 (v2)")
    p.add_argument("op_dir", type=str)
    p.add_argument("--quick", action="store_true", help="只扫 ~48 候选 (省时间)")
    p.add_argument("--msprof", action="store_true", help="用 msprof 包裹 runner (深度 profiling)")
    args = p.parse_args()
    if args.msprof:
        os.environ["SWEEP_MSPROF"] = "1"
    result = sweep(Path(args.op_dir), args.quick)
    sys.exit(0 if result else 1)
