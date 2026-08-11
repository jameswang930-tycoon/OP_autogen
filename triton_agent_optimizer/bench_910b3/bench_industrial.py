#!/usr/bin/env python3
"""工业级基准 — 各算子用"真正的工业级优化实现"测端到端耗时 (msprof 同口径).

═══ 为什么 ═══
  跟 naive torch 比不够 — 要跟工业级天花板比. Ascend 910B 上的工业级实现:
    - matmul/conv/norm 单算子: torch eager 已走 CANN aclnn vendor kernel (工业级)
    - 融合链 (MLP/matmul+epilogue/逐元素): torch.compile + TorchAir 图模式 (算子融合)
    - attention: CANN FlashAttention (torch_npu.npu_prompt_flash_attention)
  → 我们的 triton kernel 端到端 vs 这些工业级实现的端到端.

═══ 用法 (910B 服务器) ═══
  python3 bench_industrial.py <op> [--mode eager|compile|fa] [--measure 30] [--warmup 5]
  例:
    python3 bench_industrial.py matmul --mode compile    # MLP 融合链 compile
    python3 bench_industrial.py flash_attention --mode fa  # CANN FlashAttention
    python3 bench_industrial.py rms_norm --mode eager
  输出: bench_910b3/industrial_<op>_<mode>_tflops.json
        time_us(端到端, msprof Σ全部含框架) / kernel_time_us(纯kernel, Σ非aclnn)

═══ 口径 ═══
  与 verify/pt_msprof 完全一致: 一次 msprof 启动, 内层 warmup+measure 次 forward,
  跳过热身行 ÷measure. 端到端 = Σ全部 kernel 行, 纯kernel = Σ非 aclnn 行.
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR.parent))


# ═══════════════════════════════════════════════════════════════════════
#  算子注册表 — 每个算子: 构造输入 + torch forward + flops
#  尺寸 env 与项目 input/<op>/kernel_op.py 默认对齐 (2048/matmul, 2048x2048/rms, 4M/逐元素, ...)
# ═══════════════════════════════════════════════════════════════════════

def _shapes(op):
    S = {
        "matmul":          lambda: dict(M=int(os.environ.get("MATMUL_M", 2048)),
                                        K=int(os.environ.get("MATMUL_K", 2048)),
                                        N=int(os.environ.get("MATMUL_N", 2048)),
                                        H=int(os.environ.get("MLP_HIDDEN", 2048))),
        "attention_mlp":   lambda: dict(S=int(os.environ.get("AM_SEQ", 2048)),
                                        D=int(os.environ.get("AM_DIM", 2048)),
                                        H=int(os.environ.get("AM_HIDDEN", 2048))),
        "matmul_relu":     lambda: dict(M=int(os.environ.get("MATMUL_M", 2048)),
                                        K=int(os.environ.get("MATMUL_K", 2048)),
                                        N=int(os.environ.get("MATMUL_N", 2048))),
        "matmul_transpose": lambda: dict(M=int(os.environ.get("MATMUL_M", 2048)),
                                         K=int(os.environ.get("MATMUL_K", 2048)),
                                         N=int(os.environ.get("MATMUL_N", 2048))),
        "rms_norm":        lambda: dict(M=int(os.environ.get("RMS_M", 2048)),
                                        N=int(os.environ.get("RMS_N", 2048))),
        "rms_norm_residual": lambda: dict(M=int(os.environ.get("RMSR_M", 2048)),
                                          N=int(os.environ.get("RMSR_N", 4096))),
        "layernorm":       lambda: dict(M=int(os.environ.get("LN_M", 2048)),
                                        N=int(os.environ.get("LN_N", 2048))),
        "sigmoid":         lambda: dict(N=int(os.environ.get("SIGMOID_N", 4 * 1024 * 1024))),
        "softmax":         lambda: dict(M=int(os.environ.get("SOFTMAX_M", 2048)),
                                        N=int(os.environ.get("SOFTMAX_N", 2048))),
        "vector_add":      lambda: dict(N=int(os.environ.get("VEC_N", 4 * 1024 * 1024))),
        "fused_add_mul":   lambda: dict(N=int(os.environ.get("FAM_N", 4 * 1024 * 1024))),
        "flash_attention": lambda: dict(B=int(os.environ.get("FA_BATCH", 1)),
                                        S=int(os.environ.get("FA_SEQ", 2048)),
                                        NH=int(os.environ.get("FA_HEADS", 8)),
                                        D=int(os.environ.get("FA_DIM", 64))),
        "conv2d":          lambda: dict(NB=int(os.environ.get("CONV_N", 1)),
                                        C=int(os.environ.get("CONV_C", 8)),
                                        H=int(os.environ.get("CONV_H", 64)),
                                        W=int(os.environ.get("CONV_W", 64)),
                                        K=int(os.environ.get("CONV_K", 32)),
                                        R=int(os.environ.get("CONV_R", 3)),
                                        Sd=int(os.environ.get("CONV_S", 3)),
                                        P=int(os.environ.get("CONV_P", 1))),
        "conv_bias_relu":  lambda: dict(NB=int(os.environ.get("CONV_N", 1)),
                                        C=int(os.environ.get("CONV_C", 8)),
                                        H=int(os.environ.get("CONV_H", 64)),
                                        W=int(os.environ.get("CONV_W", 64)),
                                        K=int(os.environ.get("CONV_K", 32)),
                                        R=int(os.environ.get("CONV_R", 3)),
                                        Sd=int(os.environ.get("CONV_S", 3)),
                                        P=int(os.environ.get("CONV_P", 1))),
    }
    return S[op]()


def _make_forward(op, sh):
    """返回 (forward_fn, ) 无参数闭包 — 每个 op 的 torch 计算 (eager)."""
    import torch
    import torch.nn.functional as F
    npu = torch.device("npu")
    DT = torch.float32
    if op == "matmul":                       # 两层 MLP: Y=GELU(X@W1+b1)@W2
        M, K, N, H = sh["M"], sh["K"], sh["N"], sh["H"]
        x = (torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1
        w1 = (torch.rand(K, H, device=npu, dtype=DT) - 0.5) * 0.1
        b1 = (torch.rand(H, device=npu, dtype=DT) - 0.5) * 0.1
        w2 = (torch.rand(H, N, device=npu, dtype=DT) - 0.5) * 0.1
        return lambda: torch.matmul(F.gelu(x @ w1 + b1, approximate="tanh"), w2)
    if op == "attention_mlp":                # 自注意力 + MLP (QKV→S→softmax→O→GELU→FC2→+残差)
        S, D, H = sh["S"], sh["D"], sh["H"]
        scale = 1.0 / (D ** 0.5)
        x = (torch.rand(S, D, device=npu, dtype=DT) - 0.5) * 0.1
        wq = (torch.rand(D, D, device=npu, dtype=DT) - 0.5) * 0.1
        wk = (torch.rand(D, D, device=npu, dtype=DT) - 0.5) * 0.1
        wv = (torch.rand(D, D, device=npu, dtype=DT) - 0.5) * 0.1
        w1 = (torch.rand(D, H, device=npu, dtype=DT) - 0.5) * 0.1
        b1 = (torch.rand(H, device=npu, dtype=DT) - 0.5) * 0.1
        w2 = (torch.rand(H, D, device=npu, dtype=DT) - 0.5) * 0.1
        def fwd():
            q = x @ wq; k = x @ wk; v = x @ wv
            s = (q @ k.t()) * scale
            p = torch.softmax(s, dim=-1)
            o = p @ v
            y = F.gelu(o @ w1 + b1, approximate="tanh")
            z = y @ w2
            return z + o
        return fwd
    if op == "rms_norm_residual":            # (x+res) → RMSNorm → *w
        M, N = sh["M"], sh["N"]
        x = (torch.randn(M, N, device=npu, dtype=DT)) * 0.1
        res = (torch.randn(M, N, device=npu, dtype=DT)) * 0.1
        w = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        eps = 1e-6
        def fwd():
            c = x + res
            return c / torch.sqrt((c * c).mean(-1, keepdim=True) + eps) * w
        return fwd
    if op == "matmul_relu":
        M, K, N = sh["M"], sh["K"], sh["N"]
        x = (torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1
        w = (torch.rand(K, N, device=npu, dtype=DT) - 0.5) * 0.1
        return lambda: F.relu(x @ w)
    if op == "matmul_transpose":
        M, K, N = sh["M"], sh["K"], sh["N"]
        a = (torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1
        b = (torch.rand(N, K, device=npu, dtype=DT) - 0.5) * 0.1
        return lambda: a @ b.t()
    if op == "rms_norm":
        M, N = sh["M"], sh["N"]
        x = (torch.randn(M, N, device=npu, dtype=DT)) * 0.1
        w = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        eps = 1e-6
        return lambda: x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w
    if op == "layernorm":
        M, N = sh["M"], sh["N"]
        x = (torch.randn(M, N, device=npu, dtype=DT)) * 0.1
        w = torch.ones(N, device=npu, dtype=DT)
        b = torch.zeros(N, device=npu, dtype=DT)
        return lambda: F.layer_norm(x, [N], w, b, 1e-6)
    if op == "sigmoid":
        N = sh["N"]
        x = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        return lambda: torch.sigmoid(x)
    if op == "softmax":
        M, N = sh["M"], sh["N"]
        x = (torch.randn(M, N, device=npu, dtype=DT)) * 0.1
        return lambda: torch.softmax(x, dim=-1)
    if op == "vector_add":
        N = sh["N"]
        x = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        z = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        return lambda: x + z
    if op == "fused_add_mul":
        N = sh["N"]
        x = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        z = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        w = (torch.randn(N, device=npu, dtype=DT)) * 0.1
        return lambda: (x + z) * w
    if op == "flash_attention":
        B, S, NH, D = sh["B"], sh["S"], sh["NH"], sh["D"]
        scale = 1.0 / (D ** 0.5)
        q = (torch.randn(B, NH, S, D, device=npu, dtype=torch.float16)) * 0.1
        k = (torch.randn(B, NH, S, D, device=npu, dtype=torch.float16)) * 0.1
        v = (torch.randn(B, NH, S, D, device=npu, dtype=torch.float16)) * 0.1
        def fa():
            import torch_npu
            return torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=NH, scale_value=scale, input_layout="BNSD")
        return fa
    if op == "conv2d":
        NB, C, H, W, K, R, Sd, P = (sh["NB"], sh["C"], sh["H"], sh["W"],
                                     sh["K"], sh["R"], sh["Sd"], sh["P"])
        x = (torch.randn(NB, C, H, W, device=npu, dtype=DT)) * 0.1
        w = (torch.randn(K, C, R, Sd, device=npu, dtype=DT)) * 0.1
        return lambda: F.conv2d(x, w, stride=1, padding=P)
    if op == "conv_bias_relu":
        NB, C, H, W, K, R, Sd, P = (sh["NB"], sh["C"], sh["H"], sh["W"],
                                     sh["K"], sh["R"], sh["Sd"], sh["P"])
        x = (torch.randn(NB, C, H, W, device=npu, dtype=DT)) * 0.1
        w = (torch.randn(K, C, R, Sd, device=npu, dtype=DT)) * 0.1
        b = (torch.randn(K, device=npu, dtype=DT)) * 0.1
        return lambda: F.relu(F.conv2d(x, w, b, stride=1, padding=P))
    raise ValueError(f"未知算子: {op}")


def _make_cann_fused_forward(op, sh):
    """CANN 融合算子直接调用 (aclnnFusedMatmul: bias+gelu_tanh/relu; FusedConvBiasRelu) —
    工业级"厂商融合"基准, 比 eager(分 kernel) 更优, 与 TorchAir compile(GE 图融合) 同源.

    经 cann_ops_transformer / npu_ops_transformer JIT 桥 (pip install) 注册到 torch.ops.
    ⚠ 桥接库各版本注册名不同 → 模糊探测 torch.ops 里的 fused matmul/conv; 找不到/桥缺失 →
      返回 None (调用方回退 TorchAir compile — GE 图融合也会生成同一批 CANN 融合 kernel).
    ⚠ 用法需在服务器上首次确认: 若桥装好但探测不到, 看桥的文档把 fused op 名接进来."""
    try:
        import cann_ops_transformer  # noqa: F401
        _lib = "cann"
    except ImportError:
        try:
            import npu_ops_transformer  # noqa: F401
            _lib = "npu"
        except ImportError:
            print("  ⚠ cann-fused: 缺 cann_ops_transformer/npu_ops_transformer (pip install) "
                  "→ 回退 TorchAir compile (GE 图融合 = 同批 CANN 融合 kernel)")
            return None
    import torch
    npu = torch.device("npu")
    DT = torch.float32
    fused_mm = fused_conv = None
    _ops = getattr(torch.ops, _lib, None) or torch.ops
    for _n in dir(_ops):
        if _n.startswith("_"):
            continue
        _low = _n.lower()
        try:
            _f = getattr(_ops, _n)
        except Exception:
            continue
        if not callable(_f):
            continue
        if "fused" in _low and ("matmul" in _low or "mm" in _low):
            fused_mm = fused_mm or _f
        if "fused" in _low and "conv" in _low:
            fused_conv = fused_conv or _f
    if op == "matmul" and fused_mm:
        # 两层 MLP: fc1 = aclnnFusedMatmul(gelu_tanh) 融合 (y=gelu_tanh(x@w1+b1), 1 kernel),
        #           fc2 = 普通 matmul. ★签名需服务器确认 (fused_mm 的激活参数版本可能不同)
        M, K, N, H = sh["M"], sh["K"], sh["N"], sh["H"]
        x = (torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1
        w1 = (torch.rand(K, H, device=npu, dtype=DT) - 0.5) * 0.1
        b1 = (torch.rand(H, device=npu, dtype=DT) - 0.5) * 0.1
        w2 = (torch.rand(H, N, device=npu, dtype=DT) - 0.5) * 0.1
        def fwd():
            h = fused_mm(x, w1, b1)      # fc1+bias+gelu_tanh 融合 (1 kernel)
            return h @ w2                # fc2 普通 matmul
        return fwd
    if op == "matmul_relu" and fused_mm:
        M, K, N = sh["M"], sh["K"], sh["N"]
        x = (torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1
        w = (torch.rand(K, N, device=npu, dtype=DT) - 0.5) * 0.1
        b = torch.zeros(N, device=npu, dtype=DT)
        return lambda: fused_mm(x, w, b)          # y = relu(x@w + b) (fusedOpType=relu)
    if op == "conv_bias_relu" and fused_conv:
        NB, C, H, W, K, R, Sd, P = (sh["NB"], sh["C"], sh["H"], sh["W"],
                                     sh["K"], sh["R"], sh["Sd"], sh["P"])
        x = (torch.randn(NB, C, H, W, device=npu, dtype=DT)) * 0.1
        w = (torch.randn(K, C, R, Sd, device=npu, dtype=DT)) * 0.1
        b = (torch.randn(K, device=npu, dtype=DT)) * 0.1
        return lambda: fused_conv(x, w, b, stride=1, padding=P)
    print(f"  ⚠ cann-fused[{op}]: 桥可用但没探测到对应 fused op → 回退 TorchAir compile")
    return None


def _flops(op, sh):
    """各算子 FLOPs (端到端真实量)."""
    if op == "matmul":
        return 2 * sh["M"] * sh["K"] * sh["H"] + 2 * sh["M"] * sh["H"] * sh["N"]
    if op == "attention_mlp":
        s, d, h = sh["S"], sh["D"], sh["H"]
        return 6 * s * d * d + 4 * s * s * d + 2 * s * d * h + 2 * s * h * d   # QKV+S+O+MLP
    if op in ("matmul_relu", "matmul_transpose"):
        return 2 * sh["M"] * sh["K"] * sh["N"]
    if op in ("rms_norm", "rms_norm_residual"):
        return 4 * sh["M"] * sh["N"]
    if op == "layernorm":
        return 6 * sh["M"] * sh["N"]
    if op in ("sigmoid", "vector_add"):
        return 2 * sh["N"]
    if op == "fused_add_mul":
        return 3 * sh["N"]
    if op == "softmax":
        return 5 * sh["M"] * sh["N"]
    if op == "flash_attention":
        return 4 * sh["B"] * sh["S"] * sh["S"] * sh["NH"] * sh["D"]
    if op in ("conv2d", "conv_bias_relu"):
        OH = (sh["H"] + 2 * sh["P"] - sh["R"]) // 1 + 1
        OW = (sh["W"] + 2 * sh["P"] - sh["Sd"]) // 1 + 1
        f = 2 * sh["NB"] * sh["K"] * OH * OW * sh["C"] * sh["R"] * sh["Sd"]
        return f + (2 * sh["NB"] * sh["K"] * OH * OW if op == "conv_bias_relu" else 0)
    return None


# ═══════════════════════════════════════════════════════════════════════
#  测量 — 外层包 msprof (与 pt_msprof 同法); 内层跑 forward
# ═══════════════════════════════════════════════════════════════════════

def _compile_torchair(fn, op, mode):
    """TorchAir 图模式编译 (GE 图融合 → CANN 融合 kernel); 失败回退 eager."""
    import torch
    try:
        import torchair
        cfg = torchair.CompilerConfig()
        cfg.mode = os.environ.get("TORCHAIR_MODE", "max-autotune")
        backend = torchair.get_npu_backend(compiler_config=cfg)
        return torch.compile(fn, backend=backend), "compile"
    except Exception as e:
        print(f"  ⚠ [industrial] {op}/{mode}: torchair 不可用 ({str(e)[:100]}) → eager 兜底", flush=True)
        return fn, "eager"


def _build_fn(op, mode, sh):
    """返回 (forward_fn, actual_mode) — 各 mode 的 forward 构造 (供 Event 计时用).
    compile=TorchAir GE 融合; cann-fused=直接 CANN 融合算子(失败回退 compile); fa=CANN FA; eager=aclnn 厂商."""
    import torch
    actual = mode
    fn = _make_forward(op, sh)
    if mode == "compile":
        fn, actual = _compile_torchair(fn, op, mode)
    elif mode == "cann-fused":
        cf = _make_cann_fused_forward(op, sh)
        if cf is not None:
            fn, actual = cf, "cann-fused"
        else:
            fn, actual = _compile_torchair(fn, op, mode)   # 回退 GE 图融合
            actual = "cann-fused→" + actual
    elif mode == "fa":
        actual = "fa"
    return fn, actual


def _run_loop(op, mode, sh, warmup, measure):
    """warmup + measure 次 forward (供需要跑一遍的场景; 不含计时)."""
    fn, _ = _build_fn(op, mode, sh)
    import torch
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for _ in range(measure):
        fn()
    torch.npu.synchronize()


def main():
    p = argparse.ArgumentParser(description="工业级基准 (Event 设备侧端到端)")
    p.add_argument("op", type=str, help="matmul/matmul_relu/matmul_transpose/rms_norm/layernorm/"
                                        "sigmoid/softmax/vector_add/fused_add_mul/flash_attention/conv2d/conv_bias_relu")
    p.add_argument("--mode", type=str, default="eager", choices=["eager", "compile", "cann-fused", "fa"])
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")))
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    sh = _shapes(args.op)
    flops = _flops(args.op, sh)
    out_json = _BENCH_DIR / "outputs" / f"industrial_{args.op}_{args.mode}_tflops.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # ★Event 设备侧计时 (工业级, 无 msprof profiler 扰动):
    #   build fn (各 mode 实际执行的 forward) → warmup → Event 窗口包 measure 次 → ÷measure
    import torch
    fn, actual = _build_fn(args.op, args.mode, sh)
    for _ in range(args.warmup):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(args.measure):
        fn()
    e.record(); torch.npu.synchronize()
    e2e_us = s.elapsed_time(e) / args.measure * 1000.0   # ms→us, ÷measure = 单次
    data = {
        "tflops": round(flops / 1e12 / (e2e_us / 1e6), 2) if flops else None,
        "time_us": round(e2e_us, 1),
        "kernel_time_us": None,                          # Event 给不出纯kernel拆解 (要拆解走 msprof 诊断)
        "method": "event",                                # ★Event 设备侧 (工业级)
        "actual_mode": actual,                            # 实际执行 (compile 是否回退 eager)
        "op": args.op, "mode": args.mode,
        "warmup": args.warmup, "measure": args.measure,
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[industrial] {args.op}/{args.mode} (Event) → {out_json.name}: "
          f"e2e={round(e2e_us,1)}us actual={actual}")


if __name__ == "__main__":
    main()
