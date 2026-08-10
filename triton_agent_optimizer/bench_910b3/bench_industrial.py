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


def _set_actual(mode_str: str):
    """把"实际执行模式"写进文件 (内层进程写, 外层读回 → json 记录, 让用户分辨 compile 是否真编译了)."""
    try:
        f = os.environ.get("_INDUSTRIAL_ACTUAL_FILE")
        if f:
            Path(f).write_text(mode_str, encoding="utf-8")
    except Exception:
        pass


def _run_loop(op, mode, sh, warmup, measure):
    """内层 (msprof 下): warmup + measure 次 forward.
    compile 用 TorchAir (GE 图融合 → CANN 融合 kernel); cann-fused 直接调 CANN 融合算子 (失败回退 compile)."""
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
            fn, actual = _compile_torchair(fn, op, mode)   # 回退 GE 图融合 (同批 CANN 融合 kernel)
            actual = "cann-fused→" + actual               # 记录实际 (cann-fused→compile / →eager)
    elif mode == "fa":
        actual = "fa"   # flash_attention 的 forward 已是 CANN FA
    # warmup + measure
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for _ in range(measure):
        fn()
    torch.npu.synchronize()
    _set_actual(actual)   # ★记录实际执行模式 (外层读回 → json) — 让"compile 没融合"可分辨


def main():
    p = argparse.ArgumentParser(description="工业级基准 (msprof 端到端)")
    p.add_argument("op", type=str, help="matmul/matmul_relu/matmul_transpose/rms_norm/layernorm/"
                                        "sigmoid/softmax/vector_add/fused_add_mul/flash_attention/conv2d/conv_bias_relu")
    p.add_argument("--mode", type=str, default="eager", choices=["eager", "compile", "cann-fused", "fa"])
    p.add_argument("--measure", type=int, default=int(os.environ.get("BENCH_PT_MEASURE", "30")))
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    sh = _shapes(args.op)
    flops = _flops(args.op, sh)

    if os.environ.get("_INDUSTRIAL_IN_MSPROF") != "1":
        # ── 外层: 一次 msprof 包内层 → 同时算 端到端/纯kernel → 写 json ──
        # ★必须先设标记再调 measure_pytorch_msprof: 它用 dict(os.environ,...) 传给 msprof,
        #   msprof 重启本脚本时才能拿到 _INDUSTRIAL_IN_MSPROF=1 走"内层"路径 (否则无限套娃 msprof).
        os.environ["_INDUSTRIAL_IN_MSPROF"] = "1"
        # ★actual 文件: 内层进程写实际执行模式 (compile 是否回退 eager), 外层读回写进 json
        _am_file = _BENCH_DIR / "outputs" / f".actual_{args.op}_{args.mode}.txt"
        try:
            _am_file.unlink(missing_ok=True)
        except Exception:
            pass
        os.environ["_INDUSTRIAL_ACTUAL_FILE"] = str(_am_file)
        from bench_910b3.bench_common import measure_pytorch_msprof
        import subprocess
        app_cmd = (f"python3 {Path(__file__).resolve()} {args.op} --mode {args.mode} "
                   f"--measure {args.measure} --warmup {args.warmup}")
        out_json = _BENCH_DIR / "outputs" / f"industrial_{args.op}_{args.mode}_tflops.json"
        result = measure_pytorch_msprof(app_cmd, out_json, flops,
                                        measure=args.measure, warmup=args.warmup,
                                        extras={"op": args.op, "mode": args.mode})
        # 读回实际模式 → 追加进 json (让"compile 没融合"可分辨: actual_mode=eager 说明 torchair 回退)
        if result is not None:
            try:
                if _am_file.exists():
                    result["actual_mode"] = _am_file.read_text(encoding="utf-8").strip()
                    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception:
                pass
            print(f"[industrial] {args.op}/{args.mode} → {out_json.name}: "
                  f"e2e={result['time_us']}us kernel={result.get('kernel_time_us')}us "
                  f"actual={result.get('actual_mode', args.mode)}")
            return
        # msprof 失败 → 兜底 Event 计时; ★若已有旧 msprof 好结果则保留, 不覆盖 (否则好数据被差数据顶掉)
        old = None
        if out_json.exists():
            try:
                old = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                old = None
        if old and old.get("time_us") and old.get("method") == "msprof":
            print(f"  ⚠ msprof 不可用, 但 {out_json.name} 已有 msprof 结果 ({old['time_us']}us) → 保留, 不覆盖")
            return
        print("  ⚠ msprof 不可用 → 兜底 Event 计时")
        import torch
        _run_loop(args.op, args.mode, sh, args.warmup, args.measure)
        torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
        s.record()
        fn = _make_forward(args.op, sh)
        for _ in range(args.measure):
            fn()
        e.record(); torch.npu.synchronize()
        e2e_us = s.elapsed_time(e) / args.measure * 1000.0
        out_json.write_text(json.dumps({
            "tflops": round(flops / 1e12 / (e2e_us / 1e6), 2) if flops else None,
            "time_us": round(e2e_us, 1), "kernel_time_us": None,
            "method": "event", "op": args.op, "mode": args.mode,
            "warmup": args.warmup, "measure": args.measure,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[industrial] {args.op}/{args.mode} (Event兜底) e2e={e2e_us:.1f}us → {out_json.name}")
        return

    # ── 内层: 在 msprof 下, 实际跑 forward ──
    import torch
    env = dict(os.environ, _INDUSTRIAL_IN_MSPROF="1")
    os.environ.update(env)
    _run_loop(args.op, args.mode, sh, args.warmup, args.measure)


if __name__ == "__main__":
    main()
