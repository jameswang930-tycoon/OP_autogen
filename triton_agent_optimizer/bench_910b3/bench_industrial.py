#!/usr/bin/env python3
"""工业级基准 — 各算子用"真正的工业级优化实现"测端到端耗时 (★Event 设备侧).

═══ 为什么 ═══
  跟 naive torch 比不够 — 要跟工业级天花板比. Ascend 910B 上的工业级实现:
    - matmul/conv/norm 单算子: torch eager 已走 CANN aclnn vendor kernel (工业级)
    - 融合链 (MLP/matmul+epilogue/逐元素): torch.compile + TorchAir 图模式 (算子融合)
    - attention: CANN FlashAttention (torch_npu.npu_prompt_flash_attention)
  → 我们的 triton kernel 端到端 vs 这些工业级实现的端到端.

═══ 用法 (910B 服务器) ═══
  python3 bench_industrial.py <op> [--mode eager|compile|fa] [--warmup-ms 25] [--rep-ms 100] [--n-buf 32]
  例:
    python3 bench_industrial.py matmul --mode compile    # MLP 融合链 compile
    python3 bench_industrial.py flash_attention --mode fa  # CANN FlashAttention
    python3 bench_industrial.py rms_norm --mode eager
  输出: bench_910b3/industrial_<op>_<mode>_tflops.json
        time_us(★median) / time_us_min / time_us_mean / rep / warmup / n_buf / actual_mode

═══ 测量方法 (2026-08-12 对齐 triton testing.do_bench) ═══
  Event 设备侧计时 (无 profiler 扰动), 每候选:
    - 时间预算自适应: 先 5 次估时长 → warmup 25ms / rep 100ms 折算次数
    - 多窗口 median: n_rep 个独立 Event 对 (设备流水连续, 最后 sync) → median
    - ★输入轮换破 L2: 连续 forward 同一批张量, 工作集<192MB(L2) 时后 N 次全 L2 命中虚高;
      Ascend 无清 L2 API → n_buf 组输入轮换 (组数×单组工作集 > L2) 等效 do_bench clear_cache
  口径声明: 工业级 = torch 全流程 (含 host 调度/内存分配); 我们 verify 的 Event = triton 纯
  kernel launch 链 → 大算子 (ms 级) 差异可忽略, 小算子 (us 级) 我们占便宜, 对比时声明.
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
#  ★2026-08-12 修复: forward 改带参 fwd(*bufs[i]) + n_buf 组输入轮换 —
#    破坏 L2 复用 (910B3 L2=192MB; 连续 forward 同一批张量后 N 次全 L2 命中 → 数字虚高),
#    与 triton do_bench 的 clear_cache 同效 (Ascend 无清 L2 API → 轮换输入替代).
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


def _make_forward(op, sh, n_buf=32):
    """返回 (fwd, bufs) — fwd(*bufs[i]) 带参 forward; bufs = n_buf 组输入 (测量轮换破 L2).
    ★带参设计: torch.compile(fwd) 编译成"输入=图参数"的单图, 换数据不换图 (闭包捕获会被烘焙)."""
    import torch
    import torch.nn.functional as F
    npu = torch.device("npu")
    DT = torch.float32
    n = n_buf
    if op == "matmul":                       # 两层 MLP: Y=GELU(X@W1+b1)@W2
        M, K, N, H = sh["M"], sh["K"], sh["N"], sh["H"]
        def mk():
            return ((torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(K, H, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(H, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(H, N, device=npu, dtype=DT) - 0.5) * 0.1)
        def fwd(x, w1, b1, w2):
            return torch.matmul(F.gelu(x @ w1 + b1, approximate="tanh"), w2)
        return fwd, [mk() for _ in range(n)]
    if op == "attention_mlp":                # 自注意力 + MLP (QKV→S→softmax→O→GELU→FC2→+残差)
        S, D, H = sh["S"], sh["D"], sh["H"]
        scale = 1.0 / (D ** 0.5)
        def mk():
            return ((torch.rand(S, D, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(D, D, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(D, D, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(D, D, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(D, H, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(H, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(H, D, device=npu, dtype=DT) - 0.5) * 0.1)
        def fwd(x, wq, wk, wv, w1, b1, w2):
            q = x @ wq; k = x @ wk; v = x @ wv
            s = (q @ k.t()) * scale
            p = torch.softmax(s, dim=-1)
            o = p @ v
            y = F.gelu(o @ w1 + b1, approximate="tanh")
            z = y @ w2
            return z + o
        return fwd, [mk() for _ in range(n)]
    if op == "rms_norm_residual":            # (x+res) → RMSNorm → *w
        M, N = sh["M"], sh["N"]
        eps = 1e-6
        def mk():
            return ((torch.randn(M, N, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(M, N, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(N, device=npu, dtype=DT)) * 0.1)
        def fwd(x, res, w):
            c = x + res
            return c / torch.sqrt((c * c).mean(-1, keepdim=True) + eps) * w
        return fwd, [mk() for _ in range(n)]
    if op == "matmul_relu":
        M, K, N = sh["M"], sh["K"], sh["N"]
        def mk():
            return ((torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(K, N, device=npu, dtype=DT) - 0.5) * 0.1)
        def fwd(x, w):
            return F.relu(x @ w)
        return fwd, [mk() for _ in range(n)]
    if op == "matmul_transpose":
        M, K, N = sh["M"], sh["K"], sh["N"]
        def mk():
            return ((torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(N, K, device=npu, dtype=DT) - 0.5) * 0.1)
        def fwd(a, b):
            return a @ b.t()
        return fwd, [mk() for _ in range(n)]
    if op == "rms_norm":
        M, N = sh["M"], sh["N"]
        eps = 1e-6
        def mk():
            return ((torch.randn(M, N, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(N, device=npu, dtype=DT)) * 0.1)
        def fwd(x, w):
            return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w
        return fwd, [mk() for _ in range(n)]
    if op == "layernorm":
        M, N = sh["M"], sh["N"]
        def mk():
            return ((torch.randn(M, N, device=npu, dtype=DT)) * 0.1,
                    torch.ones(N, device=npu, dtype=DT),
                    torch.zeros(N, device=npu, dtype=DT))
        def fwd(x, w, b):
            return F.layer_norm(x, [N], w, b, 1e-6)
        return fwd, [mk() for _ in range(n)]
    if op == "sigmoid":
        N = sh["N"]
        def mk():
            return ((torch.randn(N, device=npu, dtype=DT)) * 0.1,)
        def fwd(x):
            return torch.sigmoid(x)
        return fwd, [mk() for _ in range(n)]
    if op == "softmax":
        M, N = sh["M"], sh["N"]
        def mk():
            return ((torch.randn(M, N, device=npu, dtype=DT)) * 0.1,)
        def fwd(x):
            return torch.softmax(x, dim=-1)
        return fwd, [mk() for _ in range(n)]
    if op == "vector_add":
        N = sh["N"]
        def mk():
            return ((torch.randn(N, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(N, device=npu, dtype=DT)) * 0.1)
        def fwd(x, z):
            return x + z
        return fwd, [mk() for _ in range(n)]
    if op == "fused_add_mul":
        N = sh["N"]
        def mk():
            return ((torch.randn(N, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(N, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(N, device=npu, dtype=DT)) * 0.1)
        def fwd(x, z, w):
            return (x + z) * w
        return fwd, [mk() for _ in range(n)]
    if op == "flash_attention":
        # ★与 input/flash_attention/kernel_op.py 对齐: 输入 fp16 (910B3 FA 工业级即 fp16)
        B, S, NH, D = sh["B"], sh["S"], sh["NH"], sh["D"]
        scale = 1.0 / (D ** 0.5)
        DTF = torch.float16
        def mk():
            return ((torch.randn(B, NH, S, D, device=npu, dtype=DTF)) * 0.1,
                    (torch.randn(B, NH, S, D, device=npu, dtype=DTF)) * 0.1,
                    (torch.randn(B, NH, S, D, device=npu, dtype=DTF)) * 0.1)
        def fwd(q, k, v):
            import torch_npu
            return torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=NH, scale_value=scale, input_layout="BNSD")
        return fwd, [mk() for _ in range(n)]
    if op == "conv2d":
        NB, C, H, W, K, R, Sd, P = (sh["NB"], sh["C"], sh["H"], sh["W"],
                                     sh["K"], sh["R"], sh["Sd"], sh["P"])
        def mk():
            return ((torch.randn(NB, C, H, W, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(K, C, R, Sd, device=npu, dtype=DT)) * 0.1)
        def fwd(x, w):
            return F.conv2d(x, w, stride=1, padding=P)
        return fwd, [mk() for _ in range(n)]
    if op == "conv_bias_relu":
        NB, C, H, W, K, R, Sd, P = (sh["NB"], sh["C"], sh["H"], sh["W"],
                                     sh["K"], sh["R"], sh["Sd"], sh["P"])
        def mk():
            return ((torch.randn(NB, C, H, W, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(K, C, R, Sd, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(K, device=npu, dtype=DT)) * 0.1)
        def fwd(x, w, b):
            return F.relu(F.conv2d(x, w, b, stride=1, padding=P))
        return fwd, [mk() for _ in range(n)]
    raise ValueError(f"未知算子: {op}")


def _make_cann_fused_forward(op, sh):
    """CANN 融合算子直接调用 (aclnnFusedMatmul: bias+gelu_tanh/relu; FusedConvBiasRelu) —
    工业级"厂商融合"基准, 比 eager(分 kernel) 更优, 与 TorchAir compile(GE 图融合) 同源.
    ★2026-08-12: 改带参 fwd(*bufs[i]) + 返回 n_buf 组输入 (与 _make_forward 同款轮换破 L2).
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
        def mk():
            return ((torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(K, H, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(H, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(H, N, device=npu, dtype=DT) - 0.5) * 0.1)
        def fwd(x, w1, b1, w2):
            h = fused_mm(x, w1, b1)      # fc1+bias+gelu_tanh 融合 (1 kernel)
            return h @ w2                # fc2 普通 matmul
        return fwd, [mk() for _ in range(32)]
    if op == "matmul_relu" and fused_mm:
        M, K, N = sh["M"], sh["K"], sh["N"]
        def mk():
            return ((torch.rand(M, K, device=npu, dtype=DT) - 0.5) * 0.1,
                    (torch.rand(K, N, device=npu, dtype=DT) - 0.5) * 0.1,
                    torch.zeros(N, device=npu, dtype=DT))
        def fwd(x, w, b):
            return fused_mm(x, w, b)          # y = relu(x@w + b) (fusedOpType=relu)
        return fwd, [mk() for _ in range(32)]
    if op == "conv_bias_relu" and fused_conv:
        NB, C, H, W, K, R, Sd, P = (sh["NB"], sh["C"], sh["H"], sh["W"],
                                     sh["K"], sh["R"], sh["Sd"], sh["P"])
        def mk():
            return ((torch.randn(NB, C, H, W, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(K, C, R, Sd, device=npu, dtype=DT)) * 0.1,
                    (torch.randn(K, device=npu, dtype=DT)) * 0.1)
        def fwd(x, w, b):
            return fused_conv(x, w, b, stride=1, padding=P)
        return fwd, [mk() for _ in range(32)]
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


def main():
    p = argparse.ArgumentParser(description="工业级基准 (Event 设备侧端到端, 多窗口median+轮换破L2)")
    p.add_argument("op", type=str, help="matmul/matmul_relu/matmul_transpose/rms_norm/layernorm/"
                                        "sigmoid/softmax/vector_add/fused_add_mul/flash_attention/conv2d/conv_bias_relu")
    p.add_argument("--mode", type=str, default="eager", choices=["eager", "compile", "cann-fused", "fa"])
    p.add_argument("--warmup-ms", type=int, default=int(os.environ.get("BENCH_WARMUP_MS", "25")),
                   help="warmup 时间预算 (ms, do_bench 同款; 按估时长折算次数)")
    p.add_argument("--rep-ms", type=int, default=int(os.environ.get("BENCH_REP_MS", "100")),
                   help="测量时间预算 (ms, do_bench 同款; 折算成 n_rep 个独立 Event 对)")
    p.add_argument("--n-buf", type=int, default=32,
                   help="轮换输入 buffer 组数 (破 L2 复用; 组数×单组工作集应 > L2 192MB)")
    args = p.parse_args()
    sh = _shapes(args.op)
    flops = _flops(args.op, sh)
    out_json = _BENCH_DIR / "outputs" / f"industrial_{args.op}_{args.mode}_tflops.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # ── 构造带参 forward + n_buf 组输入 (轮换破 L2 复用) ──
    import torch
    n_buf = 64 if args.op == "flash_attention" else args.n_buf   # FA 工作集小 → 更多组
    fn, bufs = _make_forward(args.op, sh, n_buf=n_buf)
    actual = args.mode
    if args.mode == "compile":
        fn, actual = _compile_torchair(fn, args.op, args.mode)
    elif args.mode == "cann-fused":
        cf = _make_cann_fused_forward(args.op, sh)
        if cf is not None:
            fn, bufs = cf
            actual = "cann-fused"
        else:
            fn, actual = _compile_torchair(fn, args.op, args.mode)   # 回退 GE 图融合
            actual = "cann-fused→" + actual
    elif args.mode == "fa":
        actual = "fa"

    # ── ★Event 设备侧计时 (do_bench 同款): 时间预算自适应 + 多窗口 median + 轮换破 L2 ──
    from bench_910b3.bench_common import measure_event
    m = measure_event(lambda i: fn(*bufs[i % len(bufs)]),
                      warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    e2e_us = m["median_us"]
    data = {
        "tflops": round(flops / 1e12 / (e2e_us / 1e6), 2) if flops else None,
        "time_us": round(e2e_us, 1),                     # ★median (主值)
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": len(bufs),
        "kernel_time_us": None,                          # Event 给不出纯kernel拆解 (要拆解走 msprof 诊断)
        "method": "event",                                # ★Event 设备侧 (工业级)
        "actual_mode": actual,                            # 实际执行 (compile 是否回退 eager)
        "op": args.op, "mode": args.mode,
        "note": "Event 多窗口median+输入轮换破L2(do_bench同款); 含 torch host 调度 "
                "(vs triton 纯kernel 口径偏严, 详见 ARCHITECTURE_DESIGN §6)",
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[industrial] {args.op}/{args.mode} (Event) → {out_json.name}: "
          f"e2e(median)={round(e2e_us,1)}us min={m['min_us']}us "
          f"rep={m['rep']} n_buf={len(bufs)} actual={actual}")


if __name__ == "__main__":
    main()
