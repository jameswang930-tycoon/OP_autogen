#!/usr/bin/env python3
"""工业级基准 — 各算子用"真正的工业级优化实现"测端到端耗时 (Event 设备侧 或 msprof 纯 kernel).

═══ 用法 (910B 服务器; 完整运行命令见 bench_all.py 顶部) ═══
  python3 bench_industrial.py <op> [--mode eager|compile|fa] [--warmup-ms 25] [--rep-ms 100]
                                 [--n-buf 32] [--pipelined 10] [--msprof] [--measure 100]
  例:
    python3 bench_industrial.py matmul --mode compile    # MLP 融合链 compile
    python3 bench_industrial.py flash_attention --mode fa  # CANN FlashAttention
    python3 bench_industrial.py transformer_decoder_block --mode eager  # 复杂多算子链
    python3 bench_industrial.py rms_norm --mode eager --msprof   # msprof 纯 kernel 口径
  输出: bench_910b3/outputs/industrial_<op>_<mode>_tflops.json
        time_us(★median 或 msprof 纯kernel) / method / actual_mode / pipelined_n / kernel_time_us

═══ 测量方法 (2026-08-12 对齐 triton testing.do_bench) ═══
  Event 设备侧计时 (无 profiler 扰动), 每候选:
    - 时间预算自适应: 先 5 次估时长 → warmup 25ms / rep 100ms 折算次数
    - 多窗口 median: n_rep 个独立 Event 对 (设备流水连续, 最后 sync) → median
    - ★输入轮换破 L2: 连续 forward 同一批张量, 工作集<192MB(L2) 时后 N 次全 L2 命中虚高;
      Ascend 无清 L2 API → n_buf 组输入轮换 (组数x单组工作集 > L2) 等效 do_bench clear_cache
    - ★流水化 --pipelined N (默认10): 每窗口连续 N 次 ÷N → host 开销隐藏 ≈纯设备时间,
      与 verify/measure_final_event 同口径 (小算子对比用它)
  msprof 纯 kernel (--msprof): 包 msprof 跑 forward 循环 → op_summary Task Duration 求和 ÷次数
    = 纯 kernel 执行时间 (不含 host launch, 与 verify 的 ns 同源; 小算子不受 launch 开销污染)
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
        "batchnorm2d":     lambda: dict(N=int(os.environ.get("BN_N", 1)),
                                        C=int(os.environ.get("BN_C", 8)),
                                        H=int(os.environ.get("BN_H", 64)),
                                        W=int(os.environ.get("BN_W", 64))),
        "maxpool2d":       lambda: dict(N=int(os.environ.get("MP_N", 1)),
                                        C=int(os.environ.get("MP_C", 8)),
                                        H=int(os.environ.get("MP_H", 64)),
                                        W=int(os.environ.get("MP_W", 64)),
                                        KH=int(os.environ.get("MP_KH", 3)),
                                        KW=int(os.environ.get("MP_KW", 3)),
                                        SH=int(os.environ.get("MP_SH", 2)),
                                        SW=int(os.environ.get("MP_SW", 2)),
                                        PAD=int(os.environ.get("MP_PAD", 1))),
        "conv1d":          lambda: dict(N=int(os.environ.get("C1_N", 1)),
                                        CIN=int(os.environ.get("C1_CIN", 8)),
                                        L=int(os.environ.get("C1_L", 256)),
                                        COUT=int(os.environ.get("C1_COUT", 32)),
                                        KL=int(os.environ.get("C1_KL", 3))),
        # ★复杂多算子链 (KernelBench L2/L3 风格工业级基准, 2026-08-14 新增)
        "transformer_decoder_block": lambda: dict(S=int(os.environ.get("TDB_SEQ", 2048)),
                                                  D=int(os.environ.get("TDB_DIM", 1024)),
                                                  H=int(os.environ.get("TDB_HEADS", 8)),
                                                  HD=int(os.environ.get("TDB_HDIM", 64)),
                                                  FFN=int(os.environ.get("TDB_FFN", 4096))),
        "swiglu_mlp":      lambda: dict(S=int(os.environ.get("SM_SEQ", 2048)),
                                        D=int(os.environ.get("SM_DIM", 1024)),
                                        FFN=int(os.environ.get("SM_FFN", 4096))),
        "resnet_block":    lambda: dict(N=int(os.environ.get("RB_N", 8)),
                                        C=int(os.environ.get("RB_C", 64)),
                                        H=int(os.environ.get("RB_H", 64)),
                                        W=int(os.environ.get("RB_W", 64)),
                                        K=int(os.environ.get("RB_K", 64))),
        "batched_matmul":  lambda: dict(B=int(os.environ.get("BMM_B", 16)),
                                        M=int(os.environ.get("BMM_M", 512)),
                                        K=int(os.environ.get("BMM_K", 512)),
                                        N=int(os.environ.get("BMM_N", 512))),
        # ★工业界经典长链 (2026-08-14 新增第 2 批)
        "gqa_attention":   lambda: dict(S=int(os.environ.get("GQA_SEQ", 2048)),
                                        D=int(os.environ.get("GQA_DIM", 1024)),
                                        H=int(os.environ.get("GQA_HEADS", 16)),
                                        HD=int(os.environ.get("GQA_HDIM", 64)),
                                        KV=int(os.environ.get("GQA_KV", 4))),
        "mamba_block":     lambda: dict(L=int(os.environ.get("MB_LEN", 1024)),
                                        D=int(os.environ.get("MB_DIM", 1024)),
                                        N=int(os.environ.get("MB_SSM", 16)),
                                        ED=int(os.environ.get("MB_ED", 1024)),
                                        K=int(os.environ.get("MB_KC", 4))),
        "vit_block":       lambda: dict(S=int(os.environ.get("VIT_SEQ", 197)),
                                        D=int(os.environ.get("VIT_DIM", 768)),
                                        H=int(os.environ.get("VIT_HEADS", 12)),
                                        HD=int(os.environ.get("VIT_HDIM", 64)),
                                        FFN=int(os.environ.get("VIT_FFN", 3072))),
        "bert_block":      lambda: dict(S=int(os.environ.get("BERT_SEQ", 512)),
                                        D=int(os.environ.get("BERT_DIM", 768)),
                                        H=int(os.environ.get("BERT_HEADS", 12)),
                                        HD=int(os.environ.get("BERT_HDIM", 64)),
                                        FFN=int(os.environ.get("BERT_FFN", 3072))),
        "mixture_of_experts": lambda: dict(S=int(os.environ.get("MOE_SEQ", 1024)),
                                           D=int(os.environ.get("MOE_DIM", 1024)),
                                           E=int(os.environ.get("MOE_NEXP", 8)),
                                           FFN=int(os.environ.get("MOE_FFN", 2048))),
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
            return ((torch.randn(NB, C, H, W, device=npu, dtype=DT) * 0.1),
                    (torch.randn(K, C, R, Sd, device=npu, dtype=DT) * 0.1),
                    (torch.randn(K, device=npu, dtype=DT)) * 0.1)
        def fwd(x, w, b):
            return F.relu(F.conv2d(x, w, b, stride=1, padding=P))
        return fwd, [mk() for _ in range(n)]
    if op == "batchnorm2d":                # ★新算子: BatchNorm2d 推理 (按通道归一化, 与 input/batchnorm2d 对齐)
        N, C, H, W = sh["N"], sh["C"], sh["H"], sh["W"]
        eps = 1e-5
        def mk():
            return ((torch.randn(N, C, H, W, device=npu, dtype=DT) * 0.1),
                    (torch.randn(C, device=npu, dtype=DT) * 0.1),
                    (torch.rand(C, device=npu, dtype=DT) * 0.5 + 0.5),
                    (torch.randn(C, device=npu, dtype=DT) * 0.1 + 1.0),
                    (torch.randn(C, device=npu, dtype=DT) * 0.1))
        def fwd(x, rm, rv, g, b):
            return F.batch_norm(x, rm, rv, g, b, training=False, momentum=0.0, eps=eps)
        return fwd, [mk() for _ in range(n)]
    if op == "maxpool2d":                  # ★新算子: MaxPool2d (窗口 max, 与 input/maxpool2d 对齐)
        N, C, H, W = sh["N"], sh["C"], sh["H"], sh["W"]
        KH, KW, SH, SW, PAD = (sh["KH"], sh["KW"], sh["SH"], sh["SW"], sh["PAD"])
        def mk():
            return ((torch.randn(N, C, H, W, device=npu, dtype=DT) * 0.1),)
        def fwd(x):
            return F.max_pool2d(x, (KH, KW), stride=(SH, SW), padding=PAD)
        return fwd, [mk() for _ in range(n)]
    if op == "conv1d":                     # ★新算子: Conv1d (valid, 与 input/conv1d 对齐)
        N, CIN, L, COUT, KL = sh["N"], sh["CIN"], sh["L"], sh["COUT"], sh["KL"]
        def mk():
            return ((torch.randn(N, CIN, L, device=npu, dtype=DT) * 0.1),
                    (torch.randn(COUT, CIN, KL, device=npu, dtype=DT) * 0.1),
                    (torch.randn(COUT, device=npu, dtype=DT) * 0.1))
        def fwd(x, w, b):
            return F.conv1d(x, w, b)
        return fwd, [mk() for _ in range(n)]
    if op == "transformer_decoder_block":
        # ★复杂链 (LLaMA 风格 decoder layer, KernelBench L3 代表):
        #   RMSNorm → QKV投影 → RoPE → 多头注意力 → O投影 → 残差 → RMSNorm → SwiGLU FFN → 残差
        S, D, H, HD, FFN = sh["S"], sh["D"], sh["H"], sh["HD"], sh["FFN"]
        scale = 1.0 / (HD ** 0.5)
        _freq = 1.0 / (10000 ** (torch.arange(0, HD, 2, device=npu).float() / HD))
        _t = torch.arange(S, device=npu).float()
        _cos = torch.cos(_t[:, None] * _freq[None, :]).repeat(1, 2)   # [S, HD]
        _sin = torch.sin(_t[:, None] * _freq[None, :]).repeat(1, 2)
        def _rope(x, cos, sin):
            x1 = x[..., : HD // 2]
            x2 = x[..., HD // 2:]
            return x * cos[:, None, :] + torch.cat([-x2, x1], dim=-1) * sin[:, None, :]
        def mk():
            return ((torch.randn(S, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),   # wq
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),   # wk
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),   # wv
                    (torch.randn(H * HD, D, device=npu, dtype=DT) * 0.1),   # wo
                    (torch.randn(D, FFN, device=npu, dtype=DT) * 0.1),      # wup
                    (torch.randn(D, FFN, device=npu, dtype=DT) * 0.1),      # wgate
                    (torch.randn(FFN, D, device=npu, dtype=DT) * 0.1),      # wdown
                    _cos, _sin)
        def fwd(x, wq, wk, wv, wo, wup, wgate, wdown, cos, sin):
            h = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            q = h @ wq; k = h @ wk; v = h @ wv
            q = q.view(S, H, HD); k = k.view(S, H, HD); v = v.view(S, H, HD)
            q = _rope(q, cos, sin); k = _rope(k, cos, sin)
            s = (q @ k.transpose(-1, -2)) * scale
            p = torch.softmax(s, dim=-1)
            o = (p @ v).reshape(S, H * HD) @ wo
            r = x + o
            h2 = r / torch.sqrt((r * r).mean(-1, keepdim=True) + 1e-6)
            u = h2 @ wup; g = h2 @ wgate
            return (F.silu(u) * g) @ wdown + r
        return fwd, [mk() for _ in range(n)]
    if op == "swiglu_mlp":
        # ★SwiGLU 门控 MLP (LLaMA FFN): y = (silu(x@W1) * (x@W2)) @ W3  — 3 matmul + 门控
        S, D, FFN = sh["S"], sh["D"], sh["FFN"]
        def mk():
            return ((torch.randn(S, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, FFN, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, FFN, device=npu, dtype=DT) * 0.1),
                    (torch.randn(FFN, D, device=npu, dtype=DT) * 0.1))
        def fwd(x, w1, w2, w3):
            return (F.silu(x @ w1) * (x @ w2)) @ w3
        return fwd, [mk() for _ in range(n)]
    if op == "resnet_block":
        # ★ResNet 残差块 (KernelBench L3 代表): Conv+BN+ReLU → Conv+BN → +残差 → ReLU
        N, C, H, W, K = sh["N"], sh["C"], sh["H"], sh["W"], sh["K"]
        eps = 1e-5
        def mk():
            return ((torch.randn(N, C, H, W, device=npu, dtype=DT) * 0.1),
                    (torch.randn(K, C, 3, 3, device=npu, dtype=DT) * 0.1),
                    (torch.randn(K, device=npu, dtype=DT) * 0.1),
                    (torch.randn(K, K, 3, 3, device=npu, dtype=DT) * 0.1),
                    (torch.randn(K, device=npu, dtype=DT) * 0.1),
                    (torch.rand(K, device=npu, dtype=DT) * 0.5 + 0.5),   # g1 (weight1, 正)
                    (torch.randn(K, device=npu, dtype=DT) * 0.1),        # bn1 (bias1)
                    (torch.rand(K, device=npu, dtype=DT) * 0.5 + 0.5),   # g2 (weight2, 正)
                    (torch.randn(K, device=npu, dtype=DT) * 0.1),        # bn2 (bias2)
                    (torch.rand(K, device=npu, dtype=DT) * 0.5 + 0.5),   # rm1 (mean1)
                    (torch.rand(K, device=npu, dtype=DT) * 0.5 + 0.5),   # rv1 (var1, ★必须正)
                    (torch.rand(K, device=npu, dtype=DT) * 0.5 + 0.5),   # rm2 (mean2)
                    (torch.rand(K, device=npu, dtype=DT) * 0.5 + 0.5))   # rv2 (var2, ★必须正)
        def fwd(x, w1, b1, w2, b2, g1, bn1, g2, bn2, rm1, rv1, rm2, rv2):
            y = F.relu(F.batch_norm(F.conv2d(x, w1, b1, padding=1),
                                    rm1, rv1, g1, bn1, training=False, eps=eps))
            y = F.batch_norm(F.conv2d(y, w2, b2, padding=1),
                             rm2, rv2, g2, bn2, training=False, eps=eps)
            return F.relu(y + x)
        return fwd, [mk() for _ in range(n)]
    if op == "batched_matmul":
        # ★Batched Matmul (工业级多 batch GEMM): c[b] = a[b] @ b[b]
        B, M, K, N = sh["B"], sh["M"], sh["K"], sh["N"]
        def mk():
            return ((torch.randn(B, M, K, device=npu, dtype=DT) * 0.1),
                    (torch.randn(B, K, N, device=npu, dtype=DT) * 0.1))
        def fwd(a, b):
            return torch.bmm(a, b)
        return fwd, [mk() for _ in range(n)]
    if op == "gqa_attention":
        # ★GQA + RoPE (LLaMA/DeepSeek 系推理核心): QKV投影 → RoPE → 分组注意力 → O投影 → 残差
        S, D, H, HD, KV = sh["S"], sh["D"], sh["H"], sh["HD"], sh["KV"]
        scale = 1.0 / (HD ** 0.5)
        _freq = 1.0 / (10000 ** (torch.arange(0, HD, 2, device=npu).float() / HD))
        _t = torch.arange(S, device=npu).float()
        _cos = torch.cos(_t[:, None] * _freq[None, :]).repeat(1, 2)
        _sin = torch.sin(_t[:, None] * _freq[None, :]).repeat(1, 2)
        def _rope(x, cos, sin):
            x1 = x[..., : HD // 2]
            x2 = x[..., HD // 2:]
            return x * cos[:, None, :] + torch.cat([-x2, x1], dim=-1) * sin[:, None, :]
        def mk():
            return ((torch.randn(S, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),   # wq
                    (torch.randn(D, KV * HD, device=npu, dtype=DT) * 0.1),  # wk
                    (torch.randn(D, KV * HD, device=npu, dtype=DT) * 0.1),  # wv
                    (torch.randn(H * HD, D, device=npu, dtype=DT) * 0.1),   # wo
                    _cos, _sin)
        def fwd(x, wq, wk, wv, wo, cos, sin):
            q = (x @ wq).view(S, H, HD)
            k = (x @ wk).view(S, KV, HD)
            v = (x @ wv).view(S, KV, HD)
            q = _rope(q, cos, sin)
            k = _rope(k, cos, sin)
            k = k.repeat_interleave(H // KV, dim=1)     # GQA: 每组 KV 复制给 H/KV 个 Q 头
            v = v.repeat_interleave(H // KV, dim=1)
            s = (q @ k.transpose(-1, -2)) * scale
            p = torch.softmax(s, dim=-1)
            return (p @ v).reshape(S, H * HD) @ wo + x   # 残差
        return fwd, [mk() for _ in range(n)]
    if op == "mamba_block":
        # ★Mamba 块 (KernelBench L3 点名): 门控投影 → 深度卷积 → SSM 时序扫描 → 输出投影 → 残差
        L, D, N, ED, K = sh["L"], sh["D"], sh["N"], sh["ED"], sh["K"]
        DTS = N + N                                   # dt + B + C 段宽 (简化: dt 与 B 共用 2N)
        def mk():
            return ((torch.randn(L, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, ED + 3 * N, device=npu, dtype=DT) * 0.1),   # in_proj
                    (torch.randn(3 * N, 1, K, device=npu, dtype=DT) * 0.1),     # conv_w (depthwise)
                    (torch.randn(3 * N, device=npu, dtype=DT) * 0.1),           # conv_b
                    (torch.randn(N, device=npu, dtype=DT) * 0.1 - 3.0),         # A_log (负)
                    (torch.randn(N, device=npu, dtype=DT) * 0.1),               # dt_bias
                    (torch.randn(ED + N, device=npu, dtype=DT) * 0.1),          # D 乘性
                    (torch.randn(ED, D, device=npu, dtype=DT) * 0.1))           # out_proj
        def _scan(dA, dB):
            # 对角 SSM 关联扫描 (log 域, clamp 防溢出): h_t = dA_t·h_{t-1} + dB_t
            log_dA = torch.log(dA.clamp_min(1e-8))
            logP = torch.cumsum(log_dA, dim=0)
            w = torch.cumsum(dB * torch.exp((-logP).clamp_min(-30)), dim=0) \
                * torch.exp(logP.clamp_max(30))
            return w
        def fwd(x, in_proj, conv_w, conv_b, A_log, dt_bias, Dm, out_proj):
            z_and_x = x @ in_proj                       # [L, ED + 3N]
            z = z_and_x[:, :ED]
            xb = z_and_x[:, ED:]
            xb = F.conv1d(xb.unsqueeze(0).transpose(1, 2), conv_w, conv_b,
                          groups=3 * N, padding=K - 1)[..., :L].transpose(1, 2).squeeze(0)
            dt = F.softplus(xb[:, :N] + dt_bias)        # [L, N]
            B = xb[:, N:2 * N]
            C = xb[:, 2 * N:]
            A = -torch.exp(A_log)                       # [N]
            dA = torch.exp(dt * A)                      # [L, N] 每步衰减
            dB = dt * B
            h = _scan(dA, dB)                           # [L, N]
            y = (h * C).sum(-1, keepdim=True) * Dm[:N].sum(-1, keepdim=True) + z   # 乘性 D + 门控
            return y @ out_proj + x[:, :ED]             # 残差
        return fwd, [mk() for _ in range(n)]
    if op == "vit_block":
        # ★ViT encoder 块: LN → MHA → 残差 → LN → GELU MLP → 残差
        S, D, H, HD, FFN = sh["S"], sh["D"], sh["H"], sh["HD"], sh["FFN"]
        scale = 1.0 / (HD ** 0.5)
        def mk():
            return ((torch.randn(S, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),
                    (torch.randn(H * HD, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, FFN, device=npu, dtype=DT) * 0.1),
                    (torch.randn(FFN, device=npu, dtype=DT) * 0.1),
                    (torch.randn(FFN, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, device=npu, dtype=DT) * 0.1))
        def fwd(x, wq, wk, wv, wo, w1, b1, w2, b2):
            h = F.layer_norm(x, [D])
            q = (h @ wq).view(S, H, HD)
            k = (h @ wk).view(S, H, HD)
            v = (h @ wv).view(S, H, HD)
            s = (q @ k.transpose(-1, -2)) * scale
            p = torch.softmax(s, dim=-1)
            o = (p @ v).reshape(S, H * HD) @ wo + x
            h2 = F.layer_norm(o, [D])
            return F.gelu(h2 @ w1 + b1) @ w2 + o
        return fwd, [mk() for _ in range(n)]
    if op == "bert_block":
        # ★BERT encoder 块 (post-LN): MHA → LN → GELU MLP → LN → 残差 (BERT 原始结构)
        S, D, H, HD, FFN = sh["S"], sh["D"], sh["H"], sh["HD"], sh["FFN"]
        scale = 1.0 / (HD ** 0.5)
        def mk():
            return ((torch.randn(S, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, H * HD, device=npu, dtype=DT) * 0.1),
                    (torch.randn(H * HD, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, FFN, device=npu, dtype=DT) * 0.1),
                    (torch.randn(FFN, device=npu, dtype=DT) * 0.1),
                    (torch.randn(FFN, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, device=npu, dtype=DT) * 0.1))
        def fwd(x, wq, wk, wv, wo, w1, b1, w2, b2):
            q = (x @ wq).view(S, H, HD)
            k = (x @ wk).view(S, H, HD)
            v = (x @ wv).view(S, H, HD)
            s = (q @ k.transpose(-1, -2)) * scale
            p = torch.softmax(s, dim=-1)
            a = (p @ v).reshape(S, H * HD) @ wo
            h1 = F.layer_norm(x + a, [D])
            m = F.gelu(h1 @ w1 + b1) @ w2 + b2
            return F.layer_norm(m + h1, [D])
        return fwd, [mk() for _ in range(n)]
    if op == "mixture_of_experts":
        # ★MoE 块: router topk → E 个 expert FFN (SwiGLU) → topk 加权合并
        S, D, E, FFN = sh["S"], sh["D"], sh["E"], sh["FFN"]
        TOPK = 2
        def mk():
            return ((torch.randn(S, D, device=npu, dtype=DT) * 0.1),
                    (torch.randn(D, E, device=npu, dtype=DT) * 0.1),        # router
                    (torch.randn(D, E * FFN, device=npu, dtype=DT) * 0.1),  # w1f (up)
                    (torch.randn(D, E * FFN, device=npu, dtype=DT) * 0.1),  # w2f (gate)
                    (torch.randn(E, FFN, D, device=npu, dtype=DT) * 0.1))   # w3 (down) [E,FFN,D]
        def fwd(x, router_w, w1f, w2f, w3):
            logits = x @ router_w                        # [S, E]
            topk_val, topk_idx = logits.topk(TOPK, dim=-1)
            w = F.softmax(topk_val, dim=-1)              # [S, TOPK]
            u = (x @ w1f).view(S, E, FFN)
            g = (x @ w2f).view(S, E, FFN)
            act = F.silu(u) * g
            # ★每 expert 独立 down 投影: [E,S,FFN] @ [E,FFN,D] → [E,S,D] → [S,E,D]
            y = torch.bmm(act.transpose(0, 1), w3).transpose(0, 1)
            idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
            y_topk = torch.gather(y, 1, idx)             # [S, TOPK, D]
            return (y_topk * w.unsqueeze(-1)).sum(1)     # topk 加权合并
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
    if op == "batchnorm2d":
        return 6 * sh["N"] * sh["C"] * sh["H"] * sh["W"]   # 减/除/乘/加 x2 元素
    if op == "maxpool2d":
        return (sh["KH"] * sh["KW"] - 1) * sh["N"] * sh["C"] * \
            (((sh["H"] + 2 * sh["PAD"] - sh["KH"]) // sh["SH"] + 1) *
             ((sh["W"] + 2 * sh["PAD"] - sh["KW"]) // sh["SW"] + 1))   # 每次 max 比较
    if op == "conv1d":
        LOUT = sh["L"] - sh["KL"] + 1
        return 2 * sh["N"] * sh["COUT"] * LOUT * sh["CIN"] * sh["KL"] + \
            2 * sh["N"] * sh["COUT"] * LOUT   # MAC + bias
    if op == "transformer_decoder_block":
        s, d, h, hd, ffn = sh["S"], sh["D"], sh["H"], sh["HD"], sh["FFN"]
        return (2 * s * (4 * d * h * hd)            # QKV + O 投影 (4 个 matmul)
                + 4 * s * s * h * hd                # S = Q@K^T + P@V
                + 2 * s * (2 * d * ffn + ffn * d)   # SwiGLU: up+gate(d→FFN) + down(FFN→d)
                + 4 * s * d * 2)                    # 2× RMSNorm (每元素 ~4 flop)
    if op == "swiglu_mlp":
        s, d, ffn = sh["S"], sh["D"], sh["FFN"]
        return 2 * s * d * ffn * 3                 # up/gate/down 三个 matmul
    if op == "resnet_block":
        n, c, h, w, k = sh["N"], sh["C"], sh["H"], sh["W"], sh["K"]
        oh = ow = h                                # padding=1, 3x3, stride=1
        f = 2 * n * k * oh * ow * (c * 9 + k * 9)  # 两个 conv (3x3)
        f += 6 * n * k * oh * ow * 2               # 2× BN (推理 ~6 flop/元素)
        f += 2 * n * k * oh * ow                   # 残差 add + relu
        return f
    if op == "batched_matmul":
        return 2 * sh["B"] * sh["M"] * sh["K"] * sh["N"]
    if op == "gqa_attention":
        s, d, h, hd, kv = sh["S"], sh["D"], sh["H"], sh["HD"], sh["KV"]
        return (2 * s * (d * h * hd + 2 * d * kv * hd + h * hd * d)   # QKV + O 投影
                + 4 * s * s * h * hd)                                 # S + PV
    if op == "mamba_block":
        l, d, n, ed, k = sh["L"], sh["D"], sh["N"], sh["ED"], sh["K"]
        return (2 * l * d * (ed + 3 * n)      # in_proj
                + 2 * l * 3 * n * k           # depthwise conv
                + 4 * l * n                   # scan
                + 2 * l * ed * d)             # out_proj
    if op == "vit_block":
        s, d, h, hd, ffn = sh["S"], sh["D"], sh["H"], sh["HD"], sh["FFN"]
        return 2 * s * (4 * d * h * hd) + 4 * s * s * h * hd \
            + 2 * s * (d * ffn + ffn * d) + 6 * s * d * 2
    if op == "bert_block":
        s, d, h, hd, ffn = sh["S"], sh["D"], sh["H"], sh["HD"], sh["FFN"]
        return 2 * s * (4 * d * h * hd) + 4 * s * s * h * hd \
            + 2 * s * (d * ffn + ffn * d) + 6 * s * d * 2
    if op == "mixture_of_experts":
        s, d, e, ffn = sh["S"], sh["D"], sh["E"], sh["FFN"]
        return 2 * s * d * e + 2 * s * e * (2 * d * ffn + ffn * d)   # router + E 个 SwiGLU FFN
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
                   help="轮换输入 buffer 组数 (破 L2 复用; 组数x单组工作集应 > L2 192MB)")
    p.add_argument("--pipelined", type=int, default=10, metavar="N",
                   help="流水化模式: 每窗口连续调用 N 次 /N (隐藏 host 下发开销, 近似纯设备时间; "
                        "与 verify/measure_final_event 同口径); 0=单次含 host 开销")
    p.add_argument("--msprof", action="store_true",
                   help="★msprof 纯 kernel 模式: 包 msprof 跑 app → op_summary 全部行 Task Duration "
                        "求和 /次数 = 纯 kernel 时间 (不含 host launch; 与我们 verify 的 ns 口径同源); "
                        "time_us 写纯 kernel 值, kernel_time_us 同值, method=msprof-kernel")
    p.add_argument("--measure", type=int, default=100,
                   help="msprof 模式: app 内部 forward 循环次数 (默认 100, 越多越稳)")
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

    # ── ★msprof 纯 kernel 模式 (与 verify 的 ns 口径同源: Task Duration 求和 /N) ──
    if args.msprof:
        from bench_910b3.bench_common import measure_pytorch_msprof
        _app_dir = _BENCH_DIR / "outputs"
        _app_dir.mkdir(parents=True, exist_ok=True)
        app_path = _app_dir / f"msprof_app_{args.op}_{args.mode}.py"
        # ★生成临时 app 脚本: import 本模块的 forward 构造 (闭包无法跨进程, 用模块函数重建)
        app_path.write_text(
            "#!/usr/bin/env python3\n"
            "# Auto-generated msprof app — forward 循环 (warmup + measure)\n"
            "import os, sys\n"
            f"sys.path.insert(0, {str(_BENCH_DIR.parent)!r})\n"
            "from bench_910b3.bench_industrial import _make_forward, _shapes\n"
            "import torch, torch_npu\n"
            f"_op = {args.op!r}\n"
            f"_mode = {args.mode!r}\n"
            "sh = _shapes(_op)\n"
            "fn, bufs = _make_forward(_op, sh, n_buf=64)\n"
            "if _mode == 'compile':\n"
            "    from bench_910b3.bench_industrial import _compile_torchair\n"
            "    fn, actual = _compile_torchair(fn, _op, _mode)\n"
            f"for _ in range(10):\n"
            "    fn(*bufs[0])\n"
            "torch.npu.synchronize()\n"
            f"for _ in range({args.measure}):\n"
            "    fn(*bufs[_ % len(bufs)])\n"
            "torch.npu.synchronize()\n",
            encoding="utf-8")
        m = measure_pytorch_msprof(
            f"{sys.executable or 'python3'} {app_path}", out_json, flops,
            measure=args.measure, warmup=10)
        if m is None:
            print(f"[industrial] {args.op}/{args.mode} ⚠ msprof 纯 kernel 测量失败 (欠采/无 kernel) → "
                  f"回退 Event 口径")
        else:
            print(f"[industrial] {args.op}/{args.mode} (msprof 纯 kernel) → {out_json.name}: "
                  f"kernel={m['kernel_time_us']:.1f}us /{m['measure']} rows={m.get('rows_measured')} "
                  f"kernels/遍={m.get('kernels_per_iter')} actual={actual}")
            return
    # ── ★Event 设备侧计时 (do_bench 同款): 时间预算自适应 + 多窗口 median + 轮换破 L2 ──
    from bench_910b3.bench_common import measure_event
    m = measure_event(lambda i: fn(*bufs[i % len(bufs)]),
                      warmup_ms=args.warmup_ms, rep_ms=args.rep_ms,
                      pipelined_n=args.pipelined)
    e2e_us = m["median_us"]
    data = {
        "tflops": round(flops / 1e12 / (e2e_us / 1e6), 2) if flops else None,
        "time_us": round(e2e_us, 1),                     # ★median (主值)
        "time_us_min": m["min_us"], "time_us_mean": m["mean_us"],
        "rep": m["rep"], "warmup": m["warmup"], "n_buf": len(bufs),
        "kernel_time_us": None,                          # Event 给不出纯kernel拆解 (要拆解走 msprof 诊断)
        "method": "event-pipelined" if args.pipelined and args.pipelined > 1 else "event",
        "pipelined_n": m["pipelined_n"],                 # 0=单次含 host; >1=流水化 /N
        "actual_mode": actual,                            # 实际执行 (compile 是否回退 eager)
        "op": args.op, "mode": args.mode,
        "note": "Event 多窗口median+输入轮换破L2(do_bench同款); "
                + ("pipelined=流水化/N 近似纯设备时间(含kernel间gap, 不含host调度等待)"
                   if args.pipelined and args.pipelined > 1
                   else "含 torch host 调度 (vs triton 纯kernel 口径偏严, 详见 ARCHITECTURE_DESIGN §6)"),
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[industrial] {args.op}/{args.mode} (Event{' pipelined/' + str(args.pipelined) if args.pipelined else ''}) "
          f"→ {out_json.name}: e2e(median)={round(e2e_us,1)}us min={m['min_us']}us "
          f"rep={m['rep']} n_buf={len(bufs)} actual={actual}")


if __name__ == "__main__":
    main()
