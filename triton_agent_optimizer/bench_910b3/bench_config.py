#!/usr/bin/env python3
"""bench 配置注册表 + 静态字节/算力计算 — 无 triton 依赖, 任意机器可检查.

bench_kernels.py (triton kernels, 服务器) 和 run_bench.py (实测, 服务器) 都从这里取配置.
"""
_KB = 1024

# ═══════════════════════════════════════════════════════════════════════
#  变体注册表 — 每类多个变体 (尺寸×分块×精度), 取最大
#  字段: type (kernel 类型), kernel_name (msprof op 用), variants[] (参数)
#  bytes/flops 由 variant_bytes_flops 静态算, 不重复跑 kernel (避免多余 launch)
# ═══════════════════════════════════════════════════════════════════════

BENCHES = {
    # ★流式 bench: 固定 N 的 BLOCK 扫描 (512→16384) — 展示"带宽随 BLOCK 爬升→变平"的饱和曲线
    #   小 BLOCK: 每 program 固定开销主导 (program数=N/BLOCK 多), 带宽≈BLOCK → 翻倍翻倍属正常
    #   大 BLOCK: 数据搬运主导, 带宽收敛于峰值 (取这个为 GM 峰值)
    #   UB 上限 192KB: read/write 单块 ≤49152 元素; copy 双缓冲 ≤24576; 保守取 16384 全兼容
    # ★L2 污染修复 (2026-08): 工作集必须 >> L2 (192MB), 否则 30 次重复读命中 L2 → 测到 L2 带宽
    #   fp32: 1<<27 = 512MB (2.7×L2), 1<<28 = 1GB (5.3×L2) — 保证真 GM
    "gm_read": {
        "desc": "GM 读带宽 (512MB~1GB 固定, BLOCK 512→32768 扫描)",
        "type": "read", "kernel_name": "read_kernel",
        "variants": [
            {"N": 1 << 27, "BLOCK": 512},      # 512MB, 小分块 (未饱和)
            {"N": 1 << 27, "BLOCK": 2048},
            {"N": 1 << 27, "BLOCK": 8192},
            {"N": 1 << 27, "BLOCK": 16384},    # 64KB/块
            {"N": 1 << 28, "BLOCK": 8192},     # 1GB (5.3×L2) 确认饱和
            {"N": 1 << 28, "BLOCK": 16384},
            {"N": 1 << 28, "BLOCK": 32768},    # 128KB/块 (UB 192KB 内; 若报错说明到编译器/UB 极限)
        ],
    },
    "gm_write": {
        "desc": "GM 写带宽 (512MB~1GB 固定, BLOCK 512→32768)",
        "type": "write", "kernel_name": "write_kernel",
        "variants": [
            {"N": 1 << 27, "BLOCK": 512},
            {"N": 1 << 27, "BLOCK": 2048},
            {"N": 1 << 27, "BLOCK": 8192},
            {"N": 1 << 27, "BLOCK": 16384},
            {"N": 1 << 28, "BLOCK": 8192},
            {"N": 1 << 28, "BLOCK": 16384},
            {"N": 1 << 28, "BLOCK": 32768},
        ],
    },
    "gm_copy": {
        "desc": "GM 拷贝带宽 (512MB~1GB 固定, BLOCK 512→16384; 读A写B 双向)",
        "type": "copy", "kernel_name": "copy_kernel",
        "variants": [
            {"N": 1 << 27, "BLOCK": 512},
            {"N": 1 << 27, "BLOCK": 2048},
            {"N": 1 << 27, "BLOCK": 8192},
            {"N": 1 << 27, "BLOCK": 16384},
            {"N": 1 << 28, "BLOCK": 8192},
            {"N": 1 << 28, "BLOCK": 16384},
        ],
    },
    "l2_read": {
        "desc": "L2 读带宽 (8MB 固定, BLOCK 1024→8192 扫描 — 补上之前遗漏的 BLOCK 维度)",
        "type": "l2", "kernel_name": "l2_read_kernel",
        "variants": [
            {"N": 1 << 21, "ITERS": 64, "BLOCK": 1024},
            {"N": 1 << 21, "ITERS": 64, "BLOCK": 2048},
            {"N": 1 << 21, "ITERS": 64, "BLOCK": 4096},
            {"N": 1 << 21, "ITERS": 64, "BLOCK": 8192},
        ],
    },
    "cube": {
        "desc": "cube 算力 (fp16/fp32 × 尺寸×分块扫描, 含大块 512×64) + per-path feed",
        "type": "mm", "kernel_name": "mm_kernel",
        "variants": [
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 256, "BN": 128, "BK": 64},
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 512, "BN": 64, "BK": 64},          # 大 M 块 (L0C 512×64×4=128KB 上限)
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 256, "BN": 128, "BK": 128},        # 大 K 块 (L0A 256×128×2=64KB 上限)
            {"dtype": "float16", "M": 8192, "N": 8192, "K": 8192,
             "BM": 256, "BN": 128, "BK": 64},
            {"dtype": "float32", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float32", "M": 4096, "N": 4096, "K": 4096,
             "BM": 256, "BN": 128, "BK": 64},         # fp32 大块 (L0A 256×64×4=64KB 上限)
            {"dtype": "float32", "M": 8192, "N": 8192, "K": 8192,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 256, "BK": 64},
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 256, "BK": 128},       # L0B=256×128×2=64KB 上限
            {"dtype": "float32", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 128, "BK": 128},       # fp32 L0A=128×128×4=64KB 上限
        ],
    },
    "vec": {
        "desc": "Vec 吞吐 (add/fma, BLOCK 1024→16384) + per-path",
        "type": "vec", "kernel_name": "vec_kernel",
        "variants": [
            {"OP": 0, "N": 1 << 23, "BLOCK": 1024},   # add 8M
            {"OP": 0, "N": 1 << 23, "BLOCK": 4096},
            {"OP": 2, "N": 1 << 23, "BLOCK": 1024},   # fma 8M
            {"OP": 2, "N": 1 << 23, "BLOCK": 4096},
            {"OP": 2, "N": 1 << 24, "BLOCK": 8192},   # fma 16M, 96KB
            {"OP": 2, "N": 1 << 24, "BLOCK": 16384},  # 192KB 满 (UB 极限, 可能报错=fail-fast)
        ],
    },
}


def _dt_bytes(dtype: str) -> int:
    return {"float32": 4, "float16": 2, "bfloat16": 2}.get(dtype, 4)


def variant_bytes_flops(btype: str, v: dict) -> tuple:
    """静态算 (总字节数, FLOPs) — 不跑 kernel. run_bench 用这个, 避免二次 launch.
    约定: 字节数 = 该操作的总搬运量 (读+写); FLOPs = 计算量 (非 mm 为 0)."""
    if btype == "read":
        return v["N"] * 4, 0
    if btype == "write":
        return v["N"] * 4, 0
    if btype == "copy":
        return 2 * v["N"] * 4, 0              # 读A + 写B
    if btype == "l2":
        return v["N"] * 4 * v["ITERS"], 0
    if btype == "mm":
        dtb = _dt_bytes(v["dtype"])
        m, n, k = v["M"], v["N"], v["K"]
        return (m * k + k * n + m * n) * dtb, 2 * m * n * k
    if btype == "vec":
        return 3 * v["N"] * 4, 0              # 读A + 读B + 写C
    raise ValueError(f"未知 bench 类型: {btype}")


# ═══════════════════════════════════════════════════════════════════════
#  PyTorch 基准 json 按算子映射 (bench_pytorch_*.py 输出文件名)
#  scheduler.py / trajectory_chart.py 用它找每个 op 的 torch 基准线
#  ★映射依据 kernel_op.py 的运算链, 非目录名直觉 (matmul 目录实为两层 MLP)
# ═══════════════════════════════════════════════════════════════════════
PT_BENCH_MAP = {
    "matmul": "pytorch_mlp_tflops.json",            # 两层 MLP: GELU(X@W1+b1)@W2
    "attention_mlp": "pytorch_attention_tflops.json",  # 自注意力+MLP
    "rms_norm": "pytorch_rms_norm_tflops.json",
    "flash_attention": "pytorch_flash_attention_tflops.json",
    "conv2d": "pytorch_conv2d_tflops.json",
    "conv_bias_relu": "pytorch_conv_bias_relu_tflops.json",
}
