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
    "gm_read": {
        "desc": "GM 读带宽 (多尺寸×分块扫描)",
        "type": "read", "kernel_name": "read_kernel",
        "variants": [
            {"N": 1 << 22, "BLOCK": 1024},     # 16MB
            {"N": 1 << 23, "BLOCK": 1024},     # 32MB
            {"N": 1 << 23, "BLOCK": 2048},     # 32MB 大块
            {"N": 1 << 24, "BLOCK": 2048},     # 64MB
        ],
    },
    "gm_write": {
        "desc": "GM 写带宽 (多尺寸)",
        "type": "write", "kernel_name": "write_kernel",
        "variants": [
            {"N": 1 << 22, "BLOCK": 1024},
            {"N": 1 << 23, "BLOCK": 1024},
            {"N": 1 << 24, "BLOCK": 2048},
        ],
    },
    "gm_copy": {
        "desc": "GM 拷贝带宽 (读A写B, 双向)",
        "type": "copy", "kernel_name": "copy_kernel",
        "variants": [
            {"N": 1 << 22, "BLOCK": 1024},
            {"N": 1 << 23, "BLOCK": 2048},
        ],
    },
    "l2_read": {
        "desc": "L2 读带宽 (L2 内数组反复读)",
        "type": "l2", "kernel_name": "l2_read_kernel",
        "variants": [
            {"N": 1 << 20, "ITERS": 128, "BLOCK": 1024},   # 4MB
            {"N": 1 << 21, "ITERS": 64,  "BLOCK": 1024},   # 8MB
            {"N": 1 << 22, "ITERS": 32,  "BLOCK": 1024},   # 16MB
        ],
    },
    "cube": {
        "desc": "cube 算力 (fp16/fp32 × 尺寸×分块扫描) + per-path feed",
        "type": "mm", "kernel_name": "mm_kernel",
        "variants": [
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 256, "BN": 128, "BK": 64},
            {"dtype": "float16", "M": 8192, "N": 8192, "K": 8192,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float32", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float32", "M": 8192, "N": 8192, "K": 8192,
             "BM": 128, "BN": 128, "BK": 64},
            {"dtype": "float16", "M": 4096, "N": 4096, "K": 4096,
             "BM": 128, "BN": 256, "BK": 64},
        ],
    },
    "vec": {
        "desc": "Vec 吞吐 (add/mul/fma) + per-path",
        "type": "vec", "kernel_name": "vec_kernel",
        "variants": [
            {"OP": 0, "N": 1 << 23, "BLOCK": 1024},   # add 8M
            {"OP": 2, "N": 1 << 23, "BLOCK": 1024},   # fma 8M
            {"OP": 2, "N": 1 << 24, "BLOCK": 2048},   # fma 16M
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
