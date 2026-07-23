#!/usr/bin/env python3
"""
示例 Triton Kernel: Vector Addition

这是一个简单的 vadd kernel，用于演示融合分析流程。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vadd_kernel(
    x_ptr,  # 输入向量 x
    y_ptr,  # 输入向量 y
    out_ptr,  # 输出向量
    N,  # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """
    Vector Addition Kernel

    计算: out = x + y

    参数:
        x_ptr: 输入向量 x 的指针
        y_ptr: 输入向量 y 的指针
        out_ptr: 输出向量的指针
        N: 向量长度
        BLOCK_SIZE: 每个 block 处理的元素数量
    """
    # 获取 block id
    pid = tl.program_id(0)

    # 计算 block 起始位置
    block_start = pid * BLOCK_SIZE

    # 计算当前 block 处理的范围
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建 mask（防止越界）
    mask = offsets < N

    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 计算
    out = x + y

    # 存储结果
    tl.store(out_ptr + offsets, out, mask=mask)


def vadd_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    使用 Triton kernel 执行向量加法

    参数:
        x: 输入向量 (N,)
        y: 输入向量 (N,)

    返回:
        out: 输出向量 (N,)
    """
    # 确保输入在 NPU 上
    assert x.is_npu and y.is_npu
    assert x.shape == y.shape

    # 创建输出向量
    out = torch.empty_like(x)

    # 设置 grid 大小
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # 启动 kernel
    vadd_kernel[grid](
        x, y, out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def benchmark_vadd(N: int = 32768, dtype: str = 'fp16', repeat: int = 100):
    """
    基准测试 vadd kernel

    参数:
        N: 向量长度
        dtype: 数据类型 ('fp16', 'fp32', 'bf16')
        repeat: 重复次数
    """
    import time

    # 设置数据类型
    torch_dtype = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }[dtype]

    # 创建输入数据
    x = torch.randn(N, device='npu', dtype=torch_dtype)
    y = torch.randn(N, device='npu', dtype=torch_dtype)

    # Warmup
    for _ in range(10):
        _ = vadd_triton(x, y)
    torch.npu.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        out = vadd_triton(x, y)
    torch.npu.synchronize()
    end = time.perf_counter()

    # 计算性能
    total_time_ms = (end - start) / repeat * 1000
    total_bytes = N * 2 * torch.finfo(torch_dtype).bits // 8  # 2 个输入
    bandwidth_gb_s = total_bytes / (total_time_ms / 1000) / 1e9

    print(f"\n=== vadd Benchmark ===")
    print(f"N = {N}, dtype = {dtype}")
    print(f"Time: {total_time_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"=====================\n")

    return {
        'N': N,
        'dtype': dtype,
        'time_ms': total_time_ms,
        'bandwidth_gb_s': bandwidth_gb_s,
    }


if __name__ == "__main__":
    # 检查 NPU 是否可用
    if not torch.npu.is_available():
        print("错误: NPU 不可用")
        print("请在华为昇腾 910B3 服务器上运行")
        import sys
        sys.exit(1)

    # 运行基准测试
    benchmark_vadd(N=32768, dtype='fp16')
    benchmark_vadd(N=65536, dtype='fp16')
    benchmark_vadd(N=131072, dtype='fp16')