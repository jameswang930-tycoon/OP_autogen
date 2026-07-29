#!/usr/bin/env python3
"""
向量加法 — Triton-Ascend 完整测试脚本
用法:
  1. 仅运行+验证:   python3 run_and_profile.py
  2. msprof 采集:   msprof op --application="python3 run_and_profile.py" \
                              --kernel-name=add_kernel \
                              --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
                              --output=./msprof_out
  3. HIVM dump:     TRITON_DEBUG=1 python3 run_and_profile.py
                    → ~/.triton/dump/<hash>/kernel.npuir.mlir
"""
import os
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    N = 1024
    BLOCK_SIZE = 128

    x = torch.randn(N, device='npu')
    y = torch.randn(N, device='npu')
    out = torch.empty(N, device='npu')

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    expected = x + y
    max_err = torch.max(torch.abs(out - expected)).item()
    print(f"N={N} BLOCK_SIZE={BLOCK_SIZE} max_error={max_err:.6f}", flush=True)

    if max_err < 1e-3:
        print("PASS", flush=True)
    else:
        print(f"FAIL: max_error={max_err}", flush=True)


if __name__ == "__main__":
    main()
