#!/usr/bin/env python3
"""
Softmax + GELU — Triton-Ascend 完整测试脚本 (多管线通路验证)

════════════════════════════════════════════════════════════════════════════
  服务器执行命令 (按顺序):
════════════════════════════════════════════════════════════════════════════

  # ── 1. 验证正确性 ──
  python3 run_and_profile.py

  # ── 2. 提取 HIVM MLIR ──
  export TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_CACHE=1
  python3 run_and_profile.py
  # 产出: ~/.triton/dump/<hash>/softmax_kernel.npuir.mlir
  #       ~/.triton/dump/<hash>/fused_gelu_kernel.npuir.mlir

  # ── 3. msprof 采集 softmax ──
  msprof op \
      --application="python3 run_and_profile.py" \
      --kernel-name=softmax_kernel \
      --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
      --output=./msprof_out_softmax
  # 产出: ./msprof_out_softmax/OPPROF_xxx/

  # ── 4. msprof 采集 gelu ──
  msprof op \
      --application="python3 run_and_profile.py" \
      --kernel-name=fused_gelu_kernel \
      --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
      --output=./msprof_out_gelu
  # 产出: ./msprof_out_gelu/OPPROF_xxx/

════════════════════════════════════════════════════════════════════════════
  HIVM MLIR → 分析流程 (后续步骤, 在 Windows 端执行):
════════════════════════════════════════════════════════════════════════════

  # 5. HIVM 解析
  python3 analyzers/hivmir_analyzer.py <path/to/kernel.npuir.mlir>
  # 产出: hivmir_report.json (11 字段)

  # 6. msprof trace 解析
  python3 analyzers/msprof_analyzer.py <path/to/OPPROF_xxx/>
  # 产出: pipeline_report.json (14 字段)

  # 7. 合并 → 29 字段
  python3 analyzers/dsl_merger.py hivmir_report.json pipeline_report.json
  # 产出: merged_report.json (29 字段)

  # 8. 瓶颈诊断 + 数据提取
  python3 analyzers/bottleneck_diagnoser.py merged_report.json
  python3 analyzers/data_extractor.py merged_report.json
  # → 注入 Planner prompt

════════════════════════════════════════════════════════════════════════════
"""
import os
import torch
import torch_npu
import torch.nn.functional as F
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 1: Online Softmax
#  Pipeline: load → max → sub → exp → sum → div
#  Ops: GM→UB(load) + VecUnit(reduce max) + VecUnit(sub) + VecUnit(exp)
#       + VecUnit(reduce sum) + VecUnit(div) + UB→GM(store)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Stage 1: Load (GM→UB)
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))

    # Stage 2: Find max (VecUnit reduce)
    x_max = tl.max(x, axis=0)

    # Stage 3: Subtract max + exp (VecUnit)
    x_safe = x - x_max
    x_exp = tl.math.exp(x_safe)

    # Stage 4: Sum (VecUnit reduce)
    x_sum = tl.sum(x_exp, axis=0)

    # Stage 5: Normalize (VecUnit div)
    x_softmax = x_exp / x_sum

    # Stage 6: Store (UB→GM)
    tl.store(out_ptr + offs, x_softmax, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2: GELU Activation
#  Pipeline: load → mul → mul → add → mul → tanh → add → mul → store
#  Ops: GM→UB(load) + VecUnit(x_sq/x_cu/inner/gelu chain) + UB→GM(store)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def fused_gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_sq = x * x
    x_cu = x_sq * x
    inner = 0.7978845608 * (x + 0.044715 * x_cu)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, gelu, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════
#  Test helpers
# ═══════════════════════════════════════════════════════════════════════════

def test_softmax(N: int = 1024, BLOCK_SIZE: int = 256):
    x = torch.randn(N, device='npu', dtype=torch.float32)

    # Triton result
    out_triton = torch.empty_like(x)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    softmax_kernel[grid](x, out_triton, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    # PyTorch reference
    out_torch = F.softmax(x, dim=0)

    max_err = torch.max(torch.abs(out_triton - out_torch)).item()
    status = "PASS" if max_err < 1e-3 else "FAIL"
    print(f"[softmax] N={N} BLK={BLOCK_SIZE} max_err={max_err:.6f}  {status}", flush=True)
    return status, max_err


def test_gelu(N: int = 1024, BLOCK_SIZE: int = 256):
    x = torch.randn(N, device='npu', dtype=torch.float32)

    # Triton result
    out_triton = torch.empty_like(x)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fused_gelu_kernel[grid](x, out_triton, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    # PyTorch reference
    out_torch = F.gelu(x, approximate="tanh")

    max_err = torch.max(torch.abs(out_triton - out_torch)).item()
    status = "PASS" if max_err < 1e-3 else "FAIL"
    print(f"[gelu]    N={N} BLK={BLOCK_SIZE} max_err={max_err:.6f}  {status}", flush=True)
    return status, max_err


def main():
    print("=== Triton-Ascend Softmax + GELU 测试 ===", flush=True)
    print(f"  NPU count: {torch.npu.device_count()}", flush=True)

    results = []

    # 多 shape 测试 (覆盖对齐/非对齐)
    for N in [256, 512, 1024, 1025, 4096]:
        s, e = test_softmax(N)
        results.append(("softmax", N, s, e))

    for N in [256, 512, 1024, 1025, 4096]:
        s, e = test_gelu(N)
        results.append(("gelu", N, s, e))

    # 汇总
    failures = [r for r in results if r[2] == "FAIL"]
    print(f"\n{'='*50}", flush=True)
    if failures:
        print(f"FAIL: {len(failures)}/{len(results)} tests", flush=True)
        for r in failures:
            print(f"  {r[0]} N={r[1]}  max_err={r[3]:.6f}", flush=True)
    else:
        print(f"ALL {len(results)} TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
