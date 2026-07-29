#!/usr/bin/env python3
"""
Softmax + GELU — Triton-Ascend 完整测试脚本 (多管线通路验证)

════════════════════════════════════════════════════════════════════════════
  服务器执行命令 (按顺序, 产物全部落到项目目录):
════════════════════════════════════════════════════════════════════════════

  # 进入项目目录
  cd triton_agent_optimizer
  OUT=input/softmax

  # ── 1. 验证正确性 ──
  python3 $OUT/run_and_profile.py

  # ── 2. 提取 HIVM MLIR → 拷贝到项目内 ──
  export TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_CACHE=1
  python3 $OUT/run_and_profile.py
  # 找到最新 dump
  DUMP_DIR=$(ls -dt ~/.triton/dump/*/ 2>/dev/null | head -1)
  mkdir -p $OUT/hivmir
  cp $DUMP_DIR/*.npuir.mlir $OUT/hivmir/ 2>/dev/null
  ls -la $OUT/hivmir/

  # ── 3. msprof 采集 softmax ──
  msprof op \
      --application="python3 $OUT/run_and_profile.py" \
      --kernel-name=softmax_kernel \
      --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
      --output=$OUT/msprof_softmax
  ls -la $OUT/msprof_softmax/OPPROF_*/

  # ── 4. msprof 采集 gelu ──
  msprof op \
      --application="python3 $OUT/run_and_profile.py" \
      --kernel-name=fused_gelu_kernel \
      --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
      --output=$OUT/msprof_gelu
  ls -la $OUT/msprof_gelu/OPPROF_*/

════════════════════════════════════════════════════════════════════════════
  服务器产出物 (都在 triton_agent_optimizer/input/softmax/ 下):
════════════════════════════════════════════════════════════════════════════

  input/softmax/
  ├── run_and_profile.py
  ├── triton_kernel.py
  ├── hivmir/
  │   ├── softmax_kernel.npuir.mlir    ← HIVM 语义数据
  │   └── fused_gelu_kernel.npuir.mlir
  ├── msprof_softmax/OPPROF_xxx/       ← msprof 时序数据 (softmax)
  │   ├── PipeUtilization.csv
  │   ├── Memory.csv
  │   └── simulator/*/core*_instr_exe.csv
  └── msprof_gelu/OPPROF_xxx/          ← msprof 时序数据 (gelu)

════════════════════════════════════════════════════════════════════════════
  然后把这些文件打包传回 Windows, 我来跑分析链:
════════════════════════════════════════════════════════════════════════════

  # 5. HIVM 解析 → 11 字段
  python3 analyzers/hivmir_analyzer.py input/softmax/hivmir/softmax_kernel.npuir.mlir

  # 6. msprof trace 解析 → 14 字段
  python3 analyzers/msprof_analyzer.py input/softmax/msprof_softmax/OPPROF_xxx/

  # 7. 合并 → 29 字段
  python3 analyzers/dsl_merger.py ...

  # 8. 瓶颈诊断 → Planner prompt
  ...

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

    out_triton = torch.empty_like(x)
    # 单 program 模式: 所有元素在一个 block 内, max/sum 是全局的
    # 原因: 多 program 拆分时每个 program 独立算 local max/sum, 不合并
    max_npu = 4096  # UB 192KB, fp32*4096=16KB, 放得下
    actual_block = min(N, max_npu)
    grid = (1,)
    softmax_kernel[grid](x, out_triton, N, BLOCK_SIZE=actual_block)
    torch.npu.synchronize()

    # PyTorch reference
    out_torch = F.softmax(x, dim=0)

    max_err = torch.max(torch.abs(out_triton - out_torch)).item()
    # NPU libdevice exp 精度略低于 PyTorch, 允许 1e-2 误差
    status = "PASS" if max_err < 1e-2 else "FAIL"
    print(f"[softmax] N={N:>5} BLK={actual_block:>3} grid=1 max_err={max_err:.6f}  {status}", flush=True)
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
    # gelu 涉及 tanh, NPU 上允许 5e-3 误差
    status = "PASS" if max_err < 5e-3 else "FAIL"
    print(f"[gelu]    N={N:>5} BLK={BLOCK_SIZE:>3} max_err={max_err:.6f}  {status}", flush=True)
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
