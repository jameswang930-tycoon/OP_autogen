#!/usr/bin/env python3
"""
Matmul kernel — Triton-Ascend 测试脚本 (msprof op simulator 已验证可用)
════════════════════════════════════════════════════════════════════════════
  全部在服务器上封闭执行 (CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3):

  cd triton_agent_optimizer/input/softmax

  # ── 1. 验证正确性 ──
  python3 run_and_profile.py

  # ── 2. msprof op simulator 采集 ──
  msprof op simulator \
      --application="python3 run_and_profile.py" \
      --kernel-name="matmul_kernel" \
      --soc-version=Ascend910B3 \
      --launch-count=5 --core-id=0 \
      --output=./msprof_sim

  # ── 3. HIVM MLIR 提取 ──
  rm -rf ~/.triton/cache/
  TRITON_ALWAYS_COMPILE=1 python3 run_and_profile.py
  find ~/.triton/cache -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" | while read f; do cp "$f" hivmir/; done

  # ── 4. 解析 HIVM → 11 语义字段 ──
  bash step2_parse_hivm.sh

  # ── 5. 解析 msprof trace → 14 时序字段 ──
  bash step3_parse_msprof.sh

  # ── 6. 合并 → 29 字段全填充 ──
  bash step4_merge.sh
════════════════════════════════════════════════════════════════════════════
"""
import torch
import torch_npu
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════
#  Matmul kernel: C[M,N] = A[M,K] @ B[K,N]  (fp16, fp32 accumulate → tl.dot Cube)
#  Pipeline: GM→L1(load A) + GM→L1(load B) → L1→L0A/L0B → Cube(MMA) → L0C→UB → UB→GM(store C)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  NUM_N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak)
        b = tl.load(b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc.to(tl.float16))


def main():
    M, N, K = 1024, 1024, 1024
    BM, BN, BK = 128, 128, 32

    a = torch.randn(M, K, device='npu', dtype=torch.float16)
    b = torch.randn(K, N, device='npu', dtype=torch.float16)
    c = torch.empty(M, N, device='npu', dtype=torch.float16)

    grid = (M // BM * N // BN,)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        NUM_N=N // BN, BM=BM, BN=BN, BK=BK)
    torch.npu.synchronize()

    # 正确性验证
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel = err / (ref_max + 1e-6)

    status = "PASS" if rel < 0.3 else "FAIL"
    print(f"[matmul] {M}x{N}x{K} BM={BM} BN={BN} BK={BK} max_err={err:.4f} rel={rel:.4f}  {status}", flush=True)


if __name__ == "__main__":
    main()
