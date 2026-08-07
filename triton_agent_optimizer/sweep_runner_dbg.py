#!/usr/bin/env python3
"""Auto-generated BLOCK sweep runner — DO NOT EDIT."""
import os, sys, json, math, time
import torch
import torch_npu
import triton
import triton.language as tl

# Module-level code from kernel_op.py (imports, config, kernel defs)
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    # ═══════════════════════════════════════════════════════════════════════════════
    #  单文件 kernel_op.py — 两层 MLP (X→FC1→GELU→FC2→Y) 合一体 (v4)
    #  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
    #  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
    #  运算链: Y = GELU(X@W1 + b1) @ W2
    # ═══════════════════════════════════════════════════════════════════════════════
    import os
    import sys
    import torch
    import torch_npu          # 必须先 import, 注册 NPU 后端
    import triton
    import triton.language as tl


    # ═══════════════════════════════════════════════════════════════════════════════
    #  ① 场景 config — 尺寸/精度/分块
    #   形状: X[M,K] @ W1[K,HIDDEN] → Z[M,HIDDEN] → GELU(Z+b1) → H[M,HIDDEN] @ W2[HIDDEN,N] → Y[M,N]
    # ═══════════════════════════════════════════════════════════════════════════════
    M  = int(os.environ.get("MATMUL_M", 2048))
    N  = int(os.environ.get("MATMUL_N", 2048))
    K  = int(os.environ.get("MATMUL_K", 2048))
    HIDDEN = int(os.environ.get("MLP_HIDDEN", 2048))   # 隐藏层宽度
    DTYPE = torch.float32
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    BLOCK_SIZE = 1024                        # 逐元素 kernel (bias_gelu) 分块
    # 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


    # ═══════════════════════════════════════════════════════════════════════════════
    #  ② 算子 kernel
    # ═══════════════════════════════════════════════════════════════════════════════

    # FC1: Z = X @ W1   (matmul, fp32 累加)
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr,
                      M, N, K,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
        pid = tl.program_id(axis=0)
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        pid_m = pid // grid_n
        pid_n = pid % grid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


    # FC2: Y = H @ W2   (同样 matmul, 独立 kernel 名以便 msprof 区分两次 matmul)
    @triton.jit
    def matmul_kernel2(a_ptr, b_ptr, c_ptr,
                       M, N, K,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
        pid = tl.program_id(axis=0)
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        pid_m = pid // grid_n
        pid_n = pid % grid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


    # bias + GELU: H = GELU(Z + b1)  (逐元素: 加 bias 后 tanh-GELU 激活)
    @triton.jit
    def bias_gelu_kernel(x_ptr, bias_ptr, y_ptr,
                         n_elements, N,
                         BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        b = tl.load(bias_ptr + (offs % N), mask=mask, other=0.0)   # bias 按列广播 (每行同列)
        val = x + b
        # tanh 近似 GELU (与 torch F.gelu(approximate="tanh") 一致)
        cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))
        y = val * cdf
        tl.store(y_ptr + offs, y, mask=mask)


    # ═══════════════════════════════════════════════════════════════════════════════
    #  ③ 测试 main — 分配输入/启动 kernel/同步 (一般不动)
    # ═══════════════════════════════════════════════════════════════════════════════
    def main():
        if not torch.npu.is_available():
            raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

        # 输入 (小值 ±0.05, 避免 fp32 dot 值域溢出)
        x  = (torch.rand(M, K, dtype=DTYPE, device="npu") - 0.5) * 0.1
        w1 = (torch.rand(K, HIDDEN, dtype=DTYPE, device="npu") - 0.5) * 0.1
        b1 = (torch.rand(HIDDEN, dtype=DTYPE, device="npu") - 0.5) * 0.1
        w2 = (torch.rand(HIDDEN, N, dtype=DTYPE, device="npu") - 0.5) * 0.1
        z = torch.empty(M, HIDDEN, dtype=DTYPE, device="npu")
        h = torch.empty(M, HIDDEN, dtype=DTYPE, device="npu")
        y = torch.empty(M, N, dtype=DTYPE, device="npu")

        grid1 = (triton.cdiv(M, BLOCK_M) * triton.cdiv(HIDDEN, BLOCK_N),)   # fc1: X[M,K]@W1[K,H]
        grid2 = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)        # fc2: H[M,H]@W2[H,N]
        grid_g = (triton.cdiv(M * HIDDEN, BLOCK_SIZE),)                     # bias_gelu: M*H 个元素

        print(f"[info] MLP M={M} K={K} HIDDEN={HIDDEN} N={N}  dtype={DTYPE}")
        print(f"[info] fc1 grid={grid1[0]}  fc2 grid={grid2[0]}  bias_gelu grid={grid_g[0]}  "
              f"block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

        # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
        #   默认 1 = 正常跑一遍; verify 设 VERIFY_LOOP(默认30)
        LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
        for _ in range(LOOP):
            # FC1: Z = X @ W1
            matmul_kernel[grid1](
                x, w1, z,
                M, HIDDEN, K,
                x.stride(0), x.stride(1),
                w1.stride(0), w1.stride(1),
                z.stride(0), z.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            # bias + GELU: H = GELU(Z + b1)
            bias_gelu_kernel[grid_g](
                z, b1, h,
                M * HIDDEN, HIDDEN,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            # FC2: Y = H @ W2
            matmul_kernel2[grid2](
                h, w2, y,
                M, N, HIDDEN,
                h.stride(0), h.stride(1),
                w2.stride(0), w2.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
        torch.npu.synchronize()
        print("[info] MLP launched & synced OK")

        # 正确性校验 (默认关, 要开用 MATMUL_VERIFY=1)
        if os.environ.get("MATMUL_VERIFY", "0") == "1":
            try:
                import torch.nn.functional as F
                z_ref = torch.matmul(x, w1)
                h_ref = F.gelu(z_ref + b1, approximate="tanh")
                y_ref = torch.matmul(h_ref, w2)
                abs_diff = (y - y_ref).abs().max().item()
                rel_diff = abs_diff / (y_ref.abs().max().item() + 1e-6)
                print(f"[info] result check: {'PASS' if rel_diff < 0.05 else 'CHECK'}  "
                      f"max|Y-Y_ref|={abs_diff:.5f} rel={rel_diff:.5f}")
            except Exception as e:
                print(f"[warn] result check skipped: {e}")

# Tensor setup + run_one
    # ── Tensor setup (MLP 3 kernel) ──
    M, K, N = 2048, 2048, 2048
    HIDDEN = 2048
    DTYPE = torch.float32
    device = torch.device("npu")
    x  = (torch.rand(M, K, dtype=DTYPE, device=device) - 0.5) * 0.1
    w1 = (torch.rand(K, HIDDEN, dtype=DTYPE, device=device) - 0.5) * 0.1
    b1 = (torch.rand(HIDDEN, dtype=DTYPE, device=device) - 0.5) * 0.1
    w2 = (torch.rand(HIDDEN, N, dtype=DTYPE, device=device) - 0.5) * 0.1
    z = torch.empty(M, HIDDEN, dtype=DTYPE, device=device)
    h = torch.empty(M, HIDDEN, dtype=DTYPE, device=device)
    y = torch.empty(M, N, dtype=DTYPE, device=device)

    def run_one(bm, bn, bk):
        grid1 = (triton.cdiv(M, bm) * triton.cdiv(HIDDEN, bn),)
        grid2 = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        grid_g = (triton.cdiv(M * HIDDEN, BLOCK_SIZE),)
        matmul_kernel[grid1](x, w1, z, M, HIDDEN, K,
            x.stride(0), x.stride(1), w1.stride(0), w1.stride(1),
            z.stride(0), z.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        bias_gelu_kernel[grid_g](z, b1, h, M * HIDDEN, HIDDEN, BLOCK_SIZE=BLOCK_SIZE)
        matmul_kernel2[grid2](h, w2, y, M, N, HIDDEN,
            h.stride(0), h.stride(1), w2.stride(0), w2.stride(1),
            y.stride(0), y.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)


# Sweep
CANDIDATES = [[64, 64, 64]]
WARMUP = int(os.environ.get("SWEEP_WARMUP", "2"))
LOOP   = int(os.environ.get("SWEEP_LOOP", "5"))
results = []
n_total = len(CANDIDATES)

print(f"[sweep] {n_total} candidates, warmup={WARMUP}, loop={LOOP}", flush=True)

for idx, cfg in enumerate(CANDIDATES):
    BLOCK_M=cfg[0], BLOCK_N=cfg[1], BLOCK_K=cfg[2]
    try:
        for _ in range(WARMUP):
            run_one(BLOCK_M, BLOCK_N, BLOCK_K)
        torch.npu.synchronize()
        times = []
        for _ in range(LOOP):
            ev_s = torch.npu.Event(enable_timing=True)
            ev_e = torch.npu.Event(enable_timing=True)
            ev_s.record()
            run_one(BLOCK_M, BLOCK_N, BLOCK_K)
            ev_e.record()
            torch.npu.synchronize()
            times.append(ev_s.elapsed_time(ev_e))
        avg_ns = sum(times) / len(times) * 1e6
        results.append({"block": list(cfg), "ns": round(avg_ns, 1)})
        print(f"  [{idx+1}/{n_total}] {cfg}: {avg_ns:.0f}ns", flush=True)
    except Exception as e:
        results.append({"block": list(cfg), "ns": None, "error": str(e)[:120]})
        print(f"  [{idx+1}/{n_total}] {cfg}: ERROR {str(e)[:100]}", flush=True)

valid = [r for r in results if r.get("ns")]
errs  = [r for r in results if not r.get("ns")]
valid.sort(key=lambda r: r["ns"])
out = {"measured_at": __import__("datetime").datetime.now().isoformat(),
       "total_candidates": n_total, "valid": len(valid), "errors": len(errs),
       "warmup": WARMUP, "loop": LOOP,
       "results": valid + errs}
out_path = os.environ.get("SWEEP_OUTPUT", "sweep_result.json")
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"\n[sweep] Done: {len(valid)} valid + {len(errs)} errors -> {out_path}", flush=True)
if valid:
    print(f"[sweep] Best: {valid[0]['block']} = {valid[0]['ns']:.0f}ns", flush=True)
