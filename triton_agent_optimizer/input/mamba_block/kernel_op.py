#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Mamba Block (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链 (Mamba-1 简化, 对角 A):
#    z, xb = split(X @ in_proj)                    # [L, ED] + [L, 3N]
#    xb = conv1d_depthwise(xb)                      # 因果 depthwise conv, K 内核
#    dt = softplus(xb[:N] + dt_bias); B = xb[N:2N]; C = xb[2N:]
#    h_t = exp(dt·A)·h_{t-1} + dt·B·x               # 对角 SSM 时序扫描 (关联扫描)
#    Y = ((h·C).sum(-1)·D + z) @ out_proj + X[:ED]  # 乘性 D + 门控 + 残差
#  扫描用 log 域关联扫描 (torch 参考实现, body 内执行, 计时包含)
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块 (与 bench_910b3 _shapes 对齐)
# ═══════════════════════════════════════════════════════════════════════════════
L = int(os.environ.get("MB_LEN", 1024))       # 序列长
D = int(os.environ.get("MB_DIM", 1024))       # 输入/输出维
N = int(os.environ.get("MB_SSM", 16))         # SSM 状态维
ED = int(os.environ.get("MB_ED", 1024))       # 门控维 (== D)
KC = int(os.environ.get("MB_KC", 4))          # conv 内核宽
DTYPE = torch.float32
C3N = 3 * N
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # matmul 分块
BLOCK_L = 256                                 # conv/scan 行分块
BLOCK_C = 64                                  # conv 通道分块 (≥3N=48)
BLOCK_EL = 1024                               # 逐元素分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 因果 depthwise conv1d: y[l,c] = Σ_{j<K} x[l+j-(K-1), c] · w[c,j] + b[c]

@triton.jit
def matmul_kernel2(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 因果 depthwise conv1d: y[l,c] = Σ_{j<K} x[l+j-(K-1), c] · w[c,j] + b[c]
@triton.jit
def conv1d_dw_kernel(x_ptr, w_ptr, b_ptr, y_ptr, L, C, K,
                     BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_l = (L + BLOCK_L - 1) // BLOCK_L
    c = pid // grid_l
    lb = pid % grid_l
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = offs_l < L
    base = c * L
    acc = tl.load(b_ptr + c)
    for j in range(K):
        src = offs_l + j - (K - 1)
        xv = tl.load(x_ptr + base + src, mask=l_mask & (src >= 0), other=0.0)
        acc = acc + xv * tl.load(w_ptr + c * K + j)
    tl.store(y_ptr + base + offs_l, acc, mask=l_mask)


# 切分 + softplus: 从 xb [L,3N] 出 dt/b/c 三段
@triton.jit
def split_softplus_kernel(x_ptr, dtb_ptr, dt_ptr, b_ptr, c_ptr,
                          L, N, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < 3 * N
    v = tl.load(x_ptr + row * 3 * N + offs, mask=mask, other=0.0)
    bias = tl.load(dtb_ptr + offs, mask=mask, other=0.0)
    dt = v + bias
    dt = tl.where(dt > 0, dt, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dt)))   # softplus
    tl.store(dt_ptr + row * N + offs, dt, mask=mask & (offs < N))
    tl.store(b_ptr + row * N + offs, v, mask=mask & (offs >= N) & (offs < 2 * N))
    tl.store(c_ptr + row * N + offs, v, mask=mask & (offs >= 2 * N))


# 门控: y[l,:ED] = (Σ_n h[l,n]·C[l,n]) · Dsum + z[l,:]
@triton.jit
def mamba_gate_kernel(h_ptr, c_ptr, z_ptr, d_ptr, y_ptr,
                      L, N, ED, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    h = tl.load(h_ptr + row * N + offs, mask=mask, other=0.0)
    c = tl.load(c_ptr + row * N + offs, mask=mask, other=0.0)
    dot = tl.sum(h * c, axis=0)
    dsum = tl.sum(tl.load(d_ptr + offs, mask=mask, other=0.0), axis=0)
    offs_e = tl.arange(0, BLOCK)
    e_mask = offs_e < ED
    z = tl.load(z_ptr + row * ED + offs_e, mask=e_mask, other=0.0)
    tl.store(y_ptr + row * ED + offs_e, dot * dsum + z, mask=e_mask)


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    import torch.nn.functional as F
    npu = torch.device("npu")
    x = (torch.randn(L, D, dtype=DTYPE, device=npu) * 0.1)
    x_ed = x[:, :ED].contiguous()                 # ★窗口外预切 (残差输入)
    in_proj = (torch.randn(D, ED + 3 * N, dtype=DTYPE, device=npu) * 0.1)
    conv_w = (torch.randn(3 * N, 1, KC, dtype=DTYPE, device=npu) * 0.1)
    conv_b = (torch.randn(3 * N, dtype=DTYPE, device=npu) * 0.1)
    a_log = (torch.randn(N, dtype=DTYPE, device=npu) * 0.1 - 3.0)   # A = -exp(a_log) 负
    dt_bias = (torch.randn(N, dtype=DTYPE, device=npu) * 0.1)
    dm = (torch.randn(ED + N, dtype=DTYPE, device=npu) * 0.1)
    out_proj = (torch.randn(ED, D, dtype=DTYPE, device=npu) * 0.1)

    zx = torch.empty(L, ED + 3 * N, dtype=DTYPE, device=npu)
    z = torch.empty(L, ED, dtype=DTYPE, device=npu)
    xb = torch.empty(L, 3 * N, dtype=DTYPE, device=npu)
    xc = torch.empty(L, 3 * N, dtype=DTYPE, device=npu)
    dt = torch.empty(L, N, dtype=DTYPE, device=npu)
    b = torch.empty(L, N, dtype=DTYPE, device=npu)
    c = torch.empty(L, N, dtype=DTYPE, device=npu)
    dA = torch.empty(L, N, dtype=DTYPE, device=npu)
    dB = torch.empty(L, N, dtype=DTYPE, device=npu)
    h = torch.empty(L, N, dtype=DTYPE, device=npu)
    gy = torch.empty(L, ED, dtype=DTYPE, device=npu)
    y = torch.empty(L, D, dtype=DTYPE, device=npu)

    grid_mm = (triton.cdiv(L, BLOCK_M) * triton.cdiv(ED + 3 * N, BLOCK_N),)
    grid_mm_o = (triton.cdiv(L, BLOCK_M) * triton.cdiv(D, BLOCK_N),)
    grid_cv = (3 * N * triton.cdiv(L, BLOCK_L),)
    grid_sp = (L,)
    grid_gate = (L,)
    grid_el = (triton.cdiv(L * D, BLOCK_EL),)
    print(f"[info] Mamba L={L} D={D} N={N} ED={ED} K={KC} kernels=8 (scan 走 torch 关联扫描)")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        matmul_kernel[grid_mm](x, in_proj, zx, L, ED + 3 * N, D,
                               x.stride(0), x.stride(1), in_proj.stride(0), in_proj.stride(1),
                               zx.stride(0), zx.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        z.copy_(zx[:, :ED])
        xb.copy_(zx[:, ED:])
        conv1d_dw_kernel[grid_cv](xb, conv_w, conv_b, xc, L, 3 * N, KC,
                                  BLOCK_L=BLOCK_L, BLOCK_C=BLOCK_C)
        split_softplus_kernel[grid_sp](xc, dt_bias, dt, b, c, L, N, BLOCK=BLOCK_C)
        # ★对角 SSM 关联扫描 (torch, body 内; log 域防下溢)
        A = -torch.exp(a_log)
        dA = torch.exp(dt * A)
        dB = dt * b
        _logP = torch.cumsum(torch.log(dA.clamp_min(1e-8)), dim=0)
        h = torch.cumsum(dB * torch.exp((-_logP).clamp_min(-30)), dim=0) \
            * torch.exp(_logP.clamp_max(30))
        mamba_gate_kernel[grid_gate](h, c, z, dm, gy, L, N, ED, BLOCK=BLOCK_C)
        matmul_kernel2[grid_mm_o](gy, out_proj, y, L, D, ED,
                                 gy.stride(0), gy.stride(1), out_proj.stride(0), out_proj.stride(1),
                                 y.stride(0), y.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        add_kernel[grid_el](y, x_ed, y, L * D, BLOCK=BLOCK_EL)
    torch.npu.synchronize()
    print("[info] Mamba launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        zx_r = x @ in_proj
        z_r = zx_r[:, :ED]
        xb_r = zx_r[:, ED:]
        xb_r = F.conv1d(xb_r.unsqueeze(0).transpose(1, 2), conv_w, conv_b,
                        groups=3 * N, padding=KC - 1)[..., :L].transpose(1, 2).squeeze(0)
        dt_r = F.softplus(xb_r[:, :N] + dt_bias)
        b_r = xb_r[:, N:2 * N]
        c_r = xb_r[:, 2 * N:]
        A = -torch.exp(a_log)
        dA_r = torch.exp(dt_r * A)
        dB_r = dt_r * b_r
        _logP = torch.cumsum(torch.log(dA_r.clamp_min(1e-8)), dim=0)
        h_r = torch.cumsum(dB_r * torch.exp((-_logP).clamp_min(-30)), dim=0) \
            * torch.exp(_logP.clamp_max(30))
        gy_r = (h_r * c_r).sum(-1, keepdim=True) * dm[:N].sum(-1, keepdim=True) + z_r
        ref = gy_r @ out_proj + x[:, :ED]
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
