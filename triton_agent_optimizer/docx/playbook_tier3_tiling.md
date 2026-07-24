# Tier 3: Tiling & Block Config

> **核心原则: 在融合后的稳定结构上调 tile。先调 BLOCK_SIZE, 再调 num_warps/num_stages。**

## 910B3 饱和度参数

| 引擎 | peak_clamp | k0 (半饱和点) | 饱和区 | 1KB 实际带宽 | 10KB 实际带宽 | 可靠? |
|---|---|---|---|---|---|---|
| GM→UB | 80.83 GB/s | 6.65 KB | > 13 KB | 15.8 GB/s (19.5%) | 58.8 GB/s (72.8%) | ✅ |
| UB→GM | 76.67 GB/s | 10.72 KB | > 21 KB | 16.2 GB/s (21.1%) | 46.7 GB/s (60.9%) | ✅ |
| VecUnit | 404.0 GB/s | 4.50 KB | > 9 KB | 83.8 GB/s (20.7%) | 317.9 GB/s (78.7%) | ✅ |
| GM→L1 | 37.5 GB/s | 6.65 KB | > 13 KB | — | — | ❌ |
| L1→L0 | 100.0 GB/s | 6.65 KB | > 13 KB | — | — | ❌ |
| CubeUnit | 150.0 GB/s | 0 (flat) | 始终饱和 | 150.0 | 150.0 | ❌ |
| L0→GM | 37.5 GB/s | 6.65 KB | > 13 KB | — | — | ❌ |

**公式**: `bw = vpeak × size_kb / (size_kb + k0)`, clamped to `peak_clamp`

## BLOCK_SIZE 选择

### Step 1: 计算 tile 上限

```
max_tile_kb = UB_KB / n_buffers  = 192KB / n_buffers
  n_buffers=2 → max=96KB
  n_buffers=3 → max=64KB
  n_buffers=4 → max=48KB
```

### Step 2: 找瓶颈 op 的半饱和点

```
if bw_utilization < 70% and regime in (floor, ramp):
    target_tile > k0 × 2  # 进入饱和区
    expected_bw ≈ peak_clamp × 0.90
elif bw_utilization >= 90% and regime == saturated:
    # 已饱和 → 不能靠增大 tile 提速 → 晋升 Tier 4
```

### Step 3: 选择 tile size

```
target = min(max_tile_kb, k0 × 3)  # 有余量但不过大
BLOCK_SIZE = target × 1024 / 2      # /2 = fp16 bytes, 得到元素数
```

### Step 4: 验证

- 新 tile ≤ UB 容量
- 新 tile can be divided evenly by grid
- 非整除的边界有 mask 处理

## num_warps / num_stages

| 参数 | 作用 | 范围 | 启发式 |
|---|---|---|---|
| num_warps | SM 并行度 | 1-8 | memory-bound → 4; compute-bound → 8 |
| num_stages | 流水线深度 | 0-4 | 单 GEMM → 0; 融合双 GEMM → 1; 非 GEMM → 1 |

**910B3 注意**: num_stages > 2 在 Ascend backend 有兼容性风险, 保守用 1-2

## autotune 调优 (生产环境)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [512, 1024, 2048, 4096, 8192]
        for nw in [2, 4, 8]
        for ns in [1, 2]
    ],
    key=["N"],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

## 操作步骤

1. 读 critical_path 上每个 op 的 `bw_utilization` + `regime`
2. 找 regime=floor/ramp (bw_util < 70%) 的 op → **优先增大 tile**
3. 全 saturated → 本 Tier 无改进, 晋升 Tier 4
4. 计算推荐 tile: k0×2 ~ max_tile_kb
5. 改 BLOCK_SIZE, 重跑 simulator

## 示例 Plan

```json
{
  "strategy": "increase_tile_size",
  "reason": "op0(gm_to_ub) tile=1KB, bw_util=21%, regime=ramp, k0=6.65KB",
  "target_tile_kb": 16,
  "expected_impact": "bw_util 21%→90%+, duration 从 64.7ns 降到 ~14ns, 预计总加速 1.3-1.5×",
  "verification_method": "simulator --llm 对比 bw_utilization 变化"
}
```

```json
{
  "strategy": "tune_block_config",
  "reason": "所有传输已饱和, 当前 num_warps=4, 尝试 num_warps=8 提高 SM 占用率",
  "specific_change": "num_warps: 4→8, num_stages: 1→2",
  "target_speedup": 1.05,
  "expected_impact": "SM 占用率提升, pipeline overlap 增加, 预计 5% 改善",
  "verification_method": "910B3 真机 benchmark warmup=30 repeat=200"
}
```
