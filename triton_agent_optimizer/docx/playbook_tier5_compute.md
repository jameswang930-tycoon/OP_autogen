# Tier 5: Compute & Occupancy

> **核心原则: 传输已优化, 现在看计算是否达到峰值。计算优化是最细粒度调整。**

## 910B3 VecUnit 参数

| 参数 | 值 |
|---|---|
| 单核峰值 | 404 GB/s (≈ 8 TFLOPS fp16) |
| 半饱和点 k0 | 4.5 KB (> 9 KB 饱和) |
| 聚合峰值 (40核) | 16160 GB/s (≈ 9.2 TFLOPS) |
| 校准状态 | ✅ MEASURED |

## 三类优化

### ① 计算-传输重叠

**识别**: VecUnit 时间占比 < 30%, 传输占 > 60%, `parallel_pairs=0`

**操作**: 用 double buffer 让 VecUnit 和 MTE 同时工作

**910B3 注意**: MTE2(GM→UB) 和 MTE3(UB→GM) 是不同引擎, 可以同时运行。但 VecUnit 和 MTE 共享 UB 带宽, 实际重叠效率需实测。

### ② 向量化

**识别**: 逐元素操作未对齐到向量宽度

**操作**:
1. 确保 `tl.load` 一次至少加载 16 个 fp16 元素 (256-bit SIMD)
2. 如果 BLOCK_SIZE 不是 16 的倍数 → 补齐到 16 倍数
3. 使用 `tl.vectorize` 或手动展开循环

### ③ 精度取舍

**识别**: 瓶颈在传输, fp32 精度开销大

**操作**: fp16 compute + fp32 accumulate → 减少一半传输量

**适用**: 对精度不极端敏感的算子 (大多数激活函数都可以)

**910B3**: fp16 原生支持, bf16 不支持

## 910B3 CubeUnit

| 参数 | 值 |
|---|---|
| 峰值 | 150 GB/s (PLACEHOLDER ❌) |
| k0 | 0 (flat, size-independent) |

**CubeUnit 参数不可靠 — 优化建议必须标注 UNCERTAIN。不要基于它做精确速度预测。**

## Occupancy

计算: `occupancy = min(REGISTER_LIMIT/VGPR, SMEM_LIMIT/SMEM_USED)`

910B3 参考: 每 Vec Core 约 256 VGPR, num_warps 每增加 1 多占 ~16 VGPR

## 操作步骤

1. 读 `compute_only` ops 的 `bw_utilization` (即计算吞吐利用率)
2. VecUnit 利用率 > 90% → 已达峰值, 不能直接提速
3. 检查 `parallel_pairs` 和 `engine_utilization`: VecUnit 空闲 + MTE 忙 → double buffer
4. CubeUnit → 参数不可靠, 保守建议
5. 全部已达峰值 → 晋升 Tier 6

## 示例 Plan

```json
{
  "strategy": "fp32_to_fp16_compute",
  "reason": "当前使用 fp32 compute, 传输量大。fp16 精度足够 (vadd, 无累计误差)",
  "specific_change": "tl.load 使用 fp16 dtype, compute 用 fp16, 只在最终 store 时 cast 到 fp32",
  "expected_impact": "传输量减半, 预计加速 1.1-1.3×",
  "verification_method": "CPU emulator fp32 vs fp16 数值对比, rtol=1e-2"
}
```
