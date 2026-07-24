# Tier 4: Memory Access

> **核心原则: tile 已调优, 现在优化数据的存取方式。不能靠增大 tile 提速的, 靠改访问模式。**

## 三种优化手段

### ① 小传输合并 (bw_util < 70%, regime=floor/ramp)

**识别**: 多个同类型传输 op (gm_to_ub 或 ub_to_gm) 分散在时间线上, 每个 tile 小于 k0

**操作**:
1. 找到相同 engine 的连续小传输
2. 合并为一次大传输: `new_size = sum(old_sizes)`
3. 确保合并后 `new_size × n_buffers ≤ 192KB`
4. 如果合并后 size > k0×2 → 进入饱和区 → bw_util 从 20% 跳到 90%+

**效果**: 1KB(15.8 GB/s) × 10 → 10KB(58.8 GB/s) × 1 → 省掉 9 次小传输的开销

### ② Double Buffering (传输全部 saturated, 但仍有优化空间)

**识别**: 传输 op 全 saturated, VecUnit 空闲 (>70% idle), `parallel_pairs=0`

**操作**:
```
当前结构 (串行):
  load_all → compute → store_all

改为 (double buffer):
  buffer_A = load_tile_0 → compute_tile_0 → store_tile_0
  buffer_B =                load_tile_1 → compute_tile_1 → store_tile_1
  (load_tile_1 和 store_tile_0 并行)
```

**检查**:
1. `2 × tile_size × n_buffers ≤ 192KB` (double buffer 需双倍空间)
2. 依赖链允许 split (RAW 只在一个 tile 内)
3. 910B3 支持 MTE2 和 MTE3 同时运行 (不同 engine)

**预期**: 传输和计算重叠, 省 ~30% 时间

### ③ Coalescing (非连续访问)

**识别**: 访问模式不是连续对齐的 (stride ≠ 1)

**操作**:
1. 确保 `tl.load` 的 offset 是 `arange(0, BLOCK)` 的连续偏移
2. 如果 stride > 1 → 重组数据布局或使用 gather/scatter
3. 910B3 要求 32-byte 对齐

## 910B3 引擎状态速查

| 引擎 | 校准状态 | 优化建议可信度 |
|---|---|---|
| GM→UB, UB→GM, VecUnit | ✅ MEASURED | 高 |
| GM→L1, L1→L0, CubeUnit, L0→GM | ❌ PLACEHOLDER | **不要做精确预测** |

涉及 placeholder 引擎时, 保守建议, 标注 UNCERTAIN。

## 操作步骤

1. 读 `transfer_only` ops 的 `bw_utilization` + `regime`
2. **regime=floor/ramp** → 合并小传输
3. **全 saturated + VecUnit 空闲** → double buffer
4. **全 saturated + VecUnit 也饱和** → 已最优, 晋升 Tier 5
5. 检查 UB 容量约束 + 依赖链

## 示例 Plan (小传输合并)

```json
{
  "strategy": "merge_small_transfers",
  "reason": "10×1KB gm_to_ub 各自 bw_util=21%, 合并为 1×10KB 预期 bw_util=73%",
  "specific_change": "for loop 10 iterations 1KB each → single 10KB gm_to_ub",
  "expected_impact": "GM→UB 总时间从 647ns 降到 ~174ns, 预计加速 1.2-1.5×",
  "verification_method": "simulator --llm 对比 bw_utilization 和 total_ns"
}
```

## 示例 Plan (Double Buffer)

```json
{
  "strategy": "double_buffering",
  "reason": "GM→UB+UB→GM 全 saturated, VecUnit 只占 9%, 串行 → 用 double buffer 并行化",
  "change": "拆 128KB→2×64KB tile, 交错 load(A)→compute(A)→store(A) 和 load(B)→compute(B)→store(B)",
  "expected_impact": "UB→GM 和下一次 GM→UB 重叠, 预计加速 1.15-1.2×",
  "verification_method": "simulator --llm 检查 parallelism 和 total_ns 变化"
}
```
