# Tier 6: 910B3 Architecture-Specific

> **核心原则: 所有通用优化都用尽后, 利用 910B3 硬件特性做最后微调。这些优化只对 910B3 有效。**

## 硬件配置速查

| 参数 | 值 |
|---|---|
| AI Core (transfer) | 20 核 @ 1.8 GHz |
| Vec Core (compute) | 40 核 @ 1.8 GHz |
| UB per core | 192 KB |
| L2 shared | 192 MB |
| HBM | 64 GB, ~1.54 TB/s (实测 ~1538 GB/s) |
| HCCS 互联 | 8卡全互联, 单链路 4 lane |

## 优化手段

### ① Grid 数调整

transfer grid (AI Core): 默认 20, compute grid (Vec Core): 默认 40

**诊断**: 读 `engine_utilization`:
- GM→UB/UB→GM 利用率 < 30% → transfer grid 太多, 可以减少
- VecUnit 利用率 > 80% → compute grid 可以增加
- 不均衡 → 调整比例

**910B3 约束**: transfer grid 必须是 20 的约数 (或整除20), compute grid 必须是 40 的约数 (或整除40)

### ② Pipeline 选择

| Pipeline | 路径 | 适用 |
|---|---|---|
| **Vector** | GM→UB(MTE2) → VecUnit → UB→GM(MTE3) | 小 op, 逐元素 |
| **Matrix** | GM→L1(MTE2) → L1→L0(MTE1) → CubeUnit → L0→GM(FIXP) | 大 op, 矩阵 |

**诊断**: 如果当前用 Vector Pipeline 但 op 是矩阵类 → 换 Matrix Pipeline

**注意**: GM→L1, L1→L0, CubeUnit, L0→GM 全是 PLACEHOLDER → 切换后性能不确定, 需实测

### ③ L2 驻留

**条件**: 总工作集 < 192MB (L2 容量)

**操作**:
1. 减少数据块大小, 让数据留在 L2
2. 调整 access pattern 改善 L2 命中率
3. L2 hit → ~7920 GB/s 聚合带宽 (vs HBM ~1152 GB/s)

**诊断**: 如果有 msprof 数据, 查看 L2 hit rate

### ④ 混合精度

**操作**: fp16 传输 + fp32 累加 → 传输量减半, 精度不丢

## 涉及 placeholder 引擎的处理

GM→L1, L1→L0, CubeUnit, L0→GM 全是 PLACEHOLDER:
- 可以提出优化建议 (如 "切换 pipeline")
- 但必须标注: **预计效果不确定, 需 910B3 实测**
- 不能做精确的速度预测 (如 "预计加速 1.3×")

## 操作步骤

1. 读 `engine_utilization` — 哪个引擎过载/空闲?
2. 评估 grid 调整
3. 评估 pipeline 切换 (如果是矩阵 op 却用 Vector pipeline)
4. 评估 L2 驻留 (如果工作集 < 192MB)
5. 所有建议标注实测要求

## 示例 Plan

```json
{
  "strategy": "adjust_grid_count",
  "reason": "GM→UB 利用率 44%, VecUnit 只 9% → transfer 等计算, 可以减少 transfer grid",
  "specific_change": "transfer grid: 20 → 10, compute grid: 保持不变 40",
  "expected_impact": "不确定 — 取决于硬件调度器, 需 910B3 实测",
  "verification_method": "910B3 真机 benchmark, grid=20 vs grid=10, 测 3 次取中位数"
}
```

```json
{
  "strategy": "switch_pipeline",
  "reason": "当前 op 是 matrix 类但用了 Vector Pipeline (GM→UB→VecUnit→UB→GM), Matrix Pipeline 可能更快",
  "specific_change": "换用 Matrix Pipeline: GM→L1→L0→Cube→L0→GM",
  "expected_impact": "不确定 (CubeUnit params are PLACEHOLDER), 需实测",
  "verification_method": "910B3 实测对比两个 pipeline 的 latency"
}
```
