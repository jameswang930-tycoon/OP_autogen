# Tier 2: Operator Fusion

> **核心原则: 融合在大结构确定后做, 收益最大 (2-1000×)。改融合后重新评估 Tiling。**

## 融合决策树 (2025 标准)

```
1. ops 是否 RAW 链串行?      → 否 → 不能融合
2. 中间结果被多个 op 复用?    → 是 → 不能融合 (如 skip connection)
3. 融合后 UB 总大小 < 192KB?  → 否 → 不能融合 (或分批融合)
4. tiling 规则一致?           → 否 → 不能融合 (BLOCK_SIZE 不同)
5. → 融合
```

## 三类融合

### ① 逐元素融合 (vadd+vmul+vrelu+vdiv...)

**识别**: 连续 VecUnit op 在 RAW 链上, 中间 buffer 只被 1 个 op 消费

**操作**:
1. 读 `dependencies_summary.raw` — 找 RAW 链
2. 读 `buffers` — 检查 consumers 数量
3. 检查融合后 UB 用量: `sum(size_kb of all live buffers) ≤ 192KB`
4. 合并为一个 kernel

**预期收益**: 消除中间 UB→GM→UB 往返, 每个融合 op 省 ~1 次 GM 读写, 加速 1.3-2×

### ② 激活融合 (MatMul/Cube + ReLU/SiLU/GELU)

**识别**: CubeUnit 后紧跟 VecUnit 激活

**操作**: 在 Matrix Pipeline 的 epilogue 阶段直接写入激活

**910B3 注意**: Cube→Vec 转换可能增加 L0→UB 开销, 需实测

### ③ WAR 打破

**识别**: `dependencies_summary.war` 非空

**操作**:
1. 找到 WAR 涉及的 buffer
2. 分配新 buffer (新变量名, e.g. 从 `ub_1` → `ub_1_new`)
3. 修改 dst operand → 消除 WAR 假依赖

**检查**: 新 buffer 的 UB 用量 `新 size_kb + 原有 live buffers ≤ 192KB`

**预期收益**: 解锁并行, `parallel_pairs` 从 0 → 1+, 加速 1.2-2×

## 融合后验证

1. **UB 容量检查**: 融合后所有 live buffer 的 `sum(size_kb) ≤ 192KB`
2. **数值验证**: CPU emulator, 原 kernel 和融合后 kernel 数值一致
3. **性能验证**: simulator --llm 对比, 检查 `parallel_pairs` 是否增加

## 操作步骤

1. 读 `dependencies_summary` — 找 RAW 链
2. 读 `buffers` — 看 producer→consumer 关系
3. 遍历 RAW 链: 连续 VecUnit → 融合; Cube+激活 → 融合
4. 遍历 WAR: 可避免的 → 分配新 buffer
5. 验证 UB 容量不超 192KB
6. 生成融合代码

## 示例 Plan (逐元素融合)

```json
{
  "strategy": "fuse_elementwise_ops",
  "reason": "vadd(ub_2, ub_1) → vmul(ub_3, ub_2) 串行, ub_2 只被 vmul 消费, 可融合",
  "specific_change": "合并 vadd+vmul 为单个 kernel: ub_3 = (ub_1 + scalar) * multiplier, 消除中间 ub_2",
  "expected_impact": "省 1 次 UB→GM→UB 往返, ops 从 3→2, 预计加速 1.3-1.5×",
  "verification_method": "CPU emulator + simulator --llm 对比 + 检查 UB 用量"
}
```

## 示例 Plan (WAR 打破)

```json
{
  "strategy": "break_war_dependency",
  "reason": "op2 writes ub_1, but op0 still reads ub_1 → WAR. 分配 ub_1_new 打破假依赖",
  "specific_change": "op2 的 dst 从 ub_1 改为 ub_1_new, 让 op0 和 op2 可以并行",
  "expected_impact": "解锁 op0∥op2 并行, parallel_pairs 0→1, 预计加速 1.2-1.3×",
  "verification_method": "simulator --llm 检查 parallelism 变化"
}
```
