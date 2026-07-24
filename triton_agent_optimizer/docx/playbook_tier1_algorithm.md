# Tier 1: Algorithmic Structure

> **核心原则: 算法决定数据流。改了算法后面所有优化都要重做——所以算法必须最先定。**

## 诊断依据

从 `execution_summary` + `time_breakdown` + `dependencies_summary` 读取:
- `execution_mode = sequential` 且 `num_ops > 5` → Persistent Kernel 候选
- 有归约 op (softmax/norm/reduce) → Online 算法候选
- 有 MatMul + `K > 4096` → Split-K 候选
- `num_ops > 20` → 算法层面有问题, 需要重构

## 算子 → 推荐算法

| 当前算法 | 推荐算法 | 何时用 | 预期收益 | 原因 |
|---|---|---|---|---|
| Naive softmax (3-pass) | **Online Safe Softmax** | 默认 | 2-3× | 单 pass, 消除中间 tensor |
| Two-pass LayerNorm | **One-pass Welford** | 默认 | 1.5-2× | 在线方差, 减少归约 |
| Standard Attention | **Flash Attention** | seq_len > 512 | 2-5× | SRAM-only, O(N²)→O(N) |
| Naive CrossEntropy | **Fused CE + online softmax** | 默认 | 3× | 消除中间 logits tensor |
| Large K MatMul | **Split-K** | K > 4096 | 1.6-3× | 增加 SM 利用率 |
| 多个小 kernel | **Persistent Kernel** | grid < 20 | 1.2-1.5× | 减少 launch overhead |
| Broadcast-heavy | **Streamed GEMV** | M 很小 | 2-5× | 消除 broadcast |

## 决策流程

```
1. 读 merged 报告的 execution_summary
2. 对照上表, 当前算子是哪个? → 是否有更优算法?
3. 检查 structural_issues (从 bottleneck_diagnoser)
4. 有更优算法 → 生成算法替换计划
5. 无更优算法 → 明确标注 "当前算法已最优" → 晋升 Tier 2
```

## 910B3 注意

- **Persistent Kernel**: grid ≤ 40 (Vec Core 上限), 每个 Core 分到的数据要整除
- **Split-K**: K 维度分片必须整除 16 (fp16 SIMD 宽度)
- **Flash Attention**: 需要 UB=192KB 能放得下 tile, 大 seq_len 可能溢出

## 操作步骤

1. 确认当前算子类型 (op_type 汇总)
2. 对照上表确定最优算法
3. 如果当前算法 ≠ 最优 → 生成替换计划
4. 如果当前已经是最优 → `strategy: "algorithm_already_optimal"`, 本轮仍记 KEEP

## 示例 Plan

```json
{
  "strategy": "switch_to_online_softmax",
  "reason": "当前 naive softmax 需要 3 次遍历 (max→exp→sum→div), online softmax 只需 1 次",
  "specific_change": "用 running max + online normalization 替换三 pass softmax",
  "expected_impact": "消除 2 次中间 GM 读写, op 数从 ~10 降到 ~5, 预计加速 2-3×",
  "verification_method": "CPU emulator: shape=(256,512,1024,4096), dtype=fp16, 对比 numerical error"
}
```
