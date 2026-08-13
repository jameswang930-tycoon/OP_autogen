# 14 算子验收结果（2026-08-13，910B3 真机）

> 来源：优化循环最终产出（我们 Event 端到端 vs 工业级最优端到端）。
> 口径：**验收 = 工业级最优端到端(us) ÷ 我们最优 Event 端到端(us)**（≥1.0 快于工业级 / 0.8~1.0 打平 / <0.8 有空间）。
> **链路核对结论（2026-08-13 逐行排查）**：两边计时方法完全一致（Event 设备侧 + 多窗口 median + 破 L2）；
> 窗口语义一致（都是一次"完整调用"的设备侧耗时，输入均预创建不在窗口内）；
> 差异 = 工业级 torch 多 kernel 链的固有成本（kernel 间下发 gap + forward 内部中间张量分配）vs
> 我们融合后 kernel 数更少 + 连续 launch gap≈0 + 中间结果预分配——**这是真实优化收益（融合/单遍），对比公平**。

## 验收表（字母序）

| 算子 | 我们 us | 工业级 us | 验收 | 判定 | 轮数 |
|---|---|---|---|---|---|
| attention_mlp | 1315 | 1837 | 1.40x | 快于工业级 | 20 |
| conv2d | 392 | 131 | 0.33x | 有空间 | 15 |
| conv_bias_relu | 419 | 167 | 0.40x | 有空间 | 27 |
| flash_attention | 13681 | 427 | 0.03x | 有空间（⚠数据异常） | 8 |
| fused_add_mul | 91 | 113 | 1.24x | 快于工业级 | 10 |
| layernorm | 92 | 107 | 1.16x | 快于工业级 | 13 |
| matmul | 295 | 543 | 1.84x | 快于工业级（⚠需复核 mode） | 12 |
| matmul_relu | 240 | 282 | 1.18x | 快于工业级 | 19 |
| matmul_transpose | 110 | 317 | 2.88x | 快于工业级 | 21 |
| rms_norm | 91 | 236 | 2.59x | 快于工业级 | 15 |
| rms_norm_residual | 96 | 250 | 2.60x | 快于工业级 | 12 |
| sigmoid | 82 | 80 | 0.98x | 打平 | 11 |
| softmax | 82 | 89 | 1.09x | 快于工业级 | 10 |
| vector_add | 89 | 86 | 0.97x | 打平 | 8 |

汇总：平均 1.33x | 最快 2.88x | 最慢 0.03x | 快于 9 / 打平 2 / 有空间 3

## 合理性判定（链路核对后）

### ✅ 可信（同口径下的真实优化收益）
- **attention_mlp 1.40x / fused_add_mul 1.24x / matmul_relu 1.18x / softmax 1.09x / layernorm 1.16x**：
  融合链/单遍减少 kernel 数与下发 gap——与工业级 eager（分 kernel 链）同窗口语义对比，真实收益
- **rms_norm 2.59x / rms_norm_residual 2.60x / matmul_transpose 2.88x**：
  单遍合并（1 kernel vs torch 多 kernel）+ 预转置连续访存。两边 Event 都是一次调用的设备侧耗时，
  **kernel 数少 = 优化成果本身**，不是测量差异；us 级算子 torch 的 gap/中间分配占比高，我们消除后
  差距大属合理（这正是融合/单遍优化的目标）
- **sigmoid 0.98x / vector_add 0.97x**：带宽型，两头接近，打平合理

### ⚠ 需留意（不是测量错误，但要复核对象）
- **matmul 1.84x**：工业级最优 mode 需复核——若"最优"来自 eager 的 MLP 全链（3 kernel + host 下发）
  而 compile 不可用/回退，则差距含"我们无 gap"的收益；若 compile 真融合成功，差距会缩小。
  用 `bench_all --skip-existing` 看各 mode 明细确认
- **conv2d 0.33x / conv_bias_relu 0.40x**：合理但该修——kernel 无 tl.dot（vector 模拟），结构层没做
  （见 playbook tier1 教学 2 im2col）

### ❌ 数据异常（必须重测）
- **flash_attention 0.03x（13681us = 13.7ms，仅 0.31 TFLOPS）**：即使不剪枝，fp16 FA + 合理 BLOCK
  也该有 5+ TFLOPS → 怀疑 verify Event 测量异常（注入/循环/尺寸）或优化过早停止（仅 8 轮）。
  **下一步：重测 FA**（`feedback/remeasure_best.py --op flash_attention` + 检查注入产物）

## 结论与后续
- **快 2-3 倍是真实优化收益**（融合/单遍/预转置消除 torch 多 kernel 链的 gap 与中间分配），
  不是测量作弊——两边 Event 同口径（一次调用设备侧耗时、median、破 L2）
- 待办：① FA 重测（异常）② conv 补 im2col（结构层）③ matmul mode 明细复核
