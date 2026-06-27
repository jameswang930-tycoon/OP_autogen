# vecadd_test — cost model 上板校验用例

**同一 vadd kernel**（`verify_triton.py`），在 cost model 不同**带宽口径**下的预测对照 + 上板校验。

## verify_triton.py（合作方原版 kernel，上板用）

- 来源：`costModel/cost_emulator/experiments/1.vectorAdd/verify_triton.py`
- before/after 两个 kernel（10×1KB tile vs 1×10KB single），40 threads × 10KB chunk，fp16
- 上板测：before vs after 实测 speedup
- 注：kernel 不随 cost model 带宽口径变化（带宽口径只改 cost model 预测，不改 kernel 代码）

## cost_model_predictions.md（预测对照表）

记录 4 组合（聚合/单核 × 含/不含 store）的 before/after 各 engine 时延 + 占比 + speedup：

| 口径 | after 含 store 预测 | before 含 store 预测 |
|------|--------------------|---------------------|
| 聚合（原版）| 16.67 ns | 110.90 ns |
| 单核（÷20/÷40）| 360.81 ns | 2320.39 ns |

- **聚合**：VecUnit 16000(40核), MTE2 1500(20核), MTE3 实测 1461(20核)
- **单核**：÷核数 → VecUnit 400, MTE2 75, MTE3 73

## .plan_percore.json

单核口径 plan code（含 raw_llm 全文），after 预测 219.21ns（不含 store）/ 360.81ns（含 store）。

## 对照目的

上板实测 before/after 的 total，对照两种口径预测，判断 cost model 带宽该用**聚合**还是**单核**口径。

## 注意

- `verify_triton.py` 是**上板脚本**（需 torch + triton + 真实硬件 CUDA/NPU），本地 .venv 没 triton 跑不了
- cost model 带宽修正（单核 ÷20/÷40 + UB→GM 实测 303/1445/1461）在 `cost_emulator/simulator.py` 本地，**不推远端**
