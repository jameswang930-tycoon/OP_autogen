# vecadd_test — cost model 上板校验用例

两个对照（**同 vadd kernel**，区别在 cost model 的**带宽口径**）：

## 任务1: `verify_triton.py`（合作方原版，聚合口径）

- 来源：`costModel/cost_emulator/experiments/1.vectorAdd/verify_triton.py`
- cost model 带宽：**聚合**（VecUnit 16000 GB/s = 40 核，MTE2 1500 GB/s = 20 核）
- before/after kernel（10×1KB tile vs 1×10KB single），40 threads × 10KB chunk，fp16
- 预测：after total **9.59 ns**，before→after speedup **11.2×**
- 上板测：before vs after 实测 speedup，对比预测 11.2×

## 任务3: `verify_triton_percore.py` + `.plan_percore.json`（单核口径）

- cost model 带宽：**单核**（VecUnit 400 = 16000÷40，MTE2 75 = 1500÷20，MTE3 37.5 = 750÷20）
  — 本地临时改 `simulator.py` 的 `BANDWIDTH_CURVE_GB_S` 生成，**生成后已恢复**（不推 cost_emulator 远端）
- 同 vadd after kernel（1×10KB single transfer），40 threads，fp16
- 预测：after total **219.21 ns**（单核口径，vs 聚合 9.59 ns）
- 上板测：同 kernel，对照单核预测 219 ns

## 对照目的

验证"聚合带宽 vs 单核带宽"哪个更接近实际。**同 kernel**（vadd 不变），不同 plan code 口径，上板测对照：
- 聚合口径预测 9.59 ns（偏理想，假设单核能跑满 40 核聚合算力）
- 单核口径预测 219.21 ns（每核按 1/40 算力）

上板实测落在哪个区间，就能判断 cost model 的带宽该用聚合还是单核口径。

## 注意

- `verify_triton.py` / `verify_triton_percore.py` 是**上板脚本**（需 torch + triton + 真实硬件 CUDA/NPU），本地 .venv 没 triton 跑不了，只能上板
- `.plan_percore.json` 是单核口径 plan code（含 raw_llm 全文），用于对照预测
