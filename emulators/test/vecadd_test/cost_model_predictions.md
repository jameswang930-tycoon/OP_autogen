# Cost Model 预测 vs 上板实测对比（vadd vectorAdd 实验）

verify_triton.py 的 before（10×1KB tile）vs after（1×10KB）：
- cost model 理论预测（聚合/单核口径 × 含/不含 store）
- 910B3 上板实测（grid=40，msprof）
- 两个方向的差异分析

---

## 1. cost model 理论预测（4 组合）

### 表1：不含 store（原版 DSL，load + vadd）

| 口径 | program | load（GM→UB）| vadd（VecUnit）| total | speedup |
|------|---------|-------------|---------------|-------|---------|
| 聚合 | before | 102.40 ns · 95% | 5.10 ns · 5% | **107.52 ns** | — |
| 聚合 | after | 8.20 ns · 86% | 1.40 ns · 14% | **9.59 ns** | **11.21×** |
| 单核 | before | 2048.00 ns · 91% | 204.80 ns · 9% | **2252.80 ns** | — |
| 单核 | after | 164.40 ns · 75% | 54.80 ns · 25% | **219.21 ns** | **10.28×** |

### 表2：含 store（完整 load + vadd + store）

| 口径 | program | load（GM→UB）| store（UB→GM）| vadd（VecUnit）| total | speedup |
|------|---------|-------------|--------------|---------------|-------|---------|
| 聚合 | before | 102.40 ns · 92% | 33.80 ns · 30% | 5.10 ns · 5% | **110.90 ns** | — |
| 聚合 | after | 8.20 ns · 49% | 7.10 ns · 42% | 1.40 ns · 8% | **16.67 ns** | **6.65×** |
| 单核 | before | 2048.00 ns · 88% | 675.90 ns · 29% | 204.80 ns · 9% | **2320.39 ns** | — |
| 单核 | after | 164.40 ns · 46% | 141.60 ns · 39% | 54.80 ns · 15% | **360.81 ns** | **6.43×** |

**口径说明**：
- 聚合：cost_emulator 原版（GM→UB 1500, UB→GM 实测 303/1445/1461, VecUnit 16000）— 多核聚合
- 单核：÷核数（MTE2/MTE3 ÷20, VecUnit ÷40）

---

## 2. 上板实测（grid=40，910B3 msprof）

| pipe | before | before 占比 | after | after 占比 | 实测 speedup |
|------|--------|------------|-------|-----------|-------------|
| aiv_vec（vadd 计算）| 5.504 us | 41.1% | 4.946 us | 41.8% | **1.11×** |
| aiv_scalar | 0.243 us | 4.4% | 0.161 us | 3.3% | 1.51× |
| aiv_mte2（load）| 1.576 us | 28.6% | 0.919 us | 18.6% | **1.71×** |
| aiv_mte3（store）| 1.701 us | 30.9% | 0.672 us | 13.6% | **2.53×** |

占比之和（before 105%, after 77.3%）≠ 100% —— before 有 pipe 重叠，after 有 22.7% 空闲/同步。

---

## 3. 方向1：优化幅度（before→after speedup）—— 实测为什么远小于预测

| engine | 实测 speedup | 预测 speedup（单核含 store）| 预测高估 |
|--------|------------|---------------------------|---------|
| vec（vadd）| **1.11×** | 3.74× | 3.4× |
| load（mte2）| 1.71× | 12.5× | 7.3× |
| store（mte3）| 2.53× | 4.77× | 1.9× |
| **整体** | **~1.1-1.5×** | **6.65×** | — |

### 根因 1：vec 实测不变（1.11×），但预测降 3.74× —— VecUnit 模型错了

实测 vec before/after 几乎一样（5.504 vs 4.946us）。vec 是**计算**：before 10×1KB vadd 和 after 1×10KB vadd 的总 FLOPs 完全相同（5120 elem × 40 threads）。vec 时延按总计算量，**不随 tile 大小变**。

但 cost model 把 VecUnit 建成 size-dependent（1KB floor 2000 → 24KB ramp 16000 GB/s），假设"小 vadd 慢、大 vadd 快"，预测降 3.74×。**计算单元应按 FLOPs（size-independent，像 CubeUnit 的 flat 模型），不是 transfer size** —— 这是建模错误。

### 根因 2：load/store 实测降幅（1.71/2.53×）远小于预测（12.5/4.77×）—— 多核分摊了 floor 惩罚

cost model 预测 load 12.5× 的依据：单 program 孤立 1KB 在 floor 区（100 GB/s，6.7% 利用），合并 10KB 进 ramp（1246 GB/s）→ 跃升 12.5×。

实测 grid=40（40 核各 1 program），40 个 1KB load 在 40 核**并行/流水**：多核分摊了小 tile 启动延迟，floor 惩罚没单 program 那么狠。实测 load 只降 1.71×。

### 方向1 结论

实测优化幅度小，因为 ① vec 模型错（计算不随 tile）+ ② 多核削弱搬运的 size-dependent 收益。cost model 的 size-dependent 假设在"单 program 孤立小 tile"下成立，但在 grid=40 多核 + 计算单元上不成立。

---

## 4. 方向2：after 实测 vs 理论预测 —— 为什么差这么多

以 after（1×10KB）为例：

| | 实测（grid=40）| 预测（单 program，单核含 store）|
|---|---|---|
| vec（vadd）| **4.946 us** | 54.8 ns |
| mte2（load）| 0.919 us | 164.4 ns |
| mte3（store）| 0.672 us | 141.6 ns |
| scalar | 0.161 us | 0（没建）|
| wall-clock | **~11.8 us**（反推）| 360.81 ns |

差 **~33×**（11.8us vs 360.81ns）。根因：

### 根因 1：规模不同 —— 预测单 program，实测 grid=40（40 program）

cost model 是**单核单 program** 模型（不建 grid）。预测 after 360.81ns = 一个 program（10KB chunk）时延。实测是 40 program 在 40 核跑（40×10KB=400KB 总数据）。**规模差 40×**，根本不是一回事。

即使 40 核全并行（wall-clock ≈ 单 program），实测 11.8us 也远大于 360.81ns —— 说明 40 核没做到"全并行等价单 program"，有大量开销。

### 根因 2：vec 严重没吃满（占 41.8% 但算力利用率 0.45%）

after vec 实测 4.946us（最大头）。算利用率：
- 总 vadd 计算 = 40 threads × 5120 elem × 1 FLOP = 204,800 FLOPs
- vec time 4.946us → **0.041 TFLOPS**
- 910B3 vec 聚合算力 9.216 TFLOPS → **利用率 0.45%**

grid=40 但每个 program 的 vadd 计算量太小（5120 elem），40 核的 vec 算力根本喂不饱。vec pipe 在 wall-clock 期间持续占用（4.946us，41.8%）但没满载 —— 这是 after 实测大的主因。

### 根因 3：scalar/启动/同步/空闲 —— 预测全没建

after 占比和只有 77.3%（vec 41.8 + scalar 3.3 + mte2 18.6 + mte3 13.6），**剩 22.7% 是空闲/同步/调度**。cost model 完全没建（无 scalar engine、无启动开销、无多核同步）。

### 方向2 结论

after 实测 vs 预测差 33×，主因：① 规模（单 program vs 40 program）② vec 严重没吃满（0.45% 利用）③ scalar/启动/同步（22.7%）没建。

---

## 5. 共同根因 + 修复建议

| 根因 | 方向1（优化幅度）| 方向2（绝对值）| 修复方向 |
|------|----------------|---------------|---------|
| **VecUnit size-dependent 模型错**（计算应按 FLOPs，不随 tile）| ✓ 主因 | ✓ | VecUnit 改 flat（size-independent，参照 CubeUnit）|
| **不建多核/grid** | ✓（多核削弱 floor）| ✓ 主因 | 加 grid/核数维度 |
| **不建 scalar/启动/同步** | 部分 | ✓（22.7%）| 加 scalar engine + 启动开销 |
| 小数据量算力闲置 | — | ✓（vec 0.45%）| — |

**最该修（按优先级）**：
1. **VecUnit 改 size-independent（flat，按 FLOPs）** —— 两个方向的最大偏差源，且 CubeUnit 已有 flat 模式可参照
2. **建多核维度**（grid/核数）—— 否则任何 grid>1 的 kernel 预测都对不上（单 program 假设根本不对应实际多核 kernel）

这两个是 cost model 的**结构性缺口**，单靠调带宽（聚合/单核）补不了。

---

## 6. 上板对照基准

真实 kernel（verify_triton.py）有 `tl.store`，应对照**含 store 表**：
- after 聚合预测 **16.67 ns** / 单核 **360.81 ns**
- before 聚合预测 **110.90 ns** / 单核 **2320.39 ns**
- 上板实测 after wall-clock **~11.8 us**

注：cost model 原版 summary.txt 的 11.2× 是**不含 store** 聚合口径，不对应真实 kernel；真实含 store，对照 6.65×。
