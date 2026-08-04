# 6 优化策略 → 精准数据映射 + diagnosis.json 字段清单 + 论文对比

> 2026-08-03。回答：① 每个优化策略精准看什么 → 映射哪些字段 → 字段来自哪个阶段；② diagnosis.json 输出字段清单；③ 我们的 6 层 vs 华为论文的 4 层诊断；④ 弱 LLM（不改代码）怎么针对性教它优化。

---

## 一、6 策略 → 精准数据 → 字段 → 来源

| Tier | 优化策略 | 精准看什么信号 | 映射字段 | 来源(阶段) |
|---|---|---|---|---|
| **1 算法** | 换算法/结构（online softmax/split-K/persistent） | ①执行模式(串/并) ②op 形态(是否归约) ③grid/launch ④哪些 op 占大头 | `execution_mode`, `num_ops`, `ops[].op_type序列`, `ops[].duration_ns`, `bottlenecks.tier1.top_ops` | hivm(2)+sim(3) |
| **2 融合** | 消除中间存储/激活融合/WAR打破 | ①依赖 RAW/WAR/WAW ②GM↔UB 往返次数(ub_to_gm→gm_to_ub链) ③逐元素链(vadd→vmul) ④UB容量 | `dependencies[]`, `ops[].transfer_path`, `bottlenecks.tier2.war_deps/fusion_candidates` | hivm(2) |
| **3 分块** | BLOCK_SIZE/num_stages/k0 | ①每通路真实带宽利用率 ②regime(floor/ramp/sat) ③k0半饱和点 ④max_tile | `transfer_paths[].real_bw_gb_s/bw_utilization/regime`, `ops[].size_kb` | board(4)+hivm(2) |
| **4 访存** | 小传输合并/double buffer/L2驻留 | ①小传输(<k0) ②L2命中率 ③搬运数据块大小 ④传输引擎 | `ops[].data_size_bytes/size_kb`, `summary.l2_hit_rate`, `bottlenecks.tier4.small_transfer_ops` | sim(3)+board(4) |
| **5 计算** | 计算-传输重叠/向量化/精度 | ①引擎利用率(cube/vec/mte) ②气泡(Vec等MTE) ③cube/vec fops ④指令级重叠 | `summary.engine_utilization`, `bottlenecks.tier5.cube_fops/vector_fops`, `ops[].cycles` | board(4)+sim(3) |
| **6 架构** | grid/流水/L2驻留/混合精度 | ①pipe利用率 ②L2命中 ③Block Num ④引擎占比 | `summary.engine_utilization/l2_hit_rate/num_cores`, `transfer_paths` | board(4) |

---

## 二、diagnosis.json 输出字段清单（当前实现）

### 顶层结构（6 大块）

| 块 | 说明 | 字段数 |
|---|---|---|
| `summary` | 端到端/核数/执行模式/L2/引擎占比 | 6 |
| `ops[]` | 每 op：结构+sim时序+真机带宽/L2+依赖 | 每 op ~20 |
| `transfer_paths[]` | 每通路：真实带宽/有效带宽/利用率/regime | 每通路 ~10 |
| `dependencies[]` | RAW/WAR/WAW 边 | 每边 4 |
| `bottlenecks{}` | 每 Tier 信号+优化提示 | 6 块 × ~5 信号 |
| `meta` | 来源/时间戳 | ~4 |

### ops[] 每 op 的 20 字段

```
op_id op_type transfer_path path_desc          ← 结构/引擎
dst src src2 dst_region                         ← 从哪搬到哪
size_kb dtype attrs                             ← 数据块大小/类型/tiling配置
duration_ns cycles pipe call_count             ← sim 指令时序
real_duration_ns real_bw_gb_s l2_hit           ← 真机耗时/带宽/L2
dependencies sim_instr                         ← 依赖边/对齐指令
```

### transfer_paths[] 每通路的 10 字段

```
path desc num_ops total_size_kb total_duration_ns
real_bw_gb_s peak_bw_gb_s bw_utilization regime effective_bw_gb_s
```

### bottlenecks{} 每 Tier 的信号（示例）

```
tier1_algorithm: execution_mode num_ops top_ops → hint
tier2_fusion:    war_deps war_buffers fusion_candidates → hint
tier3_tiling:    path_regimes path_effective_bw → hint
tier4_memory:    l2_hit_rate small_transfer_ops → hint
tier5_compute:   cube_fops vector_fops → hint
tier6_arch:      block_num engine_utilization → hint
```

**合计**：顶层 6 块 + ops/通路/依赖/瓶颈信号 ≈ **40+ 个不同字段概念**（每个 op/通路重复）。

---

## 三、我们 6 层 vs 华为论文 4 层诊断

### 论文：Compiler-Grounded Hierarchical Diagnosis（4 层，按需升级）

```
L1 模式三查   → 优化 pattern/启发式/快速修复
L2 Profiling  → 瓶颈检测/热点定位/指标解读/生成假设     ← 用时延+硬件计数器
L3 IR归因     → IR 模式匹配/内存布局/融合调度/向量化      ← 用 dump 的 MLIR
L4 编译器知识 → pass 行为/lowering 规则/约束/限制        ← 查编译器源码
   （浅层证据不足 → 升级到深层；最终给出带证据的源码级改写）
```

**论文关键洞察**（对我们极重要）：
> "profiling 能可靠定位**症状**（movement-heavy/scalar-heavy/serialization-heavy），
>  但定位不了**机制**（是 work 分配？IR 布局？还是后端 pass 前提不满足？）。"
> → 所以必须从 profiling 升级到 IR 归因，才能知道"为什么"。

### 我们 6 层 vs 论文 4 层

| 维度 | 我们 6 层 | 论文 4 层 |
|---|---|---|
| 是什么 | **优化空间**（可做什么优化，固定顺序结构→微调） | **诊断过程**（按证据决定做什么优化，按需升级） |
| 怎么选 | 固定顺序：算法→融合→...→架构 | 证据驱动：看 profiling/IR 信号决定该层 |
| 优点 | 覆盖全、不遗漏、结构影响从大到小 | 精准、省成本、不盲试 |
| 缺点 | 可能盲试（每层都试才知道），弱 LLM 容易跑偏 | 需要较强的诊断能力+编译器知识 |

### 结论：两者互补，应**融合**为"证据驱动的 6 层优化"

> **用论文的分层诊断机制，在我们的 6 层空间里选"该优化哪层"。**
> 即：先看 profiling/IR 信号 → 决定是算法问题(去 Tier1) 还是融合问题(去 Tier2) 还是带宽问题(去 Tier3/4) → 只动那一层。

**诊断信号 → 该去哪个 Tier**（确定性规则，脚本算，不靠 LLM）：
```
串行+多op+归约   → Tier1 算法
WAR依赖多/GM往返  → Tier2 融合
通路regime=ramp/floor → Tier3 分块
L2命中低/小传输    → Tier4 访存
cube/vec fops失衡  → Tier5 计算
pipe利用率低/核少  → Tier6 架构
```

---

## 四、弱 LLM（不改代码）怎么针对性教它

**关键原则：诊断 = 脚本确定性算；LLM 只做"读结论 + 套模板改代码"。**

### 1. 诊断完全脚本化（不靠 LLM 推理）
- `integrate.py` 输出 `bottlenecks.<tier>.hint` 是**确定性结论**（如"GM→UB regime=ramp → tile 过小，增大 BLOCK"）
- LLM 不用理解 40 个字段，只看每个 Tier 的 **hint（一句话结论 + 具体改法）**

### 2. 每个 bottleneck → 具体代码改法模板（喂给 LLM）
```
hint=通路ramp/tile小 → 改 triton_kernel.py 的 BLOCK_M/N/K 增大
hint=WAR依赖        → 给中间结果换独立 buffer（分配新变量）
hint=L2命中低       → 合并小 load 成一次大 load
hint=串行+归约      → 换 online/persistent 模板
```
LLM 不需要"想"，只需要"按模板改 kernel.py 对应行"。

### 3. 6 层空间告诉 LLM "有哪些可能"，diagnosis 告诉它 "该用哪个"
- 每轮：诊断脚本 → 输出 1 个确定性的 `hint`（该优化哪层+怎么改）
- LLM：读 hint → 改 kernel.py → 重跑诊断 → 看 hint 变没变（改进判断）
- **不盲目 6 层全试**，而是让证据指路

---

## 五、下一步增强（让 Tier1/2 更"一针见血"）

当前 tier1/2 信号较粗（op数/WAR数），可加：
1. **GM 往返检测**：`ub_to_gm → gm_to_ub` 相邻链（数据写了又读回=融合目标）
2. **RAW 逐元素链**：连续 VecUnit 且无 GM 写 = 融合候选（给 LLM 具体 op id）
3. **归约形态识别**：reduce 类 op 序列 → Tier1 算法决策
4. **每通路 peak 校准**：从 board Memory.csv 实测最大带宽 → bw_utilization 才准（Tier3/4 前提）

要不要我把这些增强写进 `integrate.py`，让 tier1/tier2 直接给出"融合哪几个 op / 换什么算法"的具体结论？
