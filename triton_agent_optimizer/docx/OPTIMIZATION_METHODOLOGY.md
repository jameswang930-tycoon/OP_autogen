# Triton Kernel 优化方法论 — 正确优化顺序

> 基于 2024-2025 年学术研究和工业实践 (TritonForge, StepTronOSS, Liger-Kernel, ThunderKittens, CuAsmRL, GoPTX)

---

## 核心原则: 从结构影响最大到最小

```
高层 (改变数据流/结构) ──────────────────────────→ 低层 (微调参数)
                                                    
  Algorithm  →  Fusion  →  Tiling  →  Memory  →  Compute  →  Arch
  (算法选择)   (算子融合)  (分块大小)  (内存访问)  (计算调优)  (硬件专属)
                                                    
  改了底层后, 如果上层又变了 → 底层白做 → 必须从上到下
```

## 为什么旧顺序是错的

旧顺序: Block Size → Memory → Fusion → Compute → Arch → Algorithm

问题:
1. **先调 tile size (Tier 1), 后融合 (Tier 3)** → 融合后 kernel 结构变了, tile size 要重新调 → 白做
2. **先做内存优化 (Tier 2), 后改算法 (Tier 6)** → 算法重构后整个数据流变了 → 白做
3. **Fusion 和 Algorithm 是结构性变更, 应该最先做** — 它们决定 kernel 的形态

## 正确的新顺序

```
Tier 1: Algorithmic Structure     ← 最先: 选择正确的算法
Tier 2: Operator Fusion           ← 其次: 融合消除中间存储
Tier 3: Tiling & Block Config     ← 然后: 在稳定的融合结构上调 tile
Tier 4: Memory Access             ← 再: 内存访问优化
Tier 5: Compute & Occupancy       ← 再: 计算调优
Tier 6: 910B3 Architecture        ← 最后: 硬件专属微调
```

---

## 详细说明

### Tier 1: Algorithmic Structure (算法结构)

**为什么最先**: 算法决定了整个 kernel 的数据流。如果后面改算法, 前面所有优化全部作废。

**触发条件**: 任何 kernel 的第一轮优化 (round0 之后)

**优化内容**:

| 算子类型 | 标准算法 | 优化变体 | 何时使用 |
|---|---|---|---|
| Softmax | Naive softmax | **Online Softmax** (数值稳定+单pass) | 默认选择 |
| MatMul | Tiled matmul | **Split-K** (大K维度分解) | K > 4096 |
| LayerNorm/RMSNorm | Two-pass (mean+variance) | **One-pass Welford** (在线方差) | 默认选择 |
| Attention | Standard attention | **Flash Attention** (SRAM-only) | 序列长度 > 512 |
| CrossEntropy | Separate softmax+CE | **Online softmax + fused CE** | 默认选择 |
| Generic | Single kernel | **Persistent Kernel** (减少launch) | 小grid, 高launch开销 |

**决策依据**:
- 分析 round0 的 `merged/final_report_llm.txt` — 看 `execution_mode`
- 如果 `execution_mode = sequential` 且 ops > 10 → 考虑 persistent kernel
- 如果是归约类算子 (softmax/norm) → 默认用 online 算法

**验证**: 算法变更后必须通过 CPU Emulator 全部 shape/dtype 测试

**晋升条件**: 算法已经是最优选择, 或无算法改进空间 → 进入 Tier 2

---

### Tier 2: Operator Fusion (算子融合)

**为什么第二**: 融合消除中间 GM 读写, 收益最大 (2-1000×)。但必须在算法确定后才能做——如果算法后面改了, 融合结构可能要重新设计。

**触发条件**: Tier 1 完成后, 分析 round0 `dependencies_summary`

**优化内容**:

| 融合类型 | 识别条件 | 预期收益 | 910B3 注意 |
|---|---|---|---|
| **逐元素融合** | RAW chain 上的 vadd→vmul→vrelu 等 | 2-3× (消除中间UB↔GM) | 融合后 UB 总量 ≤ 192KB |
| **激活融合** | GEMM/Conv + ReLU/SiLU/GELU | 1.5-2× | 同上 |
| **残差融合** | Add + LayerNorm/RMSNorm | 2-5× | 单次 GM 读写完成 add+norm |
| **WAR 打破** | `dependencies_summary.war` 非空 | 解锁并行 (1.2-2×) | 分配独立 buffer (额外 UB) |

**融合决策树**:
```
1. ops 是否串行执行? → 否 → 不能融合
2. 中间结果是否被多个后续 op 复用? → 是 → 不能融合
3. 融合后 UB 总大小 < 192KB? → 否 → 不能融合 (或分批)
4. tiling 规则是否一致? → 否 → 不能融合
5. → 融合
```

**关键**: 融合后必须重新分析 DSL 流水线。融合后的 kernel 可能暴露新的瓶颈。

**晋升条件**: 所有可融合的 op 已融合, 或 UB 容量不再允许融合 → 进入 Tier 3

---

### Tier 3: Tiling & Block Config (分块和启动配置)

**为什么第三**: 分块参数必须在融合后的稳定 kernel 结构上调整。如果在融合前调了 tile size, 融合后要重新调。

**触发条件**: Tier 2 完成后 (或无融合机会)

**优化内容**:

| 参数 | 作用 | 搜索空间 | 910B3 约束 |
|---|---|---|---|
| **BLOCK_SIZE** | 每次 load/store 的 tile 大小 | 256 ~ 65536 (2的幂) | ≤ UB_CAPACITY / n_bufs (=192KB / n_bufs) |
| **num_warps** | 每个 SM 的 warp 数 | 1~8 | — |
| **num_stages** | 软件流水线级数 | 0~4 | 0=单GEMM, 1=双GEMM/非GEMM |
| **grid** | 并行 program 数 | 20 (AI Core) / 40 (Vec Core) | 传输用20, 计算用40 |

**tile size 选择启发式**:
```
1. 计算 max_tile = UB / (n_bufs × 2B)  # 2B = fp16
2. 找到瓶颈 op 的 k0 (半饱和点)
3. 目标 tile: k0 × 2 ~ max_tile 之间
4. 如果 tile > k0 × 2 → 带宽进入饱和区 → 收益递减
5. 如果 tile < k0 → 带宽在 ramp/floor → 优先增大
```

**晋升条件**: 连续 3 轮 tile 调优无显著改进 (bw_util 变化 < 5%) → 进入 Tier 4

---

### Tier 4: Memory Access (内存访问优化)

**为什么第四**: 在 tile size 确定后, 优化数据的存取方式。

**触发条件**: Tier 3 完成后

**优化内容**:

| 优化手段 | 适用场景 | 方法 |
|---|---|---|
| **小传输合并** | 多个 < k0 的同类型传输 | 合并为一次 > k0 的传输 |
| **Coalescing** | 非连续内存访问 | 确保 load/store 是对齐的连续访问 |
| **Double Buffering** | 传输和计算串行 | 两个 buffer 轮流: 一个在计算时另一个在传输 |
| **Prefetch** | 计算前数据未就绪 | 提前 load 到 UB |
| **L2 驻留** | 工作集 < 192MB | 数据保持在 L2, 避免 HBM 访问 |

**依据**: 分析 `pipeline_report.json` 中每个 op 的 `bw_utilization` 和 `regime`
- `regime=floor` (bw_util < 50%) → 合并小传输
- `regime=ramp` (50~95%) → 可能还能增大
- `regime=saturated` (>95%) → 已达到峰值, 需要减少数据量或用更快的通路

---

### Tier 5: Compute & Occupancy (计算和占用率)

**为什么第五**: 计算优化是最细粒度的调整, 在其他结构都稳定后再做。

**触发条件**: Tier 4 完成后

**优化内容**:

| 优化手段 | 适用场景 | 方法 |
|---|---|---|
| **计算-传输重叠** | VecUnit 等待 MTE 完成 | Double buffer + pipeline |
| **向量化** | 逐元素操作 | 确保一次处理多个元素 |
| **精度取舍** | fp16 精度足够 | fp32 accumulate → fp16 compute |
| **Occupancy 调优** | SM 利用率低 | 调整 num_warps, 减少 register 压力 |
| **Warp 级原语** | 需要 warp 内通信 | tl.warp_id, tl.reduce |

---

### Tier 6: 910B3 Architecture Specific (硬件专属)

**为什么最后**: 这些是最专用的优化, 只在以上所有层级都试过后才使用。

**触发条件**: Tier 5 完成后, 或 placeholder engines (3/4/5/6) 成为瓶颈

**优化内容**:

| 优化手段 | 适用场景 | 方法 |
|---|---|---|
| **Grid 数选择** | transfer 和 compute 分配不均 | transfer=20 (AI Core), compute=40 (Vec Core) |
| **Pipeline 切换** | 当前 pipeline 不是最优 | Vector Pipeline ↔ Matrix Pipeline |
| **L2 驻留最大化** | 工作集 < 192MB | 调整 access pattern 提升 L2 hit rate |
| **Cube Core 利用** | 矩阵运算未用 Cube | 切换到 Matrix Pipeline (GM→L1→L1→L0→Cube) |
| **混合精度** | 瓶颈在传输 | fp16 传输 + fp32 累加 |

---

## 降级规则 (Descend): 结构性变更后允许回退

旧方案中 Tier 只能晋升不能降级 → 如果融合后又改了算法, 无法回到算法层。

**新规则**:

```
如果当前优化导致了结构性变更 (改变了 op 数量、顺序、或数据流):
  1. 融合了新算子 → 重新评估 Tiling (回到 Tier 3)
  2. 改变了算法结构 → 重新评估 Fusion (回到 Tier 2)
  3. 改变了 pipeline (Vector↔Matrix) → 重新评估 Tiling (回到 Tier 3)
```

## 停止条件 (全部层级)

同一 Tier 内连续 3 轮无改进 → 晋升到下一 Tier。
到达 Tier 6 且连续 3 轮无改进 → 停止 (或满足其他全局停止条件)。

## 与其他 2025 框架的对比

| | 我们的方案 | AutoKernel | TritonForge |
|---|---|---|---|
| **顺序** | Algorithm→Fusion→Tile→Mem→Compute→Arch | Tile→Mem→Fusion→Compute→Arch→Algo | Profile-driven (动态) |
| **理由** | 结构影响从大到小 | (经验性排序) | 根据 profiling 信号动态选择 |
| **降级** | ✅ 支持 (结构性变更后回退) | ❌ 不支持 | N/A (动态) |
| **融合决策** | 有明确决策树 | 无 | 无 |

---

## 参考文献

- **TritonForge** (2025): Profiling-guided agentic loop, 1.76× avg speedup
- **StepTronOSS Workflow**: 11-step checklist for production Triton kernels
- **Liger-Kernel** (2024): Partial fusion strategy
- **ThunderKittens** (2025): Fusion first, then tiling — "Tiles simply make everything easier"
- **CuAsmRL** (CGO 2025): Hierarchical search (configs → SASS scheduling)
- **GoPTX** (DAC 2025): PTX-level instruction weaving after fusion+tiling
- **AMD ROCm Triton Guide** (2025): Occupancy, autotuning, ISA analysis
