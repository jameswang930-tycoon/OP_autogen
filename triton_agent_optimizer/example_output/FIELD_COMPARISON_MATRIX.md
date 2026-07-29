# 三数据源 29 字段对比矩阵

> 项目目标: 合并 cost_emulator(DSL) + HIVM IR(MLIR) + msprof(trace) → 29 字段全填充
> 更新: 2026-07-28 (msprof op simulator 验证完成, HIVM dialect 确认)
> 文件参考: `simulator_full_output.txt`, `msprof_add_instr_exe.csv`, `msprof_matmul_cube_instr_exe.csv`

---

## 执行摘要

| 数据源 | 能提供 | 环境要求 | 状态 |
|---|---|---|---|
| **cost_emulator** (simulator.py) | 16 字段 (timing/engine/bw/regime) | 纯 Python | ✅ 已验证 |
| **HIVM IR** (bishengir-compile) | 9 字段 (buffer/deps/instruction) | WSL2 + CANN 9.0 | ✅ 已验证 |
| **msprof simulator** (trace + CSV) | 8 核心字段 (真实指令级 pipeline) | WSL2 + CANN 9.0 | ✅ 已验证 |

**综合**: 三个数据源合并可达 **28/29 字段全填充**，仅 `variable_name` 可从 HIVM 提供但精度取决于编译器。

---

## 完整 29 字段 × 三数据源 对比

### SECTION 1: EXECUTION SUMMARY (4 字段)

| # | 字段 | cost_emulator | HIVM IR | msprof trace/CSV | 合并方案 |
|---|---|---|---|---|---|
| 1 | `total_ns` | ✅ 估算值 (BW公式) | ❌ | ✅ **精确** (trace 最后 ts+dur) | **msprof 优先** |
| 2 | `num_ops` | ✅ | ✅ op 个数 | ✅ 指令条数 | HIVM (语义op) + msprof (指令数) |
| 3 | `execution_mode` | ✅ parallel/sequential | ❌ | ✅ trace 重叠检测 | **msprof 优先** |
| 4 | `num_cores` | ❌ | ❌ | ✅ core*.veccore* 目录数 | **msprof 优先** |

### SECTION 2: TIME BREAKDOWN (已覆盖)

| # | 字段 | cost_emulator | HIVM IR | msprof | 合并方案 |
|---|---|---|---|---|---|
| 5 | `time_breakdown[]` | ✅ 按 op 排序 | ❌ | ✅ 按指令排序 | **msprof timing + HIVM op 名** |

### SECTION 3: PER-OP STATISTICS (核心 20 字段)

| # | 字段 | cost_emulator | HIVM IR | msprof (instr_exe.csv) | 合并方案 |
|---|---|---|---|---|---|
| 6 | `op_id` | ✅ 按序编号 | ✅ op_id | ❌ (指令地址) | HIVM op_id 对齐 msprof pipe |
| 7 | `op_type` | ✅ gm_to_ub/vadd/... | ✅ **精确** (hivm.hir.vadd等) | ❌ (指令名: VADD/MOV_OUT_TO_UB) | **HIVM 优先** (语义op) |
| 8 | `engine` | ✅ GM→UB等 | ✅ 派生自 op_type | ✅ **pipe 列** (MTE2/VECTOR/CUBE/MTE1/MTE3) | **msprof 优先** (硬件精确) |
| 9 | `instruction` | ⚠️ 简单文本 | ✅ **完整** (gm_to_ub(buf, src)) | ✅ **指令名+detail** (VADD Dtype:F32) | **HIVM 语义 + msprof 硬件** |
| 10 | `dst` | ❌ | ✅ buffer 名 (%ub_1) | ⚠️ detail 中有 XD:X3=0x4000 | **HIVM 优先** |
| 11 | `src` | ❌ | ✅ buffer 名 (%gm_1) | ⚠️ detail 中有 XN:X0=地址 | **HIVM 优先** |
| 12 | `src2` | ❌ | ✅ (matmul) | ❌ | **HIVM 优先** |
| 13 | `size_kb` | ⚠️ 从 DSL 读取 | ✅ **精确** (memref<1024×f16>) | ⚠️ 从 detail 推算 (如 0x4000=16KB) | **HIVM 优先** |
| 14 | `variable_name` | ❌ | ✅ %ub_1, %gm_1 | ❌ | **HIVM 提供** |
| 15 | `memory_region` | ⚠️ DSL 推断 | ✅ **#hivm.address_space\<gm/ub/l1\>** | ⚠️ 从 pipe+detail 推断 | **HIVM 优先** |
| 16 | `duration_ns` | ✅ 估算 (BW÷size) | ❌ | ✅ **精确** (running_time(us)×1000) | **msprof 优先** |
| 17 | `start_ns` | ✅ 估算 | ❌ | ✅ **trace.ts 列** | **msprof 优先** |
| 18 | `end_ns` | ✅ start+dur | ❌ | ✅ **trace.ts + trace.dur** | **msprof 优先** |
| 19 | `time_ratio` | ✅ dur/total | ❌ | ✅ **dur/total (msprof)** | **msprof 优先** |
| 20 | `effective_bw_gb_s` | ✅ SATURATION_PARAMS 公式 | ❌ | ⚠️ 可从 size÷time 计算 | **cost_emulator 公式** |
| 21 | `peak_bw_gb_s` | ✅ 从 SATURATION_PARAMS 查 | ❌ | ❌ | **cost_emulator 优先** |
| 22 | `bw_utilization` | ✅ effective/peak | ❌ | ⚠️ 可从 msprof timing+size 算 | **cost_emulator 公式** |
| 23 | `regime` | ✅ floor/ramp/saturated/flat | ❌ | ❌ | **cost_emulator 优先** |
| 24 | `wait_before_start_ns` | ✅ 依赖检测 | ⚠️ SSA def-use 间接 | ✅ **trace 相邻指令 gap** | **msprof 优先** |
| 25 | `blocked_by` | ✅ engine serialization | ✅ **RAW/WAR/WAW 依赖链** | ✅ **SET_FLAG/WAIT_FLAG 同步** | **HIVM + msprof 互补** |

### SECTION 4-7: 聚合统计 (4 字段)

| # | 字段 | cost_emulator | HIVM IR | msprof | 合并方案 |
|---|---|---|---|---|---|
| 26 | `engine_utilization` | ✅ 所有 7 引擎 | ❌ | ✅ **pipe 列聚合** | **msprof 优先** (真实占比) |
| 27 | `bandwidth_utilization` | ✅ | ❌ | ❌ | **cost_emulator 优先** |
| 28 | `parallel_pairs` | ✅ overlap 检测 | ❌ | ✅ **trace 重叠检测** | **msprof 优先** |
| 29 | `critical_path` | ✅ engine serialization 链 | ⚠️ dep chain | ✅ **trace 最长依赖链** | **msprof 优先** |

---

## msprof instr_exe.csv 完整字段对照

从 `msprof_add_instr_exe.csv` 每行可提取:

| CSV 字段 | 示例值 | 对应 29 字段 |
|---|---|---|
| `instr` | `VADD`, `MOV_OUT_TO_UB` | `instruction` (硬件级) |
| `addr` | `0x282136772` | (调试用) |
| **`pipe`** | `MTE2`, `MTE3`, `VECTOR`, `CUBE`, `MTE1`, `SCALAR`, `ALL`, `FLOWCTRL` | **`engine`** (精确对应 7 引擎) |
| `call_count` | `1` | — |
| `cycles` | `88` | 用于精确周期分析 |
| **`running_time(us)`** | `0.050` | **`duration_ns`** (×1000) |
| **`detail`** | `XD:X3=0x4000,XN:X5=0,Dtype:F32` | `src`, `dst`, `size_kb`, 数据类型 |

### pipe → engine 精确映射

| msprof pipe | 7-engine 模型 | 触发方式 |
|---|---|---|
| `MTE2` | Engine 0 (GM→UB) / Engine 3 (GM→L1) | DataCopy/load | 
| `MTE3` | Engine 1 (UB→GM) | DataCopy/store |
| `VECTOR` | Engine 2 (VecUnit) | Add/VecAdd |
| `MTE1` | Engine 4 (L1→L0) | Cube pipeline |
| `CUBE` | Engine 5 (CubeUnit) | matmul |
| — | Engine 6 (L0→GM) | matmul 隐式 |
| `SCALAR` | 地址计算 (非传输/计算) | 开销 |
| `ALL` | 同步屏障 | BAR/PipeBarrier |
| `FLOWCTRL` | 控制流 | JUMP/END |

### detail 字段可提取的额外信息

```
MOV_OUT_TO_UB:  Src:OUT, Dst:UB, XD:X2=0x2000 (8KB), XM:X4=0x1000010, id:4
  → 数据大小: 8KB, 源=GM, 目的=UB, 操作ID=4

VADD:  XD:X3=0x4000, Dtype:F32, Id:49
  → 数据大小: 16KB, 数据类型=float32, 操作ID=49

MOV_UB_TO_OUT:  Src:UB, Dst:OUT, XD:X0=0x11315a00 (目标地址), id:6
  → UB→GM搬运到地址 0x11315a00
```

---

## HIVM IR (MLIR格式) 可提取字段

从 `bishengir-opt` IR dump 解析：

| HIVM 字段 | 示例 | 对应 29 字段 |
|---|---|---|
| `memref.alloc()` | `memref<1024xf16, #hivm.address_space<ub>>` | `size_kb`, `memory_region` |
| `hivm.hir.load/ store` | `ins(%arg0) outs(%alloc)` | `op_type`, `src`, `dst` |
| `hivm.hir.vadd/ vmul/...` | `ins(%a,%b) outs(%c)` | `op_type`, `src`, `src2`, `dst` |
| `hivm.hir.matmul` | 含 `block_sizes=[16,16,16]` | `op_type`, tiling 参数 |
| `#hivm.address_space<gm/ub/l1>` | 显式地址空间 | `memory_region` |
| `%buf_a = memref.alloc()` | SSA 值 | `variable_name` |
| RAW/WAR/WAW (SSA def-use) | op0写%ub1, op1读%ub1 | `blocked_by`, `dependencies` |
| `hacc.entry`, `function_kind<DEVICE>` | 函数属性 | kernel 元数据 |

---

## 距离 29 字段全填充还缺什么

### ✅ 已覆盖 (28/29)

三个数据源合并可覆盖 28 字段。

### ❌ 仍缺 (1/29)

| 字段 | 缺失原因 | 解决方式 |
|---|---|---|
| `pipeline_channel` | msprof pipe 可替代, 但 channel 粒度不同 | msprof pipe 直接映射到 engine; channel 名(如 `MTE2`) 可替代原设计的 `pipeline_channel` |

### ⚠️ PLACEHOLDER → 实测 (Engine 3-6)

| 引擎 | 旧状态 | 新状态 |
|---|---|---|
| Engine 0 (GM→UB) | ✅ MEASURED | ✅ msprof 已验证 |
| Engine 1 (UB→GM) | ✅ MEASURED | ✅ msprof 已验证 |
| Engine 2 (VecUnit) | ✅ MEASURED | ✅ msprof 已验证 |
| Engine 3 (GM→L1) | ❌ PLACEHOLDER | 🟡 msprof pipe=MTE2 可区分但需 matmul 触发 |
| Engine 4 (L1→L0) | ❌ PLACEHOLDER | ✅ msprof pipe=MTE1 已触发 |
| Engine 5 (CubeUnit) | ❌ PLACEHOLDER | ✅ msprof pipe=CUBE 已触发 |
| Engine 6 (L0→GM) | ❌ PLACEHOLDER | 🟡 隐式在 matmul, 需更多分析 |

### 核心差距总结

1. **timing 精度**: cost_emulator 的 `duration_ns` 是 BW÷size 公式估算, msprof 的是**周期精确**仿真 → **应切换到 msprof**
2. **pipeline channel 粒度**: msprof 的 `pipe` 字段已经是硬件级流水线通道, 比我们设计中的 `pipeline_channel` 更精确
3. **Engine 6 (L0→GM)**: matmul 隐式管理, 需要单独运行 L0→GM DMA 测试 kernel
4. **内存层级数据**: msprof `detail` 字段包含具体地址和数据大小, 可精确计算 UB/L1/L0 占用
