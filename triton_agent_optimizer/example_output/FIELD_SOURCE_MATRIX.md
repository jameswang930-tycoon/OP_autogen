# 字段来源矩阵: msprof vs HIVMIR

> 最终完整流水线报告 = **msprof 解析** + **HIVMIR 补充** + **合并脚本处理**
>
> 本文档标注每个字段的来源、获取方式和示例值。

---

## 完整字段列表

### 1. EXECUTION SUMMARY

| 字段 | 来源 | 获取方式 | 示例值 |
|---|---|---|---|
| `total_ns` | ✅ **msprof** | trace.json: 最大 `ts + dur` 之差 | `3655.6` |
| `num_ops` | ✅ **msprof** | trace.json: 统计 Complete events | `3` |
| `execution_mode` | ✅ **msprof** | trace.json: 并行检测结果 | `"sequential"` |
| `num_cores` | ✅ **msprof** | trace.json: 不同 tid 数量 | `1` |

### 2. PER-OP STATISTICS (每 op)

| 字段 | 来源 | 获取方式 | 示例值 |
|---|---|---|---|
| `op_id` | ✅ **msprof** | 按时间排序后的序号 | `0` |
| `op_type` | ✅ **msprof** | trace.json: 通道映射 (MTE2→gm_to_ub, VECTOR→vadd, ...) | `"gm_to_ub"` |
| `engine` | ✅ **msprof** | trace.json: 通道映射 (MTE2→GM→UB, VECTOR→VecUnit, ...) | `"GM→UB"` |
| `start_ns` | ✅ **msprof** | trace.json: `ts` 字段 (μs→ns) | `0.0` |
| `end_ns` | ✅ **msprof** | trace.json: `ts + dur` (μs→ns) | `1621.6` |
| `duration_ns` | ✅ **msprof** | trace.json: `dur` 字段 (μs→ns) | `1621.6` |
| `time_ratio` | ✅ **msprof** | 计算: `duration_ns / total_ns` | `0.4436` |
| `pipeline_channel` | ✅ **msprof** | trace.json: `cat` 或 `name` 字段 | `"MTE2"` |
| `core_id` | ✅ **msprof** | trace.json: `tid` 映射到线程名 | `"core0.veccore0"` |
| `trace_event_name` | ✅ **msprof** | trace.json: `name` 字段 | `"gm_to_ub"` |
| `instruction` | ❌ **HIVMIR** | HIVMIR: 操作指令文本, e.g. `gm_to_ub(ub_1, gm_1)` | `"gm_to_ub(ub_1, gm_1)"` |
| `dst` | ❌ **HIVMIR** | HIVMIR: 目标 buffer 名 | `"ub_1"` |
| `src` | ❌ **HIVMIR** | HIVMIR: 源 buffer 名 | `"gm_1"` |
| `src2` | ❌ **HIVMIR** | HIVMIR: 第二源 (matrixmul) | `"l0_b1"` |
| `variable_name` | ❌ **HIVMIR** | HIVMIR: `%ub_1` | `"ub_1"` |
| `size_kb` | ❌ **HIVMIR** | HIVMIR: `memref<128KB>` | `128.0` |
| `memory_region` | ❌ **HIVMIR** | HIVMIR: buffer 前缀 (gm_→GM, ub_→UB, l1_→L1, l0_→L0) | `"UB"` |
| `effective_bw` | ❌ **HIVMIR+sim** | HIVMIR 提供 size → simulator 计算 (SATURATION_PARAMS) | `80.83` |
| `peak_bw` | ❌ **HIVMIR+sim** | 从 SATURATION_PARAMS 查询引擎峰值 | `80.83` |
| `bw_utilization` | ❌ **HIVMIR+sim** | 计算: `effective / peak` | `1.0` |
| `regime` | ❌ **HIVMIR+sim** | 基于 size 和 SATURATION_PARAMS 计算 | `"saturated"` |
| `wait_before_start_ns` | ❌ **HIVMIR** | HIVMIR: 依赖 op 的 end - 本 op 的 start | `1621.6` |
| `blocked_by` | ❌ **HIVMIR** | HIVMIR: RAW/WAR/WAW 解析, buffer 名, 类型 | `["op0(RAW on ub_1)"]` |

### 3. ENGINE UTILIZATION

| 字段 | 来源 | 获取方式 | 示例值 |
|---|---|---|---|
| 各引擎利用率 | ✅ **msprof** | 计算: `sum(engine_busy_ns) / total_ns` | `{"GM→UB": 0.44, ...}` |

### 4. BANDWIDTH UTILIZATION

| 字段 | 来源 | 获取方式 | 示例值 |
|---|---|---|---|
| 整体带宽利用率 | ❌ **HIVMIR+sim** | HIVMIR size × simulator SATURATION_PARAMS | (待 HIVMIR 补充后填入) |

### 5. PARALLELISM

| 字段 | 来源 | 获取方式 | 示例值 |
|---|---|---|---|
| `parallel_pairs` | ✅ **msprof** | trace.json: 时间重叠检测 | `[]` |
| 串行根因 | ❌ **HIVMIR** | HIVMIR: RAW/WAR/WAW 分析 | `"RAW on ub_1"` |

### 6. CRITICAL PATH

| 字段 | 来源 | 获取方式 | 示例值 |
|---|---|---|---|
| `path` (op chain) | ✅ **msprof** | trace.json: 最长持续时间链 | `[0, 1, 2]` |
| `length_ns` | ✅ **msprof** | trace.json: 链总时间 | `3655.6` |
| `edges` (边原因) | ❌ **HIVMIR** | HIVMIR: RAW/WAR/WAW 类型 + buffer 名 | `"RAW on ub_1"` |

---

## 统计

| 类别 | 字段数 | 说明 |
|---|---|---|
| ✅ msprof 直接提供 | **16** | 不需要额外处理 |
| ❌ HIVMIR 补充 | **9** | 纯 HIVMIR 提取 |
| ❌ HIVMIR + simulator 计算 | **4** | HIVMIR size → SATURATION_PARAMS 计算 |
| **总字段数** | **29** | — |

---

## 合并脚本职责 (后续实现)

```
msprof pipeline_report.json          HIVMIR analysis output
        │                                      │
        └────────────┬─────────────────────────┘
                     ▼
              dsl_merger.py
          (通过 op_id 对齐, 填充 "待补充" 字段)
                     │
                     ▼
            完整 DSL 流水线报告
       (对齐 cost_emulator/simulator.py --llm 格式)
```
