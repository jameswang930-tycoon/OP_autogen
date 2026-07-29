# DSL 流水线解析文档 — 完整字段定义与解析规则

> 所有信息来自 Ascend 官方文档，禁止幻觉。

---

## 数据源 1: HIVM MLIR → 11 语义字段

### 来源

`TRITON_KERNEL_DUMP=1` 产生 `hivmir/*.mlir`

参考: https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/developer_guide/dialects/HIVMDialect.md

### HIVM MLIR 语法

```
hivm.hir.load  ins(%src : memref<WxHxdtype, #hivm.address_space<gm>>) 
               outs(%dst : memref<WxHxdtype, #hivm.address_space<ub>>)

hivm.hir.store ins(%src : memref<WxHxdtype, #hivm.address_space<ub>>) 
               outs(%dst : memref<WxHxdtype, #hivm.address_space<gm>>)

hivm.hir.vadd  ins(%lhs, %rhs : type, type) outs(%result : type)
hivm.hir.vsub  ins(%lhs, %rhs : type, type) outs(%result : type)
hivm.hir.vmul  ins(%lhs, %rhs : type, type) outs(%result : type)
hivm.hir.vdiv  ins(%lhs, %rhs : type, type) outs(%result : type)
hivm.hir.vexp  ins(%src : type) outs(%result : type)
hivm.hir.vsqrt ins(%src : type) outs(%result : type)
hivm.hir.vmax  ins(%lhs, %rhs : type, type) outs(%result : type)
hivm.hir.vmin  ins(%lhs, %rhs : type, type) outs(%result : type)
hivm.hir.vabs  ins(%src : type) outs(%result : type)
hivm.hir.vtanh ins(%src : type) outs(%result : type)
hivm.hir.vlog  ins(%src : type) outs(%result : type)
hivm.hir.vrsqrt ins(%src : type) outs(%result : type)

hivm.hir.matmul  ins(%a, %b : type, type) outs(%c : type)
hivm.hir.pipe_barrier
hivm.hir.sync_block
hivm.hir.set_flag / hivm.hir.wait_flag

memref.alloc() : memref<shape×dtype, #hivm.address_space<ub/gm/l1>>
```

### 地址空间

| 属性 | 含义 |
|------|------|
| `#hivm.address_space<gm>` | Global Memory (HBM) |
| `#hivm.address_space<ub>` | Unified Buffer (片上) |
| `#hivm.address_space<l1>` | L1 Cache |

### 解析规则 — `hivmir_analyzer.py` 输出

| # | 字段 | 解析规则 | 示例 |
|---|------|---------|------|
| 1 | `op_id` | 顺序编号, 从 0 开始 | `0` |
| 2 | `op_type` | HIVM op 名映射: `hivm.hir.load`→`gm_to_ub`, `hivm.hir.store`→`ub_to_gm`, `hivm.hir.vadd`→`vadd`, `hivm.hir.vexp`→`vexp`, `linalg.reduce`→`reduce`, `hivm.hir.matmul`→`matmul`, `memref.alloc`→`alloc` | `gm_to_ub` |
| 3 | `instruction` | HIVM 操作原文 (截取关键行) | `hivm.hir.load ins(%arg0) outs(%ub_0)` |
| 4 | `dst` | `outs(%name ...)` 中提取 SSA 变量名, 去 `%` 前缀 | `ub_0` |
| 5 | `src` | `ins(%name ...)` 中提取第一个 SSA 变量名 | `arg0` |
| 6 | `src2` | `ins(..., %name2 ...)` 中提取第二个 SSA 变量名 (vadd/vmul 等二元 op) | `arg1` |
| 7 | `size_kb` | `memref<shape×dtype>` 解析: 计算 `product(shape) * dtype_bytes / 1024`。dtype 映射: `f16`→2, `f32`→4, `i32`→4, `i16`→2, `i8`→1。shape 中 `?` 视为 1 | `1.0` |
| 8 | `memory_region` | 从 `#hivm.address_space<X>` 提取: `gm`/`ub`/`l1`; alloc 看其 memref 的地址空间; store 看 outs 的目标空间 | `ub` |
| 9 | `variable_name` | SSA 变量名 (取自 dst 或第一个 alloc 的 %name) | `%ub_0` |
| 10 | `dependencies` | SSA def-use 链: 追踪每个 SSA 值在哪个 op 被定义(def), 在哪些 op 被使用(use)。RAW=先写后读, WAR=先读后写, WAW=写后写。格式: `[{"from_op_id": 0, "type": "RAW"}]` | `[{"from_op_id":0,"type":"RAW"}]` |
| 11 | `dtype` | 从 memref 类型中提取: `f16`/`f32`/`bf16`/`i32`/`i8` | `f32` |

---

## 数据源 2: msprof op simulator trace → 14 时序字段

### 来源

`msprof op simulator` 产生 `OPPROF_xxx/simulator/`

参考: https://gitcode.com/lishuokp31/msopprof/blob/260321/mssanitizer_version_adapt/docs/zh/msopprof_simulator_performance_data.md

参考: https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/debug_guide/profiling.html

### instr_exe.csv 字段 (官方文档)

| 列 | 含义 | 示例 |
|----|------|------|
| `instr` | 硬件指令名称 | `VADD`, `MOV_OUT_TO_UB`, `SET_FLAG`, `DATA_COPY_ND` |
| `addr` | 指令 PC 地址 | `0x282136772` |
| `pipe` | 流水线通道 | `VECTOR`, `MTE2`, `MTE3`, `CUBE`, `MTE1`, `SCALAR`, `ALL`, `FLOWCTRL`, `FIXP` |
| `call_count` | 调用次数 | `1` |
| `cycles` | 该指令执行的总 cycle 数 | `88` |
| `running_time(us)` | 执行时间 (微秒) | `0.050` |
| `detail` | 指令详细参数 | `XD:X3=0x4000, Dtype:F32, Id:49` |

### pipe → engine 映射 (triton-ascend 官方文档)

| pipe | Engine 名称 | Engine ID | 说明 |
|------|-----------|-----------|------|
| `MTE2` | GM→UB | 0 | Vector 路径 DMA 读。若关联 matmul op, 映射为 GM→L1 (Engine 3) |
| `MTE3` | UB→GM | 1 | DMA 写回 |
| `VECTOR` | VecUnit | 2 | 向量计算 |
| `MTE2` (matmul路径) | GM→L1 | 3 | Cube 路径 DMA 读 |
| `MTE1` | L1→L0 | 4 | Cube 内部搬运 |
| `CUBE` | CubeUnit | 5 | 矩阵乘 |
| `FIXP` | L0→GM | 6 | FIXPIPE 写回 (A2 系列) |
| `SCALAR` | — | — | 标量运算/地址计算 |
| `ALL` | — | — | 同步屏障 |
| `FLOWCTRL` | — | — | 控制流 |

### trace.json 字段

| 字段 | 含义 |
|------|------|
| `ts` | 时间戳 (微秒) |
| `dur` | 持续时间 (微秒) |
| `name` | 指令名称 |
| `pid` | 进程 ID |
| `tid` | 线程 ID (对应 core 编号) |
| `args` | 额外参数 (pipe, 地址等) |
| `cat` | 分类 (VECTOR, MTE2, etc.) |

### 解析规则 — `msprof_analyzer.py` 输出

| # | 字段 | 解析规则 | 示例 |
|---|------|---------|------|
| 1 | `engine` | instr_exe.csv `pipe` 列 → engine 名称 (查上表) | `VecUnit` |
| 2 | `pipeline_channel` | instr_exe.csv `pipe` 列原始值 | `VECTOR` |
| 3 | `duration_ns` | instr_exe.csv `running_time(us)` × 1000 | `50.0` |
| 4 | `start_ns` | 从 trace.json 提取: `ts` (微秒) × 1000, 按指令累计 | `0.0` |
| 5 | `end_ns` | `start_ns + duration_ns` | `50.0` |
| 6 | `time_ratio` | `duration_ns / total_ns` | `0.0337` |
| 7 | `cycles` | instr_exe.csv `cycles` 列 | `88` |
| 8 | `total_ns` | trace.json 中最大 `ts + dur` × 1000, 或所有指令 duration_ns 之和取最大 | `1484.0` |
| 9 | `num_ops` | instr_exe.csv 行数 (所有核汇总) | `512` |
| 10 | `execution_mode` | trace.json 重叠检测: 任两条指令时间重叠 → `parallel`, 否则 `sequential` | `parallel` |
| 11 | `num_cores` | `core*.veccore*` 目录数量 | `8` |
| 12 | `engine_utilization` | 按 pipe 聚合 duration_ns ÷ total_ns, 再映射到 engine | `{"GM→UB":0.31, "VecUnit":0.10}` |
| 13 | `parallel_pairs` | trace.json 中时间重叠的指令对: `[{op_a:0, op_b:2, overlap_ns:810}]` | `[...]` |
| 14 | `critical_path` | 最长不重叠指令链 (贪心: 按 start_ns 排序, 选不重叠的最长序列) | `[0, 1, 3, 5]` |

---

## 数据源 3: 合并 → 29 字段

### 合并算法 — `dsl_merger.py`

**核心问题**: HIVM 的 ops 和 msprof 的 instructions 粒度不同。

- HIVM: 1 个 `gm_to_ub` + 1 个 `vadd` + 1 个 `ub_to_gm` = 3 个 ops
- msprof: 可能有数十条 MTE2/VECTOR/MTE3 指令

**合并策略**:

1. **按 engine 类型聚合 msprof 指令**: 同 pipe 的多条指令合并为一个统计单元
2. **按顺序对齐**: HIVM ops 按 op_id 顺序, msprof 聚合后按首次出现顺序
3. **互填**: HIVM 有但 msprof 没有的字段 → 用 HIVM 补充; msprof 有但 HIVM 没有的 → 用 msprof 补充
4. **特殊处理**: MTE2 pipe → 如果 HIVM op 是 matmul 相关, 映射为 Engine 3 (GM→L1); 否则映射为 Engine 0 (GM→UB)

### 910B3 硬件参数 — SATURATION_PARAMS

来源: config.py 中硬编码 (来自华为官方 benchmark 实测验证)

| Engine ID | Engine 名称 | vpeak | k0 (KB) | peak_clamp (GB/s) | 状态 |
|-----------|-----------|-------|---------|--------------------|------|
| 0 | GM→UB | 121.08 | 6.65 | 80.83 | MEASURED |
| 1 | UB→GM | 190.19 | 10.72 | 76.67 | MEASURED |
| 2 | VecUnit | 461.0 | 4.50 | 404.0 | MEASURED |
| 3 | GM→L1 | 37.5 | 6.65 | 37.5 | PLACEHOLDER |
| 4 | L1→L0 | 100.0 | 6.65 | 100.0 | PLACEHOLDER |
| 5 | CubeUnit | 150.0 | 0 | 150.0 | PLACEHOLDER |
| 6 | L0→GM | 37.5 | 6.65 | 37.5 | PLACEHOLDER |

### 计算字段

| # | 字段 | 计算规则 |
|---|------|---------|
| 25 | `effective_bw_gb_s` | `size_kb / duration_ns` (KB → GB: size_kb / 1024^2, ns → s: duration_ns / 1e9) |
| 26 | `peak_bw_gb_s` | 从 SATURATION_PARAMS 查对应 engine 的 peak_clamp |
| 27 | `bw_utilization` | `effective / peak` |
| 28 | `regime` | `bw_util ≥ 0.95` → `saturated`; `> 0.50` → `ramp`; `≤ 0.50` → `floor` |

### 完整 29 字段对照表

| # | 字段 | 来源 | 说明 |
|---|------|------|------|
| 1 | op_id | HIVM | 操作编号 |
| 2 | op_type | HIVM | 操作类型 |
| 3 | instruction | HIVM | HIVM 操作原文 |
| 4 | dst | HIVM | SSA 目标 |
| 5 | src | HIVM | SSA 源 |
| 6 | src2 | HIVM | SSA 第二源 |
| 7 | size_kb | HIVM | buffer 大小 |
| 8 | memory_region | HIVM | 地址空间 |
| 9 | variable_name | HIVM | SSA 变量名 |
| 10 | dependencies | HIVM | RAW/WAR/WAW |
| 11 | dtype | HIVM | 数据类型 |
| 12 | engine | msprof | 7-engine 名称 |
| 13 | pipeline_channel | msprof | 原始 pipe |
| 14 | duration_ns | msprof | 耗时 (ns) |
| 15 | start_ns | msprof | 起始时间 |
| 16 | end_ns | msprof | 结束时间 |
| 17 | time_ratio | msprof | 时间占比 |
| 18 | cycles | msprof | cycle 数 |
| 19 | total_ns | msprof | 总耗时 |
| 20 | num_ops | msprof | 指令数 |
| 21 | execution_mode | msprof | 并行/串行 |
| 22 | num_cores | msprof | 核数 |
| 23 | engine_utilization | msprof | engine 占比 |
| 24 | parallel_pairs | msprof | 并行对 |
| 25 | critical_path | msprof | 关键路径 |
| 26 | effective_bw_gb_s | 计算 | 有效带宽 |
| 27 | peak_bw_gb_s | 计算 | 峰值带宽 |
| 28 | bw_utilization | 计算 | 带宽利用率 |
| 29 | regime | 计算 | 饱和状态 |

---

## 910B3 内存规格 (官方文档确认)

| 层级 | 容量 | 来源 |
|------|------|------|
| HBM (Global Memory) | 64 GB | ascend-dmi 实测 |
| L2 Cache | ~MB 级, 跨核共享 | 架构文档 |
| L1 Buffer | ~128 KB, 每 AI Core | 架构文档 |
| Unified Buffer (UB) | ~128 KB, 每 Core | 架构文档 |
| L0 Buffer | 每计算单元, 极小 | 架构文档 |

## 910B3 核心数 (官方文档确认)

| 类型 | 数量 | 来源 |
|------|------|------|
| AI Cores | 32 (910B3) | 架构文档 |
| Vec Cores | 40 (每 AI Core 内) | config.py (bench 实测) |

---

## SOC 版本选择

910B3 → `--soc-version=Ascend910B3`
对应 simulator lib 路径: `tools/simulator/dav_2201/lib/`
