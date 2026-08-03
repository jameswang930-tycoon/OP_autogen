# 完整字段与流水线知识图谱 (PIPELINE_FIELDS)

> 2026-08-03 定版。本文件回答三个问题：① 完整流程怎么跑；② 29 字段分别来自哪、干什么用、哪些是真实哪些是模拟；③ 6 层优化各需要什么前提知识。
> 配套：`README.md`（架构/流程）、`example_output/FIELD_SOURCE_MATRIX.md`（旧版字段来源，已过时）、`claude_resume_summary/SESSION_SUMMARY_2026-08-03_hivm_verified_final.md`（今日工作定版）。

---

## 〇、test vs kernel：两个都要，且已正确分离

| 文件 | 角色 | 被谁用 |
|---|---|---|
| `input/matmul/triton_kernel.py` | **真正代码**：`@triton.jit matmul_kernel`，被优化的对象 | Coder 只改它；test_matmul.py import 它 |
| `input/matmul/test_matmul.py` | **测量场景**：host 驱动（建 NPU tensor → launch kernel → sync），头注释含全部命令 | `msprof ... python3 test_matmul.py` 在服务器跑 |

**结论**：不需要二合一，分离是正确设计。
- `triton_kernel.py` = 优化输入（Coder 每轮只改 kernel）
- `test_matmul.py` = 稳定测量脚手架（每轮不变，import 最新 kernel）
- 已通过 `from triton_kernel import matmul_kernel` 关联（test_matmul.py L260）
- 二合一的问题：Coder 重写 kernel 时会把 host 启动代码一起改掉，测不准

**跑通的关键**：kernel 不带 `num_warps/num_stages`（triton-ascend 禁止 tune），grid 传 tuple，`torch.npu.set_device(0)`。

---

## 一、完整流水线（端到端，服务器上跑）

```
input/matmul/triton_kernel.py + test_matmul.py
   │
   ├─① bishengir D 打印 → hivm_try.txt        (真实 HIVM, 结构字段源)
   ├─② msprof op simulator → sim_prof         (真实指令级时序源)
   └─③ msprof op 真机 → board_prof            (真实 L2/带宽/端到端源)
   │
   ├─ hivmir_analyzer(hivm_try.txt)  → 结构字段
   ├─ msprof_analyzer(sim_prof)      → 指令级时序字段 (+ 待写 parse_hardware_dir 读 board_prof)
   └─ dsl_merger(两报告)             → 29 字段 merged_report.json
   │
   └─ bottleneck_diagnoser → data_extractor → Planner(LLM) → Coder → Verifier → RecordManager
```

**每轮优化后**：Coder 改 `triton_kernel.py` → 服务器重跑 ① ② ③ → 新 merged_report → 下一轮。

---

## 二、全部字段总表（来源 + 用途 + 真实/估算）

### A. Execution Summary（4 项，聚合级）

| 字段 | 来源文件 | 获取方式 | 用途 | 真实性 |
|---|---|---|---|---|
| `total_ns` | sim_prof/trace.json 或 board_prof/OpBasicInfo | trace 最大 ts+dur；或真机 Task Duration(us)×1000 | 端到端延迟，加速比基准 | ✅ 真实 |
| `num_ops` | hivmir + msprof | HIVM op 数（含 sync） | 复杂度 | ✅ 真实 |
| `execution_mode` | sim_prof/trace.json | 时间重叠检测 → sequential/parallel | Tier1 串/并决策 | ✅ 真实 |
| `num_cores` | sim_prof 或 board_prof | trace tid 数；或真机 Block Dim | Tier6 grid/占用 | ✅ 真实 |

### B. Per-Op 结构字段（11 项，来自真实 HIVM = hivm_try.txt）

| 字段 | 来源 | 用途 | 真实性 |
|---|---|---|---|
| `op_id` | HIVM 顺序（含 sync op 计入） | 全链对齐主键 | ✅ 真实 |
| `op_type` | HIVM op 名（gm_to_ub/vadd/mmadL1/set_flag...） | 引擎映射、瓶颈分类 | ✅ 真实 |
| `engine` | OP_TO_ENGINE（mmadL1→CubeUnit, sync→Sync） | engine_utilization | ✅ 真实 |
| `instruction` | HIVM 完整 op 文本 | LLM 可读、与 simulator 指令对齐 | ✅ 真实 |
| `dst` / `src` / `src2` | HIVM operands | 依赖分析 | ✅ 真实 |
| `size_kb` | alloc memref 尺寸×dtype（2D 已支持；`?`动态维=0 未知） | 带宽计算 | ✅ 真实（动态维待补） |
| `memory_region` | address_space 映射（cbuf→L1, cc→L0C, gm→GM） | 区域判断 | ✅ 真实 |
| `variable_name` | SSA 名 | 调试 | ✅ 真实 |
| `dtype` | memref dtype | 精度/尺寸 | ✅ 真实 |
| `attrs` | HIVM `{...}`（block_sizes 等） | tiling 判断 | ✅ 真实 |
| `dependencies` | RAW/WAR/WAW | 依赖链、串行根因 | ✅ 真实 |

### C. Per-Op 指令级时序字段（来自真实 simulator = instr_exe.csv）

| 字段 | 来源 | 获取方式 | 用途 | 真实性 |
|---|---|---|---|---|
| `duration_ns` | instr_exe.csv `running_time(us)` | ×1000 | 每 op 耗时（关键路径） | ✅ 真实 |
| `start_ns` / `end_ns` | trace.json 或对齐分配 | — | 重叠/气泡 | ✅ 真实 |
| `time_ratio` | duration/total | 计算 | 占比排序 | ✅ 真实（由真实 duration 派生） |
| `cycles` | instr_exe.csv `cycles` | 直接 | 周期数 | ✅ 真实 |
| `pipeline_channel` | instr_exe.csv `pipe` | 直接 | pipe 利用率 | ✅ 真实 |
| `core_id` | instr_exe.csv 所在核 | 直接 | 多核 | ✅ 真实 |
| `wait_before_start_ns` | HIVM 依赖计算 | 依赖 op end - 本 op start | 等待/气泡 | ⚠️ 依赖真实 duration 后可算 |

### D. Per-Op 带宽/regime 字段（4 项，**当前是估算占位，需真机校准**）

| 字段 | 来源 | 获取方式 | 用途 | 真实性 |
|---|---|---|---|---|
| `effective_bw_gb_s` | size_kb / duration_ns | 计算 | 实测带宽 | ✅ 基于真实 size+duration |
| `peak_bw_gb_s` | config.py `SATURATION_PARAMS`（7 引擎 vpeak/peak_clamp） | 查表 | 理论峰值 | ❌ **占位值，需真机 Memory.csv 校准** |
| `bw_utilization` | effective/peak | 计算 | 带宽利用率 | ⚠️ 被 peak 占位污染 |
| `regime` | SATURATION_PARAMS 曲线 + size | 曲线分类 | Tier3 分块决策 | ❌ **占位，需 size 扫描曲线** |

### E. 聚合/诊断字段

| 字段 | 来源 | 用途 | 真实性 |
|---|---|---|---|
| `engine_utilization` | msprof pipe→engine 聚合 | Tier5/6 引擎忙比 | ✅ 真实 |
| `parallelism.parallel_pairs` | trace.json 重叠检测 | Tier1 串/并 | ✅ 真实 |
| `critical_path.path/length_ns` | trace.json 最长链 | 瓶颈定位 | ✅ 真实（⚠️ 索引是 trace 事件序号，非 HIVM op_id，语义待定） |
| `dependencies_summary` | HIVM RAW/WAR/WAW | 串行根因 | ✅ 真实 |
| `buffers` | HIVM alloc | 容量判断 | ✅ 真实 |
| `time_breakdown` | 计算 | LLM 报告 | ✅ 派生 |

---

## 三、Mock vs 真实 现状（务必分清）

### ❌ 当前是模拟/占位的（需替换为真实）
| 项 | 位置 | 问题 |
|---|---|---|
| `peak_bw_gb_s` / `bw_utilization` / `regime` | config.py `SATURATION_PARAMS` L879-890 | vpeak/k0/peak_clamp 是**旧占位**（GM→UB 80.83 是 ÷20 单核值，真实峰值更高）→ 需真机 Memory.csv 每核实测校准 + size 扫描曲线 |
| `timing_estimator.py` | analyzers/ | WSL2 兜底 SAT 公式估算（不是真机产物） |
| `ttir_to_hivm.py` / `hivm_to_ascendc.py` | analyzers/ | WSL2 近似转换（cube 通路近似，非真实 HIVM） |
| `example_output/mock_*.json` | example_output/ | 示例数据，非真实产物 |
| `total_ns`/`num_ops` 等早期字段 | 旧 mock 报告 | 已被真实源取代 |

### ✅ 真实产物（服务器上已拿到并核验）
| 产物 | 内容 | 覆盖字段 |
|---|---|---|
| `hivm_try.txt` | 真实 HIVM（32 op 含 mmadL1 + 21 sync + 2D alloc） | 全部结构字段 |
| `sim_prof/instr_exe.csv + trace.json` | 24 核真实指令级时序 | 全部时序字段 |
| `board_prof/OpBasicInfo + PipeUtilization + Memory` | 真机 op 级聚合 | L2/带宽/端到端校准（**待接入 parse_hardware_dir**） |

### ⚠️ 待完成（缺了就凑不齐 29 字段）
1. `msprof_analyzer.parse_hardware_dir()`（#4）：把 board_prof 接入，校准 peak_bw
2. size 扫描（Tier3/4 的 k0 曲线）：9 个尺寸档位扫带宽 → 拟合 vpeak/k0/peak_clamp
3. `critical_path` 索引语义：trace 事件号 → HIVM op_id 映射

---

## 四、6 层优化前提知识（每层要什么字段/知识）

| Tier | 决策内容 | 需要的字段/知识 | 来源 |
|---|---|---|---|
| **1 算法结构** | Online Softmax/Split-K/Persistent | 执行模式(串/并)、op 数、per-op 时长、是否归约 | execution_mode + HIVM op 序列 |
| **2 算子融合** | 逐元素融合/WAR打破/激活融合 | 依赖边 RAW/WAR/WAW + buffer 名、per-op 时长、UB 容量 | HIVM dependencies + size_kb + memory_capacity |
| **3 分块配置** | BLOCK_SIZE/num_stages | regime、bw_utilization、k0 半饱和点、max_tile | SATURATION_PARAMS 校准后 + size 扫描 |
| **4 访存** | 小传输合并/double buffer/coalescing | 小传输(<k0)、地址/stride、传输-计算重叠、L2 命中 | instr_exe 指令级 + 真机 L2Cache.csv |
| **5 计算占用** | 计算-传输重叠/向量化/精度 | Vec-MTE 气泡、向量化、GPR/占用 | instr_exe 指令级 |
| **6 架构** | Grid/Pipeline/L2驻留/混合精度 | pipe 利用率、cube 占比、L2 命中、block dim、grid 配置 | 真机 PipeUtilization/ArithmeticUtilization + HIVM launch |

**硬性前提**：
- Tier3/4 必须 **peak_bw/regime 校准完成**（否则 bw_utilization 是 13x~18x 的错误值）
- Tier6 必须 **真机 board_prof 接入**（L2/pipe 只有真机给）
- 所有 Tier 必须 **sync op 计入 op_id**（否则与 simulator SET_FLAG/WAIT_FLAG/BAR 对不齐）

---

## 五、知识补充（查哪里）

| 知识 | 位置 |
|---|---|
| HIVM 语法 / cube / sync op | HIVMDialect.md + HIVMSynchronizationOps.td（AscendNPU-IR 官方） |
| address_space 值映射 | triton-ascend al.ascend_address_space.md |
| instr_exe.csv 字段 | msopprof_simulator_performance_data.md |
| OpBasicInfo/PipeUtilization 字段 | CANN 官方文档（hiascend.com） |
| 6 层 Playbook | docx/playbook_tier1~6_*.md |
| 字段来源（旧版，已过时） | example_output/FIELD_SOURCE_MATRIX.md |
| 本日工作定版 | claude_resume_summary/SESSION_SUMMARY_2026-08-03_hivm_verified_final.md |

---

## 六、下一步（按服务器真数据逐步验证）

1. **[#4] 写 `msprof_analyzer.parse_hardware_dir()`**：读 board_prof（OpBasicInfo/PipeUtilization/Memory 标准字段），产出真机校准（peak_bw/L2/端到端）
2. **在服务器跑完整 3 源采集**（命令已在 test_matmul.py 注释）
3. **真数据验证 29 字段**：hivmir(真实 hivm_try.txt) + msprof(真实 sim_prof) + board_prof → merged_report.json，确认无「待补充」
4. **校准 SATURATION_PARAMS**：真机 Memory.csv 带宽 → 替换占位 peak
5. **size 扫描** → k0 曲线 → regime 准确
