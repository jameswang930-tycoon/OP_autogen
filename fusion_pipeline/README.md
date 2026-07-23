# Fusion Pipeline - 完整流程使用指南

## 概述

这个工具用于分析 Triton 算子的执行流水线，结合两个数据源：
1. **HIVMIR**（华为编译器中间产物）：提供详细的操作信息、变量名、依赖关系
2. **msprof op simulator**：提供时序信息、带宽利用率、性能数据

## 输入

- **Triton kernel 文件**（.py）：可运行的 Triton 算子代码
- **PyTorch 基准代码**：测试数据生成和调用代码
- **Shape 信息**：张量维度信息
- **数据类型**：fp16/fp32/bf16

## 输出

### 1. 文本报告（`complete_fusion_report.txt`）

包含每个操作的详细信息：

```
Op   操作类型        引擎          SIZE(KB)    变量名                  Times(ns)    BW util    Regime        waitFor            依赖类型      时间占比
----------------------------------------------------------------------------------------------------------------------------
   0     gm_to_ub     GM→UB           128.0        ub_1                   1621.6      100.0%  saturated                 -            -       44.36%
   1         vadd   VecUnit           128.0        ub_2                    324.4      100.0%  saturated               op0          RAW        8.88%
   2     ub_to_gm     UB→GM           128.0        gm_2                   1709.6      100.0%  saturated               op1          RAW       46.77%
```

### 2. 可视化图表（`complete_fusion_analysis.png`）

包含 4 个子图：
- 时间占比柱状图
- 时间占比饼图（Top 10）
- 引擎时间分布
- 操作时序图

## 使用方法

### 方法 1：使用示例数据运行

```bash
cd D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc
python fusion_pipeline/complete_data_merge.py
```

### 方法 2：提供自定义 DSL 和 HIVMIR

```bash
python fusion_pipeline/complete_data_merge.py \
  --dsl "alloc(gm_1, 256KB) alloc(ub_1, 256KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 1.0) ub_to_gm(gm_2, ub_2)" \
  --hivmir path/to/your/hivmir.mlir \
  --output-dir ./my_analysis
```

### 方法 3：在 910B3 服务器上完整运行

#### Step 1: 编译 Triton kernel 并提取 HIVMIR

```bash
# 在 910B3 服务器上
cd /path/to/OP_autogen_hjkc

# 编译 Triton kernel（需要华为编译器环境）
python fusion_pipeline/extract_hivmir_from_compiler.py \
  fusion_pipeline/example_kernels/vadd_kernel.py \
  --output-dir ./hivmir_output
```

#### Step 2: 运行性能基准测试

```bash
# 运行 Triton kernel 基准测试
python fusion_pipeline/example_kernels/vadd_kernel.py

# 运行 msprof 分析
msprof --op-mode ./prof_data
```

#### Step 3: 合并数据并生成报告

```bash
python fusion_pipeline/complete_data_merge.py \
  --hivmir ./hivmir_output/hivmir_output.mlir \
  --output-dir ./fusion_analysis_output
```

## 输出字段说明

### Op（操作序号）
- 从 0 开始的整数序号
- 对应 simulator 输出中的 `op0`, `op1` 等

### 操作类型
- `gm_to_ub`: 全局内存到统一缓冲区
- `ub_to_gm`: 统一缓冲区到全局内存
- `gm_to_l1`: 全局内存到 L1
- `l1_to_l0`: L1 到 L0
- `l0_to_gm`: L0 到全局内存
- `vadd`: 向量加法
- `vsub`: 向量减法
- `vmul`: 向量乘法
- `matrixmul`: 矩阵乘法

### 引擎
- **GM→UB**: 全局内存到 UB 传输引擎
- **UB→GM**: UB 到全局内存传输引擎
- **VecUnit**: 向量计算单元
- **GM→L1**: 全局内存到 L1 传输引擎
- **L1→L0**: L1 到 L0 传输引擎
- **CubeUnit**: 矩阵计算单元
- **L0→GM**: L0 到全局内存传输引擎

### SIZE（搬运数据大小）
- 单位：KB
- 来源：HIVMIR 中的 `memref<NKB>` 声明

### Times（执行时间）
- 单位：纳秒（ns）
- 来源：msprof op simulator 的时序模拟

### BW util（带宽利用率）
- 格式：百分比
- 计算：`effective_bw / peak_bw`
- 来源：simulator 的带宽模型

### Regime（带宽状态）
- `floor`: 小传输，带宽受限
- `ramp`: 中等传输，带宽爬升
- `saturated`: 大传输，达到峰值
- `flat`: 计算密集型，与大小无关

### waitFor（依赖操作）
- 格式：`op0`, `op1` 等
- 表示当前操作依赖的操作序号

### 依赖类型
- **RAW**: Read After Write（读后写，真依赖）
- **WAR**: Write After Read（写后读，反依赖）
- **WAW**: Write After Write（写后写，输出依赖）

### 时间占比
- 单位：百分比
- 计算：`duration_ns / total_ns * 100`

## 典型输出示例

### 文本报告

```
============================================================================================================================
算子融合分析报告 - 完整操作流水（数据来源：HIVMIR + msprof op simulator）
============================================================================================================================

总执行时间: 3655.57 ns
操作数量: 3

Op   操作类型        引擎          SIZE(KB)    变量名                  Times(ns)    BW util    Regime        waitFor            依赖类型      时间占比
----------------------------------------------------------------------------------------------------------------------------
   0     gm_to_ub     GM→UB           128.0        ub_1                   1621.6      100.0%  saturated                 -            -       44.36%
   1         vadd   VecUnit           128.0        ub_2                    324.4      100.0%  saturated               op0          RAW        8.88%
   2     ub_to_gm     UB→GM           128.0        gm_2                   1709.6      100.0%  saturated               op1          RAW       46.77%

============================================================================================================================

时间占比统计（从大到小排序）：
----------------------------------------------------------------------------------------------------------------------------
 1. Op  2 (    ub_to_gm):  46.77%  时长=    1709.6ns  大小=   128.0KB  引擎=    UB→GM  变量=gm_2                 依赖=op1(RAW)
 2. Op  0 (    gm_to_ub):  44.36%  时长=    1621.6ns  大小=   128.0KB  引擎=    GM→UB  变量=ub_1                 依赖=无依赖
 3. Op  1 (        vadd):   8.88%  时长=     324.4ns  大小=   128.0KB  引ux= VecUnit  变量=ub_2                 依赖=op0(RAW)

============================================================================================================================

引擎利用率统计：
----------------------------------------------------------------------------------------------------------------------------
      GM→UB: busy=    1621.6ns  utilization= 44.36%  ops=[op0]
      UB→GM: busy=    1709.6ns  utilization= 46.77%  ops=[op2]
    VecUnit: busy=     324.4ns  utilization=  8.88%  ops=[op1]
```

## 高级用法

### 自定义分析

```python
from fusion_pipeline.complete_data_merge import (
    SimulatorOutputParser,
    HIVMIRParser,
    DataMerger,
    CompleteReportGenerator
)

# 1. 解析 simulator 输出
sim_parser = SimulatorOutputParser()
sim_ops = sim_parser.parse(simulator_llm_output)

# 2. 解析 HIVMIR
hivmir_parser = HIVMIRParser()
hivmir_ops = hivmir_parser.parse(hivmir_text)

# 3. 合并数据
merger = DataMerger()
combined_ops = merger.merge(sim_ops, hivmir_ops)

# 4. 自定义分析
for op in combined_ops:
    print(f"Op{op.op_id}: {op.op_type}, {op.time_ratio:.1f}%, deps={op.dependencies}")

# 5. 生成自定义报告
report = CompleteReportGenerator.generate_text_report(
    combined_ops, 
    sim_parser.total_ns,
    output_file="custom_report.txt"
)
```

### 批量分析多个 kernel

```python
import glob

kernels = glob.glob("kernels/*.py")

for kernel_file in kernels:
    # 生成 HIVMIR
    hivmir_file = extract_hivmir(kernel_file)
    
    # 运行 simulator
    dsl = convert_to_dsl(kernel_file)
    sim_ops, _, total_ns = run_simulator_and_parse(dsl)
    
    # 解析 HIVMIR
    hivmir_ops = parse_hivmir_file(hivmir_file)
    
    # 合并并生成报告
    combined = DataMerger().merge(sim_ops, hivmir_ops)
    output_dir = f"analysis/{Path(kernel_file).stem}"
    CompleteReportGenerator.generate_text_report(
        combined, total_ns,
        output_file=f"{output_dir}/report.txt"
    )
```

## 故障排查

### 问题 1: simulator 运行失败

**症状**: `python simulator.py` 报错

**解决方案**:
```bash
# 检查 Python 环境
python --version  # 需要 Python 3.8+

# 检查依赖
pip install networkx matplotlib pandas

# 检查路径
ls costModel/cost_emulator/simulator.py
```

### 问题 2: HIVMIR 解析失败

**症状**: `No operations found in input`

**解决方案**:
- 确保 HIVMIR 文件格式正确
- 检查是否包含 `hivm.` 前缀的操作
- 示例格式：
  ```
  hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
  hivm.vadd %ub_2, %ub_1, 1.0
  hivm.ub_to_gm %gm_2, %ub_2 : memref<128KB>
  ```

### 问题 3: 在 910B3 服务器上无法运行

**症状**: 编译器或 msprof 命令找不到

**解决方案**:
```bash
# 检查 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查环境变量
echo $ASCEND_HOME
echo $PATH

# 检查 NPU 设备
npu-smi info
```

## 文件结构

```
fusion_pipeline/
├── complete_data_merge.py         # 主脚本：数据合并和报告生成
├── run_fusion_analysis.py         # 运行脚本
├── extract_hivmir_from_compiler.py # HIVMIR 提取脚本
├── combine_hivmir_msprof.py       # 基础合并脚本
├── README.md                      # 本文档
└── example_kernels/
    └── vadd_kernel.py             # 示例 Triton kernel
```

## 参考资料

- [AscendNPU-IR 官方文档](https://ascendnpu-ir.gitcode.com/)
- [MLIR Action Tracing](https://mlir.llvm.org/docs/ActionTracing/)
- [cost_emulator 机制](../costModel/cost_emulator_mechanism.md)
- [bottleneck-analysis skill](../costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md)