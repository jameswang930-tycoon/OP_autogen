# 910B3 部署指南 — Analyzers 层

> 本文档指导 AI Agent 将 triton_agent_optimizer/analyzers/ 部署到 910B3 服务器，
> 补全所有待对齐的参数、字段和流程。

---

## 0. 前置知识

### msprof op simulator — 需要编译后的 .o 二进制!

- **不需要 NPU 硬件**，但需要 CANN Toolkit (含 Bisheng 编译器)
- **不接受 .py 源码**，必须先编译为 .o 文件
- 本质是纯 CPU 软件仿真器，模拟算子执行时序

```
完整流水线 (910B3):
  ① compiler.py: bisheng -c kernel.asc -o kernel.asc.o --npu-arch=dav-2201 --run-mode=sim
  ② msprof op simulator --soc-version=Ascend910B3 ./kernel.asc.o
  ③ → OPPROF_xxx/simulator/trace.json → msprof_analyzer 解析
```

### 两种分析路径

| 环境 | 工具 | 输出 | 精度 |
|---|---|---|---|
| Linux + CANN | ① compiler.py → ② msprof op simulator → ③ msprof_analyzer | trace.json | 真实仿真 |
| Linux + CANN + NPU | ① compiler.py → ② msprof op → ③ msprof_analyzer | 真实 trace.json | 最高 |
| Windows 本地 | `cost_emulator/simulator.py --llm` | 模拟文本 | 低 (placeholder) |

---

## 1. msprof_analyzer.py — 需要对齐的内容

### 1.1 当前状态

- ✅ trace.json 解析器: Chrome Trace Event 格式 (ph/cat/pid/tid/ts/dur)
- ✅ 通道映射: MTE2→GM→UB, MTE3→UB→GM, VECTOR→VecUnit, Cube→CubeUnit, ...
- ✅ MTE2 自动修正: 区分 GM→UB (Vector) vs GM→L1 (Matrix)
- ✅ 29 字段输出对齐

### 1.2 需要在 910B3 上验证/补全

| 项目 | 命令 | 预期结果 |
|---|---|---|
| 安装 CANN | 参考 [昇腾社区](https://www.hiascend.com) | `which msprof` 返回路径 |
| 编译测试 kernel | `cd emulators/test/add && python __init__.py` 拿到 kernel 代码 | 生成 test_add_kernel |
| 运行 msprof op simulator | `msprof op simulator --soc-version=Ascend910B3 ./test_add` | 生成 OPPROF_xxx/ |
| 解析 trace.json | `python analyzers/msprof_analyzer.py --trace OPPROF_xxx/simulator/trace.json` | 输出 pipeline_report.json |

### 1.3 通道映射验证

在 910B3 上跑完后, 检查 `trace.json` 中的 `cat` 字段是否与我们的 `PIPELINE_MAP` 一致:

```bash
# 查看 trace.json 中的所有通道类型
python3 -c "
import json
with open('OPPROF_xxx/simulator/trace.json') as f:
    data = json.load(f)
events = data if isinstance(data, list) else data.get('traceEvents', [])
cats = set(e.get('cat','') for e in events if e.get('cat'))
print('Channels found:', sorted(cats))
"
```

如果出现新通道名, 在 `msprof_analyzer.py:124` 的 `PIPELINE_MAP` 中添加。

---

## 2. hivmir_analyzer.py — 需要对齐的内容

### 2.1 当前状态

- ✅ 3 种 HIVMIR 格式解析 (简化/纯文本/MLIR 全格式)
- ✅ RAW/WAR/WAW 依赖分析
- ✅ buffer 生命周期追踪 (producers/consumers)
- ✅ 29 字段输出对齐

### 2.2 需要在 910B3 上验证/补全

| 项目 | 命令 | 预期结果 |
|---|---|---|
| 编译 kernel (带 HIVMIR 插桩) | `ascendc --mlir-print-ir-after-all kernel.py` | 输出含 hivm.* 的 IR dump |
| 或使用我们的 compiler | `python execution/compiler.py` (在 910B3 上) | 自动提取 HIVMIR |
| 解析 HIVMIR | `python analyzers/hivmir_analyzer.py` (自测模式) | 输出 hivmir_report.json |

### 2.3 HIVMIR 格式验证

在 910B3 上提取到真实 HIVMIR 后, 查看格式:

```bash
cat hivmir_output.mlir | head -20
```

可能出现的格式变体:
```
格式 A: hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
格式 B: hivm.hir.load ins(%arg0) outs(%alloc) : memref<...>
格式 C: %alloc = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
```

如果出现新格式, 在 `hivmir_analyzer.py:216` 的 `_try_parse_op()` 中添加新的正则分支。

---

## 3. dsl_merger.py — 需要对齐的内容

### 3.1 当前状态

- ✅ msprof + HIVMIR 通过 op_id 对齐
- ✅ 29 字段互相填补
- ✅ bandwidth/regime 通过 SATURATION_PARAMS 计算
- ✅ LLM 文本 + Gantt 图生成

### 3.2 需要在 910B3 上验证/补全

**SATURATION_PARAMS** — 这是最重要的参数表!

当前状态:
- Engine 0/1/2 (GM→UB, UB→GM, VecUnit) — MEASURED ✅
- Engine 3/4/5/6 (GM→L1, L1→L0, CubeUnit, L0→GM) — PLACEHOLDER ❌

**在 910B3 上需要做的事情**:

```bash
# 运行 perf_test benchmark 获取真实参数
cd perf_test/910B3/vecadd/
python bench_910b3_paths.py
# 输出 bench_result.csv — 包含各通路实测带宽

# 分析结果, 更新 SATURATION_PARAMS
# 位置: dsl_merger.py 顶部的 SATURATION_PARAMS dict
# 或: costModel/cost_emulator/simulator.py 中的 SATURATION_PARAMS
```

当前在 `dsl_merger.py:42-50` 有一个内置 fallback:
```python
SATURATION_PARAMS = {
    0: {"vpeak": 121.08, "k0": 6.65, "peak_clamp": 80.83},   # GM→UB  ← bench实测
    1: {"vpeak": 190.19, "k0": 10.72, "peak_clamp": 76.67},  # UB→GM  ← bench实测
    2: {"vpeak": 461.0,  "k0": 4.50, "peak_clamp": 404.0},   # VecUnit ← bench实测
    3: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # GM→L1  ← PLACEHOLDER!
    4: {"vpeak": 100.0,  "k0": 6.65, "peak_clamp": 100.0},   # L1→L0  ← PLACEHOLDER!
    5: {"vpeak": 150.0,  "k0": 0,    "peak_clamp": 150.0},   # CubeUnit← PLACEHOLDER!
    6: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # L0→GM  ← PLACEHOLDER!
}
```

**替换 PLACEHOLDER 的步骤**:
1. 在 910B3 上跑 matrix pipeline 的 benchmark
2. 测量 GM→L1, L1→L0, CubeUnit, L0→GM 的实际带宽
3. 计算 vpeak, k0, peak_clamp
4. 更新上面的 dict

---

## 4. bottleneck_diagnoser.py — 需要对齐的内容

### 4.1 当前状态

- ✅ Tier-aware 瓶颈分类
- ✅ 6 种瓶颈类型
- ✅ HIGH/MEDIUM/LOW/UNCERTAIN 评估
- ✅ 聚合分析 (同类型 op 合并)
- ✅ PLACEHOLDER 引擎标注 UNCERTAIN

### 4.2 不需要额外操作

这个文件是纯规则引擎, 不依赖硬件。只要上游数据对, 诊断就正确。

唯一需要验证的: 在真实 910B3 数据上跑一次, 确认诊断结果合理:

```bash
python analyzers/bottleneck_diagnoser.py \
  outputs/<kernel>/round0/merged/merged_report.json 3
```

---

## 5. data_extractor.py — 不需要修改

纯数据过滤和格式化, 不需要硬件。

---

## 快速部署命令

在 910B3 服务器上按顺序执行:

```bash
# 1. 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd triton_agent_optimizer

# 2. 测试 msprof analyzer (需要 CANN + 已编译的 binary)
msprof op simulator --soc-version=Ascend910B3 ./test_binary
ls OPPROF_*/simulator/trace.json  # 确认生成

# 3. 测试 HIVMIR 提取
python execution/compiler.py  # 自测 HIVMIR 提取逻辑

# 4. 跑完整分析链
python analyzers/dsl_merger.py outputs/<kernel>/round0
python analyzers/bottleneck_diagnoser.py outputs/<kernel>/round0/merged/merged_report.json 1

# 5. 查看输出
cat outputs/<kernel>/round0/merged/final_report_llm.txt
```

## 待补全清单

| 文件 | 补全项 | 优先级 | 预期耗时 |
|---|---|---|---|
| `msprof_analyzer.py` | 通道映射验证 (跑真实 trace.json) | ⭐⭐⭐ | 30min |
| `hivmir_analyzer.py` | HIVMIR 格式适配 (新格式分支) | ⭐⭐⭐ | 1h |
| `dsl_merger.py` | SATURATION_PARAMS Engine 3-6 实测 | ⭐⭐⭐ | 2h |
| `bottleneck_diagnoser.py` | 真实数据验证 | ⭐⭐ | 30min |
| `data_extractor.py` | 无需修改 | ⭐ | 0 |
