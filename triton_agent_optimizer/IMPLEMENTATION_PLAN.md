# Triton Agent Optimizer — 逐文件实现计划

> ## ⚠️ 已废弃（2026-08-18 标记）
>
> **本文档是 v3 时期的逐文件实现计划，最后更新 2026-07-23**。
> 自 v4 以来，目录结构、`agents/orchestrator.py` / `optimizers/*.py` /
> `fusion_pipeline/*` / `feedback/round_logger.py` 等计划中的文件
> **从未实际创建**——v4 改用 `agents/scheduler.py` + `analyzers/integrate.py` +
> `memory/failed_cases.py` + `optimization_trajectory.json` 等替代方案。
>
> **请阅读**:
> - 最新架构和完整数据流 → `ARCHITECTURE_DESIGN.md`
> - 文件列表和用途 → `README.md` §4 + §6
> - 6 层优化策略原理 → `docx/OPTIMIZATION_METHODOLOGY.md`
>
> ---
>
> **保留原因**：作为 v3 思路的历史快照，便于理解 v3→v4→v4.6 三轮迁移的背景。
> **不要按本文档新建文件**——所有设计决定请走 `ARCHITECTURE_DESIGN.md` + `README.md`。

> **状态: 实现阶段已完成，本文档作为架构参考保留。**
> 最新架构和完整数据流见 `ARCHITECTURE_DESIGN.md`。
> 实际文件列表和用途见 `README.md` §4。
>
> **已发生的关键变更**:
> - 三阶段验证 → 两阶段 (Simulator 不在验证环节，在分析层)
> - Tier 顺序: Algorithm→Fusion→Tiling→Memory→Compute→Arch
> - `fusion_pipeline/` → 迁移到 `analyzers/` + `execution/compiler.py`
> - 新增 `main.py`、`prepare/`、`.claude/skills/triton-agent-*`
> - Planner/Coder 支持 AGENT_TASK 文件模式 (非 LLM 环境可用)

> 最后更新: 2026-07-23
> 实现顺序: 按依赖关系排列，每个文件一个对话，一次只做一个。

---

## 0. 总览: 实现顺序

```
Phase 1: 基础配置 (1 文件, 30min)
  └─ config.py

Phase 2: 分析层 (5 文件, 每个 1~2h)
  ├─ analyzers/msprof_analyzer.py       # 对接 simulator.py
  ├─ analyzers/hivmir_analyzer.py       # 对接 HIVMIR 解析
  ├─ analyzers/dsl_merger.py            # 合并数据
  ├─ analyzers/bottleneck_diagnoser.py   # 瓶颈诊断
  └─ analyzers/data_extractor.py        # 按需数据提取

Phase 3: 执行层 (4 文件, 每个 1~2h)
  ├─ execution/emulator_runner.py       # CPU Emulator 包装
  ├─ execution/simulator_runner.py      # Simulator 包装
  ├─ execution/compiler.py              # 编译器接口
  └─ execution/hardware_runner.py       # 910B3 真机运行

Phase 4: 反馈层 (5 文件, 每个 30min~1h)
  ├─ feedback/round_logger.py           # 每轮记录
  ├─ feedback/optimization_journal.py   # 日志管理
  ├─ feedback/stop_condition.py         # 停止条件
  ├─ feedback/trajectory_chart.py       # 轨迹图
  └─ feedback/case_template.py          # 案例模板

Phase 5: 记忆层 (3 文件, 每个 1~2h)
  ├─ memory/sliding_window.py           # 滑动窗口
  ├─ memory/context_manager.py          # 上下文管理
  └─ memory/experience_retriever.py     # 经验检索

Phase 6: 智能体层 (4 文件, 每个 1~3h)
  ├─ agents/planner.py                  # 规划智能体
  ├─ agents/coder.py                    # 编码智能体
  ├─ agents/verifier.py                 # 验证智能体
  └─ agents/orchestrator.py             # 调度器 (核心)

Phase 7: 优化器 (5 文件, 每个 1~2h)
  ├─ optimizers/base_optimizer.py       # 基类
  ├─ optimizers/tile_optimizer.py       # Tiling
  ├─ optimizers/memory_optimizer.py     # 内存
  ├─ optimizers/fusion_optimizer.py     # 融合
  └─ optimizers/compute_optimizer.py    # 计算

Phase 8: 入口 + 手册 (9 文件, 不等)
  ├─ main.py                            # 主入口
  ├─ playbooks/ (7 个 .md)              # 优化手册
  ├─ cases/template.md                  # 案例模板
  └─ tests/test_orchestrator.py         # 测试
```

---

## Phase 1: 基础配置

### 文件 1: `config.py`

**做什么**: 全局配置中心，所有其他文件通过 `from config import ...` 获取参数

**依赖**: 无

**需要包含的配置**:

```python
# 路径
PROJECT_ROOT = Path(__file__).parent.parent
SIMULATOR_PATH = PROJECT_ROOT / "costModel/cost_emulator/simulator.py"
EMULATOR_PATH = PROJECT_ROOT / "emulators"
FUSION_PIPELINE_PATH = PROJECT_ROOT / "fusion_pipeline"
MEMORY_PATH = PROJECT_ROOT / "memory"
OUTPUT_DIR = PROJECT_ROOT / "triton_agent_optimizer/output"
ROUNDS_DIR = OUTPUT_DIR / "rounds"
CASES_DIR = PROJECT_ROOT / "triton_agent_optimizer/cases"

# 硬件参数 (从 simulator.py SATURATION_PARAMS 读取, 不要硬编码)
# 只在这里定义如何从 simulator 模块加载参数

# 优化参数
MAX_ROUNDS = 200
MAX_TIME_HOURS = 6
MAX_CONSECUTIVE_REVERTS = 5
MIN_SPEEDUP_THRESHOLD = 1.01
PEAK_THRESHOLD = 0.90
PLATEAU_ROUNDS = 10
PLATEAU_VARIANCE = 0.02

# 验证参数
EMULATOR_RETRY_MAX = 3
EMULATOR_SHAPES = [1, 3, 7, 256, 512, 1024, 1025, 2049, 4096, 8192]
EMULATOR_DTYPES = ["fp16", "fp32"]
EMULATOR_TOLERANCE = {"fp16": {"rtol": 1e-2, "atol": 1e-2},
                       "fp32": {"rtol": 1e-5, "atol": 1e-5}}

# 上下文管理
HOT_WINDOW_SIZE = 5       # 保留完整上下文的最近轮次
WARM_WINDOW_SIZE = 15     # 保留摘要的轮次范围
MAX_CONTEXT_TOKENS = 800000  # 1M 窗口留 20% 余量

# 6 层策略定义
STRATEGY_TIERS = {
    1: "Block Size & Launch Config",
    2: "Memory Access & Coalescing",
    3: "Operator Fusion",
    4: "Compute Optimization",
    5: "910B3 Architecture-Specific",
    6: "Algorithmic Restructure",
}
TIER_PROMOTION_THRESHOLD = 3  # 连续 N 轮无改进→升级
```

**关键点**:
- 硬件参数**不要硬编码**——从 `costModel/cost_emulator/simulator.py` 的 `SATURATION_PARAMS` 和 `MEMORY_CAPACITY_KB` 动态读取
- 所有可调阈值集中在这里，方便后续调参
- 路径用 `pathlib.Path`，不用字符串拼接

---

## Phase 2: 分析层

### 文件 2: `analyzers/msprof_analyzer.py`

**做什么**: 包装 simulator.py，提供 Python API。**两种输出模式，用途不同。**

**依赖**: `config.py`, `costModel/cost_emulator/simulator.py`

**核心类**: `MsprofAnalyzer`

```python
class MsprofAnalyzer:
    # ═══ 主要模式: --llm (每轮必跑, AI 消费) ═══

    def generate_dsl_from_kernel(self, kernel_code: str) -> str:
        """从 Triton kernel 提取 DSL 程序 (当前最大 gap)"""
        # TODO: 复用 /triton-plan 的逻辑, 或实现新的 kernel→DSL 转换

    def run_simulator_llm(self, dsl_program: str) -> SimulatorResult:
        """★ 主要方法: 运行 simulator --llm --critical-path
        输出 = 结构化 DSL 流水线数据 (7 个 section)
        这是 Agent 每轮必跑的分析数据源
        调用: python simulator.py --llm --critical-path "<dsl_program>"
        """

    def parse_llm_output(self, raw_output: str) -> SimulatorResult:
        """解析 simulator 的 --llm 输出 → SimulatorResult
        7 个 section: EXECUTION SUMMARY / TIME BREAKDOWN / PER-OP STATISTICS /
                      ENGINE UTILIZATION / BANDWIDTH UTILIZATION / PARALLELISM /
                      CRITICAL PATH
        """

    # ═══ 次要模式: Gantt (按需生成, 人读) ═══

    def run_simulator_human(self, dsl_program: str) -> str:
        """按需调用: 生成人读的 ASCII Gantt 流水图
        包含: Pipeline Execution Graph / 操作表格 / 时间占比柱状图 /
              引擎利用率 / 带宽利用率 / 关键路径
        注意: Windows 需设置 PYTHONIOENCODING=utf-8 (µ 字符编码问题)
        调用: python simulator.py --critical-path "<dsl_program>"
        返回: 原始文本 (不解析, 直接保存或打印)
        """
```

**两种模式对比**:

| | --llm 模式 | Gantt 模式 |
|---|---|---|
| **消费者** | AI (Agent Planner/Coder) | 人 (开发者 debug) |
| **频率** | 每轮必跑 | 按需 (最终报告 / debug 时) |
| **输出** | 结构化 `SimulatorResult` | 原始文本 (ASCII 图表) |
| **体积** | ~2-8 KB (精简) | ~10-50 KB (含图表) |
| **示例** | `example_output/01_vector_add_saturated.txt` | `example_output/05_full_gantt_vector_add.txt` |

**`SimulatorResult` 数据结构** (与 simulator --llm 输出的 7 个 section 对齐):

```python
@dataclass
class SimulatorResult:
    total_ns: float
    num_ops: int
    execution_mode: str  # 'parallel' or 'sequential'
    ops: List[SimulatorOp]  # 每个 op 的详细信息
    engine_utilization: Dict[str, float]  # engine_name → utilization%
    parallel_pairs: List[Tuple[int, int]]
    critical_path: List[int]  # op indices
    critical_path_length_ns: float
    critical_path_fraction: float
```

**参考实现**: `fusion_pipeline/complete_data_merge.py` → `SimulatorOutputParser` 类 (行 60-193)，可以直接复用其解析逻辑

**搜索关键字段**: `total_ns`, `num_ops`, `execution_mode`, `op\d+`, `engine:`, `effective=`, `peak=`, `regime=`, `blocked_by:`, `critical path`

---

### 文件 3: `analyzers/hivmir_analyzer.py`

**做什么**: 解析 HIVMIR 文本，提取变量名、依赖关系、数据大小

**依赖**: `config.py`

**核心类**: `HIVMIRAnalyzer`

```python
class HIVMIRAnalyzer:
    def extract_from_compiler(self, kernel_code: str) -> str:
        """在 910B3 上编译 kernel 并提取 HIVMIR 文本"""

    def parse(self, hivmir_text: str) -> List[HIVMIROp]:
        """解析 HIVMIR 文本 → 结构化数据"""

    def extract_dependencies(self, ops: List[HIVMIROp]) -> Dict[int, List[Tuple[int, str]]]:
        """提取 RAW/WAR/WAW 依赖"""

    def extract_variable_info(self, ops: List[HIVMIROp]) -> Dict[str, VariableInfo]:
        """提取变量名 + 数据大小 + 内存区域"""
```

**`HIVMIROp` 数据结构**:

```python
@dataclass
class HIVMIROp:
    op_id: int
    op_type: str         # gm_to_ub, vadd, matrixmul, ...
    engine: str           # GM→UB, VecUnit, ...
    dst: str              # 目标 buffer 名
    src: str              # 源 buffer 名
    src2: str = ""        # matrixmul 的第二个源
    size_kb: float = 64.0
    variable_name: str = ""
    memory_region: str = ""  # GM / UB / L1 / L0
    dependencies: List[Tuple[int, str]] = field(default_factory=list)
    line_number: int = 0
```

**参考实现**: `fusion_pipeline/complete_data_merge.py` → `HIVMIRParser` 类 (行 217-350)，直接复用

**HIVMIR 解析规则**:
- `hivm.alloc %buf_name : memref<128KB>` → 变量大小
- `hivm.gm_to_ub %ub_1, %gm_1` → 操作类型 + 操作数
- 依赖关系: 分析 def-use chain (谁写了 buf A, 谁读了 buf A)
- 内存区域: `gm_` → GM, `ub_` → UB, `l1_` → L1, `l0_` → L0

---

### 文件 4: `analyzers/dsl_merger.py`

**做什么**: 合并 simulator 输出 (性能数据) + HIVMIR 输出 (语义信息)

**依赖**: `msprof_analyzer.py`, `hivmir_analyzer.py`, `config.py`

**核心类**: `DSLMerger`

```python
@dataclass
class CombinedOp:
    """合并后的完整 op 信息"""
    # 来自 simulator
    op_id: int
    op_type: str
    engine: str
    size_kb: float
    duration_ns: float
    start_ns: float
    end_ns: float
    effective_bw: float
    peak_bw: float
    bw_utilization: float
    regime: str
    time_ratio: float
    wait_before_start: float
    blocked_by: str
    # 来自 HIVMIR (新增字段)
    variable_name: str = ""
    precise_size_kb: float = 0.0
    memory_region: str = ""
    dependencies: List[Tuple[int, str]] = field(default_factory=list)
    hivmir_line: int = 0

class DSLMerger:
    def merge(self, sim_result: SimulatorResult, 
              hivmir_ops: List[HIVMIROp]) -> List[CombinedOp]:
        """通过 op_id 对齐两个数据源, 生成合并后的 op 列表"""

    def generate_pipeline_report(self, combined_ops: List[CombinedOp],
                                  output_file: str = None) -> str:
        """生成完整 DSL 流水线报告 (对齐 simulator --llm 格式)"""

    def compute_engine_utilization(self, combined_ops, total_ns) -> Dict:
        """引擎利用率统计"""

    def compute_critical_path(self, combined_ops) -> List[int]:
        """关键路径提取"""
```

**合并策略**: 通过 `op_id` 对齐——simulator 的 op0 对应 HIVMIR 的 op0 (两者按程序顺序同序)

**输出格式**: 严格对齐 simulator `--llm --critical-path` 的 7 个 section，仅新增 HIVMIR 字段 (variable_name, dependencies, precise_size_kb)

---

### 文件 5: `analyzers/bottleneck_diagnoser.py`

**做什么**: 从合并后的 DSL 流水线诊断瓶颈

**依赖**: `dsl_merger.py`, `config.py`

**核心类**: `BottleneckDiagnoser`

```python
class BottleneckDiagnoser:
    BOTTLENECK_TYPES = [
        'memory_bandwidth',    # 传输 op, bw_util > 70%, regime=saturated
        'memory_latency',      # 传输 op, bw_util < 70%, regime=floor/ramp
        'compute_vec',         # VecUnit op, 计算瓶颈
        'compute_cube',        # CubeUnit op, 计算瓶颈
        'dependency',          # 依赖链长, blocked_by > 2 个 op
        'engine_contention',   # 单引擎利用率 > 80%, 其他引擎空闲
    ]

    def diagnose(self, combined_ops: List[CombinedOp],
                 engine_utilization: Dict,
                 critical_path: List[int]) -> BottleneckReport:
        """完整诊断"""

    def classify_bottleneck(self, op: CombinedOp) -> str:
        """分类单个 op 的瓶颈类型"""

    def assess_headroom(self, op: CombinedOp) -> str:
        """评估可优化空间: HIGH / MEDIUM / LOW / NONE"""
        # 规则:
        #   regime=floor, bw_util<50% → HIGH (大幅提升空间)
        #   regime=ramp, bw_util 50~90% → MEDIUM
        #   regime=saturated → LOW (已达峰值)
        #   placeholder 引擎 → UNCERTAIN (数据不可靠)
        #   compute with k0=0 → NONE (size-independent)
```

**`BottleneckReport` 数据结构**:

```python
@dataclass
class BottleneckReport:
    bottleneck_op_id: int
    bottleneck_type: str
    time_ratio: float
    bw_utilization: float
    regime: str
    on_critical_path: bool
    optimization_headroom: str  # HIGH / MEDIUM / LOW / NONE / UNCERTAIN
    suggested_strategies: List[str]
    suggested_playbook_sections: List[str]  # 应注入的手册章节
    engine_utilization_summary: Dict
    critical_path_ops: List[int]
```

**关键搜索/判断规则**:
- 瓶颈 = 在 critical_path 上 + time_ratio 最大
- `regime` 字段直接来自 simulator (floor/ramp/saturated/flat)
- placeholder 引擎 (3/4/5/6) 的诊断标注 UNCERTAIN
- WAR 依赖 + 纯 WAR (无 RAW/WAW) → suggest "allocate new buffer"

---

### 文件 6: `analyzers/data_extractor.py`

**做什么**: 从完整 DSL 流水线中按瓶颈类型提取关键数据——全量存文件, 精简化入 prompt

**依赖**: `dsl_merger.py`, `bottleneck_diagnoser.py`, `config.py`

**核心类**: `DataExtractor`

```python
class DataExtractor:
    # 每种瓶颈类型的提取器
    EXTRACTORS = {
        'memory_bandwidth':   '_extract_memory_bottleneck',
        'memory_latency':     '_extract_memory_bottleneck',
        'compute_vec':        '_extract_compute_bottleneck',
        'compute_cube':       '_extract_compute_bottleneck',
        'dependency':         '_extract_dependency_bottleneck',
        'engine_contention':  '_extract_engine_contention',
    }

    def extract(self, combined_ops: List[CombinedOp],
                bottleneck: BottleneckReport) -> ExtractedData:
        """主入口: 根据瓶颈类型提取关键数据"""

    # 返回结构:
    # @dataclass
    # class ExtractedData:
    #     critical_data: str           # 注入 prompt 的关键数据 (~10行)
    #     playbook_file: str           # 应注入的 playbook 文件名
    #     playbook_sections: List[str] # 应注入的 playbook 章节
    #     suggested_strategies: List[str]
    #     context_summary: str         # 上下文摘要 (温层)
    #     cold_data_point: dict        # 冷层数据点
```

**每种提取器的输出格式** (以 `memory_bandwidth` 为例):

```
[关键瓶颈] op2 (ub_to_gm) time_ratio=46.77%
  引擎: UB→GM peak=76.67 GB/s/核
  当前: tile=1KB, bw=16.2 GB/s (21.1%峰值), regime=ramp
  半饱和点: k0=10.72KB → tile需 >10.72KB 才能进入饱和区
  可优化空间: HIGH (bw_util 21.1%, 合并小传输可达 90%+)

[依赖链] op0(gm_to_ub)→op1(vadd)→op2(ub_to_gm)
  全 RAW 串行, 无 WAR 可打破

[引擎利用率]
  GM→UB: 44.4%  UB→GM: 46.8%  VecUnit: 8.9%
  → 传输是主要瓶颈, 计算几乎空闲

[注入 Playbook] playbook_memory.md §1, §2
```

**各瓶颈类型对应的 Playbook 注入**:

| 瓶颈类型 | Playbook 文件 | 章节 |
|---|---|---|
| memory_bandwidth | playbook_memory.md | §1(参数速查), §2(小传输合并) |
| memory_latency | playbook_memory.md | §1, §2, §3(coalescing) |
| compute_vec | playbook_compute.md | §1(VecUnit), §3(pipeline overlap) |
| compute_cube | playbook_compute.md | §2(CubeUnit) |
| dependency | playbook_fusion.md | §1(融合识别), §3(WAR打破) |
| engine_contention | playbook_910b3_arch.md | §4(grid选择), §3(pipeline选择) |

---

## Phase 3: 执行层

### 文件 7: `execution/emulator_runner.py`

**做什么**: 包装 `emulators/common/__init__.py`，提供标准化的正确性验证接口

**依赖**: `config.py`, `emulators/common/__init__.py`

**核心类**: `EmulatorRunner`

```python
class EmulatorRunner:
    def run_basic(self, kernel_code: str, test_inputs: Dict) -> EmulatorResult:
        """单 shape + 标准输入的基础正确性测试"""

    def run_shape_sweep(self, kernel_code: str) -> EmulatorResult:
        """多 shape 测试 (从 config.EMULATOR_SHAPES)"""

    def run_dtype_sweep(self, kernel_code: str) -> EmulatorResult:
        """多 dtype 测试 (从 config.EMULATOR_DTYPES)"""

    def run_edge_cases(self, kernel_code: str) -> EmulatorResult:
        """边界条件: 全零, 极值, 空张量"""

    def full_verification(self, kernel_code: str) -> EmulatorResult:
        """完整验证 = basic + shape sweep + dtype sweep + edge cases"""
```

**`EmulatorResult` 数据结构**:

```python
@dataclass
class EmulatorResult:
    passed: bool
    max_abs_error: float
    max_rel_error: float
    failed_shapes: List[str]
    failed_dtypes: List[str]
    error_details: str  # LLM 可读的错误报告
    trace_log: str      # TraceLogger 输出 (如有)
```

**实现关键点**:
- 调用 `emulators/common/__init__.py` 的 `launch_kernel_1d/2d/3d` + `verify()`
- 参考现有算子的 `test()` 函数了解调用模式
- 借鉴 "The Correctness Illusion" 论文的 seeded fuzzing oracle —— 多 shape + 严格 tolerance
- 每个 shape/dtype 测试失败不终止，汇总所有失败

**参考实现**: `emulators/test/add/__init__.py` → `test()` 函数

---

### 文件 8: `execution/simulator_runner.py`

**做什么**: 包装 simulator.py，提供 Python API 调用。默认用 `--llm` 模式。

**依赖**: `config.py`, `costModel/cost_emulator/simulator.py`

**核心类**: `SimulatorRunner`

```python
class SimulatorRunner:
    # ── 主要方法 (每轮必跑) ──
    def run_llm(self, dsl_program: str) -> SimulatorResult:
        """运行 simulator --llm --critical-path, 返回结构化结果"""

    def run_verify(self, dsl_program: str) -> MemoryVerifyResult:
        """运行 simulator --verify (内存容量检查)"""

    def compare(self, dsl_before: str, dsl_after: str) -> SimulatorCompare:
        """优化前后 --llm 数据对比"""

    # ── 按需方法 (最终报告/debug 时) ──
    def generate_gantt(self, dsl_program: str, output_path: Path) -> Path:
        """按需生成人读的 Gantt 流水图, 保存到文件
        注意: Windows 环境需 PYTHONIOENCODING=utf-8
        """
```

**实现关键点**:
- 不 import simulator (它是独立脚本) ——用 subprocess 调用
- 解析 stdout 的 7 个 section
- 复用 `analyzers/msprof_analyzer.py` 的 `parse_output()` 方法

**调用命令**:
```bash
python costModel/cost_emulator/simulator.py --llm --critical-path "<dsl_program>"
python costModel/cost_emulator/simulator.py --verify "<dsl_program>"
```

---

### 文件 9: `execution/compiler.py`

**做什么**: Ascend 编译器接口——编译 Triton kernel + 提取 HIVMIR

**依赖**: `config.py`

**核心类**: `CompilerInterface`

```python
class CompilerInterface:
    def compile(self, kernel_code: str) -> CompileResult:
        """编译 Triton kernel → NPU 二进制"""

    def extract_hivmir(self, kernel_code: str) -> str:
        """从编译过程提取 HIVMIR 文本"""
        # 对接: fusion_pipeline/extract_hivmir_from_compiler.py

    def check_compile_errors(self, kernel_code: str) -> List[str]:
        """只检查编译错误, 不做完整编译"""
```

**注意**: 此文件只在 910B3 服务器上能用。本地环境编译会失败，需要处理异常并给出清晰的错误信息。

---

### 文件 10: `execution/hardware_runner.py`

**做什么**: 在 910B3 上运行 kernel，收集 msprof 数据和 HIVMIR

**依赖**: `config.py`, `execution/compiler.py`

**核心类**: `HardwareRunner`

```python
class HardwareRunner:
    def compile_and_run(self, kernel_code: str) -> HardwareResult:
        """编译 + 运行基准测试 (warmup=30, repeat=200)"""

    def collect_msprof(self) -> MsprofData:
        """收集 msprof 性能数据"""

    def collect_hivmir(self) -> str:
        """收集 HIVMIR 中间表示"""

    def benchmark(self, kernel_code: str, iterations: int = 200) -> float:
        """性能基准测试, 返回平均延迟 (ms)"""
```

**`HardwareResult` 数据结构**:

```python
@dataclass
class HardwareResult:
    compile_success: bool
    run_success: bool
    latency_ms: float
    throughput_gb_s: float  # for memory-bound ops
    throughput_tflops: float  # for compute-bound ops
    msprof_data: Optional[MsprofData]
    hivmir_text: Optional[str]
    error_message: Optional[str]
```

**注意**: 此文件只在 910B3 服务器上能用。参考 `perf_test/910B3/vecadd/bench_910b3_paths.py` 的 benchmark 模式 (warmup + sync + repeat)。

---

## Phase 4: 反馈层

### 文件 11: `feedback/round_logger.py`

**做什么**: 记录每轮优化的完整数据到 JSONL

**依赖**: `config.py`

**核心类**: `RoundLogger`

```python
class RoundLogger:
    def log_round(self, round_data: RoundRecord):
        """写入本轮完整数据到 optimization_journal.jsonl"""

    def log_code_diff(self, original: str, optimized: str) -> str:
        """生成 unified diff 并保存到 rounds/round_NNN_diff.patch"""

    def log_plan(self, plan_text: str, round_num: int) -> Path:
        """保存本轮优化计划到 rounds/round_NNN_plan.md"""
```

**`RoundRecord` 数据结构** (与 ARCHITECTURE_DESIGN.md §9 的 JSONL schema 一致):

```python
@dataclass
class RoundRecord:
    round: int
    timestamp: str
    kernel_fingerprint: str

    # 计划
    plan: dict  # {strategy, strategy_tier, target_speedup, plan_file}

    # 瓶颈 (前后对比)
    bottleneck_before: dict
    bottleneck_after: dict

    # 代码变更
    code_change: dict  # {diff_file, lines_changed, files_changed}

    # 验证 (两阶段)
    verification: dict

    # 决策
    decision: str  # KEEP / REVERT
    decision_reason: str
    cumulative_speedup: float
    cumulative_rounds_kept: int
    cumulative_rounds_reverted: int
```

---

### 文件 12: `feedback/optimization_journal.py`

**做什么**: 管理 JSONL 日志文件——追加、查询、统计、导出

**依赖**: `config.py`, `feedback/round_logger.py`

**核心类**: `OptimizationJournal`

```python
class OptimizationJournal:
    def __init__(self, journal_path: Path):
        self.path = journal_path

    def append(self, record: RoundRecord):
        """追加一条记录 (JSONL 一行)"""

    def query(self, round_range: slice = None) -> List[RoundRecord]:
        """查询指定轮次"""

    def get_latest(self, n: int = 1) -> List[RoundRecord]:
        """获取最近 N 轮"""

    def get_summary(self, start_round: int = 0) -> JournalSummary:
        """生成摘要: 总轮次/Keep数/Revert数/当前加速比/瓶颈变化历史"""

    def get_performance_curve(self) -> PerformanceCurve:
        """性能曲线数据: [(round, cumulative_speedup, decision), ...]"""

    def export_csv(self, output_path: Path):
        """导出为 CSV 供外部分析"""
```

---

### 文件 13: `feedback/stop_condition.py`

**做什么**: 检查是否满足停止条件

**依赖**: `config.py`, `feedback/optimization_journal.py`

**核心类**: `StopChecker`

```python
class StopChecker:
    def __init__(self, config: OptimizationConfig):
        self.max_rounds = config.MAX_ROUNDS
        self.max_consecutive_reverts = config.MAX_CONSECUTIVE_REVERTS
        # ... 从 config 读取所有阈值

    def check(self, journal: OptimizationJournal,
              current_state: dict) -> StopResult:
        """检查所有停止条件, 返回哪个条件触发 (或 CONTINUE)"""

    # 7 个检查方法 (每个返回 bool + reason):
    def _check_consecutive_reverts(self, journal) -> Tuple[bool, str]:
        """连续 N 轮 Revert?"""

    def _check_peak_threshold(self, current_state) -> Tuple[bool, str]:
        """达到理论峰值 90%?"""

    def _check_round_budget(self, journal) -> Tuple[bool, str]:
        """轮次预算耗尽?"""

    def _check_time_budget(self, start_time) -> Tuple[bool, str]:
        """时间预算耗尽?"""

    def _check_target_speedup(self, current_state) -> Tuple[bool, str]:
        """达到目标加速比?"""

    def _check_strategy_exhaustion(self, journal) -> Tuple[bool, str]:
        """所有策略层级都试过了?"""

    def _check_plateau(self, journal) -> Tuple[bool, str]:
        """最近 10 轮加速比波动 < 2%?"""
```

**`StopResult`**:

```python
@dataclass
class StopResult:
    should_stop: bool
    reason: str           # 停止原因 (或 "CONTINUE")
    condition_index: int  # 哪个条件触发的 (1~7, 0=未触发)
```

---

### 文件 14: `feedback/trajectory_chart.py`

**做什么**: 从 JSONL 生成优化轨迹可视化 (双面板图, 参照 AutoKernel)

**注意**: 这是**跨轮次的优化进度图**, 不是单轮的 Gantt 流水图。
- `trajectory_chart.py` → 多轮累计加速比曲线 (x轴=轮次, y轴=加速比)
- `SimulatorRunner.generate_gantt()` → 单轮 DSL 流水线 Gantt 图 (x轴=时间ns, y轴=引擎)

**依赖**: `config.py`, `feedback/optimization_journal.py`, matplotlib

**核心类**: `TrajectoryChart`

```python
class TrajectoryChart:
    def generate(self, journal: OptimizationJournal,
                 output_path: Path = None) -> Path:
        """从 JSONL 生成双面板图"""

    # 上图: cumulative_speedup (running-best 曲线 + 每轮散点)
    # 下图: latency (running-best 曲线 + baseline 虚线)
    # 颜色: 绿=Keep, 红=Revert
    # 标注: 最终加速比 + 最终延迟
```

**实现要点**:
- `np.maximum.accumulate()` 计算 cumulative running-best speedup
- `np.minimum.accumulate()` 计算 running-best latency
- 散点颜色: `['#2ecc71' if d=='KEEP' else '#e74c3c' for d in decisions]`
- baseline 虚线: 取自 journal 第一轮的延迟

**参考**: AutoKernel 的 `plot.py`——两面板 (TFLOPS + latency)、running-best 曲线 + scatter + baseline 虚线

---

### 文件 15: `feedback/case_template.py`

**做什么**: 当优化成功 (加速比达标) 时自动生成优秀案例文档

**依赖**: `config.py`, `feedback/optimization_journal.py`

**核心类**: `CaseGenerator`

```python
class CaseGenerator:
    def should_generate(self, final_result: dict) -> bool:
        """累计加速比 ≥ target_speedup → 生成案例"""

    def generate(self, kernel_code: str, journal: OptimizationJournal,
                 final_result: dict) -> str:
        """生成 Markdown 案例文档"""

    def fill_template(self, template_path: Path, data: dict) -> str:
        """填充模板"""

    def publish(self, case_text: str, op_name: str) -> Path:
        """发布到 cases/ 目录"""
```

**案例模板** (cases/template.md) 包含:
- 算子基本信息 (名称/类型/shape/dtype)
- 初始性能 vs 最终性能
- 优化过程总览表 (每轮: 策略/变更/加速比/决策)
- 关键变更清单 (每轮最重要的 diff)
- 瓶颈迁移图 (哪个瓶颈被逐个消除)
- 经验总结 (做了哪些有效的, 踩了什么坑)
- 适用场景 (什么类型的算子可以参考这个案例)

---

## Phase 5: 记忆层

### 文件 16: `memory/sliding_window.py`

**做什么**: 管理优化轮次的滑动窗口——热/温/冷三层

**依赖**: `config.py`

**核心类**: `SlidingWindow`

```python
class SlidingWindow:
    def __init__(self, hot_size: int = 5, warm_size: int = 15):
        self.rounds: List[RoundRecord] = []

    def add_round(self, record: RoundRecord):
        """添加轮次, 自动分类到热/温/冷"""

    def get_hot_context(self) -> str:
        """热层: 最近 5 轮完整上下文 (代码+diff+结果+决策)"""

    def get_warm_context(self) -> str:
        """温层: 6~15 轮摘要 (策略+加速比+瓶颈类型+决策)"""

    def get_cold_data(self) -> List[dict]:
        """冷层: 16+ 轮仅关键数据点 (round, speedup, bottleneck_type)"""

    def get_context_for_round(self, current_round: int) -> ContextBundle:
        """构建当前轮次应注入的全部上下文"""
```

**摘要格式** (温层):
```
Round 6: increase_tile_size (Tier 1) | 1.08× | bottleneck: ub_to_gm(memory_bandwidth→memory_latency) | KEEP
Round 7: merge_transfers (Tier 2) | 1.03× | bottleneck: vadd(compute_vec) | KEEP
...
```

---

### 文件 17: `memory/context_manager.py`

**做什么**: 管理 LLM 上下文——构建 prompt、估算 token、裁剪超限

**依赖**: `config.py`, `memory/sliding_window.py`

**核心类**: `ContextManager`

```python
class ContextManager:
    def build_prompt_context(self, round_num: int,
                             window: SlidingWindow,
                             bottleneck: BottleneckReport,
                             extracted_data: ExtractedData,
                             playbook_text: str,
                             similar_cases: List[dict]) -> str:
        """构建完整的 LLM prompt 上下文"""

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数 (保守估计: chars/2 for English, chars/1.5 for Chinese)"""

    def trim_if_needed(self, context: str, max_tokens: int) -> str:
        """裁剪: 优先保留热层 + 关键数据, 裁剪温层和案例"""

    def compact_history(self, window: SlidingWindow) -> str:
        """温层摘要压缩"""
```

**上下文优先级** (裁剪时从上到下保留):
1. Playbook 相关章节 (最高优先级, 不能裁)
2. 瓶颈关键数据 (ExtractedData.critical_data)
3. 热层上下文 (最近 5 轮)
4. 当前 kernel 代码
5. 温层摘要
6. 相似案例
7. 冷层数据点

---

### 文件 18: `memory/experience_retriever.py`

**做什么**: 从现有 `memory/` 模块检索相似算子经验和优化案例

**依赖**: `config.py`, 项目 `memory/` 包 (retrieve.py, schema.py, fingerprint.py)

**核心类**: `ExperienceRetriever`

```python
class ExperienceRetriever:
    def retrieve_similar_cases(self, kernel_fingerprint: str, n: int = 3):
        """检索相似算子的优化案例"""

    def retrieve_effective_strategies(self, bottleneck_type: str, n: int = 5):
        """检索对同类瓶颈有效的策略"""

    def retrieve_failed_approaches(self, bottleneck_type: str):
        """检索对同类瓶颈已经失败的方法 (避免重复)"""

    def format_for_prompt(self, results: List) -> str:
        """格式化为可注入 prompt 的文本"""
```

**对接方式**:
```python
from memory import retrieve, compute_fingerprint, format_context

# 检索相似经验
hits = retrieve(store, fingerprint, n=3)
context = format_context(hits)
```

**不修改 `memory/` 包**——只读调用其检索 API。

---

## Phase 6: 智能体层

### 文件 19: `agents/planner.py`

**做什么**: 规划智能体——分析瓶颈 + 生成本轮优化计划

**依赖**: `analyzers/bottleneck_diagnoser.py`, `analyzers/data_extractor.py`, `memory/context_manager.py`, `config.py`

**核心类**: `PlannerAgent`

```python
class PlannerAgent:
    def analyze_bottleneck(self, pipeline_report: str,
                           merged_ops: List[CombinedOp]) -> BottleneckReport:
        """调用 BottleneckDiagnoser, 返回瓶颈诊断"""

    def generate_round_plan(self,
                            bottleneck: BottleneckReport,
                            extracted_data: ExtractedData,
                            context_bundle: ContextBundle,
                            playbook_text: str,
                            round_num: int) -> RoundPlan:
        """★ 核心: 生成 {round_num}_plan.md

        这是 Planner 的主要工作——构建完整 prompt 调用 LLM, 让 LLM 生成:
        1. 本轮优化目标 (预期加速比)
        2. 具体优化手段 (1个, 小改动)
        3. 预期变更范围 (哪些参数/哪些行)
        4. 验证方法 (需要跑什么测试)
        """

    def select_strategy(self, bottleneck: BottleneckReport,
                        history: List[RoundRecord]) -> str:
        """根据瓶颈类型 + 历史尝试 + 策略层级选择优化方向"""

    def determine_tier(self, history: List[RoundRecord]) -> int:
        """根据最近几轮效果判断当前应在哪个策略层级"""
```

**`RoundPlan` 数据结构**:

```python
@dataclass
class RoundPlan:
    round_num: int
    strategy: str
    strategy_tier: int
    target_speedup: float
    bottleneck_description: str
    specific_change: str       # 具体的代码变更描述
    expected_impact: str       # 预期效果 (如: "GM→UB 带宽利用率 46%→80%+")
    verification_method: str   # 验证方法
    fallback_if_fail: str     # 如果失败, 备选方案
```

**LLM Prompt 结构** (Planner 构造的):

```
[System]
You are an expert Triton kernel optimizer for Huawei Ascend 910B3 NPU.
Your job: analyze the bottleneck and generate ONE specific, small optimization plan.

[Playbook] (注入相关章节)
<playbook_memory.md §1, §2>

[Hardware Context]
910B3: 20 AI Core @ 1.8GHz, 40 Vec Core, UB=192KB/core, L2=192MB

[Bottleneck Analysis]
<ExtractedData.critical_data>

[Recent History]
<ContextBundle.hot_context>
<ContextBundle.warm_context>

[Similar Cases]
<similar_cases>

[Current Kernel Code]
<kernel_code>

[Task]
Generate a detailed optimization plan for Round {N}:
{{
  "strategy": "...",
  "strategy_tier": ...,
  "target_speedup": ...,
  "specific_change": "...",
  "expected_impact": "...",
  "verification_method": "...",
  "fallback_if_fail": "..."
}}
```

**LLM 调用方式**:
```python
# 架构设计中的关键决定: Planner 不做自己的优化算法
# 它把上下文拼好 → 调用 LLM → 解析结构化输出

import anthropic  # 或使用 claude-code API

response = client.messages.create(
    model="claude-sonnet-5",  # 或 claude-opus-4-8
    max_tokens=4096,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}]
)
plan = json.loads(response.content[0].text)
```

---

### 文件 20: `agents/coder.py`

**做什么**: 编码智能体——按 Planner 的计划修改代码

**依赖**: `agents/planner.py`, `config.py`

**核心类**: `CoderAgent`

```python
class CoderAgent:
    def apply_optimization(self, kernel_code: str,
                           plan: RoundPlan,
                           context: str) -> CoderResult:
        """★ 核心: 按计划修改代码, 返回修改后的代码 + diff"""

    def generate_diff(self, original: str, optimized: str) -> str:
        """生成 unified diff"""

    def validate_syntax(self, kernel_code: str) -> bool:
        """Python 语法检查: compile(kernel_code, '<string>', 'exec')"""

    def estimate_change_size(self, diff: str) -> dict:
        """估算变更规模: {lines_changed, files_changed}"""
```

**`CoderResult`**:

```python
@dataclass
class CoderResult:
    success: bool
    optimized_code: str
    diff: str
    lines_changed: int
    error_message: str = ""
```

**原则**:
- 单文件变更 (只改 kernel 文件)
- 最小化改动 (一次只改 1 个参数/1 个模式)
- 保持可回退 (diff 记录, revert 容易)
- 如果 LLM 生成的代码有语法错误 → 自动重试 1 次 (把错误信息喂回去)

**Coder 的 LLM Prompt**:

```
[System]
You are a Triton kernel code modifier. Apply EXACTLY the change described in the plan.
Output ONLY the modified code, no explanation.

[Plan]
<plan.specific_change>

[Current Code]
<kernel_code>

[Constraints]
- Change ONLY what the plan specifies
- Maintain existing code style
- Keep all existing comments
- Output the COMPLETE modified file, not just the changed lines
```

---

### 文件 21: `agents/verifier.py`

**做什么**: 验证智能体——协调两阶段验证 (CPU Emulator + 910B3 Hardware)

**依赖**: `execution/emulator_runner.py`, `execution/simulator_runner.py`, `execution/hardware_runner.py`, `config.py`

**核心类**: `VerifierAgent`

```python
class VerifierAgent:
    def __init__(self):
        self.emulator = EmulatorRunner()
        self.simulator = SimulatorRunner()
        self.hardware = HardwareRunner()  # 可能为 None (本地环境)

    def verify_cpu(self, kernel_code: str) -> EmulatorResult:
        """Stage 1: CPU Emulator 验证"""

    def verify_simulator(self, kernel_code: str,
                         dsl_program: str) -> SimulatorResult:
        """Stage 2: Cost Simulator 预估"""

    def verify_hardware(self, kernel_code: str) -> Optional[HardwareResult]:
        """Stage 3: 910B3 真机验证 (仅在 910B3 环境可用)"""

    def full_verification(self, kernel_code: str,
                          dsl_program: str,
                          skip_hardware: bool = False) -> FullVerificationResult:
        """★ 完整验证流程: Stage 1 → Stage 2 → Stage 3"""

    def compare_before_after(self, dsl_before: str, dsl_after: str) -> dict:
        """优化前后 simulator 数据对比"""
```

**验证流程决策逻辑**:
```
Stage 1 (CPU): 每轮必跑, PASS → Stage 2, FAIL → 重试最多 3 次
Stage 2 (Sim): Stage 1 PASS 后跑, 提供参考数据, 不强制 gate
Stage 3 (HW):  Stage 1 PASS 后跑. 本地环境跳过, 910B3 环境运行
```

**`FullVerificationResult`**:

```python
@dataclass
class FullVerificationResult:
    stage1_passed: bool
    stage1_details: EmulatorResult
    stage2_passed: bool  # simulator 永远 "PASS" (它只预估, 不 gate)
    stage2_details: Optional[SimulatorResult]
    stage3_passed: Optional[bool]  # None = 未运行
    stage3_details: Optional[HardwareResult]
    overall_passed: bool  # Stage 1 必须 PASS
    speedup_estimated: float  # 来自 simulator 比较
    speedup_actual: Optional[float]  # 来自 hardware 比较
```

---

### 文件 22: `agents/orchestrator.py`

**做什么**: 调度器——协调整体优化循环 (★ 最核心的文件)

**依赖**: 所有以上模块

**核心类**: `Orchestrator`

```python
class Orchestrator:
    def __init__(self, kernel_file: Path, target_speedup: float = 1.5):
        # 初始化所有子模块
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.verifier = VerifierAgent()
        self.merger = DSLMerger()
        self.diagnoser = BottleneckDiagnoser()
        self.extractor = DataExtractor()
        self.logger = RoundLogger()
        self.journal = OptimizationJournal(...)
        self.stop_checker = StopChecker(...)
        self.context_mgr = ContextManager()
        self.window = SlidingWindow()
        self.retriever = ExperienceRetriever()

        self.current_kernel = self._load_kernel(kernel_file)
        self.best_kernel = self.current_kernel
        self.best_speedup = 1.0
        self.round = 0
        self.start_time = time.time()

    def run(self) -> FinalReport:
        """★ 主循环: 一直迭代直到停止条件满足"""
        while True:
            self.round += 1
            result = self.run_single_round()
            if self.stop_checker.check(self.journal, self._current_state()):
                break
        return self._generate_final_report()

    def run_single_round(self) -> RoundRecord:
        """单轮优化: Plan → Code → Verify → Decide → Record"""
        # Step 0: 上下文准备
        # Step 1: 瓶颈重分析
        # Step 2: 生成本轮计划
        # Step 3: 代码修改
        # Step 4: CPU Emulator 验证 (FAIL→重试最多3次)
        # Step 5: Simulator 预估
        # Step 6: Hardware 验证 (可选)
        # Step 7: 决策 (KEEP/REVERT)
        # Step 8: 记录
        # 返回 RoundRecord

    def _manage_context(self) -> ContextBundle:
        """上下文管理: 滑窗 + 检索 + 构建 prompt"""

    def _check_stop(self) -> bool:
        """停止条件检查"""

    def _generate_final_report(self) -> FinalReport:
        """生成最终报告 + 轨迹图 + 案例 (如适用)"""
```

**单轮详细流程** (与 ARCHITECTURE_DESIGN.md §3 一致):

```python
def run_single_round(self) -> RoundRecord:
    # Step 0: 上下文准备
    context_bundle = self._manage_context()

    # Step 1: 瓶颈分析
    dsl = self._generate_dsl(self.current_kernel)
    sim_result = self.msprof_analyzer.run_simulator(dsl)
    merged_ops = self.merger.merge(sim_result, hivmir_ops)
    bottleneck = self.diagnoser.diagnose(merged_ops, ...)

    # Step 2: 提取关键数据 + 注入 Playbook
    extracted = self.extractor.extract(merged_ops, bottleneck)
    playbook = self._load_playbook(extracted.playbook_file, extracted.playbook_sections)

    # Step 3: Planner 生成计划
    plan = self.planner.generate_round_plan(bottleneck, extracted,
                                             context_bundle, playbook, self.round)

    # Step 4: Coder 修改代码
    coder_result = self.coder.apply_optimization(self.current_kernel, plan, ...)

    # Step 5: CPU Emulator 验证
    emu_result = self.verifier.verify_cpu(coder_result.optimized_code)

    retry_count = 0
    while not emu_result.passed and retry_count < EMULATOR_RETRY_MAX:
        # 把错误信息喂回 Coder, 让它修复
        coder_result = self.coder.apply_optimization(
            self.current_kernel, plan, ...,
            previous_error=emu_result.error_details)
        emu_result = self.verifier.verify_cpu(coder_result.optimized_code)
        retry_count += 1

    if not emu_result.passed:
        return self._record_revert(plan, "Emulator verification failed after retries")

    # Step 6: Simulator 预估
    sim_compare = self.verifier.compare_before_after(dsl_before, dsl_after)

    # Step 7: Hardware 验证 (可选)
    hw_result = None
    if self.hardware_available:
        hw_result = self.verifier.verify_hardware(coder_result.optimized_code)

    # Step 8: 决策
    actual_speedup = hw_result.speedup if hw_result else sim_compare.estimated_speedup
    decision = 'KEEP' if actual_speedup > MIN_SPEEDUP_THRESHOLD else 'REVERT'

    if decision == 'KEEP':
        self.current_kernel = coder_result.optimized_code
        if actual_speedup > self.best_speedup:
            self.best_kernel = coder_result.optimized_code
            self.best_speedup = actual_speedup

    # Step 9: 记录
    record = self.logger.log_round(...)
    self.window.add_round(record)
    return record
```

---

## Phase 7: 优化器

### 文件 23: `optimizers/base_optimizer.py`

**做什么**: 优化器基类——定义通用接口

**依赖**: 无

```python
class BaseOptimizer(ABC):
    @abstractmethod
    def apply(self, kernel_code: str, params: dict) -> str:
        """应用优化, 返回修改后代码"""

    @abstractmethod
    def validate(self, kernel_code: str) -> bool:
        """验证优化后代码是否可编译"""

    def revert(self, original_code: str) -> str:
        """回退到原始代码"""
        return original_code

    def generate_diff(self, original: str, optimized: str) -> str:
        """生成 diff"""
```

**优化器不是纯代码逻辑**——它们依赖 LLM 来做实际的代码修改。真正的"优化逻辑"在 Playbook 和 LLM 的推理中。Optimizer 类提供的是验证和辅助逻辑 (如 tile 上限计算)。

### 文件 24-27: 具体优化器

每个都继承 `BaseOptimizer`，提供 910B3 特定的验证逻辑 + LLM prompt 模板:

| 优化器 | 验证逻辑 | LLM Prompt 模板核心 |
|---|---|---|
| `tile_optimizer.py` | tile ≤ UB_CAPACITY / n_bufs | 根据饱和度曲线计算推荐的 tile size |
| `memory_optimizer.py` | 合并后 size ≤ UB, 检查 k0 半饱和点 | 找到相邻同类型小传输, 合并 |
| `fusion_optimizer.py` | 融合后 UB 容量检查 | 识别 RAW 链上的逐元素操作, 合并 |
| `compute_optimizer.py` | VecUnit 已达峰值则跳过 | double buffer/pipeline overlap 模板 |

---

## Phase 8: 入口 + 手册

### 文件 28: `main.py`

```python
#!/usr/bin/env python3
"""
Triton Agent Optimizer — 主入口

Usage:
  python main.py --kernel <path> [--target-speedup 1.5] [--max-rounds 200]
  python main.py --kernel <path> --dry-run  # 只分析, 不优化
  python main.py --resume <output_dir>       # 从断点恢复
"""
```

### 文件 29-35: Playbooks (7 个 .md)

详细内容按 `ARCHITECTURE_DESIGN.md §1.6` 的章节结构编写。每个章节:
- 标题 + 触发条件 (什么情况下用这章)
- 910B3 相关参数 (精确数值)
- 操作步骤 (LLM 可直接执行)
- 验证方法 (如何确认优化有效)
- 注意事项 + 常见陷阱

### 文件 36: `cases/template.md`

优秀案例模板, 包含:
- 算子基本信息
- 优化前后对比表
- 每轮决策链
- 关键 diff
- 经验总结
- 适用场景

---

## 对话执行计划

按照依赖关系, 分 8 次对话, 每次做一个 Phase:

| 对话 | Phase | 文件数 | 预计对话轮次 | 输入 |
|---|---|---|---|---|
| **对话 1** | Phase 1 → `config.py` | 1 | 1 轮 | 无特殊输入 |
| **对话 2** | Phase 2 → 分析层 | 5 | 3~5 轮 | 需要参考 `fusion_pipeline/complete_data_merge.py` |
| **对话 3** | Phase 3 → 执行层 | 4 | 3~4 轮 | 需要参考 `emulators/common/`, `bench_910b3_paths.py` |
| **对话 4** | Phase 4 → 反馈层 | 5 | 3~4 轮 | 参考 AutoKernel plot.py |
| **对话 5** | Phase 5 → 记忆层 | 3 | 2~3 轮 | 参考项目 `memory/` 包 |
| **对话 6** | Phase 6 → 智能体层 | 4 | 4~6 轮 | 需要深度讨论 LLM prompt 设计 |
| **对话 7** | Phase 7 → 优化器 | 5 | 3~4 轮 | 需要参考 Playbook 内容 |
| **对话 8** | Phase 8 → 入口+手册+测试 | 10 | 2~3 轮 | 收尾 + 测试 |

**总计: ~22 个文件, 8 次对话, 预计 20~30 轮次**

---

## 每次对话开始时的 Prompt 模板

```
请阅读 claude_resume_summary/resume_2026-07-23.md 恢复上下文。
然后阅读 triton_agent_optimizer/ARCHITECTURE_DESIGN.md 了解架构全貌。
参考 IMPLEMENTATION_PLAN.md 的 Phase X。
开始实现第 N 个文件: <file_path>。

具体要求:
1. 先阅读参考文件 (如有)
2. 理解该文件在整体架构中的位置
3. 写完整代码 (包括 type hints, docstrings, 错误处理)
4. 不要依赖还未实现的模块——用 ABC/stub 隔离
5. 完成后总结: 文件做了什么、被哪些文件依赖、下一步是什么
```

---

*此文档由 Claude Code 在 2026-07-23 生成, 是一份逐文件实现 Triton Agent Optimizer 的详细操作计划。*
