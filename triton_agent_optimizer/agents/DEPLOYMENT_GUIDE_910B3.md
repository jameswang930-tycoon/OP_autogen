# 910B3 部署指南 — Agents 层

> 智能体层的 910B3 对齐和修改指南。本层有 2 个 LLM Agent (Planner/Coder)，需要 API key。

---

## 0. 前置：LLM 模式切换

Planner 和 Coder 支持两种模式：

| 模式 | 触发条件 | 行为 |
|---|---|---|
| **Stub** | 无 `ANTHROPIC_API_KEY` | 返回占位计划，不改代码 |
| **LLM** | 设置 `ANTHROPIC_API_KEY` | 真实调用 Claude API |

```bash
# 在 910B3 上设置
export ANTHROPIC_API_KEY="sk-ant-xxx"
python agents/orchestrator.py
```

检测逻辑在 `planner.py:183` 和 `coder.py:185`:
```python
self.use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
```

---

## 1. orchestrator.py — 调度器

### 1.1 当前状态

- ✅ 薄循环 (Analyzers→Plan→Code→Verify→Record)
- ✅ 6-Tier 管理 (promotion/demotion)
- ✅ 重试循环 (Verifier FAIL → Coder retry ×3)
- ✅ 所有决策委托给 feedback/record_manager.py

### 1.2 自动检测 (不需要手动修改)

`main.py` 入口自动完成:
- `detect_kernels()` → AST 解析，找到所有 `@triton.jit` 函数名
- `_classify_kernel_type()` → 分析函数体，判断 element_wise/matmul/reduction/attention
- 自动注入 `orch._kernel_fn_name` + `orch._op_type` → 全链路传递

**不再需要手动改 kernel 函数名。**

### 1.3 910B3 上需要确认

| 项目 | 说明 |
|---|---|
| `target_speedup` | 构造函数参数，默认 1.5，根据目标调整 |
| `max_rounds` | 构造函数参数，默认 200，根据预算调整 |

### 1.3 验证命令

```bash
# 本地 stub 模式测试
python agents/orchestrator.py

# 910B3 真实运行
export ANTHROPIC_API_KEY="sk-ant-xxx"
python agents/orchestrator.py outputs/vector_add_fp16_N65536
```

---

## 2. planner.py — 规划智能体

### 2.1 当前状态

- ✅ Prompt 编排: System + User prompt 构建
- ✅ Playbook 加载: 根据 tier 自动读取对应 .md
- ✅ 记忆检索: `memory/experience_retriever.py` (3级匹配)
- ✅ Stub 模式: 根据 tier/headroom 返回合理的占位计划
- ✅ Anthropic API 调用 (有 key 时)

### 2.2 需要在 910B3 上修改

| 项目 | 位置 | 当前值 | 说明 |
|---|---|---|---|
| **API Model** | `_call_llm()` ~L210 | `claude-sonnet-4-20250514` | 可换成 `claude-opus-4-20250514` 或最新模型 |
| **max_tokens** | `_call_llm()` ~L212 | `2048` | 计划 JSON 通常 < 1KB，够了 |
| **System Prompt** | `_build_system_prompt()` | 硬编码 910B3 参数 | 已包含 7 engines + 核数 + UB/L2 |
| **Playbook 路径** | `__init__()` | `_PROJECT_DIR / "docx"` | 确认 docx/ 下有 7 个 .md |

### 2.3 System Prompt 验证

当前 system prompt 包含的 910B3 参数:
- 20 AI Cores + 40 Vec Cores @ 1.8 GHz
- UB = 192 KB/core, L2 = 192 MB, HBM = 64 GB
- 7 engines with peak bandwidth
- Measured vs Placeholder 标注

**在 910B3 上首次运行时, 检查 LLM 输出是否合理**:
```bash
# 手动测试 Planner
python3 -c "
from agents.planner import PlannerAgent
from analyzers.bottleneck_diagnoser import diagnose_round
from pathlib import Path

diag = diagnose_round(Path('outputs/<kernel>/round0'), current_tier=3)
planner = PlannerAgent()
plan = planner.generate(diag, '', 3, [], 'kernel_code', 1)
print(plan.strategy)
print(plan.specific_change)
"
```

---

## 3. coder.py — 编码智能体

### 3.1 当前状态

- ✅ Prompt 编排: Plan + Code → 修改后代码
- ✅ 错误重试: 接收 `previous_error` 修复
- ✅ Python 语法检查: `_validate_python()`
- ✅ Unified diff 生成: `_generate_diff()`
- ✅ 输出清理: 去 markdown 代码块包裹

### 3.2 需要在 910B3 上修改

| 项目 | 位置 | 当前值 | 说明 |
|---|---|---|---|
| **API Model** | `_call_llm()` ~L215 | `claude-sonnet-4-20250514` | 同 Planner |
| **max_tokens** | `_call_llm()` ~L216 | `4096` | kernel 代码可能较长 |
| **System Prompt** | `_build_system_prompt()` | 已约束"只改 kernel.py" | 如需加 910B3 特定约束, 在这里加 |

### 3.3 System Prompt 定制 (可选)

如果发现 LLM 改代码时产生 910B3 不兼容的代码，在 `_build_system_prompt()` 中添加:
```python
## 910B3 Constraints
- UB = 192 KB per core — don't allocate buffers > 192KB
- Use fp16 for compute, fp32 for accumulation
- Transfer grid = 20 (AI Core), compute grid = 40 (Vec Core)
```

### 3.4 验证命令

```bash
# 测试 Coder stub
python agents/coder.py

# 测试真实 LLM 调用
export ANTHROPIC_API_KEY="sk-ant-xxx"
python3 -c "
from agents.coder import CoderAgent
c = CoderAgent()
r = c.apply('import triton; @triton.jit; def k(): pass', 'increase BLOCK_SIZE to 8192')
print(r.success, r.lines_changed)
"
```

---

## 4. verifier.py — 验证智能体

### 4.1 当前状态

- ✅ Stage 1: CPU Emulator (emulators/common 模拟) — 本地可用
- ✅ Stage 2: 910B3 Hardware — 检测编译器, 本地跳过
- ✅ FAIL → Coder 重试循环 (在 Orchestrator 中)

### 4.2 需要在 910B3 上修改

| 项目 | 位置 | 当前值 | 改为什么 |
|---|---|---|---|
| `skip_hardware_on_local` | `__init__()` | `True` | 910B3 上可设为 `False` 强制跑硬件 |
| `kernel_fn_name` | `verify()` 参数 | `"add_kernel"` | **改为实际的 kernel 函数名** |

### 4.3 Stage 2 验证

910B3 上首次验证:
```bash
# 确认编译器可用
which ascendc || which bishengir-compile

# 手动跑一次完整验证
python3 -c "
from agents.verifier import VerifierAgent
from pathlib import Path

v = VerifierAgent(skip_hardware_on_local=False)
r = v.verify(
    kernel_path=Path('round1/kernel.py'),
    round_dir=Path('round1'),
    kernel_fn_name='add_kernel',
    baseline_latency_ms=0.0183,
)
print(f'Stage1: {r.stage1_passed}')
print(f'Stage2: tested={r.stage2_tested}, speedup={r.stage2_actual_speedup}')
"
```

---

## 5. orchestration 完整流程验证

```bash
# 1. 准备
cd triton_agent_optimizer
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ANTHROPIC_API_KEY="sk-ant-xxx"

# 2. 检查环境
python prepare/env_check.py

# 3. 跑一轮完整优化
python agents/orchestrator.py outputs/vector_add_fp16_N65536

# 4. 查看轨迹
cat outputs/vector_add_fp16_N65536/optimization_trajectory.json
ls outputs/vector_add_fp16_N65536/final_output/
```

---

## 待补全清单

| 文件 | 补全项 | 优先级 | 说明 |
|---|---|---|---|
| `orchestrator.py` | `kernel_fn_name` 参数 | ⭐⭐⭐ | 改为实际函数名 |
| `planner.py` | System Prompt 910B3 参数 | ⭐⭐ | 已包含, 验证即可 |
| `coder.py` | 910B3 约束 (如需要) | ⭐ | 可选, 按需添加 |
| `verifier.py` | Stage 2 硬件验证 | ⭐⭐⭐ | 910B3 上 `skip_hardware=False` |
