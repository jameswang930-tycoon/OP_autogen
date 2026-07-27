# AI Agent 使用指南 — 单轮任务执行模式

> 适用场景: `claude -p "一句话任务"` 或任何 LLM 的 `run("一句话任务")` 单轮调用。

---

## 核心原理

**你不需要理解整个项目。** Orchestrator 会把每一轮你需要做的事写成一个自包含文件。
你只需要：读文件 → 执行 → 写结果。一问一答，不对话。

```
Orchestrator (Python) 写任务文件 → 你 (LLM) 读文件、执行、写输出 → Orchestrator 继续
```

---

## 你要处理的两种任务

### 任务 A: 生成优化计划 (`AGENT_TASK_PLAN.md`)

**何时出现**: 每一轮优化的开头。目录 `outputs/<kernel>/<tier>/roundN/` 下。

**触发命令**:
```bash
claude -p "read AGENT_TASK_PLAN.md at outputs/<kernel>/01_algorithmic_structure/round1/ and execute it"
```

**OR** 直接把文件内容复制后在对话框中执行:
```
请阅读以下任务并执行:
[粘贴 AGENT_TASK_PLAN.md 全文]
```

**你要做什么**:
1. 读文件中 `## Bottleneck Diagnosis` 了解当前瓶颈
2. 读文件中 `## Playbook` 路径指向的对应文档（如 `docx/playbook_tier3_tiling.md`）
3. 读文件中 `## Recent History` 了解之前试了什么
4. 读文件中 `## 910B3 Parameters` 检查硬件约束
5. 读文件中 `## Current Kernel Code` 看当前代码
6. 按 `## Output` 的格式生成 `plan.md` 和 `plan.json`，写入该目录

**输出**: `plan.md` + `plan.json`（放到 `## Output` 指定的目录）

### 任务 B: 修改代码 (`AGENT_TASK_CODE.md`)

**何时出现**: 任务 A 完成后。同一目录下。

**触发命令**:
```bash
claude -p "read AGENT_TASK_CODE.md at outputs/<kernel>/01_algorithmic_structure/round1/ and execute it"
```

**你要做什么**:
1. 读同目录下的 `plan.md` 和 `plan.json`（任务 A 的产出）
2. 读文件中 `## Current Kernel Code`
3. 按计划做**最小化**代码修改（只改 kernel.py）
4. 语法检查: `compile(code, "kernel.py", "exec")`
5. 910B3 约束检查: `new_tile_kb × n_buffers ≤ 192 KB`
6. 写完整 `kernel.py` + `diff.patch` 到该目录

**输出**: `kernel.py`（完整文件）+ `diff.patch`

---

## 任务文件结构详解

### AGENT_TASK_PLAN.md 的每个 Section

```
┌──────────────────────────────────────────────┐
│ # AGENT TASK: Generate Optimization Plan     │  ← 任务类型
├──────────────────────────────────────────────┤
│ ## Your Role                                 │  ← 你是谁
│ ## Instructions                              │  ← 怎么做（6步）
│ ## Bottleneck Diagnosis (Tier N)             │  ← 当前瓶颈数据
│ ## Pipeline Data                             │  ← DSL 流水线精简数据
│ ## Recent History                            │  ← 最近5轮: 策略+结果
│ ## 910B3 Parameters                          │  ← 硬件约束速查
│ ## Current Kernel Code                       │  ← 当前 kernel 源码
│ ## Output (write plan.md + plan.json)        │  ← 输出格式模板
└──────────────────────────────────────────────┘
```

**不需要读其他文件**。唯一可能需要额外读的是 `## Instructions` 中指定的 Playbook 文件:
- `docx/playbook_tier1_algorithm.md`
- `docx/playbook_tier2_fusion.md`
- `docx/playbook_tier3_tiling.md`
- `docx/playbook_tier4_memory.md`
- `docx/playbook_tier5_compute.md`
- `docx/playbook_tier6_architecture.md`

### AGENT_TASK_CODE.md 的每个 Section

```
┌──────────────────────────────────────────────┐
│ # AGENT TASK: Apply Code Change              │  ← 任务类型
├──────────────────────────────────────────────┤
│ ## Your Role                                 │  ← 你是谁
│ ## Instructions                              │  ← 怎么做
│ ## Optimization Plan                         │  ← 来自 plan.md
│ ## Previous Error (if retry)                 │  ← 上次失败原因（重试时）
│ ## Current Kernel Code                       │  ← 要修改的代码
│ ## 910B3 Constraints                         │  ← 硬件约束
│ ## Output Files                              │  ← 输出什么
└──────────────────────────────────────────────┘
```

---

## 执行模板

### 模板 A: claude 命令行

```bash
# Planner 任务
claude -p "$(cat outputs/<kernel>/<tier>/roundN/AGENT_TASK_PLAN.md)"

# Coder 任务
claude -p "$(cat outputs/<kernel>/<tier>/roundN/AGENT_TASK_CODE.md)"
```

**关键**: 用 `$(cat ...)` 把整个文件内容作为一句话传给 LLM。
单次调用，不对话。

### 模板 B: 任何 LLM 的 run() API

```python
import anthropic

task = open("outputs/<kernel>/01_algorithmic_structure/round1/AGENT_TASK_PLAN.md").read()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": task}],
)

# LLM 的回复中会包含 plan.md 的内容——解析并保存到文件
print(response.content[0].text)
```

### 模板 C: Web 界面 (ChatGPT / Claude.ai)

1. 用文本编辑器打开 `AGENT_TASK_PLAN.md`
2. Ctrl+A 全选 → Ctrl+C 复制
3. 粘贴到 ChatGPT/Claude.ai 对话框
4. 发送
5. 从 LLM 回复中提取 `plan.md` 内容 → 保存到文件

---

## 910B3 硬件约束（每次都要记住）

| 约束 | 值 | 计算公式 |
|---|---|---|
| UB 容量 | 192 KB/core | `n_buffers × tile_size_kb ≤ 192` |
| fp16 元素大小 | 2 bytes | `tile_kb = BLOCK_SIZE × 2 / 1024` |
| 最大 tile (2 buffers) | 96 KB | `192 / (2×2)` |
| 最大 tile (3 buffers) | 64 KB | `192 / (3×2)` |
| Transfer grid | 20 | AI Cores |
| Compute grid | 40 | Vec Cores |

## 执行规则（硬性）

1. `AGENT_TASK_PLAN.md` → **只读不写**。你要写的是 `plan.md` 和 `plan.json`
2. `AGENT_TASK_CODE.md` → **只读不写**。你要写的是 `kernel.py` 和 `diff.patch`
3. **不要修改任何其他文件**（msprof/hivmir/merged/trajectory/...都不要碰）
4. **每轮只改一个东西**（一个参数、一个模式）
5. **永远先做 Python 语法检查**再输出 kernel.py
6. **永远检查 UB 容量**再建议增大 tile
