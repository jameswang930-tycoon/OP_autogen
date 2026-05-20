# 文件对比分析报告

## 对比对象

| 文件 | 路径 |
|------|------|
| **现有 common/\_\_init\_\_.py** | `emulators/common/__init__.py` (973 行) |
| **副本 common/\_\_init\_\_.py** | `__init__ (1)_副本.py` (818 行) |
| **现有 triton-gen skill** | `.claude/commands/triton-gen.md` (253 行) |
| **副本 SKILL.md** | `SKILL.md` (255 行) |

---

## 一、common/\_\_init\_\_.py 对比

### 现有版本独有（副本缺失）的关键功能

| 功能 | 说明 | 重要性 |
|------|------|--------|
| **AggregatedEmulatorError** | 跨所有 program 去重聚合错误，显示每个错误的出现次数和首次触发 pid | ⭐⭐⭐ 核心 |
| **launch 的 `collect_errors` 参数** | kernel 崩溃时不立即退出，而是收集所有 program 的错误后统一抛出 | ⭐⭐⭐ 核心 |
| **EmulatorError 的 source line 追踪** | 通过 traceback 自动定位到 kernel 源码的行号和代码片段 | ⭐⭐⭐ 核心 |
| **run_with_feedback()** | LLM 自动生成闭环的顶层接口，自动启用 TraceLogger、捕获异常、格式化反馈 | ⭐⭐⭐ 核心 |
| **TraceLogger.\_invocation_id + begin_invocation()** | 追踪每次 launch 调用的批次，区分不同轮次的 trace | ⭐⭐ 有用 |
| **TraceLogger.get_flags_summary_deduped()** | 按 (行号, api, section, tensor, 类别) 去重的异常摘要 | ⭐⭐ 有用 |
| **详细的架构文档字符串** | 包含 4 层架构说明、算子约定、设计原则 | ⭐ 文档价值 |

### 副本独有的优势

| 功能 | 说明 | 重要性 |
|------|------|--------|
| **TraceLogger.format() 调用聚合** | 同一 pid 下相同 (行号, api) 的重复调用自动合并，展开前 2 次和最后 1 次，中间压缩显示 "... repeated N more times"，并汇总 overall_range | ⭐⭐ 有用 |

### 结论：**现有版本远优于副本**

副本本质上是项目**早期版本**的快照——它缺少 AggregatedEmulatorError、collect_errors、run_with_feedback、EmulatorError source line 追踪等后续开发的核心功能。这些功能是支撑 LLM 自动生成闭环的关键基础设施。

**唯一值得从副本合入的特性**：TraceLogger.format() 的调用聚合逻辑。当前版本的 format() 会逐条展开所有 trace entry，在 grid 很大或循环内频繁调用 tl.* API 时输出会极其冗长。副本的聚合方案（按 pid + line + api 分组，只展开代表样本）是更好的做法。

---

## 二、triton-gen skill 对比

### 现有版本 (.claude/commands/triton-gen.md) 的优势

| 特性 | 说明 |
|------|------|
| **双路径设计（生成 vs 修复）** | 开篇就判断场景分叉，流程清晰 |
| **错误三分类（A/B/C）** | 硬错误 / Shape 不匹配 / 数值偏差，每种有独立修复策略 |
| **错误-修复映射表** | 针对具体错误消息给出直接修复方向（如 "offsets OOB, no mask → 加 mask"） |
| **迭代规则明确** | 最多 5 轮，每次最小改动 |
| **7 条 kernel 编写约束** | 独立成段，精确且可检查 |
| **完整 add 算子示例** | 展示四件套结构 |
| **检查清单** | 可逐项核验 |

### 副本 (SKILL.md) 的优势

| 特性 | 说明 |
|------|------|
| **YAML frontmatter** | 有 name/description 元数据，适合注册为可复用 skill |
| **Emulator 架构参考** | 目录树 + API 表格（含 Signature 和 Notes），一目了然 |
| **Common Failure Patterns 表** | 5 种典型 trace 症状 → 原因 → 修复，直观 |
| **完整最小验证脚本** | 从 import 到 verify 的端到端可运行示例 |
| **Step 5: Deliver Results** | 明确交付物格式 |
| **指针包装说明** | 区分 PointerWrapper 和 raw numpy 两种传参方式 |
| **Important Emulator Behaviors** | 明确标注 `keepdims=True` 等关键差异 |

### 现有版本 vs 副本：结构对比

```
现有版本:                          副本:
──────────────────────────        ──────────────────────────
判断场景 (生成/修复)               What This Skill Does
  ├─ 生成路径                        ├─ Step 1: Understand
  │   ├─ 分析语义                    ├─ Step 2: Generate Kernel
  │   ├─ 创建文件                    ├─ Step 3: Verification Script
  │   ├─ 运行验证                    ├─ Step 4: Handle Failures
  │   └─ 注册测试                    ├─ Step 5: Deliver Results
  ├─ 修复路径                        ├─ Architecture Reference
  └─ 迭代修复 (共享)                 │   ├─ 目录树
      ├─ 类型 A: 硬错误               │   └─ API 表格
      ├─ 类型 B: Shape               ├─ Important Behaviors
      └─ 类型 C: 数值                ├─ Failure Patterns 表
API 参考                             └─ 完整示例脚本
约束 (7 条)
示例 (add 算子)
检查清单
```

### 结论：**各有千秋，理想方案是合并**

- **现有版本**的诊断和修复流程更强——错误三分类 + 映射表 + 迭代规则适合实际 debug 场景
- **副本**作为参考文档更完整——API 表格、架构图、Failure Patterns 表、完整可运行示例都是现有版本缺失的

---

## 三、总体建议

### 短期（推荐立即执行）

1. **common/\_\_init\_\_.py**：保持现有版本不变。将副本的 `TraceLogger.format()` 聚合逻辑合入现有版本（这是副本唯一有价值的部分）。

2. **triton-gen skill**：以**现有版本为主体**，从副本合入以下内容：
   - YAML frontmatter（便于 skill 注册）
   - API 参考表格（替换现在的列表形式）
   - Common Failure Patterns 表（补充到迭代修复流程中）
   - 完整的最小验证脚本示例
   - Important Emulator Behaviors 说明
   - Emulator 架构目录树

### 不推荐的合并项

- **不要**替换现有的错误三分类体系为副本的 Failure Patterns 表——两者互补，应共存
- **不要**删除现有的 add 算子完整示例——它比副本的示例更完整（展示了四件套结构）
- **不要**把副本的 Step 1-5 线性流程替换现有的双路径设计——双路径更贴合实际使用场景（有时是新建算子，有时是修复已有代码）

### 一句话总结

**common/\_\_init\_\_.py**：现有版本功能上碾压副本（副本是早期快照），只需从副本摘取 TraceLogger.format() 的聚合逻辑。**triton-gen skill**：现有版本的诊断流程更好，副本的参考文档更全，理想做法是以现有版本为骨架、以副本为血肉合并。
