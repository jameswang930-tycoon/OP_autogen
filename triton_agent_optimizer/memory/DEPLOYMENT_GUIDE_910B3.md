# 910B3 部署指南 — Memory 层

## 记忆层架构

```
每轮优化:
  │
  ├─ Planner 调用:
  │   experience_retriever.retrieve(op_type, bottleneck_type, engine, tier)
  │     → 3 级匹配: 精确→bottleneck_type→engine
  │     → 返回 top 5 案例 (SUCCESS 优先)
  │   format_for_prompt(cases)
  │     → 注入 Planner system prompt
  │
  ├─ context_manager.build_context(...)
  │     → Token 估算 + 自动裁剪 (热/温/冷三层)
  │
  └─ RecordManager 调用:
      experience_retriever.record(tier, fingerprint, strategy, speedup, status)
        → speedup > 1.05 → SUCCESS
        → speedup < 0.98 → FAIL
        → 0.98~1.05 → 不记录
```

## 三个文件详解

### 1. experience_retriever.py — 经验检索+记录 (280行)

**3 级匹配逻辑**:
```
Level 1: 精确 {op_type, bottleneck_type, engine, tier}
Level 2: 放宽 {bottleneck_type, tier}
Level 3: 再放宽 {engine, tier}
```

**存储**: 6 个 JSON 文件 `experiences/tier1_algorithm.json` ~ `tier6_architecture.json`
**上限**: 每个文件最多 50 条 (避免过大)

**已接入**:
- Planner._retrieve_similar_cases() → `retrieve()` (注入 prompt)
- RecordManager._record_experience() → `record()` (记录成功/失败)

### 2. context_manager.py — 上下文管理 (148行)

**Token 估算**: `estimate_tokens(text)` → chars/2 (保守估计)
**自动裁剪**: 超 800K tokens 时 → 先裁案例, 再裁历史, 最后裁代码
**诊断格式化**: `format_diagnosis(diag)` → Markdown 文本

**已接入**: Planner.generate() 使用 `build_context()` 构建完整 prompt

### 3. sliding_window.py — 滑动窗口 (102行)

**三层窗口**: Hot(5轮完整) / Warm(6-15轮摘要) / Cold(16+轮数据点)
**输出**: `get_context_for_prompt()` → 可注入 LLM 的文本

## 910B3 上不需要修改

记忆层是纯 Python 逻辑, 不依赖硬件。经验库自动在优化过程中填充。

**验证**:
```bash
python memory/experience_retriever.py  # 自测: 记录+检索+格式化
```
