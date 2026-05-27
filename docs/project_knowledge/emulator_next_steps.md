# Emulator Next Steps

## DualRunner 对比调试（待实现）

**问题:** 当前 emulator 约 60-70% 的常见 bug 可精确定位。剩余 30-40% 是"静默数值错误"（stride 互换、pid 解码错误、索引公式错误），emulator 只能报"数值不匹配"但无法指出根因。

**核心思路:** 在 `tl.load/tl.store` 边界上做"双轨对比"——同时跑 buggy kernel 和 reference kernel，逐 program 对比 offsets。第一个发散点就是 bug 根因。

**设计文档:** `docs/DESIGN_dual_runner.md`

## Cost Model（待实现）

**问题:** 完整闭环目标是"正确性 + 性能"。CallCapture 数据天然支持性能分析。

**扩展点:**
- `CostModelAnalyzer.analyze_memory()` — 总 bytes、唯一地址数、访问模式分类
- `CostModelAnalyzer.analyze_compute()` — FLOPs 估算、arithmetic intensity、roofline 对比

## LLM 迭代模式

推荐 **结构化 JSON + 自然语言摘要** 双输出：JSON 让 LLM 精确定位行号，自然语言帮 LLM 理解"为什么错"。
