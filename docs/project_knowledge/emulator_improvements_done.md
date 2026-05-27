# Emulator Improvements Done (2026-05-20)

对 `emulators/common/__init__.py` 进行了错误反馈精准化整改，解决了 ResNet 等场景下同一 bug 被大量 program 重复报错的问题。

## 改动清单

1. **EmulatorError 结构化行号** — 新增 `source_line`/`source_file`/`source_code` 属性，通过 `traceback.extract_stack()` 自动捕获 kernel 源码行号
2. **AggregatedEmulatorError** — 按 `(行号, API)` 聚合多 program 的同类错误
3. **TraceLogger 增强** — 新增 `_invocation_id` 批次标记、`begin_invocation()`、`get_flags_summary_deduped()` 去重摘要
4. **launch_kernel 增加 `collect_errors=True`** — 跑完所有 program 再聚合报错
5. **verify() 使用去重摘要** — 替换原来逐条列举的 verbose 格式
6. **run_with_feedback()** — 一步到位：enable TraceLogger → 运行 → verify → 产出精简反馈 → disable
7. **conv1d/conv2d 测试更新** — 新增 `run_with_feedback` + `collect_errors` 演示用例

## 效果

- conv2d 288 个 program 的 OOB 错误：从重复 288 次 → **1 行精准报告**
- TraceLogger 软错误：从 2304 条 trace → **6 行去重摘要**
- 所有现有测试向后兼容通过

## Skill 创建

创建了 `.claude/commands/triton-gen.md` 项目级 slash command，后续扩展为支持 5 种输入类型的多模态 skill。
