"""反馈层 — 优化结果产出 (v4).

各子模块职责:
  - trajectory_chart.py    结束自动出图 (加速比曲线 + tier 色带 + PyTorch/工业级对比线)
  - report.py              结束自动写 final_report.md (成功轮策略摘要 + 全部产物清单)
  - strategy_summary.py    每轮自动写 {all,successful}_strategies.md (final_output/)
  - acceptance_report.py   跨算子验收汇总 (工业级比值) — 命令行手动调
  - remeasure_best.py      best_kernel 复测 (含 L2 命中率量化) — 命令行手动调

各模块在 __init__ 不导出，外部统一通过包路径 import（如
`feedback.trajectory_chart.generate` 被 agents/scheduler.py 在结束时调用）。
"""
