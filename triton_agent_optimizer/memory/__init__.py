"""记忆层 — v4.5:
  - memory/excellent_cases: 优秀优化案例 (每 tier 一个 JSON, 阈值 1.3×, planner 参考)
  - memory/failed_cases: 失败案例库 (每 tier 一个 JSON, 指纹去重 + 两级检索 + attempted_solutions
    方案收敛守卫 + open/solved/stuck 状态机 + 负正闭环; coder 修复注入 + scheduler 累积上下文)
"""
