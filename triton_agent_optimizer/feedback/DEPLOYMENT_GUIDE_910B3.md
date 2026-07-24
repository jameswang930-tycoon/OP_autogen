# 910B3 部署指南 — Feedback 层
> 本层是决策引擎 + 图表。纯 Python，本地即可运行，不需要 NPU。

---

## 1. record_manager.py — 决策引擎

本地验证: `python feedback/record_manager.py` (导入测试)

**需要确认的参数** (在 Orchestrator 中传入):
- `target_speedup` — 默认 1.5，根据实际目标调整
- `max_rounds` — 默认 200，根据预算调整

---

## 2. trajectory_chart.py — 优化轨迹图

需要 `matplotlib`。安装: `pip install matplotlib`

验证: `python feedback/trajectory_chart.py`

---

## 待补全

| 文件 | 补全项 | 优先级 |
|---|---|---|
| `trajectory_chart.py` | 中文字体 (910B3 Linux 通常有 Noto Sans CJK) | ⭐ |
| `record_manager.py` | 无需修改 | ⭐ |
