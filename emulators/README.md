# emulators/ — 已退役（仅作历史参考）

> **状态：退役。** CPU 侧 emulator（解析式 cost-model 流水预测）已退役，仅作历史参考，新流水线不使用。

## 背景

本目录属于旧的流水线因果方向：cost model **预测**流水 → 生成 kernel。
本次改造（分支 `local-adapt`）将因果方向**反转**为：生成 kernel → 远端真实仿真实测 → 分析瓶颈 → 再生成。
解析式流水预测随之退役，真实仿真成为测量真值。详见 `docs/local_adaptation_guidance.md`。

## 为什么不删除

`common/__init__.py` 记录了 `tl.*`（emulator 侧 Triton 方言）的实现语义。
改造 `triton-gen` skill（T8b）时可能需要参考这些语义映射，故保留为只读历史参考。

## 新流水线不再使用本目录

- 新的瓶颈分析由 `control/feedback_adapter.py`（解析真实仿真反馈）+ `sim-analyze` skill 承担。
- 新的 kernel 生成产物为**真实 Triton + extension**（多段式模块），不再产出 emulator 形态代码。
- 正确性比对由发射脚本（`control/launch_template.py`）自带 reference 完成，不再依赖 emulator。

**请勿在本目录新增功能代码。**
