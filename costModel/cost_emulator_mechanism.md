# cost_emulator 机制总结

## 一句话主线

DSL 程序 → 展开成线性指令流 → 给每条指令算时长（大小相关带宽模型）→ 建数据依赖（RAW/WAW/WAR）→ ASAP 调度排时间窗 → 取占时最大的 op = 瓶颈

它不是真实硬件执行，是**静态模拟**：把你描述的"数据在各级存储间怎么搬运+计算"的程序，按引擎归属和数据依赖排成一张调度图，算出每条指令的起止时间，然后告诉你总耗时和谁拖了后腿。

## 关键的 5 步流水

1. **展开（`emulate`）**：`for` 循环 + 地址偏移（`gm_1 + m*1KB`）由前端展开成 flat 指令流；纯程序走 regex 解析。
2. **算时长（`assign_sizes` + `bandwidth_profile`）**：带宽随传输大小变化——分段线性曲线，小传输 `floor`（吃不满）、中间 `ramp`（线性爬升）、大传输 `saturated`（峰值）。`duration_ns = size_kb×1024 / bandwidth(size)`。只有 **GM→UB 是实测**（1KB→100GB/s，12KB→1500GB/s），其余引擎是 placeholder。
3. **依赖分析（`build_deps` + `hazards`）**：三种冒险 RAW/WAW/WAR；tile-aware——同 buffer 且字节区间重叠才算依赖，不重叠的 tile 能并行。
4. **ASAP 调度（`schedule`）**：每条指令 `start = max(本引擎空闲时刻, 所有依赖完成时刻)`，分到最早可执行时刻。不同引擎天然并行，同引擎排队。
5. **瓶颈提取（`render_llm`）**：`total_ns = max(op.end)`，瓶颈 = `time_ratio`（duration/total）最大的 op；再加 critical path（最长加权链，长度=makespan，告诉你哪条链卡住总时间）。
