# Example Output — 参考示例

本目录包含三个类别的参考文件：

---

## 类别 A: cost_emulator simulator.py 输出 (理论参考)

这些是 `costModel/cost_emulator/simulator.py` 的标准输出, 展示 **目标输出格式**。

| 文件 | 格式 | 说明 |
|---|---|---|
| `01_vector_add_saturated.txt` | `--llm` 文本 | Vector add, 3 ops, 全饱和, 3655ns |
| `02_for_loop_small_tile.txt` | `--llm` 文本 | 10×1KB tile loop, 20 ops, 全 floor |
| `03_single_load_1KB_floor.txt` | `--llm` 文本 | 1KB 单传输, floor 极端案例 |
| `04_matrix_pipeline_parallel.txt` | `--llm` 文本 | 双矩阵乘法, 12 ops, 7 对并行 |
| `05_full_gantt_vector_add.txt` | Gantt 图 | Vector add ASCII 流水图 (人读) |
| `06_full_gantt_for_loop.txt` | Gantt 图 | For-loop ASCII 流水图 (人读) |
| `07_full_gantt_matrix_pipeline.txt` | Gantt 图 | Matrix pipe ASCII 流水图 (人读) |

---

## 类别 B: msprof_analyzer.py 真实输出 (msprof 解析结果)

| 文件 | 说明 |
|---|---|
| `mock_pipeline_report_vector_add.json` | ★ msprof op simulator 解析产出示例。展示 Vector add (3 op) 的完整流水线报告。**msprof 提供的字段已填充, HIVMIR 字段标记为 "待补充"**。 |
| `mock_pipeline_report.json` | msprof_analyzer.py 自测生成的 mock 输出 (含 Vector + Matrix pipeline)。 |

### 关键: msprof 能拿到的 vs HIVMIR 待补充的

```
✅ msprof traces.json 直接提取 (16个字段):
   op_id, op_type, engine, start_ns, end_ns, duration_ns,
   time_ratio, pipeline_channel, core_id, trace_event_name,
   total_ns, num_ops, execution_mode, num_cores,
   engine_utilization, parallel_pairs

❌ HIVMIR 后续补充 (13个字段):
   instruction, dst, src, src2, variable_name, size_kb,
   memory_region, effective_bw, peak_bw, bw_utilization,
   regime, wait_before_start_ns, blocked_by
```

详见 **[FIELD_SOURCE_MATRIX.md](FIELD_SOURCE_MATRIX.md)**。

---

## 类别 C: 字段来源矩阵

| 文件 | 说明 |
|---|---|
| `FIELD_SOURCE_MATRIX.md` | ★ 完整字段来源表。每个字段标注来源 (msprof/HIVMIR/simulator), 获取方式, 示例值, 和合并脚本职责。 |

---

## 数据流

```
                    910B3 服务器
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
    msprof op simulator          Ascend 编译器
          │                      (--run-mode=sim -g)
          ▼                             │
    trace.json                          ▼
          │                       HIVMIR .mlir
          ▼                             │
    msprof_analyzer.py                  ▼
          │                      hivmir_analyzer.py
          ▼                             │
    pipeline_report.json                ▼
    (16 fields filled,          hivmir_data.json
     13 marked "待补充")              │
          │                             │
          └──────────┬──────────────────┘
                     ▼
               dsl_merger.py
          (通过 op_id 对齐, 填补 "待补充")
                     │
                     ▼
            完整 DSL 流水线报告
       (29 fields all filled, 对齐
        cost_emulator/simulator.py --llm 格式)
```
