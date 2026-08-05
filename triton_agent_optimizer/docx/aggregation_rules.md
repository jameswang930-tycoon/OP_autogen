# 最终 JSON 聚合规则（msprof 骨架 + msprof op 逐个填充）

> 目标：`msprof`（通用）先生成**全字段骨架 JSON**（任务级字段填满，深层字段待填），再对每个 distinct kernel 跑一次 `msprof op`，把深层字段**按 kernel 名**逐个填进骨架，最终输出完整产物。
> 环境：Ascend 910B3 / CANN 8.5.1。字段名规范见 `msprof_fields_reference.md` 第五、六节。

---

## 0. 总体架构

```
        通用 msprof（跑一遍，全部 kernel）
              │  parse
              ▼
      task.json = 骨架（全字段，部分填充）
              │  num_kernels 决定跑几次 op
              ▼
  对每个 distinct kernel 名 → msprof op --kernel-name=<名>
              │  parse（8 CSV → deep）
              ▼
      board_<i>.json（每 kernel 一个）
              │  按 kernel 名匹配
              ▼
      integrate → 最终 diagnosis.json（kernels[].deep 填满）
              │  check_fields 校验
              ▼
         完成
```

两条链在**一个键上汇合：`Op Name`（kernel 名）**。

---

## 1. msprof（通用）→ 骨架 JSON 的产生规则

### 规则 R1 — Schema-first（字段集先定死）
- 最终 JSON 的字段名**预先定义**（见 `msprof_fields_reference.md` 第五、六节），骨架里**所有字段名都在**，值填了或 `null`（待填）。
- 语义字段用 normalized 名（`main_mem_read_gb_s`），每个字段可标 `source`（工具/文件/原始列名）。
- 骨架 JSON = `task.json`。

### 规则 R2 — 文件 → 区块（一个文件对应一个 section）
| msprof 文件 | → 区块 | 粒度 |
|---|---|---|
| op_summary | `kernels[]`（每次 launch 一行，原始保留） | 行 |
| op_summary（去重） | `kernel_slots[]`（★骨架槽位，merge 目标） | 每 distinct kernel 一个 |
| op_statistic | `multi_kernel[]` | 每算子类型 |
| api_statistic | `api_overhead[]` | 每 API |
| l2_cache | `l2_hit_rate` | 单值 |
| task_time / msprof json | `raw[]` | 原样保留 |

### 规则 R3 — 行 → 槽位（按 kernel 名去重）
- op_summary 一行 = 一次 launch；同一 `Op Name` 多次 launch → **合并成一个槽位**，记 `launch_count`。
- 槽位 key = `Op Name`（后 op 填进来靠它）。
- `num_kernels` = distinct Op Name 数（不是 launch 数）。

### 规则 R4 — 列 → 字段（子串匹配 + 归一化）
- 每行遍历列名，**含所有关键字子串的第一列**就是该字段值。
  - 例：`task_duration_us` ← 列名含 `Task`+`Duration` → 命中 `Task Duration(us)`。
- 找不到 → `null`，不报错（**合法缺**，版本差异），由 check_fields 事后分类。
- 归一化：时间保持 µs；L2 命中 >1 → ÷100 成 0~1；带宽按量级 ≥1e4 MB/s→GB/s。

### 规则 R5 — 聚合
- 多次 launch 同 kernel：槽位 task 字段取首个 launch，`launch_count` 累加。
- `api_overhead_total_us` = 所有 API `Time(us)` 求和。
- `multi_kernel[]` 由 op_statistic 原样聚合（类型 → 次数/总耗时/占比）。

### 骨架产物
`summary`（num_kernels/total_ns/api_overhead/l2）+ `kernels[i]`：
- `task` 子对象：task_type / task_duration_us / block_dim / input+output shapes+dtypes / aicore+aiv_time / total_cycles / **pipes_us**（每 pipe 耗时）/ est_bytes / transfers —— 全部填满
- `deep` = null（待填）

---

## 2. msprof op → 提取规则（每 OPPROF 折叠成一个 `deep`）

### 规则 O6 — 一次 OPPROF = 一个 kernel
- 对骨架每个 distinct kernel 名，跑 `msprof op --kernel-name=<名>`（filtered，逐算子执行，与优化顺序一致）。
- 每个 OPPROF 目录的 8 CSV → 折叠成一个 `deep` 子对象。

### 规则 O7 — 8 CSV → deep 固定映射
| OPPROF CSV | → deep 字段块 |
|---|---|
| OpBasicInfo | `freq_mhz`（并核对 kernel 名/耗时） |
| PipeUtilization | `engine_utilization`（各 `*_ratio`） |
| ArithmeticUtilization | `compute`（fops/instr/ratio） |
| Memory + MemoryL0 + MemoryUB | `bandwidth_gb_s`（17 通路） |
| L2Cache | `l2_hit_rate` |
| ResourceConflictRatio | `conflict` |
| （由带宽+算力算出） | `roofline` |

### 规则 O8 — 列 → 字段走核实映射表
`aic_main_mem_read_bw` → `main_mem_read_gb_s`；`aiv_vec_bank_cflt_ratio` → `conflict.bank_cflt_ratio`；……全部子串匹配（见 `msprof_fields_reference.md` 第五节）。

### 规则 O9 — 值域归一化
- 带宽 MB/s→GB/s（≥1e4 判据）；L2 命中 >1 → 0~1；ratio 保留 0~1。

### 规则 O10 — 多 block 聚合（待细化，先定默认）
- Memory/Pipe/冲突 CSV 每 block 一行（有 `block_id`）。
- 默认：取 rows[0]；后续可改：**ratio 取 max（最差 block）、bytes 求和、时间取 max**。

---

## 3. 合并规则（op 数据进骨架）

### 规则 M11 — kernel 名匹配
- `kernels[i].deep = 对应 board 的 normalized`，匹配键 = `Op Name`。
- 骨架有 kernel、无 board（op 没跑到）→ `deep=null`，`filled_by="msprof only"`。
- board 有、骨架无 → 告警（op 跑到参考 kernel 被漏，正常）。

### 规则 M12 — roofline 用该 kernel 自己的 deep 算
- `achieved_mem_bw = max(main_mem_read/write)` 对 1.8TB/s
- `achieved_compute = (cube_fops + vector_fops)/1e12` 对 294.9TFLOPS
- → 判 `memory / compute / latency / balanced bound`（每 kernel 一个，不是全局一个）

### 规则 M13 — 校验收尾
- `filled_kernels == num_kernels` 才算完整；check_fields 逐字段报「列名不匹配(BUG) / 合法缺(正常)」。
- 退出码：有列名不匹配 → 1。

---

## 4. 最终 JSON 结构

```json
{
  "meta": {"num_kernels": N, "filled_kernels": M, "inputs": {task, boards}},
  "summary": {"num_kernels": N, "total_ns": .., "num_cores": ..,
              "api_overhead_total_us": .., "l2_hit_rate": .., "filled_kernels": M},
  "kernels": [
    {
      "kernel_name": "matmul_kernel",
      "launch_count": 2,
      "task": {"task_type": "AI_CORE", "task_duration_us": .., "block_dim": ..,
               "input_shapes": .., "input_dtypes": .., "output_shapes": .., "output_dtypes": ..,
               "aicore_time_us": .., "aiv_time_us": .., "total_cycles": ..,
               "pipes_us": {"aic_mte1_time_us": .., "aic_cube_time_us": .., ...},
               "est_bytes_in": .., "est_bytes_out": .., "transfers": [..]},
      "deep": {
        "freq_mhz": ..,
        "bandwidth_gb_s": {17 通路},
        "engine_utilization": {cube/vec/mte1/2/3/scalar/fixpipe},
        "compute": {fops/ratio/instr},
        "conflict": {bank/bankgroup/total/resc/mte},
        "l2_hit_rate": ..,
        "roofline": {"bottleneck_type": .., "achieved_memory_bw_gb_s": ..,
                     "memory_utilization": .., "achieved_compute_tflops": ..,
                     "compute_utilization": .., "arithmetic_intensity": ..}
      },
      "filled_by": "msprof op"
    }
  ],
  "api_overhead": [..],
  "multi_kernel": [..],
  "notes": [..]
}
```

---

## 5. 落地文件

| 文件 | 改动 |
|---|---|
| `pipeline_parse_task.py` | +`kernel_slots[]`（distinct kernel 去重，merge 目标） |
| `pipeline_parse_board.py` | `find_opprof` 支持直接传 OPPROF 目录 |
| `integrate.py` | 新签名 `integrate <task.json> <out.json> [board_*.json...]`，按 kernel 名合并 deep + per-kernel roofline |
| `run_optimize.sh` | msprof op 改为逐 kernel 循环 → 每 kernel 一个 board → integrate 传全部 board |
| `check_fields.py` | 最终校验对每个 board 跑一遍 |

---

## 6. 每轮 Planner/Coder 读来源（核对用）

> 调度器每轮从**相应位置读最新产物**注入 agent；agent 不自己到处找文件。
> **★核心: current_kernel 链** — round1 读源文件, 后续轮读上一轮成功输出, 永不改源文件。

| 谁 | 读什么 | 从哪读 | 最新性 |
|---|---|---|---|
| Planner | 当前 tier 字段段 | `roundN/07_tier{N}_fields/` | run_optimize 每轮重新解析 |
| Planner | 融合分析（Tier2） | `roundN/08_fusion/fusion_analysis.json` | Tier2 每轮重新 nga run |
| Planner | **当前 kernel 代码** | `round1: input/<op>/kernel_op.py`；`roundN: round(N-1)/kernel_op.py`（prompt 给绝对路径） | **上一轮成功输出** |
| Planner | 策略文档 | `docx/playbook_tier{N}.md` + `CODING_GUIDE.md` | 静态 |
| Planner | config 常量 | 当前 kernel 的 ① config 区 | 最新 |
| Planner | 历史 | `outputs/<op>/optimization_trajectory.json` | 每轮更新 |
| Coder | plan 的 changes[] | `roundN/plan.md`（old_code→new_code） | 本轮 planner 产出 |
| Coder | **当前 kernel** | `round1: input/<op>/kernel_op.py`；`roundN: round(N-1)/kernel_op.py` | 上一轮成功输出 |
| Coder | 教程/纠错 | `docx/CODING_GUIDE.md` + playbook_tier | 静态 |

**写目标**: Coder 输出 → `roundN/kernel_op.py`（每轮目录），**永不写回 `input/<op>/kernel_op.py`**。

**保留/回退**: 验证成功 → `current_kernel = roundN/kernel_op.py`（提交, 记入 trajectory）；
验证失败(重试≤3次后) → **不提交**, current_kernel 沿用上一个成功的, 本轮 kernel 不进入下一轮链。

## 7. 改码确定性规则（dumb agent 保障）

1. **planner 必须输出 `changes[]`**：每项 `{old_code, new_code, reason, section}`，old_code 必须与 kernel_op.py 某段**逐字符匹配**。
2. **coder 优先精确替换**：找到 old_code → 原样替换 new_code；**找不到就报告，绝不猜测乱改**。
3. coder 只走 LLM 的情形：有 `previous_error`（验证报错）需修报错时。
4. 替换后校验：python 语法 + 防截断（所有 def 仍在）+ no-op 检查。
