# Pipeline: 三源解析 → 统一 JSON → 合并（服务器真数据流程）

> 2026-08-03 新建。目标：在 910B3 服务器上，把三源真实产物解析成**统一格式**的 3 个 JSON，再合并成 1 个 merged.json，供后续 dsl_merger/诊断/LLM 使用。
> 命令与环境说明见 `input/matmul/test_matmul.py` 头注释（三源采集命令）与 `PIPELINE_FIELDS.md`（字段来源/用途）。

---

## 脚本清单

| 脚本 | 输入 | 输出 | 内容 |
|---|---|---|---|
| `pipeline_parse_hivm.py` | `hivm_try.txt`（真实 HIVM） | `hivm.json` | 结构字段：op_type/engine/dst/src/src2/size_kb/region/deps/attrs，含 sync op；时序=None |
| `pipeline_parse_sim.py` | `sim_prof` 目录（OPPROF_*/simulator/） | `sim.json` | 指令级时序：按(指令名,pipe)跨核聚合，per-call 耗时/cycles/call_count/data_size；summary 来自 trace.json |
| `pipeline_parse_board.py` | `board_prof` 目录（真机 msprof op） | `board.json` | op 级聚合：Task Duration→total_ns、Block Dim→num_cores、PipeUtilization→per-engine 伪 op、ArithmeticUtilization→cube/vec 占比+FLOPs、Memory*.csv→带宽 |
| `pipeline_merge.py` | 上面 3 个 JSON | `merged.json` | canonical op = hivm（含 sync），时序按引擎/pipe + sync 指令名从 sim 填 per-call，summary 优先真机 |
| `run_all.sh` | 三个数据路径 | 输出目录 4 个 JSON | 一键跑 1/4~4/4 |

公共 schema 见 `pipeline_schema.py`（三源同形：每 op 同一组字段，该源没有的 = None）。

---

## 服务器用法（一条命令）

```bash
# 1. 环境 (每次终端)
conda activate triton-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 采集三源 (命令细节见 test_matmul.py 头注释):
#    HIVM:   流程 D → hivm_try.txt (当前目录)
#    SIM:    msprof op simulator --output=./sim_prof
#    BOARD:  msprof op --output=./board_prof

# 3. 一键流水线 (在 triton_agent_optimizer/analyzers/ 下)
bash run_all.sh ../input/matmul/hivm_try.txt ./sim_prof ./board_prof ../outputs/matmul_e2e
#   或指定 python (如服务器默认 python3)
#   PYTHON=python3 bash run_all.sh ...

# 4. 查看
python3 -m json.tool ../outputs/matmul_e2e/merged.json
```

---

## 三源对齐策略（重要，务必理解）

**三种来源的操作粒度不同，不能天然 1:1：**

| 源 | 粒度 | 数量(示例) | 提供什么 |
|---|---|---|---|
| HIVM | 语义 op | ~7~32 | 结构/依赖/尺寸 |
| simulator | 指令 | ~35 组×24核 | 真实单指令耗时/cycles/搬运块 |
| 真机 board | kernel 级 | 1 行 + 引擎伪 op | 端到端延迟/引擎占比/带宽 |

**merge 如何对齐**：
1. canonical per-op = **hivm 语义 op**（含 sync op，op_id 连续）
2. 每个 hivm op 从 sim 取时序：非 sync 按 `op_type → pipe`（gm_to_ub→MTE2, mmadL1→CUBE, v*→VECTOR, ub_to_gm→MTE3）；sync 按指令名（SET_FLAG/WAIT_FLAG/BAR）
3. sim 每 (指令,pipe) 组取 **per-call 平均耗时**（1 个语义 load ≈ 1 次 MTE2 调用），不消耗组 → 循环里多个语义 op 可映射同一指令组
4. summary/engine_utilization 优先**真机 board**（真实端到端/占比）

**已知限制（mock/样例数据下可见，真实数据会缓解）**：
- 样例 csv 缺 MTE3 → ub_to_gm 时序 None；CUBE 只有 SET_FLAG → mmadL1 错配 SET_FLAG
- `critical_path` 的 path 索引是 trace 事件序号，非 hivm op_id（语义待定）

---

## 字段完整性（真 vs 待补）

| 字段 | 状态 |
|---|---|
| 结构（op_type/dst/src/size_kb/region/deps/attrs） | ✅ 真实 HIVM |
| 时序（duration/cycles/pipe/call_count/data_size） | ✅ 真实 simulator（per-call） |
| 端到端 total_ns / num_cores | ✅ 真机 Task Duration / Block Dim |
| engine_utilization（cube/vec 占比） | ✅ 真机 ArithmeticUtilization |
| 带宽 peak_bw/regime | ❌ 需真机 Memory.csv 校准 + size 扫描（见 PIPELINE_FIELDS.md） |
