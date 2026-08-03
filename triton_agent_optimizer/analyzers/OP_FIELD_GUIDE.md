# 每 Op 字段速查 — matmul 真实数据判定指南

> 2026-08-03。回答两个问题：① 每个 op 类型**应该有哪些字段、内容是什么**；② 怎么判断解析结果**正确**。
> 背景：三源（HIVM 结构 + simulator 时序 + 真机端到端）合并后，**不同 op 类型字段天然不同**——这不是 bug，是 op 语义决定的。

---

## 一、4 个 JSON 是什么

| 文件 | 内容 | 有无时序 |
|---|---|---|
| `hivm.json` | 语义 op 的结构：op_type/engine/dst/src/size_kb/region/依赖/attrs | ❌ 无（时序=None） |
| `sim.json` | 机器指令：per-call 耗时/cycles/call_count/搬运块大小 | ✅ 有 |
| `board.json` | kernel 级：total_ns/cores/引擎占比/带宽 | ✅ 有（聚合） |
| `merged.json` | per-op = HIVM 语义 op，结构真实 + 时序贴 sim + 端到端用真机 | ✅ 合并 |

**合并后每个 op 的字段**：op_id/op_name/op_type/engine/instruction/dst/src/src2/size_kb/memory_region/dtype/attrs/dependencies/duration_ns/start_ns/end_ns/time_ratio/cycles/pipeline_channel/core_id/data_size_bytes/effective_bw/peak_bw/bw_util/regime/call_count/total_duration_ns/sim_instr

---

## 二、字段含义（通用）

| 字段 | 含义 | 从哪来 |
|---|---|---|
| `op_type` | 语义 op 名（gm_to_ub/mmadL1/ub_to_gm/set_flag...） | HIVM |
| `engine` | 执行单元（GM→UB/UB→GM/CubeUnit/VecUnit/Sync/FixPipe） | HIVM OP_TO_ENGINE |
| `dst` / `src` / `src2` | 目标/源 buffer（%l1_a, %A...） | HIVM 操作数 |
| `size_kb` | **搬运/计算的数据量**（从 op 类型 tensor<MxN> 算） | HIVM op 类型 |
| `memory_region` | 数据所在（L1/L0C/UB/GM） | HIVM address_space |
| `dtype` | 数据类型（f16/f32...） | HIVM |
| `attrs` | tiling 配置 {lhs_m, rhs_n, l0b_k} 等 | HIVM |
| `dependencies` | RAW/WAR/WAW 依赖边 | HIVM |
| `duration_ns` | 该 op 单次耗时（per-call） | simulator 对齐 |
| `cycles` | 周期数 | simulator |
| `pipeline_channel` | 硬件 pipe（MTE2/MTE3/CUBE/FIXP...） | simulator |
| `call_count` | 该指令调用次数（循环展开次数） | simulator |
| `data_size_bytes` | 搬运数据块字节数（detail 里） | simulator |
| `total_ns` / `num_cores` | 端到端/核数（kernel 级） | 真机 board |

---

## 三、matmul 各 op 类型应有字段（判定正确用）

### 1. `gm_to_ub`（hivm.hir.load，GM→L1/UB 搬运）
| 字段 | 应有内容 |
|---|---|
| engine | `GM→UB` |
| dst / src | dst=L1/UB buffer，src=GM 指针（%A/%B） |
| size_kb | **>0**：A/B tile 字节/1024（如 64×32×f32=8KB） |
| memory_region | `L1`（cbuf）或 `UB` |
| dtype | f16 或 f32 |
| pipeline_channel | `MTE2` |
| duration_ns / cycles | **有值**（MTE2 指令 per-call） |
| dependencies | 可能空（首次 load）或 RAW |

**判定正确**：size_kb>0 + region=L1/UB + pipe=MTE2 + 有时序。

### 2. `mmadL1`（Cube 矩阵乘 = tl.dot）
| 字段 | 应有内容 |
|---|---|
| engine | `CubeUnit` |
| dst / src / src2 | dst=L0C，src/src2=输入 tile |
| size_kb | **>0**：C tile（如 64×64×f32=16KB） |
| memory_region | `L0C`（cc） |
| attrs | `{lhs_m, rhs_n, l0b_k}`（tile 配置） |
| pipeline_channel | `CUBE` |
| duration_ns / cycles | **有值**（CUBE 指令） |

**判定正确**：engine=CubeUnit + region=L0C + size>0 + attrs 有 + pipe=CUBE + 时序有。

### 3. `fixpipe`（L0C→UB，量化/激活融合；fp32 matmul 常见）
| 字段 | 应有内容 |
|---|---|
| engine | `FixPipe` |
| src=L0C，dst=UB | — |
| size_kb | >0 |
| attrs | `{pre_quant: F322F16}` 等 |
| pipeline_channel | `FIXP` |

### 4. `ub_to_gm`（hivm.hir.store，UB/L0C→GM 写回）
| 字段 | 应有内容 |
|---|---|
| engine | `UB→GM` |
| dst=GM 输出（%C），src=UB/L0C | — |
| size_kb | **>0**：C tile |
| memory_region | `UB` 或 `L0C` |
| pipeline_channel | `MTE3` |
| duration_ns / cycles | **有值** |

### 5. `vadd/vmul/...`（若有逐元素，如 epilogue）
| 字段 | 应有内容 |
|---|---|
| engine | `VecUnit` |
| size_kb | >0 |
| pipeline_channel | `VECTOR` |

### 6. `set_flag` / `wait_flag` / `pipe_barrier`（同步）
| 字段 | 应有内容 |
|---|---|
| engine | `Sync` |
| **size_kb** | **None/0 是正确的**（同步不搬数据！） |
| **dst/src** | **空是正确**（操作 pipe/event，不操作 buffer） |
| pipeline_channel | 同步的 pipe（SET_FLAG→CUBE/MTE2、BAR→ALL 等） |
| duration_ns / cycles | **有值**（SET_FLAG/WAIT_FLAG/BAR 指令） |
| sim_instr | SET_FLAG / WAIT_FLAG / BAR |

**判定正确**：engine=Sync + size=None + 有时序 + sim_instr=SET_FLAG/WAIT_FLAG/BAR。

---

## 四、为什么"有的有 size 有的有 duration 有的都没有"——这是对的

| 情况 | 含义 | 是否正常 |
|---|---|---|
| size_kb>0 + duration 有 | 数据 op 完全对齐 | ✅ |
| size_kb>0 + duration=None | 结构在，但 sim 没对齐到对应 pipe（MTE3/等缺指令） | ⚠️ 待对齐 |
| size_kb=None + duration 有 | **同步 op**（天然无 size） | ✅ 正确 |
| size_kb=None + duration=None | 同步 op 未对齐 | ⚠️ |
| dst/src 空 | 同步 op 天然无 buffer | ✅ |

**关键判据**：
- **数据 op（load/store/matmul/fixpipe）**：size_kb 必须 >0；若 None = 尺寸没解析到（看 op 类型签名）或动态维 `?`。
- **同步 op（set_flag/wait_flag/pipe_barrier）**：size_kb=None 是对的；必须有 duration + sim_instr。
- **所有 op**：duration=None 说明该 pipe 在 sim 里没匹配到指令 → 查 sim.json 里有没有对应 pipe 的指令。

---

## 五、对着你 64³ matmul 的预期

（DTYPE=f32，BLOCK_M/N/K=64/64/32，K 循环 2 次）

| op 类型 | 预期 size_kb | 预期 region | 预期 pipe | 预期 engine |
|---|---|---|---|---|
| gm_to_ub（load A tile） | 64×32×4/1024 = 8KB | L1 (cbuf) | MTE2 | GM→UB |
| gm_to_ub（load B tile） | 32×64×4/1024 = 8KB | L1 (cbuf) | MTE2 | GM→UB |
| mmadL1 | 64×64×4/1024 = 16KB | L0C (cc) | CUBE | CubeUnit |
| ub_to_gm（store C） | 64×64×4/1024 = 16KB | UB/L0C | MTE3 | UB→GM |
| set_flag/wait_flag/pipe_barrier | **None**（正确） | — | 同步 pipe | Sync |

> 注：真实 IR 可能因 tiling/fusion 有更多 op（fixpipe 等），size 以 op 类型签名为准；`?`动态维 → size=None。

---

## 六、操作类检查（服务器上对 merged.json 快速自查）

```bash
# 所有 op 一览 (看 engine/pipe/size/duration 是否有)
python3 -c "
import json; m=json.load(open('input/matmul/e2e_run/05_merged/merged.json'))
for o in m['per_op_statistics']:
    print(f\"op{o['op_id']:2d} {str(o['op_type'])[:12]:12s} eng={str(o['engine'])[:9]:9s} pipe={str(o.get('pipeline_channel'))[:7]:7s} size={o.get('size_kb')} dur={o.get('duration_ns')} sim={str(o.get('sim_instr'))[:16]}\")
"
# 判定:
#   load/store/matmul  size 必须 >0, 有 dur
#   set_flag/wait_flag  size=None 正确, 有 dur + sim=SET_FLAG/WAIT_FLAG/BAR
#   dur=None 的 → 看 sim.json 该 pipe 有没有指令
```
