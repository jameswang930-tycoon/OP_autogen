"""Fix Tier 4 playbook for our environment"""
path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\docx\playbook_tier4_memory.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# 1. Add environment banner after intro
old = "本手册严格适配环境约束："
new = """> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md
> - WSL2 + Python3.9 + triton3.4.0，仅用 @triton.jit
> - 所有修改通过 ast_to_ttir() → ttir_to_hivm.py → bisheng 三层编译
> - num_warps/num_stages 由 GPUTarget 设置，Coder 不修改
> - 标量索引（params[0]）在 TTIR→HIVM 中会被跳过（SCALAR op），不影响 msprof

本手册严格适配环境约束："""
if old in content:
    content = content.replace(old, new)
    fixes += 1

# 2. Fix tl.trans → tl.transpose (bug in 豆包 code)
if "tl.trans(" in content:
    content = content.replace("tl.trans(", "tl.transpose(")
    fixes += 1
    print("Fixed tl.trans -> tl.transpose")

# 3. Add bottleneck_diagnoser note
old2 = "所有优化完全由 HIVM ops 诊断数据驱动，满足对应阈值即强制执行对应优化。"
new2 = """所有优化完全由 HIVM ops 诊断数据驱动，满足对应阈值即强制执行对应优化。

> **适配我们的 bottleneck_diagnoser**：
> Agent 从 `merged_report.json` 获取以下指标判断内存优化机会：
> - `per_op_statistics[].op_type` — 统计 `gm_to_ub` (load) 和 `ub_to_gm` (store) 数量
> - 同 `src`/`size_kb`/`memory_region` 的 load op ≥ 2 → 触发冗余 Load 消除
> - `per_op_statistics[].size_kb` — 连续小 size (< 1KB) 的 load/store ≥ 4 → 触发传输合并
> - `per_op_statistics[].pipeline_channel` — 确认 MTE2/MTE3 管线占用"""
if old2 in content:
    content = content.replace(old2, new2)
    fixes += 1

# 4. Fix "最优判定" bw_util > 90% — add mock env note
old3 = "4. 带宽利用率稳定在 **90%~95%** 区间，无明显访存瓶颈"
new3 = "4. 带宽利用率稳定在 **90%~95%** 区间（mock 环境下用 SATURATION_PARAMS 估算值代替），无明显访存瓶颈"
if old3 in content:
    content = content.replace(old3, new3)
    fixes += 1

# 5. Add note to Section 3.2 After code — scalar indexing is OK (skipped by converter)
old4 = "    params = tl.load(params_ptr + tl.arange(0, 3))\n    bias = params[0]\n    scale = params[1]\n    shift = params[2]"
new4 = """    # 一次性加载3个连续参数，寄存器内切片（标量索引会被TTIR→HIVM跳过，不影响msprof）
    params = tl.load(params_ptr + tl.arange(0, 3))
    bias = params[0]
    scale = params[1]
    shift = params[2]"""
if old4 in content:
    content = content.replace(old4, new4)
    fixes += 1

# 6. Add verifier note
old5 = "### 典型场景整体收益"
new5 = """> **在我们的 pipeline 中验证 Tier 4 效果**：
> 1. `gm_to_ub` (load) op 数量应减少（冗余消除/传输合并）
> 2. `size_kb` 应增大（小传输合并为大传输）
> 3. `bw_utilization` 应提升
> 4. 若 HIVM ops 数量不变 → 优化无效 → REVERT

### 典型场景整体收益"""
if old5 in content:
    content = content.replace(old5, new5)
    fixes += 1

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Total fixes: {fixes}")
