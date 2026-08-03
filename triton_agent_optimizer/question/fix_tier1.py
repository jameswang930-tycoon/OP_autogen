"""Fix Tier 1 playbook for our specific environment"""
path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\docx\playbook_tier1_algorithm.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# 1. Add environment banner after the intro paragraph
old = "所有策略均基于 HIVM ops 诊断数据（op 类型、数量、size_kb、管线通道、RAW/WAR 依赖链、bw_utilization）驱动，严格适配 Triton 3.4.0 语法"
new = """所有策略均基于 HIVM ops 诊断数据（op 类型、数量、size_kb、管线通道、RAW/WAR 依赖链、bw_utilization）驱动。

> **环境约束（Coder Agent 必读）**：
> - WSL2 Ubuntu 24.04 + Python 3.9 + triton 3.4.0
> - 仅用 `@triton.jit`，禁用 `@triton.autotune`（mock 环境不支持）
> - `num_warps`/`num_stages` **不在** `@triton.jit` 参数中
> - `[:,None]`, `tl.zeros(2D)`, `tl.dot` 在 triton 3.4.0 中全部可用
> - 所有 BLOCK_SIZE/DIM 参数必须标记 `tl.constexpr`
> - runtime 参数不能与 `program_id` 做乘法（pointer type 报错）

严格适配 Triton 3.4.0 语法"""
if old in content:
    content = content.replace(old, new)
    fixes += 1

# 2. Fix "已最优"判定 — add bottleneck_diagnoser compatibility note
old2 = "判定规则：三个条件为与逻辑，缺一不可。若任意一条不满足，必须返回对应算法优化策略，禁止直接进入下一层优化。"
new2 = """判定规则：三个条件为与逻辑，缺一不可。若任意一条不满足，必须返回对应算法优化策略，禁止直接进入下一层优化。

> **适配我们系统的 bottleneck_diagnoser**：
> 我们的诊断器不区分算子类型，Agent 按以下优先级读取 `merged_report.json`：
> 1. `execution_summary.num_ops` — op 总数是否超标（基础阈值 > 3）
> 2. `dependencies_summary.raw_chains` — RAW 依赖链长度（阈值 > 1）
> 3. `per_op_statistics[].bw_utilization` — 仅当 msprof 是本 kernel 真实 trace 时参考；
>    否则用 `SATURATION_PARAMS` 公式估算值（标记为 ESTIMATED）
> 4. `per_op_statistics[].op_type` — 统计各类型 op 数量判断是否需要算法替换"""
if old2 in content:
    content = content.replace(old2, new2)
    fixes += 1

# 3. Fix LayerNorm core note
old3 = "核心改动：用 sum(x) + sum(x*x) 一次遍历得到均值和方差，替代 \"减均值再算方差\" 的两次遍历，减少寄存器压力与计算步骤。"
new3 = """核心改动：用 sum(x) + sum(x*x) 一次遍历得到均值和方差，替代 "减均值再算方差" 的两次遍历。

> **适配我们的 TTIR->HIVM 转换器**：
> `ttir_to_hivm.py` 将每个 `tl.load` 映射为 1 个 HIVM `gm_to_ub` op。
> 优化目标：3+ 次 load 降到 2 次（输入 + 权重），HIVM ops 从 7+ 降到 5 以内。
> `tt.load` + `arith` 组合被转换为 `gm_to_ub` + `vadd/vmul` 序列。
> Agent 可通过 `per_op_statistics` 中的 op 数量变化验证优化效果。"""
if old3 in content:
    content = content.replace(old3, new3)
    fixes += 1

# 4. Fix triton.rsqrt -> triton.math.rsqrt (in code blocks, used as tl.math.rsqrt or tl.rsqrt)
# The code examples use tl.rsqrt in a few places - fix to tl.math.rsqrt
import re
count = content.count("tl.rsqrt")
content = content.replace("tl.rsqrt", "tl.math.rsqrt")
if count > 0:
    fixes += 1
    print(f"Fixed {count} tl.rsqrt -> tl.math.rsqrt")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Total fixes applied: {fixes}")
