"""Fix Tier 2 playbook for our environment"""
path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\docx\playbook_tier2_fusion.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# 1. Add environment banner after intro
old = "本层优化完全基于 HIVM ops 诊断数据（op 类型、依赖链、访存模式、管线通道）驱动，所有修改必须兼容 Triton 3.4.0 语法、可通过 TTIR→HIVM 转换、可被 bisheng 编译器正确编译。"
new = """本层优化完全基于 HIVM ops 诊断数据（op 类型、依赖链、访存模式、管线通道）驱动。

> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md — WSL2 + Python3.9 + triton3.4.0
> - 仅用 `@triton.jit`，禁用 `@triton.autotune`
> - `num_warps`/`num_stages` 不在 `@triton.jit` 参数中
> - 所有修改必须通过 `ast_to_ttir()` → `ttir_to_hivm.py` → `bisheng` 三层编译

所有修改必须兼容 Triton 3.4.0 语法、可通过 TTIR→HIVM 转换、可被 bisheng 编译器正确编译。"""
if old in content:
    content = content.replace(old, new)
    fixes += 1

# 2. tl.rsqrt → tl.math.rsqrt
count = content.count("tl.rsqrt")
content = content.replace("tl.rsqrt", "tl.math.rsqrt")
if count > 0:
    fixes += 1
    print(f"Fixed {count} tl.rsqrt -> tl.math.rsqrt")

# 3. Add bottleneck_diagnoser note to diagnosis section
old2 = "### 2.2 辅助判定与最优标准"
new2 = """### 2.2 辅助判定与最优标准

> **适配我们的 bottleneck_diagnoser**：
> Agent 从 `merged_report.json` 获取以下指标判断融合机会：
> - `per_op_statistics[].op_type` — 统计 VecUnit op 连续出现次数（≥2 可融合）
> - `per_op_statistics[].pipeline_channel` — 确认同管线（VECTOR）且中间无 MTE3（store）
> - `dependencies_summary.raw_chains` — RAW 链长度 ≥ 3 触发算术链融合
> - `per_op_statistics[].size_kb` — 同 size 同 memory_region 的 load op ≥ 2 触发 Load 合并"""
if old2 in content:
    content = content.replace(old2, new2)
    fixes += 1

# 4. Add verifier note to After examples
old3 = "### 4.1 量化收益表"
new3 = """> **在我们的 pipeline 中验证融合效果**：
> 1. 融合后 `per_op_statistics` 中 VECTOR op 数量应减少
> 2. HIVM `gm_to_ub` op 数量应减少（Load 合并）
> 3. RAW 依赖链长度应缩短
> 4. 若 op 数量不变 = 融合失败（Coder 改了等于没改）→ REVERT

### 4.1 量化收益表"""
if old3 in content:
    content = content.replace(old3, new3)
    fixes += 1

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Total fixes applied: {fixes}")
