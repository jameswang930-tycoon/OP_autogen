"""Fix Tier 5 playbook for our environment"""
path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\docx\playbook_tier5_compute.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# 1. Fix contradicting namespace guidance: we use tl.math.rsqrt (verified working)
# Line 96 says tl.rsqrt is standard — WRONG for our env. Fix it.
old_ns = """> 说明：Triton 3.4.0 标准 API 为 `tl.rsqrt`，`tl.math.rsqrt` 为实验性别名；为保证 TTIR→HIVM 转换兼容性，统一使用顶层 `tl.*` 命名空间。"""
new_ns = """> 说明：在我们的环境（triton 3.4.0 + 自定义 TTIR→HIVM 转换器）中，
> **使用 `tl.math.rsqrt`**（已验证通过编译和转换）。
> 不要用 `tl.rsqrt`——我们的转换器的 ARITH_TO_HIVM 映射以 `math.rsqrt` 为 key。"""
if old_ns in content:
    content = content.replace(old_ns, new_ns)
    fixes += 1

# 2. tl.rsqrt → tl.math.rsqrt in code examples
count = content.count("tl.rsqrt")
content = content.replace("tl.rsqrt", "tl.math.rsqrt")
if count > 0:
    fixes += 1
    print(f"Fixed {count} tl.rsqrt -> tl.math.rsqrt")

# 3. Fix After code for rsqrt (line 91)
old_rsqrt = "inv_std = tl.math.rsqrt(var + eps)"
new_rsqrt = "inv_std = tl.math.rsqrt(var + eps)  # 单条指令替代 1.0/tl.sqrt()，减少50% VecUnit ops"
if old_rsqrt in content:
    content = content.replace(old_rsqrt, new_rsqrt)
    fixes += 1

# 4. Fix Section 3.2 FMA — add warning that tl.fma may not be recognized
old_fma_head = "### 3.2 融合乘加（FMA）指令化\n**典型场景**：线性层偏置加法、仿射变换"
new_fma_head = """### 3.2 融合乘加（FMA）指令化（谨慎使用）
**典型场景**：线性层偏置加法、仿射变换

> **⚠️ 注意**：`tl.fma` 在我们的 TTIR→HIVM 转换器中**可能不被识别**。
> `ARITH_TO_HIVM` 映射表里没有 `fma` 条目，会生成未映射的 TTIR op。
> **推荐方案**：直接写 `x * w + b`，bisheng 编译器会自动融合为 FMA 指令。
> 仅当 HIVM 明确显示两条独立 VecUnit op（mul + add）且未自动融合时，才尝试显式 `tl.fma`。"""
if old_fma_head in content:
    content = content.replace(old_fma_head, new_fma_head)
    fixes += 1

# 5. Fix weird FMA example (line 167): tl.fma(x*scale+bias, alpha, 0.0) — makes no sense
old_fma_bad = "out = tl.erf(tl.fma(x * scale + bias, alpha, 0.0))"
new_fma_good = "out = tl.erf((x * scale + bias) * alpha)  # 编译器自动融合 mul+add 为 FMA"
if old_fma_bad in content:
    content = content.replace(old_fma_bad, new_fma_good)
    fixes += 1

# 6. Fix error #2 (line 186) — namespace guidance contradicts our env
old_err2 = """### 2. API 命名空间错误导致编译失败
- **错误现象**：`ast_to_ttir` 阶段报错，提示 `tl.math` 模块不存在或函数未定义。
- **触发原因**：误用 `tl.math.rsqrt` 等子命名空间 API，Triton 3.4.0 标准数学函数统一位于 `tl` 顶层命名空间，`tl.math` 为实验性别名，自定义 TTIR→HIVM 转换器可能不支持。
- **修复方案**：统一使用顶层标准 API，如 `tl.rsqrt`、`tl.fma`、`tl.exp`、`tl.log`，避免使用 `tl.math.*` 子命名空间，保证全链路编译兼容。"""
new_err2 = """### 2. API 命名空间错误导致编译失败
- **错误现象**：`ast_to_ttir` 阶段报错，提示函数未定义。
- **触发原因**：Triton 3.4.0 中 `tl.rsqrt` 和 `tl.math.rsqrt` 均可使用，但我们的 `ttir_to_hivm.py` 的 `ARITH_TO_HIVM` 映射以 `math.rsqrt` 为 key。如果用 `tl.rsqrt` 生成的 TTIR op 名与映射不匹配，会导致 HIVM 转换丢失该 op。
- **修复方案**：在我们的环境中统一使用 `tl.math.rsqrt`、`tl.math.exp`、`tl.math.sqrt`（已验证通过全链路）。"""
if old_err2 in content:
    content = content.replace(old_err2, new_err2)
    fixes += 1

# 7. Add environment banner
old_env = "本手册严格适配环境约束："
new_env = """> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md
> - WSL2 + Python3.9 + triton3.4.0，仅用 @triton.jit
> - num_warps/num_stages 由 GPUTarget 设置，Coder 不修改
> - **数学函数用 `tl.math.*` 命名空间**（我们的 ttir_to_hivm.py 以 `math.xxx` 为 key）
> - `tl.fma` 谨慎使用——直接写 `x*w+b` 让 bisheng 自动融合更安全

本手册严格适配环境约束："""
if old_env in content:
    content = content.replace(old_env, new_env)
    fixes += 1

# 8. Add bottleneck_diagnoser note
old_diag = "所有优化动作由 HIVM ops 诊断数据驱动，满足对应阈值即执行对应优化。"
new_diag = """所有优化动作由 HIVM ops 诊断数据驱动，满足对应阈值即执行对应优化。

> **适配我们的 bottleneck_diagnoser**：
> Agent 从 `merged_report.json` 获取以下指标判断计算优化机会：
> - `per_op_statistics[].op_type` — 统计 VECTOR 管线 op 数量（vadd/vmul/vdiv）
> - `per_op_statistics[].pipeline_channel` — 确认是 VECTOR 管线
> - `execution_summary.engine_usage_pct` — VecUnit 占比 > 60% 触发计算优化
> - 连续的 `vmul` + `vadd` RAW 链（无中间 store）→ FMA 候选"""
if old_diag in content:
    content = content.replace(old_diag, new_diag)
    fixes += 1

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Total fixes: {fixes}")
