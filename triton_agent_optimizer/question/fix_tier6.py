"""Fix Tier 6 playbook for our environment"""
path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\docx\playbook_tier6_architecture.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# 1. Fix namespace guidance: we USE tl.math.rsqrt (豆包 says don't use tl.math)
old_ns = "禁止使用实验性 API、`tl.math` 子命名空间、自定义聚合类型、高阶函数，避免触发未识别 op 降级。"
new_ns = """禁止使用实验性 API、自定义聚合类型、高阶函数，避免触发未识别 op 降级。
> **注意**：我们的 `ttir_to_hivm.py` 的 `ARITH_TO_HIVM` 映射以 `math.xxx` 为 key，
> 所以 `tl.math.rsqrt`、`tl.math.exp` 等 **必须使用 `tl.math.*` 命名空间**（已验证）。
> 仅 `tl.fma` 例外——我们的转换器没有 fma 映射，建议写 `x*w+b` 让 bisheng 自动融合。"""
if old_ns in content:
    content = content.replace(old_ns, new_ns)
    fixes += 1

# 2. tl.rsqrt → tl.math.rsqrt (in standard API list and elsewhere)
count = content.count("tl.rsqrt")
content = content.replace("tl.rsqrt", "tl.math.rsqrt")
if count > 0:
    fixes += 1
    print(f"Fixed {count} tl.rsqrt -> tl.math.rsqrt")

# 3. tl.fma warning
old_fma = "乘加统一用 `tl.fma`"
new_fma = "乘加写 `x*w + b` 让编译器自动融合（我们的转换器没有 fma 映射）"
if old_fma in content:
    content = content.replace(old_fma, new_fma)
    fixes += 1

# 4. tl.trans → tl.transpose (in code examples)
count2 = content.count("tl.trans(")
content = content.replace("tl.trans(", "tl.transpose(")
if count2 > 0:
    fixes += 1
    print(f"Fixed {count2} tl.trans -> tl.transpose")

# 5. Add environment banner after intro
old = "本手册严格适配自研全链路工具与环境约束："
new = """> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md
> - WSL2 + Python3.9 + triton3.4.0，仅用 @triton.jit
> - 全链路：ast_to_ttir() → ttir_to_hivm.py → hivm_to_ascendc.py → bisheng → msprof
> - `tl.math.*` 命名空间是我们的转换器支持的（以 `math.xxx` 为 key）
> - `tl.fma` 未映射——写 `x*w+b` 让 bisheng 自动融合
> - num_warps/num_stages 由 GPUTarget 管理，Coder 不修改

本手册严格适配自研全链路工具与环境约束："""
if old in content:
    content = content.replace(old, new)
    fixes += 1

# 6. Add our converter-specific notes to section 1.1
old_ttir = "目标是让 TTIR 算子与 HIVM 算子实现 1:1 原生映射"
new_ttir = """目标是让 TTIR 算子与 HIVM 算子实现 1:1 原生映射

> **我们的转换器已知限制**：
> - `ARITH_TO_HIVM` 映射了 12 个 op（vadd/vsub/vmul/vdiv/vexp/vabs/vmax/vsqrt/vrelu/vtanh/load/store）
> - **未映射的 op 会被跳过**（SCALAR op），不影响 msprof 但不产生 HIVM 事件
> - `tl.dot` → `tt.dot` → 我们的转换器映射为 `matmul`（CubeUnit），但 AscendC 代码生成跳过 CubeUnit
> - 标量索引（`tensor[0]`）被跳过——这是预期行为，不是 bug"""
if old_ttir in content:
    content = content.replace(old_ttir, new_ttir)
    fixes += 1

# 7. Fix line 98 table — MTE unit stats
old_table = "| MTE | 全局内存读写、数据搬运 | `tl.load`、`tl.store` | 内存传输周期统计 |"
new_table = "| MTE | 全局内存读写、数据搬运 | `tl.load`、`tl.store`（映射为 HIVM `gm_to_ub`/`ub_to_gm`）| 内存传输周期统计 |"
if old_table in content:
    content = content.replace(old_table, new_table)
    fixes += 1

# 8. Add verifier notes
old_verify = "### 4.3 事件完整性校验流程"
new_verify = """> **在我们的 pipeline 中验证 Tier 6 效果**：
> 1. TTIR chars 减少但 HIVM ops 不增加 = 代码精简有效
> 2. AscendC Build OK + msprof 有 event = 工具链兼容
> 3. HIVM ops 数量不变但 msprof timing 变好 = 生成的 AscendC 代码质量提升
> 4. TTIR chars 增大但 HIVM ops 不变 = 代码膨胀 → 需要精简

### 4.3 事件完整性校验流程"""
if old_verify in content:
    content = content.replace(old_verify, new_verify)
    fixes += 1

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Total fixes: {fixes}")
