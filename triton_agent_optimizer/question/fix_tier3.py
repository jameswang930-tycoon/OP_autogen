"""Fix Tier 3 playbook for our environment"""
path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\docx\playbook_tier3_tiling.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# 1. tl.rsqrt → tl.math.rsqrt
count = content.count("tl.rsqrt")
content = content.replace("tl.rsqrt", "tl.math.rsqrt")
if count > 0:
    fixes += 1
    print(f"Fixed {count} tl.rsqrt")

# 2. Replace WRONG num_warps/stages guidance (Section 3.1-3.3)
# Our environment uses GPUTarget options, NOT call-time params
old_section = """### 3.1 官方正确用法说明
根据 Triton 3.4.0 官方规范，`num_warps`、`num_stages` **不属于 `@triton.jit` 装饰器入参**，而是内核实例化时的编译配置参数。本环境禁用 `@triton.autotune`，采用「调用时显式传递编译参数」的方式，参数与内核入参一同在调用时传入，由 Triton 前端自动识别为编译配置。

**正确调用格式**
```python
# 编译参数在调用时传入，不写入@triton.jit装饰器
kernel_func[grid](
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE=256,    # tl.constexpr 内核参数
    num_warps=4,       # 编译配置：并行warp数量
    num_stages=2       # 编译配置：软件流水线级数
)
```"""

new_section = """### 3.1 官方正确用法说明（我们的环境）

> **⚠️ 关键：我们的环境不通过调用时传参！**
> `num_warps`、`num_stages` 在我们的 pipeline 中由 `GPUTarget` 编译选项统一设置：
> ```python
> triton_compile(src, target=GPUTarget("cuda",90,32),
>                options={"num_warps": 4, "num_stages": 1, "debug": False})
> ```
> **Coder Agent 不要修改 num_warps/num_stages！** 也不要在 kernel 调用代码中传这些参数。
> 本章保留 num_warps/num_stages 调优策略作为 Planner 的参考知识，
> 但 Coder 只改 BLOCK_SIZE 参数值，不改调用代码。

根据 Triton 3.4.0 官方规范，`num_warps`、`num_stages` **不属于 `@triton.jit` 装饰器入参**，它们在我们的 pipeline 中通过编译配置传递。"""

if old_section in content:
    content = content.replace(old_section, new_section)
    fixes += 1

# 3. Remove the After code for num_warps/stages (Section 3.3 code example)
# The After code shows kernel call with num_warps=8,num_stages=4 - remove it
# since Coder shouldn't do this
old_code = """**After（显式配置流水线与并行度）**
```python
@triton.jit
def matmul_pipe_after(a_ptr, b_ptr, c_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 内核逻辑不变，仅通过调用参数调整编译配置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 调用：显式指定编译配置，匹配分块尺寸
grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
matmul_pipe_after[grid](
    a_ptr, b_ptr, c_ptr, M, N, K,
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    num_warps=8,    # 匹配128x128分块，提升并行度
    num_stages=4    # 加深流水线，掩盖访存延迟
)
```"""

new_code = """**After（仅改 BLOCK_SIZE，不改调用代码）**
```python
# 内核代码不变！Coder 只改 BLOCK_* 的值：
# BLOCK_M: 128（默认）→ 尝试 256
# BLOCK_N: 128（默认）→ 尝试 256
# BLOCK_K: 32（默认） → 尝试 64
# 约束：BLOCK_M × BLOCK_N × 4bytes × 2(buffers) ≤ 192KB(UB)
# Coder 不要修改 grid 调用代码和 num_warps/num_stages！
```"""

if old_code in content:
    content = content.replace(old_code, new_code)
    fixes += 1

# 4. Add bottleneck_diagnoser note
old_diag = "所有优化动作完全由 HIVM ops 诊断数据驱动，以下为四大优化维度的触发阈值与对应动作，满足任意一条即执行对应调优。"
new_diag = """所有优化动作完全由 HIVM ops 诊断数据驱动，以下为四大优化维度的触发阈值与对应动作，满足任意一条即执行对应调优。

> **适配我们的环境**：
> - `bw_utilization` 在 mock 环境下可能不准（使用预制 trace 时），优先参考 `SATURATION_PARAMS` 估算值
> - `num_warps`/`num_stages` 由 `GPUTarget` 编译选项设置，Coder 只改 BLOCK_SIZE
> - Agent 从 `per_op_statistics[].bw_utilization` 读取带宽利用率
> - BLOCK_SIZE 约束：`BLOCK_SIZE × 4 × n_buffers ≤ 192KB` (UB 容量)"""
if old_diag in content:
    content = content.replace(old_diag, new_diag)
    fixes += 1

# 5. Fix "调用" code examples that show grid/num_warps — add Coder note
old_call = "# 调用：BLOCK_SIZE=64"
new_call = "# 调用（Coder 不修改这部分！）: BLOCK_SIZE=64"
if old_call in content:
    content = content.replace(old_call, new_call)
    fixes += 1

old_call2 = "# 调用：BLOCK_SIZE=256，分块参数通过constexpr传递"
new_call2 = "# 调用（Coder 不修改这部分！）：BLOCK_SIZE=256"
if old_call2 in content:
    content = content.replace(old_call2, new_call2)
    fixes += 1

# 6. Add our known-bug note about pointer arithmetic
old_ptr = "a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]"
new_ptr = "a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]  # K 必须是 tl.constexpr！否则 pointer type 报错"
if old_ptr in content:
    content = content.replace(old_ptr, new_ptr, 1)  # only first occurrence (MatMul Before)
    fixes += 1

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Total fixes: {fixes}")
