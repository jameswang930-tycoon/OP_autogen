# Triton 自动优化系统 Tier 3（编译配置与分块调优）优化策略指南
## 层级定位与前置约束
Tier 3 是全链路优化的最底层参数级优化，**必须在 Tier 1（算法结构）、Tier 2（算子融合）均完成并验证最优后执行**。本层不改变核心算法逻辑与算子融合结构，仅通过调整分块尺寸、编译流水线、访存对齐、数据精度等编译期参数，匹配 Ascend NPU 微架构特性，榨取硬件极限性能。

本手册严格适配以下环境约束：
- 前端：Triton 3.4.0 + Python 3.9，仅使用 `@triton.jit` 装饰器，禁用 `@triton.autotune`
- 中间链路：TTIR MLIR → HIVM MLIR 自定义转换，所有语法必须可被解析
- 后端：CANN 9.0 + bisheng 编译器 + CMake 构建
- 验证：msprof op simulator（CPU 周期精确 NPU 模拟器）
- 核心铁律：`num_warps`、`num_stages` 不属于 `@triton.jit` 装饰器参数，必须通过编译配置传递

---

## 一、核心优化维度与诊断触发规则
所有优化动作完全由 HIVM ops 诊断数据驱动，以下为四大优化维度的触发阈值与对应动作，满足任意一条即执行对应调优。

> **适配我们的环境**：
> - `bw_utilization` 在 mock 环境下可能不准（使用预制 trace 时），优先参考 `SATURATION_PARAMS` 估算值
> - `num_warps`/`num_stages` 由 `GPUTarget` 编译选项设置，Coder 只改 BLOCK_SIZE
> - Agent 从 `per_op_statistics[].bw_utilization` 读取带宽利用率
> - BLOCK_SIZE 约束：`BLOCK_SIZE × 4 × n_buffers ≤ 192KB` (UB 容量)

| 优化维度 | 诊断指标 | 触发阈值 | 优化动作 |
|----------|----------|----------|----------|
| 分块尺寸调优 | 带宽利用率 `bw_utilization` | < 70% 且访存 op 占比 > 60% | 增大分块尺寸，提升访存连续性 |
| 分块尺寸调优 | 单块数据量 `size_kb` | > 128KB 且存在缓存换出标记 | 减小分块尺寸，避免片上缓存溢出 |
| 分块尺寸调优 | 边界掩码开销占比 | > 10% | 增大分块，降低边界处理占比 |
| 软件流水线调优 | RAW 依赖等待占比 | > 15% | 增加 `num_stages`，加深软件流水线 |
| 软件流水线调优 | 计算单元利用率 | < 60% | 调整 `num_warps`，提升并行度 |
| 软件流水线调优 | 寄存器溢出标记 | 存在 | 减少 `num_warps` 或 `num_stages` |
| 访存对齐优化 | 访存 op 对齐标记 | 存在未对齐访问 | 注入对齐提示，调整分块对齐 |
| 数据精度调优 | 带宽利用率 `bw_utilization` | > 90% 且计算 op 占比 < 20% | 存储精度降级为 FP16，减少访存带宽 |

### Tier 3 最优判定标准
同时满足以下所有条件时，判定 Tier 3 已达最优，全链路优化结束：
1. 带宽利用率稳定在 **90%~95%** 区间，无明显带宽浪费或饱和瓶颈
2. 软件流水线气泡占比 < 5%，RAW 依赖等待可忽略
3. 所有全局访存均满足对齐要求，无未对齐访问标记
4. 数据位宽已降至业务精度约束的最小值
5. 单维度参数调整均无法带来正向性能收益

---

## 二、分块参数（BLOCK_*）调优指南
分块尺寸是 Tier 3 的核心调优项，直接决定访存连续性、缓存命中率与并行效率，所有分块参数必须声明为 `tl.constexpr`。

### 2.1 通用分块原则（适配 NPU 架构）
1. **2 的幂次对齐**：所有 `BLOCK_*` 尺寸必须为 2 的幂次（32/64/128/256/512/1024），匹配 NPU 内存控制器位宽，非 2 幂次分块会导致访存效率骤降 30% 以上。
2. **单块数据量约束**：单块总数据量控制在 16KB~128KB 区间，匹配片上缓存容量；超过 128KB 会触发缓存换入换出，性能反向下降。
3. **计算访存比匹配**：计算密集型算子（如 MatMul）采用小分块多迭代，访存密集型算子（如 Norm、Softmax）采用大分块少迭代。

### 2.2 分算子推荐分块范围
| 算子类型 | 分块维度 | 推荐取值范围 | 最优初始值 | 调整方向 |
|----------|----------|--------------|------------|----------|
| 逐元素算子（GELU、Add等） | BLOCK_SIZE | 128 ~ 1024 | 256 | 带宽不足时增大，溢出时减小 |
| 归约类算子（Softmax、RMSNorm） | BLOCK_SIZE | 128 ~ 512 | 256 | 归约维度优先 256，行优先匹配缓存 |
| LayerNorm | BLOCK_SIZE | 128 ~ 512 | 256 | 与特征维度对齐，避免跨行分块 |
| 矩阵乘 MatMul | BLOCK_M / BLOCK_N | 64 ~ 256 | 128 | 计算瓶颈时增大，访存瓶颈时减小 |
| 矩阵乘 MatMul | BLOCK_K | 32 ~ 64 | 32 | 访存瓶颈时增大，精度敏感时减小 |

### 2.3 诊断驱动调优流程
1. **带宽不足场景**（bw_util < 70%）：分块尺寸 ×2，重新编译测试，直到带宽利用率进入 90% 区间或触发缓存溢出。
2. **缓存溢出场景**（size_kb > 128KB）：分块尺寸 ÷2，直到溢出标记消失且性能最优。
3. **边界开销过高**：当非对齐尺寸输入的边界掩码开销占比超过 10% 时，优先增大分块，降低边界处理的相对占比。

### 代码示例：分块参数调整
**Before（初始小分块）**
```python
@triton.jit
def rmsnorm_tune_before(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # 初始分块64，过小导致调度开销高、访存不连续
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    inv_std = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + base + offsets, x * inv_std * w, mask=mask)

# 调用（Coder 不修改这部分！）: BLOCK_SIZE=64
grid = (M,)
rmsnorm_tune_before[grid](x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK_SIZE=64)
```

**After（优化后分块）**
```python
@triton.jit
def rmsnorm_tune_after(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # 分块调整为256，访存连续、调度开销降低
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    inv_std = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + base + offsets, x * inv_std * w, mask=mask)

# 调用（Coder 不修改这部分！）：BLOCK_SIZE=256
grid = (M,)
rmsnorm_tune_after[grid](x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK_SIZE=256)
```

---

## 三、软件流水线与并行配置（num_warps / num_stages）
### 3.1 官方正确用法说明（我们的环境）

> **⚠️ 关键：我们的环境不通过调用时传参！**
> `num_warps`、`num_stages` 在我们的 pipeline 中由 `GPUTarget` 编译选项统一设置：
> ```python
> triton_compile(src, target=GPUTarget("cuda",90,32),
>                options={"num_warps": 4, "num_stages": 1, "debug": False})
> ```
> **Coder Agent 不要修改 num_warps/num_stages！** 也不要在 kernel 调用代码中传这些参数。
> 本章保留 num_warps/num_stages 调优策略作为 Planner 的参考知识，
> 但 Coder 只改 BLOCK_SIZE 参数值，不改调用代码。

根据 Triton 3.4.0 官方规范，`num_warps`、`num_stages` **不属于 `@triton.jit` 装饰器入参**，它们在我们的 pipeline 中通过编译配置传递。

### 3.2 num_warps 调优规则
`num_warps` 控制单块内的并行计算单元数量，NPU 场景下映射为向量计算单元的并行度，默认值为 4。
- **匹配规则**：通常每 warp 对应处理 32 个元素，`BLOCK_SIZE / 32 = num_warps` 为最优匹配；例如 BLOCK_SIZE=128 匹配 num_warps=4，BLOCK_SIZE=256 匹配 num_warps=8。
- **触发调优**：计算单元利用率 < 60% 且无访存瓶颈 → 增加 num_warps；出现寄存器溢出 → 减少 num_warps。
- **取值范围**：NPU 场景推荐 2/4/8 三档，禁止使用 1 或 >8 的值，避免调度异常。

### 3.3 num_stages 调优规则
`num_stages` 控制软件流水线深度，通过预取后续迭代数据掩盖访存延迟，默认值为 3。
- **访存密集型算子**（Norm、Softmax、逐元素）：推荐 2~3 级，过深会增加寄存器压力，收益边际递减。
- **计算密集型算子**（MatMul）：推荐 3~4 级，通过预取权重/输入数据掩盖访存延迟。
- **触发调优**：RAW 依赖等待占比 > 15% → 增加 1 级流水线；出现寄存器溢出 → 减少 1 级。
- **禁忌**：分块尺寸 ≤ 64 时禁止开启多级流水线，调度开销会抵消收益。

### 代码示例：流水线配置调整
**Before（默认配置，无显式优化）**
```python
@triton.jit
def matmul_pipe_before(a_ptr, b_ptr, c_ptr, M, N, K,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]  # K 必须是 tl.constexpr！否则 pointer type 报错
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

# 调用：使用默认编译配置
grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
matmul_pipe_before[grid](a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
```

**After（显式配置流水线与并行度）**
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
```

---

## 四、访存对齐与地址优化
### 4.1 优化原理
Ascend NPU 内存控制器对对齐访问有显著性能加成，128 字节对齐的连续访存比未对齐访问效率高 20%~50%。通过注入编译期对齐提示，可让编译器生成更优的访存指令。

### 4.2 实现方式
使用 Triton 3.4.0 标准 API `tl.multiple_of` 声明指针对齐属性，该语法可正常通过 TTIR→HIVM 转换与 bisheng 编译。
- 对齐字节数 = 分块元素数 × 单元素字节数；例如 FP32 + BLOCK_SIZE=256 → 对齐 1024 字节。
- 仅对基址指针声明对齐，偏移后的指针自动继承对齐属性。

### 代码示例：对齐提示注入
**Before（无对齐提示）**
```python
@triton.jit
def align_before(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)
```

**After（注入对齐提示）**
```python
@triton.jit
def align_after(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 声明指针对齐：BLOCK_SIZE个FP32元素，对齐4*BLOCK_SIZE字节
    x_ptr = tl.multiple_of(x_ptr, 4 * BLOCK_SIZE)
    out_ptr = tl.multiple_of(out_ptr, 4 * BLOCK_SIZE)
    
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)
```

---

## 五、数据类型与精度优化
### 5.1 优化原则
访存瓶颈型算子（带宽利用率 > 90%），可通过降低存储位宽减少访存数据量，核心计算保持高精度以保证数值正确性。
- **存储降级**：全局内存加载/存储使用 FP16，减少 50% 访存带宽。
- **计算保精**：中间计算、归约、累加必须使用 FP32，避免溢出与精度损失。
- **适用场景**：逐元素算子、Norm 类算子、矩阵乘的权重/输入存储。

### 5.2 禁忌场景
- 指数、对数、开方等运算禁止直接使用 FP16 计算，必须转 FP32 后运算。
- 归约求和、累加器必须保持 FP32，禁止降级。

### 代码示例：混合精度优化
**Before（全FP32）**
```python
@triton.jit
def precision_before(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 加载、计算、存储全为FP32，访存带宽占用高
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * 2.0 + 1.0
    tl.store(out_ptr + offsets, out, mask=mask)
```

**After（存储FP16 + 计算FP32）**
```python
@triton.jit
def precision_after(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # FP16加载 → 转FP32计算 → FP16存储，访存量减半
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    out = x * 2.0 + 1.0
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)
```

---

## 六、优化收益量化预期
| 优化维度 | 典型场景预期加速比 | 上限加速比 |
|----------|--------------------|------------|
| 分块尺寸调优 | 10% ~ 30% | 40%（初始分块极小时） |
| 软件流水线调优 | 10% ~ 25% | 35%（初始无流水线时） |
| 访存对齐优化 | 5% ~ 15% | 25%（完全未对齐场景） |
| 数据精度优化 | 15% ~ 40% | 50%（纯访存瓶颈场景） |

> 叠加收益：多维度优化可叠加，典型算子 Tier 3 整体收益可达 30% ~ 60%。

---

## 七、不可调优边界与禁忌
1. **禁止改变算法逻辑**：Tier 3 仅调整参数，禁止修改运算顺序、融合逻辑、归约方式，此类优化属于 Tier 1/Tier 2 范畴。
2. **禁止使用 GPU 专属配置**：禁止设置 `num_ctas > 1`、`maxnreg`、TMA 相关参数，NPU 工具链不支持，会导致转换失败。
3. **禁止分块越界**：逐元素算子分块不得小于 32，矩阵乘分块不得小于 32×32，否则调度开销会抵消所有收益。
4. **禁止盲目降精度**：业务有精度约束时，不得随意降级数据类型，必须保证数值误差在允许范围内。
5. **禁止过度流水线**：`num_stages` 最大不超过 4，过深会导致寄存器严重溢出，性能反向下降。

---

## 八、常见错误与修复方案
### 1. 编译参数写入 @triton.jit 装饰器
- **错误现象**：`ast_to_ttir` 阶段报错，提示装饰器不支持 `num_warps` 参数。
- **触发原因**：误将 `num_warps`、`num_stages` 写入 `@triton.jit()` 括号内。
- **修复方案**：将编译参数移至内核调用时传递，装饰器仅保留官方合法参数。

### 2. 分块非 2 的幂次
- **错误现象**：bisheng 编译警告，NPU 访存效率骤降，bw_util 上不去。
- **触发原因**：设置 BLOCK_SIZE=100、200 等非 2 幂次值。
- **修复方案**：所有分块参数统一为 2 的幂次，按 32/64/128/256/512 档位调整。

### 3. 流水线过深导致寄存器溢出
- **错误现象**：编译通过但性能不升反降，HIVM 出现寄存器溢出标记。
- **触发原因**：小分块配置高 `num_stages`，或同时开大 `num_warps` 与 `num_stages`。
- **修复方案**：访存密集型 `num_stages` 最高 3 级，计算密集型最高 4 级；溢出时逐级降低参数。

### 4. 对齐提示与分块不匹配
- **错误现象**：TTIR→HIVM 转换失败，或运行时出现访存越界。
- **触发原因**：`tl.multiple_of` 的对齐字节数与实际分块数据量不匹配。
- **修复方案**：对齐字节数 = 元素字节数 × BLOCK_SIZE，确保分块大小是对齐值的整数倍。

### 5. 计算精度降级导致数值错误
- **错误现象**：输出结果误差过大，甚至出现 nan/inf。
- **触发原因**：归约、指数运算直接使用 FP16 计算，导致溢出或精度丢失。
- **修复方案**：所有中间计算、累加、归约必须使用 FP32，仅全局存储可降级为 FP16。

### 6. 分块过小导致调度开销过高
- **错误现象**：op 数量少但总周期高，计算单元利用率低。
- **触发原因**：盲目减小分块，内核启动与调度开销占比超过 30%。
- **修复方案**：遵循最小分块阈值，逐元素算子不小于 128，矩阵乘不小于 64×64。

需要我补充某类算子（如 MatMul）的完整 Tier 3 调优决策树，或者对应的性能对比测试模板吗？