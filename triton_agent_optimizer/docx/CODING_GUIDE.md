# Triton Kernel 编码指南 — AI Coder 参考

> 每次修改代码前必读。修改必须精确、最小化、保证语法和语义正确。

---

## 1. 核心规则

### 1.1 最小化改动
- **只改 Plan 中指定的内容**，不要添加新功能
- 如果 Plan 说"BLOCK_SIZE 256→1024"，只改那个数字
- 不要重命名函数、不要改参数顺序、不要重构

### 1.2 保持函数签名不变
- **绝对不能改函数名**。函数名必须和原 kernel 完全一致
- **不能增删参数**。参数个数、顺序、名字必须和原来一样
- `tl.constexpr` 标记的参数保持标记不变

### 1.3 必须保持 `@triton.jit` 装饰器
- 输出代码的第一行函数定义上方必须有 `@triton.jit`

---

## 2. 常见错误及修复

### 错误: 'xxx_kernel' not found
**原因**: 你改了函数名。  
**修复**: 保持原函数名不变。只改函数体内部的实现。

### 错误: takes N positional arguments but M were given
**原因**: 你改了参数个数或删了参数。  
**修复**: 保持原参数列表不变。如需新增参数，加在已有参数后面且有默认值。

### 错误: type object 'tl' has no attribute 'xxx'
**原因**: 用了 Triton 不支持的 API。  
**修复**: 只使用 `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.sum`, `tl.max`, `tl.min`, `tl.math.exp`, `tl.math.sqrt`, `tl.math.rsqrt`, `tl.constexpr`, `triton.jit`, `triton.cdiv`

### 错误: operands could not be broadcast together
**原因**: shape 不匹配。  
**修复**: `tl.arange(0, N)` 产生 `[N]` shape。`tl.sum(x, axis=0)` 产生标量。标量和 `[N]` 运算需要 broadcast: `x - scalar` 是合法的。

### 错误: invalid value (NaN)
**原因**: 数学运算产生了非法值（如负数开方、除零）。  
**修复**: 对输入做 clamp 或加 epsilon: `tl.math.rsqrt(x + 1e-5)`

---

## 3. 输出格式

输出完整的 Python 文件，不要 markdown 包裹，不要解释文字。

正确示例：
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

错误示例：
- ❌ 用 markdown 代码块包裹
- ❌ 函数名和原 kernel 不同
- ❌ 删除了 `import triton` 或 `@triton.jit`
- ❌ 改了参数顺序或增删参数
- ❌ 添加了不需要的 import

---

## 4. 910B3 硬件约束

| 约束 | 值 |
|---|---|
| UB 容量 | 192 KB/core |
| fp16 | 2 bytes/elem |
| fp32 | 4 bytes/elem |
| tile_size × n_buffers × elem_size ≤ 192KB |
| num_warps | 1~8 |
| BLOCK_SIZE | 2 的幂 (256, 512, 1024, 2048...) |
