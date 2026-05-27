# Emulator → Real Triton 转换指南

将 emulator 校验通过的 kernel 转为可上板运行的真实 Triton kernel。

## 核心结论：零侵入转换

kernel 计算逻辑（offset 计算、mask 生成、累加、数学运算）完全不动，只改调用约定。

## 5 处机械性改动

### 1. Import 替换

```python
# emulator
from common import tl, xarray, launch_kernel_1d, launch_kernel_2d, verify, EmulatorError

# 真实 Triton
import triton
import triton.language as tl
```

### 2. 添加 @triton.jit 装饰器

```python
# emulator
def conv2d_kernel(...):

# 真实 Triton
@triton.jit
def conv2d_kernel(...):
```

### 3. tl.load 调用约定

```python
# emulator: tl.load(base_ptr, offset, mask=mask)
x_vals = tl.load(x_ptr, x_offsets, mask=combined_mask, other=0.0)

# 真实 Triton: 指针算术前移到参数中
x_vals = tl.load(x_ptr + x_offsets, mask=combined_mask, other=0.0)
```

### 4. tl.store 调用约定

```python
# emulator: tl.store(base_ptr, offset, value)
tl.store(out_ptr, np.array([out_offset], dtype=np.int64), out_val)

# 真实 Triton: 同 load，指针算术前移
tl.store(out_ptr + out_offset, out_val)
```

注意：emulator 中 `np.array([offset])` 的包裹是 emulator API 设计需要，真实 Triton 不需要。

### 5. Launch 方式

```python
# emulator: launch_kernel_Nd 函数
launch_kernel_1d(
    kernel, x_flat, out_flat, ...,
    BLOCK_SIZE,
    grid_size=grid_size,
)

# 真实 Triton: kernel[grid] 调用语法
kernel[(grid_size,)](
    x, out, ...,
    BLOCK_SIZE=BLOCK_SIZE,
)
```

附带变化：
- 数据从 numpy 平铺数组（`.ravel()`）变为 torch.Tensor（直接传，stride 由 `.stride()` 获取）
- 输出分配从 `np.zeros(...)` 变为 `torch.empty(..., device='cuda')`
- grid 计算从 `tl.cdiv()` 变为 `triton.cdiv()`

## 需要新增的 kernel

emulator 中部分操作在 Python/numpy 层完成，真实 Triton 需要补写 kernel：

| 操作 | emulator 方式 | 真实 Triton 方式 |
|------|-------------|---------------|
| 残差 add | numpy `out + identity` | `add_kernel` (element-wise) |
| FC / linear | 无 emulator kernel | `linear_kernel` (per-output-element 累加) |
| reshape / flatten | numpy `.reshape()` / `.view()` | `tensor.view()` (torch，不需要 kernel) |

## 转换验证：ResNet18 示例

文件：`emulators/test/resnet18/triton_real.py`

输入 [B, 3, 224, 224] → 输出 [B, 1000]，包含：
- 7 个 `@triton.jit` kernel（conv2d, bn, relu, maxpool, avgpool, add, linear）
- 7 个 launcher 函数
- 配置驱动的 ResNet18 forward pass
- `load_resnet18_weights()` 权重加载
- 数值验证 test

### ResNet18 层配置

```python
RESNET18_LAYERS = [
    ('layer1', 64,  64,  1, False),   # no downsample
    ('layer2', 64,  128, 2, True),    # block 0 has 1x1 projection
    ('layer3', 128, 256, 2, True),
    ('layer4', 256, 512, 2, True),
]
```

### 权重格式

权重 dict 的 key 与 torchvision `resnet18.state_dict()` 完全一致，可通过
`load_resnet18_weights(device)` 直接加载预训练权重。

## 转换清单模板

对任意 emulator 算子目录 `emulators/test/<op>/__init__.py`，生成真实 Triton 版本的步骤：

1. 复制 kernel 函数体
2. 添加 `import triton; import triton.language as tl`
3. 每个 kernel 函数前加 `@triton.jit`
4. 全局替换 `tl.load(ptr, off,` → `tl.load(ptr + off,`
5. 全局替换 `tl.store(ptr, off,` / `tl.store(ptr, np.array([off])` → `tl.store(ptr + off,`
6. 将 `emulate_xxx()` 改写为 launcher：numpy → torch tensor, `launch_kernel_Nd` → `kernel[grid]`
7. 补写缺失的 kernel（如 add、linear）
8. 添加 weight loader 和 test
