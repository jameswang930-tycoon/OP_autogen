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

## Triton Ascend (NPU) 后端约束

### Grid size 限制

Triton Ascend 后端要求 grid size（coreDim）不超过 UINT16_MAX（65535），超出报错：
`coreDim=xxxx can't be greater than UINT16_MAX`。

ResNet18 中 grid size 超限的 kernel（B=1, 输入 224×224）：

| 算子 | 输出 shape | Grid size |
|------|-----------|-----------|
| stem conv2d | [1, 64, 112, 112] | 802,816 |
| stem maxpool | [1, 64, 56, 56] | 200,704 |
| layer1 conv2d | [1, 64, 56, 56] | 200,704 |
| layer2 conv2d (s2) | [1, 128, 28, 28] | 100,352 |
| layer3 conv2d (s2) | [1, 256, 14, 14] | 50,176 |
| layer4 conv2d (s2) | [1, 512, 7, 7] | 25,088 |

### 应对方案（按阶段选择）

**阶段 1：跑通正确性（快速验证）**

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

注意：此方案会触发分批调度（grid > 物理核数时分多批执行），引入额外设备侧开销，**仅用于正确性验证，不适合性能评估**。

**阶段 2：性能优化（按 NPU 核数重设计 grid）**

官方推荐：grid size = 物理 aicore 数量，kernel 内部用两级 tiling 处理。

```python
import triton.runtime.driver as driver

# 获取物理核数
props = driver.active.utils.get_device_properties(device)
num_aicore = props["num_aicore"]           # 含 tl.dot 的算子
num_vectorcore = props["num_vectorcore"]   # 纯 vector 算子

# 两级 tiling
#   block_size = 总元素数 / 核数（核间切分）
#   sub_block_size = 控制片上内存 ≤ 192KB（核内切分）
block_size = total_elements // num_core
sub_block_size = 8192  # 示例值，建议用 autotune 寻优
```

```python
# kernel 示例：两级 tiling
@triton.jit
def kernel(x_ptr, out_ptr, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for sub_offset in range(0, XBLOCK, XBLOCK_SUB):
        idx = xoffset + sub_offset + tl.arange(0, XBLOCK_SUB)
        mask = idx < numel
        x = tl.load(x_ptr + idx, mask=mask)
        tl.store(out_ptr + idx, result, mask=mask)
```

### 其他 NPU 约束

- 内存对齐：vector 算子需 32 字节对齐，cube+vector 融合算子需 512 字节对齐
- 片上内存：单 aicore UB ≤ 192KB（Atlas 800T/I A2），超出报 `ub overflow`
- 2D grid 自动合并为 1D：`(4, 5)` 和 `(20,)` 等价
- 去掉 GPU 特有逻辑：`torch.cuda.*`、CUDA stream/event、`assert x.is_cuda` 等

### 设备适配

kernel 和 launcher 不包含任何 CUDA 硬编码，device 由调用方通过 `torch.Tensor` 的 placement 决定：

```python
import torch_npu  # 注册 'npu' device
device = 'npu:0'
x = torch.randn(B, 3, 224, 224, device=device)
weights = make_resnet18_weights(device)
out = resnet18_forward(x, weights)
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
