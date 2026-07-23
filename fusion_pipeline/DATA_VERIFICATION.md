# 数据验证状态说明

## ⚠️ 重要提示

**当前实现的 HIVMIR 解析器基于文档推断，尚未在真实华为编译器上验证！**

---

## ✅ 已验证的真实数据

### 1. Simulator 数据（来自 costModel/cost_emulator/simulator.py）

这些数据**已经验证**，来自真实测量：

#### 引擎定义
```python
ENG_NAME = {
    0: 'GM→UB',
    1: 'UB→GM',
    2: 'VecUnit',
    3: 'GM→L1',
    4: 'L1→L0',
    5: 'CubeUnit',
    6: 'L0→GM',
}
```
**来源**: ✅ 基于 910B3 真实测量

#### 带宽参数（SATURATION_PARAMS）
```python
SATURATION_PARAMS = {
    0: {"vpeak": 121.08, "k0": 6.65, "peak_clamp": 80.83},    # GM→UB - 实测
    1: {"vpeak": 190.19, "k0": 10.72, "peak_clamp": 76.67},   # UB→GM - 实测
    2: {"vpeak": 461.0, "k0": 4.50, "peak_clamp": 404.0},     # VecUnit - 实测
    3: {"vpeak": 37.5, "k0": 6.65, "peak_clamp": 37.5},        # GM→L1 - placeholder
    4: {"vpeak": 100.0, "k0": 6.65, "peak_clamp": 100.0},      # L1→L0 - placeholder
    5: {"vpeak": 150.0, "k0": 0, "peak_clamp": 150.0},         # CubeUnit - placeholder
    6: {"vpeak": 37.5, "k0": 6.65, "peak_clamp": 37.5},        # L0→GM - placeholder
}
```
**来源**:
- ✅ GM→UB: perf_test 实测
- ✅ UB→GM: perf_test 实测
- ✅ VecUnit: perf_test 实测
- ⚠️ 其他: placeholder（需要实测）

#### 内存区域定义
```python
MEMORY_CAPACITY_KB = {
    'UB': 512.0,    # Unified Buffer
    'L1': 2048.0,   # L1 SRAM (2 MB)
    'L0': 1024.0,   # L0 register file (1 MB)
    'GM': None,     # Global Memory (unbounded)
}
```
**来源**: ✅ 来自 910B3 硬件规格

---

## ⚠️ 基于推断的数据

### 1. HIVMIR 语法格式

**当前假设的格式**:
```mlir
hivm.gm_to_ub %ub_1, %gm_1 : memref<64KB>
hivm.vadd %ub_2, %ub_1, 1.0
hivm.ub_to_gm %gm_2, %ub_2 : memref<64KB>
```

**来源**: ⚠️ 基于 AscendNPU-IR 文档推断

**需要验证**:
- [ ] 真实的语法格式是什么？
- [ ] 操作名称是否准确？
- [ ] 操作数顺序是否正确？
- [ ] memref 语法是否正确？

### 2. 操作类型到引擎的映射

**当前映射**:
```python
OP_TO_ENGINE = {
    'gm_to_ub': 'GM→UB',
    'ub_to_gm': 'UB→GM',
    'gm_to_l1': 'GM→L1',
    'l1_to_l0': 'L1→L0',
    'l0_to_gm': 'L0→GM',
    'vadd': 'VecUnit',
    'vsub': 'VecUnit',
    'vmul': 'VecUnit',
    'matrixmul': 'CubeUnit',
}
```

**来源**: ⚠️ 推断

**需要验证**:
- [ ] 这些操作名称在 HIVMIR 中是否存在？
- [ ] 是否有其他操作类型？
- [ ] 映射是否准确？

### 3. 缓冲区命名规则

**当前假设**:
```python
BUFFER_REGION = {
    'gm': 'Global Memory',
    'ub': 'Unified Buffer',
    'l1': 'L1 SRAM',
    'l0': 'L0 Register',
}
```

**来源**: ⚠️ 基于前缀推断

**需要验证**:
- [ ] 真实的缓冲区命名规则是什么？
- [ ] 前缀是否准确？

---

## 🔍 如何验证真实数据

### 步骤 1: 在 910B3 服务器上运行验证脚本

```bash
# 运行验证脚本
python fusion_pipeline/verify_real_data.py \
    --kernel fusion_pipeline/example_kernels/vadd_kernel.py \
    --output-dir ./verify_output

# 查看输出
cat ./verify_output/parser_update.json
```

### 步骤 2: 编译并捕获真实 IR

```bash
# 方法 1: 使用华为编译器
bishengir-compile \
    --mlir-print-ir-after-all \
    your_kernel.py \
    -o output.om \
    > ir_dump.txt

# 方法 2: 使用 Triton 编译
python -c "
import torch
import triton
from your_kernel import kernel_fn

# 强制打印 IR
import os
os.environ['TRITON_PRINT_IR'] = '1'
kernel_fn[grid](*args)
" > ir_dump.txt
```

### 步骤 3: 分析真实格式

查看生成的 IR 文件：
```bash
# 查找 HIVM 相关内容
grep -i "hivm" ir_dump.txt

# 查找操作定义
grep -E "(gm_to|ub_to|vadd|matrix)" ir_dump.txt

# 查找引擎定义
grep -E "(GM→|UB→|VecUnit|CubeUnit)" ir_dump.txt
```

### 步骤 4: 更新解析器

根据真实格式更新 `complete_data_merge.py`:

```python
# 更新操作模式
OP_PATTERNS = {
    r'真实格式正则表达式': '操作类型',
}

# 更新引擎映射
OP_TO_ENGINE = {
    '真实操作名': '真实引擎名',
}
```

---

## 📋 验证清单

在 910B3 服务器上需要验证的内容：

### HIVMIR 格式
- [ ] 操作语法格式
- [ ] 操作名称（gm_to_ub, vadd 等）
- [ ] 操作数顺序
- [ ] memref 语法

### 字段名称
- [ ] 引擎名称（GM→UB, VecUnit 等）
- [ ] 缓冲区命名规则
- [ ] 数据类型标识

### 带宽参数
- [ ] GM→L1 的真实带宽
- [ ] L1→L0 的真实带宽
- [ ] L0→GM 的真实带宽
- [ ] CubeUnit 的真实带宽

---

## 🎯 总结

### ✅ 已验证（可直接使用）
1. GM→UB 带宽参数
2. UB→GM 带宽参数
3. VecUnit 带宽参数
4. 内存容量规格
5. 引擎定义（部分）

### ⚠️ 需要验证（当前为推断）
1. HIVMIR 语法格式
2. 操作类型名称
3. 操作到引擎的映射
4. 其他引擎的带宽参数

### 📝 建议

**立即可用**:
- Simulator 数据（时序、带宽利用率）✅
- 内存分析 ✅

**需要在 910B3 上验证**:
- HIVMIR 解析器格式 ⚠️
- 操作类型和引擎映射 ⚠️

**使用建议**:
1. 先用当前版本测试流程
2. 在 910B3 上运行 `verify_real_data.py`
3. 根据真实输出更新解析器
4. 验证更新后的结果

---

## 📞 下一步

1. **在 910B3 服务器运行验证脚本**:
   ```bash
   python fusion_pipeline/verify_real_data.py --kernel your_kernel.py
   ```

2. **查看真实 IR 输出**:
   ```bash
   cat verify_output/full_ir_dump.txt
   ```

3. **更新解析器**:
   根据 `parser_update.json` 中的建议更新代码

---

**重要**: 当前实现提供了完整的框架和流程，但 HIVMIR 解析部分需要在真实环境中验证和调整！