# 910B3 部署指南 — Execution 层
> 本层 2 个文件本地可用，2 个文件需 910B3。

---

## 0. 关键: msprof op simulator 需要编译后的 .o 文件

**msprof op simulator 不接受 .py 源码!** 必须先编译为 .o 二进制。

完整流水线:
```
Triton .py → compiler.py (bisheng编译) → .asc.o 二进制
  → msprof op simulator ./kernel.asc.o → trace.json
  → msprof_analyzer.parse(trace.json) → pipeline_report.json
```

编译命令 (compiler.py 自动执行):
```bash
# Step 1: 编译 →仿真模式 .o
bisheng -c kernel.asc -o kernel.asc.o --npu-arch=dav-2201 --run-mode=sim

# Step 2: 链接仿真库 → 可执行文件
bisheng -L${INSTALL_DIR}/tools/simulator/dav_2201/lib \
  -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++ \
  kernel.asc.o -o kernel_sim

# Step 3: 运行仿真
msprof op simulator --soc-version=Ascend910B3 ./kernel_sim

# 910B3 的 --npu-arch 对应 dav-2201 (可通过 npu-smi info 确认)
```

**本地 (Windows/无 CANN)**: compiler.py 不可用，整个 msprof 分析链 fallback 到 cost_emulator simulator。

---

## 1. emulator_runner.py — CPU 仿真验证

**本地即可运行，不需要 NPU。支持智能识别算子类型。**

`auto_verify(kernel_path, kernel_fn_name, op_type)`:
- `op_type="element_wise"` → `_emulate_element_wise` + `_reference_add`
- `op_type="matmul"` → `_emulate_matmul` + `_reference_matmul`
- `op_type="add"` → `_emulate_element_wise` + `_reference_add`
- `op_type="unknown"` → `_emulate_generic` (仅语法检查)
- `op_type` 由 main.py 自动检测，不需要手动指定

验证命令:
```bash
python execution/emulator_runner.py
```

**需要在 910B3 上扩展**: 如果新算子类型需要特殊的 emulate/reference，在 `KNOWN_KERNEL_TYPES` dict 中添加。

---

## 2. compiler.py — 编译 + HIVMIR 提取

**需要 CANN + 编译器。**

### 2.1 确认编译器可用

```bash
which ascendc || which bishengir-compile || which bisheng
ls /usr/local/Ascend/ascend-toolkit/latest/compiler/bin/
```

如果找不到，在 `CompilerInterface._find_compiler()` 中添加路径：
```python
# compiler.py ~L245
standard_paths = [
    "/usr/local/Ascend/ascend-toolkit/latest/compiler/bin",
    "/usr/local/Ascend/cann/compiler/bin",
    "/your/custom/path/here",  # ← 添加
]
```

### 2.2 验证 HIVMIR 提取

```bash
python execution/compiler.py
# 预期: "Extraction OK — found vadd op"
```

---

## 3. hardware_runner.py — 910B3 性能测试

**需要 CANN + msprof + NPU。**

### 3.1 确认 msprof 可用

```bash
which msprof
msprof --version
```

### 3.2 手动测试

```bash
# 编译 kernel
python3 -c "from execution.compiler import CompilerInterface; ..."
# benchmark
msprof op ./binary_path
```

---

## 待补全清单

| 文件 | 补全项 | 优先级 |
|---|---|---|
| `compiler.py` | 编译器路径确认 | ⭐⭐⭐ |
| `hardware_runner.py` | msprof benchmark 实现 | ⭐⭐⭐ (当前 stub) |
| `emulator_runner.py` | 无需修改 | ⭐ |
