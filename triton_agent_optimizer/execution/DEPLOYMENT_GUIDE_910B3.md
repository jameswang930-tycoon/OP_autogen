# 910B3 部署指南 — Execution 层
> 本层 2 个文件本地可用，2 个文件需 910B3。

---

## 1. emulator_runner.py — CPU 仿真验证

**本地即可运行，不需要 NPU。**

验证命令:
```bash
python execution/emulator_runner.py
```

**需要在 910B3 上确认**: kernel 函数名是否正确 (`add_kernel` → 你的实际函数名)。

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
