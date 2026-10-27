# WSL2 搭建 HIVMIR 编译环境 — 完整指南

> 目标：在 Windows 本地通过 WSL2 安装 CANN + Bisheng 编译器，获取 HIVM 中间表示并解析。
> 不需要 NPU 硬件。纯 CPU 编译。

---

## 整体流程

```
Windows 10/11 (你当前的机器)
  │
  ├─ Step 1: 启用 WSL2 + 安装 Ubuntu 22.04
  │
  ├─ Step 2: 在 WSL/Ubuntu 中安装 CANN 社区版 Toolkit
  │     └─ 包含 bishengir-compile / bishengir-opt
  │
  ├─ Step 3: 验证: 手写 .mlir → bishengir-opt → HIVM IR dump
  │
  └─ Step 4: 重写 hivmir_analyzer.py 适配真实 MLIR 格式
```

---

## Step 1: 启用 WSL2 + 安装 Ubuntu 22.04

### 1.1 管理员 PowerShell 中执行：

```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机平台
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启电脑
Restart-Computer
```

### 1.2 重启后，下载并安装 WSL2 内核更新包：

```
https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
```

### 1.3 设置 WSL2 为默认版本 + 安装 Ubuntu 22.04：

```powershell
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

安装完成后，首次进入会提示创建用户名和密码。

### 1.4 进入 WSL 后更新系统：

```bash
sudo apt update && sudo apt upgrade -y
```

---

## Step 2: 安装 CANN 社区版

### 2.1 安装系统依赖

```bash
sudo apt install -y gcc g++ make cmake python3 python3-pip \
    zlib1g zlib1g-dev libssl-dev libffi-dev unzip \
    libsqlite3-dev pciutils net-tools libblas-dev gfortran

# CANN 需要 Python 3.7~3.9，Ubuntu 22.04 自带 Python 3.10
# 安装 Python 3.9
sudo apt install -y python3.9 python3.9-dev python3.9-venv
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 protobuf scipy requests
```

### 2.2 下载 CANN 社区版

从昇腾社区下载对应 `.run` 包：

👉 **https://www.hiascend.com/developer/download/community/result?module=cann**

选择：
- 操作系统: Linux
- 架构: x86_64
- 版本: 最新社区版 (推荐 8.2.RC1 以上)
- 包类型: Toolkit 开发套件

示例文件名: `Ascend-cann-toolkit_8.2.RC1.alpha003_linux-x86_64.run`

### 2.3 安装 CANN

```bash
# 赋予执行权限
chmod +x Ascend-cann-toolkit_*.run

# 安装（不需要 root，安装到 $HOME 目录）
./Ascend-cann-toolkit_*.run --full --install-path=$HOME/Ascend

# 按提示操作。检测不到 NPU 硬件时选择 "仅安装开发环境"
```

### 2.4 验证安装

```bash
ls $HOME/Ascend/ascend-toolkit/latest/tools/bishengir/
# 应该看到: bishengir-compile, bishengir-opt, bishengir-translate 等
```

### 2.5 配置环境变量（写入 ~/.bashrc）

```bash
cat >> ~/.bashrc << 'EOF'
# CANN 环境
export ASCEND_HOME=$HOME/Ascend/ascend-toolkit/latest
export PATH=$ASCEND_HOME/tools/bishengir:$ASCEND_HOME/compiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=$ASCEND_HOME/opp
export ASCEND_GLOBAL_LOG_LEVEL=3
EOF

source ~/.bashrc
```

### 2.6 验证工具可用

```bash
bishengir-opt --version
bishengir-compile --help
```

---

## Step 3: 获取 HIVM IR — 两种方式

### 方式 A：直接手写 MLIR（最快，不需要 Triton kernel）

创建一个测试文件 `vec_add.mlir`：

```mlir
func.func @add_kernel(%arg0: memref<1024xf16, #hivm.address_space<gm>>,
                       %arg1: memref<1024xf16, #hivm.address_space<gm>>,
                       %arg2: memref<1024xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {

    %buf_a = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_b = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_c = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>

    // GM → UB
    hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>)
                 outs(%buf_a : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<1024xf16, #hivm.address_space<gm>>)
                 outs(%buf_b : memref<1024xf16, #hivm.address_space<ub>>)

    // vadd
    hivm.hir.vadd ins(%buf_a, %buf_b : memref<1024xf16, #hivm.address_space<ub>>,
                                       memref<1024xf16, #hivm.address_space<ub>>)
                 outs(%buf_c : memref<1024xf16, #hivm.address_space<ub>>)

    // UB → GM
    hivm.hir.store ins(%buf_c : memref<1024xf16, #hivm.address_space<ub>>)
                  outs(%arg2 : memref<1024xf16, #hivm.address_space<gm>>)

    return
}
```

编译 + 获取 HIVM IR dump：

```bash
# 方式 1: 直接编译到 .o（会经过完整 HIVM pipeline）
bishengir-compile vec_add.mlir -o vec_add.o

# 方式 2: 用 bishengir-opt 跑 HIVM lowering pass，打印中间 IR
bishengir-opt vec_add.mlir \
    --convert-hfusion-to-hivm \
    --print-ir-after-all > hivm_dump.txt 2>&1

# 方式 3: 编译时开启 IR dump
bishengir-compile vec_add.mlir -o vec_add.o \
    --mlir-print-ir-after-all 2>&1 | tee compile_ir.log
```

### 方式 B：Triton kernel → MLIR → HIVM（完整链路）

```
Triton .py → triton-ascend (TritonNext) → MLIR → bishengir-compile → HIVM .o
```

更复杂的全链路（需要安装 triton-ascend 编译框架），建议先用方式 A 验证工具链。

---

## Step 4: 我们需要重写的 hivmir_analyzer.py

### 真实格式 vs 我们当前假设的格式

| 维度 | 当前 hivmir_analyzer.py | 真实官方格式 |
|---|---|---|
| alloc | `hivm.alloc %ub_1 : memref<128KB>` | `%buf = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>` |
| size_kb | 直接解析 `128KB` | 从 `1024×f16` 计算: 1024*2/1024 = 2KB |
| 地址空间 | 变量名前缀推断 `gm_`→GM | `#hivm.address_space<gm/ub/l1>` |
| GM→UB | `hivm.gm_to_ub %ub_1, %gm_1` | `hivm.hir.load ins(%gm) outs(%ub)` |
| UB→GM | `hivm.ub_to_gm %gm_2, %ub_2` | `hivm.hir.store ins(%ub) outs(%gm)` |
| vadd | `hivm.vadd %ub_2, %ub_1, 2.0` | `hivm.hir.vadd ins(%a, %b : type, type) outs(%c)` |
| 依赖分析 | 基于文本匹配 RAW/WAR/WAW | 基于 MLIR SSA use-def chain |
| 函数属性 | 无 | `hacc.entry`, `hacc.function_kind<DEVICE>` |

### 需要重写的核心部分

1. **`HIVMIRParser._try_parse_alloc()`** — 匹配 `memref.alloc() : memref<N×type, #hivm.address_space<ub>>`
2. **`HIVMIRParser._try_parse_op()`** — 匹配 `hivm.hir.load/store/vadd/matmul` 的 ins/outs 格式
3. **`HIVMIROp` 数据结构** — 增加 dtype 字段（从 `memref<1024×f16>` 解析）
4. **`_analyze_dependencies()`** — 基于 SSA value (%buf_a 等) 做 use-def 分析
5. **`_build_op()` 的 engine 映射** — `hivm.hir.load` → GM→UB, `hivm.hir.store` → UB→GM
6. **`size_kb` 计算** — 从 shape × dtype_size 计算（如 1024×f16 = 2KB），不再直接读取

### 额外可提取的信息（真实格式比我们假设的更丰富）

- matmul 的 `block_sizes`, `tiling_params`, `a_transpose` 等属性
- 函数签名中的参数个数和类型（识别是 element-wise 还是 matmul）
- GPU→NPU 对应: MTE2=GM→UB, MTE3=UB→GM, VECTOR=VecUnit, Cube=CubeUnit

---

## 常用命令速查

```bash
# 检查 CANN 环境
echo $ASCEND_HOME
ls $ASCEND_HOME/tools/bishengir/

# 编译 .mlir → .o
bishengir-compile input.mlir -o kernel.o

# 获取 IR dump
bishengir-opt input.mlir --print-ir-after-all 2>&1 | less

# 查看 HIVM lowering pass 名称
bishengir-opt --help | grep -i hivm

# 从 HIR 转 HIVM
bishengir-opt input.mlir \
    --canonicalize \
    -convert-hfusion-to-hivm \
    --print-ir-after-all > hivm_full.log 2>&1
```

---

## 注意事项

1. **磁盘空间**：CANN 安装约需 10-15GB，WSL2 虚拟磁盘会自动增长
2. **Python 版本**：必须 3.7~3.9，Ubuntu 22.04 自带 3.10 需要额外安装 3.9
3. **不需要 GPU/NPU 驱动**：只做编译和 IR 提取
4. **msprof op simulator 无法运行**：msprof 需要真实 trace 数据，在 WSL2 上不可用
5. **WSL2 文件访问**：Windows 文件在 `/mnt/c/`, `/mnt/d/` 下可访问，但编译建议在 WSL 原生文件系统 (`~/`) 下进行（性能更好）
6. **CANN 版本配套**：bishengir-compile 版本随 CANN 一起发布，注意版本兼容性

---

## 下一步

1. 按此指南在 WSL2 中安装 CANN
2. 运行 Step 3 的测试 .mlir 文件验证工具链
3. 将生成的 HIVM IR dump 保存一份
4. 基于真实格式重写 `hivmir_analyzer.py`
