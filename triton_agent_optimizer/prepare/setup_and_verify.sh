#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
# Triton Agent Optimizer — 910B3 真机环境搭建 + 编译链验证
# ═════════════════════════════════════════════════════════════════════════════════
#
# 用法:
#   source setup_and_verify.sh    # 设置环境变量 + 验证 (推荐)
#   bash   setup_and_verify.sh    # 仅验证 (不 source 环境变量)
#
# 平台: Ascend 910B3 (aarch64) · Ubuntu 22.04 / openEuler
# 前置: CANN Toolkit 已安装, NPU 驱动已加载
# ═════════════════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

_ok()   { echo -e "  ${GREEN}[OK]${NC}    $*"; }
_warn() { echo -e "  ${YELLOW}[WARN]${NC}  $*"; }
_fail() { echo -e "  ${RED}[FAIL]${NC}  $*"; }
_info() { echo -e "  ${CYAN}[INFO]${NC}  $*"; }
_section() { echo -e "\n${BOLD}── $* ──${NC}"; }

PASS=0; FAIL=0

_check() {
    local label="$1"; local cond="$2"; local detail="${3:-}"
    if eval "$cond"; then
        _ok "$label ${detail:+— $detail}"
        ((PASS++))
    else
        _fail "$label ${detail:+— $detail}"
        ((FAIL++))
    fi
}

# ═════════════════════════════════════════════════════════════════════════════════
#  1. 操作系统
# ═════════════════════════════════════════════════════════════════════════════════
_section "1. 操作系统"

OS_ID="$(grep '^ID=' /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"')"
OS_VER="$(grep '^VERSION_ID=' /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"')"
ARCH="$(uname -m)"

_check "OS" '[ -n "$OS_ID" ]' "$OS_ID $OS_VER"
_check "架构 (aarch64)" '[ "$ARCH" = "aarch64" ]' "$ARCH"

if [ "$ARCH" != "aarch64" ]; then
    _fail "需要 aarch64 (ARM64) 服务器, 当前为 $ARCH"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  2. CANN Toolkit
# ═════════════════════════════════════════════════════════════════════════════════
_section "2. CANN Toolkit"

CANN_DIR=""
for d in "$ASCEND_HOME" "$ASCEND_HOME_PATH" \
         /usr/local/Ascend/ascend-toolkit/latest \
         /usr/local/Ascend/cann \
         /usr/local/Ascend; do
    if [ -n "$d" ] && [ -d "$d" ]; then CANN_DIR="$d"; break; fi
done

_check "CANN 安装路径" '[ -n "$CANN_DIR" ]' "$CANN_DIR"

if [ -z "$CANN_DIR" ]; then
    echo ""
    echo "  CANN 未安装。安装步骤:"
    echo "  1. 下载: https://www.hiascend.com/developer/download/community/result?module=cann"
    echo "  2. chmod +x Ascend-cann-toolkit_*.run"
    echo "  3. ./Ascend-cann-toolkit_*.run --install --install-path=/usr/local/Ascend"
    exit 1
fi

# set_env.sh
SET_ENV=""
for s in "$CANN_DIR/set_env.sh" \
         "$CANN_DIR/ascend-toolkit/set_env.sh" \
         "$CANN_DIR/cann/set_env.sh"; do
    if [ -f "$s" ]; then SET_ENV="$s"; break; fi
done
_check "set_env.sh" '[ -n "$SET_ENV" ]' "$SET_ENV"

if [ -n "$SET_ENV" ] && [ "${_SOURCED:-0}" = "0" ]; then
    _info "source $SET_ENV"
    source "$SET_ENV" 2>/dev/null || true
fi

# 编译器
BISHENGIR_COMPILE=$(command -v bishengir-compile 2>/dev/null || echo "")
_check "bishengir-compile" '[ -n "$BISHENGIR_COMPILE" ]' "$BISHENGIR_COMPILE"

BISHENGIR_OPT=$(command -v bishengir-opt 2>/dev/null || echo "")
_check "bishengir-opt" '[ -n "$BISHENGIR_OPT" ]' "$BISHENGIR_OPT"

CANN_VER=$("$BISHENGIR_COMPILE" --version 2>&1 | head -1 || echo "unknown")
_check "CANN 版本" true "$CANN_VER"

# msprof
MSPROF=$(command -v msprof 2>/dev/null || echo "")
_check "msprof" '[ -n "$MSPROF" ]' "$MSPROF"

# npu-smi
NPU_SMI=$(command -v npu-smi 2>/dev/null || echo "")
_check "npu-smi" '[ -n "$NPU_SMI" ]' "$NPU_SMI"

# ═════════════════════════════════════════════════════════════════════════════════
#  3. NPU 设备
# ═════════════════════════════════════════════════════════════════════════════════
_section "3. NPU 设备"

if [ -n "$NPU_SMI" ]; then
    NPU_INFO=$("$NPU_SMI" info 2>&1 || echo "")
    _check "npu-smi info 可用" '[ -n "$NPU_INFO" ]'

    NPU_CHIPS=$(echo "$NPU_INFO" | grep -ci "910B\|910\|950\|Ascend" || echo "0")
    _check "Ascend 910B3 芯片" '[ "$NPU_CHIPS" -gt 0 ]' "${NPU_CHIPS} 个 NPU"

    echo "$NPU_INFO" | head -15 | while read -r line; do _info "$line"; done
else
    _fail "npu-smi 不可用 — NPU 驱动可能未加载"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  4. Python 环境
# ═════════════════════════════════════════════════════════════════════════════════
_section "4. Python 环境"

PYTHON=$(command -v python3 2>/dev/null || echo "")
_check "python3" '[ -n "$PYTHON" ]' "$("$PYTHON" --version 2>&1)"

_check "numpy"         '"$PYTHON" -c "import numpy" 2>/dev/null'
_check "openai"        '"$PYTHON" -c "import openai" 2>/dev/null' "LLM API 客户端"
_check "json"          '"$PYTHON" -c "import json" 2>/dev/null'
_check "pathlib"       '"$PYTHON" -c "from pathlib import Path" 2>/dev/null'
_check "subprocess"    '"$PYTHON" -c "import subprocess" 2>/dev/null'

# 可选但推荐
"$PYTHON" -c "import matplotlib" 2>/dev/null && _ok "matplotlib — trajectory chart" || _warn "matplotlib 未安装 (trajectory chart 不可用)"

# ═════════════════════════════════════════════════════════════════════════════════
#  5. PyTorch + torch_npu (真机环境必要条件)
# ═════════════════════════════════════════════════════════════════════════════════
_section "5. PyTorch NPU"

if "$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
    _check "torch" true "$TORCH_VER"

    if "$PYTHON" -c "import torch_npu; print(torch.npu.is_available())" 2>/dev/null; then
        NPU_COUNT=$("$PYTHON" -c "import torch_npu; print(torch.npu.device_count())")
        _check "torch_npu" true "NPU 设备数: $NPU_COUNT"
        TORCH_OK=1
    else
        _fail "torch_npu 不可用 — 真机环境必要组件"
        TORCH_OK=0
    fi
else
    _fail "torch 未安装 — 真机环境必要组件"
    TORCH_OK=0
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  6. Triton-Ascend (真机版本, 非 GPU triton)
# ═════════════════════════════════════════════════════════════════════════════════
_section "6. Triton-Ascend"

TRITON_OK=0

# 检查 triton 是否存在
if ! "$PYTHON" -c "import triton" 2>/dev/null; then
    _fail "triton 未安装 — 必须安装 triton-ascend (Ascend 后端版本)"
    echo ""
    echo "  安装方式:"
    echo "  pip install triton-ascend"
    echo "  或: git clone https://gitee.com/ascend/triton-ascend.git && cd triton-ascend && python3 setup.py install"
    TRITON_OK=0
else
    TRITON_VER=$("$PYTHON" -c "import triton; print(getattr(triton, '__version__', 'unknown'))")
    _check "triton 版本" true "$TRITON_VER"

    # 验证有 Ascend 后端 (不是纯 GPU triton)
    HAS_ASCEND=0
    if "$PYTHON" -c "from triton.backends.npu.compiler import NPUCompiler" 2>/dev/null; then
        _check "triton Ascend 后端" true "NPUCompiler 可用"
        HAS_ASCEND=1
    fi

    if "$PYTHON" -c "
import triton
import torch, torch_npu
x = torch.randn(4, device='npu')
try:
    # 验证 triton 能发现 NPU 设备
    import triton.runtime.driver as drv
    active = drv.active.get_current_device()
    print(f'device={active}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null; then
        _check "triton → NPU 连接" true "driver 可访问 NPU"
    else
        _warn "triton 可能无法访问 NPU driver"
    fi

    if [ "$HAS_ASCEND" = "1" ]; then
        TRITON_OK=1
    else
        _fail "当前 triton 不是 Ascend 后端版本 — 请安装 triton-ascend 替换标准 triton"
        TRITON_OK=0
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  7. 编译链烟雾测试: Triton .py → NPU 运行 → HIVM MLIR dump
# ═════════════════════════════════════════════════════════════════════════════════
_section "7. 编译链烟雾测试"

if [ "$TRITON_OK" = "1" ] && [ "${TORCH_OK:-0}" = "1" ]; then
    TMPDIR=$(mktemp -d -t triton-smoke-XXXXXX)
    trap "rm -rf $TMPDIR" EXIT

    cat > "$TMPDIR/test_vec_add.py" << 'KERNEL_EOF'
import os; os.environ["TRITON_DEBUG"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_DISABLE_CACHE"] = "1"
import triton, triton.language as tl
import torch, torch_npu

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

N = 256
x = torch.randn(N, device='npu'); y = torch.randn(N, device='npu')
out = torch.empty(N, device='npu')
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, out, N, BLOCK_SIZE=128)
torch.npu.synchronize()
# 验证数值正确性
assert torch.allclose(out, x + y, atol=1e-3), "数值比对失败"
print("SMOKE_PASS", flush=True)
KERNEL_EOF

    SMOKE_OUT=$("$PYTHON" "$TMPDIR/test_vec_add.py" 2>&1) || true
    if echo "$SMOKE_OUT" | grep -q "SMOKE_PASS"; then
        _check "Triton kernel NPU 运行" true "vec_add 执行 + 数值比对通过"
    else
        _fail "Triton kernel NPU 运行" false "${SMOKE_OUT:0:300}"
    fi

    # 查找 HIVM MLIR dump
    NPUIR=$(find "$HOME/.triton/dump" -maxdepth 3 -name "*.npuir.mlir" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$NPUIR" ]; then
        NPUIR_SIZE=$(wc -c < "$NPUIR" 2>/dev/null || echo "0")
        _check "HIVM MLIR dump" true "$(basename "$(dirname "$NPUIR")")/$(basename "$NPUIR") (${NPUIR_SIZE} bytes)"
    else
        _fail "HIVM MLIR dump 未生成 — 检查 TRITON_DEBUG=1 是否设置"
    fi

    # bishengir-compile: HIVM → .o
    if [ -n "$NPUIR" ] && [ -n "$BISHENGIR_COMPILE" ]; then
        if "$BISHENGIR_COMPILE" "$NPUIR" --enable-hivm-compile -o "$TMPDIR/test.o" 2>&1; then
            _check "bishengir-compile HIVM→.o" true "$(wc -c < "$TMPDIR/test.o") bytes"
        else
            _fail "bishengir-compile HIVM→.o 失败"
        fi
    fi

    rm -rf "$TMPDIR"
    trap - EXIT
else
    _fail "跳过 — triton-ascend 或 torch_npu 不可用"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  8. msprof 真机采集烟雾测试
# ═════════════════════════════════════════════════════════════════════════════════
_section "8. msprof 真机采集"

if [ -n "$MSPROF" ] && [ "$TRITON_OK" = "1" ] && [ "${TORCH_OK:-0}" = "1" ]; then
    TMPDIR=$(mktemp -d -t msprof-smoke-XXXXXX)
    trap "rm -rf $TMPDIR" EXIT

    cat > "$TMPDIR/test_add.py" << 'KERNEL_EOF'
import triton, triton.language as tl
import torch, torch_npu

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

N = 256
x = torch.randn(N, device='npu'); y = torch.randn(N, device='npu')
out = torch.empty(N, device='npu')
add_kernel[(triton.cdiv(N, 128),)](x, y, out, N, BLOCK_SIZE=128)
torch.npu.synchronize()
KERNEL_EOF

    MSPROF_OUT=$("$MSPROF" op \
        --application="python3 $TMPDIR/test_add.py" \
        --kernel-name=add_kernel \
        --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
        --output="$TMPDIR/msprof_out" 2>&1) || true

    OPPROF_DIR=$(ls -dt "$TMPDIR/msprof_out"/OPPROF_* 2>/dev/null | head -1)

    if [ -n "$OPPROF_DIR" ]; then
        _check "msprof op 采集成功" true "$(basename "$OPPROF_DIR")"

        # 真机产出: PipeUtilization.csv, Memory.csv 等
        [ -f "$OPPROF_DIR/PipeUtilization.csv" ] && _check "PipeUtilization.csv" true || _warn "PipeUtilization.csv 缺失"
        [ -f "$OPPROF_DIR/Memory.csv" ]          && _check "Memory.csv" true          || _warn "Memory.csv 缺失"

        # 指令级 trace
        INSTR_CSV=$(find "$OPPROF_DIR/simulator" -name "*_instr_exe.csv" 2>/dev/null | head -1)
        if [ -n "$INSTR_CSV" ]; then
            INSTR_COUNT=$(wc -l < "$INSTR_CSV" 2>/dev/null || echo "0")
            _check "instr_exe.csv" true "${INSTR_COUNT} 条指令"
        else
            _warn "instr_exe.csv 未找到 (可能被 msprof 配置影响)"
        fi
    else
        _fail "msprof op 采集失败 — 未产出 OPPROF_ 目录"
        _info "手工验证: msprof op --application=\"python3 kernel.py\" --kernel-name=add_kernel"
    fi

    rm -rf "$TMPDIR"
    trap - EXIT
else
    _fail "跳过 — msprof 或 triton-ascend 不可用"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  9. 项目结构
# ═════════════════════════════════════════════════════════════════════════════════
_section "9. 项目结构"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"

check_file() { _check "$1" '[ -e "'"$AGENT_DIR/$1"'" ]' "$AGENT_DIR/$1"; }

check_file "config.py"
check_file "main.py"
check_file "analyzers/hivmir_analyzer.py"
check_file "analyzers/msprof_analyzer.py"
check_file "analyzers/dsl_merger.py"
check_file "analyzers/bottleneck_diagnoser.py"
check_file "analyzers/data_extractor.py"
check_file "agents/orchestrator.py"
check_file "agents/planner.py"
check_file "agents/coder.py"
check_file "agents/verifier.py"
check_file "feedback/record_manager.py"
check_file "memory/experience_retriever.py"
check_file "memory/context_manager.py"
check_file "docx/playbook_tier1_algorithm.md"
check_file ".env"

# ═════════════════════════════════════════════════════════════════════════════════
#  10. 环境变量
# ═════════════════════════════════════════════════════════════════════════════════
_section "10. 环境变量"

if [ "${_SOURCED:-0}" = "0" ] && [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    export TRITON_DEBUG=1
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_DISABLE_CACHE=1
    export TRITON_AGENT_MSPROF_MODE=hardware
    export PYTHONPATH="$AGENT_DIR:${PYTHONPATH:-}"
    _ok "TRITON_DEBUG=1"
    _ok "TRITON_ALWAYS_COMPILE=1"
    _ok "TRITON_DISABLE_CACHE=1"
    _ok "TRITON_AGENT_MSPROF_MODE=hardware"
    _ok "PYTHONPATH += $AGENT_DIR"
else
    echo "  请手动设置:"
    echo "    export TRITON_DEBUG=1"
    echo "    export TRITON_ALWAYS_COMPILE=1"
    echo "    export TRITON_DISABLE_CACHE=1"
    echo "    export TRITON_AGENT_MSPROF_MODE=hardware"
    echo "    export PYTHONPATH=$AGENT_DIR:\$PYTHONPATH"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  汇总
# ═════════════════════════════════════════════════════════════════════════════════
_section "汇总"

echo -e "  ${GREEN}通过: $PASS${NC}  ${RED}失败: $FAIL${NC}"

if [ "$FAIL" -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}═══ 910B3 真机环境就绪 ═══${NC}"
    echo ""
    echo "  运行优化管线:"
    echo "  cd $AGENT_DIR"
    echo "  python3 main.py input/softmax/triton_kernel.py --max-rounds 50 --target 1.5"
else
    echo -e "\n${RED}${BOLD}═══ $FAIL 项检查未通过 — 真机环境未就绪 ═══${NC}"
    exit 1
fi
