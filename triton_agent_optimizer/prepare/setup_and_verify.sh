#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
# Triton Agent Optimizer — 910B3 真机环境搭建 + 编译链验证
# ═════════════════════════════════════════════════════════════════════════════════
#
# 用法:
#   source setup_and_verify.sh    # 设置环境变量 + 验证
#   bash   setup_and_verify.sh    # 仅验证
#
# 平台: Ascend 910B3 (aarch64) · Ubuntu 22.04 / openEuler
# 前置: 先 source CANN 的 set_env.sh, 再跑本脚本
# ═════════════════════════════════════════════════════════════════════════════════

# 注: 不使用 set -e, 容器环境难以排查静默退出
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

_ok()   { echo -e "  ${GREEN}[OK]${NC}    $*"; }
_warn() { echo -e "  ${YELLOW}[WARN]${NC}  $*"; }
_fail() { echo -e "  ${RED}[FAIL]${NC}  $*"; }
_info() { echo -e "  ${CYAN}[INFO]${NC}  $*"; }
_section() { echo -e "\n${BOLD}── $* ──${NC}"; }

PASS=0; FAIL=0; SKIP_SECTIONS=""

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

_skip_section() {
    # 标记后续某 section 整体跳过 (因为前置条件不满足)
    SKIP_SECTIONS="$SKIP_SECTIONS $1"
}

_is_skipped() { [[ " $SKIP_SECTIONS " == *" $1 "* ]]; }

# ═════════════════════════════════════════════════════════════════════════════════
#  1. 操作系统
# ═════════════════════════════════════════════════════════════════════════════════
_section "1. 操作系统"

OS_ID="$(grep '^ID=' /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"' || true)"
OS_VER="$(grep '^VERSION_ID=' /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"' || true)"
ARCH="$(uname -m)"

_check "OS" '[ -n "$OS_ID" ]' "${OS_ID:-?} ${OS_VER:-?}"
_check "架构" '[ "$ARCH" = "aarch64" ]' "$ARCH"

# ═════════════════════════════════════════════════════════════════════════════════
#  2. CANN Toolkit
# ═════════════════════════════════════════════════════════════════════════════════
_section "2. CANN Toolkit"

# 先尝试 source set_env.sh (如果用户还没 source 过)
for env_script in "${ASCEND_HOME:-}/set_env.sh" \
                  "${ASCEND_HOME:-}/ascend-toolkit/set_env.sh" \
                  "${ASCEND_HOME_PATH:-}/set_env.sh" \
                  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
                  /usr/local/Ascend/cann/set_env.sh \
                  /usr/local/Ascend/set_env.sh; do
    if [ -f "$env_script" ]; then
        _info "source $env_script"
        source "$env_script" 2>/dev/null || true
        break
    fi
done

CANN_DIR=""
for d in "${ASCEND_HOME:-}" "${ASCEND_HOME_PATH:-}" \
         /usr/local/Ascend/ascend-toolkit/latest \
         /usr/local/Ascend/cann \
         /usr/local/Ascend; do
    if [ -n "$d" ] && [ -d "$d" ]; then CANN_DIR="$d"; break; fi
done

_check "CANN 安装路径" '[ -n "$CANN_DIR" ]' "${CANN_DIR:-未找到}"

# 编译器 (无论 CANN 是否找到都继续检查)
BISHENGIR_COMPILE=$(command -v bishengir-compile 2>/dev/null || echo "")
_check "bishengir-compile" '[ -n "$BISHENGIR_COMPILE" ]' "${BISHENGIR_COMPILE:-未找到}"

BISHENGIR_OPT=$(command -v bishengir-opt 2>/dev/null || echo "")
_check "bishengir-opt" '[ -n "$BISHENGIR_OPT" ]' "${BISHENGIR_OPT:-未找到}"

# CANN 版本
if [ -n "$BISHENGIR_COMPILE" ]; then
    CANN_VER=$("$BISHENGIR_COMPILE" --version 2>&1 | head -1 || echo "unknown")
else
    CANN_VER="unknown"
fi
_check "CANN 版本" true "${CANN_VER}"

# msprof
MSPROF=$(command -v msprof 2>/dev/null || echo "")
_check "msprof" '[ -n "$MSPROF" ]' "${MSPROF:-未找到}"

# npu-smi
NPU_SMI=$(command -v npu-smi 2>/dev/null || echo "")
_check "npu-smi" '[ -n "$NPU_SMI" ]' "${NPU_SMI:-未找到}"

# ═════════════════════════════════════════════════════════════════════════════════
#  3. NPU 设备
# ═════════════════════════════════════════════════════════════════════════════════
_section "3. NPU 设备"

if [ -n "$NPU_SMI" ]; then
    NPU_INFO=$("$NPU_SMI" info 2>&1 || echo "")
    _check "npu-smi info 可用" '[ -n "$NPU_INFO" ]'

    NPU_CHIPS=$(echo "$NPU_INFO" | grep -ci "910B\|910\|950\|Ascend" || echo "0")
    _check "Ascend 芯片" '[ "$NPU_CHIPS" -gt 0 ]' "检测到 ${NPU_CHIPS} 个"

    echo "$NPU_INFO" | head -15 | while read -r line; do _info "$line" || true; done
else
    _fail "npu-smi 不可用 — NPU 驱动可能未加载，跳过后续 NPU 相关检查"
    _skip_section "pytorch"
    _skip_section "triton"
    _skip_section "smoke"
    _skip_section "msprof"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  4. Python 环境
# ═════════════════════════════════════════════════════════════════════════════════
_section "4. Python 环境"

PYTHON=$(command -v python3 2>/dev/null || echo "")
_check "python3" '[ -n "$PYTHON" ]' "$("$PYTHON" --version 2>&1 || true)"

if [ -z "$PYTHON" ]; then
    _skip_section "pytorch"
    _skip_section "triton"
    _skip_section "smoke"
    _skip_section "msprof"
else
    _check "numpy"      '"$PYTHON" -c "import numpy" 2>/dev/null'
    _check "openai"     '"$PYTHON" -c "import openai" 2>/dev/null' "LLM API 客户端"
    _check "json"       '"$PYTHON" -c "import json" 2>/dev/null'
    _check "pathlib"    '"$PYTHON" -c "from pathlib import Path" 2>/dev/null'
    _check "subprocess" '"$PYTHON" -c "import subprocess" 2>/dev/null'
    "$PYTHON" -c "import matplotlib" 2>/dev/null && _ok "matplotlib — trajectory chart" || _warn "matplotlib 未安装 (trajectory chart 不可用)"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  5. PyTorch + torch_npu
# ═════════════════════════════════════════════════════════════════════════════════
_section "5. PyTorch NPU"

TORCH_OK=0
if _is_skipped "pytorch"; then
    _warn "跳过 — 前置条件不满足"
elif "$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
    _check "torch" true "$TORCH_VER"

    if "$PYTHON" -c "import torch_npu; print(torch.npu.is_available())" 2>/dev/null; then
        NPU_COUNT=$("$PYTHON" -c "import torch_npu; print(torch.npu.device_count())")
        _check "torch_npu" true "NPU 设备数: $NPU_COUNT"
        TORCH_OK=1
    else
        _fail "torch_npu 不可用"
    fi
else
    _fail "torch 未安装"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  6. Triton-Ascend
# ═════════════════════════════════════════════════════════════════════════════════
_section "6. Triton-Ascend"

TRITON_OK=0
if _is_skipped "triton"; then
    _warn "跳过 — 前置条件不满足"
elif ! "$PYTHON" -c "import triton" 2>/dev/null; then
    _fail "triton 未安装 — 需要 triton-ascend (Ascend 后端版本)"
    _info "pip install triton-ascend"
else
    TRITON_VER=$("$PYTHON" -c "import triton; print(getattr(triton, '__version__', 'unknown'))")
    _check "triton 版本" true "$TRITON_VER"

    HAS_ASCEND=0
    if "$PYTHON" -c "from triton.backends.npu.compiler import NPUCompiler" 2>/dev/null; then
        _check "triton Ascend 后端" true "NPUCompiler 可用"
        HAS_ASCEND=1
    elif "$PYTHON" -c "import triton_ascend" 2>/dev/null; then
        _check "triton-ascend 包" true "triton_ascend 已安装"
        HAS_ASCEND=1
    else
        _fail "当前 triton 不是 Ascend 后端版本"
    fi

    if [ "$HAS_ASCEND" = "1" ] && [ "$TORCH_OK" = "1" ]; then
        if "$PYTHON" -c "
import torch, torch_npu, triton
x = torch.randn(4, device='npu')
import triton.runtime.driver as drv
drv.active.get_current_device()
print('driver_ok')
" 2>/dev/null; then
            _check "triton → NPU 连接" true "driver 可访问 NPU"
        else
            _warn "triton 可能无法访问 NPU driver"
        fi
    fi

    [ "$HAS_ASCEND" = "1" ] && TRITON_OK=1
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  7. 编译链烟雾测试
# ═════════════════════════════════════════════════════════════════════════════════
_section "7. 编译链烟雾测试"

if _is_skipped "smoke"; then
    _warn "跳过 — 前置条件不满足"
elif [ "$TRITON_OK" = "1" ] && [ "$TORCH_OK" = "1" ]; then
    TMPDIR=$(mktemp -d -t triton-smoke-XXXXXX)
    trap "rm -rf $TMPDIR" EXIT

    cat > "$TMPDIR/test_vec_add.py" << 'PYEOF'
import os
os.environ["TRITON_DEBUG"] = "1"
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
add_kernel[(triton.cdiv(N, 128),)](x, y, out, N, BLOCK_SIZE=128)
torch.npu.synchronize()
assert torch.allclose(out, x + y, atol=1e-3), "mismatch"
print("SMOKE_PASS", flush=True)
PYEOF

    SMOKE_OUT=$("$PYTHON" "$TMPDIR/test_vec_add.py" 2>&1) || true
    if echo "$SMOKE_OUT" | grep -q "SMOKE_PASS"; then
        _check "Triton kernel NPU 运行" true "vec_add 执行 + 数值比对通过"
    else
        _fail "Triton kernel NPU 运行" false "${SMOKE_OUT:0:300}"
    fi

    # HIVM MLIR dump
    NPUIR=$(find "$HOME/.triton/dump" -maxdepth 3 -name "*.npuir.mlir" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$NPUIR" ]; then
        _check "HIVM MLIR dump" true "$(basename "$(dirname "$NPUIR")")/$(basename "$NPUIR") ($(wc -c < "$NPUIR") bytes)"
    else
        _fail "HIVM MLIR dump 未生成"
    fi

    # bishengir-compile
    if [ -n "$NPUIR" ] && [ -n "$BISHENGIR_COMPILE" ]; then
        if "$BISHENGIR_COMPILE" "$NPUIR" --enable-hivm-compile -o "$TMPDIR/test.o" 2>/dev/null; then
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

if _is_skipped "msprof"; then
    _warn "跳过 — 前置条件不满足"
elif [ -n "$MSPROF" ] && [ "$TRITON_OK" = "1" ] && [ "$TORCH_OK" = "1" ]; then
    TMPDIR=$(mktemp -d -t msprof-smoke-XXXXXX)
    trap "rm -rf $TMPDIR" EXIT

    cat > "$TMPDIR/test_add.py" << 'PYEOF'
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
PYEOF

    "$MSPROF" op \
        --application="python3 $TMPDIR/test_add.py" \
        --kernel-name=add_kernel \
        --aic-metrics=PipeUtilization,ResourceConflictRatio,PMSampling \
        --output="$TMPDIR/msprof_out" 2>&1 || true

    OPPROF_DIR=$(ls -dt "$TMPDIR/msprof_out"/OPPROF_* 2>/dev/null | head -1)

    if [ -n "$OPPROF_DIR" ]; then
        _check "msprof op 采集成功" true "$(basename "$OPPROF_DIR")"
        [ -f "$OPPROF_DIR/PipeUtilization.csv" ] && _check "PipeUtilization.csv" true || _warn "PipeUtilization.csv 缺失"
        [ -f "$OPPROF_DIR/Memory.csv" ]          && _check "Memory.csv" true          || _warn "Memory.csv 缺失"

        INSTR_CSV=$(find "$OPPROF_DIR" -name "*_instr_exe.csv" 2>/dev/null | head -1)
        if [ -n "$INSTR_CSV" ]; then
            _check "instr_exe.csv" true "$(wc -l < "$INSTR_CSV") 条指令"
        else
            _warn "instr_exe.csv 未找到"
        fi
    else
        _fail "msprof op 采集未产出 OPPROF_ 目录"
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
check_file "analyzers/check_fields.py"
check_file "analyzers/integrate.py"
check_file "analyzers/pipeline_parse_task.py"
check_file "analyzers/pipeline_parse_board.py"
check_file "analyzers/pipeline_schema.py"
check_file "analyzers/merge_single_file.py"
check_file "analyzers/filter_hivm_for_fusion.py"
check_file "analyzers/run_hivm_fusion.py"
check_file "agents/planner.py"
check_file "agents/coder.py"
check_file "agents/verifier.py"
check_file "agents/llm_client.py"
check_file "agents/scheduler.py"
check_file "feedback/trajectory_chart.py"
check_file "docx/playbook_tier1_algorithm.md"
check_file ".env"

# ═════════════════════════════════════════════════════════════════════════════════
#  10. 环境变量
# ═════════════════════════════════════════════════════════════════════════════════
_section "10. 环境变量"

if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # 被 source 时自动设置
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
    _info "请手动设置环境变量:"
    echo ""
    echo "  export TRITON_DEBUG=1"
    echo "  export TRITON_ALWAYS_COMPILE=1"
    echo "  export TRITON_DISABLE_CACHE=1"
    echo "  export TRITON_AGENT_MSPROF_MODE=hardware"
    echo "  export PYTHONPATH=$AGENT_DIR:\$PYTHONPATH"
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
    echo "  python3 main.py input/softmax --max-rounds 50 --target 1.5"
else
    echo -e "\n${RED}${BOLD}═══ $FAIL 项检查未通过 — 真机环境未就绪 ═══${NC}"
fi
