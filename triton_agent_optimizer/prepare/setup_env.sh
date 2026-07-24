#!/bin/bash
# =============================================================================
# Triton Agent Optimizer — 环境设置 + 完整验证脚本
# =============================================================================
#  用法:
#    source prepare/setup_env.sh           # 设置环境
#    bash prepare/setup_env.sh --verify    # 只验证, 不设置
#    bash prepare/setup_env.sh --json      # JSON 格式输出验证结果
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$AGENT_DIR")"

echo "=============================================="
echo " Triton Agent Optimizer — Environment Setup"
echo " Project: $PROJECT_DIR"
echo "=============================================="

# ── 1. 检测运行环境 ──────────────────────────────────────────────────────

detect_environment() {
    if [ -n "$ASCEND_HOME" ] || [ -n "$ASCEND_HOME_PATH" ]; then
        ENV_TYPE="ascend_910b3"
    elif [ -d "/usr/local/Ascend" ]; then
        ENV_TYPE="ascend_910b3"
        export ASCEND_HOME="/usr/local/Ascend"
    elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
        ENV_TYPE="ascend_910b3"
        export ASCEND_HOME="/usr/local/Ascend"
        export ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
    else
        ENV_TYPE="local_dev"
    fi
    echo "  Environment: $ENV_TYPE"
}

# ── 2. CANN 环境设置 (仅 910B3) ──────────────────────────────────────────

setup_cann_env() {
    if [ "$ENV_TYPE" != "ascend_910b3" ]; then
        echo "  [SKIP] CANN — local dev, no Ascend hardware"
        return 0
    fi

    echo "  Setting up CANN environment..."

    # 查找 set_env.sh
    SET_ENV=""
    for candidate in \
        "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
        "${ASCEND_HOME}/ascend-toolkit/latest/set_env.sh" \
        "${ASCEND_HOME}/ascend-toolkit/set_env.sh" \
        "${ASCEND_HOME}/cann/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"; do
        if [ -f "$candidate" ]; then
            SET_ENV="$candidate"
            break
        fi
    done

    if [ -z "$SET_ENV" ]; then
        echo "  [WARN] set_env.sh not found. CANN may not be installed."
        echo "  Expected: /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
        return 1
    fi

    echo "  Source: $SET_ENV"
    source "$SET_ENV"

    # 设置常用环境变量
    export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-1}

    # 查找 toolkit 路径
    if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
        for candidate in \
            "${ASCEND_HOME}/ascend-toolkit/latest" \
            "${ASCEND_HOME}/cann"; do
            if [ -d "$candidate" ]; then
                export ASCEND_TOOLKIT_HOME="$candidate"
                break
            fi
        done
    fi

    # 添加工具路径到 PATH
    TOOLS_BIN="${ASCEND_TOOLKIT_HOME}/tools/profiler/bin"
    if [ -d "$TOOLS_BIN" ]; then
        export PATH="$TOOLS_BIN:$PATH"
    fi
    COMPILER_BIN="${ASCEND_TOOLKIT_HOME}/compiler/bin"
    if [ -d "$COMPILER_BIN" ]; then
        export PATH="$COMPILER_BIN:$PATH"
    fi

    # 添加库路径
    LIB64="${ASCEND_TOOLKIT_HOME}/lib64"
    if [ -d "$LIB64" ]; then
        export LD_LIBRARY_PATH="$LIB64:$LD_LIBRARY_PATH"
    fi

    echo "  ASCEND_TOOLKIT_HOME = ${ASCEND_TOOLKIT_HOME}"
    echo "  ASCEND_HOME         = ${ASCEND_HOME}"
    echo "  CANN environment set."
}

# ── 3. Python 虚拟环境 ──────────────────────────────────────────────────

setup_python_env() {
    echo "  Setting up Python environment..."

    # 检查 Python 版本
    PYTHON=""
    for candidate in python3.10 python3.11 python3.12 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done

    if [ -z "$PYTHON" ]; then
        echo "  [ERROR] Python not found"
        return 1
    fi

    PY_VER=$("$PYTHON" --version 2>&1)
    echo "  Using: $PY_VER ($(which "$PYTHON"))"

    # 检查是否有 .venv
    VENV_DIR="$PROJECT_DIR/.venv"
    if [ -d "$VENV_DIR" ]; then
        echo "  Virtual env found: $VENV_DIR"
        source "$VENV_DIR/bin/activate" 2>/dev/null || true
    elif command -v conda &>/dev/null; then
        # 尝试 conda 环境
        CONDA_ENV="OP_autogen_hjkc"
        if conda env list 2>/dev/null | grep -q "$CONDA_ENV"; then
            echo "  Conda env found: $CONDA_ENV"
            conda activate "$CONDA_ENV" 2>/dev/null || true
        else
            echo "  Creating conda env: $CONDA_ENV (Python 3.12)..."
            conda create -n "$CONDA_ENV" python=3.12 -y
            conda activate "$CONDA_ENV" 2>/dev/null || true
        fi
    fi

    # 安装必要包
    echo "  Checking required packages..."
    pip install -q numpy 2>/dev/null || true

    # 可选包 (静默安装, 已有则跳过)
    pip install -q matplotlib networkx 2>/dev/null || true

    # Triton 环境 (仅 910B3)
    if [ "$ENV_TYPE" = "ascend_910b3" ]; then
        echo "  Triton/Ascend packages (checking)..."
        python -c "import torch; import torch_npu; import triton" 2>/dev/null && \
            echo "  Triton OK" || \
            echo "  [WARN] triton/torch_npu not importable — install manually"
    fi

    echo "  Python environment ready."
}

# ── 4. 验证 ─────────────────────────────────────────────────────────────

verify() {
    echo ""
    echo "=============================================="
    echo " Verification"
    echo "=============================================="

    PYTHON="${1:-python}"
    cd "$AGENT_DIR"

    if [ -f "prepare/env_check.py" ]; then
        "$PYTHON" prepare/env_check.py "$@"
    else
        echo "[WARN] env_check.py not found"
    fi
}

# ── 5. 打印环境摘要 ─────────────────────────────────────────────────────

print_summary() {
    echo ""
    echo "=============================================="
    echo " Environment Summary"
    echo "=============================================="
    echo "  Type:       $ENV_TYPE"
    echo "  Project:    $PROJECT_DIR"
    echo "  Agent:      $AGENT_DIR"
    echo "  Python:     $(python --version 2>&1 || echo 'not found')"
    echo "  numpy:      $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'no')"
    echo "  msprof:     $(which msprof 2>/dev/null || echo 'not found')"
    echo "  npu-smi:    $(which npu-smi 2>/dev/null || echo 'not found')"
    echo ""
    echo "  Ready to run:"
    echo "    cd triton_agent_optimizer"
    echo "    python agents/orchestrator.py"
    echo "=============================================="
}

# ── 主流程 ───────────────────────────────────────────────────────────────

main() {
    detect_environment
    setup_cann_env
    setup_python_env

    case "${1:-}" in
        --verify)
            verify "$@"
            ;;
        --json)
            verify "--json"
            ;;
        *)
            print_summary
            verify
            ;;
    esac
}

main "$@"
