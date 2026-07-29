#!/bin/bash
# ==============================================================================
#  WSL2 HIVMIR 环境安装脚本 — 在 WSL Ubuntu 内执行
# ==============================================================================
# 用法:
#   1. 打开 Windows Terminal / PowerShell，输入:
#        wsl -d Ubuntu
#   2. 在 WSL 终端内执行:
#        bash /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/wsl_hivmir_setup/install_cann.sh
# ==============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

WORKDIR="$HOME/cann_setup"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

# ============================== Step 1: Python 环境 ==============================
log "Step 1/5: 配置 Python 3.9..."
if ! python3.9 --version &>/dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.9 python3.9-dev python3.9-venv python3.9-distutils
fi

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
sudo update-alternatives --set python3 /usr/bin/python3.9

log "安装 pip 包..."
pip3 install --upgrade pip -q 2>/dev/null || true
pip3 install numpy attrs decorator sympy cffi pyyaml requests 2>&1 | tail -3
python3.9 -c "import numpy; print('  numpy:', numpy.__version__)"
log "Python 环境就绪"

# ============================== Step 2: 下载 CANN ==============================
log "Step 2/5: 下载 CANN Toolkit..."

CANN_RUNFILE=$(ls Ascend-cann-toolkit_*.run 2>/dev/null | head -1)
if [ -n "$CANN_RUNFILE" ]; then
    log "发现已下载的 CANN 包: $CANN_RUNFILE"
else
    warn ""
    warn "============================================================"
    warn "  CANN 需要手动下载 (需登录昇腾社区)"
    warn ""
    warn "  1. 浏览器打开:"
    warn "     https://www.hiascend.com/developer/download/community/result?module=cann"
    warn ""
    warn "  2. 选择: Linux + x86_64 + 最新社区版 + Toolkit"
    warn ""
    warn "  3. 下载 .run 文件后放到:"
    warn "     \\\\wsl.localhost\\Ubuntu\\home\\<用户名>\\cann_setup\\"
    warn ""
    warn "  4. 然后重新执行本脚本继续后续步骤"
    warn "============================================================"
    warn ""
    echo "按 Enter 跳过 CANN 安装 (仅完成 Python 环境配置), 或 Ctrl+C 退出..."
    read -r
    log "跳过 CANN 安装。Python 环境配置完成。"
    log "在 WSL 中运行以下命令验证 Python:"
    log "  python3 --version    # 应显示 Python 3.9.x"
    log "  python3 -c 'import numpy; print(numpy.__version__)'"
    exit 0
fi

# ============================== Step 3: 安装 CANN ==============================
log "Step 3/5: 安装 CANN Toolkit..."
chmod +x "$CANN_RUNFILE"

# 安装到用户目录（不需要 root）
INSTALL_PATH="$HOME/Ascend"
./"$CANN_RUNFILE" --full --install-path="$INSTALL_PATH" --quiet 2>&1 | tail -10 || {
    err "CANN 安装失败。检查日志或尝试手动安装:"
    err "  ./$CANN_RUNFILE --full --install-path=$INSTALL_PATH"
    exit 1
}

log "CANN 安装完成: $INSTALL_PATH"

# ============================== Step 4: 配置环境变量 ==============================
log "Step 4/5: 配置环境变量..."

# 找实际 latest 目录
ASCEND_LATEST=$(ls -d "$INSTALL_PATH"/ascend-toolkit/latest 2>/dev/null || ls -d "$INSTALL_PATH"/latest 2>/dev/null || echo "$INSTALL_PATH")

if [ ! -d "$ASCEND_LATEST" ]; then
    err "找不到 ascend-toolkit 目录。检查安装是否成功。"
    ls -la "$INSTALL_PATH/"
    exit 1
fi

# 写入 ~/.bashrc
cat >> ~/.bashrc << 'CANNEVAR'
# === CANN + Bisheng 环境变量 (auto-generated) ===
export INSTALL_PATH=$HOME/Ascend
export ASCEND_HOME=$(ls -d $INSTALL_PATH/ascend-toolkit/latest 2>/dev/null || echo $INSTALL_PATH)
export PATH=$ASCEND_HOME/tools/bishengir:$ASCEND_HOME/compiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=$ASCEND_HOME/opp
export ASCEND_GLOBAL_LOG_LEVEL=3
CANNEVAR

source ~/.bashrc 2>/dev/null || true

# ============================== Step 5: 验证工具链 ==============================
log "Step 5/5: 验证工具链..."

# 找 bishengir 工具
BISHENGIR_DIR="$ASCEND_LATEST/tools/bishengir"
if [ -d "$BISHENGIR_DIR" ]; then
    export PATH="$BISHENGIR_DIR:$PATH"
    log "bishengir 工具目录: $BISHENGIR_DIR"
    ls -la "$BISHENGIR_DIR"/ 2>/dev/null | grep -E "bishengir-compile|bishengir-opt"
else
    warn "未找到 bishengir 目录，尝试搜索..."
    find "$INSTALL_PATH" -name "bishengir-compile" -type f 2>/dev/null | head -5
fi

# 验证 bishengir-compile
if command -v bishengir-compile &>/dev/null; then
    log "✅ bishengir-compile 可用: $(which bishengir-compile)"
    bishengir-compile --help 2>&1 | head -5
elif [ -f "$BISHENGIR_DIR/bishengir-compile" ]; then
    log "✅ bishengir-compile 找到: $BISHENGIR_DIR/bishengir-compile"
    "$BISHENGIR_DIR/bishengir-compile" --help 2>&1 | head -5
else
    warn "⚠️ bishengir-compile 未自动找到。"
    warn "   尝试手动: find $INSTALL_PATH -name 'bishengir-compile'"
fi

if command -v bishengir-opt &>/dev/null; then
    log "✅ bishengir-opt 可用: $(which bishengir-opt)"
else
    warn "⚠️ bishengir-opt 未自动找到。"
fi

log ""
log "============================================================"
log "  ✅ 环境安装完成！"
log ""
log "  使环境变量生效: source ~/.bashrc"
log "  验证: bishengir-compile --version"
log "  编译测试: bishengir-compile test_vec_add.mlir -o vec_add.o"
log ""
log "  测试 MLIR 文件在 (Windows 路径):"
log "    D:\\vscodeproject\\huawei_work\\OP_autogen\\OP_autogen_hjkc\\wsl_hivmir_setup\\test_vec_add.mlir"
log "  对应 WSL 路径:"
log "    /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/wsl_hivmir_setup/test_vec_add.mlir"
log "============================================================"
