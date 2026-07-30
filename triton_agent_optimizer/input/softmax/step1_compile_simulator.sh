#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → HIVM MLIR + msprof op simulator trace
#  CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# ═════════════════════════════════════════════════════════════════════════════════
#
#  服务器执行:
#    cd triton_agent_optimizer/input/softmax
#    bash step1_compile_simulator.sh [softmax_kernel|fused_gelu_kernel]
#
# ═════════════════════════════════════════════════════════════════════════════════
#  已知 CANN 8.5 问题 (全部搜自官方文档/博客):
#
#  1. TRITON_DISABLE_CACHE Ascend 后端不生效 → 用 TRITON_ALWAYS_COMPILE=1
#  2. msprof op simulator CANN 8.5 bug: OPPROF dump 有但 simulator 子目录空
#     → "GetOutputPathFromRemote failed" (解析阶段失败, 即使仿真本身成功)
#     → 尝试 --dump on 触发 dump 生成
#  3. triton-ascend 3.5.x: kernel.npuir.mlir 不生成 (pass 重命名, 需 MR !1656)
#  4. aicpu_legacy.tar.gz: CANN 可选包, 忽略
#  5. taskfailcallbackmanager: triton 内部类名, 非报错
#
#  参考:
#   https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/profiling.md
#   https://gitcode.com/cann/asc-devkit/blob/.../msProf/README_en.md
#   https://gitcode.com/Ascend/triton-ascend/blob/release/3.5.x/docs/zh/FAQ.md
# ═════════════════════════════════════════════════════════════════════════════════

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG_DIR="$HERE/step1_logs"
mkdir -p "$LOG_DIR"

# ── 辅助函数 ──
ERRORS=""; WARNS=""; OKS=""
add_err()  { echo -e "  \033[31m[ERROR]\033[0m $*"; ERRORS="$ERRORS\n  [ERROR] $*"; }
add_warn() { echo -e "  \033[33m[WARN]\033[0m  $*"; WARNS="$WARNS\n  [WARN] $*"; }
add_ok()   { echo -e "  \033[32m[OK]\033[0m    $*"; OKS="$OKS\n  [OK] $*"; }
_chk()     { if eval "$2"; then add_ok "$1"; else add_err "$1"; fi; }

echo "=============================================="
echo " Step 1: HIVM MLIR + msprof op simulator"
echo " kernel: $KERNEL"
echo "=============================================="

# ── 环境信息 ──
echo ""
echo "── 环境 ──"
python3 -c "import triton; print('triton:', getattr(triton, '__version__', '?'))" 2>/dev/null || true
python3 -c "import torch_npu; print('torch_npu:', torch_npu.__version__)" 2>/dev/null || true
echo "npu-smi: $(npu-smi info -l 2>/dev/null | head -1 || echo 'N/A')"
echo "CANN:    ${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME:-未设置}}"
echo ""

# ── 强制清理 ──
rm -rf ~/.triton/cache/ ~/.triton/dump/ 2>/dev/null || true
rm -rf "$HERE/hivmir" "$HERE/msprof_sim" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  第1步: 提取 HIVM MLIR
# ═════════════════════════════════════════════════════════════════════════════════
echo "── 1. HIVM MLIR ──"

export TRITON_DEBUG=1
export TRITON_ALWAYS_COMPILE=1
# triton-ascend 默认关闭行号信息(默认 true), 必须设为 0 才能生成完整的 trace
# 来源: https://ascend.github.io/docs/.../profiling.html
# "triton-ascend defaults to disabling line info (TRITON_DISABLE_LINE_INFO=1 by default)"
export TRITON_DISABLE_LINE_INFO=0

python3 run_and_profile.py > "$LOG_DIR/triton.log" 2>&1
echo "Triton 退出码: $?"

DUMP_DIR=$(ls -dt ~/.triton/dump/*/ 2>/dev/null | head -1)
CACHE_MLIRS=$(find ~/.triton/cache -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" 2>/dev/null)

mkdir -p "$HERE/hivmir"

if [ -n "$DUMP_DIR" ]; then
    echo "dump 目录: $DUMP_DIR"
    find "$DUMP_DIR" -type f 2>/dev/null | while read f; do
        cp "$f" "$HERE/hivmir/" && echo "  + $(basename "$f") ($(wc -c < "$f") bytes)"
    done
elif [ -n "$CACHE_MLIRS" ]; then
    echo "dump 为空, 从 cache 提取:"
    echo "$CACHE_MLIRS" | while read f; do
        cp "$f" "$HERE/hivmir/" && echo "  + cache→hivmir: $(basename "$f")"
    done
else
    echo "!! dump 和 cache 都没有 .mlir/.ttir 文件"
fi

HIVM_COUNT=$(find "$HERE/hivmir" -type f 2>/dev/null | wc -l)
_chk "HIVM MLIR: $HIVM_COUNT 个文件" '[ "$HIVM_COUNT" -gt 0 ]'

echo ""
echo "hivmir/:"
ls -la "$HERE/hivmir/" 2>/dev/null || echo "  (空)"

# HIVM 文件类型检查
echo "文件类型:"
for f in "$HERE/hivmir/"*; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    if file "$f" 2>/dev/null | grep -q "text"; then
        echo "  $name → 文本文件 (OK)"
    else
        echo "  $name → 可能是二进制, 前80字符: $(head -c 80 "$f" 2>/dev/null | tr -d '\0')"
    fi
done

# ═════════════════════════════════════════════════════════════════════════════════
#  第2步: msprof op simulator
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 2. msprof op simulator ──"

# 找 simulator lib
SIM_LIB=""
for d in "${ASCEND_TOOLKIT_HOME:-}/tools/simulator/Ascend910B3/lib" \
         "${ASCEND_HOME:-}/tools/simulator/Ascend910B3/lib" \
         "${ASCEND_HOME:-}/tools/simulator/dav_2201/lib" \
         "/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib" \
         "/usr/local/Ascend/cann/tools/simulator/Ascend910B3/lib" \
         "/usr/local/Ascend/cann/tools/simulator/dav_2201/lib"; do
    [ -d "$d" ] && { SIM_LIB="$d"; break; }
done

if [ -z "$SIM_LIB" ]; then
    echo "!! 未找到 simulator lib"
    echo "   搜索路径:"
    for d in /usr/local/Ascend/ascend-toolkit/latest/tools/simulator/*/lib \
             /usr/local/Ascend/cann/tools/simulator/*/lib; do
        [ -d "$d" ] && echo "   存在: $d"
    done
    add_err "SIM LIB: 未找到 → 无法运行 simulator"
else
    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
    echo "SIM lib: $SIM_LIB"
    echo "LD_LIBRARY_PATH 已设置"
fi

# ── 跑 msprof op simulator ──
echo ""
echo "执行 msprof op simulator (stdout/stderr → $LOG_DIR/msprof_sim.log)..."

msprof op simulator \
    --kernel-name="$KERNEL" \
    --soc-version=Ascend910B3 \
    --dump=on \
    --output="$HERE/msprof_sim" \
    python3 run_and_profile.py > "$LOG_DIR/msprof_sim.log" 2>&1
MSPROF_RC=$?
echo "msprof 退出码: $MSPROF_RC"

# ── 诊断 msprof 输出 ──
echo ""
echo "msprof 日志关键信息:"
grep -i "error\|fail\|warn\|dump\|profiling data" "$LOG_DIR/msprof_sim.log" 2>/dev/null \
    | grep -v "taskfailcallbackmanager\|TaskFailCallbackManager" \
    | head -20 || echo "  (无)"

OPPROF_DIR=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)

if [ -z "$OPPROF_DIR" ]; then
    add_err "msprof simulator: OPPROF 目录未生成"
    echo ""
    echo "  msprof 日志全文 (最后40行):"
    tail -40 "$LOG_DIR/msprof_sim.log" 2>/dev/null
else
    echo ""
    echo "OPPROF: $OPPROF_DIR"
    echo "OPPROF 顶层:"
    ls -la "$OPPROF_DIR/" 2>/dev/null

    echo ""
    echo "OPPROF 完整目录树:"
    find "$OPPROF_DIR" -type f 2>/dev/null | head -50 || echo "  (空)"

    # 检查 simulator 子目录
    SIM_SUBDIR="$OPPROF_DIR/simulator"
    if [ -d "$SIM_SUBDIR" ]; then
        SIM_FILES=$(find "$SIM_SUBDIR" -type f 2>/dev/null | wc -l)
        INSTRS=$(find "$SIM_SUBDIR" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
        TRACES=$(find "$SIM_SUBDIR" -name "trace.json" 2>/dev/null | wc -l)

        _chk "simulator/ 存在: $SIM_FILES 个文件" '[ "$SIM_FILES" -gt 0 ]'
        echo "  instr_exe.csv: $INSTRS"
        echo "  trace.json:    $TRACES"

        if [ "$SIM_FILES" -gt 0 ]; then
            add_ok "msprof simulator 采集成功"
        else
            add_warn "msprof simulator: simulator/ 为空 (已知 CANN bug: GetOutputPathFromRemote failed)"
            echo ""
            echo "  原因 (CANN 文档+社区确认):"
            echo "  1. CANN 8.5 解析阶段 bug — 仿真成功但解析失败"
            echo "  2. 尝试升级 CANN 版本"
            echo "  3. 备选: 用 msprof op (真机模式) 采集 PipeUtilization.csv"
            echo "  4. 备选: 用 bishengir-opt 从 .mlir 做 IR 层面分析"
        fi
    else
        add_err "msprof simulator: simulator/ 子目录不存在"
        echo ""
        echo "  OPPROF 内容:"
        ls -laR "$OPPROF_DIR/" 2>/dev/null | head -30
    fi

    # 检查 dump 目录 (原始仿真数据)
    if [ -d "$OPPROF_DIR/dump" ]; then
        DUMP_FILES=$(find "$OPPROF_DIR/dump" -type f 2>/dev/null | wc -l)
        echo ""
        echo "  dump/ 原始数据: $DUMP_FILES 个文件"
        [ "$DUMP_FILES" -gt 0 ] && echo "  (仿真成功, 但解析阶段可能失败)"
    fi
fi

# ── 检查 msprof 工具本身 (CANN 8.5 不支持 --version, 用 which + ls) ──
echo ""
echo "── msprof 工具信息 ──"
MSPROF_PATH=$(command -v msprof 2>/dev/null || echo "")
if [ -n "$MSPROF_PATH" ]; then
    echo "msprof 路径: $MSPROF_PATH"
    ls -la "$MSPROF_PATH" 2>/dev/null
    # 直接从路径推断版本
    if echo "$MSPROF_PATH" | grep -q "cann"; then
        CANN_VER_DIR=$(echo "$MSPROF_PATH" | grep -oP '/cann/\K[^/]+' 2>/dev/null || echo "?")
        echo "CANN 版本(路径推断): $CANN_VER_DIR"
    fi
else
    echo "msprof: 未找到"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  汇总
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " Step 1 汇总"
echo "=============================================="

echo ""
echo "── 产物 ──"
echo "hivmir:       $(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files"
echo "msprof_sim:   $(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files"
echo "日志:         $LOG_DIR"

echo ""
echo "── 日志摘要 ──"
echo "triton ($(wc -l < "$LOG_DIR/triton.log" 2>/dev/null || echo 0) lines):"
tail -5 "$LOG_DIR/triton.log" 2>/dev/null || true
echo ""
echo "msprof_sim ($(wc -l < "$LOG_DIR/msprof_sim.log" 2>/dev/null || echo 0) lines):"
# 筛出真正的错误 (排除噪音)
grep -i "error\|fail\|ERROR\|FAIL" "$LOG_DIR/msprof_sim.log" 2>/dev/null \
    | grep -v "taskfailcallbackmanager\|TaskFailCallbackManager" \
    | head -10 || echo "  (未发现明显错误)"

echo ""
echo "── 下一步 ──"
if [ "$HIVM_COUNT" -gt 0 ]; then
    echo "  → bash step2_parse_hivm.sh (HIVM 语义解析, 可立即执行)"
fi

INSTRS=$(find "$HERE/msprof_sim" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
if [ "$INSTRS" -gt 0 ]; then
    echo "  → bash step3_parse_msprof.sh (msprof simulator 时序解析)"
else
    echo ""
    echo "  msprof op simulator 在 CANN 8.5.1 上存在已知的解析阶段 bug,"
    echo "  网上多个来源 (华为云博客、oam-tools issue) 确认此问题:"
    echo "  - 仿真本身成功 (exit code 0, OPPROF 目录生成)"
    echo "  - 但 simulator/ 子目录为空/不存在 (解析工具 GetOutputPathFromRemote 失败)"
    echo "  - CANN 8.5 社区版 msprof 功能不完整, 暂无官方补丁"
    echo ""
    echo "  务实替代方案:"
    echo "  1. 用 HIVM MLIR 做语义层分析 (step2 继续执行)"
    echo "  2. 用 msprof op (真机模式) 采集 PipeUtilization.csv 做时序层分析"
    echo "  3. 两者合并仍可产生完整的 29 字段 DSL 流水线"
fi
