#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → msprof op simulator → trace.json
#  自动尝试多种方案, 详细诊断每步失败原因
# ═════════════════════════════════════════════════════════════════════════════════
#  服务器: cd triton_agent_optimizer/input/softmax
#          bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
# ═════════════════════════════════════════════════════════════════════════════════

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG_DIR="$HERE/step1_logs"
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

# ═════════════════════════════════════════════════════════════════════════════════
#  辅助: 诊断输出
# ═════════════════════════════════════════════════════════════════════════════════
RED='\033[31m'; GRN='\033[32m'; YLW='\033[33m'; CYN='\033[36m'; NC='\033[0m'
_err()  { echo -e "${RED}[ERR]${NC} $*" | tee -a "$LOG_DIR/summary.log"; }
_wrn()  { echo -e "${YLW}[WARN]${NC} $*" | tee -a "$LOG_DIR/summary.log"; }
_ok()   { echo -e "${GRN}[OK]${NC}  $*" | tee -a "$LOG_DIR/summary.log"; }
_inf()  { echo -e "${CYN}[INFO]${NC} $*"; }
_hdr()  { echo ""; echo "═══════════════════════════════════════════════"; echo " $*"; echo "═══════════════════════════════════════════════"; }
_dump_log() { local log="$1" lines="${2:-20}"; echo "  --- $log (最后${lines}行) ---"; tail -"$lines" "$log" 2>/dev/null || echo "  (文件不存在)"; }
_check_file() { if [ -f "$1" ]; then _ok "$2 ($(wc -c < "$1") bytes)"; return 0; else _err "$2: 文件不存在 ($1)"; return 1; fi; }

# ═════════════════════════════════════════════════════════════════════════════════
#  0. 环境诊断
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "0. 环境诊断"

echo "时间: $(date)"
echo "主机: $(hostname)"
echo "架构: $(uname -m)"
echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '"' || echo '?')"
echo "SHELL: $SHELL"
echo "PWD: $PWD"

# CANN
CANN_HOME="${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME}}"
if [ -z "$CANN_HOME" ]; then
    for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann /usr/local/Ascend; do
        [ -d "$d" ] && { CANN_HOME="$d"; break; }
    done
fi
[ -n "$CANN_HOME" ] && _ok "CANN: $CANN_HOME" || _err "CANN: 未找到!"

# Python
PY=$(command -v python3 2>/dev/null || echo "")
[ -n "$PY" ] && _ok "python3: $($PY --version 2>&1)" || _err "python3: 未找到"

# Triton
$PY -c "import triton; print('triton:', getattr(triton, '__version__', '?'))" 2>/dev/null && _ok "triton OK" || _err "triton import 失败"
$PY -c "import torch_npu; print('torch_npu:', torch_npu.__version__); print('NPU count:', torch.npu.device_count())" 2>/dev/null && _ok "torch_npu OK" || _wrn "torch_npu import 失败"

# npu-smi
NPU_SMI=$(command -v npu-smi 2>/dev/null || echo "")
[ -n "$NPU_SMI" ] && _ok "npu-smi: $NPU_SMI" || _wrn "npu-smi: 未找到"
[ -n "$NPU_SMI" ] && $NPU_SMI info 2>/dev/null | head -5 || true

# bisheng
BISHENG=$(command -v bisheng 2>/dev/null || echo "")
[ -n "$BISHENG" ] && _ok "bisheng: $BISHENG" || _wrn "bisheng: 未找到"
BISHENGIR_COMPILE=$(command -v bishengir-compile 2>/dev/null || echo "")
[ -n "$BISHENGIR_COMPILE" ] && _ok "bishengir-compile: $BISHENGIR_COMPILE" || _wrn "bishengir-compile: 未找到"

# msprof
MSPROF=$(command -v msprof 2>/dev/null || echo "")
[ -n "$MSPROF" ] && _ok "msprof: $MSPROF" || _err "msprof: 未找到!"

# simulator lib
SIM_LIB=""; SIM_SO=""
for d in "$CANN_HOME/tools/simulator/Ascend910B3/lib" \
         "$CANN_HOME/tools/simulator/dav_2201/lib" \
         "$CANN_HOME/tools/simulator/Ascend910B1/lib" \
         "$CANN_HOME/tools/simulator/Ascend910B2/lib" \
         "$CANN_HOME/tools/simulator/Ascend910B4/lib"; do
    if [ -f "$d/libruntime_camodel.so" ]; then SIM_LIB="$d"; break; fi
done
if [ -z "$SIM_LIB" ]; then
    SIM_SO=$(find /usr/local/Ascend -maxdepth 6 -name "libruntime_camodel.so" -type f 2>/dev/null | head -1)
    [ -n "$SIM_SO" ] && SIM_LIB=$(dirname "$SIM_SO")
fi
[ -n "$SIM_LIB" ] && _ok "SIM lib: $SIM_LIB" || _err "SIM lib: 未找到 libruntime_camodel.so!"
[ -n "$SIM_LIB" ] && ls -la "$SIM_LIB"/lib*_camodel.so 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  公共: 清理 + 基础 env
# ═════════════════════════════════════════════════════════════════════════════════
rm -rf ~/.triton/cache/ ~/.triton/dump/ 2>/dev/null || true
rm -rf "$HERE/hivmir" "$HERE/msprof_sim" "$HERE/sim_build" 2>/dev/null || true
mkdir -p "$HERE/hivmir"

# ═════════════════════════════════════════════════════════════════════════════════
#  方案 A: LD_PRELOAD 注入 → msprof simulator 直接跑 Python 脚本
#  思路: 用 libruntime_camodel.so 拦截 triton-ascend 的 NPU 运行时调用
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "方案 A: LD_PRELOAD 注入 simulator libs"

if [ -z "$SIM_LIB" ]; then
    _err "方案 A 跳过: 未找到 simulator lib"
else
    export TRITON_DEBUG=1
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_DISABLE_LINE_INFO=0
    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$SIM_LIB/libruntime_camodel.so:$SIM_LIB/libnpu_drv_camodel.so"

    _inf "LD_LIBRARY_PATH 头部: $(echo $LD_LIBRARY_PATH | tr ':' '\n' | head -3 | tr '\n' ' ')"
    _inf "LD_PRELOAD=$LD_PRELOAD"

    # A1: 先验证 Python 脚本能跑
    _inf "A1: 运行 python3 run_and_profile.py (验证正确性)..."
    $PY run_and_profile.py > "$LOG_DIR/a1_run.log" 2>&1
    A1_RC=$?
    if [ "$A1_RC" = "0" ] && grep -q "PASS\|ALL.*PASSED" "$LOG_DIR/a1_run.log" 2>/dev/null; then
        _ok "A1: kernel 在 NPU 上运行成功 (退出码 0)"
        grep "max_err\|PASS\|FAIL" "$LOG_DIR/a1_run.log" 2>/dev/null | head -10
    else
        _err "A1: kernel 运行失败! 退出码=$A1_RC"
        _dump_log "$LOG_DIR/a1_run.log" 30
    fi

    # A2: 检查编译产物
    _inf "A2: 检查编译产物..."
    DUMP_FILES=$(find ~/.triton/dump -type f 2>/dev/null)
    CACHE_FILES=$(find ~/.triton/cache -type f 2>/dev/null)
    echo "  dump:  $(echo "$DUMP_FILES" | grep -c '.' 2>/dev/null || echo 0) files"
    echo "  cache: $(echo "$CACHE_FILES" | grep -c '.' 2>/dev/null || echo 0) files"

    # 列出 .o 文件
    ALL_O=$(find ~/.triton -name "*.o" -type f 2>/dev/null)
    O_COUNT=$(echo "$ALL_O" | grep -c '.' 2>/dev/null || echo 0)
    [ "$O_COUNT" -gt 0 ] && _ok "A2: 找到 $O_COUNT 个 .o 文件" || _wrn "A2: 未找到 .o 文件"
    echo "$ALL_O" | head -10 | while read f; do echo "    $(basename "$f") ($(wc -c < "$f") bytes)"; done

    # 列出 .mlir 文件
    ALL_MLIR=$(find ~/.triton -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" 2>/dev/null)
    MLIR_COUNT=$(echo "$ALL_MLIR" | grep -c '.' 2>/dev/null || echo 0)
    [ "$MLIR_COUNT" -gt 0 ] && _ok "A2: 找到 $MLIR_COUNT 个 MLIR 文件" || _wrn "A2: 未找到 MLIR 文件"

    if [ "$MLIR_COUNT" -gt 0 ]; then
        echo "$ALL_MLIR" | while read f; do cp "$f" "$HERE/hivmir/" 2>/dev/null; done
        _ok "A2: HIVM → hivmir/ ($(find "$HERE/hivmir" -type f | wc -l) files)"
    fi

    # A3: 用 .o 文件尝试 msprof simulator --config 模式
    if [ "$O_COUNT" -gt 0 ]; then
        _inf "A3: msprof simulator --config 模式..."

        FIRST_O=$(echo "$ALL_O" | head -1)
        cat > "$HERE/sim_config.json" << JSONEOF
{
    "kernel_name": "$KERNEL",
    "op_type": "AI_CORE",
    "kernel_file": "$FIRST_O"
}
JSONEOF
        _inf "config: $(cat $HERE/sim_config.json)"

        export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
        msprof op simulator \
            --config="$HERE/sim_config.json" \
            --output="$HERE/msprof_sim" > "$LOG_DIR/a3_msprof.log" 2>&1
        A3_RC=$?
        _inf "msprof 退出码: $A3_RC"

        OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
        if [ -n "$OPPROF" ]; then
            _ok "A3: OPPROF 目录生成: $(basename $OPPROF)"
            _inf "A3: OPPROF 目录树:"
            find "$OPPROF" -type f 2>/dev/null | head -30
            TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
            INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
            [ "$TRACES" -gt 0 ] && _ok "A3 成功! trace.json: $TRACES 个" || _wrn "A3: 有 OPPROF 但无 trace.json"
            [ "$INSTRS" -gt 0 ] && _ok "A3: instr_exe.csv: $INSTRS 个"
        else
            _err "A3: OPPROF 目录未生成"
            _dump_log "$LOG_DIR/a3_msprof.log" 30
            # 检查 msprof 日志中的关键错误
            _inf "A3: msprof 日志错误关键词:"
            grep -i "error\|fail\|ERROR\|FAIL\|not found\|cannot" "$LOG_DIR/a3_msprof.log" 2>/dev/null | head -10 || echo "  (未发现关键词)"
        fi
    fi

    # A4: LD_PRELOAD + msprof simulator 直接跑 Python
    _inf "A4: LD_PRELOAD + msprof simulator 直接跑 Python 脚本..."
    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$SIM_LIB/libruntime_camodel.so:$SIM_LIB/libnpu_drv_camodel.so"

    msprof op simulator \
        --kernel-name="$KERNEL" \
        --soc-version=Ascend910B3 \
        --output="$HERE/msprof_sim" \
        $PY run_and_profile.py > "$LOG_DIR/a4_msprof.log" 2>&1
    A4_RC=$?
    _inf "msprof 退出码: $A4_RC"

    OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
    if [ -n "$OPPROF" ]; then
        _ok "A4: OPPROF 目录生成: $(basename $OPPROF)"
        find "$OPPROF" -type f 2>/dev/null | head -30
        TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
        [ "$TRACES" -gt 0 ] && _ok "A4 成功! trace.json: $TRACES 个"
    else
        _err "A4: OPPROF 目录未生成"
        _dump_log "$LOG_DIR/a4_msprof.log" 30
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  方案 B: HIVM MLIR → AscendC .asc → bisheng 编译 → 链接 → msprof simulator
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "方案 B: HIVM MLIR → AscendC → bisheng → msprof simulator"

if [ -z "$SIM_LIB" ]; then
    _err "方案 B 跳过: 未找到 simulator lib"
elif [ -z "$BISHENG" ]; then
    _err "方案 B 跳过: 未找到 bisheng"
else
    # B1: 提取 HIVM MLIR
    _inf "B1: 提取 HIVM MLIR..."
    export TRITON_DEBUG=1
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_DISABLE_LINE_INFO=0

    rm -rf ~/.triton/dump/ 2>/dev/null || true
    $PY run_and_profile.py > "$LOG_DIR/b1_run.log" 2>&1
    B1_RC=$?
    echo "Triton 退出码: $B1_RC"

    DUMP_DIR=$(ls -dt ~/.triton/dump/*/ 2>/dev/null | head -1)
    [ -n "$DUMP_DIR" ] && _ok "B1: dump 目录: $DUMP_DIR" || _wrn "B1: dump 目录为空"

    # 找 MLIR 文件
    MLIR_FILES=$(find ~/.triton -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" 2>/dev/null)
    MLIR_CNT=$(echo "$MLIR_FILES" | grep -c '.' 2>/dev/null || echo 0)

    if [ "$MLIR_CNT" -gt 0 ]; then
        _ok "B1: 找到 $MLIR_CNT 个 MLIR 文件"
        echo "$MLIR_FILES" | while read f; do cp "$f" "$HERE/hivmir/" 2>/dev/null; done

        # B2: HIVM → AscendC
        _inf "B2: HIVM MLIR → AscendC 代码生成..."
        rm -rf "$HERE/sim_build" && mkdir -p "$HERE/sim_build"

        $PY -c "
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path('$HERE').parent.parent))

# 找第一个 MLIR 文件并解析
mlir_files = sorted(Path('$HERE/hivmir').glob('*.mlir')) + \
             sorted(Path('$HERE/hivmir').glob('*.ttir')) + \
             sorted(Path('$HERE/hivmir').glob('*.ttadapter'))

if not mlir_files:
    print('B2_ERROR: no MLIR files in hivmir/')
    sys.exit(1)

mlir_path = mlir_files[0]
print(f'B2: 解析 {mlir_path.name} ({mlir_path.stat().st_size} bytes)')

try:
    from analyzers.hivmir_analyzer import HIVMIRAnalyzer
    ha = HIVMIRAnalyzer()
    report = ha.analyze_file(mlir_path)
    data = ha.to_dict(report)
    ops = data.get('ops', [])
    print(f'B2: {len(ops)} ops:')

    from analyzers.hivm_to_ascendc import generate as gen_asc

    # 从 hivmir_analyzer 的 report 获取原始指令
    hivm_ops = []
    for op in ops:
        hivm_ops.append({
            'op_type': op.get('op_type', '?'),
            'dst': op.get('dst', ''),
            'src': op.get('src', ''),
            'src2': op.get('src2', ''),
            'size_kb': op.get('size_kb', 0),
            'memory_region': op.get('memory_region', ''),
            'dtype': op.get('dtype', 'f32'),
        })
        print(f'  op{op[\"op_id\"]} {op[\"op_type\"]:<12} {op.get(\"size_kb\",0):.2f}KB')

    asc_code = gen_asc('$KERNEL', hivm_ops, total_elems=1024, dtype='f32')
    Path('$HERE/sim_build/kernel.asc').write_text(asc_code)
    print(f'B2: AscendC → {len(asc_code)} chars')
except Exception as e:
    print(f'B2_ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" > "$LOG_DIR/b2_asc_gen.log" 2>&1
        B2_RC=$?

        if [ "$B2_RC" != "0" ]; then
            _err "B2: HIVM→AscendC 生成失败!"
            _dump_log "$LOG_DIR/b2_asc_gen.log" 30
        elif [ -f "$HERE/sim_build/kernel.asc" ]; then
            _ok "B2: kernel.asc ($(wc -c < "$HERE/sim_build/kernel.asc") chars)"
            _inf "B2: AscendC 代码前 30 行:"
            head -30 "$HERE/sim_build/kernel.asc"

            # B3: bisheng 编译 AscendC → .o
            _inf "B3: bisheng 编译 AscendC → .o (simulator mode)..."
            bisheng -c "$HERE/sim_build/kernel.asc" \
                -o "$HERE/sim_build/kernel.o" \
                --npu-arch=dav-2201 \
                --run-mode=sim \
                -DASCENDC_TRACE_ON \
                > "$LOG_DIR/b3_compile.log" 2>&1
            B3_RC=$?

            if [ "$B3_RC" != "0" ]; then
                _err "B3: bisheng 编译失败! 退出码=$B3_RC"
                _dump_log "$LOG_DIR/b3_compile.log" 30
                _inf "B3: 编译错误关键词:"
                grep -i "error\|undefined\|cannot find\|not found\|fatal" "$LOG_DIR/b3_compile.log" 2>/dev/null | head -10 || echo "  (未发现)"
            elif [ -f "$HERE/sim_build/kernel.o" ]; then
                _ok "B3: kernel.o ($(wc -c < "$HERE/sim_build/kernel.o") bytes)"

                # B4: 链接 → 可执行文件
                _inf "B4: bisheng 链接 simulator libs → 可执行文件..."
                bisheng -Wl,--disable-new-dtags \
                    -L"$SIM_LIB" \
                    -Wl,-rpath,"$SIM_LIB" \
                    -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++ \
                    -lascendcl -lascendc_runtime -lprofapi -lunified_dlog \
                    -lmmpa -lascend_dump -lc_sec -lerror_manager -lnpu_drv \
                    "$HERE/sim_build/kernel.o" \
                    -o "$HERE/sim_build/kernel_app" \
                    > "$LOG_DIR/b4_link.log" 2>&1
                B4_RC=$?

                if [ "$B4_RC" != "0" ]; then
                    _err "B4: 链接失败! 退出码=$B4_RC"
                    _dump_log "$LOG_DIR/b4_link.log" 30
                    _inf "B4: 链接错误关键词:"
                    grep -i "undefined reference\|cannot find\|no such file\|error" "$LOG_DIR/b4_link.log" 2>/dev/null | head -10 || echo "  (未发现)"
                    _inf "B4: 检查 SIM_LIB 下的 .so 文件:"
                    ls -la "$SIM_LIB"/*.so 2>/dev/null | head -20 || echo "  (无 .so 文件)"
                elif [ -f "$HERE/sim_build/kernel_app" ]; then
                    _ok "B4: kernel_app ($(wc -c < "$HERE/sim_build/kernel_app") bytes, file: $(file "$HERE/sim_build/kernel_app" 2>/dev/null || echo '?'))"

                    # B5: msprof op simulator
                    _inf "B5: msprof op simulator 采集..."
                    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
                    rm -rf "$HERE/msprof_sim"
                    msprof op simulator \
                        --soc-version=Ascend910B3 \
                        --output="$HERE/msprof_sim" \
                        "$HERE/sim_build/kernel_app" \
                        > "$LOG_DIR/b5_msprof.log" 2>&1
                    B5_RC=$?
                    _inf "msprof 退出码: $B5_RC"

                    OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
                    if [ -n "$OPPROF" ]; then
                        _ok "B5: OPPROF 目录: $(basename $OPPROF)"
                        _inf "B5: 完整目录树:"
                        find "$OPPROF" -type f 2>/dev/null | while read f; do
                            echo "  $f ($(wc -c < "$f") bytes)"
                        done
                        TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
                        INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
                        [ "$TRACES" -gt 0 ] && _ok "★★★ B5: trace.json $TRACES 个 ★★★"
                        [ "$INSTRS" -gt 0 ] && _ok "★★★ B5: instr_exe.csv $INSTRS 个 ★★★"
                    else
                        _err "B5: OPPROF 目录未生成"
                        _dump_log "$LOG_DIR/b5_msprof.log" 30
                        _inf "B5: msprof 错误关键词:"
                        grep -i "error\|fail\|not found\|cannot\|assertion\|UNKNOWN\|ERR" "$LOG_DIR/b5_msprof.log" 2>/dev/null | head -15 || echo "  (未发现)"
                    fi
                fi
            fi
        fi
    else
        _err "B1: 未找到 MLIR 文件"
        _inf "B1: Triton 运行日志:"
        _dump_log "$LOG_DIR/b1_run.log" 20
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  方案 C: 直接用 msprof op (真机模式) 作为最低保障
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "方案 C: msprof op 真机模式 (备选保障)"

MSPROF=$(command -v msprof 2>/dev/null || echo "")
if [ -n "$MSPROF" ]; then
    rm -rf "$HERE/msprof_hw" 2>/dev/null || true
    msprof op \
        --kernel-name="$KERNEL" \
        --output="$HERE/msprof_hw" \
        $PY run_and_profile.py > "$LOG_DIR/c_msprof.log" 2>&1
    C_RC=$?
    _inf "msprof op 退出码: $C_RC"

    OPPROF_HW=$(ls -dt "$HERE/msprof_hw"/OPPROF_*/ 2>/dev/null | head -1)
    if [ -n "$OPPROF_HW" ]; then
        _ok "方案 C: OPPROF $(basename $OPPROF_HW) ($(ls "$OPPROF_HW"/*.csv 2>/dev/null | wc -l) CSV, $(find "$OPPROF_HW" -name "trace.json" | wc -l) trace.json)"
        ls -la "$OPPROF_HW"/*.csv 2>/dev/null | while read f; do echo "  $(echo "$f" | awk '{print $NF}')"; done
    else
        _wrn "方案 C: OPPROF 未生成"
    fi
else
    _err "方案 C 跳过: msprof 未找到"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  最终汇总
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "最终汇总"

echo ""
echo "── 环境 ──"
echo "CANN:     ${CANN_HOME:-未找到}"
echo "SIM lib:  ${SIM_LIB:-未找到}"
echo "bisheng:  ${BISHENG:-未找到}"
echo "msprof:   ${MSPROF:-未找到}"
echo "python3:  ${PY:-未找到}"

echo ""
echo "── 产物 ──"
for d in hivmir msprof_sim msprof_hw sim_build; do
    cnt=$(find "$HERE/$d" -type f 2>/dev/null | wc -l)
    echo "  $d: $cnt files"
done
echo "  日志: $LOG_DIR ($(find "$LOG_DIR" -type f | wc -l) files)"

echo ""
echo "── trace.json 搜索结果 ──"
TRACES=$(find "$HERE" -name "trace.json" 2>/dev/null)
if [ -n "$TRACES" ]; then
    echo "$TRACES" | while read f; do
        echo "  ★ $f ($(wc -c < "$f") bytes)"
    done
else
    echo "  !! 未找到任何 trace.json"
fi

echo ""
echo "── 全部错误日志汇总 ──"
for log in "$LOG_DIR"/*.log; do
    [ -f "$log" ] || continue
    errs=$(grep -c -i "error\|fail\|ERROR\|FAIL\|cannot\|not found\|undefined\|fatal\|ERR\|assertion" "$log" 2>/dev/null || echo 0)
    if [ "$errs" -gt 0 ]; then
        echo "  $(basename $log): ${errs} 条疑似错误"
    fi
done

echo ""
echo "完整日志: ls -la $LOG_DIR/"
echo "按方案查看: cat $LOG_DIR/<方案>.log"
echo "汇总日志: cat $LOG_DIR/summary.log"
