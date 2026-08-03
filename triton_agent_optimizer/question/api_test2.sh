#!/bin/bash
# === CANN 9.0 API 速测 v2 — 加入 InitSocState + 多种分配方式 ===
OUT=/tmp/api_test2_$$.log
exec > >(tee "$OUT") 2>&1
echo "=== CANN 9.0 AscendC API 速测 v2 ==="
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null

wcmake() {
  cat > CMakeLists.txt << 'CMEOF'
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "arch")
set(CMAKE_ASC_RUN_MODE "sim" CACHE STRING "mode")
find_package(ASC REQUIRED)
project(t LANGUAGES ASC CXX)
add_executable(app test.asc)
target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>)
target_link_libraries(app PRIVATE tiling_api platform)
CMEOF
}

# Test A: InitSocState + LocalMemAllocator
echo "=== A: InitSocState + LocalMemAllocator ==="
rm -rf /tmp/ta && mkdir /tmp/ta && cd /tmp/ta
cat > test.asc << 'EOF'
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void test(__gm__ float* x, __gm__ float* y, uint32_t n) {
    InitSocState();
    LocalMemAllocator<Hardware::UB> ub;
    LocalTensor<float> t1 = ub.Alloc<float, 256>();
    GlobalTensor<float> gx;
    gx.SetGlobalBuffer(x, 256);
    DataCopy(t1, gx, 256);
    PipeBarrier<PIPE_ALL>();
}
EOF
wcmake
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -3
make -j1 2>&1
[ -f app ] && echo ">>> PASS A" || echo ">>> FAIL A"

# Test B: GM_ADDR + InitSocState + LocalMemAllocator
echo "=== B: GM_ADDR + InitSocState ==="
rm -rf /tmp/tb && mkdir /tmp/tb && cd /tmp/tb
cat > test.asc << 'EOF'
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void test(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    InitSocState();
    LocalMemAllocator<Hardware::UB> ub;
    LocalTensor<float> t1 = ub.Alloc<float, 256>();
    GlobalTensor<float> gx;
    gx.SetGlobalBuffer((__gm__ float*)x, 256);
    DataCopy(t1, gx, 256);
    PipeBarrier<PIPE_ALL>();
}
EOF
wcmake
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -3
make -j1 2>&1
[ -f app ] && echo ">>> PASS B" || echo ">>> FAIL B"

# Test C: InitSocState + class-based + TPipe
echo "=== C: InitSocState + class + TPipe ==="
rm -rf /tmp/tc && mkdir /tmp/tc && cd /tmp/tc
cat > test.asc << 'EOF'
#include "kernel_operator.h"
using namespace AscendC;
class KernelTest {
public:
    __aicore__ void Init(__gm__ float* x, __gm__ float* y) {
        xGm.SetGlobalBuffer(x, 256);
        yGm.SetGlobalBuffer(y, 256);
    }
    __aicore__ void Process() {
        LocalTensor<float> t1 = Alloc<float, 256>();
        DataCopy(t1, xGm, 256);
        PipeBarrier<PIPE_ALL>();
    }
private:
    GlobalTensor<float> xGm, yGm;
};
extern "C" __global__ __aicore__ void test(__gm__ float* x, __gm__ float* y, uint32_t n) {
    InitSocState();
    KernelTest op;
    op.Init(x, y);
    op.Process();
}
EOF
wcmake
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -3
make -j1 2>&1
[ -f app ] && echo ">>> PASS C" || echo ">>> FAIL C"

# Test D: 最小 — 只有 InitSocState，看 bisheng 版本
echo "=== D: bisheng version check ==="
bisheng --version 2>&1 || true
echo ""
echo "=== 头文件检查 ==="
grep -r "Alloc" /usr/local/Ascend/cann-9.0.0/x86_64-linux/tikcpp/ascendc_kernel_cmake/../include/ 2>/dev/null | grep -i "template\|inline\|LocalTensor" | head -5 || true

echo ""
echo "=== 日志: $OUT ==="
