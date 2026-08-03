#!/bin/bash
# === CANN 9.0 AscendC API 速测 — 定位 Alloc 替代方案 ===
OUT=/tmp/api_test_$$.log
exec > >(tee "$OUT") 2>&1
echo "=== CANN 9.0 AscendC API 速测 ==="

source /usr/local/Ascend/cann/set_env.sh 2>/dev/null

# 准备 CMakeLists.txt
write_cmake() {
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

# 测试1: LocalMemAllocator (我们 v1.0 用过)
echo "=== Test 1: LocalMemAllocator ==="
rm -rf /tmp/t1 && mkdir /tmp/t1 && cd /tmp/t1
cat > test.asc << 'EOF'
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void test1(__gm__ float* x, __gm__ float* y, uint32_t n) {
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<float> t1 = ub.Alloc<float, 256>();
    AscendC::LocalTensor<float> t2 = ub.Alloc<float, 256>();
    AscendC::GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer(x, 256); gy.SetGlobalBuffer(y, 256);
    AscendC::DataCopy(t1, gx, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
    AscendC::Add(t2, t1, t1, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
    AscendC::DataCopy(gy, t2, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
}
EOF
write_cmake
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -3
make -j1 2>&1 | tail -5
[ -f app ] && echo ">>> PASS: LocalMemAllocator" || echo ">>> FAIL: LocalMemAllocator"

# 测试2: LocalTensor<T>::Alloc(N) 静态方法
echo ""
echo "=== Test 2: LocalTensor<T>::Alloc(N) ==="
rm -rf /tmp/t2 && mkdir /tmp/t2 && cd /tmp/t2
cat > test.asc << 'EOF'
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void test2(__gm__ float* x, __gm__ float* y, uint32_t n) {
    AscendC::LocalTensor<float> t1 = AscendC::LocalTensor<float>::Alloc(256);
    AscendC::LocalTensor<float> t2 = AscendC::LocalTensor<float>::Alloc(256);
    AscendC::GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer(x, 256); gy.SetGlobalBuffer(y, 256);
    AscendC::DataCopy(t1, gx, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
    AscendC::Add(t2, t1, t1, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
    AscendC::DataCopy(gy, t2, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
}
EOF
write_cmake
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -3
make -j1 2>&1 | tail -5
[ -f app ] && echo ">>> PASS: LocalTensor<T>::Alloc(N)" || echo ">>> FAIL: LocalTensor<T>::Alloc(N)"

# 测试3: using namespace + LocalMemAllocator
echo ""
echo "=== Test 3: using namespace AscendC + LocalMemAllocator ==="
rm -rf /tmp/t3 && mkdir /tmp/t3 && cd /tmp/t3
cat > test.asc << 'EOF'
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void test3(__gm__ float* x, __gm__ float* y, uint32_t n) {
    LocalMemAllocator<Hardware::UB> ub;
    LocalTensor<float> t1 = ub.Alloc<float, 256>();
    LocalTensor<float> t2 = ub.Alloc<float, 256>();
    GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer(x, 256); gy.SetGlobalBuffer(y, 256);
    DataCopy(t1, gx, 256);
    PipeBarrier<PIPE_ALL>();
    Add(t2, t1, t1, 256);
    PipeBarrier<PIPE_ALL>();
    DataCopy(gy, t2, 256);
    PipeBarrier<PIPE_ALL>();
}
EOF
write_cmake
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -3
make -j1 2>&1 | tail -5
[ -f app ] && echo ">>> PASS: using namespace + LocalMemAllocator" || echo ">>> FAIL: using namespace + LocalMemAllocator"

echo ""
echo "=== 完成, 日志: $OUT ==="
