#!/bin/bash
# ============================================================================
# AscendC 编译诊断脚本 — 定位 bisheng 拒绝 Alloc 的原因
# 用法: 在 WSL2 中执行 bash /mnt/d/vscodeproject/.../question/diagnose_ascendc.sh
# ============================================================================
set -e
OUT=/tmp/ascendc_diag_$$.log
exec > >(tee "$OUT") 2>&1
echo "=== AscendC 编译诊断 ==="
echo "时间: $(date)"
echo "输出: $OUT"
echo ""

# ── Step 0: 环境检测 ──
echo "=== Step 0: 环境检测 ==="
CANN_SETUP=""
for f in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
  if [ -f "$f" ]; then CANN_SETUP="$f"; break; fi
done
if [ -z "$CANN_SETUP" ]; then echo "ERROR: CANN set_env.sh not found!"; exit 1; fi
echo "CANN setup: $CANN_SETUP"
source "$CANN_SETUP" 2>/dev/null || true
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-NOT_SET}"
echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-NOT_SET}"
echo "bisheng: $(which bisheng 2>/dev/null || echo NOT_FOUND)"
echo "msprof:  $(which msprof 2>/dev/null || echo NOT_FOUND)"
echo ""

# ── Step 1: 确认工作参考代码能否编译 (kernel_vecadd.asc) ──
echo "=== Step 1: 编译工作参考 kernel_vecadd.asc ==="
REF_DIR=/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/msprof_simulator_test
WORKDIR=/tmp/ascendc_test1
rm -rf "$WORKDIR" && mkdir -p "$WORKDIR" && cd "$WORKDIR"
cp "$REF_DIR/kernel_vecadd.asc" "$REF_DIR/host_main.cpp" ./
cat > CMakeLists.txt << 'ENDOFFILE'
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "arch")
set(CMAKE_ASC_RUN_MODE "sim" CACHE STRING "mode")
find_package(ASC REQUIRED)
project(ref_test LANGUAGES ASC CXX)
add_executable(ref_app kernel_vecadd.asc host_main.cpp)
target_compile_options(ref_app PRIVATE $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>)
target_compile_options(ref_app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
target_link_libraries(ref_app PRIVATE tiling_api platform)
ENDOFFILE
echo "cmake..."
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -5
echo "make..."
make -j1 2>&1
if [ -f ref_app ]; then
  echo "RESULT: kernel_vecadd.asc BUILD PASS"
else
  echo "RESULT: kernel_vecadd.asc BUILD FAIL (这不应该 — 之前是成功的)"
fi
echo ""

# ── Step 2: 最小 test_alloc.asc (验证 Alloc 语法) ──
echo "=== Step 2: 最小 Alloc 语法测试 ==="
WORKDIR2=/tmp/ascendc_test2
rm -rf "$WORKDIR2" && mkdir -p "$WORKDIR2" && cd "$WORKDIR2"
cat > test_alloc.asc << 'ENDOFFILE'
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void test_alloc(__gm__ float* x, __gm__ float* y, uint32_t n) {
    LocalTensor<float> t = Alloc<float, 256>();
    GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer(x, 256);
    gy.SetGlobalBuffer(y, 256);
    DataCopy(t, gx, 256);
    PipeBarrier<PIPE_ALL>();
    Add(t, t, t, 256);
    PipeBarrier<PIPE_ALL>();
    DataCopy(gy, t, 256);
    PipeBarrier<PIPE_ALL>();
}
ENDOFFILE
cat > CMakeLists.txt << 'CMEOF'
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "arch")
set(CMAKE_ASC_RUN_MODE "sim" CACHE STRING "mode")
find_package(ASC REQUIRED)
project(alloc_test LANGUAGES ASC CXX)
add_executable(alloc_app test_alloc.asc)
target_compile_options(alloc_app PRIVATE $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>)
target_link_libraries(alloc_app PRIVATE tiling_api platform)
CMEOF
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -5
echo "make..."
make -j1 2>&1
if [ -f alloc_app ]; then
  echo "RESULT: test_alloc.asc (Alloc<float,256>) BUILD PASS"
else
  echo "RESULT: test_alloc.asc (Alloc<float,256>) BUILD FAIL"
fi
echo ""

# ── Step 3: 尝试不带命名空间的 API ──
echo "=== Step 3: 不带 using namespace 的 API ==="
WORKDIR3=/tmp/ascendc_test3
rm -rf "$WORKDIR3" && mkdir -p "$WORKDIR3" && cd "$WORKDIR3"
cat > test_nons.asc << 'ENDOFFILE'
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void test_nons(__gm__ float* x, __gm__ float* y, uint32_t n) {
    AscendC::LocalTensor<float> t = AscendC::Alloc<float, 256>();
    AscendC::GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer(x, 256);
    gy.SetGlobalBuffer(y, 256);
    AscendC::DataCopy(t, gx, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
    AscendC::Add(t, t, t, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
    AscendC::DataCopy(gy, t, 256);
    AscendC::PipeBarrier<AscendC::PIPE_ALL>();
}
ENDOFFILE
cp "$WORKDIR2/CMakeLists.txt" ./
sed -i 's/test_alloc/test_nons/g' CMakeLists.txt
sed -i 's/alloc_app/nons_app/g' CMakeLists.txt
cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -5
echo "make..."
make -j1 2>&1
if [ -f nons_app ]; then
  echo "RESULT: AscendC::Alloc (带命名空间) BUILD PASS → 需要加前缀!"
else
  echo "RESULT: AscendC::Alloc BUILD FAIL"
fi
echo ""

# ── Step 4: 编译我们的 softmax_kernel.asc ──
echo "=== Step 4: 编译我们的 softmax_kernel.asc ==="
OUR_ASC=/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/softmax/02_operator_fusion/round2/ascendc_build/softmax_kernel.asc
if [ ! -f "$OUR_ASC" ]; then
  echo "ERROR: softmax_kernel.asc not found at $OUR_ASC"
else
  WORKDIR4=/tmp/ascendc_test4
  rm -rf "$WORKDIR4" && mkdir -p "$WORKDIR4" && cd "$WORKDIR4"
  cp "$OUR_ASC" ./
  # 同时需要一个最小 host_main.cpp
  cat > host_main.cpp << 'CPPEOF'
#include "acl/acl.h"
#include <cstdio>
extern "C" void softmax_kernel(float* arg0, uint32_t totalLen);
int main() {
    constexpr uint32_t N = 2048;
    size_t bytes = N * sizeof(float);
    aclInit(nullptr);
    aclrtContext ctx; aclrtCreateContext(&ctx, 0);
    float *h; aclrtMallocHost((void**)&h, bytes);
    float *d; aclrtMalloc((void**)&d, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    for (uint32_t i=0;i<N;i++) h[i]=(float)i;
    aclrtMemcpy(d, bytes, h, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    softmax_kernel(d, N);
    aclrtSynchronizeDevice();
    aclrtFree(d); aclrtFreeHost(h);
    aclrtDestroyContext(ctx); aclFinalize();
    return 0;
}
CPPEOF
  cat > CMakeLists.txt << 'CMEOF'
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "arch")
set(CMAKE_ASC_RUN_MODE "sim" CACHE STRING "mode")
find_package(ASC REQUIRED)
project(softmax_test LANGUAGES ASC CXX)
add_executable(softmax_app softmax_kernel.asc host_main.cpp)
target_compile_options(softmax_app PRIVATE $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>)
target_compile_options(softmax_app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
target_link_libraries(softmax_app PRIVATE tiling_api platform)
CMEOF
  cmake . -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -5
  echo "make (完整错误输出)..."
  make -j1 2>&1
  if [ -f softmax_app ]; then
    echo "RESULT: softmax_kernel.asc BUILD PASS"
  else
    echo "RESULT: softmax_kernel.asc BUILD FAIL"
  fi
fi

echo ""
echo "=== 诊断完成 ==="
echo "完整日志: $OUT"
