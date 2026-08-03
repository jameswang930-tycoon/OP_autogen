#!/bin/bash
# CANN 8.5.1 完整 msprof 链路验证
set -e
source $HOME/Ascend/cann-8.5.1/set_env.sh
echo "=== version ==="
bisheng --version 2>&1 | head -1
which msprof

cd /tmp && rm -rf v851 && mkdir v851 && cd v851

# 生成 AscendC (用 CANN 8.5.1 兼容 API: extern "C" __global__ __aicore__ + Alloc)
cat > k.asc << 'ASCEOF'
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void test_kernel(__gm__ float* x, __gm__ float* y, uint32_t n) {
    constexpr uint32_t BLK = 256;
    uint32_t blockNum = (n + BLK - 1) / BLK;
    for (uint32_t i = 0; i < blockNum; i++) {
        uint32_t off = i * BLK;
        uint32_t sz = (i == blockNum - 1 && n % BLK) ? n % BLK : BLK;
        GlobalTensor<float> gx, gy;
        gx.SetGlobalBuffer(x + off, sz); gy.SetGlobalBuffer(y + off, sz);
        LocalTensor<float> t1 = Alloc<float, 256>();
        LocalTensor<float> t2 = Alloc<float, 256>();
        DataCopy(t1, gx, sz); PipeBarrier<PIPE_ALL>();
        DataCopy(t2, gx, sz); PipeBarrier<PIPE_ALL>();
        Add(t1, t1, t2, sz); PipeBarrier<PIPE_ALL>();
        DataCopy(gy, t1, sz); PipeBarrier<PIPE_ALL>();
    }
}
ASCEOF

cat > host.cpp << 'CPPEOF'
#include <cstdint>
#include <cstdio>
#include <cstdlib>
extern "C" void test_kernel(float* x, float* y, uint32_t n);
int main(){uint32_t N=2048;float*b=(float*)malloc(N*4);for(uint32_t i=0;i<N;i++)b[i]=(float)i;test_kernel(b,b,N);free(b);printf("OK\n");return 0;}
CPPEOF

echo "=== build ==="
bisheng k.asc host.cpp -o app --npu-arch=dav-2201 2>&1 | grep -v WARNING
echo "BUILD: $?"

if [ -f app ]; then
  echo "=== run ==="
  ./app 2>&1; echo "RUN: $?"

  echo "=== msprof ==="
  msprof op simulator --soc-version=Ascend910B3 --output=./prof --timeout=10 ./app 2>&1
  echo "MS: $?"

  echo "=== results ==="
  find prof -type f 2>/dev/null | head -30
  echo "TOTAL: $(find prof -type f 2>/dev/null | wc -l)"
  find prof -name "*.csv" 2>/dev/null | head -5
fi
