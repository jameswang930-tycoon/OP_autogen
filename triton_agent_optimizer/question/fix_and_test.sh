#!/bin/bash
# === CANN 9.0 AscendC 编译+msprof 完整链路测试 ===
# 关键修复: C host (避免 C++ ABI 不兼容) + bisheng 链接
set -e

BIS=/usr/local/Ascend/cann-9.0.0/bin/bisheng
SIMLIB=/usr/local/Ascend/cann-9.0.0/x86_64-linux/simulator/dav_2201/lib
DEVLIB=/usr/local/Ascend/cann-9.0.0/x86_64-linux/devlib
LIB64=/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64
INCLUDE=/usr/local/Ascend/cann-9.0.0/x86_64-linux/include
MSPROF=/usr/local/Ascend/cann-9.0.0/bin/msprof

cd /tmp && rm -rf c9_test && mkdir c9_test && cd c9_test

# 1. 生成 AscendC 代码
echo "=== 1. Generate AscendC ==="
python3 -c "
import sys; sys.path.insert(0, '/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer')
from analyzers.hivm_to_ascendc import generate_kernel_asc
import json
ops = json.load(open('/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/softmax/02_operator_fusion/round2/hivmir/hivm_ops.json'))
with open('softmax_kernel.asc', 'w') as f: f.write(generate_kernel_asc('softmax_kernel', ops, 256, 'f32'))
print('OK: ' + str(len(open('softmax_kernel.asc').read())) + ' chars')
"

# 2. C host (C 链接避免 C++ ABI 问题)
echo "=== 2. C host ==="
cat > host.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern void softmax_kernel(float* x, uint32_t n);
int main() {
    const uint32_t N = 2048;
    float* buf = (float*)malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) buf[i] = (float)i;
    printf("CALL\n");
    softmax_kernel(buf, N);
    printf("DONE\n");
    free(buf);
    return 0;
}
CEOF

# 3. 编译
echo "=== 3. Compile ==="
$BIS -c softmax_kernel.asc -o softmax_kernel.o --npu-arch=dav-2201 --run-mode=sim 2>&1
gcc -c host.c -o host.o -std=c11 2>&1
echo "Compile OK"

# 4. 链接
echo "=== 4. Link ==="
$BIS softmax_kernel.o host.o -o softmax_app \
  -L$SIMLIB -L$DEVLIB -L$LIB64 \
  -lruntime_camodel -lnpu_drv_camodel -lascend_hal -lm 2>&1
echo "Link OK"

# 5. 检查依赖
echo "=== 5. ldd check ==="
export LD_LIBRARY_PATH=$SIMLIB:$DEVLIB:$LIB64
ldd softmax_app 2>&1 | grep "not found" || echo "All libs found"

# 6. 运行
echo "=== 6. Run ==="
./softmax_app 2>&1
echo "RUN_EXIT: $?"

# 7. msprof
echo "=== 7. msprof ==="
$MSPROF op simulator --soc-version=Ascend910B3 --output=./msprof_out --timeout=10 ./softmax_app 2>&1

echo "=== 8. Results ==="
find msprof_out -name "*.csv" -o -name "*.json" 2>/dev/null | head -10
echo "=== DONE ==="
