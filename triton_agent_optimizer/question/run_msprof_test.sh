#!/bin/bash
# === CANN 9.0: 完整 AscendC compile + link + run + msprof ===
# 关键: LD_LIBRARY_PATH 必须包含所有 CANN 库目录
set -e

BIS=/usr/local/Ascend/cann-9.0.0/bin/bisheng
MSPROF=/usr/local/Ascend/cann-9.0.0/bin/msprof
CANN=/usr/local/Ascend/cann-9.0.0/x86_64-linux

# 所有 CANN 库目录 (按依赖优先级排序)
SIMLIB=$CANN/simulator/dav_2201/lib
DEVLIB=$CANN/devlib
LIB64=$CANN/lib64
DEVDEVICE=$CANN/devlib/device

export LD_LIBRARY_PATH=$SIMLIB:$DEVLIB:$LIB64:$DEVDEVICE:$LD_LIBRARY_PATH

cd /tmp && rm -rf msprof_final && mkdir msprof_final && cd msprof_final

# 1. 生成 AscendC
echo "=== Generate AscendC ==="
python3 -c "
import sys; sys.path.insert(0, '/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer')
from analyzers.hivm_to_ascendc import generate_kernel_asc
import json
ops = json.load(open('/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/softmax/02_operator_fusion/round2/hivmir/hivm_ops.json'))
with open('k.asc', 'w') as f: f.write(generate_kernel_asc('softmax_kernel', ops, 256, 'f32'))
print('ASC: OK')
"

# 2. C host
cat > h.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern void softmax_kernel(float* x, uint32_t n);
int main() {
    uint32_t N = 2048;
    float* buf = (float*)malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) buf[i] = (float)i;
    softmax_kernel(buf, N);
    free(buf);
    printf("DONE\n");
    return 0;
}
CEOF

# 3. Compile + Link
echo "=== Compile + Link ==="
$BIS -c k.asc -o k.o --npu-arch=dav-2201 --run-mode=sim 2>&1 | grep -v WARNING
gcc -c h.c -o h.o -std=c11
$BIS k.o h.o -o app \
  -L$SIMLIB -L$DEVLIB -L$LIB64 \
  -lruntime_camodel -lnpu_drv_camodel -lascend_hal \
  -lplatform -lstdc++ -lm 2>&1
echo "Build: OK ($(stat -c%s app) bytes)"

# 4. Verify ldd
echo "=== LDD Check ==="
MISSING=$(ldd app 2>&1 | grep "not found" | wc -l)
echo "Missing libs: $MISSING"
if [ "$MISSING" -gt 0 ]; then ldd app 2>&1 | grep "not found"; fi

# 5. Run standalone
echo "=== Run ==="
./app 2>&1; echo "Run exit: $?"

# 6. msprof
echo "=== msprof ==="
$MSPROF op simulator --soc-version=Ascend910B3 --output=./prof_out --timeout=10 ./app 2>&1

# 7. Results
echo "=== Results ==="
find prof_out -name "instr_exe.csv" 2>/dev/null | head -5
find prof_out -type f 2>/dev/null | wc -l
echo "Files: $(find prof_out -type f 2>/dev/null | wc -l)"
ls prof_out/ 2>/dev/null
echo "=== DONE ==="
