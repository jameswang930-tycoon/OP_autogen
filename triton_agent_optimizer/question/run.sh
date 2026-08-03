#!/bin/bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
CANN=/usr/local/Ascend/cann-9.0.0/x86_64-linux
CANNCXX=/usr/local/Ascend/cann-9.0.0/tools/hcc/sysroot/usr/lib64

# ★ CANN old libstdc++ FIRST to fix ABI mismatch ★
export LD_LIBRARY_PATH=$CANNCXX:$CANN/simulator/dav_2201/lib:$CANN/devlib:$CANN/lib64

cd /tmp && rm -rf run && mkdir run && cd run

# Generate AscendC
python3 -c "
import sys; sys.path.insert(0, '/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer')
from analyzers.hivm_to_ascendc import generate_kernel_asc
import json
ops=json.load(open('/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/softmax/02_operator_fusion/round2/hivmir/hivm_ops.json'))
open('k.asc','w').write(generate_kernel_asc('softmax_kernel',ops,256,'f32'))
print('GEN OK')
"

# C host
cat > h.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern void softmax_kernel(float* x, uint32_t n);
int main(){uint32_t N=2048;float*b=malloc(N*4);for(uint32_t i=0;i<N;i++)b[i]=(float)i;softmax_kernel(b,N);free(b);printf("OK\n");return 0;}
CEOF

# Compile
echo "=== compile ==="
/usr/local/Ascend/cann-9.0.0/bin/bisheng -c k.asc -o k.o --npu-arch=dav-2201 --run-mode=sim 2>&1 | grep -v WARNING
echo "BISHENG: $?"
gcc -c h.c -o h.o -std=c11 2>&1
echo "GCC: $?"

# Link
echo "=== link ==="
/usr/local/Ascend/cann-9.0.0/bin/bisheng k.o h.o -o app \
  -L$CANN/simulator/dav_2201/lib -L$CANN/devlib -L$CANN/lib64 \
  -lruntime_camodel -lnpu_drv_camodel -lascend_hal -lplatform -lstdc++ -lm 2>&1
echo "LINK: $?"

# Run
echo "=== run ==="
./app 2>&1
echo "RUN_EXIT: $?"

# msprof
echo "=== msprof ==="
/usr/local/Ascend/cann-9.0.0/bin/msprof op simulator --soc-version=Ascend910B3 --output=./prof --timeout=10 ./app 2>&1
echo "MS_EXIT: $?"

# Results
find prof -type f 2>/dev/null | head -20
echo "TOTAL_FILES: $(find prof -type f 2>/dev/null | wc -l)"
