#!/bin/bash
# === CANN 9.0 完整链路: 编译 → 链接 → 运行 → msprof ===
# 关键: 显式设置 ASCEND_HOME_PATH (source set_env.sh 在 WSL2 里不工作)

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
CANN=/usr/local/Ascend/cann-9.0.0/x86_64-linux
export LD_LIBRARY_PATH=$CANN/simulator/dav_2201/lib:$CANN/devlib:$CANN/lib64:$LD_LIBRARY_PATH

BIS=/usr/local/Ascend/cann-9.0.0/bin/bisheng
MSPROF=/usr/local/Ascend/cann-9.0.0/bin/msprof

cd /tmp && rm -rf final && mkdir final && cd final

# 生成 AscendC
python3 -c "
import sys; sys.path.insert(0, '/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer')
from analyzers.hivm_to_ascendc import generate_kernel_asc
import json
ops=json.load(open('/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/softmax/02_operator_fusion/round2/hivmir/hivm_ops.json'))
open('k.asc','w').write(generate_kernel_asc('softmax_kernel',ops,256,'f32'))
print('GEN OK')
"

cat > h.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern void softmax_kernel(float* x, uint32_t n);
int main(){uint32_t N=2048;float*b=malloc(N*4);for(uint32_t i=0;i<N;i++)b[i]=(float)i;softmax_kernel(b,N);free(b);printf("OK\n");return 0;}
CEOF

echo "=== bisheng compile ==="
$BIS -c k.asc -o k.o --npu-arch=dav-2201 --run-mode=sim 2>&1
echo "BISHENG: $?"

echo "=== gcc host ==="
gcc -c h.c -o h.o -std=c11 2>&1
echo "GCC: $?"

echo "=== link ==="
$BIS k.o h.o -o app -L$CANN/simulator/dav_2201/lib -L$CANN/devlib -L$CANN/lib64 -lruntime_camodel -lnpu_drv_camodel -lascend_hal -lplatform -lstdc++ -lm 2>&1
echo "LINK: $?"

if [ -f app ]; then
  echo "=== ldd ==="
  ldd app 2>&1 | grep "not found" || echo "ALL OK"
  echo "=== run ==="
  ./app 2>&1; echo "RUN: $?"
  echo "=== msprof ==="
  $MSPROF op simulator --soc-version=Ascend910B3 --output=./prof --timeout=10 ./app 2>&1
  echo "MS: $?"
  FILES=$(find prof -type f 2>/dev/null | wc -l)
  echo "FILES: $FILES"
  find prof -name "*.csv" 2>/dev/null | head -5
  find prof -name "*.json" 2>/dev/null | head -5
fi
