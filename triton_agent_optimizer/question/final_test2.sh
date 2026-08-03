#!/bin/bash
# === CANN 9.0 完整链路 v2 — LD_PRELOAD stub 绕过 ErrorManager ABI 不匹配 ===

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
CANN=/usr/local/Ascend/cann-9.0.0/x86_64-linux
BIS=/usr/local/Ascend/cann-9.0.0/bin/bisheng
MSPROF=/usr/local/Ascend/cann-9.0.0/bin/msprof

cd /tmp && rm -rf final2 && mkdir final2 && cd final2

# 1. 生成 stub 库 (提供缺失的 ErrorManager::ATCReportErrMessage 符号)
cat > stub_error.c << 'CEOF'
#include <string>
#include <vector>
#include <cstdio>
namespace { int dummy; }
// 提供 libruntime_common.so 需要的 ATCReportErrMessage
// mangled: _ZN12ErrorManager19ATCReportErrMessageESsRKSt6vectorISsSaISsEES4_
class ErrorManager {
public:
    static ErrorManager* GetInstance() { static ErrorManager* p=nullptr; return p; }
    std::string ATCReportErrMessage(std::string const& s, std::vector<std::string> const& v, std::string const& s2) {
        fprintf(stderr, "STUB: ErrorManager::ATCReportErrMessage called\n");
        return "";
    }
    std::string GetLogHeader() { return ""; }
};
extern "C" {
    void* _ZN12ErrorManager11GetInstanceEv() { return ErrorManager::GetInstance(); }
    void _ZN12ErrorManager19ATCReportErrMessageESsRKSt6vectorISsSaISsEES4_(void* self, void* ret, void* s, void* v, void* s2) {
        // 只是一个 stub, 不做实际事情
    }
    void _ZN12ErrorManager12GetLogHeaderEv(void* self, void* ret) {}
}
CEOF

echo "=== compile stub ==="
g++ -shared -fPIC stub_error.c -o libstub_error.so -std=c++11 2>&1
echo "STUB: $?"

# 2. 生成 AscendC
python3 -c "
import sys; sys.path.insert(0, '/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer')
from analyzers.hivm_to_ascendc import generate_kernel_asc
import json
ops=json.load(open('/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/softmax/02_operator_fusion/round2/hivmir/hivm_ops.json'))
open('k.asc','w').write(generate_kernel_asc('softmax_kernel',ops,256,'f32'))
print('GEN OK')
"

# 3. C host
cat > h.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern void softmax_kernel(float* x, uint32_t n);
int main(){uint32_t N=2048;float*b=malloc(N*4);for(uint32_t i=0;i<N;i++)b[i]=(float)i;softmax_kernel(b,N);free(b);printf("OK\n");return 0;}
CEOF

# 4. 编译 + 链接
echo "=== compile ==="
$BIS -c k.asc -o k.o --npu-arch=dav-2201 --run-mode=sim 2>&1 | grep -v WARNING
gcc -c h.c -o h.o -std=c11 2>&1
echo "COMPILE OK"

echo "=== link ==="
$BIS k.o h.o -o app -L$CANN/simulator/dav_2201/lib -L$CANN/devlib -L$CANN/lib64 -lruntime_camodel -lnpu_drv_camodel -lascend_hal -lplatform -lstdc++ -lm 2>&1
echo "LINK: $?"

if [ -f app ]; then
  export LD_LIBRARY_PATH=$(pwd):$CANN/simulator/dav_2201/lib:$CANN/devlib:$CANN/lib64:$LD_LIBRARY_PATH
  echo "=== run with stub ==="
  LD_PRELOAD=./libstub_error.so ./app 2>&1; echo "RUN: $?"

  echo "=== msprof with stub ==="
  LD_PRELOAD=./libstub_error.so $MSPROF op simulator --soc-version=Ascend910B3 --output=./prof --timeout=10 ./app 2>&1
  echo "MS: $?"

  echo "=== results ==="
  find prof -type f 2>/dev/null | head -20
  echo "TOTAL FILES: $(find prof -type f 2>/dev/null | wc -l)"
  find prof -name "instr_exe.csv" 2>/dev/null | head -5
fi
