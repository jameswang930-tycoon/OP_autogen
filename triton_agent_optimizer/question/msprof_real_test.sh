#!/bin/bash
# 正确的 CANN 环境 + msprof 测试
set -e

# ★ Step 1: 从 CANN 安装目录 source set_env.sh ★
cd /usr/local/Ascend/cann
source ./set_env.sh

# ★ Step 2: 加上 simulator 路径到最前面 ★
SIMLIB=/usr/local/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B3/lib
export LD_LIBRARY_PATH=$SIMLIB:$LD_LIBRARY_PATH

cd ~/v4b/build

echo "=== ldd check ==="
MISSING=$(ldd demo 2>&1 | grep -c "not found" || true)
echo "Missing libs: $MISSING"
if [ "$MISSING" -gt 0 ]; then
  ldd demo 2>&1 | grep "not found" | head -10
fi

echo ""
echo "=== run demo ==="
timeout 5 ./demo 2>&1 || true
echo "RUN: $?"

echo ""
echo "=== msprof ==="
rm -rf ~/prof_real && mkdir ~/prof_real
timeout 45 msprof op simulator \
  --soc-version=Ascend910B3 \
  --output=$HOME/prof_real \
  --timeout=1 \
  ./demo 2>&1 || true
echo "MS: $?"

echo ""
echo "=== results ==="
find ~/prof_real -type d 2>/dev/null
find ~/prof_real -name "*.csv" 2>/dev/null | head -10
find ~/prof_real -name "*.json" 2>/dev/null | head -5
echo "TOTAL FILES: $(find ~/prof_real -type f 2>/dev/null | wc -l)"
