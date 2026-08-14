# 服务器运行命令指南 (910B3 / CANN 8.5.1 / torch_npu 2.9.0)

> 用途: ① 环境检查与安装 ② 4 算子修复验证 ③ 工业级基准全量测量 ④ 生成对比表
> 原则: 按顺序执行; 每步有明确判定标准; 贴回输出即可定位问题

---

## 第 0 步: 拉取最新代码

```bash
cd ~/OP_autogen && git pull origin hjkc
```

---

## 第 1 步: 环境检查 (定 torchair/桥库安装版本 — 把输出按 1~8 编号全部贴回来)

```bash
# 1. python 版本
python3 --version
# 2. torch 版本
python3 -c "import torch; print(torch.__version__)"
# 3. torch_npu 版本
python3 -c "import torch_npu; print(torch_npu.__version__)"
# 4. CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null \
  || cat /usr/local/Ascend/ascend-toolkit/latest/version.info 2>/dev/null \
  || ls /usr/local/Ascend/ascend-toolkit/latest/
# 5. 服务器架构 (x86_64 / aarch64)
uname -m
# 6. 系统 (欧拉/麒麟/Ubuntu)
grep -E "^(NAME|VERSION)=" /etc/os-release
# 7. triton / 已装相关包
pip list 2>/dev/null | grep -iE "triton|npu|cann|air"
# 8. pip 能否联网 (能连→出错误信息里带 999 找不到; 连不上→超时/网络错误)
pip download torchair==999 2>&1 | head -2
```

**各项用途 (定版本依据)**:
| # | 决定什么 |
|---|---|
| 1~3 | torchair 版本对 **torch 小版本 + torch_npu 版本** 有硬性配对 (错一个 import 即崩) |
| 4 | CANN 版本 → 配套 torchair 大版本 |
| 5 | x86_64 / aarch64 决定下载哪个 whl 平台包 |
| 6 | 华为官网按 OS (欧拉/麒麟/Ubuntu) 分包 |
| 7 | 有没有 triton-ascend (装 torchair 时的依赖冲突排查) |
| 8 | 服务器能否联网 pip — 不通就走离线 whl 上传路线 |

**判定**:
- 有 `torchair` → 跳过第 2 步的 torchair 安装
- 有 `npu_ops_transformer` 或 `cann_ops_transformer` → 跳过第 2 步的桥库安装

---

## 第 2 步: 按需安装 (可选, 不装也能跑, 只是缺融合数字)

### ① torchair (compile 列后端; 版本必须与 CANN 8.5.1 配对)

```bash
# 优先: CANN toolkit 自带 whl
ls /usr/local/Ascend/ascend-toolkit/latest/tools/ | grep -i air
pip install /usr/local/Ascend/ascend-toolkit/latest/tools/torchair*.whl

# 备选: 华为云源 (版本号以官网兼容性表为准)
pip install torchair==<匹配版本> -i https://repo.huaweicloud.com/repository/pypi/simple/
```

### ② 桥库 npu_ops_transformer (cann-fused 列)

```bash
# 先看 torch_npu 2.9.0 是否自带
python3 -c "from torch_npu.contrib import *" 2>&1 | head -1

# 不自带则: pip install npu_ops_transformer 或从 Ascend 官网下载配套 whl
```

**判定**: 装完 `python3 -c "import torchair"` 无报错 = OK

---

## 第 3 步: 4 算子修复验证 (上次 HIVM 编译失败, 已修复, 必须裸跑确认)

```bash
cd triton_agent_optimizer

for op in conv2d conv_bias_relu maxpool2d flash_attention; do
  echo "══ $op ══"
  cd input/$op && python3 kernel_op.py 2>&1 | tail -6 && cd ../..
done

# 正确性校验 (裸跑通过后)
cd input/conv2d && MATMUL_VERIFY=1 python3 kernel_op.py
cd ../conv_bias_relu && MATMUL_VERIFY=1 python3 kernel_op.py
cd ../maxpool2d && MATMUL_VERIFY=1 python3 kernel_op.py
cd ../flash_attention && MATMUL_VERIFY=1 python3 kernel_op.py
cd ../..
```

**判定**:
- 裸跑见 `conv2d launched & synced OK` / `maxpool2d launched & synced OK` → 编译修复成功
- 裸跑还有 `ub overflow` / `root alloc` 报错 → 贴回来, 我继续修
- MATMUL_VERIFY 见 `result check: PASS` → 正确性 OK

---

## 第 4 步: 工业级基准全量测量 (17 算子 × 各自方法, 需几分钟~几十分钟)

```bash
cd triton_agent_optimizer/bench_910b3
python3 bench_all.py
```

**判定**:
- 末尾见 `成功 17/17 个算子有工业级最优端到端` → 全部完成
- eager 或 fa 有 `⚠` 未产出 → 是问题, 贴 stderr 回来
- compile/cann-fused 显示 `⚠ 回退 eager/compile` → 正常 (没装 torchair/桥库)

---

## 第 5 步: 生成对比表 (4 方法列 + 我们优化结果空列)

```bash
python3 make_summary_table.py
```

**产出**: `bench_910b3/outputs/industrial_summary_table.md` — 表格式:
列 = eager / compile / cann-fused / fa / 我们优化结果(留空手动填); 行 = 17 个算子
加粗 = 该算子工业级最优; ⚠回退 = 该方法的数字是别的实现的重复测量, 不作数

---

## 第 6 步: 跑我们的优化循环 (出"我们优化结果"列数据)

```bash
cd triton_agent_optimizer && python3 main.py --op <算子>     # 逐算子跑, 全部 17 个
```

> 每个算子最后轮次记录 e2e_event_ns → 手动填进表格"我们优化结果"列
> 与工业级同尺 (Event 设备侧端到端), 直接对比
