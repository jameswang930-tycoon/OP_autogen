# 910B3 硬件基准测量 (bench_910b3)

纯 triton 基准 kernel + msprof 实测, 测出 910B3 的真实带宽/算力,
填掉 `analyzers/timing_estimator.py` 里的 PLACEHOLDER 占位参数。

## 测什么

| bench | 测什么 | 怎么算 |
|---|---|---|
| `read_bw` | GM 读带宽 (只读 16MB) | bytes / time |
| `write_bw` | GM 写带宽 (只写 16MB) | bytes / time |
| `copy_bw` | GM 拷贝带宽 (读A写B, 32MB) | (读+写) / time |
| `l2_bw` | L2 读带宽 (L2 内 4MB 反复读 64 次) | bytes / time |
| `mm` | cube 算力 (4096³ fp16 matmul) | 2·M·N·K / time |
| `vec` | Vec 吞吐 (大向量 add) | bytes / time |

## 需要安装什么 (910B3 服务器)

```bash
# 1. conda 环境 (triton-npu, 已有 torch_npu/triton/triton-ascend)
conda activate triton-npu
# 2. CANN (msprof 工具)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 3. 验证 msprof 可用
which msprof
```

依赖: `torch`, `torch_npu`, `triton`, `triton.language` (已在 triton-npu 环境) + CANN 的 `msprof`。

## 怎么用

```bash
cd bench_910b3

# ① 全部测 (warmup 1 次 + msprof 3 轮取平均)
python3 run_bench.py

# ② 只测某个 (如 cube 算力)
python3 run_bench.py --bench mm

# ③ 调整轮数/热身 (默认 warmup=1, rounds=3)
python3 run_bench.py --rounds 5 --warmup 2
# 或环境变量: BENCH_ROUNDS=5 BENCH_WARMUP=2 python3 run_bench.py

# ④ 单测某个 kernel 是否能跑 (不带 msprof)
python3 bench_kernels.py --bench read_bw
```

## 输出

- `results.json` — 结构化结果 (每 bench: avg_us / durations / bytes / bw_gb_s / tflops)
- `results.txt` — 可读表格
- `out/<bench>/msprof_*/` — 每次 msprof 原始数据

## 结果对照 (填占位用)

把 `results.json` 里的值回填到:
- GM 读/写带宽 → `integrate.py` 的 `PEAK_MEM_BW_GB_S` (当前 1800, 用实测校准)
- cube TFLOPS → `PEAK_COMPUTE_TFLOPS` (当前 294.9)
- 各通路带宽 → `analyzers/timing_estimator.py` 的 `SATURATION_PARAMS` (替换 PLACEHOLDER)

## 注意

- 尺寸用 env 覆盖: `BENCH_BW_N`, `BENCH_MM`, `BENCH_VEC_N` 等 (见 bench_kernels.py 顶部)
- 大尺寸 (4096³) 编译/运行较慢, 首次会 JIT 编译 (warmup 已覆盖)
- 保密服务器: 结果只含数字, 可直接贴出

---

## 完整流程（bench + PyTorch 基准 + 优化 + 轨迹图）

```bash
# ── 步骤 0: 环境 (一次性) ──
conda activate triton-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ── 步骤 1: 测 910B3 硬件基准 (可选, 校准峰值用) ──
cd bench_910b3
python3 run_bench.py            # 6 个 bench → results.json (GM带宽/cube算力/vec吞吐)

# ── 步骤 2: 测 PyTorch 基准线 (轨迹图虚线用, 推荐先跑) ──
python3 bench_pytorch.py        # torch.matmul 512³ → pytorch_tflops.json

# ── 步骤 3: 跑优化主流程 (scheduler 自动读 pytorch_tflops) ──
cd ..
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15 --target 2.0

# ── 步骤 4: 生成轨迹图 (含 PyTorch 虚线) ──
python3 feedback/trajectory_chart.py outputs/matmul
# 输出: outputs/matmul/final_output/trajectory_chart.png
```

## 加速比说明

- **加速比 = 时间比**: baseline_time / current_time (msprof Task Duration)
- 图上 **左轴 Speedup = 时间比**; 右轴 TFLOPS = 同一比值的换算 (M/N/K 固定时 2MNK/time)
- **vs PyTorch**: 图标题末尾 `vs PyTorch: XX%` = 我们算力 / torch.matmul 算力
