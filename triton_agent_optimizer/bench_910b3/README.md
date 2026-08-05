# 910B3 硬件基准测量 (bench_910b3) — 全链路实测版

纯 triton 基准 kernel + msprof 实测, 测出 910B3 的**真实**带宽/算力,
校准 roofline 峰值 (`integrate.py`) 并填充各通路带宽。

**科学原则**:
1. **多变体扫描**: 每类 bench 跑多个变体 (尺寸×分块×精度) — 任何单次测量是真值的下界
2. **取最大**: 每个度量取全变体 max = 该路径峰值的最佳估计
3. **10 热身 + 30 测量平均 (一次 msprof 内循环)**: app 内部 launch warmup+measure 次,
   读 op_summary **跳过前 10 次 (JIT/冷cache), 平均后 30 次 (稳态)** — 与主优化循环同源
   (msprof 测设备侧时间确定性高, 10/30 统计上够稳, 无需 100;
    也不拆 N 次独立 msprof — 每次有 ~30s 工具开销)
4. **双数据源**: 聚合峰值 (GB/s/TFLOPS) 用通用 msprof (与主优化循环同源);
   per-path 带宽 (l0a/l0b feed, gm_to_ub, mte1/mte2) 用 msprof op + board.json (`--single` 单次)

## 测什么 (7 类, 覆盖全链路)

| bench | 测什么 | 变体数 | 聚合输出 |
|---|---|---|---|
| `gm_read` | GM 读带宽 | 4 (16/32/64MB × 分块) | `gm_bw_gb_s` (GM 读峰值) |
| `gm_write` | GM 写带宽 | 3 | `gm_bw_gb_s` (写) |
| `gm_copy` | GM 拷贝 (读A写B) | 2 | `gm_bw_gb_s` (双向) |
| `l2_read` | L2 读带宽 | 3 (4/8/16MB 反复读) | `l2_read_gb_s` |
| `cube` | cube 算力 fp16/**fp32** | 6 (4096/8192³ × 分块) | `cube_fp16/fp32_tflops` |
| `vec` | Vec 吞吐 (add/mul/fma) | 3 | `vec_bw_gb_s` |
| — | cube/vec 额外 msprof op | — | `l0a/l0b_feed`, `gm_to_ub`, `mte1/2_ratio` ... |

## 预期值 (联网核实, 校准对照)

| 指标 | 我们原配置 | 实测预期 | 备注 |
|---|---|---|---|
| cube fp16 | 294.9 TFLOPS | **~280-313** | 官网 313 (910B3 为 B 系最低配) |
| cube fp32 | (未设) | **~60-74** | 原生 FP32 GEMM 峰值 73.7 |
| GM 带宽 | 1800 GB/s (理论) | **~1200-1600** | ascend-dmi 实测 1.54TB/s; 理论 1.8 |
| HBM | 64GB | 64GB | 一致 |
| L2 读 | — | 数 TB/s | 远高于 GM |
| VecUnit | 404 GB/s | ~380-450 | 已实测 404 |
| l0a/l0b feed | 占位 150/100 | 每核 ~百 GB/s | MemoryL0.csv 实测 |

**关键提示**: 若实测 cube fp16 > 294.9 或 GM < 1600, 说明我们原硬编码偏了 → 用实测校准。

## 怎么运行 (910B3 服务器)

```bash
conda activate triton-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd bench_910b3

# ① 全套实测 (推荐; 每个变体一次 msprof, 内循环 10 热身 + 30 测量取稳态平均)
python3 run_bench.py

# ② 快速模式 (只聚合峰值, 跳过 per-path msprof op)
python3 run_bench.py --skip-op

# ③ 只测某类 / 调热身/测量次数 (一般不用调, 10/30 已够稳)
python3 run_bench.py --bench cube
python3 run_bench.py --warmup 20 --rounds 50     # 想更稳可加大
# env 覆盖: BENCH_WARMUP_ITERS=10 BENCH_MEASURE_ITERS=30
```

## 输出与校准

| 文件 | 内容 | 谁消费 |
|---|---|---|
| `results.json` | 每个变体完整测量 + 每度量 max | 人工 review |
| `hardware_peak.json` | 校准峰值 (max) | **integrate.py 自动读** (roofline 用) |
| `results.txt` | 可读表格 | 人工 |
| `out/<bench>/msprof_*` | 每次 msprof 原始 | 排错 |

**校准动作**:
- `hardware_peak.json` 生成后, `integrate.py` 自动用 `gm_bw_gb_s` / `cube_fp16_tflops` 替换硬编码 1800 / 294.9
- 想微调某个峰值: 手动编辑 `hardware_peak.json` 后重跑 `main.py` 即可 (不需重测)

**per-path → timing_estimator 引擎映射** (若想回填 v3 遗留的 SATURATION_PARAMS):
| SATURATION_PARAMS 引擎 | 实测字段 |
|---|---|
| 0 GM→UB | `gm_to_ub_gb_s` |
| 1 UB→GM | `ub_to_gm_gb_s` |
| 2 VecUnit | `vec_bw_gb_s` (≈ results 里 vec 聚合) |
| 3 GM→L1 | `main_mem_read_gb_s` / `l1_read_gb_s` |
| 4 L1→L0 (MTE1) | `l1_read_gb_s` + `mte1_ratio` |
| 5 CubeUnit feed | `l0a_feed_gb_s` / `l0b_feed_gb_s` |
| 6 L0→GM | `main_mem_write_gb_s` / `ub_to_gm_gb_s` |

> ⚠ v4 主优化循环不依赖 timing_estimator (每轮用 msprof op 逐 kernel 实测),
> 回填引擎 3-6 仅为 v3 遗留成本模型 — 优先级低。

## 完整流程 (bench + 优化 + 轨迹图)

```bash
# 环境 (一次性)
conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ① 测硬件基准 → hardware_peak.json (校准 roofline)
cd bench_910b3 && python3 run_bench.py && cd ..

# ② 测 PyTorch 基准线
python3 bench_910b3/bench_pytorch.py            # 单 matmul (512³ fp16)
python3 bench_910b3/bench_pytorch_mlp.py        # ★MLP 原算子 (2048³ fp32, 与 kernel_op.py 同形状)
   # speedup vs PyTorch = torch_mlp_time / 我们优化后 triton 端到端时间

# ③ 跑优化主流程 (scheduler 自动读 pytorch_tflops + integrate 读 hardware_peak)
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15 --target 2.0

# ④ 轨迹图 (含 PyTorch 虚线)
python3 feedback/trajectory_chart.py outputs/matmul
```

## 注意
- 尺寸 env 覆盖: `BENCH_MM` / `BENCH_BW_N` / `BENCH_VEC_N`; 热身/测量: `BENCH_WARMUP_ITERS`(10) / `BENCH_MEASURE_ITERS`(30)
- 大尺寸 (8192³ fp32) 编译/运行较慢, 首次 JIT 编译 (warmup 已覆盖)
- 保密服务器: 结果只含数字, 可直接贴出
- `run_bench.py --skip-op` 跳过 per-path (只校准 GM/cube/vec 峰值, 不填微路径)
