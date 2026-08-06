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

## 预期值 (联网核实 2026-08, 校准对照)

> 理论峰值由 `bench_theory.py` 从硬件参数公式推导, **不再硬编码**。关键来源:
> - **HBM 带宽 = JEDEC HBM2E (JESD235C)**: 单 stack = 3.2Gbps × 1024bit / 8 = **409.6 GB/s**;
>   910B3 = 4 stack → **1638.4 GB/s**。npu-smi 实测 HBM2e 64GB@1600MHz(三星), ascend-dmi 实测 ~1.54TB/s (94%)。
>   ⚠ 之前代码硬编码 1800 GB/s 是错的 (HBM2e 无此规格)。
> - **cube fp16 = 20 核 × 8192 FLOP/cyc × 1.8GHz = 294.9 TFLOPS** (标称频率推导);
>   官方标称 **313 TFLOPS** (≈1.91GHz boost)。fp32 = fp16/4 = **73.7 / 78.3**。
> - vec fp16 ≈ 64 TFLOPS (全片 INT8 128 TOPS ÷2, 推断); 片上 L2/UB 带宽无官方规格, 只取实测。

| 指标 | 理论值 | 实测预期 | 备注 |
|---|---|---|---|
| cube fp16 | **294.9** (标称1.8GHz) / **313** (官方) | ~280-313 | 20核×16×16×16 cube; 本机看标称 |
| cube fp32 | **73.7** (标称) / **78.3** (官方) | ~60-74 | fp16/4 (cube fp32 半 lane 双字节) |
| GM 带宽 | **1638.4 GB/s** (HBM2e 4×409.6) | **~1200-1600** | ascend-dmi 实测 1.54TB/s (94%) |
| HBM | 64GB | 64GB | npu-smi 一致 |
| L2 读 | 无官方规格 | 数 TB/s | 片上缓存, 远高于 GM |
| VecUnit | 无官方规格 | ~380-450 | 已实测 404 GB/s |
| l0a/l0b feed | 无官方规格 | 每核 ~百 GB/s | MemoryL0.csv 实测 |

**roofline 转折点 (算术强度)**: fp16 ≈ **180** FLOP/byte, fp32 ≈ **45** FLOP/byte
(kernel 实测强度 > 转折点 → compute-bound; < → memory-bound)

**关键提示**: 若实测 cube fp16 > 294.9 (标称) 或 GM < 1600, 用实测校准 (hardware_peak.json 自动覆盖理论回退值)。

## 怎么运行 (910B3 服务器)

```bash
conda activate triton-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd bench_910b3

# ① 理论峰值 + 理论/实测对照 (纯本地可跑, 无需 NPU)
python3 bench_theory.py                 # 打印理论峰值表 + 对照 (有 hardware_peak.json 则显示真机实测效率)

# ② 全套实测 (推荐; 每个变体一次 msprof, 内循环 10 热身 + 30 测量取稳态平均)
python3 run_bench.py

# ③ 快速模式 (只聚合峰值, 跳过 per-path msprof op)
python3 run_bench.py --skip-op

# ④ 只测某类 / 调热身/测量次数 (一般不用调, 10/30 已够稳)
python3 run_bench.py --bench cube
python3 run_bench.py --warmup 20 --rounds 50     # 想更稳可加大
# env 覆盖: BENCH_WARMUP_ITERS=10 BENCH_MEASURE_ITERS=30
```

## 输出与校准

| 文件 | 内容 | 谁消费 |
|---|---|---|
| `hardware_theory.json` | 理论峰值 + 公式来源 + 对照表 | `bench_theory.py`/人工 |
| `results.json` | 每个变体完整测量 + 每度量 max + 理论/实测对照 | 人工 review |
| `hardware_peak.json` | 校准峰值 (max) + 理论值 | **integrate.py 自动读** (roofline 用) |
| `results.txt` | 可读表格 + 理论/实测对照表 | 人工 |
| `out/<bench>/msprof_*` | 每次 msprof 原始 | 排错 |

**校准动作**:
- `hardware_peak.json` 生成后, `integrate.py` 自动用 `gm_bw_gb_s` / `cube_fp16_tflops` / `cube_fp32_tflops` 替换理论回退值 (1638.4 / 294.9 / 73.7)
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

# ① 理论峰值核对 (本地) → hardware_theory.json
cd bench_910b3 && python3 bench_theory.py --json && cd ..

# ② 测硬件基准 → hardware_peak.json (校准 roofline, 自动写理论/实测对照)
cd bench_910b3 && python3 run_bench.py && cd ..

# ③ 测 PyTorch 基准线 (每算子一个文件, 默认对齐对应 kernel_op.py 形状/精度; 输出 *_tflops.json)
#   speedup vs PyTorch = torch_时间 / 我们优化后 triton 端到端时间 (同尺寸同 dtype 才可比)
python3 bench_910b3/bench_pytorch_mlp.py            # matmul(两层MLP 2048³ fp32)
python3 bench_910b3/bench_pytorch_attention.py      # attention_mlp (2048² fp32)
python3 bench_910b3/bench_pytorch_rms_norm.py       # rms_norm (2048² fp32)
python3 bench_910b3/bench_pytorch_flash_attention.py # flash_attention (2048×8×64 因果 fp32)
python3 bench_910b3/bench_pytorch_conv2d.py         # conv2d (1×8×64²→32×64² fp32)
python3 bench_910b3/bench_pytorch_conv_bias_relu.py # conv_bias_relu (同上+ bias+relu)
   # scheduler/轨迹图按 op 自动读对应 json (bench_config.PT_BENCH_MAP)

# ④ 跑优化主流程 (scheduler 自动读 pytorch_tflops + integrate 读 hardware_peak)
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15 --target 2.0

# ⑤ 轨迹图 (含 PyTorch 虚线)
python3 feedback/trajectory_chart.py outputs/matmul
```

## 注意
- **GM 带宽测量必须用 >L2 的工作集**: L2=192MB, 旧 128MB 数组 30 次重复读会命中 L2 → 测到 L2 带宽(虚高)。
  现 gm_read/write/copy 用 **512MB~1GB** (2.7~5.3×L2), 保证测到真 GM。
- 尺寸 env 覆盖: `BENCH_MM` / `BENCH_BW_N` / `BENCH_VEC_N`; 热身/测量: `BENCH_WARMUP_ITERS`(10) / `BENCH_MEASURE_ITERS`(30)
- 大尺寸 (8192³ fp32 / 1GB GM) 编译/运行较慢, 首次 JIT 编译 (warmup 已覆盖); 1GB 数组需 HBM 余量充足
- 保密服务器: 结果只含数字, 可直接贴出
- `run_bench.py --skip-op` 跳过 per-path (只校准 GM/cube/vec 峰值, 不填微路径)
- `bench_theory.py` 纯本地可跑 (无 NPU 依赖), 服务器/本机都能看理论值
