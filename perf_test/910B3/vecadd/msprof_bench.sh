#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  msprof 普通模式采集整个 benchmark 所有 kernel → op_summary.csv → 解析
#  (msprof op 是单算子模式只采1个, 这里用普通模式采全部)
#  用法: bash msprof_bench.sh [bench脚本] [输出csv]
# ═══════════════════════════════════════════════════════════════════════════════
set -e
SCRIPT="${1:-bench_910b3_paths.py}"
OUT_CSV="${2:-msprof_result.csv}"
PROF_OUT="./prof_data"

if ! command -v msprof &>/dev/null; then
    for p in /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
             /usr/local/Ascend/cann-*/set_env.sh; do
        [ -f "$p" ] && source "$p" && break
    done
fi
command -v msprof &>/dev/null || { echo "[ERROR] 找不到 msprof"; exit 1; }
[ -f "${SCRIPT}" ] || { echo "[ERROR] 找不到 ${SCRIPT}"; exit 1; }

echo ">>> [1/3] 清理旧数据 ..."
rm -rf "${PROF_OUT}" PROF_* OPPROF_* 2>/dev/null || true

echo ">>> [2/3] msprof 采集整个进程 (所有 kernel) ..."
# 普通模式: 采集整个 python 进程的所有 kernel
# --aic-metrics 一次只能填一个值; PipeUtilization 含 MTE2/MTE3/Vec 带宽+占比
msprof --output="${PROF_OUT}" \
       --application="python ${SCRIPT}" \
       --aic-metrics=PipeUtilization \
       --ai-core=on --l2=on 2>&1 | tail -20

echo ">>> [3/3] 解析 → ${OUT_CSV} ..."
# 先确认采集是否产出了数据
if [ ! -d "${PROF_OUT}" ] || [ -z "$(find ${PROF_OUT} -name '*.csv' 2>/dev/null)" ]; then
    echo "  [ERROR] 采集未产出 CSV, msprof 可能失败. 检查上面的报错."
    echo "  ${PROF_OUT} 目录内容:"
    find "${PROF_OUT}" -type f 2>/dev/null | head -20 || echo "  (目录为空或不存在)"
    exit 1
fi
python3 - "${OUT_CSV}" "${PROF_OUT}" << 'PYEOF'
import sys, os, glob, csv
from collections import defaultdict

out_csv = sys.argv[1]
prof_out = sys.argv[2]

# 找 summary csv (普通模式可能叫 op_summary / OpBasicInfo / 等)
csvs = glob.glob(os.path.join(prof_out, "**", "op_summary*.csv"), recursive=True)
if not csvs:
    csvs = glob.glob(os.path.join(prof_out, "**", "*OpBasicInfo*.csv"), recursive=True)
if not csvs:
    csvs = glob.glob(os.path.join(prof_out, "**", "*op_statistic*.csv"), recursive=True)
if not csvs:
    print(f"  未找到 summary csv, 列出 {prof_out} 下所有 csv:")
    for f in glob.glob(os.path.join(prof_out, "**", "*.csv"), recursive=True):
        print("   ", f)
    print("\n  把上面的文件名贴出来, 我调整匹配规则")
    sys.exit(1)
src = max(csvs, key=os.path.getmtime)
print(f"  源文件: {src}")

with open(src, newline="") as fp:
    rows = list(csv.DictReader(fp))
if not rows:
    print("  空"); sys.exit(1)

cols = list(rows[0].keys())
print(f"  列名: {cols[:12]}{'...' if len(cols)>12 else ''}")

def find(*keys):
    for c in cols:
        cl = c.lower().replace(" ","").replace("_","").replace("(","").replace(")","")
        if all(k in cl for k in keys):
            return c
    return None

c_name = find("opname") or find("name")
c_dur  = find("task","duration") or find("aicore","time") or find("duration")
c_type = find("optype") or find("type")
c_mte2 = find("mte2","ratio")
c_mte2bw = find("mte2","bw")
c_mte3 = find("mte3","ratio")
c_mte3bw = find("mte3","bw")
c_vec  = find("vec","ratio")
c_cube = find("cube","ratio")

# ── 从 kernel 名反推单次调用的搬运字节数 (复用 bench 脚本逻辑) ──────────
import re
ELEM = 2; N_CORES = 20; FREQ_GHZ = 1.8

def align_data(mb, grid, max_tile):
    unit = grid * max_tile * 3
    target = int(mb*1024*1024/ELEM)
    return max(unit, (target//unit)*unit)

def infer_bytes(name):
    """返回该 kernel 单次调用搬运字节数; 算力类(FMA/ADD)返回 None"""
    g = re.search(r"_g(\d+)", name)
    grid = int(g.group(1)) if g else None
    # [1-HIT]: grid×BLK_ELEMS×OUTER_HIT×ELEM
    if "HIT" in name and grid:
        BLK_ELEMS = 49152*3*10; OUTER_HIT = 256   # =1474560, 和 bench/analyze 对齐 (原 98304*24 偏高 1.6x)
        return grid * BLK_ELEMS * OUTER_HIT * ELEM
    # [1-MISS]: N×ELEM (纯读)
    m = re.search(r"MISS(\d+)", name)
    if m and grid:
        mb = int(m.group(1))
        unit = grid*16384*3
        N = max(unit, int(mb*1024*1024/ELEM)//unit*unit)
        return N * ELEM
    # [2] write: N×ELEM
    if "write_kernel" in name and grid:
        N = align_data(256, grid, 32768)
        return N * ELEM
    # [3] copy: 2N×ELEM (1读1写)
    if "copy_kernel" in name and grid:
        N = align_data(256, grid, 32768)
        return 2 * N * ELEM
    # 算力类 FMA/ADD: 不算带宽
    return None

# ── 诊断: msprof 区不区分不同 TILE ──────────────────────────────────
all_names = [(r.get(c_name,"") or "").strip() for r in rows]
uniq = sorted(set(all_names))
print(f"\n  ━━ 诊断: 共 {len(rows)} 行, {len(uniq)} 个不同算子名 ━━")
print(f"  (sweep 了多少 TILE 就该有多少个不同 hash; 若远少于 TILE 数 → msprof 没区分)")
print()

# 按 op 名聚合
print(f"  {'Op名(前40)':<42s} {'总时延us':>10s} {'次数':>4s} {'单次us':>9s} "
      f"{'GB/s':>9s} {'B/cyc/core':>10s} {'Vec%':>6s}")
print("  " + "-"*96)

groups = defaultdict(list)
for r in rows:
    nm = (r.get(c_name,"?") or "?").strip()
    try: dur = float(r.get(c_dur,"") or 0)
    except ValueError: dur = 0.0
    groups[nm].append((dur, r))

final = []
for nm, items in sorted(groups.items(), key=lambda x: -max(d for d,_ in x[1])):
    total_dur = sum(d for d,_ in items)
    n = len(items)
    per_call_us = total_dur / n if n else 0
    r0 = items[0][1]
    def g(c): return r0.get(c,"") if c else ""

    # 算带宽 + B/cyc/core
    nbytes = infer_bytes(nm)
    if nbytes and per_call_us > 0:
        gbs = nbytes / (per_call_us * 1e-6) / 1e9
        bcyc = gbs / N_CORES / FREQ_GHZ   # 折算单核 B/cyc
        gbs_s = f"{gbs:.1f}"
        bcyc_s = f"{bcyc:.2f}"
    else:
        gbs_s = "-"; bcyc_s = "-"

    print(f"  {nm[:40]:<42s} {total_dur:>10.1f} {n:>4d} {per_call_us:>9.3f} "
          f"{gbs_s:>9s} {bcyc_s:>10s} {str(g(c_vec)):>6s}")
    final.append({"Op名":nm,"总时延us":round(total_dur,3),"调用次数":n,
                  "单次时延us":round(per_call_us,3),
                  "搬运字节":nbytes if nbytes else "",
                  "GBs": round(gbs,2) if nbytes and per_call_us>0 else "",
                  "B_cyc_core": round(bcyc,3) if nbytes and per_call_us>0 else "",
                  "MTE2占比":g(c_mte2),"MTE3占比":g(c_mte3),
                  "Vec占比":g(c_vec),"Cube占比":g(c_cube)})

with open(out_csv,"w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(final[0].keys()))
    w.writeheader(); w.writerows(final)
print(f"\n  ✓ {out_csv} ({len(final)} 个不同 kernel)")
print(f"""
  说明:
    单次us  = 总时延 / 调用次数 (msprof 时延是所有调用累加)
    GB/s    = 搬运字节 / 单次时延 (字节数从 kernel 名反推)
    B/cyc/core = GB/s ÷ 20核 ÷ 1.8GHz (折算单 AI Core)
    FMA/ADD 算力类不算带宽 (看 TFLOPS, 显示 -)
""")
PYEOF
echo ""
echo "  完成! CSV: ${OUT_CSV}"
