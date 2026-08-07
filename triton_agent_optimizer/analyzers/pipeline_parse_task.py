#!/usr/bin/env python3
"""源4: 解析通用 msprof (真机任务级) → task.json — 完整提取全部文件的所有字段。

文件 (官网核实):
  op_summary / op_statistic / task_time / api_statistic / l2_cache / msprof*.json
输出:
  - raw[]         每文件 所有列 + 所有行 (不遗漏)
  - normalized    每kernel耗时/核数/多kernel分解/launch开销/L2
用法: python pipeline_parse_task.py <task_prof目录> <out.json>
"""
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_schema import write_json  # noqa: E402

# ── dtype → 字节 (估算搬运块大小用) ──
_DTYPE_BYTES = {"float64": 8, "int64": 8, "float32": 4, "float": 4, "int32": 4,
                "float16": 2, "half": 2, "bfloat16": 2, "bf16": 2, "int16": 2,
                "int8": 1, "uint8": 1, "bool": 1}


def _dtype_bytes(s):
    """从 dtype 字符串猜字节数 (默认 4)。"""
    s = (s or "").lower()
    for k, v in _DTYPE_BYTES.items():
        if k in s:
            return v
    return 4


def _num_tensors(s):
    """从 "Input Data Type(s)" 猜张量数 (逗号/分号分隔)。"""
    if not s:
        return None
    return len([x for x in re.split(r"[,;]", str(s)) if x.strip()])


def _shape_groups(s, n_tensors=None):
    """把 InputShape 拆成 [张量dim元组,...]。
    优先 () 分组; 其次 ; 分隔; 否则整串, 若实际有 n_tensors 个张量则按维数平均拆分
    (多输入逗号分隔的 "512,512,512,512" 会被当一个 4D, 用 dtype 张量数纠正)。"""
    if not s:
        return []
    s = str(s).strip()
    groups = re.findall(r"\([^)]*\)", s) or [g for g in re.split(r"[;]", s) if g.strip()]
    dims_list = []
    for g in groups:
        dims = tuple(int(d) for d in re.split(r"[,\s*xX]+", g.strip("()[]{}")) if d.strip().isdigit())
        if dims:
            dims_list.append(dims)
    if len(dims_list) == 1 and n_tensors and n_tensors > 1:
        flat = [int(d) for d in re.split(r"[,\s*xX]+", s.strip("()[]{}")) if d.strip().isdigit()]
        if len(flat) > 1 and len(flat) % n_tensors == 0:
            per = len(flat) // n_tensors
            dims_list = [tuple(flat[i * per:(i + 1) * per]) for i in range(n_tensors)]
    return dims_list


def _shape_counts(s, n_tensors=None):
    """→ [元素数,...]"""
    out = []
    for d in _shape_groups(s, n_tensors):
        c = 1
        for x in d:
            c *= x
        out.append(c)
    return out


def _shape_dims(s, n_tensors=None):
    return _shape_groups(s, n_tensors)


def _matmul_mnk(shapes, op_type):
    """两输入两维且内维匹配 → (M,N,K); 否则 None。"""
    if not op_type:
        return None
    t = op_type.lower()
    if not any(x in t for x in ("matmul", "gemm", "mm", "mat_mul", "cube")):
        return None
    if len(shapes) < 2:
        return None
    a, b = shapes[0], shapes[1]
    if len(a) != 2 or len(b) != 2:
        return None
    m, k1 = a
    k2, n = b
    if k1 != k2:
        return None
    return (m, n, k1)


def _bw_gb_s(bytes_, time_us):
    """字节 → GB/s (时间 us)"""
    if not bytes_ or not time_us:
        return None
    return round(bytes_ / 1e9 / (time_us / 1e6), 2)


def find_prof_dir(base):
    """找 mindstudio_profiler_output 目录 (目录名拼写可能不一, 宽找)。
    优先选目录名含 mindstudio/profiler 的; 其次选含最多 op_summary 文件的。"""
    best, best_score = None, -1
    for f in Path(base).rglob("op_summary*.csv"):
        parent = f.parent
        name = str(parent).lower()
        score = 0
        if "mindstudio" in name or "profiler" in name:
            score += 10
        if "mindstudio_profiler_output" in name.replace("_", ""):
            score += 5
        if score > best_score:
            best, best_score = parent, score
    return best


def read_csv_all(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    columns = [c.strip() for c in rows[0]]
    data = []
    for r in rows[1:]:
        if r:
            data.append({columns[i]: r[i].strip()
                         for i in range(min(len(r), len(columns)))})
    return columns, data


def _f(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("n/a", "na", "nan", "none", "-", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _first(row, *keys):
    if not row:
        return None
    for k, v in row.items():
        if all(x.lower() in k.lower() for x in keys):
            return v
    return None


def parse(base):
    prof_out = find_prof_dir(base)
    if prof_out is None:
        raise SystemExit(f"[task] 找不到 {base} 下 op_summary*.csv\n"
                         f"  需先跑: msprof --output=<dir> --application='python3 test_matmul.py' --ai-core=on")

    # ── raw: 全部文件全字段 (文件名带时间戳, 按前缀匹配) ──
    def _find(prefix):
        for p in sorted(prof_out.glob(f"{prefix}*.csv")):
            return p
        return None

    raw = {}
    for prefix in ("op_summary", "op_statistic", "task_time", "api_statistic", "l2_cache"):
        p = _find(prefix)
        if p:
            cols, rows = read_csv_all(p)
            raw[prefix] = {"file": p.name, "columns": cols, "rows": rows}

    # ── normalized ──
    op_sum = raw.get("op_summary", {}).get("rows", [])
    op_stat = raw.get("op_statistic", {}).get("rows", [])
    api_stat = raw.get("api_statistic", {}).get("rows", [])
    l2 = raw.get("l2_cache", {}).get("rows", [])

    # 每 kernel 耗时 (op_summary 每行一个 kernel)
    kernels = []
    for r in op_sum:
        # per-pipe 耗时列 (aic_=cube 核, aiv_=vector 核; 列名带 (us) 后缀; fixp/fixpipe 版本差异都兼容)
        pipes = {}
        for p in ("cube", "mac", "scalar", "mte1", "mte2", "mte3", "fixpipe", "fixp"):
            aic = _f(_first(r, "aic", p, "time"))
            if aic is not None:
                pipes.setdefault(f"aic_{p}_time_us", aic)
        for p in ("vec", "scalar", "mte2", "mte3"):
            aiv = _f(_first(r, "aiv", p, "time"))
            if aiv is not None:
                pipes.setdefault(f"aiv_{p}_time_us", aiv)
        # 归一化: cube 耗时缺用 mac (真实列常是 aic_mac_time(us));
        #         cube 的 mte3 缺用 aiv_mte3 (结果 L0C→fixpipe→UB→GM, UB→GM 是 aiv_mte3)
        if "aic_cube_time_us" not in pipes and "aic_mac_time_us" in pipes:
            pipes["aic_cube_time_us"] = pipes["aic_mac_time_us"]
        if "aic_mte3_time_us" not in pipes and "aiv_mte3_time_us" in pipes:
            pipes["aic_mte3_time_us"] = pipes["aiv_mte3_time_us"]
        kernels.append({
            "op_name": _first(r, "Op", "Name") or _first(r, "op", "name"),
            "op_type": _first(r, "Op", "Type") or _first(r, "op", "type"),
            "task_type": _first(r, "Task", "Type"),
            "task_duration_us": _f(_first(r, "Task", "Duration")),
            "task_start_us": _f(_first(r, "Task", "Start")),
            "task_wait_us": _f(_first(r, "Task", "Wait")),
            "block_dim": _f(_first(r, "Block", "Dim")),
            "input_shapes": _first(r, "Input", "Shape"),
            "input_dtypes": _first(r, "Input", "Data", "Type") or _first(r, "Input", "Dtype"),
            "output_shapes": _first(r, "Output", "Shape"),
            "output_dtypes": _first(r, "Output", "Data", "Type") or _first(r, "Output", "Dtype"),
            "aicore_time_us": _f(_first(r, "aicore", "time")),
            "aiv_time_us": _f(_first(r, "aiv", "time")),
            "total_cycles": _f(_first(r, "total", "cycle")),
            **pipes,   # aic_cube/mac/scalar/mte1/2/3/fixpipe_time_us, aiv_vec/scalar/mte2/3_time_us
        })

    # ── per-op 搬运分析: 从 shape+dtype 估搬运块, 用 per-pipe 耗时算每通路带宽 ──
    ops = []
    for k in kernels:
        dtb = _dtype_bytes(k.get("input_dtypes") or k.get("output_dtypes"))
        in_counts = _shape_counts(k.get("input_shapes"))
        out_counts = _shape_counts(k.get("output_shapes"))
        bytes_in = sum(c * dtb for c in in_counts)
        bytes_out = sum(c * dtb for c in out_counts)
        # cube matmul: MTE1(→L0A/B) ≈ M*K+K*N, GM读 ≈ M*K+K*N, 写回 ≈ M*N
        mnk = _matmul_mnk(_shape_dims(k.get("input_shapes")), k.get("op_type"))
        if mnk:
            m, n, kk = mnk
            mte1_vol = (m * kk + kk * n) * dtb
            cube_macs = 2 * m * n * kk
        else:
            mte1_vol = None
            cube_macs = None
        mte2_t = k.get("aic_mte2_time_us")
        mte1_t = k.get("aic_mte1_time_us")
        store_t = k.get("aic_mte3_time_us") or k.get("aiv_mte3_time_us") or k.get("aic_fixpipe_time_us")
        cube_t = k.get("aic_cube_time_us") or k.get("aic_mac_time_us")
        transfers = []
        if mte2_t and bytes_in:
            transfers.append({"path": "GM读→L1/UB(MTE2)", "bytes": bytes_in,
                              "time_us": mte2_t, "bw_gb_s": _bw_gb_s(bytes_in, mte2_t)})
        if mte1_t and mte1_vol:
            transfers.append({"path": "L1→L0A/L0B(MTE1)", "bytes": mte1_vol,
                              "time_us": mte1_t, "bw_gb_s": _bw_gb_s(mte1_vol, mte1_t)})
        if store_t and bytes_out:
            transfers.append({"path": "L0C/UB→GM(写)", "bytes": bytes_out,
                              "time_us": store_t, "bw_gb_s": _bw_gb_s(bytes_out, store_t)})
        if cube_t and cube_macs:
            transfers.append({"path": "Cube MAC", "macs": cube_macs,
                              "time_us": cube_t,
                              "tflops": round(cube_macs / 1e12 / (cube_t / 1e6), 2)})
        ops.append({
            "op_name": k.get("op_name"), "op_type": k.get("op_type"),
            "task_type": k.get("task_type"), "block_dim": k.get("block_dim"),
            "task_duration_us": k.get("task_duration_us"),
            "input_shapes": k.get("input_shapes"), "input_dtypes": k.get("input_dtypes"),
            "est_bytes_in": bytes_in or None, "est_bytes_out": bytes_out or None,
            "transfers": transfers,
        })

    # ── 骨架槽位 kernel_slots: 按 distinct Op Name 合并 (★merge 目标, 每 kernel 一个) ──
    #     task 字段取首个 launch, launch_count 累加, deep 留空待 msprof op 填充
    _PIPE_KEYS = ("aic_cube_time_us", "aic_mac_time_us", "aic_scalar_time_us",
                  "aic_mte1_time_us", "aic_mte2_time_us", "aic_mte3_time_us",
                  "aic_fixpipe_time_us", "aic_fixp_time_us",
                  "aiv_vec_time_us", "aiv_scalar_time_us",
                  "aiv_mte2_time_us", "aiv_mte3_time_us")
    slots = {}
    for k, o in zip(kernels, ops):
        name = k.get("op_name") or "unknown"
        if name not in slots:
            slots[name] = {
                "kernel_name": name,
                "framework": bool(name.lower().startswith("aclnn")),  # torch_npu 内部 kernel (数据准备/参考), 非我们优化目标
                "launch_count": 0,
                "task": {
                    "task_type": k.get("task_type"),
                    "task_duration_us": k.get("task_duration_us"),
                    "block_dim": k.get("block_dim"),
                    "input_shapes": k.get("input_shapes"),
                    "input_dtypes": k.get("input_dtypes"),
                    "output_shapes": k.get("output_shapes"),
                    "output_dtypes": k.get("output_dtypes"),
                    "aicore_time_us": k.get("aicore_time_us"),
                    "aiv_time_us": k.get("aiv_time_us"),
                    "total_cycles": k.get("total_cycles"),
                    "pipes_us": {p: k[p] for p in _PIPE_KEYS if k.get(p) is not None},
                    "est_bytes_in": o.get("est_bytes_in"),
                    "est_bytes_out": o.get("est_bytes_out"),
                    "transfers": o.get("transfers"),
                },
                "deep": None, "filled_by": None,
            }
        slots[name]["launch_count"] += 1
    # 优化目标 = 非框架 kernel; 框架 kernel (aclnn* torch 数据准备) 单列, 不参与逐 kernel 优化
    kernel_slots = [s for s in slots.values() if not s["framework"]]
    framework_kernels = [s for s in slots.values() if s["framework"]]

    # 多 kernel 分解 (op_statistic: 每类算子 次数/总耗时)
    multi_kernel = []
    for r in op_stat:
        multi_kernel.append({
            "op_type": _first(r, "OP", "Type") or _first(r, "top", "type"),
            "core_type": _first(r, "Core", "Type"),
            "count": _f(_first(r, "Count")),
            "total_time_us": _f(_first(r, "Total", "Time")),
            "avg_us": _f(_first(r, "Avg")), "min_us": _f(_first(r, "Min")),
            "max_us": _f(_first(r, "Max")), "ratio": _f(_first(r, "Ratio")),
        })

    # launch/API 开销
    api_overhead = []
    for r in api_stat:
        api_overhead.append({
            "level": _first(r, "Level"), "api_name": _first(r, "API", "Name"),
            "total_us": _f(_first(r, "Time")), "count": _f(_first(r, "Count")),
            "avg_us": _f(_first(r, "Avg")), "max_us": _f(_first(r, "Max")),
        })

    # L2 (Hit Rate 可能是 0~1 或百分数, 统一归一化到 0~1)
    l2_hit = None
    if l2:
        for k, v in l2[0].items():
            if "hit" in k.lower() and _f(v) is not None:
                val = _f(v)
                l2_hit = round(val / 100.0, 4) if val > 1 else round(val, 4)
                break

    # summary
    def _max(k):
        vals = [x.get(k) for x in kernels if x.get(k)]
        return max(vals) if vals else None
    # ★端到端耗时 = 目标 kernel (非 aclnn) Task Duration 之和
    #   (多 kernel 如 MLP: fc1+bias_gelu+fc2 求和才是总耗时; 单 kernel 即本身)
    #   排除 aclnn 框架 kernel — 与 verifier.verify_end_to_end 口径一致
    target_durs = [k["task_duration_us"] for k in kernels
                   if k.get("task_duration_us")
                   and not (k.get("op_name") or "").lower().startswith("aclnn")]
    total_ns = (sum(target_durs) * 1000) if target_durs else None
    kernel = next((k["op_name"] for k in kernels if k["op_name"]), None)
    # num_kernels = 去重后的 op 名数 (op_summary 每行是一次启动, 同名只算一个 kernel)
    #   ★只算非框架 kernel (优化目标); 框架 kernel (aclnn* torch 数据准备) 单列 framework_kernels
    distinct = {k["op_name"] for k in kernels if k["op_name"]}
    n_total = len(distinct) if distinct else len(kernels)
    n_target = len(kernel_slots)
    n_kernels = n_target if n_target else n_total

    report = {
        "meta": {"source": "task", "generated_at": datetime.now().isoformat(),
                 "input_files": [str(prof_out)], "schema_version": "2.0"},
        "execution_summary": {"total_ns": total_ns,
                              # ★num_cores 实际是 launch grid (Block Dim), 不是物理核数 (910B3 固定 20 核)
                              "num_cores": _max("block_dim"),
                              "kernel_name": kernel,
                              "num_kernels": n_kernels,
                              "num_kernels_total": n_total},
        "raw": raw,                       # ★ 全文件全字段 (不遗漏)
        "normalized": {
            "kernels": kernels,
            "kernel_slots": kernel_slots,   # ★骨架槽位 (非框架 kernel, merge 目标, deep 待 msprof op 填)
            "framework_kernels": framework_kernels,  # torch 框架 kernel (aclnn*), 非优化目标, 仅保留观察
            "ops": ops,                     # per-op 搬运分析 (shape+dtype→字节, pipe耗时→每通路带宽)
            "multi_kernel": multi_kernel,
            "api_overhead": api_overhead,
            "l2_hit_rate": l2_hit,
        },
        "notes": ["task.json = 通用 msprof 全字段; normalized 是 LLM 用关键字段",
                  "op_summary 每 kernel 一行; api_overhead 判断 launch 开销; multi_kernel 判断是否值得 kernel 融合",
                  "kernel_slots 按 distinct Op Name 去重 (merge 目标); deep 由 msprof op 按名填充",
                  "ops[].transfers 的 bytes 是估算 (每元素每通路搬一次的近似), bw_gb_s = bytes/pipe耗时"],
    }
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_task.py <task_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[task] {sys.argv[1]} -> {sys.argv[2]}")
