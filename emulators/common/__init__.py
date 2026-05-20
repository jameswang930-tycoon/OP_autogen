"""
Triton Language CPU Emulator — 公共基础设施
============================================

## 这个仓库是做什么的

GPU 算子开发需要在真实 GPU 上反复调试，迭代慢、反馈周期长。本仓库提供了一套
**Triton Language 的 CPU 侧模拟层**，让你可以在 CPU 上编写、运行、调试 Triton
kernel，验证算子逻辑的正确性，全程不需要 GPU。

工作流：
  1. 用 Triton 语法写 kernel（只用 `tl.*` API，和真实 Triton 写法完全一样）
  2. 在 CPU 上跑 emulator，得到输出
  3. 和 NumPy reference 实现做数值对比验证
  4. 验证通过后，kernel 代码几乎可以直接搬到真实 Triton 环境使用

## 整体架构（4 层）

### 第 1 层：`tl` 类 — Triton API 静态打桩

`tl` 是一个纯静态类，mock 了 Triton Language 的核心 API，所有方法签名与真实
Triton 保持一致。底层用 NumPy 实现，不做 GPU 相关的事情。

覆盖的 API 类别：
  - 内存操作：`load`, `store`, `atomic_add`
  - 创建：`zeros`, `full`, `arange`
  - 数学：`exp`, `log`, `sqrt`, `sigmoid`, `tanh`, `abs`
  - 二元：`maximum`, `minimum`, `where`, `clamp`
  - 归约：`sum`, `max`, `min`
  - 矩阵：`dot`
  - 调度：`program_id`, `num_programs`, `cdiv`
  - 调试：`static_print`, `static_assert`, `debug_barrier`

类型别名（`tl.float32`, `tl.int32` 等）也做了映射，直接指向 NumPy dtype。

### 第 2 层：`xarray` — 带内存层级追踪的 ndarray

`xarray` 继承 `np.ndarray`，增加了一个 `_in_fast_mem` 属性来标记数据当前在
SRAM（已 `tl.load`）还是 DRAM（未 load）。所有算术运算的结果自动标记为
`in_fast_mem=True`，模拟 Triton 中数据从 DRAM 加载到 SRAM 后才能参与计算。

设计意图：如果某块数据没有经过 `tl.load` 就被送入计算，虽然 NumPy 不会报错，
但真实 Triton 的行为会不同。`xarray` 的 `_in_fast_mem` 标记提供了可追踪的
元信息，配合 TraceLogger 可以辅助发现这类问题。

### 第 3 层：指针系统 — 模拟 Triton 指针算术

Triton 中 block pointers 是关键抽象。本层提供两种包装器：
  - `PointerWrapper`：包装一个 flatten 后的数据数组，支持 `ptr + offset` 语法
  - `OffsetPointer`：包装一组索引偏移，用于 block pointer 的索引计算

`tl.load` 和 `tl.store` 支持多种调用约定（`base_ptr + offsets`、`ptr + offsets`、
`base_data, indices`），自动解析并执行安全的 gather/scatter。

### 第 4 层：Grid Launch — 模拟 Triton 的 program 调度

`launch_kernel_1d/2d/3d` 三个函数模拟 Triton 的 grid 启动：
  - 设置 `tl._num_programs`
  - 遍历所有 `(pid_0, pid_1, pid_2)` 组合
  - 每次迭代设置 `tl._program_ids`，然后调用一次 kernel 函数

这就是模拟了 "每个 program 执行同一个 kernel 但处理不同数据块" 的 SPMD 模型。

## 算子约定

每个算子目录（如 `add/`, `matmul/`）下的 `__init__.py` 统一包含 4 个部分：
  1. **xxx_kernel()** — 纯粹使用 `tl.*` API 的 Triton 风格 kernel
  2. **emulate_xxx()** — 封装：展平输入 → 启动 grid → reshape 输出
  3. **reference_xxx()** — NumPy 参考实现，用于数值对比
  4. **test()** — 自测：正常路径 + 边界条件 + 错误路径

## TraceLogger — 调试管线

`TraceLogger` 是一个可选的调用追踪器（默认关闭，不影响性能）：
  - 记录每次 `tl.*` 调用的输入输出摘要（shape、数值范围、异常标记）
  - 精确到 kernel 源码行号
  - `verify()` 失败时自动附加 trace 中的异常 flag 摘要
  - 支持将 trace 格式化为字符串，适合喂给 LLM 做进一步分析

使用方式：
  ```python
  TraceLogger.enable()
  # ... run kernel ...
  TraceLogger.dump()
  trace_str = TraceLogger.format()  # 喂给 LLM
  TraceLogger.disable()
  ```

## 设计原则

  1. API 签名与真实 Triton 完全一致，不添加额外参数
  2. 每个 API 调用时做参数合法性检查，产生详细错误信息
  3. xarray 追踪 fast_mem 状态，为内存层级相关的分析提供元信息
  4. TraceLogger 记录每次 tl.* 调用的输入输出摘要，用于 debug 和 LLM 交互
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import traceback


# ============================================================
# TraceLogger: tl.* 调用追踪
# ============================================================

class TraceLogger:
    """
    记录每次 tl.* API 调用的摘要信息。
    
    默认关闭 (enabled=False), 不影响性能。
    verify 失败后可以开启重跑, 拿到完整 trace 用于 debug。
    
    使用方式:
        TraceLogger.enable()              # 开启
        ... run kernel ...
        TraceLogger.dump()                # 打印 trace
        trace_str = TraceLogger.format()  # 获取字符串 (喂给 LLM)
        TraceLogger.disable()             # 关闭并清空
    
    每条日志包含:
        pid:    当前 program_id
        api:    tl.xxx 函数名
        inputs: 输入参数的 shape 和数值摘要
        output: 输出的 shape 和数值摘要
        flags:  异常标记 (ALL_ZERO, HAS_NAN, HAS_INF, MOSTLY_ZERO)
    """

    enabled = False
    logs: list = []
    _max_logs = 10000
    _invocation_id = 0

    @classmethod
    def enable(cls):
        cls.enabled = True
        cls.logs = []
        cls._invocation_id = 0

    @classmethod
    def begin_invocation(cls):
        cls._invocation_id += 1

    @classmethod
    def disable(cls):
        cls.enabled = False
        cls.logs = []

    @classmethod
    def reset(cls):
        cls.logs = []

    @classmethod
    def _summarize(cls, x, name="tensor") -> dict:
        """生成一个 tensor 的数值摘要"""
        if x is None:
            return {"name": name, "value": "None"}
        if isinstance(x, (int, float, np.integer, np.floating)):
            return {"name": name, "type": "scalar", "value": float(x)}
        if isinstance(x, str):
            return {"name": name, "type": "str", "value": x}
        if isinstance(x, (PointerWrapper, OffsetPointer)):
            return {"name": name, "type": type(x).__name__}
        if isinstance(x, (tuple, list)):
            return {"name": name, "type": "tuple", "value": str(x)}

        try:
            arr = np.asarray(x, dtype=np.float64)
        except (ValueError, TypeError):
            return {"name": name, "type": str(type(x).__name__), "value": str(x)[:80]}

        summary = {
            "name": name,
            "shape": tuple(arr.shape),
            "dtype": str(getattr(x, 'dtype', arr.dtype)),
        }
        if arr.size > 0:
            summary["min"] = float(arr.min())
            summary["max"] = float(arr.max())
            summary["mean"] = float(arr.mean())

        # 异常标记
        flags = []
        if arr.size > 0:
            if np.all(arr == 0):
                flags.append("ALL_ZERO")
            if np.any(np.isnan(arr)):
                flags.append(f"HAS_NAN({int(np.sum(np.isnan(arr)))})")
            if np.any(np.isinf(arr)):
                flags.append(f"HAS_INF({int(np.sum(np.isinf(arr)))})")
            zero_frac = float(np.sum(arr == 0)) / arr.size
            if zero_frac > 0.5 and not np.all(arr == 0):
                flags.append(f"MOSTLY_ZERO({zero_frac:.0%})")
        if flags:
            summary["flags"] = flags

        return summary

    @classmethod
    def log(cls, api_name: str, inputs: dict, output=None):
        """记录一次 API 调用, 包含调用者在 kernel 中的行号"""
        if not cls.enabled:
            return
        if len(cls.logs) >= cls._max_logs:
            return

        pid = tuple(tl._program_ids)

        # 提取调用者行号: 从调用栈中找到不在 common/__init__.py 里的最近一帧
        # 那一帧就是 kernel 代码中调用 tl.xxx 的位置
        caller_line = None
        caller_file = None
        caller_code = None
        stack = traceback.extract_stack()
        for frame in reversed(stack):
            # 跳过 TraceLogger 自身和 tl 类内部的帧
            if 'common' in frame.filename and '__init__' in frame.filename:
                continue
            caller_file = frame.filename
            caller_line = frame.lineno
            caller_code = frame.line  # 该行的源代码文本
            break

        entry = {
            "pid": pid,
            "api": api_name,
            "line": caller_line,
            "file": caller_file,
            "code": caller_code,
            "invocation": cls._invocation_id,
            "inputs": {},
            "output": None,
        }

        for k, v in inputs.items():
            entry["inputs"][k] = cls._summarize(v, name=k)

        if output is not None:
            entry["output"] = cls._summarize(output, name="result")

        cls.logs.append(entry)

    @classmethod
    def _format_summary(cls, s: dict) -> str:
        """将一个 summary dict 格式化为紧凑字符串"""
        if s.get("type") == "scalar":
            return f"{s['name']}={s['value']:.4g}"
        if s.get("value") == "None":
            return f"{s['name']}=None"
        if s.get("type") in ("PointerWrapper", "OffsetPointer", "str", "tuple"):
            return f"{s['name']}={s.get('value', s.get('type'))}"

        parts = [f"{s['name']}"]
        if "shape" in s:
            parts.append(f"shape={s['shape']}")
        if "min" in s and s["min"] is not None:
            parts.append(f"range=[{s['min']:.4g}, {s['max']:.4g}]")
            parts.append(f"mean={s['mean']:.4g}")
        if "flags" in s:
            parts.append(f"!! {','.join(s['flags'])}")
        return " ".join(parts)

    @classmethod
    def format(cls, max_lines=200, pid_filter=None) -> str:
        """
        格式化 trace 日志, 同一 pid 下同一行号的重复调用聚合显示。

        聚合规则:
          - 同一 pid 内连续或非连续出现的同一 (line, api) 调用合并
          - 只展开前 2 次和最后 1 次的详细信息
          - 中间用 "... repeated N more times" 压缩
          - 汇总所有调用的输入输出 range
          - 单次调用的行不聚合, 正常展示
        """
        # 按 pid 过滤
        filtered = [e for e in cls.logs
                    if pid_filter is None or e["pid"] == tuple(pid_filter)]

        if not filtered:
            return "=== Trace Log (0 entries) ==="

        total = len(filtered)
        lines = [f"=== Trace Log ({total} entries, aggregated) ==="]

        # 按 pid 分组, 保持 pid 出现顺序
        pid_groups = {}
        for entry in filtered:
            pid = entry["pid"]
            if pid not in pid_groups:
                pid_groups[pid] = []
            pid_groups[pid].append(entry)

        count = 0
        for pid, entries in pid_groups.items():
            pid_str = ",".join(str(p) for p in pid)

            # 按 (line, api) 全局分组, 同时记录首次出现的顺序
            seen_order = []     # 保持首次出现顺序
            group_map = {}      # key -> [entries]
            for entry in entries:
                key = (entry.get("line"), entry["api"])
                if key not in group_map:
                    group_map[key] = []
                    seen_order.append(key)
                group_map[key].append(entry)

            for key in seen_order:
                if count >= max_lines:
                    lines.append(f"  ... truncated")
                    break

                grp_entries = group_map[key]
                line_num, api_name = key
                line_info = f"L{line_num}" if line_num else "L?"
                n = len(grp_entries)

                if n == 1:
                    # 单次调用, 正常展示
                    e = grp_entries[0]
                    in_str = ", ".join(cls._format_summary(s) for s in e["inputs"].values())
                    out_str = cls._format_summary(e["output"]) if e["output"] else "void"
                    code_suffix = ""
                    if e.get("code"):
                        code_text = e["code"].strip()
                        if len(code_text) > 60: code_text = code_text[:57] + "..."
                        code_suffix = f"  # {code_text}"
                    lines.append(f"  [pid=({pid_str})] {line_info}: tl.{api_name}({in_str}) -> {out_str}{code_suffix}")
                    count += 1
                else:
                    # 聚合: 汇总 + 展开前2次和最后1次
                    code_suffix = ""
                    if grp_entries[0].get("code"):
                        code_text = grp_entries[0]["code"].strip()
                        if len(code_text) > 60: code_text = code_text[:57] + "..."
                        code_suffix = f"  # {code_text}"

                    # 汇总所有调用的输出 range
                    all_out_mins = []
                    all_out_maxs = []
                    all_flags = set()
                    for e in grp_entries:
                        out = e.get("output")
                        if isinstance(out, dict):
                            if "min" in out and out["min"] is not None:
                                all_out_mins.append(out["min"])
                                all_out_maxs.append(out["max"])
                            if "flags" in out:
                                all_flags.update(out["flags"])

                    range_str = ""
                    if all_out_mins:
                        range_str = f", overall_range=[{min(all_out_mins):.4g}, {max(all_out_maxs):.4g}]"
                    flag_str = f" !! {','.join(sorted(all_flags))}" if all_flags else ""

                    lines.append(f"  [pid=({pid_str})] {line_info}: tl.{api_name}() x{n} calls{range_str}{flag_str}{code_suffix}")
                    count += 1

                    # 展开前 2 次
                    show_entries = grp_entries[:2]
                    for i, e in enumerate(show_entries):
                        in_str = ", ".join(cls._format_summary(s) for s in e["inputs"].values())
                        out_str = cls._format_summary(e["output"]) if e["output"] else "void"
                        lines.append(f"    call[{i}]: ({in_str}) -> {out_str}")
                        count += 1

                    # 中间省略
                    if n > 3:
                        lines.append(f"    ... {n - 3} more calls ...")
                        count += 1

                    # 展开最后 1 次
                    if n > 2:
                        e = grp_entries[-1]
                        in_str = ", ".join(cls._format_summary(s) for s in e["inputs"].values())
                        out_str = cls._format_summary(e["output"]) if e["output"] else "void"
                        lines.append(f"    call[{n-1}]: ({in_str}) -> {out_str}")
                        count += 1

            if count >= max_lines:
                break

        lines.append(f"=== End Trace ===")
        return "\n".join(lines)

    @classmethod
    def dump(cls, max_lines=200, pid_filter=None):
        """打印 trace 到 stdout"""
        print(cls.format(max_lines=max_lines, pid_filter=pid_filter))

    @classmethod
    def get_flags_summary(cls) -> dict:
        """
        统计所有日志中出现的异常 flag。
        返回 {"ALL_ZERO": [(pid, api, tensor, flag), ...], ...}
        """
        summary = {}
        for entry in cls.logs:
            line = entry.get("line")
            code = entry.get("code", "")
            # 检查 inputs
            for name, s in entry.get("inputs", {}).items():
                if isinstance(s, dict) and "flags" in s:
                    for f in s["flags"]:
                        key = f.split("(")[0]
                        summary.setdefault(key, []).append({
                            "pid": entry["pid"], "api": entry["api"],
                            "tensor": name, "section": "input", "flag": f,
                            "line": line, "code": code,
                        })
            # 检查 output
            out = entry.get("output")
            if isinstance(out, dict) and "flags" in out:
                for f in out["flags"]:
                    key = f.split("(")[0]
                    summary.setdefault(key, []).append({
                        "pid": entry["pid"], "api": entry["api"],
                        "tensor": "result", "section": "output", "flag": f,
                        "line": line, "code": code,
                    })
        return summary

    @classmethod
    def get_flags_summary_deduped(cls) -> str:
        groups = {}
        for entry in cls.logs:
            line = entry.get("line")
            api = entry["api"]
            pid = entry["pid"]
            for name, s in entry.get("inputs", {}).items():
                if isinstance(s, dict) and "flags" in s:
                    for f in s["flags"]:
                        cat = f.split("(")[0]
                        key = (line, api, "input", name, cat)
                        g = groups.setdefault(key, {"count": 0, "pids": set(), "sample_flag": f, "code": entry.get("code", "")})
                        g["count"] += 1
                        g["pids"].add(pid)
            out = entry.get("output")
            if isinstance(out, dict) and "flags" in out:
                for f in out["flags"]:
                    cat = f.split("(")[0]
                    key = (line, api, "output", "result", cat)
                    g = groups.setdefault(key, {"count": 0, "pids": set(), "sample_flag": f, "code": entry.get("code", "")})
                    g["count"] += 1
                    g["pids"].add(pid)
        if not groups:
            return ""
        lines = []
        for (line, api, section, tensor, cat), g in sorted(groups.items(), key=lambda x: (x[0][0] or 0)):
            code_hint = f"  # {g['code'].strip()[:50]}" if g.get("code") else ""
            lines.append(
                f"L{line}: tl.{api}() {section}.{tensor} -> {g['sample_flag']}"
                f"  ({g['count']}x across {len(g['pids'])} pids){code_hint}")
        return "\n".join(lines)


# ============================================================
# PointerWrapper: 模拟 Triton 指针算术
# ============================================================

class PointerWrapper:
    def __init__(self, data: np.ndarray, offset: int = 0):
        self.data = data.ravel() if isinstance(data, np.ndarray) else data
        self.offset = offset

    def __add__(self, other):
        if isinstance(other, (int, np.integer)):
            return PointerWrapper(self.data, self.offset + int(other))
        elif isinstance(other, np.ndarray):
            return OffsetPointer(self.data, np.asarray(other, dtype=np.int64) + self.offset)
        elif isinstance(other, PointerWrapper):
            raise EmulatorError("pointer_arithmetic",
                "Cannot add two pointers.")
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        raise EmulatorError("pointer_arithmetic",
            "Cannot multiply a pointer.")

    def __repr__(self):
        return f"PointerWrapper(data_len={len(self.data)}, offset={self.offset})"


class OffsetPointer:
    def __init__(self, data: np.ndarray, offsets: np.ndarray):
        self.data = data
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def __repr__(self):
        return f"OffsetPointer(data_len={len(self.data)}, offsets_shape={self.offsets.shape})"


def wrap_ptr(data: np.ndarray) -> PointerWrapper:
    flat = data.ravel()
    if flat.dtype != np.float32:
        flat = flat.astype(np.float32)
    return PointerWrapper(flat)


# ============================================================
# xarray: 带内存层级标注的 ndarray
# ============================================================

class xarray(np.ndarray):
    def __new__(cls, input_array, in_fast_mem=False):
        obj = np.asarray(input_array).view(cls)
        obj._in_fast_mem = in_fast_mem
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._in_fast_mem = getattr(obj, '_in_fast_mem', False)

    def _wrap(self, result):
        if not isinstance(result, xarray):
            result = xarray(result, in_fast_mem=True)
        else:
            result._in_fast_mem = True
        return result

    def __add__(self, other):       return self._wrap(self.view(np.ndarray).__add__(np.asarray(other)))
    def __radd__(self, other):      return self._wrap(np.asarray(other).__add__(self.view(np.ndarray)))
    def __sub__(self, other):       return self._wrap(self.view(np.ndarray).__sub__(np.asarray(other)))
    def __rsub__(self, other):      return self._wrap(np.asarray(other).__sub__(self.view(np.ndarray)))
    def __mul__(self, other):       return self._wrap(self.view(np.ndarray).__mul__(np.asarray(other)))
    def __rmul__(self, other):      return self._wrap(np.asarray(other).__mul__(self.view(np.ndarray)))
    def __truediv__(self, other):   return self._wrap(self.view(np.ndarray).__truediv__(np.asarray(other)))
    def __rtruediv__(self, other):  return self._wrap(np.asarray(other).__truediv__(self.view(np.ndarray)))
    def __floordiv__(self, other):  return self._wrap(self.view(np.ndarray).__floordiv__(np.asarray(other)))
    def __mod__(self, other):       return self._wrap(self.view(np.ndarray).__mod__(np.asarray(other)))
    def __pow__(self, other):       return self._wrap(self.view(np.ndarray).__pow__(np.asarray(other)))
    def __neg__(self):              return self._wrap(self.view(np.ndarray).__neg__())
    def __abs__(self):              return self._wrap(np.abs(self.view(np.ndarray)))

    def __gt__(self, other):  return np.asarray(self.view(np.ndarray) > np.asarray(other))
    def __lt__(self, other):  return np.asarray(self.view(np.ndarray) < np.asarray(other))
    def __ge__(self, other):  return np.asarray(self.view(np.ndarray) >= np.asarray(other))
    def __le__(self, other):  return np.asarray(self.view(np.ndarray) <= np.asarray(other))
    def __ne__(self, other):
        if isinstance(other, (xarray, np.ndarray)):
            return np.asarray(self.view(np.ndarray) != np.asarray(other))
        return True
    def __eq__(self, other):
        if isinstance(other, (xarray, np.ndarray)):
            return np.array_equal(self.view(np.ndarray), np.asarray(other))
        return False

    def to(self, dtype):
        return xarray(self.view(np.ndarray).astype(dtype), in_fast_mem=self._in_fast_mem)

    def __repr__(self):
        mem = "SRAM" if self._in_fast_mem else "DRAM"
        return f"xarray({self.view(np.ndarray).__repr__()}, mem={mem})"


# ============================================================
# EmulatorError
# ============================================================

class EmulatorError(Exception):
    def __init__(self, api_name: str, message: str, details: dict = None):
        self.api_name = api_name
        self.message = message
        self.details = details or {}
        self.source_line = None
        self.source_file = None
        self.source_code = None
        stack = traceback.extract_stack()
        for frame in reversed(stack):
            if 'common' in frame.filename and '__init__' in frame.filename:
                continue
            self.source_file = frame.filename
            self.source_line = frame.lineno
            self.source_code = frame.line
            break
        full_msg = f"\n[Triton Emulator Error] in tl.{api_name}():\n  {message}\n"
        if self.source_line:
            full_msg += f"  at: L{self.source_line}: {self.source_code}\n"
        if details:
            for k, v in details.items():
                full_msg += f"  {k}: {v}\n"
        super().__init__(full_msg)


class AggregatedEmulatorError(Exception):
    def __init__(self, errors_seen: dict, total_programs: int):
        self.errors_seen = errors_seen
        self.total_programs = total_programs
        lines = [f"[Triton Emulator] {len(errors_seen)} unique error(s) across {total_programs} programs:"]
        for (src_line, api), info in errors_seen.items():
            e = info["error"]
            code = e.source_code.strip() if e.source_code else ""
            lines.append(
                f"  L{src_line}: tl.{api}() — {e.message}"
                f"  ({info['count']}x, first at pid={info['first_pid']})")
            if code:
                lines.append(f"    code: {code}")
        super().__init__("\n".join(lines))


# ============================================================
# tl: Triton Language API 打桩
# ============================================================

class tl:
    _program_ids  = [0, 0, 0]
    _num_programs = [1, 1, 1]

    constexpr = int
    float16 = np.float16; float32 = np.float32; float64 = np.float64
    int8 = np.int8; int16 = np.int16; int32 = np.int32; int64 = np.int64
    uint8 = np.uint8; uint16 = np.uint16; uint32 = np.uint32
    bool_ = np.bool_; bfloat16 = np.float32

    @staticmethod
    def program_id(axis=0):
        if axis not in (0, 1, 2):
            raise EmulatorError("program_id", f"axis must be 0, 1, or 2, got {axis}")
        val = tl._program_ids[axis]
        TraceLogger.log("program_id", {"axis": axis}, val)
        return val

    @staticmethod
    def num_programs(axis=0):
        if axis not in (0, 1, 2):
            raise EmulatorError("num_programs", f"axis must be 0, 1, or 2, got {axis}")
        return tl._num_programs[axis]

    @staticmethod
    def arange(start, end):
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
            raise EmulatorError("arange", f"start/end must be int, got {type(start)}, {type(end)}")
        if end <= start:
            raise EmulatorError("arange", f"end must be > start, got {start}, {end}")
        r = np.arange(start, end)
        TraceLogger.log("arange", {"start": start, "end": end}, r)
        return r

    @staticmethod
    def cdiv(x, y):
        if y == 0: raise EmulatorError("cdiv", "division by zero")
        return (x + y - 1) // y

    # ---- 内存操作 ----
    @staticmethod
    def load(first, second=None, mask=None, other=0.0):
        # 解析调用约定
        if isinstance(first, OffsetPointer):
            base_data, offsets = first.data, first.offsets
            if second is not None and not isinstance(second, np.ndarray):
                if mask is None: mask = second; second = None
        elif isinstance(first, PointerWrapper):
            if isinstance(second, OffsetPointer):
                base_data, offsets = second.data, second.offsets
            elif second is not None:
                base_data = first.data
                offsets = np.asarray(second, dtype=np.int64) + first.offset
            else:
                raise EmulatorError("load", "PointerWrapper needs offsets.")
        elif isinstance(first, np.ndarray) and first.ndim == 1:
            base_data = first
            if second is not None and isinstance(second, np.ndarray):
                offsets = np.asarray(second, dtype=np.int64)
            elif second is None:
                offsets = np.arange(len(first), dtype=np.int64)
            else:
                offsets = np.asarray(second, dtype=np.int64)
        else:
            raise EmulatorError("load",
                f"Unsupported args: first={type(first).__name__}, second={type(second).__name__ if second is not None else 'None'}",
                {"hint": "Use tl.load(base_ptr, offsets, mask=...) or tl.load(ptr + offsets, mask=...)"})

        offsets = np.asarray(offsets, dtype=np.int64)
        result = np.full(offsets.shape, other,
                         dtype=base_data.dtype if hasattr(base_data, 'dtype') else np.float32)
        safe = np.clip(offsets, 0, len(base_data) - 1)

        oob = int(np.sum((offsets < 0) | (offsets >= len(base_data))))
        if oob > 0 and mask is None:
            raise EmulatorError("load",
                f"{oob}/{offsets.size} offsets OOB [0, {len(base_data)}), no mask.",
                {"offsets_range": f"[{int(offsets.min())}, {int(offsets.max())}]",
                 "hint": "Add mask to guard OOB."})

        if mask is None:
            result = base_data[safe].copy()
        else:
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.shape != offsets.shape:
                raise EmulatorError("load", f"mask shape {mask.shape} != offsets shape {offsets.shape}")
            result[np.where(mask)] = base_data[safe[np.where(mask)]]

        r = xarray(result, in_fast_mem=True)
        TraceLogger.log("load", {"offsets": offsets}, r)
        return r

    @staticmethod
    def store(first, second, third=None, mask=None):
        if isinstance(first, OffsetPointer):
            base_data, offsets = first.data, first.offsets
            values_np = np.asarray(second)
            if third is not None: mask = third
        elif isinstance(first, PointerWrapper):
            if isinstance(second, OffsetPointer):
                base_data, offsets = second.data, second.offsets
                if third is None: raise EmulatorError("store", "Missing values.")
                values_np = np.asarray(third)
            elif third is not None:
                base_data = first.data
                offsets = np.asarray(second, dtype=np.int64) + first.offset
                values_np = np.asarray(third)
            else:
                raise EmulatorError("store", "PointerWrapper needs offsets + values.")
        elif isinstance(first, np.ndarray) and first.ndim == 1:
            if third is None: raise EmulatorError("store", "Missing values.")
            base_data = first
            offsets = np.asarray(second, dtype=np.int64)
            values_np = np.asarray(third)
        else:
            raise EmulatorError("store", f"Unsupported args: first={type(first).__name__}",
                {"hint": "Use tl.store(base_ptr, offsets, values, mask=...)"})

        offsets = np.asarray(offsets, dtype=np.int64)
        if values_np.shape != offsets.shape:
            raise EmulatorError("store", f"values shape {values_np.shape} != offsets shape {offsets.shape}")
        safe = np.clip(offsets, 0, len(base_data) - 1)

        TraceLogger.log("store", {"offsets": offsets, "values": values_np})

        if mask is None:
            base_data[safe] = values_np
        else:
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.shape != offsets.shape:
                raise EmulatorError("store", f"mask shape {mask.shape} != offsets shape {offsets.shape}")
            for i in np.where(mask.ravel())[0]:
                base_data[safe.ravel()[i]] = values_np.ravel()[i]

    # ---- 创建 ----
    @staticmethod
    def zeros(shape, dtype=np.float32):
        if isinstance(shape, (int, np.integer)): shape = (int(shape),)
        r = xarray(np.zeros(shape, dtype=dtype), in_fast_mem=True)
        TraceLogger.log("zeros", {"shape": shape}, r)
        return r

    @staticmethod
    def full(shape, value, dtype=np.float32):
        if isinstance(shape, (int, np.integer)): shape = (int(shape),)
        r = xarray(np.full(shape, value, dtype=dtype), in_fast_mem=True)
        TraceLogger.log("full", {"shape": shape, "value": value}, r)
        return r

    # ---- 数学函数 ----
    @staticmethod
    def _unary_math(fn_name, np_fn, x):
        x_np = np.asarray(x)
        with np.errstate(all='raise'):
            try:
                result = np_fn(x_np)
            except FloatingPointError as e:
                raise EmulatorError(fn_name, f"Floating point error: {e}",
                    {"input_range": f"[{float(x_np.min()):.4e}, {float(x_np.max()):.4e}]",
                     "has_nan": bool(np.any(np.isnan(x_np))),
                     "has_inf": bool(np.any(np.isinf(x_np)))})
        r = xarray(result, in_fast_mem=getattr(x, '_in_fast_mem', True))
        TraceLogger.log(fn_name, {"x": x}, r)
        return r

    @staticmethod
    def exp(x):     return tl._unary_math("exp", np.exp, x)
    @staticmethod
    def log(x):     return tl._unary_math("log", np.log, x)
    @staticmethod
    def log2(x):    return tl._unary_math("log2", np.log2, x)
    @staticmethod
    def sqrt(x):    return tl._unary_math("sqrt", np.sqrt, x)
    @staticmethod
    def abs(x):     return tl._unary_math("abs", np.abs, x)

    @staticmethod
    def sigmoid(x):
        x_np = np.asarray(x)
        r = xarray(1.0 / (1.0 + np.exp(-x_np)), in_fast_mem=getattr(x, '_in_fast_mem', True))
        TraceLogger.log("sigmoid", {"x": x}, r)
        return r

    @staticmethod
    def tanh(x):    return tl._unary_math("tanh", np.tanh, x)

    @staticmethod
    def maximum(x, y):
        r = xarray(np.maximum(np.asarray(x), np.asarray(y)), in_fast_mem=True)
        TraceLogger.log("maximum", {"x": x, "y": y}, r)
        return r

    @staticmethod
    def minimum(x, y):
        r = xarray(np.minimum(np.asarray(x), np.asarray(y)), in_fast_mem=True)
        TraceLogger.log("minimum", {"x": x, "y": y}, r)
        return r

    @staticmethod
    def where(condition, x, y):
        c = np.asarray(condition)
        r = xarray(np.where(c, np.asarray(x), np.asarray(y)), in_fast_mem=True)
        TraceLogger.log("where", {"cond_true_count": int(np.sum(c)), "cond_size": c.size}, r)
        return r

    @staticmethod
    def clamp(x, min_val=None, max_val=None):
        x_np = np.asarray(x)
        if min_val is not None: x_np = np.maximum(x_np, min_val)
        if max_val is not None: x_np = np.minimum(x_np, max_val)
        r = xarray(x_np, in_fast_mem=getattr(x, '_in_fast_mem', True))
        TraceLogger.log("clamp", {"x": x, "min": min_val, "max": max_val}, r)
        return r

    # ---- 归约 ----
    @staticmethod
    def sum(x, axis=0):
        x_np = np.asarray(x)
        if axis >= x_np.ndim:
            raise EmulatorError("sum", f"axis={axis} OOR for ndim={x_np.ndim}", {"shape": x_np.shape})
        r = xarray(np.sum(x_np, axis=axis, keepdims=True), in_fast_mem=True)
        TraceLogger.log("sum", {"x": x, "axis": axis}, r)
        return r

    @staticmethod
    def max(x, axis=0):
        x_np = np.asarray(x)
        if axis >= x_np.ndim:
            raise EmulatorError("max", f"axis={axis} OOR for ndim={x_np.ndim}", {"shape": x_np.shape})
        r = xarray(np.max(x_np, axis=axis, keepdims=True), in_fast_mem=True)
        TraceLogger.log("max", {"x": x, "axis": axis}, r)
        return r

    @staticmethod
    def min(x, axis=0):
        x_np = np.asarray(x)
        if axis >= x_np.ndim:
            raise EmulatorError("min", f"axis={axis} OOR for ndim={x_np.ndim}", {"shape": x_np.shape})
        r = xarray(np.min(x_np, axis=axis, keepdims=True), in_fast_mem=True)
        TraceLogger.log("min", {"x": x, "axis": axis}, r)
        return r

    # ---- 矩阵运算 ----
    @staticmethod
    def dot(a, b, allow_tf32=True):
        a_np, b_np = np.asarray(a), np.asarray(b)
        if a_np.ndim != 2 or b_np.ndim != 2:
            raise EmulatorError("dot", f"Both must be 2D, got a.ndim={a_np.ndim}, b.ndim={b_np.ndim}")
        if a_np.shape[1] != b_np.shape[0]:
            raise EmulatorError("dot", f"Shape mismatch: a={a_np.shape}, b={b_np.shape}")
        r = xarray(a_np @ b_np, in_fast_mem=True)
        TraceLogger.log("dot", {"a": a, "b": b}, r)
        return r

    # ---- 原子操作 ----
    @staticmethod
    def atomic_add(base_ptr, offsets, values, mask=None):
        offsets = np.asarray(offsets, dtype=np.int64)
        safe = np.clip(offsets, 0, len(base_ptr) - 1)
        vals = np.asarray(values)
        mask_arr = np.ones(offsets.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        TraceLogger.log("atomic_add", {"offsets": offsets, "values": vals})
        for i in np.where(mask_arr.ravel())[0]:
            base_ptr[safe.ravel()[i]] += vals.ravel()[i]

    # ---- 调试 ----
    @staticmethod
    def static_print(*args):  print("[tl.static_print]", *args)
    @staticmethod
    def static_assert(cond, msg=""):
        if not cond: raise EmulatorError("static_assert", f"Assertion failed: {msg}")
    @staticmethod
    def debug_barrier(): pass


# ============================================================
# Kernel Launch
# ============================================================

def launch_kernel_1d(kernel_fn, *args, grid_size: int, collect_errors: bool = False, **kwargs):
    tl._num_programs = [grid_size, 1, 1]
    if TraceLogger.enabled:
        TraceLogger.begin_invocation()
    errors_seen = {}
    for pid in range(grid_size):
        tl._program_ids = [pid, 0, 0]
        try:
            kernel_fn(*args, **kwargs)
        except EmulatorError as e:
            if not collect_errors:
                raise
            key = (e.source_line, e.api_name)
            if key not in errors_seen:
                errors_seen[key] = {"count": 1, "first_pid": pid, "error": e}
            else:
                errors_seen[key]["count"] += 1
    if errors_seen:
        raise AggregatedEmulatorError(errors_seen, grid_size)

def launch_kernel_2d(kernel_fn, *args, grid: Tuple[int, int], collect_errors: bool = False, **kwargs):
    tl._num_programs = [grid[0], grid[1], 1]
    if TraceLogger.enabled:
        TraceLogger.begin_invocation()
    errors_seen = {}
    for p0 in range(grid[0]):
        for p1 in range(grid[1]):
            tl._program_ids = [p0, p1, 0]
            try:
                kernel_fn(*args, **kwargs)
            except EmulatorError as e:
                if not collect_errors:
                    raise
                key = (e.source_line, e.api_name)
                if key not in errors_seen:
                    errors_seen[key] = {"count": 1, "first_pid": (p0, p1), "error": e}
                else:
                    errors_seen[key]["count"] += 1
    if errors_seen:
        raise AggregatedEmulatorError(errors_seen, grid[0] * grid[1])

def launch_kernel_3d(kernel_fn, *args, grid: Tuple[int, int, int], collect_errors: bool = False, **kwargs):
    tl._num_programs = [grid[0], grid[1], grid[2]]
    if TraceLogger.enabled:
        TraceLogger.begin_invocation()
    errors_seen = {}
    for p0 in range(grid[0]):
        for p1 in range(grid[1]):
            for p2 in range(grid[2]):
                tl._program_ids = [p0, p1, p2]
                try:
                    kernel_fn(*args, **kwargs)
                except EmulatorError as e:
                    if not collect_errors:
                        raise
                    key = (e.source_line, e.api_name)
                    if key not in errors_seen:
                        errors_seen[key] = {"count": 1, "first_pid": (p0, p1, p2), "error": e}
                    else:
                        errors_seen[key]["count"] += 1
    if errors_seen:
        raise AggregatedEmulatorError(errors_seen, grid[0] * grid[1] * grid[2])


# ============================================================
# 验证工具
# ============================================================

def verify(emulator_out, reference_out, op_name="unknown", rtol=1e-3, atol=1e-5):
    """
    对比 emulator 输出与 reference。
    如果 TraceLogger 有数据, 自动附加异常 flag 摘要到 error_msg。
    """
    emu = np.asarray(emulator_out, dtype=np.float64)
    ref = np.asarray(reference_out, dtype=np.float64)

    if emu.shape != ref.shape:
        msg = f"Shape mismatch: emulator {emu.shape} vs reference {ref.shape}"
        print(f"  [FAIL] {op_name}: {msg}")
        return {"passed": False, "error_msg": msg, "max_abs_error": float('inf'),
                "max_rel_error": float('inf'), "mean_abs_error": float('inf')}

    abs_diff = np.abs(emu - ref)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    denom = np.maximum(np.abs(ref), 1e-12)
    max_rel = float(np.max(abs_diff / denom))
    passed = np.allclose(emu, ref, rtol=rtol, atol=atol)

    report = {
        "passed": passed, "op_name": op_name, "shape": emu.shape,
        "max_abs_error": max_abs, "mean_abs_error": mean_abs, "max_rel_error": max_rel,
        "rtol": rtol, "atol": atol, "error_msg": None, "trace": None,
    }

    if not passed:
        flat_idx = int(np.argmax(abs_diff))
        multi_idx = np.unravel_index(flat_idx, emu.shape)
        report["error_indices"] = multi_idx
        report["error_msg"] = (
            f"Numerical mismatch in {op_name}: "
            f"max_abs_err={max_abs:.4e} (atol={atol}), max_rel_err={max_rel:.4e} (rtol={rtol}). "
            f"Worst position {multi_idx}: emulator={emu[multi_idx]:.6e}, reference={ref[multi_idx]:.6e}. "
            f"Sample emulator[:5]={emu.ravel()[:5].tolist()}, "
            f"sample reference[:5]={ref.ravel()[:5].tolist()}"
        )

        # 附加 trace 异常摘要 (去重版)
        if TraceLogger.enabled and TraceLogger.logs:
            deduped = TraceLogger.get_flags_summary_deduped()
            if deduped:
                report["error_msg"] += "\n\nTrace anomalies (deduplicated):\n" + deduped
            report["trace"] = TraceLogger.format(max_lines=100)

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {op_name}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}, shape={emu.shape}")
    return report


# ============================================================
# run_with_feedback: LLM 自动生成循环的顶层接口
# ============================================================

def _format_error_feedback(e) -> str:
    if isinstance(e, AggregatedEmulatorError):
        return str(e)
    parts = [f"ERROR tl.{e.api_name}(): {e.message}"]
    if e.source_line:
        code = e.source_code.strip() if e.source_code else ""
        parts.append(f"  at L{e.source_line}: {code}")
    if e.details:
        for k, v in e.details.items():
            parts.append(f"  {k}: {v}")
    return "\n".join(parts)


def run_with_feedback(emulate_fn, reference_fn, op_name="unknown",
                      rtol=1e-3, atol=1e-5) -> dict:
    """
    运行 emulator 并产出精简去重的错误反馈。
    返回: {"passed": bool, "feedback": str, "details": dict}
    """
    TraceLogger.enable()
    try:
        emu_out = emulate_fn()
    except (EmulatorError, AggregatedEmulatorError) as e:
        feedback = _format_error_feedback(e)
        TraceLogger.disable()
        return {"passed": False, "feedback": feedback, "details": {"exception": e}}

    ref_out = reference_fn()
    report = verify(emu_out, ref_out, op_name, rtol=rtol, atol=atol)

    feedback = ""
    if not report["passed"]:
        parts = [
            f"FAIL {op_name}: max_abs_err={report['max_abs_error']:.2e}, "
            f"max_rel_err={report['max_rel_error']:.2e}"
        ]
        deduped = TraceLogger.get_flags_summary_deduped()
        if deduped:
            parts.append("Anomalies:")
            parts.append(deduped)
        feedback = "\n".join(parts)

    TraceLogger.disable()
    return {"passed": report["passed"], "feedback": feedback, "details": report}
