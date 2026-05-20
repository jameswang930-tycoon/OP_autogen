# Emulator 错误覆盖扩展方案：Reference-Guided Comparison Debugging

## Context

项目目标是 **PyTorch → Triton kernel → Emulator (正确性+性能) → LLM 反馈 → 迭代** 的自动算子生成闭环。

上一轮整改解决了错误去重问题（EmulatorError 结构化行号、AggregatedEmulatorError、TraceLogger 去重摘要、run_with_feedback）。但 emulator 仍有 ~30-40% 的 "静默数值错误" 无法精确定位根因：

| 错误类型 | 当前 emulator 能力 | 缺失 |
|----------|-------------------|------|
| stride 公式互换 | 只报 max_abs_err=14.4 | 不知道哪个 stride 写反 |
| pid 解码错误 | 只报最终数值不对 | pid 解码不经过 tl.*，不可观测 |
| 索引计算错误 | 只报 OOB 或数值偏差 | 中间变量不可观测 |

**核心思路**：不需要追踪"不可见"的中间变量，而是在 `tl.load/tl.store` 的 **边界** 上，对比 buggy kernel vs reference kernel 的 offsets 和 values。第一个发散点就是 bug 的根因。

---

## 整体架构

```
run_with_feedback()
        │
        ▼
    verify()  ← 检测到数值不匹配
        │ 如果提供了 reference_kernel
        ▼
    DualRunner.compare()  ← NEW: 逐 program 对比 tl.* 调用
        │
        ▼
    DivergenceReport  ← 结构化 + 自然语言反馈
        │
        ▼
    _generate_hint()  ← 分析 delta 模式，给出修复方向
```

---

## 改动 1：tl 类增加 CallCapture 模式

**文件**: `emulators/common/__init__.py`

在 `tl` 类上增加轻量捕获机制，仅在 DualRunner 对比时开启：

```python
class tl:
    _capture_mode = False
    _captured_calls: list = []

    @classmethod
    def _start_capture(cls):
        cls._capture_mode = True
        cls._captured_calls = []

    @classmethod
    def _stop_capture(cls) -> list:
        cls._capture_mode = False
        calls = cls._captured_calls
        cls._captured_calls = []
        return calls
```

在 `tl.load` 和 `tl.store` 中，当 `_capture_mode=True` 时记录完整的 offsets 和 values（需要 copy 防止后续 mutation）：

```python
# tl.load 内部，在返回 result 之前：
if cls._capture_mode:
    # 复用 TraceLogger 的栈回溯逻辑提取行号
    caller_line, caller_code = cls._extract_caller_info()
    cls._captured_calls.append({
        "api": "load",
        "line": caller_line,
        "code": caller_code,
        "offsets": np.array(offsets, copy=True),
        "result": np.array(result, copy=True),
        "mask": np.array(mask, copy=True) if mask is not None else None,
    })
```

需要将行号提取逻辑从 TraceLogger.log() 提取为 `tl._extract_caller_info()` 共享方法，避免重复代码。

---

## 改动 2：新建 dual_runner.py

**文件**: `emulators/common/dual_runner.py`（新文件）

### compare_kernels() — 核心对比函数

```python
def compare_kernels(buggy_kernel, reference_kernel,
                    shared_args, shared_kwargs,
                    grid_size, sample_pids=None, atol=1e-5) -> dict | None:
    """
    逐 program 对比两个 kernel 的 tl.load/tl.store 调用序列。
    返回第一个发散点的 DivergenceReport，或 None（无发散）。

    流程:
      1. 选取 sample_pids（默认: pid=0 + 边界 + 随机采样，最多 15 个）
      2. 对每个 pid:
         a. 用独立的 output buffer 跑 reference_kernel + capture
         b. 用独立的 output buffer 跑 buggy_kernel + capture
         c. 逐调用对比 offsets 和 values
      3. 在第一个发散点停止并返回报告
    """
```

### _find_first_divergence() — 调用序列对比

```python
def _find_first_divergence(ref_calls, bug_calls, pid, atol):
    """
    对比两个 captured_calls 列表，返回第一个不一致的位置。

    检查顺序（从粗到细）:
      1. 调用数量不同 → "call_count_mismatch"（控制流错误）
      2. 同一位置 API 不同 → "api_mismatch"（load vs store 混淆）
      3. offsets 不同 → "offset_divergence"（索引公式错误）★ 主要目标
      4. 同 offsets 但 values 不同 → "value_divergence"（上游计算错误）
    """
```

### _generate_hint() — Delta 模式分析

```python
def _generate_hint(divergence, kernel_strides=None):
    """
    分析 offset delta 模式，生成可操作的修复建议。

    策略:
      - delta 是某个已知 stride 的倍数 → "stride_xh 和 stride_xw 可能互换"
      - delta 恒定 → "基地址偏移错误"
      - delta 线性增长 → "循环变量累加错误"
      - delta 周期性 → "维度分解错误（如 kH*kW 的 // 和 % 写反）"
      - 部分匹配部分不匹配 → "mask/boundary 条件错误"
    """
```

### _select_sample_pids() — 智能采样

```python
def _select_sample_pids(grid_size, max_samples=15):
    """
    智能选择要对比的 pid:
      - pid=0（第一个 program，最大概率发散）
      - 最后一个 pid（边界条件）
      - 维度边界处的 pid（如 conv2d 中第一个 oh=0 但 ow>0 的 pid）
      - 随机采样补齐
    优化: 如果 pid=0 就发散了，立即返回不再采样。
    """
```

### DivergenceReport 输出格式

```python
# 文本模式（LLM 直接消费）:
"""
DIVERGENCE at L63: tl.load() offsets differ (pid=5, call #2)
  buggy  offsets[3:6] = [921, 922, 923]
  ref    offsets[3:6] = [897, 898, 899]
  delta = +24 (= 3 × stride_xh=8, suggest: kh_idx may have +3 offset)
"""

# 结构化模式（JSON，机器解析）:
{
  "error_type": "offset_divergence",
  "line": 63,
  "code": "x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)",
  "api": "load",
  "pid": 5,
  "call_index": 2,
  "buggy_sample": [921, 922, 923],
  "reference_sample": [897, 898, 899],
  "delta_pattern": "+24 (3 × stride_xh)",
  "hint": "kh_idx offset may be +3 too large in the height index calculation"
}
```

两种格式同时输出，不需要用户提前选择迭代模式。

---

## 改动 3：扩展 run_with_feedback()

**文件**: `emulators/common/__init__.py`

```python
def run_with_feedback(emulate_fn, reference_fn, op_name="unknown",
                      rtol=1e-3, atol=1e-5,
                      buggy_kernel=None, reference_kernel=None,
                      kernel_args=None, grid_size=None) -> dict:
    """
    扩展: 当 verify 失败且提供了 reference_kernel 时，
    自动调用 DualRunner 进行对比调试。
    """
    # ... 现有逻辑 ...
    if not report["passed"] and buggy_kernel and reference_kernel:
        from .dual_runner import compare_kernels
        divergence = compare_kernels(
            buggy_kernel, reference_kernel,
            kernel_args, grid_size=grid_size, atol=atol)
        if divergence:
            report["divergence"] = divergence
            feedback += "\n" + divergence["text_report"]
    # ...
```

---

## 改动 4：抽取 caller_info 共享逻辑

当前 TraceLogger.log() 和 EmulatorError.__init__ 各自有一份栈回溯逻辑。CallCapture 也需要同样的逻辑。统一为一个共享函数：

```python
def _extract_kernel_caller():
    """从调用栈中提取 kernel 源码的行号和代码文本。"""
    stack = traceback.extract_stack()
    for frame in reversed(stack):
        if 'common' in frame.filename and '__init__' in frame.filename:
            continue
        return frame.filename, frame.lineno, frame.line
    return None, None, None
```

TraceLogger.log()、EmulatorError.__init__、CallCapture 三处都调用这个函数。

---

## 改动 5：conv2d/conv1d 增加 DualRunner 测试用例

**文件**: `emulators/conv2d/__init__.py`、`emulators/conv1d/__init__.py`

```python
# conv2d test() 新增：
print("\n--- Test 7: DualRunner stride-swap 精确定位 ---")
from common.dual_runner import compare_kernels
divergence = compare_kernels(
    conv2d_kernel_bug_weight_stride, conv2d_kernel,
    shared_args=(x_flat, w_flat, b_flat, out_flat, ...),
    grid_size=grid_size)
if divergence:
    print(f"  {divergence['text_report']}")
    # 预期输出: "DIVERGENCE at L250: tl.load() offsets differ..."
    #           "delta = stride_wkh - stride_wkw swap pattern"
```

---

## 对 LLM 迭代模式的建议

基于这套 emulator 反馈体系，推荐的 LLM 迭代流程：

```
Round 1: LLM 生成 kernel
  ↓
Round 2: run_with_feedback() → 
  - 如果 hard error (OOB/axis): 文本反馈即可修复
  - 如果 numerical mismatch: DualRunner 给出精确发散点 + hint
  ↓
Round 3: LLM 根据 hint 修改 → 再次 run_with_feedback()
  ↓
... 直到 passed=True
```

反馈格式建议用**结构化 JSON + 一句话自然语言摘要**双输出。原因：
- JSON 让 LLM 可以精确定位要修改的行号和变量
- 自然语言摘要帮助 LLM 理解"为什么错"而非仅仅"哪里错"
- 两者并存让你后续可以灵活切换 prompt 策略，不被格式锁定

---

## 为 Cost Model 预留的扩展点

CallCapture 的数据天然支持性能分析。当前架构下，后续可以无侵入地添加：

```python
class CostModelAnalyzer:
    @staticmethod
    def analyze_memory(captured_calls) -> dict:
        # 从 captured load/store 计算:
        # - 总 bytes 读写量
        # - 唯一地址数（数据复用率）
        # - 访问模式分类（连续/跨步/随机）
        # - 估算 bandwidth utilization

    @staticmethod  
    def analyze_compute(captured_calls, kernel_source) -> dict:
        # 估算 FLOPs，计算 arithmetic intensity
        # 与 roofline model 对比，标记瓶颈
```

这不在本次实现范围内，但 CallCapture 已经记录了所需的全部数据。

---

## 验证方式

1. `python emulators/run_all_tests.py` — 现有测试全部通过（向后兼容）
2. conv2d stride-swap bug: DualRunner 应报出 `L250: tl.load() offsets diverge`，hint 应包含 "stride swap"
3. conv2d window-OOB bug: DualRunner 应在 pid=0 第一个 `tl.load` 就报出 offset 发散
4. conv1d axis bug: 这是 hard error，DualRunner 不需要介入（EmulatorError 直接捕获）

---

## 实现顺序

1. 改动 4（抽取 `_extract_kernel_caller()` 共享函数）
2. 改动 1（tl 类增加 CallCapture）
3. 改动 2（新建 dual_runner.py，核心对比逻辑）
4. 改动 3（扩展 run_with_feedback）
5. 改动 5（conv2d/conv1d 测试用例）

---

## 修改文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `emulators/common/__init__.py` | 修改 | 抽取 `_extract_kernel_caller()`；tl 类增加 CallCapture；扩展 `run_with_feedback` |
| `emulators/common/dual_runner.py` | 新建 | DualRunner、compare_kernels()、divergence 分析、hint 生成 |
| `emulators/conv2d/__init__.py` | 修改 | 增加 DualRunner 测试用例 |
| `emulators/conv1d/__init__.py` | 修改 | 增加 DualRunner 测试用例 |
