# Memory 模块接入指南 —— 把 memory/ 接进 OP_autogen 流水线

> 本文是自包含的实现说明,单看这一个文件即可动手,无需其他上下文。
> 目标:把项目根的 `memory/` 包接入现有流水线,只做两处接线,先只接
> 正确性这一路,并用一个开关对照实验验证"记忆是否有用"。
>
> 涉及的真实接口签名、文件路径,均来自仓库 `wsx` 分支实际代码,不是假设。

---

## 0 范围(第一版只做这些)

- **做**:检索历史经验注入生成上下文;每次生成尝试写回一条记录并更新经验统计;正确性
  写回(用模拟器结果)。
- **先不做**:上板延迟写回(异步,依赖精炼循环)、自动经验蒸馏、语义/向量检索、服务化。
  这些骨架里已留位置,后面再接。

---

## 1 骨架提供的 API(在 `memory/` 包内)

```python
from memory import (
    ExperienceStore,        # 经验库(JSON 文件后端)
    RunLog,                 # 运行日志(JSONL,只追加)
    fingerprint_from_plan_json, # 从 triton-plan 写出的 .plan.json 派生算子特征(见 §2)
    retrieve,               # 按算子特征检索经验,排序取前 n 条
    format_context,         # 把检索到的经验拼成可注入提示词的文本
    record_attempt,         # 写回:记一条日志 + 更新经验统计
    add_experience,         # 手工新增一条经验(自动蒸馏是后续工作)
)
```

初始化一次(全流程共用同一对象):

```python
store = ExperienceStore("memory/experience/experience.json")
log   = RunLog("memory/runlog/runlog.jsonl")
```

---

## 2 三条必须先知道的修正(最容易踩的坑)

**修正一:`cost_planner.py` 已废弃,规划产物是纯文本,不是结构化字典。**
真实规划路径(见 `.claude/commands/triton-plan.md`):`triton-plan` 写一段 cost_emulator
DSL,**直接**跑 `costModel/cost_emulator/simulator.py --verify` 和 `--llm --critical-path`,
把 `--llm` 的 **7 段纯文本**原样写进 `.plan.json`:

```jsonc
// 正常:
{"op": "matmul", "shapes": {...}, "dtype": "fp16", "dsl": "...", "raw_llm": "……7 段文本……"}
// 模拟器失败的降级桩:
{"mock": true, "op": "matmul", "shapes": {...}, "dtype": "fp16", "note": "simulator failed"}
```

7 段是:execution summary / time breakdown / per-op / engine util / bandwidth util /
parallelism / critical path。瓶颈信息只以自然语言埋在这段文本里,`triton-plan` **不解释**,
真正读懂它的是 `triton-gen`。所以**不要去解析这段文本来做算子特征**。

**修正二:算子特征就用 `op_kind`(+dtype、+粗粒度 shape),不放瓶颈标签。**
`fingerprint_from_plan_json(plan_json)` 直接读 `.plan.json` 的 `op` / `shapes`,mock 桩
也能正常降级。想要瓶颈标签时,**不要写文本解析器**——让 `triton-gen` 顺手吐出它已经推断
的那个瓶颈标签(它本就在读 TIME BREAKDOWN / CRITICAL PATH),作为可选入参回喂:
`fingerprint_from_plan_json(plan_json, bottleneck="cube|compute_bound")`。缺失时主键自动
退回 `op_kind`。

**修正三:写回点就是 `run_with_feedback` 的返回值。**
`emulators/common/__init__.py` 里:

```python
def run_with_feedback(emulate_fn, reference_fn, op_name="unknown",
                      rtol=1e-3, atol=1e-5) -> dict:
    # 返回: {"passed": bool, "feedback": str, "details": dict}
```

`result["passed"]` 正好是写回要用的布尔量,不需要另造校验接口。

---

## 3 接入点一:注入(在 `triton-gen` 之前)

现有流程里,`triton-plan` 技能把规划产物写到 `emulators/test/<op>/.plan.json`,结构是
`{op, shapes, dtype, dsl, raw_llm}`,随后 `triton-gen` 读它的 `raw_llm` 拼提示词。
检索就插在这两步之间。

现有流程里,`triton-plan` **已经**把 `.plan.json` 写好了。记忆步骤**不调用任何规划器**,
只做一件事:读这份已有的 `.plan.json`,检索经验,把结果**追加**回同一个文件的一个新字段
`retrieved_experience`,供 `triton-gen` 读取。

```python
import json
from memory import fingerprint_from_plan_json, retrieve, format_context

def inject_experience(plan_json_path, store, bottleneck=None):
    """在 triton-plan 之后、triton-gen 之前调用。不碰任何规划器。"""
    # 1. 读 triton-plan 已写好的 .plan.json
    with open(plan_json_path, encoding="utf-8") as f:
        plan_json = json.load(f)

    # 2. 派生算子特征(mock 桩也能降级;bottleneck 可由 triton-gen 回喂,见修正二)
    fp = fingerprint_from_plan_json(plan_json, bottleneck=bottleneck)

    # 3. 检索历史经验
    hits = retrieve(store, fp, n=3)
    retrieved_ids = [e.id for e in hits]

    # 4. 追加字段写回同一个 .plan.json(其余字段原样保留)
    plan_json["retrieved_experience"] = format_context(hits)
    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(plan_json, f, ensure_ascii=False, indent=2)

    # fp 与 retrieved_ids 要带到写回点(见接入点二)
    return fp, retrieved_ids
```

然后在 `.claude/commands/triton-gen.md` 里加一句:除了 `raw_llm`,也读取
`.plan.json` 的 `retrieved_experience` 字段,把其中的历史经验作为生成参考。

> 备选做法:不改 `.plan.json`,而是在调用 `triton-gen` 的驱动里,直接把
> `format_context(hits)` 拼进传给它的上下文。二选一即可。

---

## 4 接入点二:写回(在 `run_with_feedback` 之后)

现有生成/自测里对模拟器的调用形如(见 `emulators/test/conv1d/llm_fix_demo.py`):

```python
result = run_with_feedback(emulate_fn, reference_fn,
                           op_name="conv1d_iter", rtol=1e-3, atol=1e-4)
passed = result["passed"]
```

在这一行之后写回。把它包一层最省事:

```python
from memory import record_attempt

def verify_and_record(emulate_fn, reference_fn, op_name,
                      fp, retrieved_ids, log, store,
                      kernel_ref=None, rtol=1e-3, atol=1e-5):
    result = run_with_feedback(emulate_fn, reference_fn,
                               op_name=op_name, rtol=rtol, atol=atol)
    record_attempt(
        log, store, fp,
        retrieved_ids=retrieved_ids,     # 接入点一返回的那一份
        passed=result["passed"],         # 直接用它的布尔
        kernel_ref=kernel_ref,           # 例如 f"emulators/test/{op_name}"
        latency_us=None,                 # 上板延迟先不接,保持 None
    )
    return result                        # result["feedback"] 仍照旧喂给修复循环
```

`record_attempt` 内部会:追加一条日志;对 `retrieved_ids` 里的每条经验 `used += 1`,
若 `passed` 则 `helped += 1`。修复循环(`triton-fix`)照旧用 `result["feedback"]`,不受影响。

---

## 5 新增经验(第一版手工)

某算子首次跑通、且效果好时,手工存一条经验。内容可以取自这次的 `.plan.json` 关键做法,
或 `docs/emulator_observations/*` 里的相关条目:

```python
from memory import add_experience
add_experience(store, fp, text="……这次有效的关键做法或踩过的坑……")
```

自动蒸馏(离线读日志、提炼、去重、替换)是后续工作,当前先手工让经验库有内容。

---

## 6 目录布局(建议)

```
OP_autogen/
  memory/                      # 骨架包(已有)
  memory/runlog/runlog.jsonl   # 运行日志(自动生成)
  memory/experience/experience.json   # 经验库(自动生成)
```

`memory/runlog/` 与 `memory/experience/` 首次运行时自动创建,建议加进 `.gitignore`
(运行产物,不入库)。

---

## 7 验证:记忆 on/off 对照实验

这是判断"记忆是否有用"的唯一硬标准,必须先做。注意:`run_all_tests.py` 跑的是**已经
写好的固定 kernel**,记忆在那里不起作用;记忆起作用的是**生成环节**。所以对照实验要
包在"生成一次尝试"这个动作上。

最小骨架(把 `run_one_generation` 的内部换成你实际的 plan→gen→verify 调用):

```python
from memory import ExperienceStore, RunLog, retrieve

def run_one_generation(op_kind, shapes, dtype, store, log, use_memory: bool):
    """跑一次完整生成尝试,返回 (是否首次通过, 修复迭代次数)。"""
    fp, retrieved_ids = inject_experience(plan_path(op_kind), store)
    if not use_memory:
        retrieved_ids = []          # 关闭记忆:不注入、不计入统计
    # …这里调用 triton-gen 生成 kernel,再用 verify_and_record 校验、循环修复…
    # 返回本次的首次通过与迭代数
    ...

def ab_test(op_list):
    for use_memory in (False, True):
        store = ExperienceStore(f"memory/ab_{use_memory}/experience.json")
        log   = RunLog(f"memory/ab_{use_memory}/runlog.jsonl")
        passed_first, total_iters = 0, 0
        for op_kind, shapes, dtype in op_list:
            ok, iters = run_one_generation(op_kind, shapes, dtype, store, log, use_memory)
            passed_first += int(ok); total_iters += iters
        n = len(op_list)
        print(f"memory={use_memory}: 首次通过率={passed_first/n:.2f}, 平均迭代={total_iters/n:.2f}")
```

看两个指标:**首次通过率**(越高越好)和**达到正确的平均迭代数**(越低越好)。
打开记忆若这两项改善,说明记忆有效。**看到正向信号之前,不要往 §0 列的"先不做"里
任何一项扩展。**

---

## 8 一句话总览

只加两处接线:triton-plan 之后 `inject_experience`(读 .plan.json + 检索 + 写回该文件),模拟器校验后
`verify_and_record`(写回 + 更新统计)。其余照旧。真实签名见 §2、§3、§4。
