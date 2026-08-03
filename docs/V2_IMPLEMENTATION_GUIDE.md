# local-adapt-v2 实施指南（GLM 5.2 直接读着写）

> 在 `local-adapt` 分支基础上实现 v2。P1/P2/P4 的代码改动已在副本验证可行，按此写；
> P3 部分环境侧、P5 是 agent+skill 架构升级。每块改完跑 `pytest -q`（`test_costmodel_untouched`
> 需 wsx 分支存在才通过，其余应全绿）。**不生成任何总结文档。**

## 架构决定（先读，贯穿全篇）

**分界线：agent 自主管"信息加载"，orchestrator 确定性管"流程和决策"。**
- 信息获取（extension 原语、经验读取、按场景加载）→ 交给 agent 的灵活性（skill + lazy-load）
- 流程控制 + 关键决策（编排、判停、best 回退、避坑、场景过滤候选）→ 留给代码的确定性

**已实测确认：目标环境的 agent 是类 claude-code runtime，命令行单次调用（`agent "query"`）会按 description 隐式触发 skill。** 所以 skill + lazy-loading 成立。但"无状态子进程"不变——每次独立 session、上下文干净不膨胀，经验靠记忆文件（experience.json）跨 session 传递，不靠 session 连续。

---

## P1 — choose_lever 重试 + 超时回退

**问题：** Round3 卡在 choose_lever 的 nga 子进程超时，`_resolve_lever` 一次超时就炸整个 run。

**改 `control/job_spec.py`：** `Budget` 类加字段
```python
lever_retries: int = 3     # choose_lever 超时/失败重试，不计入轮数；耗尽 -> 回退 vocabulary lever
```
`from_dict` 里加 `lever_retries=int(d.get("lever_retries", 3)),`

**改 `control/orchestrator.py`：**

1. import 区加：`from .llm_backend import LLMInvocationError, LLMTimeout`

2. `_resolve_lever` 中，把单次 `resp = self.llm.choose_lever(prompt)` 换成重试 + 回退：
```python
        if t is not None:
            t.write("01b_prompt_lever.txt", prompt)
        resp = None
        for _ in range(max(1, self.job.budget.lever_retries)):
            try:
                resp = self.llm.choose_lever(prompt)
                break
            except (LLMTimeout, LLMInvocationError):
                continue
        if resp is None:
            if t is not None:
                t.note(lever_fallback="vocabulary")
            return vocabulary.lever_for(verdict.bottleneck), None
        if t is not None:
            t.write("02b_response_lever.txt", resp)
        chosen = parse_lever_response(resp, cands)
        return chosen, chosen
```

**环境侧配套（env.sh）：** `NGA_CHOOSE_LEVER_MODEL=<小模型如 MiniMax-M2.7>`（choose_lever 只返回一行 json，不需要大模型）、`NGA_CHOOSE_LEVER_TIMEOUT_S=180`（调宽余量，实测 85s、默认 120 余量太小）。

---

## P2 — choose_lever 候选严格场景过滤（治 softmax 系统性误选）

**问题：** reduce_sum_int8 场景 LLM 系统性选 softmax。根因：`_scene_applies` 缺省 applies_to 视为"通用放行"，漏标的专用原语混进候选，LLM 靠"fusion 省 memory round-trip"通用论据误选。

**改 `control/orchestrator.py`：**

1. `_scene_applies` 加 strict 参数（不靠 category 猜——img2col 归 memory 类却卷积专用，category 不可靠，唯一可信信号是 applies_to）：
```python
def _scene_applies(entry: dict, op_kind, *, strict: bool = False) -> bool:
    """applies_to 缺省视为通用；给出则要求 op_kind 命中。
    strict=True（choose_lever 候选池）：applies_to 缺省的原语不进候选——
    "要 LLM 选一个去用"的候选池宁可漏（漏标的选不到）也不能错（漏标专用原语被误选）。
    index 展示仍宽松（信息多点无害）。"""
    scenes = entry.get("applies_to") or []
    if scenes:
        return op_kind in scenes
    return not strict
```

2. `_relevant_entries` 签名加 `*, strict: bool = False`，内部两处 `_scene_applies(e, op_kind)` → `_scene_applies(e, op_kind, strict=strict)`。

3. `_resolve_lever` 里候选构造加 strict：
```python
        cands = [e["name"] for e in _relevant_entries(self.job.op, verdict.bottleneck, strict=True)]
```

**注意：** index 展示（`extension_index_text` 调 `_relevant_entries` 处）**不加 strict**，保持宽松。只有 choose_lever 候选池严格。根治仍需环境侧给 softmax/img2col 等专用原语的 yaml 标对 `applies_to`。

---

## P4 — memory 三职责（最有价值，已验证生效）

**现状缺失：** 下一轮生成基准始终是原始 baseline（浪费 best-so-far）；失败原语记进了 experience 但没告诉 LLM 避开。schema 字段（`helped`/`failed`/`extension_used`）都已存在，**无需扩 schema**；`_best` 三元组 `(round_n, cycles, kernel_src)` 主循环已在维护。

**改 `control/orchestrator.py`：**

1. `_produce_kernel` 里，`while True:` 循环体内 `build_gen_prompt(...)` 之前插入：
```python
        # 职责②：迭代基准用 best-so-far kernel（有则用，无则原始 baseline）。
        iter_basis = self._best[2] if self._best is not None else self.job.baseline_src
        # 职责③：避坑清单——历史失败过的 extension 原语，提示 LLM 避开。
        avoid = self._avoided_primitives()
        retrieved_aug = retrieved_experience or ""
        if avoid:
            retrieved_aug = (retrieved_aug + "\n" if retrieved_aug else "") + \
                "避免以下曾导致失败的原语/用法（除非有确切正确用法）：" + ", ".join(sorted(avoid))
```
并把 `build_gen_prompt` 的两个参数改为：`baseline_src=iter_basis,` 和 `retrieved_experience=retrieved_aug or None,`

2. 新增方法（放 `_evidence_by_primitive` 前）：
```python
    def _avoided_primitives(self) -> set[str]:
        """职责③：曾导致失败的 extension 原语黑名单。
        信号 (helped==0 and failed>0)：推荐该原语的经验在场时从未通过、且失败过。"""
        if self.store is None:
            return set()
        avoid: set[str] = set()
        for exp in self.store.all():
            eu = getattr(exp, "extension_used", None)
            if eu and getattr(exp, "helped", 0) == 0 and getattr(exp, "failed", 0) > 0:
                avoid.add(eu)
        return avoid
```

**验证（必须真跑，不只 pytest）：** 新增 `tests/test_v2_memory_duties.py`，用一个记录每轮 prompt 的 CapturingLLM：
- 职责②：round1 的 prompt 含原始 baseline 标记；round2 的 prompt **不含**原始 baseline 标记、**含** round1 产物 kernel 标记（证明基准换成了 best-so-far）。构造：launcher 让 round1 成功且更优、round2 更差，best 停在 round1。
- 职责③：预置一条 `helped=0, failed=2, extension_used="tlext_bad"` 的经验，run 后确认某轮 prompt 含 `tlext_bad`。
- FakeLLM 的 gen 响应 json 必须含 `lever` 字段（否则 parse 失败），参考 `tests/test_t11_orchestrator.py` 的 GEN_RESP 格式。

---

## P3 — prompt 精简

**公开分支能做（通用精简）：**
- `triton-gen/SKILL.md`（96 行）body 瘦身：Step 0-N 的说明合并、去重复措辞，只留必要指令。指令性内容（怎么填、输出契约）保留，冗长解释删。
- 数据与规则分离已由之前 P1（GLM52）做过，确认 verdict/experience 不再内联进规则文本。

**环境侧做（verify_method，公开分支无此套）：** 保密环境的 launchable 模板有 ref_func / operator 两种验证示范，当前 prompt 把两种都附上。改为：组装时读 `job.verify_method`，只附对应那一种示范 kernel + case，另一种不附。这是保密环境的 build_gen_prompt / 模板改动，环境侧 5.1 做。

---

## P5 — extension-guide 拆分 + agent+skill 化（架构升级）

**目标：** 生成环节从"orchestrator 塞满 extension index 喂哑后端"，升级为"orchestrator 给精简任务描述、agent 自主按场景触发 extension skill 加载原语"。prompt 大幅精简，原语按需 lazy-load。

### P5.1 — 新增 AgentBackend（关键：接口不变，实现替换）

生成环节是 `self.llm.generate(prompt) -> str`（orchestrator.py:706）。新增一个 backend 实现**相同接口**，orchestrator 主体不动：

`control/llm_backend.py` 加：
```python
class AgentBackend:
    """类 claude-code agent：单次命令行调用会按 description 隐式触发 skill。
    与 NgaBackend 接口一致（generate/choose_lever 返回 str），但内部调 agent，
    让它自主 lazy-load extension-guide 等 skill，而非把全量 index 塞进 prompt。"""
    def __init__(self, config=None):
        # 命令行模板、模型名、skill 目录等从 config/env 读，不硬编码。
        # 环境侧适配真实 agent 命令行格式（见 P5.4）。
        ...
    def generate(self, task_prompt: str) -> str:
        # 调 agent，如 subprocess.run([<agent_cmd>, task_prompt], ...)
        # agent 内部自主触发 skill、多步执行，返回最终 kernel 响应文本。
        ...
    def choose_lever(self, prompt: str) -> str:
        ...
```
**公开分支用 mock 验证**（不接真实 agent）：写一个 FakeAgentBackend，模拟"收到精简 prompt → 假装触发 extension skill 读某原语 → 返回 kernel"，验证 orchestrator 调用链、skill 目录结构正确。真实 agent 命令行环境侧接（P5.4）。

### P5.2 — 精简 gen prompt（agent 模式下不再塞全量 index）

agent 模式下，`build_gen_prompt` 的 `EXTENSION_INDEX` 不再塞全量原语索引——改为**只给场景提示**（"当前算子 X、瓶颈 Y，按需查 extension-guide skill 获取该场景原语"）。agent 自己触发 extension-guide skill 按场景 lazy-load 具体原语。

加一个开关（config/env）区分两种模式：
- **哑后端模式（nga）**：保持现状，orchestrator 塞 index（agent 不会触发 skill）；
- **agent 模式**：prompt 精简，只给场景提示，原语靠 agent 触发 skill 加载。

### P5.3 — extension-guide 拆分成按场景的独立 skill

当前 `extension-guide` 是一个 skill + references/ 一堆 yaml。拆分为按算子大类的独立 skill，每类一个 SKILL.md（description 精准，便于 agent 隐式触发），原语 yaml 按类归入：

建议大类（环境侧按真实原语库调整）：
- `ext-reduction`（归约类：reduce_sum/max/min/mean）
- `ext-activation`（激活/逐元素数学：softmax/gelu/relu/div/sqrt）
- `ext-matmul`（矩阵类：mmul/mmaddm/img2col）
- `ext-shape`（形状变换/切片/块指针）
- `ext-quant`（量化相关）

每个 skill 的 description 写清"适用场景"，让 agent 遇到对应算子时精准触发**那一个** skill、只加载相关原语（天然场景过滤，治 softmax 混入——reduce_sum 场景根本不会触发 activation skill）。

拆分时保留：
- 每个原语 yaml 的字段契约（name/semantics/signature/category/example/pitfalls/module/applies_to）不变；
- `check_extension_cheatsheet.py` / `check_yaml_signatures.py` 校验仍适用（可能要改成遍历多个 skill 目录）；
- 读取路径契约：环境侧的 `EXT_REFS` 若指向单一 references/，拆分后要相应调整（环境侧适配）。

### P5.4 — 环境侧适配（5.1 做，公开分支只留接口）

以下在公开分支只留接口/mock，环境侧接真实：
- AgentBackend 的真实 agent 命令行格式、模型名、variant、skill 目录路径（配置驱动，写进 env.sh 或 config）；
- 验证真实 agent 会正确隐式触发拆分后的 extension skill（用 agent 命令行描述一个符合某 skill description 的任务，确认终端显示触发了预期 skill）；
- extension skill 的读取路径、多目录校验适配；
- 编译手册原语 / txt 格式 / triton 扩展包接口/范例的准备与对接。

### P5.5 — memory 是否 skill 化：不

memory 的三职责（P4：回退/不重跑/避坑）是**确定性决策**，留在 orchestrator 代码，**不做成 skill、不交给 agent 自主判断**。只有"读经验拼进上下文"这类信息获取可考虑 agent 触发，但当前 orchestrator 的 retrieve 已足够，无需 skill 化。理由：预算和正确性的关键决策不该赌 agent 每次判对。

---

## 保留不动的接口（P5 之外，环境侧稳定接缝）

- 远端仿真脚本 / 配置 / 输入输出目录：`launch(kernel_file)->dict`、raw_sim_output schema、SIM_* 环境变量、结果目录约定，全部不动；
- contracts.py（Event/Verdict/SimResult）、build_sim_result、parse_raw 契约不动；
- 编译手册原语 / txt / 扩展包接口：环境侧准备，框架侧只按契约消费。

---

## 实施顺序与验证

```
P1（choose_lever 重试）      → pytest 绿
P2（候选严格场景过滤）        → pytest 绿
P4（memory 三职责）          → pytest 绿 + test_v2_memory_duties 证明真实生效
P3（prompt 精简，公开分支部分）→ pytest 绿
P5（agent+skill）：
  P5.1 AgentBackend + FakeAgentBackend mock  → mock 验证调用链
  P5.2 gen prompt 双模式（agent 模式精简）    → pytest 绿
  P5.3 extension-guide 拆分成按场景 skill     → 校验脚本适配、pytest 绿
  （P5.4 环境侧真实 agent 对接，拉回环境后 5.1 做）
```

**每步先跑全量 pytest 确认不破坏契约。P4 和 P5 的 mock 验证必须真跑，不能只看 pytest 绿。**
