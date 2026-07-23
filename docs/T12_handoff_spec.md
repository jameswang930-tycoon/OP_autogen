# T12 规范：交接包 `HANDOFF_GLM47.md`（推导程序式）

> **读者：Claude Code + GLM 5.2**（非保密开发机）。本文档规定 T12 应当写成什么。
> **核心要求**：交接包的五个槽位**必须写成推导程序（derivation procedure），不是内容规格**。
>
> 理由：真实仿真输出格式、真实 stall 类型、真实 extension 原语，只有保密环境里的 GLM 4.7 能看到。开发侧（用户与你）预设这些内容既不可靠也无必要。**4.7 的任务不是"按我们给的规格填空"，而是"按我们给的方法，从它眼前的材料里推导出内容"。**
>
> 因此每个槽位任务的写法统一为：**输入材料从哪来 → 逐步推导方法 → 自检命令 → 缺材料时如何停下上报**。

---

## 0. 写作前先做的事

**重新扫描 `control/` 与 `.claude/skills/` 下的实际代码**，逐一核对交接包中出现的每个文件路径、函数签名、字段名。签名对不上是最常见也最致命的错误。以下为已核实的冻结契约（T2 定义，写进交接包时逐字使用）：

```python
Event:      name:str, start:int, end:int, duration:int, unit:str, stall_class:str, bytes:Optional[int]
Verdict:    bottleneck:str, lever:str, cycles:int, expected_gain:float
SimResult:  correct:bool, max_abs_err:float, cycles:Optional[int], pipeline:dict

parse_raw(raw_sim_output) -> list[Event]          # control/feedback_adapter.py   槽位
launch(kernel_file) -> raw_sim_output             # control/launch_template.py    槽位
check_extension_calls(kernel_src) -> list[str]    # control/presim_gate.py        槽位
adapt(events, k=TOP_K) -> AdapterOutput(.verdict, .summary)
```

---

## 1. 交接包的写作原则（务必遵守）

1. **不要让 4.7 读架构文档、执行手册或本规范**——它读不完也不需要。交接包**自包含且极短**。
2. **每个槽位写成推导程序**：编号步骤、每步一个动作、可机械执行。不写设计论证。
3. **每步都要有可执行的自检**，不依赖模型自我判断。
4. **每个槽位都必须有"缺材料 / 推不出来"的分支**，明确写"停下，记录到 `PROGRESS.md` 待确认事项，不要猜"。
5. 中文，句子短，一项一段，避免嵌套条件。
6. **贯穿始终的一句话**（写在交接包开头）：**能观察到的就转录，观察不到的就上报——任何情况下都不要编造字段名、原语名或类别名。**

---

## 2. 材料自举（bootstrap）——写在五个槽位之前

交接包的第一节不是任务，而是**告诉 4.7 怎么自己拿到材料**。这样用户无需预先准备任何东西。

> **第 0 步：取样**
> 1. 找到保密环境中**已经能跑通的仿真调用方式**（用户已打通闭环，这段代码或命令是现成的）。
> 2. 用它跑 **3–5 个**已有的小算子，把每次的**原始仿真输出**完整保存到 `materials/sim_samples/`。样本尽量覆盖不同情况（正常、明显访存慢、明显计算慢、以及一次数值不对的）。
> 3. 找到 extension 包的 API 文档或头文件，路径记录在 `materials/README.md`。
> 4. 若第 1 步找不到可跑通的调用方式 → **停下上报**，后续四个槽位都依赖它。

**这一节是整个交接包的地基**：五个槽位的内容全部从这批样本与文档中推导，而非来自任何人的预设。

---

## 3. 五个槽位的推导程序（按此顺序，顺序本身有意义）

> **排序理由（写进交接包）**：词表界定了速查表的范围。先定词表，速查表就只需收录能解决这些类别的原语，工作量收敛且可逐类完成，适配小上下文窗口。反之会试图转录整个 API，超窗口且大部分用不上。

### 槽位 1：`control/vocabulary.yaml`（先做）

**这一项是分类判断，产出需用户确认。**

> 1. 打开 `materials/sim_samples/` 的全部样本，列出其中出现的**所有代表"代价/停顿/瓶颈"的信号**（各类 stall 计数、占用率、流量等），先不做合并，原样罗列。
> 2. 对每一个信号，回答一个问题：**"如果这是主要原因，我会怎么优化这个 kernel？"** 把答案写在旁边。
> 3. **按第 2 步的答案分组**——应对动作相同的合并成一类，动作不同的分开。
> 4. 收敛到 **5–8 类**。若不足 5 类，说明样本覆盖不够，回第 0 步补样本；若超过 8 类，继续按动作合并。
> 5. 为每类写 `id`（小写下划线）、`desc`（一句话）、`lever`（第 2 步得到的优化动作）；`primitives` 先留 `[]`。
> 6. 替换 `vocabulary.yaml` 中的 3 条示例条目。**字段结构不得改动。**
> 7. 自检：`.venv/bin/python -m control.check_vocab_consistency`
> 8. **把这份草案连同"每类对应哪些原始信号"的依据，写进 `PROGRESS.md` 待确认事项，停下等用户确认后再继续。**
>
> 判断粒度的唯一标准：**两类是否分开，只看它们是否导向不同的优化动作。** 应对相同就合并——分开只会让后续多做无意义的选择。若某类没有任何可做的优化，它属于"不可约"，无需单独成类。

（第 8 步是刻意设计：词表是三方共用的单一定义源，错误会级联到 adapter 标签、memory 检索键、速查表索引。让 4.7 **草拟**、用户**确认**，既避免用户从零撰写，又避免弱模型独断。）

### 槽位 2：`parse_raw()` in `control/feedback_adapter.py`

> 1. 取 `materials/sim_samples/` 中**任意一份**样本，打印其结构（若是 JSON 就打印键与嵌套层级；若是文本就看行格式）。
> 2. **先写映射表，再写代码。** 在函数上方以注释写出下面这张表并填满右列：
>
>    | Event 冻结字段 | 类型 | 来自原始输出的哪个字段/位置 |
>    |---|---|---|
>    | `name` | str | ? |
>    | `start` | int | ? |
>    | `end` | int | ? |
>    | `duration` | int | ?（若原始只给 start/end，则 end-start） |
>    | `unit` | str | ?（执行单元/角色） |
>    | `stall_class` | str | ?（必须映射到槽位 1 词表中的 id） |
>    | `bytes` | Optional[int] | ?（无则 None） |
>
> 3. **若某个必填字段在原始输出中找不到对应** → 停下上报，不要用常量或猜测值填充。
> 4. 按映射表实现函数，返回 `list[Event]`。
> 5. 自检：`.venv/bin/python -m pytest tests/test_t5_adapter.py -q`
> 6. 端到端确认：用一份真实样本跑 `adapt(parse_raw(sample))`，检查产出的 7 段摘要与 Verdict——`verdict.bottleneck` 必须是词表中的 id。

### 槽位 3：`.claude/skills/extension-guide/references/`

> 1. 打开 `vocabulary.yaml`，取出全部类别 id。**只为这些类别找原语，不要通读整个 API。**
> 2. 逐个类别处理（一次只做一类，控制上下文占用）：在 extension 文档中找出能解决该类别的原语。
> 3. 每个原语照 `references/` 下**已有的样例条目**格式填写：名称 / 一句话语义 / 签名 / 解决哪个瓶颈类别 / 一个最小示例 / 常见坑。**格式不得改动。**
> 4. 把该原语名回填进 `vocabulary.yaml` 对应类别的 `primitives` 列表。
> 5. 若某个类别在 extension 中找不到对应原语 → `primitives` 保持 `[]` 并在条目中注明，**不要为了填满而牵强对应**。
> 6. 自检：`.venv/bin/python -m control.check_extension_cheatsheet`（每条的类别必须在词表内）

### 槽位 4：`launch()` in `control/launch_template.py`

> 1. 找到第 0 步中那份**已经能跑通的仿真调用代码/命令**。
> 2. **包一层壳适配签名**：入参是 kernel 文件路径，返回值是**原始仿真输出**（即槽位 2 的 `parse_raw` 能吃进去的东西）。不要在这里做解析——解析是 `parse_raw` 的职责，两者分工不得混淆。
> 3. 加超时与错误处理：设施故障（超时、连接断）应抛出可识别的异常，编排器会按 `sim_retries` 退避重试且不计入轮数。
> 4. 自检：`.venv/bin/python -m pytest tests/test_t6_launch.py -q`
> 5. 端到端确认：跑一个小 kernel，确认能拿到合法 `SimResult`（`correct` / `max_abs_err` / `cycles` / `pipeline` 齐全）。

### 槽位 5：`check_extension_calls()` in `control/presim_gate.py`

> 1. 从槽位 3 完成的速查表中取出所有原语的**签名**。
> 2. 实现静态检查：kernel 源码中出现的 extension 调用，其原语名是否存在、参数个数/类型是否符合签名。**只做静态检查，不执行代码。**
> 3. 返回问题列表，空列表表示通过（签名与占位实现一致）。
> 4. 自检：`.venv/bin/python -m pytest tests/test_t7_gate.py -q`
> 5. 端到端确认：写一个故意用错原语的 kernel，确认被挡下。

---

## 4. 交接包的其余四节

**① 三条纪律**（与 `AGENTS.md` 一致）：不改函数签名/schema/词表结构；不做架构判断（遇到设计问题停下上报）；每改一处跑对应自检、通过再做下一处。

**② 运行说明**：怎么写 job 文件（三种输入形态，当前完整支持 `triton_file`）、怎么跑 `.venv/bin/python -m control.orchestrator --job jobs/xxx.yaml`、产物在哪、怎么读 `report.json`（特别说明 `recommended` 与 `final_round` 的区别：前者是实测最优、可能就是 baseline，后者是最后一轮、可能更慢但保留了诊断价值）。

**③ 验证清单**（五个槽位全部完成后自证）：
1. 小 kernel 跑通完整闭环，产出 `report.json` + 两份 kernel。
2. 停止逻辑三用例（复用 `tests/test_t3_controller.py`，应仍全绿）。
3. 记忆冷启动价值：清空经验库跑一遍、再带经验跑一遍同类算子，对比首轮 kernel 质量与总轮数。
4. 词表一致性：`check_vocab_consistency` + `check_extension_cheatsheet` 均通过。
5. 全量：`.venv/bin/python -m pytest tests/ -v` 全绿。

**④ 保密纪律**：填入的真实 stall 类型名、原语名、仿真字段名、`launch` 实现**均不得回流公开分支**。公开分支保持占位，测试靠 fixture（预置测试数据）跑。`materials/` 加入 `.gitignore`。

---

## 5. T12 的验收（`tests/test_t12_handoff.py`）

- `HANDOFF_GLM47.md` 存在，含材料自举节 + 五个槽位 + 四节其余内容。
- **五个槽位的文件路径与函数签名，与 `control/` 下实际代码逐一对得上**（自动检查，最易错也最致命）。
- 每个槽位任务均包含：编号步骤、自检命令、"缺材料则停下上报"分支。
- 交接包中出现的所有自检命令均可被解析出来且路径真实存在。
- 文档不引用架构文档 / 执行手册 / 本规范（自包含性检查）。

---

## 6. 禁令

- 不在交接包中预设任何真实硬件内容（字段名、stall 类型、原语名）——**只给推导方法**。示例条目必须显式标注为"示例，待替换"。
- 不改动 T1–T11 已冻结的契约、三个 skill 的 description、以及编排器的流程判断归属。
- 不臆造：交接包里出现的每个路径与签名，都必须能在仓库中找到。
