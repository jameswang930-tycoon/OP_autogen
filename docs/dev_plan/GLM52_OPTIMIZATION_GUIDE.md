# GLM 5.2 框架优化指导（local-adapt 分支）

> 读者:开发机 GLM 5.2。基于当前 GitHub `local-adapt` 分支的公开框架优化。
> 目标:改进 prompt 组织、memory、原语呈现等,让弱模型少走弯路;**但绝不破坏保密环境已适配的两个接缝**。
> 所有内容不涉密——讲的是架构与方法,不含真实原语名/路径/硬件细节。

---

## 第零部分:硬约束——绝不可破坏的两个接口(最高优先,先读)

保密环境花了极长时间适配 `ext_distill`(蒸馏语料/签名表)和 `remote_dsl`(远端仿真调用)两个接缝。你的任何优化**必须保持这两个接口的契约不变**,否则框架拉回保密环境要重做适配。这是不可谈判的。

### 接口A:ext_distill 对接(extension 语料/签名表的读取契约)

**契约(不可改)**:
- `orchestrator.py:40` `EXT_REFS = SKILLS_DIR / "extension-guide" / "references"` —— extension 速查表从这个目录读,**路径约定不可改**。
- 速查表以 **index(索引)形式**被读入 prompt(`placeholders.py:28` 注明"extension-guide is read as an index — no placeholders")——**这个"作为索引读、无占位符"的方式不可改**。
- `signature_table` 通过 `PRESIM_SIGNATURE_TABLE` 环境变量指向,`presim_gate.load_signature_table` 读取——**环境变量名、加载函数签名不可改**。
- `build_signature_table.py` 从 `api_inventory.txt` 生成签名表的**输入输出格式不可改**(保密环境靠它生成真实签名表)。
- `check_extension_cheatsheet.py` / `check_vocab_consistency.py` 的校验契约不可改。

**你可以改的**:extension index **怎么组织进 prompt**(见第二部分——按场景检索、加模块归属),但**读取路径、索引形式、校验接口这些契约不动**。

### 接口B:remote_dsl 对接(远端仿真调用契约)

**契约(不可改)**:
- `launch_template.py:105` `launch(kernel_file: str) -> dict` —— 签名不可改。它是保密环境实现的槽位。
- `raw_sim_output` 规范 schema(launch 必须按此返回,launch_template.py:14 注明)——**字段契约不可改**:`correct/max_abs_err/cycles/pipeline/compiled/compile_log`。
- `new_run_id()`(:99)+ 多轮结果隔离约定(T13-2)——**不可改**,保密环境的时间戳目录定位依赖它。
- launch 失败分类异常(E4,:37-42)——异常体系可扩展但**已有的不可删/改语义**。
- `build_sim_result(raw)`(:11)把 raw 转 SimResult 的契约不可改。
- SIM_* 环境变量(SIM_ROOT/SIM_SCRIPT/SIM_INPUT_DIR/SIM_RESULT_DIR/SIM_TIMEOUT 等)——**变量名不可改**,保密环境的 env.sh 靠它们配置。

**你可以改的**:launch **内部**你碰不到(是保密环境槽位,本就是 NotImplementedError);你能碰的是**它的契约周边**(失败分类、build_sim_result),这些**只扩展不破坏**。

### 契约冻结清单(动它们=破坏适配)

```
contracts.py: Event / Verdict / SimResult 三个 dataclass 字段
launch_template.py: launch() 签名、raw_sim_output schema、new_run_id、build_sim_result
feedback_adapter.py: parse_raw(raw_sim_output) -> list[Event] 签名
presim_gate.py: check_extension_calls 签名、load_signature_table
orchestrator.py: EXT_REFS 路径、extension作为index读取
所有 SIM_* / PRESIM_* / NGA_* / LAUNCHABLE_* 环境变量名
```

**优化前先问:这个改动会不会改到上述任何一项?会 → 停,换个不破坏契约的做法。**

---

## 第一部分:prompt 组织重构(核心优化)

### 问题(保密环境实测踩的坑)

当前 `build_gen_prompt`(orchestrator.py:98)把所有东西揉进一个大字符串:角色定义、规则、verdict、经验、extension index。踩的坑:
- **数据内联进规则区**:verdict/experience 被复制进了规则说明文本(出现两次),污染了 prompt 结构,让模型困惑;
- prompt 又长又杂,弱模型抓不住重点。

### 优化方向:角色(instruction) + 数据(结构化)分离

借鉴"instruction 传稳定角色 + 单独通道传变化数据"的模式:
- **instruction/system 部分(稳定)**:"你是 kernel 生成器,输出格式是 python块+json块,只填占位符不写完整文件" —— 每轮一样,固定;
- **数据部分(每轮变化)**:verdict、检索到的经验、相关原语 —— 作为**结构化的数据段**给,清晰标注"这是输入数据",**不要内联进规则/指令文本**;
- **规则区只做指向性引用**:如"根据下方 verdict 的 bottleneck 选择优化方向",而不是把 verdict 的完整内容抄进规则条件。

**效果**:消除数据内联污染、prompt 清爽、角色稳定可复用。

### prompt 精简原则

精简 = **相关性**,不是长度。只给当前任务相关的:
- 相关原语(见第二部分,按场景检索,不全量塞);
- top-k 相关经验(memory retrieve,已有界);
- 不塞无关规则、不重复内联。

---

## 第二部分:extension 原语的呈现(解决幻觉与误用)

### 问题(保密环境实测)

1. **幻觉原语**:模型写出不存在的 `tlext1.add`——因为清单给了原语名(add),few-shot 是 `tlext1.X` 模式,模型合理推断出 tlext1.add,但 add 不在 tlext1。**根因:清单有名字,缺"模块归属"。**
2. **误归类污染候选**:`img2col` 按"减访存"归到 memory_bound_throughput,于是出现在所有 memory bound 算子(含 element-wise add)的候选里,但 img2col 只适用卷积。**根因:只按瓶颈归类,缺"适用场景"维度。**

### 优化方向:extension index 加两个维度 + 按场景检索

**维度补充**(index 组织时,不改读取路径契约):
- **模块归属**:每个原语给完整限定名(`模块.原语(签名)`),而非孤零零的名字——避免模型猜模块;
- **适用场景**:每个原语标"用于什么操作"(如 img2col→仅卷积),避免污染不相关候选。

**按场景检索(lazy-loading 的正确形态)**:
- 不把全部原语塞进 prompt,也不按瓶颈把整类几十个都给;
- **仿照 memory 的 retrieve**,写一个"按当前算子/瓶颈检索相关原语子集"的逻辑,只把相关的拼进 prompt;
- 这既省上下文,又天然过滤不适用的(add 算子不会看到 img2col)。
- **注意**:检索逻辑是新增的确定性代码,读取的仍是 EXT_REFS 下的 index(契约不变),只是"读多少/读哪些"变智能了。

### 语料质量原则(重要教训)

保密环境发现:**语料的"有"不等于"对"**。
- `api_inventory.txt` 只是名字索引,**模块归属不可信**(实测系统性错标);
- 错误示例会**主动教坏模型**(如错误的 tlext1.add 示例让模型照抄);
- **证据优先级:真实kernel用法 > 手册 > api_inventory**。
- 框架侧你无法接触真实语料(在保密环境),但**要在设计上支持"证据分级"**:比如 index 里每个原语可带"证据来源/置信度"字段,让检索时能优先给高置信度的。

---

## 第三部分:memory 修复(核心欠债)

### 两个 bug(保密环境实测,让 memory 名存实亡)

**Bug 1:fingerprint key mismatch**
- `_retrieve_experience` 用 `Fingerprint(op_kind, bottleneck=None)`,key 变成 `"op|unknown"`,但存的是 `"op|真实bottleneck"` → 精确匹配永远 miss,只靠 op_kind 回退命中 → 失去瓶颈区分度。
- **修**:retrieve 时用**已知的当前 bottleneck**(round2 时已有上轮 verdict,bottleneck 是知道的)构建 fingerprint,不要用 None。

**Bug 2:retrieved_ids=[] 硬编码**
- `record_attempt(..., retrieved_ids=[])` 传空列表 → 经验的 used/helped 计数永不更新 → 分数永远初始值 → **无法区分好经验和坏经验**。
- **修**:把 retrieve 实际返回的 experience IDs 传给 record_attempt,让分数真正迭代。

### 深层风险:误导性经验自我强化

保密环境观察到:round1 的错误做法(如用了幻觉原语)被 record 成经验,round2 retrieve 又喂回来,**错误被强化**。
- Bug 2 是根源之一:分数不更新 → 烂经验和好经验一样被捞回。
- **修好 Bug 2 后**,失败/低质的尝试分数会低、被降权,不再无脑喂回。
- 建议设计上支持:经验带"来源/证据强度",retrieve 对低质经验降权;支持标记经验失效(否决机制)。

### memory 的定位(别退回 agent)

memory 的 retrieve 是**有界 top-k**(不随轮数膨胀)——这是对的,保持。它是"确定性检索相关子集",不是 agent 自主决策。

---

## 第四部分:职责分离原则(别拆成多个 LLM agent)

### 原则:代码做确定的事,LLM 只做创造的事

保密环境和 cost model 的演进历史都验证了这条线:

| 职责 | 性质 | 谁做 |
|---|---|---|
| 瓶颈分析 | 确定性算法(取占时最大+critical path) | **代码**(feedback_adapter) |
| 编排统筹 | 确定性流程控制 | **代码**(orchestrator) |
| 远程校验 | 确定性调用+解析 | **代码**(launch) |
| **代码生成** | **创造** | **LLM**(唯一的模型调用) |

**关键**:瓶颈分析最初是 skill(agent形态,cost_emulator/Skills/bottleneck-analysis),但已**主动演进成 feedback_adapter.py 代码**(继承了"展开→算时长→依赖→调度→取占时最大"的思想,数据换成真实仿真)。**这个演进是对的,不要退回 skill/agent 形态。**

**所以**:
- 不要把瓶颈分析/编排/校验"拆成 LLM agent"——它们是确定性算法,该是代码;
- "职责分离"已经做到了(代码扮演大部分角色,LLM 只扮演生成);
- 真正要改进的是"喂给生成 LLM 的 prompt 组织"(第一部分),不是拆 agent。

---

## 第五部分:静默失效教训(防御性设计)

保密环境踩的坑,都是"核心模块失效但不报错":
- memory `store=None` 时 record_attempt **静默 return** → memory 全程没工作,没人发现;
- example 模板的 `_compare()` 是死代码,`bringup PASS` 只验语法不验语义,gap 被掩盖。

**原则**:
- 核心模块失效必须**报 warning,不许静默 return/跳过**;
- mock 层/示例的 PASS **不等于**真实层对齐——测试要尽量贴近真实契约;
- 提供的 example(如 launchable_template.example.py)要和真实结构同构,不要凭空造(保密环境发现 example 模板缺必需段、含死代码)。

---

## 第六部分:文档卫生(硬规矩)

**只保留一个 PROGRESS.md 作为进展文档,中间过程的临时 markdown 用完即清,不要残留。**

- 优化过程中的分析/审计类临时 md,完成后删除,不要堆积;
- **PROGRESS.md 更新规则:只追加、只改状态标记,绝不删除已有的"决定/踩坑记录"章节**(保密环境吃过亏——整体重写覆盖丢失了宝贵记录);
- 如需重构 PROGRESS.md 结构,先确认无信息丢失;
- 重要的"为什么这么改"的决定,记进 PROGRESS.md 的决定区(只增不减)。

---

## 优化优先级

```
0. 先读第零部分——确认不破坏 ext_distill / remote_dsl 契约(硬约束)
1. prompt 组织重构(instruction+数据分离,消除内联污染)   ← 最实在的红利
2. memory 两个 bug(fingerprint / retrieved_ids)          ← 核心欠债
3. extension index 加模块归属+场景 + 按场景检索           ← 解决幻觉/误用
4. 静默失效防御(核心模块失效报warning)
5. 文档卫生(只留PROGRESS.md,只追加)
```

**每项改完跑全量 pytest 确认不破坏契约。改完在 PROGRESS.md 追加记录。**

---

## 一句话总结

在**不碰 ext_distill / remote_dsl 两个接缝契约**的前提下,优化"喂给生成 LLM 的信息组织"——prompt 角色/数据分离、原语按场景检索、memory 真正工作。保持确定性骨架(分析/编排/校验是代码),LLM 只做生成。这些改完,框架拉回保密环境能秒适配(契约没变),而弱模型能少走弯路。
