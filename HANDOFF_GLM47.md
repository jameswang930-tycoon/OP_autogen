# 交接包：保密环境适配（给 GLM 4.7）

> **一句话贯穿始终**：能观察到的就转录，观察不到的就上报——任何情况下都不要编造字段名、原语名或类别名。
>
> 你不读架构文档、执行手册或任何 spec；本文自包含。每步都有可执行自检；缺材料就**停下、记到 `PROGRESS.md` 的「待确认事项」、上报**，不要猜。

## 环境约定

- 执行脚本一律用 `.venv/bin/python`（系统 `python3` 太旧）。
- 不改任何函数签名 / schema / 词表结构（已冻结）。遇到「这里该怎么设计」→ 停下上报，不要自行发挥。
- 每改一处，跑对应自检，过了再做下一处。

## 第 0 步：材料自举（五个槽位的地基，先做）

1. 找到保密环境里**已经能跑通的仿真调用方式**（用户已打通，是现成代码或命令）。
2. 用它跑 **3–5 个**小算子，把每次的**原始仿真输出**完整保存到 `materials/sim_samples/`。样本覆盖：正常、明显访存慢、明显计算慢、以及一次数值不对的。
3. 找到 extension 包的 API 文档或头文件，路径记到 `materials/README.md`。
4. 若第 1 步找不到可跑通的调用方式 → **停下上报**；下面四个槽位都依赖它。

## 槽位 0：LLM 后端（材料自举之后、词表之前）

编排器需要一个能调通的 LLM 后端，否则第一次 `generate` 即失败。

1. 拿到保密环境里 LLM 服务的**基地址与模型名**（用户已知；是现成服务）。
2. 设环境变量：`LLM_BASE_URL`、`LLM_MODEL`；可选 `LLM_TEMPERATURE`（默认 `0.0`，低值利于复现）、`LLM_TIMEOUT`（默认 `30`）。
3. `control/llm_backend.py` 的 `ConfigurableLLMBackend` 默认按 OpenAI 兼容 chat 格式发请求。若真实后端接口不同，**只改 `_post` 一个方法的请求体/响应解析**，不动类签名与 `generate`/`choose_lever`。
4. 注入编排器：`Orchestrator(job, llm=ConfigurableLLMBackend(), ...)`。
5. 自检：`.venv/bin/python -c "from control.llm_backend import ConfigurableLLMBackend; print(ConfigurableLLMBackend().generate('ping'))"`——能返回内容，且返回内容能过解析闸门（恰好一个 python 块 + 一个 json 块）。
   - 拿不到基地址/模型名，或调不通 → 停下上报，不要猜端点。

## 槽位 1：`control/vocabulary.yaml`（先做，产出需用户确认）

分类判断。冻结字段：每条 `id` / `desc` / `lever` / `primitives`，结构不得改。

1. 打开 `materials/sim_samples/` 全部样本，列出所有代表「代价/停顿/瓶颈」的信号（各类 stall 计数、占用率、流量等），原样罗列，先不合并。
2. 对每个信号回答：「若它是主因，我怎么优化这个 kernel？」答案写在旁边。
3. 按第 2 步答案分组——应对动作相同的合并，动作不同的分开。
4. 收敛到 **5–8 类**。不足 5 类 → 回第 0 步补样本；超过 8 类 → 继续按动作合并。
5. 每类写：`id`（小写下划线）、`desc`（一句话）、`lever`（第 2 步的优化动作）；`primitives` 先留 `[]`。
6. 替换 `control/vocabulary.yaml` 里的 3 条示例条目。
   - 判断粒度唯一标准：两类是否分开，只看它们是否导向**不同的优化动作**。动作相同就合并；无可优化的类属「不可约」，不必单列。
7. 自检：`.venv/bin/python -m control.check_vocab_consistency`
8. 把草案连同「每类对应哪些原始信号」写进 `PROGRESS.md` 待确认事项，停下等用户确认再继续。
   - 缺材料 / 推不出来 → 停下上报，不要猜。

## 槽位 2：`parse_raw()` in `control/feedback_adapter.py`

冻结签名：`parse_raw(raw_sim_output) -> list[Event]`。`Event` 字段已冻结（见下），不得改。

1. 取 `materials/sim_samples/` 中任一份样本，打印其结构（JSON 看键与嵌套层级；文本看行格式）。
2. **先写映射表，再写代码**——在函数上方以注释填满右列：

   | Event 字段 | 类型 | 来自原始输出的哪个字段/位置 |
   |---|---|---|
   | `name` | str | ? |
   | `start` | int | ? |
   | `end` | int | ? |
   | `duration` | int | ?（只有 start/end 则 end−start） |
   | `unit` | str | ?（执行单元/角色） |
   | `stall_class` | str | ?（必须映射到槽位 1 词表的 id） |
   | `bytes` | Optional[int] | ?（无则 None） |

3. 某必填字段在原始输出找不到对应 → 停下上报，不要用常量或猜测值填。
4. 按映射表实现，返回 `list[Event]`。
5. 自检：`.venv/bin/python -m pytest tests/test_t5_adapter.py -q`
6. 端到端：用一份真实样本跑 `adapt(parse_raw(sample))`，检查 7 段摘要与 Verdict；`verdict.bottleneck` 必须是词表里的 id。
   - 缺材料 / 推不出来 → 停下上报，不要猜。

冻结契约（逐字使用）：

- `Event`: name:str, start:int, end:int, duration:int, unit:str, stall_class:str, bytes:Optional[int]
- `Verdict`: bottleneck:str, lever:str, cycles:int, expected_gain:float
- `SimResult`: correct:bool, max_abs_err:float, cycles:Optional[int], pipeline:dict

## 槽位 3：`.claude/skills/extension-guide/references/`

只为词表里的类别找原语，不要通读整个 API。

1. 打开 `control/vocabulary.yaml`，取全部类别 id。
2. 一次只做一类（控制上下文占用）：在 extension 文档里找出能解决该类的原语。
3. 每个原语照 `references/sample_entry.yaml` 的格式填一个 YAML 文件：`name` / `semantics` / `signature` / `category` / `example` / `pitfalls`。格式不得改。
4. 把原语名回填进 `control/vocabulary.yaml` 对应类别的 `primitives` 列表。
5. 某类在 extension 里找不到原语 → `primitives` 保持 `[]` 并在条目标注，不要牵强对应。
6. 自检：`.venv/bin/python -m control.check_extension_cheatsheet`（每条 `category` 必须在词表内）
   - 缺材料 / 推不出来 → 停下上报，不要猜。

## 槽位 4：`launch()` in `control/launch_template.py`

冻结签名：`launch(kernel_file) -> raw_sim_output`（返回值是槽位 2 `parse_raw` 能吃进去的东西）。真实调用是**目录式**：写入输入目录 → 远程脚本执行 → 读输出目录。须交代三个细节：

1. 找到第 0 步那份已能跑通的仿真调用代码/命令。输入目录、输出目录、远程脚本路径**全部走环境变量或配置，不得硬编码**（公开分支不得出现真实路径）。
2. **多轮隔离（最易出错）**：编排器单作业跑 5–8 轮，每轮调一次 `launch()`。每次用 `control/launch_template.py` 的 `new_run_id()` 生成唯一 id，用 run id 区分输入/输出文件名或子目录；**读取结果时校验其确实属于本次 run id**，不匹配视为故障。共享目录不做区分会让性能数据**静默错位**，极难排查。
3. **等待与超时**：目录式提交是异步的，脚本返回不代表结果已写完。轮询等待完成标志 + 设超时；超时或连接故障抛**可识别异常**，编排器按 `sim_retries` 退避重试且不计入轮数。
4. 包一层壳适配签名：入参是 kernel 文件路径，返回**原始仿真输出**。规范 dict 须含 `correct` / `max_abs_err` / `cycles` / `pipeline` / `compiled` / `compile_log`——其中 `compiled` 与 `compile_log` 取自远端**编译结果**（编译是否通过 + 日志），正确性字段取自 compare 段输出、流水取自仿真器产物。编译失败时 `compile_log` 非空、`correct` 置 false、`cycles` 置 null。解析分工见槽位 2 / 槽位 4 末尾说明。
5. 自检：`.venv/bin/python -m pytest tests/test_t6_launch.py -q`
6. 端到端：跑一个小 kernel，确认拿到合法 `SimResult`（`correct` / `max_abs_err` / `cycles` / `pipeline` 齐全）。
   - 缺材料 / 推不出来 → 停下上报，不要猜。

## 槽位 5：`check_extension_calls()` in `control/presim_gate.py`

冻结签名：`check_extension_calls(kernel_src) -> list[str]`（返回空列表表示通过，与占位实现一致）。

1. 从槽位 3 完成的速查表取出所有原语的**签名**。
2. 实现静态检查：kernel 源码中出现的 extension 调用，其原语名是否存在、参数个数/类型是否符合签名。**只静态检查，不执行代码。**
3. 返回问题列表，空列表表示通过。
4. 自检：`.venv/bin/python -m pytest tests/test_t7_gate.py -q`
5. 端到端：写一个故意用错原语的 kernel，确认被挡下。
   - 缺材料 / 推不出来 → 停下上报，不要猜。

## 三条纪律

1. 不改任何函数签名、schema、词表结构——只填内容。
2. 不做架构判断——遇到「这里该怎么设计」停下上报，不要自行发挥。
3. 每改一处，跑对应自检，通过再做下一处。

## 运行说明

- 写 job 文件（YAML）。三种输入形态：`triton_file` / `pytorch` / `shape_only`；**当前完整支持 `triton_file`**。字段：`op` / `input{form,path}` / `shapes` / `dtype` / `budget{max_rounds,epsilon,llm_retries,presim_retries,sim_retries}`。样例见 `jobs/matmul.yaml`。
- 跑：`.venv/bin/python -m control.orchestrator --job jobs/xxx.yaml`
- 产物在 `outputs/<run_时间戳>/`：`report.json`、`recommended.py`、`final_round.py`、`log/`。
- 读 `report.json`：`recommended` 是实测 cycles 最优的那版（可能就是 baseline）；`final_round` 是最后一轮（可能更慢，但保留「为什么变差」的诊断价值）。二者**并列输出**。`rounds[]` 含失败轮（`correct=false` 的轮 `cycles` 为 null）。`stop.reason` 给停止原因（如 `EPSILON` / `max_rounds` / `UNKNOWN_BOTTLENECK` / `BUDGET_*`）。

## 验证清单（五个槽位全完成后自证）

1. 小 kernel 跑通完整闭环，产出 `report.json` + 两份 kernel（`recommended.py` / `final_round.py`）。
2. 停止逻辑三用例：`.venv/bin/python -m pytest tests/test_t3_controller.py -q` 应全绿。
3. 记忆冷启动价值：清空经验库跑一遍、再带经验跑一遍同类算子，对比首轮 kernel 质量与总轮数。
4. 词表一致性：`.venv/bin/python -m control.check_vocab_consistency` 与 `.venv/bin/python -m control.check_extension_cheatsheet` 均通过。
5. 全量：`.venv/bin/python -m pytest tests/ -v` 全绿。

## 保密纪律

- 你填入的真实 stall 类型名、原语名、仿真字段名、`launch` 实现，**均不得回流公开分支**。公开分支保持占位，测试靠 fixture（预置测试数据）跑。
- `materials/` 加入 `.gitignore`，不要提交。
