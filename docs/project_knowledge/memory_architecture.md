# Memory 模块架构

纯标准库、文件后端,不依赖任何外部服务,也不碰 cost model 仓库。本文档描述 `memory/`
包的内部结构与设计;接入流水线的方式见 `memory_integration.md`。

## 结构

```
memory/
  schema.py       数据结构:算子特征 / 经验条目 / 日志记录
  fingerprint.py  从算子规格算出算子特征(检索与写回的键)
  log.py          运行日志(只追加 JSONL)
  store.py        经验库(JSON,含用过/帮上忙/failed/harmed 统计)
  retrieve.py     检索 + 排序 + 拼上下文
  writeback.py    写回:记日志 + 更新统计;手工新增经验
  demo.py         用桩函数演示整条回路
```

## 跑起来

从项目根:
```
python memory/demo.py
```

## 接入现有系统(只两处)

**注入点** —— 在 triton-plan / triton-gen 构造提示词之前:

```python
fp = compute_fingerprint(op_spec, bottleneck=cost_model_bottleneck)
hits = retrieve(store, fp, n=3)
context += format_context(hits)          # 拼进提示词
```

**写回点** —— 在 triton-verify(模拟器)给出结果之后:

```python
record_attempt(log, store, fp,
               retrieved_ids=[e.id for e in hits],
               passed=is_correct,        # 模拟器结论
               kernel_ref=path)
```

新增经验(最小版,成功后手工调用;将来由自动蒸馏替换):

```python
add_experience(store, fp, text="……这次有效的关键做法……")
```

## 明确的预留位(当前不做)

- 自动经验蒸馏 —— 现在 `add_experience` 手工调用,将来由离线蒸馏读日志后自动入库。
- 上板延迟写回 —— `record_attempt` 已留 `latency_us` 字段,精炼循环就绪后再接。
- 语义 / 向量检索 —— 现在是精确 + 放宽键匹配。
- 上下文策略、来源可信度、经验替换、服务化 —— 均未实现,留空。

## 先验证再扩展

接入后立刻做开关对照实验:同一批算子,`retrieve` 返回空 vs 正常返回,比较通过率和达到
正确的迭代数。看到正向信号之前,不要碰上面任何预留位。
