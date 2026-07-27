# Orchestrator 调度逻辑

> Orchestrator = 薄循环。所有决策委托给 RecordManager。

---

## 1. 主循环

```python
run():
    _run_round0()              # Baseline

    while True:
        should_stop, reason, updates = StopChecker.check(state, history)

        if updates:             # Tier 晋升
            state.update(updates)
            save_trajectory()

        if should_stop:
            break

        record = _run_one_round()
        update_state(record)
        append_history(record)
        save_trajectory()

    _finalize()
```

---

## 2. 单轮执行

```python
_run_one_round():
    rd = tier_dir / roundN/    # 自动选目录: 03_tiling_block_config/round5/

    ① _run_analyzers(rd)      # 5 脚本: msprof→hivmir→merger→diagnoser→extractor
    ② _call_planner(diag)     # LLM or AGENT_TASK_PLAN.md
    ③ _call_coder(plan)       # LLM or AGENT_TASK_CODE.md
    ④ vr = _verify_with_retry() # CPU仿真→FAIL→Coder重试×3→Hardware
    ⑤ _decide(vr, plan)       # speedup>1.01? KEEP:REVERT
    ⑥ 返回 RoundRecord
```

### 目录自动选择

```
_round_dir(tier, rn):
  TIER_DIRS[3] = "03_tiling_block_config"
  → outputs/<kernel>/03_tiling_block_config/round5/
```

每个 round 自动进入正确的 Tier 子文件夹。Orchestrator 不需要显式管理路径。

---

## 3. Tier 晋升规则 (StopChecker)

在 `feedback/record_manager.py` 的 `StopChecker.check()` 中:

```python
# 规则 1: 连续 5 轮 REVERT → 晋升
if history[-5:] all REVERT:
    if tier >= 6:  → STOP "All tiers exhausted"
    else:          → tier += 1  (晋升)

# 规则 2: 平台期 → STOP
if history[-10:] speedup variance < 2%: → STOP

# 规则 3: 轮次预算 → STOP
if round >= max_rounds: → STOP

# 规则 4: 目标达成 → STOP
if best_speedup >= target: → STOP

# 规则 5: Tier6 + 3连败 → STOP
if tier >= 6 and consecutive_reverts >= 3: → STOP

# 规则 6: 连续10轮无改进 → STOP
if consecutive_no_improvement >= 10: → STOP
```

### 晋升状态更新

```python
if tier_promoted:
    state.tier = new_tier
    state.consecutive_reverts = 0
    state.consecutive_no_improvement = 0
```

---

## 4. 降级规则

```python
# 在 _run_one_round 内部判断:
if 融合了新算子 (op 数量减少):  → 回到 Tier 3 (Tiling)
if 改了算法 (kernel 结构变化): → 回到 Tier 2 (Fusion)
if 换了 Pipeline (Vector↔Matrix): → 回到 Tier 3

# 实现: 直接设置 state.tier = new_tier
```

降级不是 StopChecker 管的，是 Orchestrator 在 `_run_one_round()` 中根据本轮 plan.strategy 类型判断。

---

## 5. KEEP / REVERT 决策

```python
_decide(vr, plan):
    if not vr.overall_passed:
        return "REVERT", "Verification failed"

    if vr.speedup <= 1.01:
        return "REVERT", f"Speedup {vr.speedup:.3f}x <= 1.01x"

    return "KEEP", f"Speedup {vr.speedup:.3f}x > 1.01x"
```

**REVERT 的自然回退**: `self.current_kernel` 不更新 → 下一轮从上一轮的代码开始。

---

## 6. state 更新逻辑

```python
update_state(record):
    state.round += 1
    state.last_updated = now()

    if record.decision == "KEEP":
        state.consecutive_reverts = 0
        if record.actual_speedup > state.best_speedup:
            state.best_speedup = record.cumulative_speedup
        if record.actual_speedup < 1.01:
            state.consecutive_no_improvement += 1
        else:
            state.consecutive_no_improvement = 0
    else:
        state.consecutive_reverts += 1
        state.consecutive_no_improvement += 1
```

---

## 7. 经验记录规则

```python
record_experience(tier, diag, strategy, speedup, decision):
    if speedup > 1.05:   → status = "SUCCESS"
    elif speedup < 0.98: → status = "FAIL"
    else:                → 不记录 (中性)
```

---

## 8. 完整状态文件

`optimization_trajectory.json` 由 RecordManager 读写。所有 Agent 只读，Orchestrator 通过 RecordManager 写。

```json
{
  "state": {
    "tier": 3,
    "round": 12,
    "best_speedup": 1.52,
    "consecutive_reverts": 0,
    "consecutive_no_improvement": 0
  },
  "history": [
    {"round": 1, "tier": 1, "strategy": "...",
     "actual_speedup": 1.0, "cumulative_speedup": 1.0,
     "decision": "KEEP", "decision_reason": "..."}
  ]
}
```

---

## 9. 文件位置

| 组件 | 文件 |
|---|---|
| 主循环 | `agents/orchestrator.py` |
| 停止条件 + 决策 | `feedback/record_manager.py` → `StopChecker` |
| 经验记录 | `feedback/record_manager.py` → `_record_experience()` |
| 中枢状态 | `outputs/<kernel>/optimization_trajectory.json` |
| 详细架构 | `ARCHITECTURE_DESIGN.md` §2 |
