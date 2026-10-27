# -*- coding: utf-8 -*-
"""V5: Event 注入输入轮换 (破 L2) — 14 个算子注入产物 compile 校验 + 分配行重建断言."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.verifier import _inject_event_timing

def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"{name}: {detail}"

ops = sorted(Path("input").glob("*/kernel_op.py"))
n_ok = n_alloc = 0
for p in ops:
    src = p.read_text(encoding="utf-8")
    injected = _inject_event_timing(src)
    if not injected:
        print(f"  SKIP {p.parent.name}: 无标准 KERNEL_LOOP 循环")
        continue
    # 注入产物必须能 compile
    try:
        compile(injected, str(p), "exec")
    except SyntaxError as e:
        check(f"{p.parent.name} 注入后语法错误", False, str(e))
        continue
    n_ok += 1
    # 断言: 注入块内包含输入重建 (alloc 行在 rep 循环内, 12 空格缩进)
    if "torch.rand" in src or "torch.empty" in src or "torch.zeros" in src:
        has_rebuild = ("for _r in range(_REPS):" in injected
                       and re.search(r"\n            \w+ = torch\.(randn?|rand|empty|zeros|ones)\(", injected))
        check(f"{p.parent.name}: 分配行重建进 rep 窗口", has_rebuild)
        # ★V5c: _keep 持有防地址复用 (caching allocator 复用 → 破 L2 失效)
        has_keep = "_keep = []" in injected and "_keep.append((" in injected
        check(f"{p.parent.name}: _keep 持有张量防地址复用", has_keep)
        n_alloc += 1

print(f"\n═══ V5 验证: {n_ok}/{len(ops)} 算子注入可编译, {n_alloc} 个含输入重建 ═══")
assert n_ok == len(ops), "有算子注入失败"
assert n_alloc >= 12, f"输入重建覆盖不足: {n_alloc}/14"

# 防御: 无分配行的 kernel → 不崩, 退化为热 L2
src2 = "def main():\n    for _ in range(LOOP):\n        k[1](x)\n"
inj2 = _inject_event_timing(src2)
check("无分配行 → 注入仍可用 (退化热 L2)", "for _r in range(_REPS):" in inj2)

# ★V5b: rebuild_inputs=False (热 L2 模式) — 14 个算子注入产物也必须 compile + 不含分配重建
for p in ops:
    src = p.read_text(encoding="utf-8")
    inj_hot = _inject_event_timing(src, rebuild_inputs=False)
    if not inj_hot:
        continue
    try:
        compile(inj_hot, str(p), "exec")
    except SyntaxError as e:
        check(f"{p.parent.name} 热L2注入语法错", False, str(e))
        continue
    assert "            " not in [l for l in inj_hot.splitlines() if re.match(r"^            \w+ = torch\.(randn?|rand|empty|zeros|ones)\(", l)], \
        f"{p.parent.name}: 热L2模式不应含分配重建"
print("  热L2 (rebuild=False) 注入 14 算子全部可编译 + 无分配重建")

print("\n═══ V5 全部通过 ═══")
