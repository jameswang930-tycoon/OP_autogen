# -*- coding: utf-8 -*-
"""验证 coder 抄写防御: prompt 禁止指令 + 语法校验拦截 + 注释不误伤."""
import sys
sys.path.insert(0, ".")
from agents.coder import _build_user_prompt, _validate_python

# 1) prompt 含禁止抄写指令
prev_err = "unsupported op for finding the root alloc: 这条 load 夹了 vsel(来自 tl.where)..."
p = _build_user_prompt('{"strategy":"x"}', "code", previous_error=prev_err)
assert "只读参考, 严禁抄写" in p, "prompt 缺禁止抄写指令"
assert "unterminated string literal" in p, "prompt 缺后果提示"
print("1) prompt 禁止抄写指令 OK")

# 2) 模拟 LLM 把解释文本抄进代码 (裸文本行, 含英文单引号) → 语法校验必须拦下
bad = """import triton, triton.language as tl
@triton.jit
def k(a, b, c, BLOCK: tl.constexpr):
    pass
    per the coding_guide/skill.md, this is caused by 'vsel' (vector select) in the ...
"""
ok, err = _validate_python(bad)
assert not ok and "line" in err, err
print("2) 裸文本抄写被拦下 OK:", err[:70])

# 3) 单引号字符串未闭合场景 (LLM 最常犯) → 拦下
bad2 = """import triton
x = 'per the coding_guide, this is caused by 'vsel' (vector select) in the ...
"""
ok2, err2 = _validate_python(bad2)
assert not ok2 and "unterminated" in err2, err2
print("3) 未闭合字符串被拦下 OK:", err2[:80])

# 4) 注释版抄写 (合法, 不应误伤)
ok3, _ = _validate_python("""import triton, triton.language as tl
# per the coding_guide/skill.md, this is caused by 'vsel' (vector select) in the load chain
@triton.jit
def k(a, b, c, BLOCK: tl.constexpr):
    pass
""")
assert ok3, "注释版应合法"
print("4) 注释版抄写不误伤 OK")
print("ALL_OK")
