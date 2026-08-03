"""
Upgrade main.py and orchestrator.py from triton 2.3.1 to 3.4.0 API.
Replaces:
  - MagicMock driver setup → GPUTarget
  - src.make_ir(opts) → triton_compile(src, target=GPUTarget(...), options=...)
  - sig[i] → sig[name] (string keys)
  - constants → constexprs
"""
import re

def upgrade_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content

    # 1. Remove MagicMock imports (4 lines: from unittest... through _comp.CompiledKernel...)
    content = re.sub(
        r'\n\s+from unittest\.mock import MagicMock\n'
        r'\s+import triton\.runtime\.driver as _drv\n'
        r'\s+_drv\._obj = MagicMock\([^)]+\)\n'
        r'\s+import triton\.compiler\.compiler as _comp\n'
        r'\s+_comp\.CompiledKernel = MagicMock\(\)\n',
        '\n', content)

    # 2. Replace 'import triton\nfrom triton.compiler import ASTSource\nfrom types import SimpleNamespace'
    content = re.sub(
        r"(\n\s+)import triton\n"
        r"\s+from triton\.compiler import ASTSource\n"
        r"\s+from types import SimpleNamespace\n",
        r"\1from triton.backends.compiler import GPUTarget\n"
        r"\1from triton.compiler import ASTSource, compile as triton_compile\n",
        content)

    # 3. Change sig[i] to sig[name] (all occurrences in these functions)
    content = re.sub(r'sig\[i\] = ', r'sig[name] = ', content)

    # 4. Change signature building loop
    # Old: for i, name in enumerate(...)
    # New: for name in kernel_fn.arg_names:
    content = re.sub(
        r'for i, name in enumerate\(kernel_fn\.arg_names\):',
        r'for name in kernel_fn.arg_names:', content)

    # 5. Remove the n_args/ptrcount logic that's no longer needed
    content = re.sub(
        r'\n\s+n_args = len\(kernel_fn\.arg_names\)\n\s+sig = \{\}\n\s+consts = \{\}\n\s+ptr_count = 0\n',
        r'\n        sig = {}\n        consts = {}\n', content)

    # Also remove the i32 fallback block
    content = re.sub(
        r'\n\s+# 确保至少一个 i32 参数[^\n]*\n.*?last_idx in consts:[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*if last_idx >= 0:\n[^\n]*sig\[last_idx\][^\n]*\n',
        '\n', content, flags=re.DOTALL)

    # 6. Replace ASTSource + make_ir call
    content = re.sub(
        r"src = ASTSource\(fn=kernel_fn, signature=sig, constants=consts\)\n"
        r"\s+opts = SimpleNamespace\([^)]+\)\n"
        r"\s+ttir_module = src\.make_ir\(opts\)\n"
        r"\s+ttir_text = str\(ttir_module\)",
        'src = ASTSource(fn=kernel_fn, signature=sig, constexprs=consts)\n'
        '        target = GPUTarget("cuda", 90, 32)\n'
        '        result = triton_compile(src, target=target,\n'
        '                               options={"num_warps": 4, "num_stages": 1, "debug": False})\n'
        '        ttir_text = str(result.asm["ttir"])',
        content)

    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

if __name__ == "__main__":
    base = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer"
    for f in ["main.py", "agents/orchestrator.py"]:
        path = base + "\\" + f
        ok = upgrade_file(path)
        print(f"{f}: {'UPGRADED' if ok else 'no changes needed'}")
