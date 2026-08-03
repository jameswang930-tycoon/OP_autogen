"""Patch main.py + orchestrator.py for triton 3.4.0"""
import re, sys

# Fix main.py
main_path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\main.py"
orch_path = r"D:\vscodeproject\huawei_work\OP_autogen\OP_autogen_hjkc\triton_agent_optimizer\agents\orchestrator.py"

for path in [main_path, orch_path]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace MagicMock imports
    content = re.sub(
        r"from unittest\.mock import MagicMock\n.*?_drv\._obj = MagicMock\([^)]+\)\n.*?_comp\.CompiledKernel = MagicMock\(\)\n",
        "", content, flags=re.DOTALL)

    # Replace import triton + from triton.compiler + from types
    content = re.sub(
        r'\n\s+import triton\n\s+from triton\.compiler import ASTSource\n\s+from types import SimpleNamespace\n',
        '\n        from triton.backends.compiler import GPUTarget\n'
        '        from triton.compiler import ASTSource, compile as triton_compile\n',
        content)

    # Replace signature construction loop (sig[i] → sig[name])
    content = re.sub(
        r"sig\[i\] = \"(\*fp32|i32)\"",
        r'sig[name] = "\1"',
        content
    )

    # Replace ASTSource + make_ir calls
    content = re.sub(
        r'src = ASTSource\(fn=kernel_fn, signature=sig, constants=consts\)\n\s+opts = SimpleNamespace\([^)]+\)\n\s+ttir_module = src\.make_ir\(opts\)\n\s+ttir_text = str\(ttir_module\)',
        'src = ASTSource(fn=kernel_fn, signature=sig, constexprs=consts)\n'
        '        target = GPUTarget("cuda", 90, 32)\n'
        '        result = triton_compile(src, target=target,\n'
        '                               options={"num_warps": 4, "num_stages": 1, "debug": False})\n'
        '        ttir_text = str(result.asm["ttir"])',
        content
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched: {path.split(chr(92))[-1]}")

print("Done")
