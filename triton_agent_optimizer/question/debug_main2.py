import sys, os, json
sys.path.insert(0, '.')
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile
import importlib.util
from pathlib import Path

kernel_path = Path("input/rms_norm/triton_kernel.py")
spec = importlib.util.spec_from_file_location(kernel_path.stem, str(kernel_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, "rms_norm_kernel")
print(f"fn={fn}, has fn={hasattr(fn, 'fn')}, arg_names={fn.arg_names}")

# Same as main.py
sig = {}
consts = {}
for name in fn.arg_names:
    nu = name.upper()
    if "BLOCK" in nu or nu.startswith("BLOCK"):
        consts[name] = 256
    elif "DIM" in nu or "HIDDEN" in nu:
        consts[name] = 4096
    elif "EPS" in nu or "eps" == name:
        consts[name] = 1e-5
    elif nu.endswith("_PTR") or nu.endswith("_ptr") or name.lower() in (
            "x_ptr", "y_ptr", "a_ptr", "b_ptr", "c_ptr", "out_ptr",
            "weight_ptr", "residual_ptr"):
        sig[name] = "*fp32"
    else:
        sig[name] = "i32"

print(f"sig={json.dumps(sig)}, consts={json.dumps(consts)}")

src = ASTSource(fn=fn, signature=sig, constexprs=consts)
target = GPUTarget("cuda", 90, 32)
print(f"src created, calling compile...")
try:
    result = triton_compile(src, target=target,
                           options={"num_warps": 4, "num_stages": 1, "debug": False})
    ttir = str(result.asm["ttir"])
    print(f"TTIR: {len(ttir)} chars")
    print(f"First 200: {ttir[:200]}")
except Exception as e:
    print(f"compile FAILED: {e}")
    import traceback; traceback.print_exc()
