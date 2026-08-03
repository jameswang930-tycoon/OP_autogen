import sys; sys.path.insert(0, '.')
from main import run_triton_to_hivm
from pathlib import Path
hivm = run_triton_to_hivm(
    Path("input/rms_norm/triton_kernel.py"),
    Path("/tmp/hivm_debug"),
    "rms_norm_kernel")
print("RESULT:", hivm[:300] if hivm else "NONE")
