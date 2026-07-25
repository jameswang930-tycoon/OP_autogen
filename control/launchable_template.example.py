# Launchable unit template — public-branch placeholder (任务 C).
# Confidential env: replace this file (at LAUNCHABLE_TEMPLATE_PATH) with the real
# triton.py-based template using the SAME placeholder contract. The framework loads it
# via load_launchable_template(); gen assembles against it. Don't hardcode the format.
#
# Frozen placeholders (see control/launch_template.py LAUNCHABLE_PLACEHOLDERS):
#   OP, SHAPES, DTYPE, KERNEL_BODY, REFERENCE
# The compare section MUST emit the canonical raw_sim_output fields (fixed contract,
# confidential template must satisfy too):
#   correct / max_abs_err / cycles / pipeline / compiled / compile_log

OP = "{{OP}}"
SHAPES = {{SHAPES}}
DTYPE = "{{DTYPE}}"

# === kernel (gen fills) ===
{{KERNEL_BODY}}

# === reference (gold standard, gen fills) ===
{{REFERENCE}}

# === compare + emit raw_sim_output (fixed contract) ===
def _compare():
    # out_kernel = run_kernel(...)
    # out_ref    = run_reference(...)
    # max_abs_err = float(max abs diff)
    # measured_cycles = int(...)
    return {
        "correct": bool(max_abs_err <= TOL),
        "max_abs_err": float(max_abs_err),
        "cycles": int(measured_cycles),
        "pipeline": {unit: cycles},
        "compiled": bool(compiled_ok),
        "compile_log": str(compile_log),
    }
