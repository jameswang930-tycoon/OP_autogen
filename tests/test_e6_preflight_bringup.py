"""E6 gate: preflight（配置自检）+ bringup（逐点联调）。

preflight: 四态状态表（OK/STUB/MISSING/EXAMPLE），纯本地秒级。
bringup: 每个子命令只碰一个接缝，PASS/FAIL 明确。开发机用 mock。
"""
import json

import pytest

from control import preflight, bringup

GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": null, "extension_used": null, "notes": "ok"}\n```')


# ---------------- preflight ----------------

def _states(env=None):
    return {name: state for name, state, _ in preflight.run_checks(env=env)}


def test_preflight_all_placeholder_env_marks_stub_example():
    st = _states(env={})  # nothing configured
    assert st["llm"] == "STUB"
    assert st["nga"] == "STUB"            # dev machine has no nga
    assert st["parse_raw"] == "STUB"      # slot still NotImplementedError
    assert st["vocabulary"] == "EXAMPLE"  # still the 3 example entries


def test_preflight_llm_ok_when_model_configured():
    st = _states(env={"NGA_GENERATE_MODEL": "real/model", "NGA_CHOOSE_LEVER_MODEL": "w3/GLM-4.7"})
    assert st["llm"] == "OK"


def test_preflight_missing_path_marked_missing(tmp_path):
    st = _states(env={"PRESIM_SIGNATURE_TABLE": str(tmp_path / "nope.yaml")})
    assert st["signature_table"] == "MISSING"


def test_preflight_signature_table_ok_when_real_file(tmp_path):
    p = tmp_path / "sig.yaml"
    p.write_text("- name: real_prim\n  param_counts: [2]\n", encoding="utf-8")
    st = _states(env={"PRESIM_SIGNATURE_TABLE": str(p)})
    assert st["signature_table"] == "OK"


def test_preflight_states_are_known_vocab():
    allowed = {"OK", "STUB", "MISSING", "EXAMPLE"}
    for _, state, _ in preflight.run_checks(env={}):
        assert state in allowed, f"unknown state {state!r}"


# ---------------- bringup ----------------

class _OKLLM:
    def generate(self, p): return GEN
    def choose_lever(self, p): return GEN


class _BadLLM:
    def generate(self, p): return "no fenced blocks at all"
    def choose_lever(self, p): return ""


def test_bringup_template_passes():
    passed, msg = bringup.bringup_template()
    assert passed, msg


def test_bringup_llm_pass_with_good_backend():
    passed, msg = bringup.bringup_llm(_OKLLM())
    assert passed, msg


def test_bringup_llm_fail_with_bad_backend():
    passed, msg = bringup.bringup_llm(_BadLLM())
    assert not passed
    assert "python" in msg.lower() or "json" in msg.lower() or "block" in msg.lower()


def test_bringup_launch_pass_and_saves_raw(tmp_path):
    raw = {"correct": True, "max_abs_err": 0.0, "cycles": 100, "pipeline": {},
           "compiled": True, "compile_log": ""}

    def launcher(path): return raw
    passed, msg = bringup.bringup_launch(launcher, save_dir=tmp_path)
    assert passed, msg
    assert json.loads((tmp_path / "last_raw.json").read_text(encoding="utf-8"))["cycles"] == 100


def test_bringup_parse_pass_on_fixture_raw(tmp_path):
    # use a fixture events list as the "raw" -> parse_raw mock returns it; adapt must succeed
    from control.fixtures import COMPUTE_BOUND

    def parse_raw(raw): return COMPUTE_BOUND
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps({"correct": True}), encoding="utf-8")
    passed, msg = bringup.bringup_parse(raw_file, parse_raw_fn=parse_raw)
    assert passed, msg


def test_bringup_extcheck_pass_on_clean_kernel(tmp_path):
    k = tmp_path / "k.py"
    k.write_text("def kernel(a, b, c):\n    return a @ b\n", encoding="utf-8")
    passed, msg = bringup.bringup_extcheck(k)
    assert passed, msg
