# Environment & Running

How to run this project's Python components. If a command fails with a `torch`
import error or a `str | None` SyntaxError, it is almost certainly the wrong
interpreter — see the rule below.

## The one rule: use `.venv/bin/python`

Always run Python from the repo root with `.venv/bin/python`. The system `python3`
does **not** work for anything in this project:

| Interpreter | Version | numpy | torch | Runs |
|---|---|---|---|---|
| `.venv/bin/python` | 3.13 | 2.4.6 | 2.12.0 | everything |
| system `python3` | 3.7.12 | 1.21.6 | **missing** | nothing useful |

The system `python3` (3.7) fails two independent ways:
- **No `torch`** — every emulator `reference_*()` and `run_all_tests.py` imports torch.
- **Below 3.10** — `costModel/cost_emulator/simulator.py` uses PEP 604 unions (`str | None`).

`.venv/` is a [uv](https://docs.astral.sh/uv/)-managed CPython 3.13 venv and is gitignored
(not in the repo). There is **no checked-in dependency manifest** (`pyproject.toml` /
`requirements.txt` / `uv.lock`) — the runtime deps are just `numpy` and `torch` (the
emulator is otherwise pure-Python). Recreate with uv and `uv pip install numpy torch`.

## Running each component

All commands assume `cwd = repo root` (`OP_autogen/`).

### Cost-model simulator (used by `/triton-plan`)

```bash
.venv/bin/python costModel/cost_emulator/simulator.py --verify "<DSL>"
.venv/bin/python costModel/cost_emulator/simulator.py --llm --critical-path "<DSL>"
```

The simulator parses `sys.argv` by hand — there is **no `--help`**; with no DSL it
prints `(no operations found in input)`. The `--llm` output is dumped verbatim as
`raw_llm` (see `plan_code_contract.md`). DSL/engine reference:
`costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md`.

### Emulator self-tests

```bash
.venv/bin/python emulators/test/run_all_tests.py        # all operators
```

`run_all_tests.py` puts its own directory on `sys.path`, so run it from the repo
root, not from inside `emulators/test/`. Single-operator inline verify:

```bash
cd emulators && ../.venv/bin/python -c "from test.<op> import test; test()"
```

or in Python via `run_with_feedback(emulate_<op>, reference_<op>)` — see
`emulators/common/__init__.py`.
