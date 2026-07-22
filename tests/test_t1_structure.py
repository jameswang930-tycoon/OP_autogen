"""T1 gate: 退役与保留 (retire-and-keep scope).

T1 only retires and keeps; it does NOT create/modify skill bodies.
The final 3-skill set {sim-analyze, triton-gen, extension-guide} is established
by T8 (sim-analyze is triton-plan renamed; extension-guide is new). So this gate
checks exactly what T1 owns:

  - the three retired skills (triton-convert, triton-verify, triton-fix) are gone
  - the kept skills (triton-gen, triton-plan) are NOT over-deleted
  - emulators/ is marked retired (README notice), not deleted
  - costModel/ is untouched by this branch (read-only collaborator repo)
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
RETIRED = {"triton-convert", "triton-verify", "triton-fix"}
KEPT = {"triton-gen", "triton-plan"}  # survive T1; triton-plan -> sim-analyze in T8


def _skills_present():
    if not SKILLS_DIR.exists():
        return set()
    return {p.name for p in SKILLS_DIR.iterdir() if p.is_dir()}


def test_retired_skills_are_deleted():
    present = _skills_present()
    leftover = RETIRED & present
    assert not leftover, f"retired skills still present: {sorted(leftover)}"


def test_kept_skills_not_overdeleted():
    present = _skills_present()
    missing = KEPT - present
    assert not missing, f"kept skills were over-deleted: {sorted(missing)}"


def test_emulators_marked_retired_not_deleted():
    readme = REPO_ROOT / "emulators" / "README.md"
    assert (REPO_ROOT / "emulators").is_dir(), "emulators/ must be kept (not deleted)"
    assert readme.is_file(), "emulators/README.md retirement notice must exist"
    text = readme.read_text(encoding="utf-8")
    assert "退役" in text, "README must state the emulator is retired (退役)"
    assert "历史参考" in text or "新流水线" in text, (
        "README must say it is historical-only / unused by the new pipeline"
    )


def _git(args):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout


def test_costmodel_untouched_by_this_branch():
    # costModel/ is a read-only collaborator repo; this refactor must not touch it.
    # 1) no uncommitted changes to costModel in the working tree
    status = _git(["status", "--porcelain", "--", "costModel"]).strip()
    # 2) no committed changes to costModel introduced on this branch since its fork point
    base = _git(["merge-base", "wsx", "HEAD"]).strip()
    assert base, "could not determine fork point (merge-base wsx HEAD)"
    committed = _git(["diff", "--name-only", base, "HEAD", "--", "costModel"]).strip()
    assert not status, f"uncommitted costModel changes: {status!r}"
    assert not committed, f"committed costModel changes since {base[:8]}: {committed!r}"
