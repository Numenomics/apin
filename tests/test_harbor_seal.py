"""The Harbor benchmark containers hold no answers and give the agent no open network.

Fast tests check the staged build context (scripts/build_harbor_base.py) and the task
files. The slow test inspects built task images; build them first with
`uv run python scripts/build_harbor_base.py --tasks`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "harbor" / "tasks"
TASK_NAMES = sorted(p.parent.name for p in TASKS.glob("*/task.toml"))

# Paths that must never reach an agent container: evolved code, example samplers,
# hidden rewind fixtures and reference fixes, the verifier, results, history.
LEAK_PATHS = (
    ".git", "tests", "results", "scripts", "flywheel", "examples", "benchmarks",
    "upstream", "harbor", "runs", "build", "scores", "jobs", ".venv", ".cache",
    "autoresearch/lanes/S1", "autoresearch/lanes/S2", "autoresearch/lanes/S3",
    "autoresearch/task_catalog.json", "REPORT.md", "ARCHIVE.md",
)
LEAK_NAMES = {"runs", "archive", "incumbent.py", "__pycache__", "fixtures", ".git"}
ALLOWED_TOP = {"LICENSE", "pyproject.toml", "uv.lock", "harness", "models", "data", "codex",
               "autoresearch", "Dockerfile"}


def _load_builder():
    path = ROOT / "scripts" / "build_harbor_base.py"
    spec = importlib.util.spec_from_file_location("build_harbor_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task(name: str) -> dict:
    return tomllib.loads((TASKS / name / "task.toml").read_text())


def _optimize_baselines() -> dict[str, bytes]:
    out = {}
    for name in TASK_NAMES:
        meta = _task(name)["metadata"]
        if meta["category"] == "optimize":
            out[meta["lane"]] = (TASKS / name / "environment" / "baseline.py").read_bytes()
    return out


def _task_copies(name: str) -> dict[str, bytes]:
    """Container path -> bytes for each `COPY <file> <dest>` in a task Dockerfile."""
    env = TASKS / name / "environment"
    out = {}
    for line in (env / "Dockerfile").read_text().splitlines():
        m = re.match(r"\s*COPY\s+(\S+)\s+(/\S+)\s*$", line)
        if m:
            out[m.group(2)] = (env / m.group(1)).read_bytes()
    return out


def _expected_files(name: str) -> dict[str, bytes]:
    """Every submission file the agent of task `name` starts from, by container path."""
    files = {f"/app/autoresearch/lanes/{lane}/submission.py": code
             for lane, code in _optimize_baselines().items()}
    files.update(_task_copies(name))
    return files


@pytest.fixture(scope="module")
def context(tmp_path_factory) -> Path:
    return _load_builder().stage(tmp_path_factory.mktemp("harbor") / "ctx")


def test_staged_context_holds_only_the_allowlist(context: Path):
    top = {p.name for p in context.iterdir()}
    assert top <= ALLOWED_TOP, f"unexpected top-level entries: {sorted(top - ALLOWED_TOP)}"
    for rel in LEAK_PATHS:
        assert not (context / rel).exists(), f"leak path staged: {rel}"
    for p in context.rglob("*"):
        assert p.name not in LEAK_NAMES, f"leak path staged: {p.relative_to(context)}"
    lanes = {p.name for p in (context / "autoresearch" / "lanes").iterdir()}
    assert lanes == set(_optimize_baselines())


def test_staged_context_is_enough_to_run_the_harness(context: Path):
    for rel in ("pyproject.toml", "uv.lock", "harness/cli.py", "harness/judge.py",
                "models/manifest.json", "codex/INDEX.yaml", "autoresearch/RULINGS.md"):
        assert (context / rel).is_file(), rel
    for lane in _optimize_baselines():
        for doc in ("program.md", "NEGATIVE_MEMORY.md"):
            assert (context / "autoresearch" / "lanes" / lane / doc).is_file(), (lane, doc)


def test_every_lane_starts_from_its_frozen_baseline(context: Path):
    for lane, code in _optimize_baselines().items():
        staged = (context / "autoresearch" / "lanes" / lane / "submission.py").read_bytes()
        assert staged == code, f"lane {lane} submission.py is not its frozen starting code"


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_image_copies_its_own_starting_code(name: str):
    builder = _load_builder()
    text = (TASKS / name / "environment" / "Dockerfile").read_text()
    assert text.splitlines()[0] == f"FROM {builder.BASE_IMAGE}:{builder.RELEASE}"
    copies = _task_copies(name)
    assert copies, "task image copies no starting code"
    baseline = (TASKS / name / "environment" / "baseline.py").read_bytes()
    assert set(copies.values()) == {baseline}
    if _task(name)["metadata"]["category"] == "rewind":
        hidden = (TASKS / name / "tests" / "tasks.json").read_text()
        for task in json.loads(hidden)["tasks"]:
            assert "fixtures" not in text and task["reference_submission"] not in text


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_timeouts_and_network(name: str):
    cfg = _task(name)
    category = cfg["metadata"]["category"]
    assert cfg["agent"]["timeout_sec"] == 86400.0
    assert cfg["verifier"]["timeout_sec"] == (7200.0 if category == "optimize" else 1800.0)
    assert cfg["agent"]["network_mode"] == "allowlist"
    assert cfg["agent"].get("allowed_hosts", []) == []
    assert cfg["verifier"]["network_mode"] == "no-network"
    assert "allow_internet" not in cfg["environment"]


@pytest.mark.parametrize("name", TASK_NAMES)
def test_verifier_grades_only_the_submission_in_a_fresh_container(name: str):
    cfg = _task(name)
    assert cfg["verifier"]["environment_mode"] == "separate"
    assert cfg["verifier"]["environment"]["network_mode"] == "no-network"
    # No tests/Dockerfile: Harbor builds the verifier from environment/, the sealed image.
    assert not (TASKS / name / "tests" / "Dockerfile").exists()
    test_sh = (TASKS / name / "tests" / "test.sh").read_text()
    submission = re.search(r"--submission\s+(\S+)", test_sh).group(1)
    assert cfg["artifacts"] == [submission]
    assert submission in _task_copies(name)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_instruction_states_no_time_budget(name: str):
    text = (TASKS / name / "instruction.md").read_text().lower()
    assert not re.search(r"\b\d+\s*(min|minutes|hours?|h)\b|time (budget|limit)", text)


_PROBE = r"""
import hashlib, json, os, sys
leaks = [p for p in sys.argv[1:] if os.path.lexists(os.path.join("/app", p))]
for d, dirs, files in os.walk("/app"):
    if d.startswith("/app/.venv"):
        continue
    leaks += [os.path.join(d, n) for n in dirs + files
              if n in ("runs", "archive", "incumbent.py", "fixtures", ".git")]
subs = {}
for d, _, files in os.walk("/app"):
    if not d.startswith("/app/.venv"):
        subs.update({os.path.join(d, f): hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
                     for f in files if f == "submission.py"})
print(json.dumps({"leaks": leaks, "subs": subs}))
"""


@pytest.mark.slow
@pytest.mark.parametrize("name", TASK_NAMES)
def test_built_task_image_is_sealed(name: str):
    builder = _load_builder()
    image = os.environ.get("APIN_SEAL_IMAGE", f"apin-harbor-task-{name}:{builder.RELEASE}")
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip(f"{image} not built")
    in_image = [p for p in LEAK_PATHS if p != ".venv"]  # the image's own venv is expected
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image, "python3", "-c", _PROBE, *in_image],
        capture_output=True, text=True, check=True,
    )
    probe = json.loads(out.stdout)
    assert probe["leaks"] == []
    import hashlib
    expected = {k: hashlib.sha256(v).hexdigest() for k, v in _expected_files(name).items()
                if k.endswith("submission.py")}
    assert probe["subs"] == expected
