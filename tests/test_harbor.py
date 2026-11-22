"""Unit tests for the apin Harbor Agent Benchmark task suite (`harbor/`)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from harness.models import ROOT

HARBOR_TASKS_DIR = ROOT / "harbor" / "tasks"
EXPECTED_TASKS = {
    "s3-position-gated-mh",
    "s3-missing-jacobian-a4",
    "s3-unadjusted-ula",
    "a0-ground-up",
    "a1-preconditioning",
    "a2-funnel-dynamics",
    "a3-integrator-trajectory",
    "a4-nonlinear-transport",
}


def test_harbor_task_directories_and_schemas():
    assert HARBOR_TASKS_DIR.is_dir()
    found = {p.name for p in HARBOR_TASKS_DIR.iterdir() if p.is_dir()}
    assert found == EXPECTED_TASKS

    for name in sorted(EXPECTED_TASKS):
        tdir = HARBOR_TASKS_DIR / name
        toml_path = tdir / "task.toml"
        instr_path = tdir / "instruction.md"
        docker_path = tdir / "environment" / "Dockerfile"
        test_sh = tdir / "tests" / "test.sh"
        base_py = tdir / "environment" / "baseline.py"

        assert toml_path.is_file(), f"{name}: missing task.toml"
        assert instr_path.is_file() and len(instr_path.read_text().strip()) > 80
        assert docker_path.is_file()
        assert test_sh.is_file() and os.access(test_sh, os.X_OK), f"{name}: tests/test.sh must be executable"
        assert base_py.is_file(), f"{name}: missing frozen baseline.py"

        cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        for sec in ("metadata", "agent", "verifier", "environment"):
            assert sec in cfg, f"{name}: task.toml missing [{sec}]"
        assert cfg["agent"]["timeout_sec"] > 0
        assert cfg["verifier"]["timeout_sec"] > 0
        assert cfg["metadata"]["category"] in ("rewind", "optimize")


def test_harbor_verify_rewind_writes_rewards(tmp_path):
    verify_py = ROOT / "harbor" / "verify.py"

    # Base fixture fails -> reward 0.0
    out_base = tmp_path / "base"
    r_base = subprocess.run(
        [
            sys.executable, str(verify_py), "rewind",
            "--task-id", "s3_missing_jacobian_transport",
            "--submission", str(HARBOR_TASKS_DIR / "s3-missing-jacobian-a4" / "environment" / "baseline.py"),
            "--out-dir", str(out_base),
        ],
        capture_output=True, text=True,
    )
    assert r_base.returncode == 1
    assert float((out_base / "reward.txt").read_text().strip()) == 0.0
    assert json.loads((out_base / "reward.json").read_text())["passed"] == 0

    # Reference fix passes -> reward 1.0
    out_ref = tmp_path / "ref"
    r_ref = subprocess.run(
        [
            sys.executable, str(verify_py), "rewind",
            "--task-id", "s3_missing_jacobian_transport",
            "--submission", str(ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "missing_jacobian_a4_ref.py"),
            "--out-dir", str(out_ref),
        ],
        capture_output=True, text=True,
    )
    assert r_ref.returncode == 0, r_ref.stdout + r_ref.stderr
    assert float((out_ref / "reward.txt").read_text().strip()) == 1.0
    assert json.loads((out_ref / "reward.json").read_text())["passed"] == 1
