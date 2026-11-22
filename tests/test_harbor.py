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

    # Reference fix passes -> reward 1.0 (also verify hyphenated Harbor slug alias works)
    out_ref = tmp_path / "ref"
    r_ref = subprocess.run(
        [
            sys.executable, str(verify_py), "rewind",
            "--task-id", "s3-missing-jacobian-a4",
            "--submission", str(ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "missing_jacobian_a4_ref.py"),
            "--out-dir", str(out_ref),
        ],
        capture_output=True, text=True,
    )
    assert r_ref.returncode == 0, r_ref.stdout + r_ref.stderr
    assert float((out_ref / "reward.txt").read_text().strip()) == 1.0
    assert json.loads((out_ref / "reward.json").read_text())["passed"] == 1

    # Verify default --out-dir outside Docker writes to cwd/verifier_out instead of crashing on /logs/verifier
    r_def = subprocess.run(
        [
            sys.executable, str(verify_py), "rewind",
            "--task-id", "s3-position-gated-mh",
            "--submission", str(ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "position_gated_ref.py"),
        ],
        cwd=tmp_path,
        capture_output=True, text=True,
    )
    assert r_def.returncode == 0, r_def.stdout + r_def.stderr
    assert float((tmp_path / "verifier_out" / "reward.txt").read_text().strip()) == 1.0


def test_harbor_verify_records_solution_complexity_and_collector_builds_page(tmp_path):
    """harbor/verify.py records LOC/cyclomatic/AST stats & archives submission.py; collector renders HTML."""
    import importlib.util

    verify_py = ROOT / "harbor" / "verify.py"
    out_dir = tmp_path / "job1" / "trial_a" / "verifier"
    r = subprocess.run(
        [
            sys.executable, str(verify_py), "rewind",
            "--task-id", "s3-position-gated-mh",
            "--submission", str(ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "position_gated_ref.py"),
            "--out-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    # reward.json must contain strictly numeric values for Harbor's verifier parser
    reward_num = json.loads((out_dir / "reward.json").read_text())
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in reward_num.values()), reward_num
    assert reward_num["reward"] == 1.0 and reward_num["passed"] == 1 and reward_num["solution_loc"] > 10
    details = json.loads((out_dir / "details.json").read_text())
    sol = details["solution"]
    assert sol["loc"] > 10 and sol["cyclomatic_complexity"] >= 1 and sol["ast_nodes"] > 20
    assert (out_dir / "submission.py").is_file()

    # Create synthetic Harbor result.json + opencode-v2.txt next to verifier/ and test collector
    trial_dir = tmp_path / "job1" / "trial_a"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({
        "trial_name": "s3-position-gated-mh__abc",
        "task_name": "s3-position-gated-mh",
        "config": {
            "agent": {
                "name": "opencode-v2",
                "model_name": "zai-coding-plan/glm-5.3-flash",
                "kwargs": {"reasoning_effort": "high"},
            },
            "task": {"path": "harbor/tasks/s3-position-gated-mh"},
        },
        "started_at": "2026-11-22T04:00:00Z",
        "finished_at": "2026-11-22T04:01:30Z",
        "agent_execution": {"started_at": "2026-11-22T04:00:05Z", "finished_at": "2026-11-22T04:01:20Z"},
        "verifier_result": {"rewards": {"reward": 1.0}},
    }))
    (agent_dir / "opencode-v2.txt").write_text("\n".join([
        json.dumps({"type": "tool_use", "part": {"state": {"input": {"command": "uv run apin invariance --sampler /app/submission.py --dim 4"}}}}),
        json.dumps({"type": "text", "part": {"text": "Replaced radial gate with state-independent 50/50 coin mixture."}}),
        json.dumps({"type": "step_finish", "part": {"tokens": {"input": 10000, "output": 500, "reasoning": 1200, "cache": {"read": 40000, "write": 2000}}, "cost": 0.0}}),
    ]) + "\n")

    spec = importlib.util.spec_from_file_location("collect_harbor_results", ROOT / "scripts" / "collect_harbor_results.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    pricing = json.loads((ROOT / "benchmarks" / "pricing.json").read_text())
    data = mod.collect_jobs([tmp_path / "job1"], pricing, release_tag="bench-v0.1.0")
    assert len(data["trials"]) == 1 and len(data["leaderboard"]) == 1
    tr = data["trials"][0]
    assert tr["input_tokens"] == 10000 and tr["cached_tokens"] == 40000 and tr["reasoning_tokens"] == 1200
    assert tr["cache_hit_rate"] == 0.8 and tr["test_runs"] == 1 and tr["tool_calls"] == 1 and tr["turns"] == 1
    assert tr["cost_usd"] > 0 and tr["agent_wall_s"] == 75.0
    assert "coin mixture" in tr["approach_summary"]
    html = mod.render_html(data)
    assert "Price Pareto Frontier" in html and "GLM-5.3-Flash" in html
