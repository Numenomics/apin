"""Unit tests for the apin Harbor Agent Benchmark task suite (`harbor/`)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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


def test_harbor_verify_records_solution_complexity_and_collector_builds_site_data(tmp_path):
    """harbor/verify.py records LOC/cyclomatic/AST stats & archives submission.py; collector builds the site's benchmark data."""
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
    data = mod.collect_jobs([tmp_path / "job1"], pricing, release_tag="bench-v0.2.0")
    rel = data["releases"][0]
    assert data["current_release"] == rel["release"] == "bench-v0.2.0"
    assert len(rel["trials"]) == 1
    tr = rel["trials"][0]
    assert tr["input_tokens"] == 10000 and tr["cached_tokens"] == 40000 and tr["reasoning_tokens"] == 1200
    assert tr["cache_hit_rate"] == 0.8 and tr["test_runs"] == 1 and tr["tool_calls"] == 1 and tr["turns"] == 1
    assert tr["cost_usd"] > 0 and tr["agent_wall_s"] == 75.0
    assert "coin mixture" in tr["approach_summary"]
    assert tr["termination"] == "finished" and tr["outcome"] == "solved"
    # A bug-fix attempt is kept as a record, but the benchmark is the speed-up tasks: it adds
    # no leaderboard row and no task to the release.
    assert tr["category"] == "rewind" and rel["leaderboard"] == [] and rel["tasks"] == []
    # Every task on the page is described from its own task.toml, never restated by hand.
    assert mod.load_task_meta(["a1-preconditioning"])["a1-preconditioning"]["lane"] == "A1"


def _load_collector():
    import importlib.util

    spec = importlib.util.spec_from_file_location("collect_harbor_results", ROOT / "scripts" / "collect_harbor_results.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _attempt(trial_id, cfg, task, *, release="bench-v0.2.0", passed=True, outcome=None, ref=None, start=None,
             termination="finished", cost=1.0):
    model, effort = cfg.split("#")
    return {
        "trial_id": trial_id, "release": release, "model": model, "effort": effort, "config_key": cfg,
        "agent": "opencode-v2", "task_id": task, "category": "rewind" if task.startswith("s3-") else "optimize",
        "passed": passed, "termination": termination, "rate_limited": termination == "provider_limit",
        "reward": ref or 0.0, "verifier_outcome": outcome, "ref_speedup": ref, "start_speedup": start,
        "cost_usd": cost, "agent_wall_s": 60.0, "total_tokens": 10, "input_tokens": 5, "cached_tokens": 5,
    }


def test_collector_ranks_by_speedup_and_never_scores_interrupted_runs():
    mod = _load_collector()
    trials = [
        # Setting A: one task attempted three times, one of them invalid. A bug-fix attempt too.
        _attempt("a1", "m/a#low", "s3-unadjusted-ula", outcome="solved"),
        _attempt("a2", "m/a#low", "a0-ground-up", outcome="solved", ref=1.5, start=6.0),
        _attempt("a3", "m/a#low", "a0-ground-up", outcome="improved", ref=0.5, start=2.0),
        _attempt("a4", "m/a#low", "a0-ground-up", passed=False, outcome="invalid", ref=0.0, start=0.0),
        _attempt("a5", "m/a#low", "a1-preconditioning", outcome="no_gain", ref=0.7, start=1.0),
        # Setting B: the largest speed-up of all, but its second task was cut off by the provider.
        _attempt("b1", "m/b#low", "a1-preconditioning", termination="provider_limit", outcome="solved", ref=3.0, start=5.0),
        _attempt("b2", "m/b#low", "a0-ground-up", outcome="solved", ref=9.0, start=30.0),
        # Setting C: one modest gain, one broken sampler.
        _attempt("c1", "m/c#low", "a1-preconditioning", passed=False, outcome="invalid"),
        _attempt("c2", "m/c#low", "a0-ground-up", outcome="improved", ref=0.4, start=2.0),
        # A valid result the verifier could not score (its own reference failed) is nobody's zero.
        _attempt("c3", "m/c#low", "a0-ground-up", outcome="unscorable"),
    ]
    rel = mod.aggregate_results(trials, {"models": {}}, release_tag="bench-v0.2.0")["releases"][0]
    assert rel["scoring"] == "reference"
    # The benchmark is the speed-up tasks. The bug-fix attempt stays in the record only.
    assert rel["tasks"] == ["a0-ground-up", "a1-preconditioning"] and any(t["trial_id"] == "a1" for t in rel["trials"])
    board = {r["config_key"]: r for r in rel["leaderboard"]}

    a = board["m/a#low"]
    task = a["tasks"]["a0-ground-up"]
    assert task["attempts"] == 3 and task["valid"] == 2 and task["outcomes"] == {"solved": 1, "improved": 1, "invalid": 1}
    # The invalid attempt counts as zero in the mean; it is not dropped.
    assert task["speedup"] == round((6.0 + 2.0 + 0.0) / 3, 4) and task["speedup_min"] == 0.0 and task["speedup_max"] == 6.0
    assert task["ref_speedup"] == 1.0 and task["best_trial"] == "a2"
    assert a["complete"] and a["attempts_per_task"] == 3 and "s3-unadjusted-ula" not in a["tasks"]
    assert a["speedup"] == pytest.approx(math.sqrt((8.0 / 3) * 1.0), abs=1e-3)

    b = board["m/b#low"]
    # The interrupted run is neither a result nor a zero, whatever the verifier wrote for it.
    assert b["interrupted"] == 1 and b["tasks"]["a1-preconditioning"]["attempts"] == 0 and not b["complete"]
    assert [t["outcome"] for t in rel["trials"] if t["trial_id"] == "b1"] == ["interrupted"]
    assert b["speedup"] == 30.0

    c = board["m/c#low"]
    # A task with no valid attempt enters the geometric mean at the floor.
    assert c["speedup"] == pytest.approx(math.sqrt(2.0 * mod.INVALID_FLOOR), abs=1e-3) and c["valid"] == 1.0
    assert c["tasks"]["a0-ground-up"]["attempts"] == 1 and c["interrupted"] == 1

    # Complete settings first, then the larger speed-up. B's bigger ratio does not lift it over A.
    assert [r["config_key"] for r in rel["leaderboard"]] == ["m/a#low", "m/c#low", "m/b#low"]


def test_collector_keeps_releases_apart_and_maps_earlier_trials():
    mod = _load_collector()
    legacy = {
        "trial_id": "old1", "release": "bench-v0.1.1", "model": "m/a", "effort": "low", "config_key": "m/a#low",
        "task_id": "a0-ground-up", "category": "optimize", "passed": True, "infra_error": False, "rate_limited": False,
        "reward": 1.0, "speedup": 1.0, "baseline_soft_min": 0.001, "error": "AgentTimeoutError: timed out", "tool_calls": 40,
    }
    limited = dict(legacy, trial_id="old2", task_id="a1-preconditioning", speedup=6.4, rate_limited=True,
                   error="ApiRateLimitError: Command failed")
    new = _attempt("new1", "m/a#low", "a0-ground-up", outcome="solved", ref=1.2, start=4.0)
    data = mod.aggregate_results([legacy, limited, new], {"models": {}}, release_tag="bench-v0.2.0")
    assert [r["release"] for r in data["releases"]] == ["bench-v0.2.0", "bench-v0.1.1"]
    old = data["releases"][1]
    assert old["scoring"] == "start" and "Not comparable" in old["note"]
    by_id = {t["trial_id"]: t for t in old["trials"]}
    # Unchanged starting code used to count as a pass at 1.00x. It is "no gain".
    assert by_id["old1"]["outcome"] == "no_gain" and by_id["old1"]["termination"] == "time_limit"
    assert by_id["old1"]["start_speedup"] == 1.0 and "speedup" not in by_id["old1"]
    # A run cut off by the provider is excluded even though it scored well.
    assert by_id["old2"]["outcome"] == "interrupted" and by_id["old2"]["start_speedup"] is None
    assert old["leaderboard"][0]["speedup"] == 1.0 and old["leaderboard"][0]["time_limit_hits"] == 1


def test_collector_writes_json_a_browser_can_parse(tmp_path):
    # A sampler that diverges reports b2_max = inf. Python would write `Infinity`, which
    # JSON.parse rejects, and the whole benchmark page would fail to load.
    mod = _load_collector()
    trial = _attempt("x1", "m/a#low", "a0-ground-up", passed=False, outcome="invalid")
    trial["b2_max"] = float("inf")
    trial["by_class"] = {"funnel": float("nan")}
    out = tmp_path / "benchmark.json"
    mod.write_site_data(mod.aggregate_results([trial], {"models": {}}, release_tag="bench-v0.2.0"), out)
    text = out.read_text()
    assert "Infinity" not in text and "NaN" not in text
    saved = json.loads(text, parse_constant=lambda name: pytest.fail(f"non-standard JSON constant {name}"))["releases"][0]["trials"][0]
    assert saved["b2_max"] is None and saved["by_class"]["funnel"] is None


def test_collector_termination_rule_does_not_score_runs_that_never_had_a_fair_chance():
    mod = _load_collector()
    kind = mod.classify_termination
    assert kind(None, rate_limited=False, tool_calls=9) == "finished"
    assert kind("AgentTimeoutError", rate_limited=True, tool_calls=9) == "time_limit"
    # Cancelled runs, setup timeouts and provider outages are not the agent's result.
    assert kind("CancelledError", rate_limited=False, tool_calls=9) == "infrastructure"
    assert kind("AgentSetupTimeoutError", rate_limited=False, tool_calls=0) == "infrastructure"
    assert kind("ApiProviderOverloadedError", rate_limited=False, tool_calls=9) == "provider_limit"
    assert kind("ApiRateLimitError", rate_limited=False, tool_calls=9) == "provider_limit"
    # An exception with no verifier result cannot be an invalid attempt: nothing was graded.
    assert kind("SomeNewHarborError", rate_limited=False, tool_calls=9, graded=False) == "infrastructure"
    assert kind("AgentTimeoutError", rate_limited=False, tool_calls=9, graded=False) == "infrastructure"
    assert kind("AgentTimeoutError", rate_limited=True, tool_calls=9, graded=False) == "provider_limit"
    assert kind("NonZeroAgentExitCodeError", rate_limited=False, tool_calls=9) == "agent_error"
    assert {"provider_limit", "infrastructure", "agent_crash"} == set(mod.INTERRUPTED)
