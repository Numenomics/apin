"""Unit tests for the apin Harbor Agent Benchmark task suite (`harbor/`)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tomllib
from pathlib import Path

import pytest

from harness import grade
from harness.models import ROOT

HARBOR_TASKS_DIR = ROOT / "harbor" / "tasks"
EXPECTED_TASKS = {
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
        assert cfg["metadata"]["category"] == "optimize"


@pytest.mark.parametrize("name", sorted(EXPECTED_TASKS))
def test_instruction_names_the_measuring_command_and_the_reward(name):
    lane = tomllib.loads((HARBOR_TASKS_DIR / name / "task.toml").read_text())["metadata"]["lane"]
    text = (HARBOR_TASKS_DIR / name / "instruction.md").read_text()
    cmd = f'uv run apin autoresearch eval --lane {lane} --run-id harbor_dev --no-queue --hypothesis "<one sentence>"'
    assert cmd in text
    assert "--dims" not in text and "--repeats" not in text
    assert "hidden seeds" in text and "under 5% is noise" in text and "scores zero" in text
    assert "Stan-style reference" in text and "baseline_soft_min" not in text
    for bad in ("\u2014", ";", "**"):
        assert bad not in text.split("```")[0] + text.split("```")[-1], (name, bad)


def _verify():
    spec = importlib.util.spec_from_file_location("apin_harbor_verify_harbor", ROOT / "harbor" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_grader(monkeypatch, soft_min: float = 6e-4):
    def score_file(path, lane, cfg, work_dir, **kw):
        sm = soft_min if Path(path).name == "submission.py" else 2e-4
        return {"valid": True, "tier0_ok": True, "tier1_ok": True, "soft_min_eff": sm,
                "short_horizon_soft_min": sm, "long_horizon_soft_min": sm, "b2_max": 0.01,
                "overhead_ratio": 1.0, "by_class": {"banana": sm}, "per_repeat_soft_min": [sm],
                "confirmation": {"ran": False}, "failed_targets": [], "wall_s": 0.1}
    monkeypatch.setattr(grade, "score_file", score_file)


def test_verifier_default_out_dir_outside_docker(tmp_path, monkeypatch):
    verify = _verify()
    _stub_grader(monkeypatch)
    monkeypatch.chdir(tmp_path)
    sub = ROOT / "autoresearch" / "lanes" / "A2" / "start.py"
    assert verify.main(["optimize", "--lane", "A2", "--submission", str(sub)]) == 0
    assert float((tmp_path / "verifier_out" / "reward.txt").read_text().strip()) > 0


def test_harbor_verify_records_solution_complexity_and_collector_builds_site_data(tmp_path, monkeypatch):
    """harbor/verify.py records LOC/cyclomatic/AST stats & archives submission.py; collector builds the site's benchmark data."""
    verify = _verify()
    _stub_grader(monkeypatch)
    out_dir = tmp_path / "job1" / "trial_a" / "verifier"
    sub = ROOT / "autoresearch" / "lanes" / "A0" / "submission.py"
    assert verify.verify_optimize("A0", sub, out_dir) == 0
    reward_num = json.loads((out_dir / "reward.json").read_text())
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in reward_num.values()), reward_num
    assert reward_num["passed"] == 1 and reward_num["solution_loc"] > 10
    details = json.loads((out_dir / "details.json").read_text())
    sol = details["solution"]
    assert sol["loc"] > 10 and sol["cyclomatic_complexity"] >= 1 and sol["ast_nodes"] > 20
    assert (out_dir / "submission.py").is_file()

    trial_dir = tmp_path / "job1" / "trial_a"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({
        "trial_name": "a0-ground-up__abc",
        "task_name": "a0-ground-up",
        "config": {
            "agent": {
                "name": "opencode-v2",
                "model_name": "zai-coding-plan/glm-5.3-flash",
                "kwargs": {"reasoning_effort": "high"},
            },
            "task": {"path": "harbor/tasks/a0-ground-up"},
        },
        "started_at": "2026-11-22T04:00:00Z",
        "finished_at": "2026-11-22T04:01:30Z",
        "agent_execution": {"started_at": "2026-11-22T04:00:05Z", "finished_at": "2026-11-22T04:01:20Z"},
        "verifier_result": {"rewards": {"reward": reward_num["reward"]}},
    }))
    (agent_dir / "opencode-v2.txt").write_text("\n".join([
        json.dumps({"type": "tool_use", "part": {"state": {"input": {"command": "uv run apin autoresearch eval --lane A0 --run-id harbor_dev --no-queue --hypothesis h"}}}}),
        json.dumps({"type": "text", "part": {"text": "Replaced the random walk with windowed dense HMC."}}),
        json.dumps({"type": "step_finish", "part": {"tokens": {"input": 10000, "output": 500, "reasoning": 1200, "cache": {"read": 40000, "write": 2000}}, "cost": 0.0}}),
    ]) + "\n")

    spec = importlib.util.spec_from_file_location("collect_harbor_results", ROOT / "scripts" / "collect_harbor_results.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    assert mod.DEFAULT_RELEASE == "bench-v0.3.0"
    pricing = json.loads((ROOT / "benchmarks" / "pricing.json").read_text())
    # A job whose name carries no release tag is skipped unless a release is named for it.
    assert mod.collect_jobs([tmp_path / "job1"], pricing)["releases"] == []
    data = mod.collect_jobs([tmp_path / "job1"], pricing, release_tag=mod.DEFAULT_RELEASE)
    rel = data["releases"][0]
    assert data["current_release"] == rel["release"] == "bench-v0.3.0"
    assert len(rel["trials"]) == 1
    tr = rel["trials"][0]
    assert tr["input_tokens"] == 10000 and tr["cached_tokens"] == 40000 and tr["reasoning_tokens"] == 1200
    assert tr["cache_hit_rate"] == 0.8 and tr["test_runs"] == 1 and tr["tool_calls"] == 1 and tr["turns"] == 1
    assert tr["cost_usd"] > 0 and tr["agent_wall_s"] == 75.0
    assert "dense HMC" in tr["approach_summary"]
    assert tr["termination"] == "finished" and tr["category"] == "optimize"
    assert tr["outcome"] == details["outcome"] and tr["start_speedup"] == details["start_speedup"]
    assert rel["tasks"] == mod.speedup_tasks() == sorted(EXPECTED_TASKS)
    assert [r["config_key"] for r in rel["leaderboard"]] == [tr["config_key"]]
    # Every task on the page is described from its own task.toml, never restated by hand.
    assert mod.load_task_meta(["a1-preconditioning"])["a1-preconditioning"]["lane"] == "A1"


def test_collector_reaggregates_the_published_benchmark_file(tmp_path):
    """Older releases in the site's data file (with their bug-fix tasks) still aggregate."""
    mod = _load_collector()
    published = ROOT / "flywheel" / "public" / "data" / "benchmark.json"
    if not published.is_file():
        pytest.skip("no published benchmark data")
    out = tmp_path / "x.json"
    assert mod.main(["--out-json", str(out), "--input-json", str(published)]) == 0
    before = {r["release"]: r for r in json.loads(published.read_text())["releases"]}
    after = json.loads(out.read_text())["releases"]
    assert after
    # An earlier release keeps the task details it was published with: today's task files
    # describe only the current release.
    earlier = [r for r in after if r["release"] != mod.DEFAULT_RELEASE]
    assert earlier and all(r["task_meta"] == before[r["release"]]["task_meta"] for r in earlier)


def test_a_listed_rule_break_counts_as_a_failed_attempt():
    """An attempt the verifier passed and a reader of its submission did not."""
    mod = _load_collector()
    trial = {"trial_id": "a3__x", "task_id": "a3-integrator-trajectory", "category": "optimize", "release": mod.DEFAULT_RELEASE,
             "config_key": "m#default", "model": "m", "effort": "default", "termination": "finished",
             "verifier_outcome": "solved", "passed": True, "start_speedup": 1737.0, "ref_speedup": 1222.0}
    ruling = {"rules": "R-003", "reason": "identifies the target"}
    pricing = {"models": {}}
    clean = mod.aggregate_results([trial], pricing, release_tag=mod.DEFAULT_RELEASE, rule_breaks={})
    assert clean["releases"][0]["trials"][0]["outcome"] == "solved"
    data = mod.aggregate_results([trial], pricing, release_tag=mod.DEFAULT_RELEASE,
                                 rule_breaks={f"{mod.DEFAULT_RELEASE}/a3__x": ruling})
    rel = data["releases"][0]
    got = rel["trials"][0]
    assert got["outcome"] == "invalid" and got["rule_break"] == ruling and got["start_speedup"] == 1737.0
    cell = rel["leaderboard"][0]["tasks"]["a3-integrator-trajectory"]
    assert cell["valid"] == 0 and cell["speedup"] == mod.INVALID_FLOOR
    # Taking the entry off the list counts the attempt again, also when it is read back from the data file.
    again = mod.aggregate_results(rel["trials"], pricing, release_tag=mod.DEFAULT_RELEASE, rule_breaks={})
    assert again["releases"][0]["trials"][0]["outcome"] == "solved" and "rule_break" not in again["releases"][0]["trials"][0]


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


def test_collector_ranks_by_speedup_and_never_scores_interrupted_runs(monkeypatch):
    mod = _load_collector()
    # A benchmark of two tasks, so that a row can be complete with few attempts.
    monkeypatch.setattr(mod, "speedup_tasks", lambda: ["a0-ground-up", "a1-preconditioning"])
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
    # The invalid attempt enters the mean at the floor; it is not dropped.
    assert task["speedup"] == round((6.0 + 2.0 + mod.INVALID_FLOOR) / 3, 4) and task["speedup_min"] == mod.INVALID_FLOOR and task["speedup_max"] == 6.0
    assert task["ref_speedup"] == 1.0 and task["best_trial"] == "a2"
    assert a["complete"] and a["attempts_per_task"] == 3 and "s3-unadjusted-ula" not in a["tasks"]
    assert a["speedup"] == pytest.approx(math.sqrt((8.1 / 3) * 1.0), abs=1e-3)

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

    # Against the real task list, a row that ran two of the five tasks is not complete,
    # even though no row in the release ran the other three.
    monkeypatch.undo()
    full = mod.aggregate_results(trials, {"models": {}}, release_tag="bench-v0.2.0")["releases"][0]
    assert len(full["tasks"]) == 5 and not any(r["complete"] for r in full["leaderboard"])
    assert {r["config_key"]: r["tasks_run"] for r in full["leaderboard"]} == {"m/a#low": 2, "m/b#low": 1, "m/c#low": 2}


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
