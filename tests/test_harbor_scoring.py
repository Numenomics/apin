"""Scoring of the Harbor verifier (harbor/verify.py): outcome rule, frozen denominators, output
shape, isolation of test.sh, and that every copy under harbor/tasks/*/tests/ matches its source."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness import judge
from harness.models import ROOT

TASKS = ROOT / "harbor" / "tasks"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def verify():
    return _load(ROOT / "harbor" / "verify.py", "apin_harbor_verify_scoring")


def _den(sm, valid=True):
    return {"valid": valid, "soft_min_eff": sm}


# ------------------------------------------------------------------------------- outcome rule

def test_outcome_rule_on_synthetic_numbers(verify):
    m = verify.IMPROVEMENT_MARGIN
    start, ref, front = _den(1.0), _den(2.0), _den(4.0)

    r = verify.classify(False, 0.0, start, ref, front)
    assert r["outcome"] == "invalid" and r["reward"] == 0.0
    assert r["ref_speedup"] == r["start_speedup"] == r["frontier_speedup"] == 0.0 and r["gap_closed"] is None

    r = verify.classify(True, 1.0, start, ref, front)  # unchanged starting code
    assert r["outcome"] == "no_gain" and r["start_speedup"] == 1.0 and r["reward"] == pytest.approx(0.5)
    assert r["gap_closed"] == 0.0

    just_below = m * 0.999
    assert verify.classify(True, just_below, start, ref, front)["outcome"] == "no_gain"
    r = verify.classify(True, m, start, ref, front)
    assert r["outcome"] == "improved" and r["ref_speedup"] == pytest.approx(m / 2.0)

    r = verify.classify(True, 2.0, start, ref, front)
    assert r["outcome"] == "solved" and r["reward"] == r["ref_speedup"] == 1.0
    assert r["frontier_speedup"] == 0.5 and r["gap_closed"] == pytest.approx(0.5)  # log 2 / log 4

    r = verify.classify(True, 8.0, start, ref, front)
    assert r["gap_closed"] == pytest.approx(1.5)  # beyond the frontier


def test_solved_needs_the_margin_even_when_the_reference_is_weak(verify):
    # A reference weaker than the starting code: unchanged code beats it but gained nothing.
    r = verify.classify(True, 1.0, _den(1.0), _den(0.5), None)
    assert r["outcome"] == "no_gain" and r["ref_speedup"] == 2.0 and r["reward"] == 2.0


def test_gap_closed_is_null_unless_the_frontier_is_valid_and_ahead(verify):
    assert verify.classify(True, 2.0, _den(1.0), _den(1.0), _den(0.8))["gap_closed"] is None
    assert verify.classify(True, 2.0, _den(1.0), _den(1.0), _den(1.0))["gap_closed"] is None
    r = verify.classify(True, 2.0, _den(1.0), _den(1.0), _den(3.0, valid=False))
    assert r["gap_closed"] is None and r["frontier_speedup"] is None
    assert verify.classify(True, 2.0, _den(1.0), _den(1.0), None)["gap_closed"] is None


def test_failed_denominators_make_the_run_unscorable_never_1x(verify):
    for start, ref in ((_den(1.0, False), _den(1.0)), (_den(1.0), _den(1.0, False)), (_den(0.0), _den(1.0))):
        r = verify.classify(True, 3.0, start, ref, None)
        assert r["outcome"] == "unscorable" and r["reward"] == 0.0 and r["unscorable_reason"]
    # The candidate's own failure still reads as invalid.
    assert verify.classify(False, 0.0, _den(1.0, False), _den(1.0), None)["outcome"] == "invalid"


def test_scoring_constants_are_named_once(verify):
    import inspect

    assert inspect.signature(verify.classify).parameters["margin"].default == verify.IMPROVEMENT_MARGIN
    assert verify.IMPROVEMENT_MARGIN > 1.0 and verify.VERIFIER_REPEATS >= 2
    assert verify.scoring_config()["repeats"] == verify.VERIFIER_REPEATS
    assert verify.HIDDEN_VERIFIER_SEED != 88421077  # retired with bench-v0.2.0


# ----------------------------------------------------------------- copies and frozen scores

def test_every_copy_matches_its_canonical_source():
    sync = _load(ROOT / "scripts" / "sync_harbor_tests.py", "apin_sync_harbor_tests")
    assert sync.drifted() == [], "run: uv run python scripts/sync_harbor_tests.py"
    assert set(sync.OPTIMIZE_TASKS) == set(_load(ROOT / "harbor" / "verify.py", "apin_v_lanes").LANE_TASKS)


@pytest.mark.parametrize("lane", ["A0", "A1", "A2", "A3", "A4"])
def test_frozen_scores_are_current(verify, lane):
    """Every frozen denominator is keyed to today's judge, scoring config and file bytes, so the
    verifier reads it instead of scoring live. Fails after a harness/ or verifier change: rerun
    scripts/freeze_harbor_references.py."""
    tests = TASKS / verify.LANE_TASKS[lane] / "tests"
    frozen = verify.load_frozen(tests)
    assert frozen, "missing or wrong-schema frozen_scores.json"
    assert frozen["lane"] == lane
    digest = judge.judge_digest()
    cfg = verify.scoring_config()
    for role, name in verify.DENOMINATOR_FILES.items():
        sha = verify._sha256(tests / name)
        assert verify._frozen_match(frozen["scores"][role], lane, role, sha, cfg, digest), (lane, role)
    assert frozen["scores"]["start"]["valid"], "the starting code must pass on the verifier seeds"


def test_test_sh_grades_in_isolated_mode_from_the_tests_dir(verify):
    for task in sorted(p for p in TASKS.iterdir() if (p / "tests" / "test.sh").is_file()):
        text = (task / "tests" / "test.sh").read_text()
        assert "/app/.venv/bin/python -I /tests/verify.py" in text, task.name
        assert "--baseline" not in text
        if task.name in verify.LANE_TASKS.values():
            assert "--tests-dir /tests" in text
            lane = next(k for k, v in verify.LANE_TASKS.items() if v == task.name)
            assert f"--submission /app/autoresearch/lanes/{lane}/submission.py" in text


def test_denominators_never_ship_in_the_agent_image():
    for task in TASKS.iterdir():
        for name in ("reference.py", "frontier.py", "frozen_scores.json"):
            assert not (task / "environment" / name).exists(), (task.name, name)


# ------------------------------------------------------------------------------ output shape

def _stub_score(soft_min_by_file: dict[str, float | None]):
    def score_file(path, lane, cfg, work_dir):
        sm = soft_min_by_file[Path(path).name]
        if sm is None:
            return {"valid": False, "tier0_ok": False, "tier1_ok": False, "error": "tier0 invariance_smoke: stub",
                    "tier0_info": {"stage": "invariance_smoke"}, "wall_s": 0.0}
        return {"valid": True, "tier0_ok": True, "tier1_ok": True, "soft_min_eff": sm,
                "short_horizon_soft_min": sm / 2, "long_horizon_soft_min": sm * 2, "b2_max": 0.01,
                "overhead_ratio": 1.5, "by_class": {"banana": sm, "funnel": sm * 3},
                "per_repeat_soft_min": [sm, sm], "per_repeat_by_class": [], "failed_targets": [], "wall_s": 0.1}
    return score_file


def _tests_dir(tmp_path: Path, lane_task: str) -> Path:
    d = tmp_path / "tests"
    d.mkdir()
    for name in ("baseline.py", "reference.py", "frontier.py"):
        shutil.copyfile(TASKS / lane_task / "tests" / name, d / name)
    return d


def test_optimize_output_shape(verify, tmp_path, monkeypatch):
    tests = _tests_dir(tmp_path, "a2-funnel-dynamics")
    sub = tmp_path / "cand.py"
    sub.write_text((tests / "baseline.py").read_text() + "\n# edited\n")
    monkeypatch.setattr(verify, "score_file", _stub_score(
        {"submission.py": 3.0e-4, "baseline.py": 1.0e-4, "reference.py": 2.0e-4, "frontier.py": 4.0e-4}))
    rc = verify.verify_optimize("A2", sub, tmp_path / "out", tests_dir=tests)
    assert rc == 0
    reward = json.loads((tmp_path / "out" / "reward.json").read_text())
    details = json.loads((tmp_path / "out" / "details.json").read_text())
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in reward.values()), reward
    assert reward["reward"] == reward["ref_speedup"] == 1.5 and reward["start_speedup"] == 3.0
    assert reward["solved"] == 1 and reward["improved"] == 0 and reward["unscorable"] == 0 and reward["passed"] == 1
    assert reward["frontier_speedup"] == 0.75 and reward["gap_closed"] == pytest.approx(math.log(3) / math.log(4), abs=1e-4)
    assert reward["soft_min_eff"] == 3.0e-4 and reward["start_soft_min"] == 1.0e-4
    assert reward["reference_soft_min"] == 2.0e-4 and reward["frontier_soft_min"] == 4.0e-4
    assert details["outcome"] == "solved" and details["lane"] == "A2" and details["task"] == "a2-funnel-dynamics"
    assert details["reference"] == {"name": "stan-nuts-diag-2.37", "file": "reference.py", "source": "live", "valid": True}
    assert details["judge_digest"] == judge.judge_digest()
    assert set(details["denominators"]) == {"start", "reference", "frontier"}
    assert details["denominators"]["start"]["source"] == "live" and details["by_class"]["funnel"] == 9.0e-4
    assert details["config"] == verify.scoring_config() and details["improvement_margin"] == verify.IMPROVEMENT_MARGIN
    assert (tmp_path / "out" / "submission.py").read_text() == sub.read_text()


def test_optimize_unscorable_when_the_reference_fails(verify, tmp_path, monkeypatch):
    tests = _tests_dir(tmp_path, "a0-ground-up")
    monkeypatch.setattr(verify, "score_file", _stub_score(
        {"submission.py": 3.0e-4, "baseline.py": 1.0e-4, "reference.py": None, "frontier.py": 4.0e-4}))
    rc = verify.verify_optimize("A0", tests / "frontier.py", tmp_path / "out", tests_dir=tests)
    reward = json.loads((tmp_path / "out" / "reward.json").read_text())
    details = json.loads((tmp_path / "out" / "details.json").read_text())
    assert rc == 2 and reward["reward"] == 0.0 and reward["unscorable"] == 1 and reward["passed"] == 1
    assert "ref_speedup" not in reward and details["ref_speedup"] is None and details["start_speedup"] == 3.0
    assert details["outcome"] == "unscorable" and "reference" in details["unscorable_reason"]
    assert details["denominators"]["reference"]["valid"] is False


def test_frozen_denominators_are_used_only_when_the_key_matches(verify, tmp_path, monkeypatch):
    tests = _tests_dir(tmp_path, "a3-integrator-trajectory")
    calls = []
    stub = _stub_score({"submission.py": 2.0e-4, "baseline.py": 1.0e-4, "reference.py": 1.0e-4, "frontier.py": 1.0e-4})

    def counting(path, lane, cfg, work_dir):
        calls.append(Path(path).name)
        return stub(path, lane, cfg, work_dir)

    monkeypatch.setattr(verify, "score_file", counting)
    cfg, digest = verify.scoring_config(), judge.judge_digest()
    scores = {role: verify.frozen_entry("A3", role, tests / name, cfg, digest, stub(tests / name, "A3", cfg, None))
              for role, name in verify.DENOMINATOR_FILES.items()}
    scores["frontier"]["sha256"] = "0" * 64  # stale: the file changed after freezing
    (tests / verify.FROZEN_FILE).write_text(json.dumps({"schema": verify.FROZEN_SCHEMA, "lane": "A3", "scores": scores}))
    verify.verify_optimize("A3", tests / "baseline.py", tmp_path / "out", tests_dir=tests)
    details = json.loads((tmp_path / "out" / "details.json").read_text())
    assert sorted(calls) == ["frontier.py", "submission.py"]
    assert {r: d["source"] for r, d in details["denominators"].items()} == {
        "start": "frozen", "reference": "frozen", "frontier": "live"}
    assert details["start_speedup"] == 2.0 and details["ref_speedup"] == 2.0
    assert details["outcome"] == verify.classify(True, 2.0e-4, scores["start"], scores["reference"])["outcome"]


# --------------------------------------------------------------------------- real verifier

def test_biased_sampler_is_invalid(verify, tmp_path):
    """examples/samplers/ula_wrong.py (unadjusted Langevin) fails the invariance checks: outcome
    invalid, reward 0. Uses the frozen denominators, so only the candidate is scored."""
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-I", str(TASKS / "a0-ground-up" / "tests" / "verify.py"), "optimize", "--lane", "A0",
         "--submission", str(ROOT / "examples" / "samplers" / "ula_wrong.py"), "--out-dir", str(out)],
        capture_output=True, text=True, cwd=tmp_path, check=False,
    )
    assert r.returncode == 1, r.stdout[-2000:] + r.stderr[-2000:]
    reward = json.loads((out / "reward.json").read_text())
    details = json.loads((out / "details.json").read_text())
    assert reward["reward"] == 0.0 and reward["passed"] == 0 and reward["solved"] == 0
    assert details["outcome"] == "invalid" and details["error"]
    assert all(d["source"] == "frozen" for d in details["denominators"].values()), details["denominators"]


def test_rewind_reports_outcome(verify, tmp_path):
    fixtures = ROOT / "autoresearch" / "lanes" / "S3" / "fixtures"
    for sub, want in ((fixtures / "missing_jacobian_a4_ref.py", "solved"),
                      (TASKS / "s3-missing-jacobian-a4" / "environment" / "baseline.py", "invalid")):
        out = tmp_path / want
        verify.verify_rewind("s3-missing-jacobian-a4", sub, out, TASKS / "s3-missing-jacobian-a4" / "tests" / "tasks.json")
        reward = json.loads((out / "reward.json").read_text())
        details = json.loads((out / "details.json").read_text())
        assert details["outcome"] == want and reward["solved"] == int(want == "solved") == reward["passed"]


@pytest.mark.slow
def test_unchanged_starting_code_gives_no_gain_and_matches_its_frozen_score(verify, tmp_path):
    """The real verifier on lane A3's own starting code: outcome no_gain, and the live candidate
    score equals the frozen starting-code score exactly (scoring is deterministic)."""
    tests = TASKS / "a3-integrator-trajectory" / "tests"
    rc = verify.verify_optimize("A3", tests / "baseline.py", tmp_path / "out", tests_dir=tests)
    details = json.loads((tmp_path / "out" / "details.json").read_text())
    assert rc == 0 and details["outcome"] == "no_gain"
    assert details["start_speedup"] == 1.0
    assert all(d["source"] == "frozen" for d in details["denominators"].values())
