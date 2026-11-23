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


# ----------------------------------------------------------------- confirmation of rejections

class _Tgt:
    def __init__(self, family, multimodal=False):
        self.family, self.multimodal = family, multimodal


_FAMILIES = ("gaussian_illcond", "funnel", "banana")


def _fake_res(verify, effs, reasons):
    """A judge-shaped Tier 1 result for one-target-per-class suites; reasons[r][i] '' = passed."""
    suites, runs = [], []
    for r, (eff_r, why_r) in enumerate(zip(effs, reasons, strict=True)):
        suite = [_Tgt(f) for f in _FAMILIES]
        per_target = [{"family": f, "dim": 6, "rotated": True, "passed": not why, "reason": why,
                       "emp_eff": e, "short_horizon_eff": e / 2, "long_horizon_eff": e * 2}
                      for f, e, why in zip(_FAMILIES, eff_r, why_r, strict=True)]
        suites.append((r, suite))
        runs.append({"per_target": per_target, "passed": not any(why_r)})
    res = {"per_repeat": runs, "per_target": [t for run in runs for t in run["per_target"]],
           "passed": all(run["passed"] for run in runs), "b2_max": 0.01, "overhead_ratio": 1.0}
    rebuilt = verify.scores_from_targets(res, suites)
    ok = res["passed"]
    for run, sm in zip(runs, rebuilt["per_repeat_soft_min"], strict=True):
        run["soft_min_eff"] = sm if run["passed"] else 0.0
        run["by_class"] = {}
    for k in ("soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min"):
        res[k] = rebuilt[k] if ok else 0.0  # the judge zeroes the soft-min of a failed pass
    return res, suites


REJECT = "invariance rejected on z0_sq: |z|=4.1 > 3.9 (b2_max=0.3)"
STUCK = "stuck/identity kernel: late mean latent step 0 < 0.001"


@pytest.fixture
def judge_stub(verify, monkeypatch):
    """Replace Tier 0 and Tier 1 with scripted results keyed by seed; records the Tier 1 calls."""
    script: dict = {}
    calls: list[int] = []
    monkeypatch.setattr(verify.autoresearch, "_eval_tier0", lambda *a, **k: (True, {"stage": "passed"}))

    def tier1(path, lane, cfg, seed):
        calls.append(seed)
        return _fake_res(verify, *script[seed])

    monkeypatch.setattr(verify, "_tier1", tier1)
    return script, calls


def _cfg(verify):
    return verify.scoring_config(repeats=2)


GOOD = ([[1e-3, 2e-3, 3e-3], [1e-3, 2e-3, 3e-3]], [["", "", ""], ["", "", ""]])


def test_statistical_rejection_confirmed_as_chance_keeps_first_run_numbers(verify, judge_stub, tmp_path):
    script, calls = judge_stub
    cfg = _cfg(verify)
    first = ([[1e-3, 2e-3, 3e-3], [2e-3, 2e-3, 3e-3]], [["", REJECT, ""], ["", "", ""]])
    script[cfg["seed"]] = first
    script[cfg["confirmation_seed"]] = ([[9e-3, 9e-3, 9e-3]] * 2, GOOD[1])
    tmp_path.joinpath("s.py").write_text("x = 1\n")
    r = verify.score_file(tmp_path / "s.py", "A2", cfg, tmp_path / "w")
    assert calls == [cfg["seed"], cfg["confirmation_seed"]]
    assert r["valid"] and r["tier1_ok"] and r["failed_targets"] == []
    c = r["confirmation"]
    assert c["ran"] and c["passed"] and c["seed"] == cfg["confirmation_seed"]
    assert c["efficiency_source"] == "first_run_paired" and "banana" not in str(c["failed_targets"])
    assert "funnel" in c["failed_targets_first_run"][0]
    res, suites = _fake_res(verify, *first)
    assert r["soft_min_eff"] == pytest.approx(verify.scores_from_targets(res, suites)["soft_min_eff"])
    assert r["soft_min_eff"] < 9e-3  # not the confirmation run's numbers


def test_confirmed_rejection_is_invalid(verify, judge_stub, tmp_path):
    script, _calls = judge_stub
    cfg = _cfg(verify)
    bad = (GOOD[0], [["", REJECT, ""], ["", "", ""]])
    script[cfg["seed"]] = bad
    script[cfg["confirmation_seed"]] = (GOOD[0], [["", "", ""], [REJECT, "", ""]])
    tmp_path.joinpath("s.py").write_text("x = 1\n")
    r = verify.score_file(tmp_path / "s.py", "A2", cfg, tmp_path / "w")
    assert not r["valid"] and r["confirmation"]["ran"] and not r["confirmation"]["passed"]
    assert str(cfg["confirmation_seed"]) in r["error"] and "gaussian_illcond" in r["error"]
    assert r["soft_min_eff"] == 0.0 and r["per_repeat_soft_min"][0] == 0.0 and r["per_repeat_soft_min"][1] > 0


@pytest.mark.parametrize("reasons", [
    [["", STUCK, ""], ["", "", ""]],                       # stuck kernel: not a statistical test
    [["", REJECT, STUCK], ["", "", ""]],                   # mixed
    [["adapt() failed: BudgetExceeded: more than 300.0 s", "", ""], ["", "", ""]],  # timeout
    [["", "", ""], ["adapt cost 300000 exceeded cap 204800", "", ""]],
])
def test_non_statistical_failures_are_invalid_at_once(verify, judge_stub, tmp_path, reasons):
    script, calls = judge_stub
    cfg = _cfg(verify)
    script[cfg["seed"]] = (GOOD[0], reasons)
    tmp_path.joinpath("s.py").write_text("x = 1\n")
    r = verify.score_file(tmp_path / "s.py", "A2", cfg, tmp_path / "w")
    assert not r["valid"] and calls == [cfg["seed"]] and not r["confirmation"]["ran"]


def test_tier0_failure_gets_no_tier1_and_no_confirmation(verify, judge_stub, tmp_path, monkeypatch):
    _script, calls = judge_stub
    monkeypatch.setattr(verify.autoresearch, "_eval_tier0",
                        lambda *a, **k: (False, {"stage": "invariance_smoke", "error": "rejected"}))
    tmp_path.joinpath("s.py").write_text("x = 1\n")
    r = verify.score_file(tmp_path / "s.py", "A2", _cfg(verify), tmp_path / "w")
    assert not r["valid"] and calls == [] and r["confirmation"] == {"ran": False}
    assert not verify.score_file(tmp_path / "missing.py", "A2", _cfg(verify), tmp_path / "w")["valid"]


def test_no_ratio_when_the_first_run_has_no_usable_numbers(verify, judge_stub, tmp_path):
    script, _calls = judge_stub
    cfg = _cfg(verify)
    script[cfg["seed"]] = ([[1e-3, 0.0, 3e-3], [1e-3, 2e-3, 3e-3]], [["", REJECT, ""], ["", "", ""]])
    script[cfg["confirmation_seed"]] = ([[4e-3, 4e-3, 4e-3]] * 2, GOOD[1])
    tmp_path.joinpath("s.py").write_text("x = 1\n")
    r = verify.score_file(tmp_path / "s.py", "A2", cfg, tmp_path / "w")
    # A score from the confirmation seed is never divided by primary-seed denominators.
    assert r["valid"] and r["unpaired"] and r["confirmation"]["efficiency_source"] == "none"
    # No efficiency survives, so the file cannot serve as a denominator either, live or frozen.
    assert r["soft_min_eff"] == 0.0 and verify.classify(True, 1.0, r, r)["outcome"] == "unscorable"


def test_confirmed_pass_in_the_verifier_output(verify, judge_stub, tmp_path, monkeypatch):
    script, _calls = judge_stub
    cfg = verify.scoring_config()
    script[cfg["seed"]] = ([[3e-4, 3e-4, 3e-4]] * cfg["repeats"], [["", REJECT, ""]] + [["", "", ""]] * (cfg["repeats"] - 1))
    script[cfg["confirmation_seed"]] = ([[3e-4, 3e-4, 3e-4]] * cfg["repeats"], [["", "", ""]] * cfg["repeats"])
    tests = _tests_dir(tmp_path, "a2-funnel-dynamics")
    real = verify.score_file
    monkeypatch.setattr(verify, "score_file", lambda path, lane, cfg, work_dir, **k: (
        real(path, lane, cfg, work_dir, **k) if Path(path).name == "submission.py"
        else _stub_score({"baseline.py": 1e-4, "reference.py": 2e-4, "frontier.py": 4e-4})(path, lane, cfg, work_dir)))
    sub = tmp_path / "cand.py"
    sub.write_text("x = 1\n")
    assert verify.verify_optimize("A2", sub, tmp_path / "out", tests_dir=tests) == 0
    reward = json.loads((tmp_path / "out" / "reward.json").read_text())
    details = json.loads((tmp_path / "out" / "details.json").read_text())
    assert reward["confirmed"] == 1 and reward["passed"] == 1 and details["outcome"] == "solved"
    c = details["confirmation"]
    assert c["ran"] and c["passed"] and c["seed"] == verify.CONFIRMATION_SEED and c["failed_targets_first_run"]
    assert reward["start_speedup"] == pytest.approx(3.0, rel=1e-4)


def test_confirmation_seed_is_disjoint_from_the_verifier_seeds(verify):
    used = verify.HIDDEN_VERIFIER_SEED + 7919 * verify.VERIFIER_REPEATS + 100_000
    assert abs(verify.CONFIRMATION_SEED - verify.HIDDEN_VERIFIER_SEED) > used - verify.HIDDEN_VERIFIER_SEED
    assert verify.scoring_config()["confirmation_seed"] == verify.CONFIRMATION_SEED


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
    """The reference fix is solved; each of the three buggy starting files stays invalid. The
    rewind check is one target on a fixed seed, so it gets no confirmation run (README)."""
    fixtures = ROOT / "autoresearch" / "lanes" / "S3" / "fixtures"
    cases = [("s3-missing-jacobian-a4", fixtures / "missing_jacobian_a4_ref.py", "solved")]
    cases += [(t, TASKS / t / "environment" / "baseline.py", "invalid")
              for t in ("s3-missing-jacobian-a4", "s3-position-gated-mh", "s3-unadjusted-ula")]
    for task, sub, want in cases:
        out = tmp_path / f"{task}-{want}"
        verify.verify_rewind(task, sub, out, TASKS / task / "tests" / "tasks.json")
        reward = json.loads((out / "reward.json").read_text())
        details = json.loads((out / "details.json").read_text())
        assert details["outcome"] == want, (task, details.get("reason"))
        assert reward["solved"] == int(want == "solved") == reward["passed"]


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


def test_a_statistical_rejection_does_not_hide_a_stuck_kernel(verify):
    # The judge joins every failed gate of one target with "; ".
    both = {"per_target": [{"passed": False, "reason": REJECT + "; " + STUCK}]}
    only = {"per_target": [{"passed": False, "reason": REJECT}, {"passed": True, "reason": ""}]}
    assert not verify._only_statistical(both)
    assert verify._only_statistical(only)
