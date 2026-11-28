"""The lane trial loop (harness/autoresearch.py): pre-checks, one grader run per trial, and the
decision from the speed-up over the lane's yardstick.

Decision tests replace harness/grade.py::score_file with a stub that reads the score from the
candidate file, so they run in milliseconds. Tests of the real Tier 0 and Tier 1 path run at a
small exploratory configuration (dims (4,), one repeat, short chains)."""

import json
import re
import shutil
from pathlib import Path

import pytest

from harness import autoresearch, flywheel, grade, invariance, judge, real_gate, shapes

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "samplers"
A1_SUB = ROOT / "autoresearch" / "lanes" / "A1" / "submission.py"
A2_START = ROOT / "autoresearch" / "lanes" / "A2" / "start.py"

# Small enough for the fast suite. Any non-standard configuration is exploratory.
SMALL = grade.config(dims=(4,), repeats=1, n_chains=36, k_steps=10, adapt_draws=60)


# --------------------------------------------------------------------------- stubbed grader

def _score(sm: float | None, *, tier0: bool = True, unpaired: bool = False, by_class: dict | None = None,
           overhead: float = 2.0) -> dict:
    """A grader result. sm=None means a failed Tier 1 check."""
    if not tier0:
        return {"valid": False, "tier0_ok": False, "tier1_ok": False, "error": "tier0 invariance_smoke: stub",
                "tier0_info": {"stage": "invariance_smoke"}, "confirmation": {"ran": False}, "wall_s": 0.1}
    valid = sm is not None
    sm = float(sm or 0.0)
    return {"valid": valid, "tier0_ok": True, "tier1_ok": valid, "unpaired": unpaired,
            "soft_min_eff": 0.0 if unpaired else sm, "short_horizon_soft_min": sm / 2,
            "long_horizon_soft_min": sm * 2, "median_adapt_cost": 100.0, "b2_max": 0.01,
            "overhead_ratio": overhead, "by_class": by_class or {"banana": sm, "funnel": 3 * sm},
            "per_repeat_soft_min": [sm] * 8, "failed_targets": [] if valid else ["banana(d=6): stub"],
            "error": None if valid else "tier1: banana(d=6): stub",
            "confirmation": {"ran": False}, "wall_s": 0.1}


_MARK = re.compile(r"^# GRADE (\S+)(.*)$", re.M)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """Lane A2 in a temporary autoresearch dir, with a yardstick of soft-min 1.0 and a grader
    stub. A candidate file names its result in a `# GRADE <soft-min | tier0 | tier1 | unpaired>`
    line, optionally followed by class overrides such as `funnel=9`. Returns (autoresearch_dir,
    lane_dir, grader calls, posted rows)."""
    lane_dir = tmp_path / "lanes" / "A2"
    lane_dir.mkdir(parents=True)
    shutil.copy2(A2_START, lane_dir / "start.py")
    shutil.copy2(A2_START, lane_dir / "submission.py")
    yd = grade.yardstick_entry("A2", lane_dir / "start.py", grade.config(), judge.judge_digest(), _score(1.0))
    (lane_dir / "yardstick.json").write_text(json.dumps(yd))
    calls, posted = [], []

    def stub(path, lane_name, cfg, work_dir, **kw):
        calls.append((Path(path).name, cfg))
        m = _MARK.search(Path(path).read_text())
        mark = m.group(1)
        if mark == "tier0":
            return _score(None, tier0=False)
        if mark == "tier1":
            return _score(None)
        if mark == "unpaired":
            return _score(5.0, unpaired=True)
        sm = float(mark)
        classes = {"banana": sm, "funnel": 3 * sm}
        classes.update({k: float(v) for k, v in (kv.split("=") for kv in m.group(2).split())})
        return _score(sm, by_class=classes)

    monkeypatch.setattr(grade, "score_file", stub)
    monkeypatch.setattr(flywheel, "post_trial_row", lambda row, **kw: posted.append(row))
    return tmp_path, lane_dir, calls, posted


def _trial(ar_dir, lane_dir, text: str | None, hyp: str = "h", **kw) -> dict:
    if text is not None:
        (lane_dir / "submission.py").write_text(text)
    return autoresearch.evaluate_lane_trial("A2", hyp, autoresearch_dir=ar_dir, run_id="r", use_queue=False, **kw)


def _long(score: float | str) -> str:
    """A candidate with more lines than start.py, so it is never simpler."""
    return f"# GRADE {score}\n" + A2_START.read_text() + f"\n_PAD = '{score}'\n"


def _start(lane_dir) -> None:
    (lane_dir / "submission.py").write_bytes(A2_START.read_bytes())


def _short(score: float | str) -> str:
    return f"# GRADE {score}\nimport numpy as np\n_V = '{score}'\n"


def test_baseline_identical_to_start_scores_one_without_a_grader_run(lane):
    ar_dir, lane_dir, calls, posted = lane
    row = _trial(ar_dir, lane_dir, None, "baseline")
    assert calls == []
    assert row["decision"] == "keep_baseline" and row["speedup"] == 1.0 and row["graded_from"] == "yardstick"
    yd = json.loads((lane_dir / "yardstick.json").read_text())
    assert row["yardstick"] == yd["id"] and row["soft_min_eff"] == 1.0 and row["exploratory"] is False
    assert (lane_dir / "runs" / "r" / "incumbent.py").read_bytes() == A2_START.read_bytes()
    assert posted[0]["speedup"] == 1.0 and posted[0]["yardstick"] == yd["id"]
    report = autoresearch.trial_report(row, "r")
    metrics = [ln for ln in report if ln.startswith("METRIC ")]
    assert metrics[0] == "METRIC speedup=1.0" and "METRIC soft_min_eff=1.0" in metrics
    assert report[-1] == "Best so far in run r: 1.0x the starting code (T0001). This trial was kept (keep_baseline)."


def test_a_baseline_that_differs_from_start_is_graded(lane):
    ar_dir, lane_dir, calls, _ = lane
    row = _trial(ar_dir, lane_dir, _long(0.8))
    assert len(calls) == 1 and calls[0][1] == grade.config()
    assert row["decision"] == "keep_baseline" and row["speedup"] == 0.8 and row["graded_from"] == "grader"


def test_kept_at_the_margin_and_not_below_it(lane):
    ar_dir, lane_dir, _, _ = lane
    _trial(ar_dir, lane_dir, None)
    below = _trial(ar_dir, lane_dir, _long(1.04))
    assert below["decision"] == "discard_no_gain" and below["speedup"] == 1.04 and below["best_speedup"] == 1.0
    at = _trial(ar_dir, lane_dir, _long(1.05))
    assert at["decision"] == "keep" and at["ratio_vs_best"] == 1.05
    assert (lane_dir / "runs" / "r" / "incumbent.py").read_text() == _long(1.05)
    # The next keep is measured against the new best, never against the starting code.
    nxt = _trial(ar_dir, lane_dir, _long(1.10))
    assert nxt["decision"] == "discard_no_gain" and nxt["best_speedup"] == 1.05 and nxt["best_trial"] == "T0003"
    report = autoresearch.trial_report(nxt, "r")
    assert report[-1] == "Best so far in run r: 1.05x the starting code (T0003). This trial was not kept (discard_no_gain)."


def test_tie_band_keeps_only_a_simpler_file(lane):
    ar_dir, lane_dir, _, _ = lane
    _trial(ar_dir, lane_dir, None)
    assert _trial(ar_dir, lane_dir, _long(0.99))["decision"] == "discard_no_gain"
    row = _trial(ar_dir, lane_dir, _short(0.99))
    assert row["decision"] == "keep_simpler"
    # Outside the band, simpler is not enough.
    assert _trial(ar_dir, lane_dir, _short(0.97))["decision"] == "discard_no_gain"
    # The best stays the baseline's 1.0, not the simpler file's 0.99.
    assert _trial(ar_dir, lane_dir, _long(1.04))["best_speedup"] == 1.0


def test_a_failed_check_never_yields_a_keep(lane):
    ar_dir, lane_dir, _, posted = lane
    first = _trial(ar_dir, lane_dir, _long("tier1"))
    assert first["decision"] == "discard_tier1" and first["speedup"] is None and first["error"]
    assert not (lane_dir / "runs" / "r" / "incumbent.py").exists()  # an invalid file is never the baseline
    _start(lane_dir)
    assert _trial(ar_dir, lane_dir, None)["decision"] == "keep_baseline"
    for mark, decision in (("tier0", "discard_tier0"), ("tier1", "discard_tier1"), ("unpaired", "discard_tier1")):
        row = _trial(ar_dir, lane_dir, _short(mark) + f"# {decision}\n")
        assert row["decision"] == decision and row["speedup"] is None and row["archive_cells"] == []
        assert autoresearch.trial_report(row, "r")[1] == "METRIC speedup=0"
    assert (lane_dir / "runs" / "r" / "incumbent.py").read_bytes() == A2_START.read_bytes()
    assert all(p["decision"] not in flywheel.KEPT_DECISIONS or p["speedup"] for p in posted)


def test_exploratory_trials_are_never_kept(lane):
    ar_dir, lane_dir, calls, posted = lane
    explore = grade.config(repeats=2)
    row = _trial(ar_dir, lane_dir, _long(9.0), cfg=explore)
    assert calls[-1][1] == explore
    assert row["decision"] == "exploratory" and row["exploratory"] is True
    assert row["speedup"] is None and row["yardstick"] is None and row["soft_min_eff"] == 9.0
    assert not (lane_dir / "runs" / "r" / "incumbent.py").exists()
    assert not (lane_dir / "runs" / "r" / "archive").exists()
    assert posted[-1]["speedup"] is None and posted[-1]["yardstick"] is None
    report = autoresearch.trial_report(row, "r")
    assert not any(ln.startswith("METRIC speedup") for ln in report)
    assert report[-1].startswith("EXPLORATORY: ") and "was not kept" in report[-1]
    # An exploratory trial is not the run's baseline.
    _start(lane_dir)
    assert _trial(ar_dir, lane_dir, None)["decision"] == "keep_baseline"
    # A different seed is exploratory too.
    assert _trial(ar_dir, lane_dir, _long(9.5), cfg=grade.config(seed=1))["decision"] == "exploratory"
    # The file of an exploratory trial can be graded at the standard setup afterwards, but the
    # same file is still a duplicate of an earlier trial at the same kind of setup.
    (lane_dir / "submission.py").write_text(_long(9.0))
    assert _trial(ar_dir, lane_dir, None)["decision"] != "discard_duplicate"
    dup = _trial(ar_dir, lane_dir, None)
    assert dup["decision"] == "discard_duplicate"
    assert any("--allow-duplicate" in ln for ln in autoresearch.trial_report(dup, "r"))
    assert _trial(ar_dir, lane_dir, _long(9.5), cfg=explore)["decision"] == "discard_duplicate"


def test_archive_only_when_a_class_sets_a_record(lane):
    ar_dir, lane_dir, _, _ = lane
    base = _trial(ar_dir, lane_dir, None)
    assert sorted(base["archive_cells"]) == ["banana__oh_med", "funnel__oh_med"]
    # No overall gain, but the funnel class beats its specialist by more than the margin.
    row = _trial(ar_dir, lane_dir, _long(1.0) + "# GRADE 1.0 funnel=9\n".replace("# GRADE", "# grade"))
    assert row["decision"] == "discard_no_gain" and row["archive_cells"] == []
    (lane_dir / "submission.py").write_text("# GRADE 1.0 funnel=9\n" + A2_START.read_text() + "\n_V = 2\n")
    row = _trial(ar_dir, lane_dir, None)
    assert row["decision"] == "archive_only" and row["archive_cells"] == ["funnel__oh_med"]
    cell = json.loads((lane_dir / "runs" / "r" / "archive" / "funnel__oh_med" / "best.json").read_text())
    assert cell["score"] == 9.0 and cell["trial_id"] == row["trial_id"] and cell["yardstick"] == row["yardstick"]
    assert (lane_dir / "runs" / "r" / "incumbent.py").read_bytes() == A2_START.read_bytes()


def test_ledger_and_results_tsv_carry_speedup_and_yardstick(lane):
    ar_dir, lane_dir, _, _ = lane
    _trial(ar_dir, lane_dir, None)
    _trial(ar_dir, lane_dir, _long(1.2))
    run = lane_dir / "runs" / "r"
    rows = [json.loads(ln) for ln in (run / "ledger.jsonl").read_text().splitlines()]
    assert [r["speedup"] for r in rows] == [1.0, 1.2] and len({r["yardstick"] for r in rows}) == 1
    lines = (run / "results.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    assert {"speedup", "yardstick", "best_speedup", "soft_min_eff", "decision"} <= set(header)
    assert lines[2].split("\t")[header.index("speedup")] == "1.2"


def test_a_stale_yardstick_stops_a_standard_trial(lane):
    ar_dir, lane_dir, calls, _ = lane
    (lane_dir / "start.py").write_text(A2_START.read_text() + "\n# edited\n")
    with pytest.raises(autoresearch.YardstickError, match="start.py changed"):
        _trial(ar_dir, lane_dir, _long(1.2))
    yd = json.loads((lane_dir / "yardstick.json").read_text())
    (lane_dir / "start.py").write_bytes(A2_START.read_bytes())
    (lane_dir / "yardstick.json").write_text(json.dumps({**yd, "judge_digest": "0" * 16}))
    with pytest.raises(autoresearch.YardstickError, match="judge digest"):
        _trial(ar_dir, lane_dir, _long(1.2))
    (lane_dir / "yardstick.json").write_text(json.dumps({**yd, "soft_min_eff": 2.0}))
    with pytest.raises(autoresearch.YardstickError, match="id does not match"):
        _trial(ar_dir, lane_dir, _long(1.2))
    assert calls == []
    # Exploratory trials need no yardstick.
    assert _trial(ar_dir, lane_dir, _long(1.2), cfg=SMALL)["decision"] == "exploratory"


def test_decide_rule():
    m, band = autoresearch.IMPROVEMENT_MARGIN, autoresearch.TIE_BAND
    assert m == grade.IMPROVEMENT_MARGIN == 1.05 and band == 0.02
    assert autoresearch.decide(0.5, None, False) == "keep_baseline"
    assert autoresearch.decide(1.05, 1.0, False) == "keep"
    assert autoresearch.decide(1.0499, 1.0, True) == "keep_simpler"
    assert autoresearch.decide(1.0499, 1.0, False) == "discard_no_gain"
    assert autoresearch.decide(0.98, 1.0, True) == "keep_simpler"
    assert autoresearch.decide(0.979, 1.0, True) == "discard_no_gain"


def test_rerun_reproduces_the_posted_speedup_or_reports_a_mismatch(lane, monkeypatch):
    ar_dir, lane_dir, _, _ = lane
    yd = json.loads((lane_dir / "yardstick.json").read_text())
    cand = lane_dir / "cand.py"
    cand.write_text(_long(1.3))
    gate_calls = []
    gate = real_gate.load()
    rows = [{"model": m, "status": "right", "reason": "right", "ess_per_cost": 0.01} for m in gate["required"]]
    rows += [{"model": m, "status": "unproven", "reason": "chains have not mixed"} for m in gate["reported"]]

    def fake_gate(path, lane_name, **kw):
        gate_calls.append((path, lane_name, kw))
        return {**real_gate.summarize(gate, rows), "submission_sha256": "abc"}

    monkeypatch.setattr(real_gate, "run", fake_gate)
    ok = autoresearch.rerun_for_verification(cand, "A2", claimed_speedup=1.3, claimed_yardstick=yd["id"],
                                             autoresearch_dir=ar_dir, use_queue=False)
    assert ok["comparable"] and ok["passed"] and ok["metrics"]["speedup"] == 1.3 and "reproduced" in ok["reason"]
    # A reproduced rerun then runs the full real-posterior gate on the same file.
    assert len(gate_calls) == 1 and gate_calls[0][0] == cand and gate_calls[0][1] == "A2"
    assert gate_calls[0][2]["use_queue"] is False and "models" not in gate_calls[0][2]
    assert ok["gate"]["ok"] and ok["gate_error"] is None
    assert ok["metrics"]["real_gate"] == {
        "id": gate["id"], "ok": True, "required_passed": len(gate["required"]), "required_total": len(gate["required"]),
        "models": {**{m: "right" for m in gate["required"]}, **{m: "unproven" for m in gate["reported"]}},
        "submission_sha256": "abc",
    }
    # --no-real-gate skips it; a crash in it leaves the synthetic result standing.
    skipped = autoresearch.rerun_for_verification(cand, "A2", claimed_speedup=1.3, claimed_yardstick=yd["id"],
                                                  autoresearch_dir=ar_dir, use_queue=False, run_gate=False)
    assert skipped["passed"] and skipped["gate"] is None and "real_gate" not in skipped["metrics"] and len(gate_calls) == 1
    monkeypatch.setattr(real_gate, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no bridgestan")))
    crashed = autoresearch.rerun_for_verification(cand, "A2", claimed_speedup=1.3, claimed_yardstick=yd["id"],
                                                  autoresearch_dir=ar_dir, use_queue=False)
    assert crashed["passed"] and crashed["gate"] is None and "no bridgestan" in crashed["gate_error"]
    monkeypatch.setattr(real_gate, "run", lambda *a, **k: pytest.fail("a failed rerun must not run the gate"))
    bad = autoresearch.rerun_for_verification(cand, "A2", claimed_speedup=1.5, claimed_yardstick=yd["id"],
                                              autoresearch_dir=ar_dir, use_queue=False)
    assert bad["comparable"] and not bad["passed"] and "mismatch" in bad["reason"]
    other = autoresearch.rerun_for_verification(cand, "A2", claimed_speedup=1.3, claimed_yardstick="A2-old",
                                                autoresearch_dir=ar_dir, use_queue=False)
    assert not other["comparable"] and "A2-old" in other["reason"]
    cand.write_text(_long("tier1"))
    failed = autoresearch.rerun_for_verification(cand, "A2", claimed_speedup=1.3, claimed_yardstick=yd["id"],
                                                 autoresearch_dir=ar_dir, use_queue=False)
    assert failed["comparable"] and not failed["passed"] and "failed a check" in failed["reason"]


def test_trial_payload_posts_speedup_and_yardstick():
    graded = {"trial_id": "T0002", "lane": "A2", "decision": "keep", "speedup": 1.2, "yardstick": "A2-abc",
              "soft_min_eff": 3e-4, "ratio_vs_best": 1.1}
    body = flywheel.trial_payload(graded, run_id="r")
    assert body["speedup"] == 1.2 and body["yardstick"] == "A2-abc" and body["soft_min_eff"] == 3e-4
    assert body["ratio_vs_incumbent"] == 1.1
    explore = flywheel.trial_payload({**graded, "decision": "exploratory", "yardstick": None, "speedup": None},
                                     run_id="r")
    assert explore["speedup"] is None and explore["yardstick"] is None
    assert "keep_incumbent_failed" not in flywheel.KEPT_DECISIONS


# ------------------------------------------------------------------------- real grader, small

def test_a1_narrow_contract_and_novelty_filter(tmp_path):
    """Lane A1 accepts only adapt_metric(), rejects a transition override, and dedupes comments."""
    lane_dir = tmp_path / "lanes" / "A1"
    lane_dir.mkdir(parents=True)
    shutil.copy2(A1_SUB, lane_dir / "submission.py")

    def run(hyp, **kw):
        return autoresearch.evaluate_lane_trial("A1", hyp, cfg=SMALL, autoresearch_dir=tmp_path,
                                                run_id="test_run", use_queue=False, **kw)

    r1 = run("A1 adapt_metric at a small configuration")
    assert r1["tier0_ok"] is True and r1["tier1_ok"] is True, r1
    assert r1["decision"] == "exploratory" and r1["soft_min_eff"] > 0 and r1["speedup"] is None
    assert r1["dims"] == [4] and r1["repeats"] == 1 and r1["horizon_steps"] == 500 and r1["magic_constants"] > 0
    run_dir = lane_dir / "runs" / "test_run"
    assert not (run_dir / "incumbent.py").exists() and not (lane_dir / "incumbent.py").exists()

    dry = run("dry run duplicate", dry_run=True)
    # A dry run grades nothing, so its row carries no verdict of the checks.
    assert "tier0_ok" not in dry and "tier1_ok" not in dry
    assert dry["decision"] == "dry_run_duplicate" and dry["is_duplicate"] is True
    assert len((run_dir / "ledger.jsonl").read_text().splitlines()) == 1

    assert run("duplicate")["decision"] == "discard_duplicate"
    (lane_dir / "submission.py").write_text(A1_SUB.read_text() + "\n# cosmetic comment only\n")
    assert run("comment-only change")["decision"] == "discard_duplicate"

    shutil.copy2(EXAMPLES / "hmc.py", lane_dir / "submission.py")
    r4 = run("replace the fixed A1 transition kernel")
    assert r4["tier0_ok"] is False and r4["decision"] == "discard_tier0"
    assert "adapt_metric" in r4["tier0_info"]["error"]

    ledger = [json.loads(ln) for ln in (run_dir / "ledger.jsonl").read_text().splitlines()]
    tsv = (run_dir / "results.tsv").read_text().splitlines()
    assert len(ledger) == 4 and len(tsv) == 5


def test_full_kernel_lane_catches_stuck_kernels_and_logs_errors(tmp_path, monkeypatch):
    lane_dir = tmp_path / "lanes" / "A0"
    lane_dir.mkdir(parents=True)

    def run(text, hyp):
        (lane_dir / "submission.py").write_text(text)
        return autoresearch.evaluate_lane_trial("A0", hyp, cfg=SMALL, autoresearch_dir=tmp_path,
                                                run_id="a0_run", use_queue=False)

    hmc = (EXAMPLES / "hmc.py").read_text()
    r1 = run(hmc, "full HMC")
    assert r1["tier0_ok"] and r1["tier1_ok"] and r1["decision"] == "exploratory", r1
    r2 = run(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng):\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    return chain.position, None\n",
        "identity kernel",
    )
    assert r2["tier0_ok"] is False and r2["decision"] == "discard_tier0"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated tier1 failure")

    monkeypatch.setattr(grade, "eval_suite", boom)
    r3 = run(hmc + "\nX_TAG = 1\n", "crash resilience")
    assert r3["decision"] == "discard_error" and "simulated tier1 failure" in r3["error"]
    assert len((lane_dir / "runs" / "a0_run" / "ledger.jsonl").read_text().splitlines()) == 3


def test_default_seed_schedule_no_overflow_on_banana_and_rosenbrock():
    """The shipped A1 metric must not overflow on banana or hybrid Rosenbrock at the public seed."""
    for i in range(3):
        seed = grade.PUBLIC_SEED + i * 7919
        for tgt in (shapes.Banana(4, seed + 4, rotated=True), shapes.HybridRosenbrock(4, seed + 5, b=5.0)):
            res = invariance.check_target_invariance(
                tgt, A1_SUB, seed=seed, n_chains=24, k_steps=8, adapt_draws=40, lane="A1", isolated=False,
            )
            assert "OverflowError" not in res.reason


def test_horizon_amortization_and_simplicity_and_queue(tmp_path, monkeypatch):
    import threading
    import time
    from harness import queue

    tgt = shapes.IllCondGaussian(4, seed=7, rotated=True, kappa=10.0)
    r_amort = invariance.check_target_invariance(
        tgt, A1_SUB, seed=7, n_chains=24, k_steps=10, adapt_draws=40,
        lane="A1", isolated=False, horizon_steps=500,
    )
    assert r_amort.horizon_steps == 500
    assert abs(r_amort.adapt_cost_amortized - r_amort.adapt_cost * (10.0 / 500.0)) < 1e-6

    assert autoresearch._is_simpler(100, 100, 5, 6, 4.0, 4.0) is True
    assert autoresearch._is_simpler(100, 100, 6, 6, 3.0, 4.0) is True
    assert autoresearch._is_simpler(100, 100, 6, 6, 3.8, 4.0) is False
    assert autoresearch._is_simpler(105, 100, 2, 10, 1.0, 4.0) is False  # larger LOC not simpler

    monkeypatch.setenv("APIN_FORCE_QUEUE", "1")
    lock_file = tmp_path / "q.lock"
    events: list[tuple[str, float]] = []

    def worker(tag: str, hold: float):
        with queue.eval_slot(tag, enabled=True, lock_path=lock_file):
            events.append((f"{tag}:enter", time.perf_counter()))
            time.sleep(hold)
            events.append((f"{tag}:exit", time.perf_counter()))

    t1 = threading.Thread(target=worker, args=("a", 0.15))
    t2 = threading.Thread(target=worker, args=("b", 0.05))
    t1.start()
    time.sleep(0.03)
    t2.start()
    t1.join()
    t2.join()
    assert len(events) == 4
    assert events[1][0].endswith(":exit") and events[2][0].endswith(":enter")
    assert events[2][1] >= events[1][1] - 1e-4


def test_aux_spec_ndarray_json_and_dual_horizon_and_dim_ladder():
    import numpy as np
    from harness import logdensity

    spec = logdensity.normalize_aux_spec({"kind": "normal", "diag": np.array([1.0, 2.0, 3.0])})
    assert spec["diag"] == [1.0, 2.0, 3.0]
    assert json.dumps(spec)

    suite = grade.build_suite(11, (6, 20))
    assert len(suite) == 13  # 8 targets at d=6 + 5 targets at d=20
    assert any(t.dim == 20 for t in suite) and any(t.dim == 6 for t in suite)

    tgt = shapes.IllCondGaussian(4, seed=9, rotated=True, kappa=10.0)
    r = invariance.check_target_invariance(
        tgt, A1_SUB, seed=9, n_chains=24, k_steps=10, adapt_draws=40, lane="A1", isolated=False,
    )
    assert r.short_horizon_eff > 0.0 and r.long_horizon_eff > 0.0 and r.dual_horizon_eff > 0.0
    assert r.long_horizon_eff >= r.short_horizon_eff
    assert abs(r.dual_horizon_eff - (r.short_horizon_eff * r.long_horizon_eff) ** 0.5) < 1e-12


def test_evaluator_json_contract_and_research_brief_stall_detection(tmp_path):
    lane_dir = tmp_path / "lanes" / "A1"
    run_dir = lane_dir / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (lane_dir / "NEGATIVE_MEMORY.md").write_text(
        "- Do not shrink step size below 1e-4\n" + "".join(f"- refuted idea {i}\n" for i in range(40)))
    (run_dir / "RESEARCH_HANDOFF.md").write_text("Pinned commit abc123; funnel still bottleneck.\n")

    rows = [
        {
            "trial_id": "T0001", "lane": "A1", "tier0_ok": True, "tier1_ok": True,
            "decision": "keep_baseline", "speedup": 1.0, "yardstick": "A1-x", "soft_min_eff": 0.42,
            "short_horizon_soft_min": 0.30, "long_horizon_soft_min": 0.55, "median_adapt_cost": 1200.0,
            "overhead_ratio": 1.5, "b2_max": 0.002, "loc": 80, "magic_constants": 5, "wall_s": 12.3,
            "by_class": {"Funnel": 0.11, "Gaussian": 0.90},
        },
        *[
            {"trial_id": f"T{i:04d}", "lane": "A1", "tier0_ok": True, "tier1_ok": True,
             "decision": "discard_no_gain", "speedup": 0.95, "soft_min_eff": 0.40, "hypothesis": f"tweak {i}"}
            for i in range(2, 7)
        ],
    ]
    (run_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run_dir / "results.tsv").write_text("trial_id\nT0001\n")

    ev = autoresearch.format_evaluator_json(rows[0], run_id="r1", autoresearch_dir=tmp_path)
    assert ev["valid"] is True and ev["score"] == 1.0 and ev["kept"] is True and ev["yardstick"] == "A1-x"
    assert list(ev["metrics"])[0] == "speedup"
    assert ev["metrics"]["class_Funnel"] == 0.11 and ev["metrics"]["soft_min_eff"] == 0.42
    assert len(ev["evidence"]) >= 2

    ev_fail = autoresearch.format_evaluator_json(
        {"trial_id": "T0009", "lane": "A1", "tier0_ok": False, "tier1_ok": False,
         "decision": "discard_tier0", "speedup": None, "soft_min_eff": float("nan")},
        run_id="r1", autoresearch_dir=tmp_path,
    )
    assert ev_fail["valid"] is False and ev_fail["score"] == 0.0 and ev_fail["kept"] is False
    assert "soft_min_eff" not in ev_fail["metrics"] and "speedup" not in ev_fail["metrics"]

    brief = autoresearch.build_research_brief("A1", run_id="r1", autoresearch_dir=tmp_path, offline=True)
    assert "STALL DETECTED (5 consecutive non-kept trials)" in brief
    assert "speedup=1.0x" in brief and "Funnel=0.11" in brief
    assert "Do not shrink step size below 1e-4" in brief and "refuted idea 39" in brief
    assert "Pinned commit abc123" in brief
    assert "--repeats" not in brief


def test_submission_is_never_imported_in_the_judging_process(tmp_path):
    """Import-time code in a submission must not reach the process that computes the verdict."""
    import numpy as np

    sub = tmp_path / "submission.py"
    sub.write_text(
        "import numpy as np\n"
        "np.APIN_TAMPERED = True\n"
        "def adapt(oracle, budget, rng):\n    return {}, (np.zeros(oracle.dim), None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n    return chain.position, None\n"
    )
    grade.load_check(sub, None)  # loads fine, in a child
    assert not hasattr(np, "APIN_TAMPERED")

    broken = tmp_path / "broken.py"
    broken.write_text("def adapt(oracle, budget, rng):\n    return None\n")
    with pytest.raises(RuntimeError, match="must define both adapt"):
        grade.load_check(broken, None)


def test_research_brief_shows_live_frontier_pickup_and_systems_checklists(tmp_path, monkeypatch):
    from harness import flywheel

    lane_dir = tmp_path / "lanes" / "A1"
    lane_dir.mkdir(parents=True)
    (lane_dir / "submission.py").write_text("# local start\n")
    (lane_dir / "yardstick.json").write_text(json.dumps({"id": "A1-yd1"}))

    monkeypatch.setattr(flywheel, "get_frontier", lambda **kw: {
        "frontier": [{
            "id": "A1:r0:T0004", "lane": "A1", "speedup": 2.35, "yardstick": "A1-yd1",
            "author": "alice", "code_sha": "deadbeef12345678", "verified": 0,
            "hypothesis": "shrunk diagonal metric with dual averaging",
            "metrics_json": json.dumps({"by_class": {"funnel": 0.01, "banana": 0.05, "gaussian": 0.2}}),
        }]
    })
    monkeypatch.setattr(flywheel, "get_nodes", lambda **kw: {
        "nodes": [
            {"id": "A1:r0:T0004", "decision": "keep", "speedup": 2.35, "yardstick": "A1-yd1",
             "author": "alice", "has_source": True, "hypothesis": "shrunk diagonal metric"},
            {"id": "A1:r0:T0003", "decision": "discard_no_gain", "speedup": 0.9, "yardstick": "A1-yd1",
             "author": "alice", "hypothesis": "tiny step size"},
        ]
    })
    monkeypatch.setattr(flywheel, "get_claims", lambda **kw: {"claims": []})
    monkeypatch.setattr(flywheel, "list_notes", lambda **kw: {"notes": []})

    brief_a1 = autoresearch.build_research_brief("A1", autoresearch_dir=tmp_path, flywheel_url="http://mock")
    assert "Starting Point on A1" in brief_a1
    assert "fetch --node A1:r0:T0004 --out autoresearch/lanes/A1/submission.py --force" in brief_a1
    assert "--fork-of A1:r0:T0004" in brief_a1
    assert "funnel=0.01" in brief_a1
    assert "Top Recent Kept Trials on A1" in brief_a1
    assert "shrunk diagonal metric" in brief_a1

    brief_s2 = autoresearch.build_research_brief("S2", autoresearch_dir=tmp_path, offline=True)
    assert "eval-s2" in brief_s2 and "apin autoresearch eval --lane S2" not in brief_s2
    brief_s3 = autoresearch.build_research_brief("S3", autoresearch_dir=tmp_path, offline=True)
    assert "eval-s3" in brief_s3 and "apin autoresearch eval --lane S3" not in brief_s3
