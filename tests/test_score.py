"""Aggregation from per-rep rows to models, labels and baseline ratios."""

import json
import math
import shutil
from pathlib import Path

import pytest

from harness import score

FIX = Path(__file__).parent / "fixtures"


def _row(label, model, rep, ess_min, wall, chain_sum, grads, rhat=1.0):
    return {"label": label, "model": model, "rep": rep, "wall_s": wall, "chain_s_sum": chain_sum,
            "grads_sampling": grads, "divergences": 0, "ess_bulk_min": ess_min, "ess_bulk_geomean": ess_min,
            "ess_tail_min": ess_min, "rhat_max": rhat, "ess_per_s_wall": ess_min / wall,
            "ess_per_s_chain": ess_min / chain_sum, "ess_per_grad": ess_min / grads}


def test_median_over_reps_then_geomean_over_models():
    rows = [_row("base", "m1", r, e, 1.0, 4.0, 1000) for r, e in enumerate((100, 300, 200))]
    rows += [_row("base", "m2", 0, 400, 2.0, 8.0, 1000, rhat=1.05)]
    models = score.per_model(rows)
    m1 = next(m for m in models if m["model"] == "m1")
    assert m1["ess_bulk_min"] == 200 and m1["reps"] == 3
    (s,) = score.summary(models)
    assert s["geo_ess_per_s_wall"] == pytest.approx(math.sqrt(200 * 200))
    assert s["n_models_rhat_bad"] == 1


def test_ratios_against_baseline():
    rows = [_row("base", "m1", 0, 100, 1.0, 4.0, 1000), _row("patched", "m1", 0, 150, 0.5, 2.0, 1000)]
    (r,) = score.ratios(score.per_model(rows), "base")
    assert r["ratio_ess_per_s_wall"] == pytest.approx(3.0)
    assert r["ratio_ess_per_grad"] == pytest.approx(1.5)
    with pytest.raises(KeyError):
        score.ratios(score.per_model(rows), "missing")


def test_score_end_to_end_on_fixture(tmp_path):
    rep = tmp_path / "runs" / "demo" / "tiny" / "rep0"
    shutil.copytree(FIX / "tiny", rep)
    timing = {"wall_s": 2.0, "chain_s": [1.0] * 4, "warmup_s": [0.5] * 4, "sampling_s": [0.5] * 4,
              "grads_sampling": 4 * 101 * 7, "grads_warmup": None, "divergences": 0, "extra": {}}
    (rep / "run.json").write_text(json.dumps({"label": "demo", "sampler": "fixture", "model": "tiny", "rep": 0,
                                              "timing": timing}))
    (rep / "DONE").touch()
    out = score.score(tmp_path / "runs", tmp_path / "scores", log=lambda *_: None)
    (row,) = out["per_run"]
    assert row["ess_per_s_wall"] == pytest.approx(row["ess_bulk_min"] / 2.0)
    assert row["ess_per_s_chain"] == pytest.approx(row["ess_bulk_min"] / 4.0)
    assert row["ess_per_grad"] == pytest.approx(row["ess_bulk_min"] / (4 * 101 * 7))
    assert (tmp_path / "scores" / "summary.csv").is_file()


def _gated(label, model, rep, ess_per_eval, gate="pass", detail="pass"):
    r = _row(label, model, rep, 100, 1.0, 4.0, 1000)
    r.update(gate=gate, gate_detail=detail, evaluations=1000, ess_per_eval=ess_per_eval)
    if gate != "pass":
        for k in score.COST:
            r[k] = math.nan
    return r


def test_wrong_model_has_no_speed_and_shows_its_worst_variable():
    rows = [_gated("x", "m1", 0, math.nan, "wrong", "wrong: tau x2 b2=0.3"),
            _gated("x", "m1", 1, math.nan, "failed", "failed: boom"),
            _gated("x", "m1", 2, 0.05)]
    (m,) = score.per_model(rows)
    assert m["gate"] == "wrong" and m["reps_wrong"] == 2
    assert all(math.isnan(m[k]) for k in score.COST)
    assert "wrong: tau x2 b2=0.3" in score.table([m], ["x"])
    (s,) = score.summary([m])
    assert s["n_models_wrong"] == 1


def test_one_bad_rep_of_three_is_not_a_wrong_model():
    rows = [_gated("x", "m1", 0, math.nan, "wrong", "wrong: a x b2=1")] + [_gated("x", "m1", r, 0.05) for r in (1, 2)]
    (m,) = score.per_model(rows)
    assert m["gate"] == "pass" and m["reps_wrong"] == 1 and m["ess_per_eval"] == pytest.approx(0.05)


def test_bench_score_penalty_and_baseline_exclusion():
    rows = [_gated("base", m, 0, 0.01) for m in ("m1", "m2")]
    rows += [_gated("base", "m3", 0, math.nan, "wrong", "wrong: z x b2=1")]
    rows += [_gated("cand", "m1", r, v) for r, v in enumerate((0.02, 0.04, 0.03))]  # median ratio 3
    rows += [_gated("cand", "m2", r, math.nan, "wrong", "wrong") for r in range(3)]  # PENALTY
    rows += [_gated("cand", "m3", 0, 0.5)]  # baseline wrong here: left out
    b = score.bench_score(rows, score.per_model(rows), "cand", "base")
    assert b["per_model"] == {"m1": pytest.approx(3.0), "m2": score.PENALTY}
    assert b["score"] == pytest.approx(math.sqrt(3.0 * score.PENALTY))
    assert b["excluded"] == "m3"


def test_failed_rep_becomes_a_row(tmp_path):
    rep = tmp_path / "lab" / "m1" / "rep0"
    rep.mkdir(parents=True)
    (rep / "FAILED").write_text("Traceback...\nRuntimeError: submission killed after 2100 s\n")
    r = score.failed_row(rep)
    assert (r["label"], r["model"], r["rep"], r["gate"]) == ("lab", "m1", 0, "failed")
    assert "killed" in r["gate_detail"] and math.isnan(r["ess_per_eval"])


def test_bench_interval_brackets_the_score():
    rows = [_gated("base", m, 0, 0.01) for m in ("m1", "m2")]
    rows += [_gated("cand", "m1", r, v) for r, v in enumerate((0.01, 0.02, 0.04))]
    rows += [_gated("cand", "m2", r, v) for r, v in enumerate((0.01, 0.01, 0.01))]
    b = score.bench_score(rows, score.per_model(rows), "cand", "base")
    assert b["score_lo"] <= b["score"] <= b["score_hi"]
    assert b["score_lo"] >= 1.0 - 1e-12 and b["score_hi"] <= 2.0 + 1e-12


def test_bench_score_leaves_out_models_without_a_reference():
    rows = [_gated("base", m, 0, 0.01) for m in ("m1", "m2")]
    rows += [_gated("cand", "m1", 0, 0.02), _gated("cand", "m2", 0, 0.5)]
    for r in rows:
        if r["model"] == "m2":
            r["gate"] = "no reference"
    b = score.bench_score(rows, score.per_model(rows), "cand", "base")
    assert b["per_model"] == {"m1": pytest.approx(2.0)} and b["excluded"] == "m2"
