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
