"""The scorer: a declared grid, three verdicts per model, the rank (BENCH.md, rules 1, 2 and 4)."""

import json
import math
import zlib

import numpy as np
import pytest

from harness import gate, reference, score
from harness.adapters.nutpie import write_csv

MU = np.array([0.0, 3.0, -1.0])
SIGMA = np.array([1.0, 0.5, 2.0])
NAMES = ["a", "b.1", "b.2"]


def lp(draws: dict) -> np.ndarray:
    return -0.5 * sum(((draws[n] - m) / s) ** 2 for n, m, s in zip(NAMES, MU, SIGMA))


@pytest.fixture(scope="module")
def grid():
    rng = np.random.default_rng(5)
    x = MU + SIGMA * rng.standard_normal((1000, 10, 3))
    d = {n: x[..., i] for i, n in enumerate(NAMES)}
    ref = reference.summarise(d, lp=lp(d), model="sim")
    entries = {m: score.Entry(m, ref, chains=4, warmup=10, draws=4000, lp=lp) for m in ("m1", "m2")}
    return score.Grid(entries, reps=2, base_seed=100)


def write_rep(runs, label, model, rep, grid, *, shift=0.0, evaluations=10**5, seconds=2.0, identity=None,
              seed=None, draws=None, seed_rng=None):
    d = runs / label / model / f"rep{rep}"
    d.mkdir(parents=True)
    e = grid.entries[model]
    n = draws or e.draws
    rng = np.random.default_rng(seed_rng if seed_rng is not None else zlib.crc32(f"{label}/{model}/{rep}".encode()))
    x = MU + SIGMA * rng.standard_normal((e.chains, n, 3))
    x[..., 0] += shift
    for c in range(e.chains):
        write_csv(d / f"chain_{c}.csv", NAMES, x[c])
    timing = {"wall_s": seconds, "chain_s": [seconds / e.chains] * e.chains, "warmup_s": [None] * e.chains,
              "sampling_s": [None] * e.chains, "grads_sampling": None, "grads_warmup": None, "divergences": None,
              "evaluations": evaluations,
              "extra": {} if evaluations is None else {"density_evals": evaluations // 4,
                                                       "gradient_evals": evaluations - evaluations // 4}}
    rec = {"label": label, "sampler": "fake", "model": model, "rep": rep,
           "seed": grid.base_seed + 1000 * rep if seed is None else seed, "base_seed": grid.base_seed,
           "chains": e.chains, "warmup": e.warmup, "draws": n, "identity": identity or {"name": label},
           "timing": timing}
    (d / "run.json").write_text(json.dumps(rec))
    (d / "DONE").touch()
    return d


def test_omitting_a_model_ranks_below_attempting_it(tmp_path, grid):
    """S7: the old score was 3 for a candidate three times as fast on m1 with no rows on m2."""
    for r in range(2):
        for m in ("m1", "m2"):
            write_rep(tmp_path, "base", m, r, grid)
        write_rep(tmp_path, "cand", "m1", r, grid, evaluations=10**5 // 3)
    out = score.score(tmp_path, tmp_path / "scores", grid, baseline="base", log=lambda *_: None)
    ranking = {r["label"]: r for r in out["ranking"]}
    assert ranking["base"]["right"] == 2 and ranking["cand"]["right"] == 1
    assert [r["label"] for r in out["ranking"]] == ["base", "cand"]
    assert ranking["cand"]["speed_eval"] == pytest.approx(3.0, rel=0.05)
    m2 = next(m for m in out["per_model"] if m["label"] == "cand" and m["model"] == "m2")
    assert m2["status"] == "wrong" and "missing" in m2["reason"]


def test_a_missing_rep_makes_the_model_wrong(tmp_path, grid):
    write_rep(tmp_path, "x", "m1", 0, grid)
    out = score.score(tmp_path, tmp_path / "s", grid, log=lambda *_: None)
    m1 = next(m for m in out["per_model"] if m["model"] == "m1")
    assert m1["status"] == "wrong" and "rep1" in m1["reason"]


def test_a_rep_with_the_wrong_seed_or_budget_is_wrong(tmp_path, grid):
    """Rule 4: a saved rep counts only for the seed and budget the grid declares."""
    write_rep(tmp_path, "x", "m1", 0, grid)
    write_rep(tmp_path, "x", "m1", 1, grid, seed=7)
    write_rep(tmp_path, "x", "m2", 0, grid)
    write_rep(tmp_path, "x", "m2", 1, grid, draws=2000)
    out = score.score(tmp_path, tmp_path / "s", grid, log=lambda *_: None)
    reasons = {(r["model"], r["rep"]): r["reason"] for r in out["per_run"]}
    assert "seed" in reasons[("m1", 1)] and "budget" in reasons[("m2", 1)]
    assert all(m["status"] == "wrong" for m in out["per_model"])


def test_a_label_mixing_submissions_is_refused(tmp_path, grid):
    write_rep(tmp_path, "x", "m1", 0, grid, identity={"submission_sha256": "aa"})
    write_rep(tmp_path, "x", "m1", 1, grid, identity={"submission_sha256": "bb"})
    with pytest.raises(ValueError, match="identit"):
        score.score(tmp_path, tmp_path / "s", grid, log=lambda *_: None)


def test_a_biased_sampler_is_wrong_and_ranks_last(tmp_path, grid):
    for r in range(2):
        for m in ("m1", "m2"):
            write_rep(tmp_path, "base", m, r, grid)
            write_rep(tmp_path, "fast", m, r, grid, shift=0.5, evaluations=10**3)
    out = score.score(tmp_path, tmp_path / "s", grid, baseline="base", log=lambda *_: None)
    fast = next(r for r in out["ranking"] if r["label"] == "fast")
    assert fast["right"] == 0 and math.isnan(fast["speed_eval"]) and out["ranking"][-1]["label"] == "fast"
    assert "wrong" in out["table"]


def test_costs_per_evaluation_and_per_second_and_both_counts(tmp_path, grid):
    for r in range(2):
        for m in ("m1", "m2"):
            write_rep(tmp_path, "base", m, r, grid, evaluations=10**5, seconds=1.0)
            write_rep(tmp_path, "cand", m, r, grid, evaluations=5 * 10**4, seconds=4.0)
    out = score.score(tmp_path, tmp_path / "s", grid, baseline="base", log=lambda *_: None)
    row = next(r for r in out["per_run"] if r["label"] == "cand")
    assert row["density_evals"] + row["gradient_evals"] == row["evaluations"] == 5 * 10**4
    assert row["ess_per_eval"] == pytest.approx(row["ess"] / 5e4)
    assert row["ess_per_s"] == pytest.approx(row["ess"] / 4.0)
    cand = next(r for r in out["ranking"] if r["label"] == "cand")
    assert cand["speed_eval"] == pytest.approx(2.0, rel=0.1) and cand["speed_s"] == pytest.approx(0.25, rel=0.1)
    assert cand["speed_eval_lo"] <= cand["speed_eval"] <= cand["speed_eval_hi"]
    assert (tmp_path / "s" / "ranking.csv").is_file() and (tmp_path / "s" / "table.md").is_file()


def test_an_unavailable_cost_earns_no_speed(tmp_path, grid):
    """S12: a rep that cannot count its evaluations is not credited per evaluation."""
    for r in range(2):
        for m in ("m1", "m2"):
            write_rep(tmp_path, "base", m, r, grid)
            write_rep(tmp_path, "cand", m, r, grid, evaluations=None if m == "m2" else 10**4)
    out = score.score(tmp_path, tmp_path / "s", grid, baseline="base", log=lambda *_: None)
    cand = next(r for r in out["ranking"] if r["label"] == "cand")
    assert cand["right"] == 2 and math.isnan(cand["speed_eval"]) and "m2" in cand["note"]


def test_the_interval_resamples_the_baseline():
    """S15: with the candidate's reps all equal, the interval still shows the baseline's spread."""
    rows = [{"label": "base", "model": "m", "status": "right", "eval": [1.0, 2.0, 4.0], "sec": [1.0] * 3},
            {"label": "cand", "model": "m", "status": "right", "eval": [2.0] * 3, "sec": [1.0] * 3}]
    (b, c) = score.rank(rows, "base", names=["m"])
    assert c["speed_eval_lo"] < c["speed_eval"] < c["speed_eval_hi"]


def test_rank_orders_by_models_right_then_speed():
    def rows(label, status, v):
        return [{"label": label, "model": m, "status": s, "eval": [x] * 2, "sec": [x] * 2}
                for m, s, x in zip(("m1", "m2", "m3"), status, v)]

    allrows = (rows("base", ["right", "right", "unproven"], [1, 1, 1])
               + rows("slow3", ["right", "right", "right"], [0.5, 0.5, 0.5])
               + rows("fast2", ["right", "right", "wrong"], [4, 4, 4]))
    out = score.rank(allrows, "base", names=["m1", "m2", "m3"])
    assert [r["label"] for r in out] == ["slow3", "fast2", "base"]
    assert out[0]["speed_eval"] == pytest.approx(0.5) and out[1]["speed_eval"] == pytest.approx(4)


def test_verdicts_are_cached_by_reference_and_reps(tmp_path, grid):
    d = write_rep(tmp_path, "x", "m1", 0, grid)
    write_rep(tmp_path, "x", "m1", 1, grid)
    score.score(tmp_path, tmp_path / "s", grid, log=lambda *_: None)
    cached = json.loads((d / "gate.json").read_text())
    assert cached["key"]["reps"] == 2 and cached["verdict"]["status"] in gate.STATUSES


def _reasons(tmp_path, grid):
    out = score.score(tmp_path, tmp_path / "s", grid, log=lambda *_: None)
    return {(r["model"], r["rep"]): (r["status"], r["reason"]) for r in out["per_run"]}


def test_a_rep_short_of_its_chains_or_draws_is_wrong(tmp_path, grid):
    """Rule 4: the files must hold the budget run.json declares. A lost chain or a chain cut short
    (a crash, a full disk) is missing draws, so the rep is wrong rather than gated on what is left."""
    write_rep(tmp_path, "x", "m1", 0, grid)
    (write_rep(tmp_path, "x", "m1", 1, grid) / "chain_3.csv").unlink()
    d = write_rep(tmp_path, "x", "m2", 0, grid)
    lines = (d / "chain_1.csv").read_text().splitlines()
    (d / "chain_1.csv").write_text("\n".join(lines[:-100]) + "\n")
    reasons = _reasons(tmp_path, grid)
    assert reasons[("m1", 0)][0] != "wrong"
    assert reasons[("m1", 1)][0] == "wrong" and "3 chains" in reasons[("m1", 1)][1]
    assert reasons[("m2", 0)][0] == "wrong" and "draws" in reasons[("m2", 0)][1]


def test_a_truncated_chain_file_is_wrong_not_a_crash(tmp_path, grid):
    d = write_rep(tmp_path, "x", "m1", 0, grid)
    text = (d / "chain_0.csv").read_text()
    (d / "chain_0.csv").write_text(text[: len(text) - 7])  # the last row loses its last fields
    status, reason = _reasons(tmp_path, grid)[("m1", 0)]
    assert status == "wrong" and "chain_0.csv" in reason


def test_a_corrupt_verdict_cache_is_recomputed(tmp_path, grid):
    d = write_rep(tmp_path, "x", "m1", 0, grid)
    write_rep(tmp_path, "x", "m1", 1, grid)
    first = _reasons(tmp_path, grid)[("m1", 0)]
    (d / "gate.json").write_text('{"key": {"reps": 2, "refer')
    assert _reasons(tmp_path, grid)[("m1", 0)] == first
    assert json.loads((d / "gate.json").read_text())["verdict"]["status"] == first[0]
