"""The ESS and R-hat port against R's posterior 1.7.0 on a fixed fixture."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from harness import ess, score

FIX = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIX / "tiny_expected.json").read_text())
DRAWS = score.chain_draws(FIX / "tiny")


@pytest.mark.parametrize("i,var", list(enumerate(EXPECTED["variable"])))
def test_matches_r_posterior(i, var):
    x = DRAWS[var]
    assert x.shape == (101, 4)  # odd length exercises the dropped middle draw in split chains
    assert ess.ess_bulk(x) == pytest.approx(EXPECTED["ess_bulk"][i], rel=1e-9)
    assert ess.ess_tail(x) == pytest.approx(EXPECTED["ess_tail"][i], rel=1e-9)
    assert ess.rhat(x) == pytest.approx(EXPECTED["rhat"][i], rel=1e-9)


def test_diagnostic_columns_are_not_scored():
    assert set(DRAWS) == {"a", "b.1", "b.2", "c"}


def test_summary_minima():
    s = ess.summarise(DRAWS)
    assert s["ess_bulk_min"] == pytest.approx(min(EXPECTED["ess_bulk"]), rel=1e-9)
    assert s["ess_tail_min"] == pytest.approx(min(EXPECTED["ess_tail"]), rel=1e-9)
    assert s["rhat_max"] == pytest.approx(max(EXPECTED["rhat"]), rel=1e-9)
    assert s["worst_variables"][0] == "a"
    assert s["n_variables"] == 4


def test_constant_variable_is_left_out():
    draws = dict(DRAWS, const=np.ones((101, 4)))
    assert math.isnan(ess.ess_bulk(draws["const"]))
    assert ess.summarise(draws)["n_variables"] == 4


def test_nonfinite_gives_nan():
    x = DRAWS["a"].copy()
    x[3, 1] = np.inf
    assert math.isnan(ess.ess_bulk(x)) and math.isnan(ess.rhat(x))


def test_nextn():
    assert [ess._nextn(n) for n in (1, 7, 11, 101, 202, 1000)] == [1, 8, 12, 108, 216, 1000]


def test_ragged_chains_are_trimmed(tmp_path):
    for c in range(2):
        rows = 10 + 3 * c
        lines = ["x,lp__"] + [f"{0.1 * i + c},0" for i in range(rows)]
        (tmp_path / f"chain_{c}.csv").write_text("\n".join(lines) + "\n")
    d = score.chain_draws(tmp_path)
    assert d["x"].shape == (10, 2)
