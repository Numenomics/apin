"""End to end with the real CmdStan: build, sample and score two small models.

Run with `CMDSTAN=... uv run pytest -m slow`. Takes about a minute on first
build and seconds after.
"""

import os

import pytest

from harness import grid, models, sampler, score


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("CMDSTAN"), reason="CMDSTAN not set")
def test_smoke_grid(tmp_path):
    s = sampler.get("cmdstan", sampler.Settings(build_dir=tmp_path / "build"))
    ms = models.select("eight_schools_noncentered,blr")
    done = grid.run_grid(s, ms, reps=1, runs_dir=tmp_path / "runs", log=lambda *_: None)
    assert len(done) == 2
    out = score.score(tmp_path / "runs", tmp_path / "scores", log=lambda *_: None)
    for row in out["per_run"]:
        assert row["ess_bulk_min"] > 400, row
        assert row["rhat_max"] < 1.01, row
        assert row["grads_sampling"] > 4000
        assert row["ess_per_s_wall"] > 0
