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
    g = score.declared("eight_schools_noncentered,blr", reps=1, build_dir=tmp_path / "build")
    out = score.score(tmp_path / "runs", tmp_path / "scores", g, log=lambda *_: None)
    for row in out["per_run"]:
        assert row["status"] != "wrong", row
        assert row["evaluations"] > 4000 and row["gradient_evals"] == row["evaluations"]
