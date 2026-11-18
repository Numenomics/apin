"""The grid runner resumes only a rep whose seed, sampler identity and budget match (S13, rule 4)."""

import json

from harness import grid, models
from harness.sampler import Sampler, Settings, Timing


class Fake(Sampler):
    name = "fake"

    def __init__(self, ident="a"):
        super().__init__(Settings())
        self.ident, self.calls = ident, []

    def describe(self):
        return {"submission_sha256": self.ident}

    def run(self, req):
        self.calls.append((req.seed, req.draws))
        for c in range(req.chains):
            (req.out_dir / f"chain_{c}.csv").write_text("x\n0.0\n")
        return Timing(1.0, [1.0] * req.chains, [None] * req.chains, [None] * req.chains, None, None, None,
                      evaluations=10)


def _run(tmp_path, s, **kw):
    ms = models.select("blr")
    return grid.run_grid(s, ms, reps=2, runs_dir=tmp_path, label="x", log=lambda *_: None, **kw)


def test_a_matching_rep_is_reused(tmp_path):
    _run(tmp_path, Fake())
    again = Fake()
    _run(tmp_path, again)
    assert again.calls == []


def test_a_changed_base_seed_reruns(tmp_path):
    _run(tmp_path, Fake(), base_seed=1)
    again = Fake()
    _run(tmp_path, again, base_seed=2)
    assert [s for s, _ in again.calls] == [2, 1002]
    rec = json.loads((tmp_path / "x" / "blr" / "rep1" / "run.json").read_text())
    assert rec["seed"] == 1002 and rec["base_seed"] == 2


def test_a_changed_submission_reruns(tmp_path):
    _run(tmp_path, Fake("a"))
    again = Fake("b")
    _run(tmp_path, again)
    assert len(again.calls) == 2


def test_a_changed_budget_reruns(tmp_path):
    _run(tmp_path, Fake(), draws=10)
    again = Fake()
    _run(tmp_path, again, draws=20)
    assert [d for _, d in again.calls] == [20, 20]


def test_the_default_budget_is_the_models(tmp_path):
    s = Fake()
    _run(tmp_path, s)
    assert {d for _, d in s.calls} == {models.load()["blr"].draws}
