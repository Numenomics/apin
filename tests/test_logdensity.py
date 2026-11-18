"""The log-density seam: counting, budget, rejections and the import lint, on a fake model.

The end-to-end tests at the bottom build two models with BridgeStan (about a
minute and a half each on first build) and run the example samplers; they are
marked slow.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from harness import grid, logdensity, models, sampler, score
from harness.logdensity import Budget, BudgetExceeded, Counter, check_imports

EXAMPLES = Path(__file__).parent.parent / "examples" / "samplers"


class FakeModel:
    """A standard normal in 3 dimensions that rejects points with x[0] > 5, as Stan would."""

    def param_unc_num(self):
        return 3

    def log_density(self, x, propto, jacobian):
        if x[0] > 5:
            raise RuntimeError("normal_lpdf: Scale parameter is 0, but must be positive!")
        return -0.5 * float(x @ x)

    def log_density_gradient(self, x, propto, jacobian):
        return self.log_density(x, propto, jacobian), -x


def test_counter_counts_both_kinds_against_one_budget():
    c = Counter(FakeModel(), Budget(chains=1, draws=1, max_evaluations=5, max_seconds=60))
    for _ in range(3):
        c.log_density(np.zeros(3))
    c.log_density_gradient(np.ones(3))
    lp, g = c.log_density_gradient(np.ones(3))
    assert (c.density, c.gradient) == (3, 2)
    assert lp == -1.5 and np.array_equal(g, -np.ones(3))
    with pytest.raises(BudgetExceeded):
        c.log_density(np.zeros(3))
    assert c.exceeded and (c.density, c.gradient) == (3, 2)


def test_budget_stays_exceeded_after_the_error_is_caught():
    c = Counter(FakeModel(), Budget(1, 1, max_evaluations=1, max_seconds=60))
    c.log_density(np.zeros(3))
    for _ in range(2):
        with pytest.raises(BudgetExceeded):
            c.log_density_gradient(np.zeros(3))


def test_time_limit():
    c = Counter(FakeModel(), Budget(1, 1, max_evaluations=100, max_seconds=0.0))
    c.t0 -= 1.0
    with pytest.raises(BudgetExceeded, match="s$"):
        c.log_density(np.zeros(3))


def test_rejected_point_is_minus_infinity():
    c = Counter(FakeModel(), Budget(1, 1, 10, 60))
    assert c.log_density(np.array([6.0, 0, 0])) == -math.inf
    lp, g = c.log_density_gradient(np.array([6.0, 0, 0]))
    assert lp == -math.inf and g.shape == (3,) and np.all(np.isnan(g))
    assert c.density + c.gradient == 2  # rejected points still count


def test_wrong_shape_is_an_error():
    c = Counter(FakeModel(), Budget(1, 1, 10, 60))
    with pytest.raises(ValueError, match="shape"):
        c.log_density(np.zeros(4))


@pytest.mark.parametrize("src", ["import os", "from subprocess import run", "import numpy, torch",
                                 "from . import helper", "import importlib"])
def test_import_lint_rejects(src):
    with pytest.raises(ImportError):
        check_imports(src)


def test_import_lint_allows_numpy_scipy_and_stdlib_helpers():
    check_imports("from __future__ import annotations\nimport math\nimport numpy as np\n"
                  "from scipy.special import logsumexp\nfrom dataclasses import dataclass\n")


def test_examples_pass_the_lint():
    for f in EXAMPLES.glob("*.py"):
        check_imports(f.read_text())


def test_load_submission_needs_sample(tmp_path):
    f = tmp_path / "s.py"
    f.write_text("import numpy as np\n")
    with pytest.raises(AttributeError, match="sample"):
        logdensity.load_submission(f)


def test_example_hmc_on_fake_model():
    """The example HMC, run on the fake target through the Counter, recovers a standard normal."""
    budget = Budget(chains=2, draws=500, max_evaluations=10**6, max_seconds=60)
    c = Counter(FakeModel(), budget)
    rng = np.random.default_rng(1)
    problem = logdensity.Problem(3, c.log_density, c.log_density_gradient, logdensity.initial_points(rng, 2, 3))
    draws = logdensity.load_submission(EXAMPLES / "hmc.py")(problem, budget, rng)
    assert draws.shape == (2, 500, 3)
    flat = draws.reshape(-1, 3)
    assert np.all(np.abs(flat.mean(axis=0)) < 0.2) and np.all(np.abs(flat.var(axis=0) - 1) < 0.25)
    assert 0 < c.gradient < budget.max_evaluations


@pytest.mark.slow
def test_examples_end_to_end(tmp_path):
    """The example HMC passes the gate on the two smallest models; ULA with a large step fails."""
    ms = models.select("eight_schools_noncentered,blr")
    for label, file in (("hmc", "hmc.py"), ("ula", "ula_wrong.py")):
        s = sampler.get("logdensity", sampler.Settings(build_dir=tmp_path / "build",
                                                       options={"sampler": str(EXAMPLES / file)}))
        done = grid.run_grid(s, ms, reps=1, runs_dir=tmp_path / "runs", label=label, log=lambda *_: None)
        assert len(done) == 2
    out = score.score(tmp_path / "runs", tmp_path / "scores", log=lambda *_: None)
    gates = {(r["label"], r["model"]): r for r in out["per_run"]}
    for m in ("eight_schools_noncentered", "blr"):
        hmc, ula = gates["hmc", m], gates["ula", m]
        assert hmc["gate"] == "pass", hmc["gate_detail"]
        assert hmc["evaluations"] > 0 and hmc["ess_per_eval"] > 0 and hmc["ess_per_s_wall"] > 0
        assert ula["gate"] == "wrong", ula["gate_detail"]
        assert math.isnan(ula["ess_per_eval"])


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("CMDSTAN"), reason="CMDSTAN not set")
def test_cmdstan_nuts_passes_the_gate(tmp_path):
    s = sampler.get("cmdstan", sampler.Settings(build_dir=tmp_path / "build"))
    ms = models.select("eight_schools_noncentered,blr,kidscore_momiq")
    grid.run_grid(s, ms, reps=1, runs_dir=tmp_path / "runs", log=lambda *_: None)
    out = score.score(tmp_path / "runs", tmp_path / "scores", log=lambda *_: None)
    assert len(out["per_run"]) == 3
    for row in out["per_run"]:
        assert row["gate"] == "pass", (row["model"], row["gate_detail"])
        assert row["evaluations"] > row["grads_sampling"]  # warmup is counted


def test_provenance_marks_only_evaluated_points():
    c = Counter(FakeModel(), Budget(1, 2, 10, 60))
    a, b = np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0])
    c.log_density(a)
    c.log_density_gradient(list(b))  # a list is converted to the same float64 point
    draws = np.array([[a, b], [b, a + 1e-12]])
    assert c.first_call(draws).tolist() == [[0, 1], [1, -1]]


@pytest.mark.parametrize("first, match", [([[0, 1], [2, -1]], "never evaluated"),
                                          ([[1, 0], [2, 3]], "order"),
                                          ([[0, 1], [1, 2]], "share a draw")])
def test_provenance_rejects(first, match):
    with pytest.raises(ValueError, match=match):
        logdensity.check_provenance(np.array(first))


def test_provenance_allows_repeats_within_a_chain():
    logdensity.check_provenance(np.array([[0, 0, 3, 3, 5], [1, 2, 2, 4, 6]]))


@pytest.mark.parametrize("src", ["data = open('ref.json').read()", "eval('1')", "__import__('os')"])
def test_lint_rejects_file_and_code_builtins(src):
    with pytest.raises(ImportError, match="not allowed"):
        check_imports(src)


def _stay_put(problem, budget, rng):
    """Evaluates each initial point once and returns it as every draw."""
    for x in problem.initial_points:
        problem.log_density(x)
    return np.repeat(problem.initial_points[:, None, :], budget.draws, axis=1)


def test_execute_accepts_evaluated_draws():
    draws, c, _ = logdensity.execute(FakeModel(), _stay_put, Budget(2, 4, 10, 60), seed=1)
    assert draws.shape == (2, 4, 3) and c.density == 2


def _never_evaluated(problem, budget, rng):
    return np.zeros((budget.chains, budget.draws, problem.dim))


def _short(problem, budget, rng):
    return _stay_put(problem, budget, rng)[:, :-1]


def _over_budget_but_caught(problem, budget, rng):
    try:
        for _ in range(budget.max_evaluations + 1):
            problem.log_density(problem.initial_points[0])
    except BudgetExceeded:
        pass
    return np.repeat(problem.initial_points[:, None, :], budget.draws, axis=1)


@pytest.mark.parametrize("fn, error, match", [(_never_evaluated, ValueError, "never evaluated"),
                                              (_short, ValueError, "shape"),
                                              (_over_budget_but_caught, BudgetExceeded, "caught")])
def test_execute_rejects(fn, error, match):
    with pytest.raises(error, match=match):
        logdensity.execute(FakeModel(), fn, Budget(2, 4, 5, 60), seed=1)
