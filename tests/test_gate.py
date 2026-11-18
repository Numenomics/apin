"""The correctness gate: its false-fail rate on correct samplers and its power on broken ones."""

import math

import numpy as np
import pytest

from harness import gate, reference

MU = np.array([0.0, 3.0, -1.0, 10.0, 0.5])
SIGMA = np.array([1.0, 0.5, 2.0, 1.0, 0.1])
NAMES = [f"x.{i + 1}" for i in range(len(MU))]


def ar1(rng, n, chains, rho):
    """Stationary AR(1) chains with N(MU, SIGMA^2) marginals: dict name -> (n, chains)."""
    e = rng.standard_normal((n, chains, len(MU)))
    z = np.empty_like(e)
    z[0] = e[0]
    for t in range(1, n):
        z[t] = rho * z[t - 1] + math.sqrt(1 - rho**2) * e[t]
    x = MU + SIGMA * z
    return {name: x[:, :, i] for i, name in enumerate(NAMES)}


def estimated_reference(rng, n=10000):
    """A reference estimated from n independent draws, as posteriordb's are (10 chains x 1000,
    thinned). The draws are independent, so each MCSE is the sample sd over sqrt(n); this is
    what reference.summarise gives for them, without the cost of n-draw ESS estimates."""
    x = MU + SIGMA * rng.standard_normal((n, len(MU)))
    m = x.mean(axis=0)
    cols = {"x": x, "x2": x**2, "var": (x - m) ** 2}
    sd = {k: v.std(axis=0, ddof=1) for k, v in cols.items()}
    stats = {"mean": m, "sd": sd["x"], "mcse_mean": sd["x"] / math.sqrt(n),
             "sq_mean": cols["x2"].mean(axis=0), "sq_sd": sd["x2"], "mcse_sq": sd["x2"] / math.sqrt(n),
             "var": cols["var"].mean(axis=0), "var_sd": sd["var"], "mcse_var": sd["var"] / math.sqrt(n)}
    return reference.Reference("sim", "test", list(NAMES), stats, {})


def test_false_fail_rate_is_at_most_alpha():
    """Correct samplers with few and correlated draws, against an estimated reference.

    The gate is designed for a false-fail rate of at most ALPHA = 1 % per run
    (Bonferroni over 3 x 5 tests). 300 runs at a true rate of 1 % give 3
    fails on average; 10 or more would happen with probability about 0.1 %.
    The naive rule (point estimate of b2 above 0.01, no MCSE) fails most of
    the same runs, which is why the gate accounts for Monte Carlo error.
    """
    rng = np.random.default_rng(1)
    fails = naive = runs = 0
    for n, rho in ((100, 0.0), (250, 0.5), (500, 0.9)):
        for _ in range(100):
            ref = estimated_reference(rng)
            v = gate.check(ar1(rng, n, 4, rho), ref, rhat_bad=math.inf)
            runs += 1
            fails += not v.passed
            naive += v.b2 > gate.THRESHOLD
    assert runs == 300
    assert fails < 10, fails
    assert naive > 120, naive


def test_threshold_and_z():
    assert gate.THRESHOLD == 0.01 and gate.ALPHA == 0.01
    assert gate.critical_z(1) == pytest.approx(2.5758, abs=1e-4)
    assert gate.critical_z(30) == pytest.approx(3.5879, abs=1e-3)


def test_correct_sampler_passes_with_many_draws():
    rng = np.random.default_rng(2)
    v = gate.check(ar1(rng, 1000, 4, 0.3), estimated_reference(rng))
    assert v.passed and v.reason == "" and v.label() == "pass"
    assert v.n_tests == 15


def test_scaled_draws_fail():
    """Draws multiplied by 1.2: the mean of x.4 (mu 10) is off by 2 standard deviations."""
    rng = np.random.default_rng(3)
    draws = {k: 1.2 * x for k, x in ar1(rng, 1000, 4, 0.3).items()}
    v = gate.check(draws, estimated_reference(rng))
    assert not v.passed and v.reason == "bias"
    assert v.worst_variable == "x.4" and v.b2 > 1
    assert v.label().startswith("wrong: x.4")


def test_overdispersed_around_the_mean_fails():
    """Variance 1.44x too large, means right: only the centred second moment catches it on x.4."""
    rng = np.random.default_rng(4)
    ref = estimated_reference(rng)
    draws = {k: MU[i] + 1.2 * (x - MU[i]) for i, (k, x) in enumerate(ar1(rng, 1000, 4, 0.0).items())}
    v = gate.check(draws, ref)
    assert not v.passed and v.worst_moment == "var"


def test_stuck_chain_fails_on_rhat():
    rng = np.random.default_rng(5)
    draws = ar1(rng, 1000, 4, 0.3)
    for x in draws.values():
        x[:, 0] = x[0, 0]
    v = gate.check(draws, estimated_reference(rng))
    assert not v.passed and v.reason == "rhat" and v.rhat_max > gate.RHAT_BAD


def test_nan_and_missing_fail():
    rng = np.random.default_rng(6)
    ref = estimated_reference(rng)
    draws = ar1(rng, 200, 4, 0.0)
    draws["x.2"][5, 1] = np.nan
    assert gate.check(draws, ref).reason == "nonfinite"
    draws = ar1(rng, 200, 4, 0.0)
    del draws["x.3"]
    v = gate.check(draws, ref)
    assert v.reason == "missing" and v.worst_variable == "x.3"


def test_extra_run_variables_are_ignored_by_the_bias_test():
    rng = np.random.default_rng(7)
    draws = dict(ar1(rng, 1000, 4, 0.0), extra=rng.standard_normal((1000, 4)) + 50)
    assert gate.check(draws, estimated_reference(rng)).passed
