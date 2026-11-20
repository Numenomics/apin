"""Slice 4 tests: analytic shapes (finite-diff gradients, exact draws, tempered mixtures,
Student-t nu=1.5 defined functionals) and exact-draw invariance (good HMC passes, ULA /
identity / biased MALA fail, lifted aux kernels work, soft-min LCB)."""

import math
from pathlib import Path

import numpy as np
import pytest

from harness import invariance, shapes
from harness.logdensity import Chain

EXAMPLES = Path(__file__).parent.parent / "examples" / "samplers"


@pytest.mark.parametrize("factory", [
    lambda: shapes.IllCondGaussian(5, seed=1, rotated=True, scale_stress=2.0, kappa=50.0),
    lambda: shapes.NealsFunnel(5, seed=2, rotated=True, scale_stress=0.5),
    lambda: shapes.Banana(5, seed=3, rotated=True, scale_stress=1.5),
    lambda: shapes.HybridRosenbrock(4, seed=4, rotated=True),
    lambda: shapes.StudentT(4, seed=5, rotated=True, nu=1.5),
    lambda: shapes.StudentT(4, seed=6, rotated=False, nu=5.0),
    lambda: shapes.GaussianMixture(4, seed=7, rotated=True, separation=3.0),
    lambda: shapes.GaussianMixture(4, seed=8, rotated=False, separation=7.0, weights=(0.3, 0.7), multimodal=True),
])
def test_analytic_shapes_gradient_and_exact_moments(factory):
    tgt = factory()
    rng = np.random.default_rng(123)
    draws = tgt.exact_draws(4000, rng)
    # 1. Finite-difference gradient check at an exact draw
    x0 = draws[0]
    lp0, g0 = tgt.log_density_gradient(x0)
    assert math.isfinite(lp0) and np.all(np.isfinite(g0))
    eps = 1e-6
    fd = np.empty_like(g0)
    for j in range(tgt.dim):
        e = np.zeros(tgt.dim)
        e[j] = eps
        fd[j] = (tgt.log_density(x0 + e) - tgt.log_density(x0 - e)) / (2.0 * eps)
    assert np.allclose(g0, fd, atol=1e-4, rtol=1e-4)

    # 2. Exact draws match exact_truth() within Monte Carlo error
    z = tgt.to_latent(draws)
    funcs = tgt.latent_functionals(z)
    truth = tgt.exact_truth()
    for name, (mu, sd) in truth.items():
        err = abs(float(np.mean(funcs[name])) - mu)
        assert err < 5.0 * sd / math.sqrt(len(draws)), (tgt.family, name, err, sd)


def test_student_t_nu_1_5_excludes_undefined_moments():
    t15 = shapes.StudentT(3, seed=1, nu=1.5)
    truth15 = t15.exact_truth()
    assert "z0_mean" not in truth15 and "z0_sq" not in truth15
    assert {"z0_lt_q05", "z0_lt_q95", "z0_in_iqr", "z0_pos"} == set(truth15)

    t5 = shapes.StudentT(3, seed=1, nu=5.0)
    assert {"z0_mean", "z0_sq"} <= set(t5.exact_truth())


def test_tempered_gaussian_mixture_exact_rejection_sampler():
    """Council B2: exact rejection draws from pi^beta match numerical importance weights."""
    mix = shapes.GaussianMixture(2, seed=42, rotated=False, separation=4.0, weights=(0.25, 0.75), multimodal=True)
    rng = np.random.default_rng(9)
    beta = 0.4
    draws = mix.exact_tempered_draws(8000, beta, rng)
    z = mix.to_latent(draws)
    # Importance-reweight draws from pi^beta back to pi (w_i propto exp((1 - beta) log pi(x)))
    lps = np.array([mix.log_density(x) for x in draws])
    log_w = (1.0 - beta) * lps
    w = np.exp(log_w - np.max(log_w))
    w /= w.sum()
    est_mode1 = float(np.sum(w * (z[:, 0] > 0.0)))
    true_mode1 = mix.exact_truth()["mode1_weight"][0]
    assert abs(est_mode1 - true_mode1) < 0.03


def test_invariance_passes_good_hmc_and_catches_ula_and_identity():
    tgt = shapes.IllCondGaussian(4, seed=11, rotated=True, kappa=10.0)

    # 1. Good HMC passes both invariance and mixing
    good = invariance.check_target_invariance(
        tgt, EXAMPLES / "hmc.py", seed=7, n_chains=96, k_steps=15, adapt_draws=120
    )
    assert good.passed and good.invariance_ok and good.mixing_ok, good

    # 2. Identity kernel passes stationarity but FAILS mixing (Council B3)
    def id_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density(x)
        return {}, (x, None), [1.0], "none"

    def id_trans(tuning, chain, oracle, rng):
        return chain.position, None

    stuck = invariance.check_target_invariance(tgt, (id_adapt, id_trans), seed=7, n_chains=48, k_steps=10)
    assert not stuck.passed and stuck.invariance_ok and not stuck.mixing_ok

    # 3. Deliberately biased ULA (large step) FAILS invariance
    def big_ula_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density_gradient(x)
        return {"h": 0.9}, (x, None), [1.0], "none"

    def big_ula_trans(tuning, chain, oracle, rng):
        x = chain.position
        _, g = oracle.log_density_gradient(x)
        y = x + 0.5 * tuning["h"] * g + math.sqrt(tuning["h"]) * rng.standard_normal(x.size)
        if math.isfinite(oracle.log_density(y)):
            return y, None
        return x, None

    bad = invariance.check_target_invariance(tgt, (big_ula_adapt, big_ula_trans), seed=7, n_chains=256, k_steps=20)
    assert not bad.passed and not bad.invariance_ok, bad


def test_lifted_nonreversible_nrpt_parity_kernel_passes_tempered_mixture_invariance():
    """Exact-draw joint stationarity on pi^beta0 x pi^1 x Unif({0,1}) for a lifted parity kernel."""
    mix = shapes.GaussianMixture(2, seed=5, rotated=False, separation=3.0)

    def adapt(oracle, budget, rng):
        pos = rng.standard_normal((2, oracle.dim))
        for w in range(2):
            oracle.log_density(pos[w])
        return {"step": 0.5, "betas": [0.5, 1.0]}, Chain(pos, 0), [0.5, 1.0], {"kind": "uniform_discrete", "k": 2}

    def transition(tuning, chain, oracle, rng):
        pos = chain.position.copy()
        parity = int(chain.aux)
        betas = tuning["betas"]
        lps = [oracle.log_density(pos[w]) for w in range(2)]
        # Exact RWM step per walker at its inverse temperature beta
        for w, b in enumerate(betas):
            cand = pos[w] + tuning["step"] * rng.standard_normal(oracle.dim)
            lpc = oracle.log_density(cand)
            if math.isfinite(lpc) and math.log(rng.uniform()) < b * (lpc - lps[w]):
                pos[w], lps[w] = cand, lpc
        # DEO swap when parity == 0
        if parity == 0:
            log_a = (betas[1] - betas[0]) * (lps[0] - lps[1])
            if math.log(rng.uniform()) < log_a:
                pos[0], pos[1] = pos[1].copy(), pos[0].copy()
        return Chain(pos, 1 - parity)

    res = invariance.check_target_invariance(mix, (adapt, transition), seed=19, n_chains=128, k_steps=15)
    assert res.passed, res


def test_soft_min_power_mean_and_lcb():
    # Soft-min heavily penalizes a weak class unlike geomean
    sm_bal = invariance.soft_min_power_mean([1.0, 1.0, 1.0], p=-4.0)
    sm_weak = invariance.soft_min_power_mean([2.0, 2.0, 0.2], p=-4.0)
    assert sm_bal == pytest.approx(1.0)
    assert sm_weak < 0.3  # dominated by the 0.2 funnel failure
    pt, lcb, geo = invariance.soft_min_lcb({"a": [1.0, 1.1, 0.9], "b": [0.3, 0.28, 0.32]})
    assert 0 < lcb <= pt < geo
