"""Slice 4 tests: analytic shapes (finite-diff gradients, exact draws, tempered mixtures,
Student-t nu=1.5 defined functionals) and exact-draw invariance (good HMC passes, ULA /
identity / biased MALA fail, lifted aux kernels work, soft-min LCB)."""

import math
from pathlib import Path

import numpy as np
import pytest

from harness import invariance, judge, shapes
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


def test_shapes_handle_extreme_overflow_inputs_cleanly():
    """Review 3 #4: extreme proposals during unstable leapfrog must return (-inf, nan), never OverflowError."""
    suite = shapes.make_suite(seed=7, dim=4, include_multimodal=True)
    for tgt in suite:
        for bad in (np.full(tgt.dim, 1e200), np.full(tgt.dim, -1e200), np.full(tgt.dim, math.nan)):
            lp, g = tgt.log_density_gradient(bad)
            assert lp == -math.inf and np.all(np.isnan(g)), (tgt.family, lp, g)


def test_invariance_catches_aux_law_violation_and_walker_copying():
    """Review 1 #1: resetting a declared uniform_discrete aux bit to 0 or copying walkers MUST fail."""
    tgt = shapes.IllCondGaussian(3, seed=5, rotated=False, kappa=4.0)

    # 1. Exact position draws (RWM on Gaussian) that reset uniform_discrete(2) aux bit to 0
    def bad_aux_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density(x)
        return {}, Chain(x, 0), [1.0], {"kind": "uniform_discrete", "k": 2}

    def bad_aux_trans(tuning, chain, oracle, rng):
        x = chain.position
        lpx = oracle.log_density(x)
        cand = x + 0.8 * rng.standard_normal(x.size)
        lpc = oracle.log_density(cand)
        if math.isfinite(lpc) and math.log(rng.uniform()) < (lpc - lpx):
            x = cand
        return Chain(x, 0)  # violates Unif({0, 1})!

    res_aux = invariance.check_target_invariance(
        tgt, (bad_aux_adapt, bad_aux_trans), seed=3, n_chains=64, k_steps=10,
    )
    assert not res_aux.passed and not res_aux.invariance_ok and not res_aux.aux_ok, res_aux

    # 2. 2-walker kernel that copies walker 0 into walker 1 (destroys independence)
    def copy_adapt(oracle, budget, rng):
        pos = np.zeros((2, oracle.dim))
        oracle.log_density(pos[0])
        return {}, Chain(pos, None), [1.0, 1.0], "none"

    def copy_trans(tuning, chain, oracle, rng):
        x = chain.position[0]
        lpx = oracle.log_density(x)
        cand = x + 0.8 * rng.standard_normal(x.size)
        lpc = oracle.log_density(cand)
        if math.isfinite(lpc) and math.log(rng.uniform()) < (lpc - lpx):
            x = cand
        return Chain(np.vstack([x, x]), None)

    res_copy = invariance.check_target_invariance(
        tgt, (copy_adapt, copy_trans), seed=3, n_chains=64, k_steps=10,
    )
    assert not res_copy.passed and not res_copy.invariance_ok, res_copy


def test_calibrated_funnel_threshold_controls_false_rejections():
    """Review 1 #2: calibrated cutoff avoids 5x over-rejection on rare funnel neck indicator."""
    funnel = shapes.NealsFunnel(dim=4, seed=11, rotated=False, v_sd=3.0)
    n_chains = 64
    alpha = 0.001
    cal_thresh = invariance.calibrated_omnibus_threshold(
        funnel, {"kind": "none"}, 1, True, [0], n_chains, alpha, n_null=4000,
    )
    naive_sidak = invariance._omnibus_threshold(len(funnel.exact_truth()), alpha)
    assert cal_thresh > naive_sidak

    # Evaluate false rejection rate on 2,000 fresh exact-draw batches
    rng = np.random.default_rng(98765)
    n_batches = 2000
    z_all = funnel._latent_exact_draws(n_batches * n_chains, rng).reshape(n_batches, n_chains, funnel.dim)
    funcs = funnel.latent_functionals(z_all)
    truth = funnel.exact_truth()
    max_z = np.zeros(n_batches, dtype=float)
    for name, (mu, sd) in truth.items():
        z_vec = np.abs(funcs[name].mean(axis=1) - mu) / (sd / math.sqrt(n_chains))
        max_z = np.maximum(max_z, z_vec)
    rej_rate = float(np.mean(max_z > cal_thresh))
    assert rej_rate <= 0.003, f"expected calibrated rate ~0.001, got {rej_rate:.4f}"


def test_adapt_cost_is_billed_and_step0_nudge_kernel_fails_mixing():
    """Review 2 P0#1 & P0#3: expensive adapt() lowers empirical efficiency; step-0-only movement fails mixing."""
    tgt = shapes.IllCondGaussian(3, seed=17, rotated=False, kappa=4.0)

    def cheap_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density(x)
        return {"step": 0.9}, (x, None), [1.0], "none"

    def bloated_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        for _ in range(2000):
            oracle.log_density_gradient(x)
        return {"step": 0.9}, (x, None), [1.0], "none"

    def rwm_trans(tuning, chain, oracle, rng):
        x = chain.position
        lpx = oracle.log_density(x)
        cand = x + tuning["step"] * rng.standard_normal(x.size)
        lpc = oracle.log_density(cand)
        if math.isfinite(lpc) and math.log(rng.uniform()) < (lpc - lpx):
            return cand, None
        return x, None

    r_cheap = invariance.check_target_invariance(tgt, (cheap_adapt, rwm_trans), seed=5, n_chains=48, k_steps=14)
    r_bloat = invariance.check_target_invariance(tgt, (bloated_adapt, rwm_trans), seed=5, n_chains=48, k_steps=14)
    assert r_cheap.passed and r_bloat.passed
    assert r_bloat.adapt_cost > r_cheap.adapt_cost + 3000.0
    assert r_bloat.empirical_efficiency < 0.1 * r_cheap.empirical_efficiency

    # Step-0 nudge then freeze: moves slightly only if aux==0, then freezes (aux stays 1) -> fails late-step mixing
    def nudge_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density(x)
        return {}, Chain(x, None), [1.0], "none"

    def stuck_trans(tuning, chain, oracle, rng):
        return chain.position, None

    r_stuck = invariance.check_target_invariance(tgt, (nudge_adapt, stuck_trans), seed=5, n_chains=32, k_steps=12)
    assert not r_stuck.passed and not r_stuck.mixing_ok


def test_ipc_blocks_closure_introspection_and_global_smuggling(tmp_path):
    """Review 3 #3 & Review 2 P0#2: out-of-process oracle prevents exact-draw theft and adapt->transition globals."""
    tgt = shapes.IllCondGaussian(4, seed=21, rotated=True, kappa=10.0)

    # Exploit 1: closure / target introspection trying to reach Counter or AnalyticTarget.exact_draws()
    cheat_introspect = tmp_path / "cheat_introspect.py"
    cheat_introspect.write_text(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng):\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    for cell in (oracle.log_density.__closure__ or ()):\n"
        "        obj = cell.cell_contents\n"
        "        if hasattr(obj, '__self__') and hasattr(obj.__self__, '_model'):\n"
        "            raise RuntimeError('SECURITY BREACH: reached Counter._model')\n"
        "    # Try to steal exact_draws from closure -> must fail\n"
        "    oracle.log_density.__closure__[0].cell_contents.__self__._model.exact_draws(1, rng)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    return chain.position, None\n"
    )
    res_cheat = invariance.check_target_invariance(
        tgt, cheat_introspect, seed=1, n_chains=32, k_steps=8, isolated=True,
    )
    assert not res_cheat.passed and "SECURITY BREACH" not in res_cheat.reason

    # Exploit 2: module global smuggled from adapt() to transition()
    cheat_global = tmp_path / "cheat_global.py"
    cheat_global.write_text(
        "import numpy as np\n"
        "_STASH = None\n"
        "def adapt(oracle, budget, rng):\n"
        "    global _STASH\n"
        "    _STASH = 42\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    if _STASH is None:\n"
        "        raise RuntimeError('global blocked by clean post-adapt worker')\n"
        "    return chain.position, None\n"
    )
    res_glob = invariance.check_target_invariance(
        tgt, cheat_global, seed=1, n_chains=32, k_steps=8, isolated=True,
    )
    assert not res_glob.passed and "global blocked" in res_glob.reason


def test_a1_fixed_kernel_supports_diag_chol_and_low_rank_metrics():
    """Verify Lane A1 harness-owned HMC kernel preserves exact invariance for diag, dense Cholesky, and low-rank."""
    tgt = shapes.IllCondGaussian(4, seed=31, rotated=True, kappa=16.0)
    # Exact covariance in observed coordinates: Q @ diag(stds^2) @ Q.T
    cov = tgt._Q @ np.diag(tgt.stds ** 2) @ tgt._Q.T
    chol = np.linalg.cholesky(cov)

    # Low-rank decomposition on standardized correlation
    d_diag = np.diag(cov)
    inv_sqrt_d = 1.0 / np.sqrt(d_diag)
    corr = (cov * inv_sqrt_d[:, None]) * inv_sqrt_d[None, :]
    eigs, vecs = np.linalg.eigh(corr)

    configs = [
        {"kind": "diag", "eps": 0.35, "inv_m": d_diag},
        {"kind": "chol", "eps": 0.45, "inv_m_chol": chol},
        {"kind": "low_rank", "eps": 0.45, "inv_m_diag": d_diag, "low_rank_u": vecs[:, -2:], "low_rank_eigs": eigs[-2:]},
    ]
    for idx, tuning in enumerate(configs):
        def _ad(oracle, budget, rng, _t=tuning):
            x = np.zeros(oracle.dim)
            oracle.log_density_gradient(x)
            return _t, (x, None), [1.0], "none"

        res = invariance.check_target_invariance(
            tgt, (_ad, judge.a1_transition), seed=10 + idx, n_chains=80, k_steps=14,
        )
        assert res.passed, (tuning["kind"], res)


def test_calibrate_mdb_detects_known_bad_kernels():
    """Review 2 P0#3: graded MDB calibration detects ULA, (1-eps)-MALA, and uncorrected HMC."""
    tgt = shapes.IllCondGaussian(3, seed=9, rotated=False, kappa=9.0)
    rows = invariance.calibrate_mdb(
        tgt, n_chains=220, k_steps=20, alpha=0.001, seed=3, ula_steps=(0.9,), mala_discounts=(0.5,),
    )
    by = {r["kernel"]: r for r in rows}
    assert by["ula_h=0.9"]["detected"], by["ula_h=0.9"]
    assert by["mala_discount=0.5"]["detected"], by["mala_discount=0.5"]
    assert by["hmc_missing_jacobian"]["detected"], by["hmc_missing_jacobian"]
