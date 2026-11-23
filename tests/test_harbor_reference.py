"""The Stan-style Harbor references (harbor/references, generated into harbor/tasks/*/tests/reference.py).

Direct unit tests of the NUTS port (leapfrog and tree-builder reversibility, invariance of a known
Gaussian, Stan's window schedule), plus Tier 0 / Tier 1 validity of every lane's reference.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from harness import autoresearch
from harness.models import ROOT

TASKS = ROOT / "harbor" / "tasks"
REFERENCE_TASKS = {
    "A0": "a0-ground-up",
    "A1": "a1-preconditioning",
    "A2": "a2-funnel-dynamics",
    "A3": "a3-integrator-trajectory",
    "A4": "a4-nonlinear-transport",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref():
    mod = _load(TASKS / "a0-ground-up" / "tests" / "reference.py", "apin_test_stan_reference")
    yield mod
    mod.SHORT_WARMUP_RESTART = True


class GaussOracle:
    """N(0, cov) with a dense covariance; counts gradient calls."""

    def __init__(self, cov: np.ndarray):
        self.cov = cov
        self.prec = np.linalg.inv(cov)
        self.dim = cov.shape[0]
        self.calls = 0

    def log_density_gradient(self, x):
        self.calls += 1
        x = np.asarray(x, dtype=np.float64)
        g = -self.prec @ x
        return 0.5 * float(x @ g), g


def _rotated_cov(dim: int, kappa: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q @ np.diag(np.logspace(0.0, math.log10(kappa), dim)) @ q.T


def test_leapfrog_is_reversible(ref):
    oracle = GaussOracle(_rotated_cov(5, 30.0, 1))
    rng = np.random.default_rng(2)
    inv_m = rng.uniform(0.5, 3.0, size=5)
    q = rng.standard_normal(5)
    lp, g = oracle.log_density_gradient(q)
    z0 = ref._Point(q, rng.standard_normal(5), lp, g)
    z = z0
    for _ in range(25):
        z = ref._leapfrog(z, oracle, inv_m, 0.3)
    z = ref._Point(z.q, -z.p, z.lp, z.g)
    for _ in range(25):
        z = ref._leapfrog(z, oracle, inv_m, 0.3)
    assert np.allclose(z.q, z0.q, atol=1e-10) and np.allclose(-z.p, z0.p, atol=1e-10)


def _chain(ref, oracle, z0, inv_m, eps, n):
    """Leapfrog states at integer offsets -n..n from z0 (index i -> states[i + n])."""
    fwd, bck = [z0], [z0]
    for _ in range(n):
        fwd.append(ref._leapfrog(fwd[-1], oracle, inv_m, eps))
        bck.append(ref._leapfrog(bck[-1], oracle, inv_m, -eps))
    return bck[::-1] + fwd[1:]


def _directions_from(i: int, lo: int, depth: int) -> list[int]:
    """Doubling directions that rebuild the dyadic orbit [lo, lo + 2**depth) starting from index i."""
    return [1 if ((i - lo) >> j) % 2 == 0 else -1 for j in range(depth)]


def test_nuts_tree_builder_is_reversible(ref):
    """From every state of a NUTS orbit, the tree builder rebuilds the same orbit: same accepted
    depth, same total weight sum_k exp(-H_k) and the same stopping decision. This is the property
    that makes multinomial NUTS reversible with respect to the orbit."""
    oracle = GaussOracle(_rotated_cov(3, 20.0, 3))
    rng = np.random.default_rng(4)
    inv_m = np.array([1.0, 2.0, 0.5])
    max_depth = 7
    checked = 0
    for trial in range(12):
        q = rng.standard_normal(3) * 2.0
        lp, g = oracle.log_density_gradient(q)
        z0 = ref._Point(q, ref._sample_momentum(inv_m, rng), lp, g)
        eps = float(rng.uniform(0.15, 0.6))
        dirs = [int(d) for d in rng.choice([-1, 1], size=max_depth + 1)]
        t0 = ref._Trajectory(z0, oracle, inv_m, eps, np.random.default_rng(trial))
        t0.run(max_depth, directions=dirs)
        depth = t0.depth
        weight = t0.log_sum_weight - t0.h0
        lo = hi = 0
        for j in range(depth):
            if dirs[j] > 0:
                hi += 2**j
            else:
                lo -= 2**j
        states = _chain(ref, oracle, z0, inv_m, eps, 2**max_depth)
        n = 2**max_depth
        for i in range(lo, hi + 1):
            zi = states[i + n]
            d_i = _directions_from(i, lo, depth)
            # The extension that ended the original run, in the same absolute direction.
            d_i += [dirs[depth]] if depth < max_depth else []
            d_i += [1] * (max_depth + 1 - len(d_i))
            ti = ref._Trajectory(ref._Point(zi.q, zi.p, zi.lp, zi.g), oracle, inv_m, eps, np.random.default_rng(99))
            ti.run(max_depth, directions=d_i)
            assert ti.depth == depth, (trial, i)
            assert ti.log_sum_weight - ti.h0 == pytest.approx(weight, abs=1e-8), (trial, i)
            checked += 1
    assert checked > 50


def test_nuts_transition_leaves_a_gaussian_invariant(ref):
    """Exact draws from a rotated, ill-conditioned Gaussian stay distributed as that Gaussian after
    NUTS transitions with a mismatched diagonal metric (first and second moments, z-tests)."""
    dim, n = 4, 3000
    cov = _rotated_cov(dim, 25.0, 5)
    oracle = GaussOracle(cov)
    rng = np.random.default_rng(6)
    chol = np.linalg.cholesky(cov)
    x = rng.standard_normal((n, dim)) @ chol.T
    inv_m = np.array([1.0, 3.0, 0.7, 2.0])
    out = np.empty_like(x)
    for c in range(n):
        q = x[c]
        lp, g = oracle.log_density_gradient(q)
        for _ in range(2):
            z, _, _ = ref.nuts_transition(q, lp, g, oracle, inv_m, 0.45, rng)
            q, lp, g = z.q, z.lp, z.g
        out[c] = q
    sd = np.sqrt(np.diag(cov))
    z_mean = np.abs(out.mean(axis=0)) / (sd / math.sqrt(n))
    var_hat = out.var(axis=0, ddof=1)
    z_var = np.abs(var_hat - sd**2) / (sd**2 * math.sqrt(2.0 / (n - 1)))
    cross = out[:, 0] * out[:, 1]
    z_cross = abs(cross.mean() - cov[0, 1]) / (cross.std(ddof=1) / math.sqrt(n))
    assert np.all(z_mean < 4.5), z_mean
    assert np.all(z_var < 4.5), z_var
    assert z_cross < 4.5
    # and the chain moved
    assert float(np.mean(np.linalg.norm(out - x, axis=1))) > 0.5


def test_stan_window_schedule(ref):
    """Stan's windows: 75/25/50 doubling for 1000 warmup draws; one 15..89 window for 100."""
    def windows(num_warmup):
        var = ref._VarAdaptation(2, num_warmup)
        for _ in range(num_warmup):
            var.learn(np.random.default_rng(len(var.windows)).standard_normal(2))
        return var.windows

    assert windows(1000) == [(75, 99), (100, 149), (150, 249), (250, 449), (450, 949)]
    assert windows(100) == [(15, 89)]
    assert windows(19) == []
    ref.SHORT_WARMUP_RESTART = False
    try:
        assert windows(100) == []  # CmdStan 2.37's actual behaviour: the metric is never updated
        assert windows(1000) == [(75, 99), (100, 149), (150, 249), (250, 449), (450, 949)]
    finally:
        ref.SHORT_WARMUP_RESTART = True


def test_variance_regularization_matches_stan(ref):
    var = ref._VarAdaptation(3, 100)
    rng = np.random.default_rng(7)
    xs = rng.standard_normal((100, 3)) * np.array([0.5, 2.0, 7.0])
    got = None
    for x in xs:
        v = var.learn(x)
        if v is not None:
            got = v
    window = xs[15:90]
    n = len(window)
    expected = (n / (n + 5.0)) * window.var(axis=0, ddof=1) + 1e-3 * (5.0 / (n + 5.0))
    assert np.allclose(got, expected, rtol=1e-12)


def test_warmup_adapts_metric_and_step_size(ref):
    sd = np.array([0.1, 1.0, 10.0])
    oracle = GaussOracle(np.diag(sd**2))
    w = ref.stan_warmup(oracle, 100, np.random.default_rng(8))
    assert w["windows"] == [(15, 89)]
    assert np.all(np.isfinite(w["inv_m"])) and w["inv_m"][2] > 10 * w["inv_m"][0]
    assert 0.05 < w["eps"] < 5.0
    assert max(w["depths"]) <= ref.MAX_TREE_DEPTH


def test_reference_files_pass_the_import_lint_and_lane_seams():
    from harness import judge, logdensity

    for lane, task in REFERENCE_TASKS.items():
        path = TASKS / task / "tests" / "reference.py"
        logdensity.check_imports(path.read_text())
        judge.resolve_lane_submission(path, lane)


@pytest.mark.parametrize("lane", sorted(REFERENCE_TASKS))
def test_reference_passes_tier0(lane):
    verify = _load(ROOT / "harbor" / "verify.py", "apin_harbor_verify_t0")
    path = TASKS / REFERENCE_TASKS[lane] / "tests" / "reference.py"
    with tempfile.TemporaryDirectory() as d:
        ok, info = autoresearch._eval_tier0(path, seed=verify.HIDDEN_VERIFIER_SEED, work_dir=Path(d), lane=lane)
    assert ok, info


@pytest.mark.parametrize("lane", sorted(REFERENCE_TASKS))
def test_frozen_reference_passed_tier0_and_tier1_on_the_verifier_seeds(lane):
    """The frozen score of every reference is a pass on the verifier's own seeds and config. The
    freeze is current: test_harbor_scoring checks its key against today's judge and files."""
    frozen = json.loads((TASKS / REFERENCE_TASKS[lane] / "tests" / "frozen_scores.json").read_text())
    entry = frozen["scores"]["reference"]
    assert entry["valid"] and entry["tier0_ok"] and entry["tier1_ok"], entry.get("error")
    assert entry["soft_min_eff"] > 0
    assert entry["config"]["seed"] == _load(ROOT / "harbor" / "verify.py", "apin_harbor_verify_fz").HIDDEN_VERIFIER_SEED


@pytest.mark.slow
def test_reference_passes_tier1_live_on_the_verifier_seeds():
    """Full live Tier 0 + Tier 1 of the NUTS reference (lane A2) at the verifier config; must equal
    the frozen score exactly (scoring is deterministic given the seeds)."""
    verify = _load(ROOT / "harbor" / "verify.py", "apin_harbor_verify_live")
    tests = TASKS / REFERENCE_TASKS["A2"] / "tests"
    with tempfile.TemporaryDirectory() as d:
        live = verify.score_file(tests / "reference.py", "A2", verify.scoring_config(), Path(d))
    assert live["valid"], live.get("error")
    frozen = json.loads((tests / "frozen_scores.json").read_text())["scores"]["reference"]
    assert live["soft_min_eff"] == pytest.approx(frozen["soft_min_eff"], rel=1e-9)
