"""Slice 4 exact-draw invariance suite, dispersed-start mixing, Gandy-Scott ranks,
simulation-calibrated omnibus thresholds, adapt-billed empirical efficiency, and MDB calibration.

Implements Council Rulings B2, B3, and D3:
  - Runs file-path submissions in isolated out-of-process brokered workers (`harness.ipc`)
    so `adapt()` and `transition()` never share globals and cannot introspect `Counter` or
    `AnalyticTarget.exact_draws()`.
  - Samples exact joint starts `(position, aux) ~ (prod_w pi^{beta_w}) ⊗ aux_law` and tests:
      * every cold walker's unrotated latent functionals,
      * declared `aux_spec` marginal stationarity and `aux`-position independence,
      * inter-walker independence across all walker pairs (rank correlation),
      * Gandy-Scott exact-draw rank uniformity (location + spread, including hot walkers),
      * simulation-calibrated omnibus threshold against exact draws (prevents rare-indicator
        over-rejection such as Neal's funnel neck `v < -6` at small batch sizes).
  - Always pairs exact-draw invariance with a dedicated dispersed-start mixing pass:
      * verifies late-step latent displacement (`s >= k_steps // 2`, killing step-0-nudge
        and identity kernels),
      * computes true exact empirical efficiency `min_f(Var_pi(f) / MSE_f) / cost_per_chain`
        from dispersed starts with `adapt_counter` billed into `cost_per_chain`.
  - Supports a calibrated Minimum-Detectable-Bias (MDB) power-curve check (`calibrate_mdb`).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtri
from scipy.stats import rankdata

from . import ipc, judge, logdensity
from .logdensity import Budget, Chain, Counter, _resolve_submission, canonical_digest, normalize_aux_spec, normalize_chain
from .shapes import AnalyticTarget


def sample_aux(aux_spec: dict, walkers: int, dim: int, is_1d: bool, rng: np.random.Generator) -> Any:
    """Draw an exact independent sample from the stationary aux law declared in aux_spec."""
    spec = normalize_aux_spec(aux_spec)
    kind = spec["kind"]
    if kind == "none":
        return None
    if kind == "normal":
        shape = (dim,) if is_1d else (walkers, dim)
        mean = np.asarray(spec.get("mean", 0.0), dtype=float)
        if "diag" in spec:
            sd = np.sqrt(np.asarray(spec["diag"], dtype=float))
            return mean + rng.standard_normal(shape) * sd
        if "cov" in spec:
            cov = np.asarray(spec["cov"], dtype=float)
            chol = np.linalg.cholesky(cov)
            z = rng.standard_normal(shape)
            return mean + (z @ chol.T)
        return mean + rng.standard_normal(shape)
    if kind == "unit_sphere":
        d = int(spec.get("dim", dim))
        shape = (d,) if is_1d else (walkers, d)
        z = rng.standard_normal(shape)
        norms = np.linalg.norm(z, axis=-1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return z / norms
    if kind == "uniform_discrete":
        k = int(spec["k"])
        return int(rng.integers(0, k))
    if kind == "product":
        return tuple(sample_aux(sub, walkers, dim, is_1d, rng) for sub in spec["parts"])
    raise ValueError(f"unsupported aux_spec kind {kind!r}")


def _sample_aux_batch(aux_spec: dict, n_batches: int, n_chains: int, walkers: int, dim: int,
                      is_1d: bool, rng: np.random.Generator) -> Any:
    """Vectorized batch sampler of shape `(n_batches, n_chains, ...)` for null calibration."""
    spec = normalize_aux_spec(aux_spec)
    kind = spec["kind"]
    if kind == "none":
        return None
    if kind == "normal":
        shape = (n_batches, n_chains, dim) if is_1d else (n_batches, n_chains, walkers, dim)
        mean = np.asarray(spec.get("mean", 0.0), dtype=float)
        if "diag" in spec:
            sd = np.sqrt(np.asarray(spec["diag"], dtype=float))
            return mean + rng.standard_normal(shape) * sd
        if "cov" in spec:
            cov = np.asarray(spec["cov"], dtype=float)
            chol = np.linalg.cholesky(cov)
            z = rng.standard_normal(shape)
            return mean + (z @ chol.T)
        return mean + rng.standard_normal(shape)
    if kind == "unit_sphere":
        d = int(spec.get("dim", dim))
        shape = (n_batches, n_chains, d) if is_1d else (n_batches, n_chains, walkers, d)
        z = rng.standard_normal(shape)
        norms = np.linalg.norm(z, axis=-1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return z / norms
    if kind == "uniform_discrete":
        k = int(spec["k"])
        return rng.integers(0, k, size=(n_batches, n_chains))
    if kind == "product":
        return tuple(_sample_aux_batch(sub, n_batches, n_chains, walkers, dim, is_1d, rng) for sub in spec["parts"])
    raise ValueError(f"unsupported aux_spec kind {kind!r}")


def _aux_functionals_and_truth(
    aux_spec: dict,
    aux_arr: Any,
    z_ref: np.ndarray,
    dim: int,
    is_1d: bool,
    prefix: str = "aux_",
) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    """Compute standardized functionals and exact (mean, sd) truth for `aux` arrays.

    Works for both single-run shape `(n_chains, ...)` and batched null shape `(n_null, n_chains, ...)`.
    `z_ref` has shape `(..., n_chains)` with exact mean 0 and sd 1 from the primary cold walker.
    """
    spec = normalize_aux_spec(aux_spec)
    kind = spec["kind"]
    funcs: dict[str, np.ndarray] = {}
    truth: dict[str, tuple[float, float]] = {}
    if kind == "none":
        return funcs, truth
    if kind == "uniform_discrete":
        k = int(spec["k"])
        vals = np.asarray(aux_arr, dtype=int)
        p = 1.0 / k
        sd_p = math.sqrt(p * (1.0 - p))
        for j in range(k):
            name = f"{prefix}bin_{j}"
            funcs[name] = (vals == j).astype(float)
            truth[name] = (p, sd_p)
        ind0 = ((vals == 0).astype(float) - p) / sd_p
        funcs[f"{prefix}indep"] = ind0 * z_ref
        truth[f"{prefix}indep"] = (0.0, 1.0)
        return funcs, truth
    if kind == "normal":
        arr = np.asarray(aux_arr, dtype=float)
        a = arr if is_1d else arr[..., 0, :]
        mean = np.asarray(spec.get("mean", 0.0), dtype=float)
        diff = a - mean
        if "diag" in spec:
            u = diff / np.sqrt(np.asarray(spec["diag"], dtype=float))
        elif "cov" in spec:
            chol = np.linalg.cholesky(np.asarray(spec["cov"], dtype=float))
            flat = diff.reshape(-1, dim)
            u = solve_triangular(chol, flat.T, lower=True).T.reshape(diff.shape)
        else:
            u = diff
        funcs[f"{prefix}u0"] = u[..., 0]
        truth[f"{prefix}u0"] = (0.0, 1.0)
        if dim > 1:
            funcs[f"{prefix}u_last"] = u[..., -1]
            truth[f"{prefix}u_last"] = (0.0, 1.0)
        funcs[f"{prefix}rad_sq"] = np.mean(u ** 2, axis=-1)
        truth[f"{prefix}rad_sq"] = (1.0, math.sqrt(2.0 / dim))
        funcs[f"{prefix}indep"] = u[..., 0] * z_ref
        truth[f"{prefix}indep"] = (0.0, 1.0)
        return funcs, truth
    if kind == "unit_sphere":
        d = int(spec.get("dim", dim))
        arr = np.asarray(aux_arr, dtype=float)
        a = arr if is_1d else arr[..., 0, :]
        sd0 = 1.0 / math.sqrt(d)
        funcs[f"{prefix}s0"] = a[..., 0]
        truth[f"{prefix}s0"] = (0.0, sd0)
        funcs[f"{prefix}s_last"] = a[..., -1]
        truth[f"{prefix}s_last"] = (0.0, sd0)
        sd_sq = math.sqrt(2.0 * (d - 1.0) / (d * d * (d + 2.0)))
        funcs[f"{prefix}s0_sq"] = a[..., 0] ** 2
        truth[f"{prefix}s0_sq"] = (1.0 / d, sd_sq)
        funcs[f"{prefix}indep"] = (a[..., 0] / sd0) * z_ref
        truth[f"{prefix}indep"] = (0.0, 1.0)
        return funcs, truth
    if kind == "product":
        for i, sub in enumerate(spec["parts"]):
            sub_arr = aux_arr[i] if isinstance(aux_arr, tuple) else [item[i] for item in aux_arr]
            sf, st = _aux_functionals_and_truth(sub, sub_arr, z_ref, dim, is_1d, prefix=f"{prefix}p{i}_")
            funcs.update(sf)
            truth.update(st)
        return funcs, truth
    raise ValueError(f"unsupported aux_spec kind {kind!r}")


def _stack_aux_list(aux_spec: dict, aux_list: list[Any]) -> Any:
    """Convert a list of per-chain aux values into array/tuple structure matching `_aux_functionals_and_truth`."""
    spec = normalize_aux_spec(aux_spec)
    kind = spec["kind"]
    if kind == "none":
        return None
    if kind == "product":
        return tuple(
            _stack_aux_list(sub, [item[i] for item in aux_list])
            for i, sub in enumerate(spec["parts"])
        )
    return np.asarray(aux_list)


def _spearman_z(x: np.ndarray, y: np.ndarray) -> float:
    """Standardized Spearman rank correlation |r_s| * sqrt(n - 1) under independence."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if denom <= 0.0:
        return 0.0
    r_s = float(rx @ ry) / denom
    return abs(r_s) * math.sqrt(n - 1.0)


@dataclass
class InvarianceResult:
    target: str
    rotated: bool
    passed: bool
    invariance_ok: bool
    mixing_ok: bool
    max_z: float
    z_threshold: float
    b2_max: float
    empirical_efficiency: float  # min_f(Var_pi(f) / MSE_f) / cost_per_chain from dispersed starts
    mean_movement_sd: float  # mean late-step latent displacement (kills identity/step-0-nudge kernels)
    cost_total: float
    overhead_ratio: float
    reason: str = ""
    adapt_cost: float = 0.0
    transition_cost: float = 0.0
    cost_per_chain: float = 0.0
    aux_ok: bool = True
    rank_ok: bool = True
    dispersed_ess_min: float = 0.0


def _omnibus_threshold(n_funcs: int, alpha: float) -> float:
    """Bonferroni/Sidak normal threshold for max_{j=1..m} |z_j| at level alpha."""
    p_each = 1.0 - (1.0 - alpha) ** (1.0 / max(n_funcs, 1))
    return float(-ndtri(p_each / 2.0))


_CALIBRATION_CACHE: dict[tuple, float] = {}


def calibrated_omnibus_threshold(
    target: AnalyticTarget,
    aux_spec: dict,
    walkers: int,
    is_1d: bool,
    cold_idx: list[int],
    n_chains: int,
    alpha: float,
    *,
    n_null: int = 4000,
) -> float:
    """Calibrate the omnibus max-|z| cutoff empirically against exact draws at `(n_chains, alpha)`.

    Eliminates finite-sample normal-approximation over-rejection on skewed / rare-indicator
    functionals (such as Neal's funnel `neck_v_lt_m6` at n=64) while staying at least as
    strict as the analytical Sidak threshold on Gaussian functionals.
    """
    spec = normalize_aux_spec(aux_spec)
    key = (
        target.params_key(),
        canonical_digest(spec),
        int(walkers),
        bool(is_1d),
        len(cold_idx),
        int(n_chains),
        round(float(alpha), 8),
        int(n_null),
    )
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached

    seed_bytes = hashlib.sha256(repr(key).encode("utf-8")).digest()
    null_seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.default_rng(null_seed)

    truth_base = target.exact_truth()
    first_fn_name = next(iter(truth_base))
    first_mu, first_sd = truth_base[first_fn_name]

    max_z_null = np.zeros(n_null, dtype=float)
    n_total_funcs = 0

    # Cold walker latent functionals
    z_ref_primary = None
    for idx_pos in range(len(cold_idx)):
        z_all = target._latent_exact_draws(n_null * n_chains, rng).reshape(n_null, n_chains, target.dim)
        funcs = target.latent_functionals(z_all)
        if idx_pos == 0:
            z_ref_primary = (funcs[first_fn_name] - first_mu) / max(first_sd, 1e-12)
        for name, (mu_true, sd_true) in truth_base.items():
            se = sd_true / math.sqrt(n_chains)
            if se > 0.0:
                means = funcs[name].mean(axis=1)
                z_vec = np.abs(means - mu_true) / se
                max_z_null = np.maximum(max_z_null, z_vec)
            n_total_funcs += 1

    # Auxiliary state functionals & aux-position independence
    if spec["kind"] != "none":
        aux_batch = _sample_aux_batch(spec, n_null, n_chains, walkers, target.dim, is_1d, rng)
        aux_funcs, aux_truth = _aux_functionals_and_truth(spec, aux_batch, z_ref_primary, target.dim, is_1d)
        for name, (mu_true, sd_true) in aux_truth.items():
            se = sd_true / math.sqrt(n_chains)
            if se > 0.0:
                means = aux_funcs[name].mean(axis=1)
                z_vec = np.abs(means - mu_true) / se
                max_z_null = np.maximum(max_z_null, z_vec)
            n_total_funcs += 1

    # Inter-walker independence null (for walkers > 1)
    if walkers > 1:
        for _ in range(walkers - 1):
            # Under independence, Spearman rank correlation * sqrt(n - 1) is asymptotically N(0, 1)
            # Simulate exact rank correlation of independent uniform draws
            u1 = rng.random((n_null, n_chains))
            u2 = rng.random((n_null, n_chains))
            r1 = np.argsort(np.argsort(u1, axis=1), axis=1).astype(float)
            r2 = np.argsort(np.argsort(u2, axis=1), axis=1).astype(float)
            r1 -= r1.mean(axis=1, keepdims=True)
            r2 -= r2.mean(axis=1, keepdims=True)
            denom = np.sqrt(np.sum(r1 * r1, axis=1) * np.sum(r2 * r2, axis=1))
            r_s = np.where(denom > 0.0, np.sum(r1 * r2, axis=1) / denom, 0.0)
            max_z_null = np.maximum(max_z_null, np.abs(r_s) * math.sqrt(max(1.0, n_chains - 1.0)))
            n_total_funcs += 1

    sidak = _omnibus_threshold(n_total_funcs, alpha)
    emp_q = float(np.quantile(max_z_null, 1.0 - alpha))
    thresh = max(sidak, emp_q)
    _CALIBRATION_CACHE[key] = thresh
    return thresh


def _gandy_scott_rank_check(
    target: AnalyticTarget,
    final_pos: np.ndarray,
    betas: np.ndarray,
    cold_idx: list[int],
    alpha: float,
    rng: np.random.Generator,
    n_ranks: int = 19,
) -> tuple[bool, str]:
    """Non-parametric exact-draw rank uniformity check (Gandy-Scott / SBC style).

    Ranks each chain's endpoint functional against `n_ranks` fresh independent exact draws.
    Under exact invariance, normalized ranks u = (rank + 0.5) / (n_ranks + 1) follow exact
    Discrete Uniform on `n_ranks + 1` points — testing BOTH location (mean u) and spread
    (mean |u - 0.5|) with zero normal-tail assumptions, including hot walkers (beta < 1).
    """
    n_chains = final_pos.shape[0]
    m_bins = n_ranks + 1
    # Exact variance of U ~ DiscreteUniform({0.5/m, 1.5/m, ..., (m-0.5)/m})
    var_u = (m_bins * m_bins - 1.0) / (12.0 * m_bins * m_bins)
    sd_mean_u = math.sqrt(var_u / n_chains)
    # Exact mean & variance of |U - 0.5| on the m_bins grid
    grid = (np.arange(m_bins) + 0.5) / m_bins
    abs_dev = np.abs(grid - 0.5)
    mu_abs = float(np.mean(abs_dev))
    sd_abs = math.sqrt(float(np.var(abs_dev)) / n_chains)

    checks: list[tuple[str, float]] = []

    # 1. Cold walkers: check continuous latent functionals
    truth = target.exact_truth()
    ref_z = target._latent_exact_draws(n_chains * n_ranks, rng).reshape(n_chains, n_ranks, target.dim)
    ref_funcs = target.latent_functionals(ref_z)

    for idx_pos, w in enumerate(cold_idx):
        prefix = "" if len(cold_idx) == 1 else f"w{w}_"
        z_end = target.to_latent(final_pos[:, w, :])
        funcs = target.latent_functionals(z_end)
        for name in truth:
            vals = funcs[name]
            ref_v = ref_funcs[name]
            # Skip binary indicators (already tested by calibrated omnibus exact count)
            if len(np.unique(ref_v)) <= 2:
                continue
            less = np.sum(ref_v < vals[:, None], axis=1)
            eq = np.sum(ref_v == vals[:, None], axis=1)
            ranks = less + np.where(eq > 0, rng.integers(0, np.maximum(1, eq + 1)), 0)
            u = (ranks + 0.5) / m_bins
            z_loc = abs(float(np.mean(u)) - 0.5) / sd_mean_u
            z_spr = abs(float(np.mean(np.abs(u - 0.5))) - mu_abs) / sd_abs
            checks.append((f"{prefix}{name}:rank_loc", z_loc))
            checks.append((f"{prefix}{name}:rank_spread", z_spr))

    # 2. Hot walkers (beta < 1.0): check marginal stationarity along latent z[0]
    for w, b in enumerate(betas.tolist()):
        if b == 1.0:
            continue
        try:
            ref_hot = target.exact_tempered_draws(n_chains * n_ranks, b, rng).reshape(n_chains, n_ranks, target.dim)
        except NotImplementedError:
            continue
        z_hot = target.to_latent(final_pos[:, w, :])[:, 0]
        ref_z0 = target.to_latent(ref_hot)[..., 0]
        less = np.sum(ref_z0 < z_hot[:, None], axis=1)
        eq = np.sum(ref_z0 == z_hot[:, None], axis=1)
        ranks = less + np.where(eq > 0, rng.integers(0, np.maximum(1, eq + 1)), 0)
        u = (ranks + 0.5) / m_bins
        z_loc = abs(float(np.mean(u)) - 0.5) / sd_mean_u
        z_spr = abs(float(np.mean(np.abs(u - 0.5))) - mu_abs) / sd_abs
        checks.append((f"hot_w{w}:rank_loc", z_loc))
        checks.append((f"hot_w{w}:rank_spread", z_spr))

    if not checks:
        return True, ""
    thresh = _omnibus_threshold(len(checks), alpha)
    worst_name, worst_z = max(checks, key=lambda kv: kv[1])
    if worst_z > thresh:
        return False, f"rank test rejected on {worst_name}: |z|={worst_z:.2f} > {thresh:.2f}"
    return True, ""


def _counter_cost(counter: Counter, r_grad: float) -> float:
    return float(counter.density + r_grad * counter.gradient + r_grad * counter.hvp_calls + r_grad * counter.hess_calls)


def check_target_invariance(
    target: AnalyticTarget,
    submission: Path | str | tuple[Callable, Callable],
    *,
    seed: int = 1,
    n_chains: int = 64,
    k_steps: int = 25,
    adapt_draws: int = 150,
    alpha: float = 0.001,
    min_movement_sd: float = 1e-3,
    r_grad: float = 2.0,
    capabilities: set[str] | None = None,
    lane: str | None = None,
    isolated: bool = True,
    n_dispersed: int | None = None,
    check_ranks: bool = True,
) -> InvarianceResult:
    """Run adapt() once, freeze tuning, then run:
      1. Exact-draw joint invariance: `n_chains` independent starts drawn from
         `(prod_w pi^{beta_w}) ⊗ aux_law`, evolved `k_steps`; test calibrated omnibus max-|z|
         across all cold walkers, declared `aux_spec` marginals & independence, inter-walker
         independence, and Gandy-Scott exact-draw rank uniformity.
      2. Dispersed-start mixing & empirical efficiency: `n_dispersed` independent starts from
         `target.dispersed_starts()`, verifying late-step movement (`s >= k_steps // 2`) and
         computing `min_f(Var_pi(f) / MSE_f) / cost_per_chain` with `adapt_counter` billed.
    """
    use_ipc = isolated and isinstance(submission, (str, Path))
    rng = np.random.default_rng(seed)
    n_disp = int(n_dispersed if n_dispersed is not None else max(24, n_chains // 2))

    # 1. Adapt once and freeze tuning
    adapt_counter = Counter(target, max_evaluations=10**7, max_seconds=300.0, capabilities=capabilities)
    adapt_budget = Budget(1, adapt_draws, 10**7, 300.0)
    t_adapt0 = logdensity._now()
    if use_ipc:
        try:
            tuning, pos0_2d, aux0, is_1d, betas, aux_spec, tuning_sha = ipc.run_isolated_adapt(
                adapt_counter, Path(submission), adapt_budget, seed, lane=lane, seed_seq=[seed],
            )
            walkers = pos0_2d.shape[0]
        except Exception as e:
            return InvarianceResult(
                target.family, target.rotated, False, False, False, math.inf, 0.0,
                math.inf, 0.0, 0.0, 0.0, 0.0, f"adapt() failed: {e}",
            )
    else:
        if isinstance(submission, (str, Path)):
            adapt_fn, transition_fn = judge.resolve_lane_submission(Path(submission), lane)
        else:
            adapt_fn, transition_fn = _resolve_submission(submission)
        try:
            raw = adapt_fn(adapt_counter.oracle(), adapt_budget, np.random.default_rng(seed))
        except Exception as e:
            return InvarianceResult(
                target.family, target.rotated, False, False, False, math.inf, 0.0,
                math.inf, 0.0, 0.0, 0.0, 0.0, f"adapt() raised: {e}",
            )
        if not isinstance(raw, tuple) or len(raw) != 4:
            return InvarianceResult(
                target.family, target.rotated, False, False, False, math.inf, 0.0,
                math.inf, 0.0, 0.0, 0.0, 0.0, "adapt() did not return 4-tuple",
            )
        tuning, chain0_raw, betas_raw, aux_spec_raw = raw
        try:
            pos0_2d, aux0, is_1d = normalize_chain(chain0_raw, target.dim)
            walkers = pos0_2d.shape[0]
            betas = logdensity.validate_betas(betas_raw, walkers)
            aux_spec = logdensity.normalize_aux_spec(aux_spec_raw)
            logdensity.validate_aux(aux_spec, aux0, walkers, target.dim, is_1d)
            tuning_sha = logdensity.canonical_digest(tuning)
        except Exception as e:
            return InvarianceResult(
                target.family, target.rotated, False, False, False, math.inf, 0.0,
                math.inf, 0.0, 0.0, 0.0, 0.0, f"adapt() contract error: {e}",
            )
    adapt_elapsed = logdensity._now() - t_adapt0

    cold_idx = [w for w, b in enumerate(betas.tolist()) if b == 1.0]
    exact_starts = np.empty((n_chains, walkers, target.dim), dtype=float)
    for w, b in enumerate(betas.tolist()):
        exact_starts[:, w, :] = target.exact_tempered_draws(n_chains, b, rng)

    disp_starts = np.empty((n_disp, walkers, target.dim), dtype=float)
    for w, b in enumerate(betas.tolist()):
        if b == 1.0:
            disp_starts[:, w, :] = target.dispersed_starts(n_disp, rng)
        else:
            disp_starts[:, w, :] = target.exact_tempered_draws(n_disp, b, rng)

    # 2. Run Phase A (exact-draw invariance) & Phase B (dispersed-start mixing) in a clean post-adapt worker
    inv_counter = Counter(target, max_evaluations=10**8, max_seconds=300.0, capabilities=capabilities)
    disp_counter = Counter(target, max_evaluations=10**8, max_seconds=300.0, capabilities=capabilities)

    final_pos = np.empty((n_chains, walkers, target.dim), dtype=float)
    final_aux: list[Any] = []

    half_start = max(0, k_steps // 2)
    n_late = max(1, k_steps - half_start)
    disp_late_latent = np.empty((n_disp, n_late, target.dim), dtype=float)
    late_displacements: list[float] = []

    t_trans0 = logdensity._now()
    try:
        if use_ipc:
            with ipc.IsolatedTransitionSession(
                inv_counter, Path(submission), tuning, tuning_sha, walkers, is_1d, aux_spec, lane=lane,
            ) as sess:
                # Phase A: exact-draw invariance
                for c in range(n_chains):
                    pos = exact_starts[c].copy()
                    aux = sample_aux(aux_spec, walkers, target.dim, is_1d, rng)
                    for s in range(k_steps):
                        pos, aux = sess.step(pos, aux, [seed, 10, c, s])
                    final_pos[c] = pos
                    final_aux.append(aux)
                sess.verify_tuning_frozen()

                # Switch parent counter to disp_counter for Phase B (same frozen post-adapt worker)
                sess.counter = disp_counter
                for c in range(n_disp):
                    pos = disp_starts[c].copy()
                    aux = sample_aux(aux_spec, walkers, target.dim, is_1d, rng)
                    prev_z = target.to_latent(pos[cold_idx[0]])
                    for s in range(k_steps):
                        pos, aux = sess.step(pos, aux, [seed, 20, c, s])
                        cur_z = target.to_latent(pos[cold_idx[0]])
                        if s >= half_start:
                            late_displacements.append(float(np.linalg.norm(cur_z - prev_z)))
                            disp_late_latent[c, s - half_start] = cur_z
                        prev_z = cur_z
                sess.verify_tuning_frozen()
        else:
            inv_oracle = inv_counter.oracle()
            for c in range(n_chains):
                pos = exact_starts[c].copy()
                aux = sample_aux(aux_spec, walkers, target.dim, is_1d, rng)
                for s in range(k_steps):
                    inv_counter.reset_step()
                    arg = Chain(pos[0].copy() if is_1d else pos.copy(), aux)
                    raw_next = transition_fn(tuning, arg, inv_oracle, np.random.default_rng([seed, 10, c, s]))
                    pos_next, aux_next, _ = normalize_chain(
                        raw_next, target.dim, expected_walkers=walkers, expected_is_1d=is_1d
                    )
                    logdensity.validate_aux(aux_spec, aux_next, walkers, target.dim, is_1d)
                    inv_counter.check_step_provenance(pos, pos_next)
                    pos, aux = pos_next, aux_next
                final_pos[c] = pos
                final_aux.append(aux)

            disp_oracle = disp_counter.oracle()
            for c in range(n_disp):
                pos = disp_starts[c].copy()
                aux = sample_aux(aux_spec, walkers, target.dim, is_1d, rng)
                prev_z = target.to_latent(pos[cold_idx[0]])
                for s in range(k_steps):
                    disp_counter.reset_step()
                    arg = Chain(pos[0].copy() if is_1d else pos.copy(), aux)
                    raw_next = transition_fn(tuning, arg, disp_oracle, np.random.default_rng([seed, 20, c, s]))
                    pos_next, aux_next, _ = normalize_chain(
                        raw_next, target.dim, expected_walkers=walkers, expected_is_1d=is_1d
                    )
                    logdensity.validate_aux(aux_spec, aux_next, walkers, target.dim, is_1d)
                    disp_counter.check_step_provenance(pos, pos_next)
                    cur_z = target.to_latent(pos_next[cold_idx[0]])
                    if s >= half_start:
                        late_displacements.append(float(np.linalg.norm(cur_z - prev_z)))
                        disp_late_latent[c, s - half_start] = cur_z
                    prev_z = cur_z
                    pos, aux = pos_next, aux_next

            if logdensity.canonical_digest(tuning) != tuning_sha:
                raise logdensity.ContractError("tuning mutated during invariance test")
    except Exception as e:
        return InvarianceResult(
            target.family, target.rotated, False, False, False, math.inf, 0.0,
            math.inf, 0.0, 0.0, 0.0, 0.0, f"transition() failed: {e}",
        )
    trans_elapsed = logdensity._now() - t_trans0

    # 3. Evaluate calibrated omnibus statistic across all cold walkers, aux, and walker dependence
    truth_base = target.exact_truth()
    first_fn_name = next(iter(truth_base))
    first_mu, first_sd = truth_base[first_fn_name]

    z_scores: dict[str, float] = {}
    aux_z_scores: dict[str, float] = {}
    b2_vals: dict[str, float] = {}
    z_ref_primary = None

    for idx_pos, w in enumerate(cold_idx):
        prefix = "" if len(cold_idx) == 1 else f"w{w}_"
        z_end = target.to_latent(final_pos[:, w, :])
        funcs = target.latent_functionals(z_end)
        if idx_pos == 0:
            z_ref_primary = (funcs[first_fn_name] - first_mu) / max(first_sd, 1e-12)
        for name, (mu_true, sd_true) in truth_base.items():
            vals = funcs[name]
            err = float(np.mean(vals)) - mu_true
            se_null = sd_true / math.sqrt(n_chains)
            z_val = abs(err) / se_null if se_null > 0 else (0.0 if err == 0 else math.inf)
            z_scores[f"{prefix}{name}"] = z_val
            b2_vals[f"{prefix}{name}"] = (err / sd_true) ** 2 if sd_true > 0 else 0.0

    # Auxiliary state stationarity & aux-position independence
    if aux_spec["kind"] != "none":
        aux_stacked = _stack_aux_list(aux_spec, final_aux)
        aux_funcs, aux_truth = _aux_functionals_and_truth(aux_spec, aux_stacked, z_ref_primary, target.dim, is_1d)
        for name, (mu_true, sd_true) in aux_truth.items():
            vals = aux_funcs[name]
            err = float(np.mean(vals)) - mu_true
            se_null = sd_true / math.sqrt(n_chains)
            z_val = abs(err) / se_null if se_null > 0 else (0.0 if err == 0 else math.inf)
            z_scores[name] = z_val
            aux_z_scores[name] = z_val

    # Inter-walker independence across all walker pairs involving primary cold walker
    if walkers > 1:
        z_primary0 = target.to_latent(final_pos[:, cold_idx[0], :])[:, 0]
        for w in range(walkers):
            if w == cold_idx[0]:
                continue
            z_w0 = target.to_latent(final_pos[:, w, :])[:, 0]
            z_scores[f"walker_dep_{w}"] = _spearman_z(z_primary0, z_w0)

    z_thresh = calibrated_omnibus_threshold(
        target, aux_spec, walkers, is_1d, cold_idx, n_chains, alpha,
    )
    max_z = max(z_scores.values())
    worst_fn = max(z_scores, key=z_scores.get)
    b2_max = max(b2_vals.values()) if b2_vals else 0.0
    aux_ok = all(math.isfinite(v) and v <= z_thresh for v in aux_z_scores.values())
    omnibus_ok = bool(math.isfinite(max_z) and max_z <= z_thresh)

    # 4. Gandy-Scott exact-draw rank uniformity test (location + spread, incl. hot walkers)
    rank_ok, rank_reason = (True, "")
    if check_ranks:
        rank_ok, rank_reason = _gandy_scott_rank_check(
            target, final_pos, betas, cold_idx, alpha, np.random.default_rng([seed, 999]),
        )

    invariance_ok = omnibus_ok and aux_ok and rank_ok

    # 5. Dispersed-start mixing & true empirical efficiency Var_pi(f) / MSE per billed cost unit
    mean_move = float(np.mean(late_displacements)) if late_displacements else 0.0
    mixing_ok = bool(mean_move >= min_movement_sd)

    disp_funcs = target.latent_functionals(disp_late_latent)  # each value: (n_disp, n_late)
    ess_effs: list[float] = []
    for name, (mu_true, sd_true) in truth_base.items():
        if sd_true <= 0:
            continue
        chain_means = np.mean(disp_funcs[name], axis=1)
        mse = float(np.mean((chain_means - mu_true) ** 2))
        ess_effs.append((sd_true ** 2) / max(mse, 1e-12))

    # Bill adapt_counter + per-chain transition cost (no free warmup!)
    adapt_cost = _counter_cost(adapt_counter, r_grad)
    inv_trans_cost = _counter_cost(inv_counter, r_grad)
    disp_trans_cost = _counter_cost(disp_counter, r_grad)
    cost_total = adapt_cost + inv_trans_cost + disp_trans_cost
    cost_per_chain = max(1.0, adapt_cost + (disp_trans_cost / max(1, n_disp)))

    disp_ess_min = float(min(ess_effs)) if ess_effs else 0.0
    emp_eff = disp_ess_min / cost_per_chain

    total_elapsed = adapt_elapsed + trans_elapsed
    total_eval_s = adapt_counter.eval_s + inv_counter.eval_s + disp_counter.eval_s
    overhead_ratio = max(0.0, total_elapsed - total_eval_s) / total_eval_s if total_eval_s > 0 else 0.0

    passed = invariance_ok and mixing_ok
    reason = ""
    if not omnibus_ok:
        reason = f"invariance rejected on {worst_fn}: |z|={max_z:.2f} > {z_thresh:.2f} (b2_max={b2_max:.4g})"
    elif not rank_ok:
        reason = rank_reason
    elif not mixing_ok:
        reason = f"stuck/identity kernel: late mean latent step {mean_move:.3g} < {min_movement_sd:.3g}"

    return InvarianceResult(
        target=target.family,
        rotated=target.rotated,
        passed=passed,
        invariance_ok=invariance_ok,
        mixing_ok=mixing_ok,
        max_z=max_z,
        z_threshold=z_thresh,
        b2_max=b2_max,
        empirical_efficiency=emp_eff,
        mean_movement_sd=mean_move,
        cost_total=cost_total,
        overhead_ratio=overhead_ratio,
        reason=reason,
        adapt_cost=adapt_cost,
        transition_cost=inv_trans_cost + disp_trans_cost,
        cost_per_chain=cost_per_chain,
        aux_ok=aux_ok,
        rank_ok=rank_ok,
        dispersed_ess_min=disp_ess_min,
    )


def calibrate_mdb(
    target: AnalyticTarget,
    *,
    n_chains: int = 128,
    k_steps: int = 20,
    alpha: float = 0.001,
    seed: int = 1,
    ula_steps: tuple[float, ...] = (0.4, 0.9),
    mala_discounts: tuple[float, ...] = (0.15, 0.4),
) -> list[dict]:
    """Run graded known-bad kernels (ULA, (1-eps)-MALA, unadjusted HMC) to calibrate MDB power."""
    rows: list[dict] = []

    def _zero_adapt(oracle, budget, rng):
        x = np.zeros(oracle.dim, dtype=float)
        oracle.log_density_gradient(x)
        return {}, (x, None), [1.0], "none"

    # 1. Graded ULA (no MH step)
    for idx, h in enumerate(ula_steps):
        def ula_trans(tuning, chain, oracle, rng, _h=h):
            x = chain.position
            _, g = oracle.log_density_gradient(x)
            y = x + 0.5 * _h * g + math.sqrt(_h) * rng.standard_normal(x.size)
            return (y, None) if math.isfinite(oracle.log_density(y)) else (x, None)

        r = check_target_invariance(
            target, (_zero_adapt, ula_trans), seed=seed + 10 * idx,
            n_chains=n_chains, k_steps=k_steps, alpha=alpha,
        )
        rows.append({"kernel": f"ula_h={h}", "detected": not r.invariance_ok, "max_z": r.max_z,
                     "z_threshold": r.z_threshold, "b2_max": r.b2_max, "reason": r.reason})

    # 2. Graded (1 - eps)-MALA (missing fraction of MH log-ratio)
    for idx, eps in enumerate(mala_discounts):
        def bad_mala_trans(tuning, chain, oracle, rng, _eps=eps):
            x = chain.position
            h = 1.0
            lp, g = oracle.log_density_gradient(x)
            y = x + 0.5 * h * g + math.sqrt(h) * rng.standard_normal(x.size)
            lpy, gy = oracle.log_density_gradient(y)
            if not math.isfinite(lpy):
                return x, None
            dq_xy = -0.5 * float(np.sum((y - (x + 0.5 * h * g)) ** 2)) / h
            dq_yx = -0.5 * float(np.sum((x - (y + 0.5 * h * gy)) ** 2)) / h
            log_a = (1.0 - _eps) * (lpy - lp) + (dq_yx - dq_xy)
            if math.log(rng.uniform()) < min(0.0, log_a):
                return y, None
            return x, None

        r = check_target_invariance(
            target, (_zero_adapt, bad_mala_trans), seed=seed + 100 + 10 * idx,
            n_chains=n_chains, k_steps=k_steps, alpha=alpha,
        )
        rows.append({"kernel": f"mala_discount={eps}", "detected": not r.invariance_ok, "max_z": r.max_z,
                     "z_threshold": r.z_threshold, "b2_max": r.b2_max, "reason": r.reason})

    # 3. Missing-Jacobian HMC (non-volume-preserving leapfrog with MH omitting log|det J|)
    def bad_hmc_trans(tuning, chain, oracle, rng):
        x = chain.position
        lp, g = oracle.log_density_gradient(x)
        eps = 0.45
        p = rng.standard_normal(x.size)
        h0 = lp - 0.5 * float(p @ p)
        x1, g1, lp1 = x.copy(), g.copy(), lp
        p1 = p + 0.5 * eps * g1
        for i in range(4):
            p1 = 1.08 * p1  # non-volume-preserving drift whose Jacobian is omitted from MH
            x1 = x1 + eps * p1
            lp1, g1 = oracle.log_density_gradient(x1)
            if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
                return x, None
            if i < 3:
                p1 = p1 + eps * g1
        p1 = p1 + 0.5 * eps * g1
        h1 = lp1 - 0.5 * float(p1 @ p1)
        if math.isfinite(h1) and math.log(rng.uniform()) < min(0.0, h1 - h0):
            return x1, None
        return x, None

    r = check_target_invariance(
        target, (_zero_adapt, bad_hmc_trans), seed=seed + 200,
        n_chains=n_chains, k_steps=k_steps, alpha=alpha,
    )
    rows.append({"kernel": "hmc_missing_jacobian", "detected": not r.invariance_ok, "max_z": r.max_z,
                 "z_threshold": r.z_threshold, "b2_max": r.b2_max, "reason": r.reason})
    return rows


def soft_min_power_mean(values: list[float], p: float = -4.0) -> float:
    """Power mean M_p(x) = (mean(x^p))^(1/p) with p = -4 (Council D3 soft-min)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        return 0.0
    return float(np.mean(arr ** p) ** (1.0 / p))


def soft_min_lcb(class_scores: dict[str, list[float]], p: float = -4.0, n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Return (point_soft_min, lcb_95, overall_geomean) over non-multimodal class score lists."""
    classes = sorted(class_scores)
    means = [float(np.exp(np.mean(np.log(np.maximum(class_scores[c], 1e-18))))) for c in classes]
    point = soft_min_power_mean(means, p=p)
    geo = float(np.exp(np.mean(np.log(np.maximum(means, 1e-18)))))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        c_vals = []
        for c in classes:
            v = np.asarray(class_scores[c], dtype=float)
            pick = v[rng.integers(0, len(v), size=len(v))]
            c_vals.append(float(np.exp(np.mean(np.log(np.maximum(pick, 1e-18))))))
        boots[b] = soft_min_power_mean(c_vals, p=p)
    lcb = float(np.percentile(boots, 5.0))
    return point, lcb, geo
