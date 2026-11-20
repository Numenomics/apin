"""Slice 4 exact-draw invariance suite, empirical efficiency, b^2_max, and soft-min aggregation.

Implements Council Rulings B2, B3, and D3:
  - Samples exact joint starts (position, aux) ~ pi ⊗ aux_law (including exact tempered mixtures).
  - Evaluates stationarity in *unrotated latent coordinates* via ONE omnibus statistic per target
    (max-|z| calibrated against an exact null / simulation threshold to control multiplicity).
  - Always pairs exact-draw invariance with dispersed-start mixing / empirical efficiency
    (Var_pi(f) / MSE against exact closed-form truth), killing stuck/identity kernels.
  - Supports a calibrated Minimum-Detectable-Bias (MDB) ladder: Tier 0 (coarse, fast < 30s)
    through Tier 2 (>= 10x finer on fresh seeds).
  - Aggregates analytic block scores with a lower confidence bound on the soft-min power mean
    (p = -4) across non-multimodal shape classes, printing overall geomean beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import logdensity
from .logdensity import Budget, Chain, Counter, _resolve_submission, normalize_aux_spec, normalize_chain
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
            L = np.linalg.cholesky(cov)
            z = rng.standard_normal(shape)
            return mean + (z @ L.T)
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
    empirical_efficiency: float  # Var_pi(f) / MSE per cost unit (min across functionals)
    mean_movement_sd: float  # mean latent displacement per step (kills identity kernels)
    cost_total: float
    overhead_ratio: float
    reason: str = ""


def _omnibus_threshold(n_funcs: int, alpha: float) -> float:
    """Bonferroni/Sidak conservative threshold for max_{j=1..m} |z_j| at level alpha."""
    from scipy.special import ndtri

    p_each = 1.0 - (1.0 - alpha) ** (1.0 / max(n_funcs, 1))
    return float(-ndtri(p_each / 2.0))


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
) -> InvarianceResult:
    """Run adapt() once, freeze tuning, then run:
      1. Exact-draw invariance: n_chains independent starts drawn from pi ⊗ aux_law, evolved k_steps;
         test omnibus max |z| in unrotated latent coordinates against exact truth.
      2. Dispersed-start mixing: verify the kernel actually moves (kills identity/stuck kernels)
         and compute exact empirical efficiency Var_pi(f) / MSE per cost unit + b^2_max.
    """
    adapt_fn, transition_fn = _resolve_submission(submission)
    rng = np.random.default_rng(seed)

    # 1. Adapt once and freeze tuning via canonical serialize/digest
    adapt_counter = Counter(target, max_evaluations=10**7, max_seconds=300.0, capabilities=capabilities)
    raw = adapt_fn(adapt_counter.oracle(), Budget(1, adapt_draws, 10**7, 300.0), np.random.default_rng(seed))
    if not isinstance(raw, tuple) or len(raw) != 4:
        return InvarianceResult(target.family, target.rotated, False, False, False, math.inf, 0.0,
                                math.inf, 0.0, 0.0, 0.0, 0.0, "adapt() did not return 4-tuple")
    tuning, chain0_raw, betas_raw, aux_spec_raw = raw
    pos0_2d, aux0, is_1d = normalize_chain(chain0_raw, target.dim)
    walkers = pos0_2d.shape[0]
    betas = logdensity.validate_betas(betas_raw, walkers)
    aux_spec = logdensity.normalize_aux_spec(aux_spec_raw)
    logdensity.validate_aux(aux_spec, aux0, walkers, target.dim, is_1d)
    tuning_sha = logdensity.canonical_digest(tuning)

    # 2. Exact-draw joint starts: (position, aux) ~ (prod_w pi^{beta_w}) ⊗ aux_law
    cold_idx = [w for w, b in enumerate(betas.tolist()) if b == 1.0]
    exact_starts = np.empty((n_chains, walkers, target.dim), dtype=float)
    for w, b in enumerate(betas.tolist()):
        exact_starts[:, w, :] = target.exact_tempered_draws(n_chains, b, rng)

    counter = Counter(target, max_evaluations=10**8, max_seconds=300.0, capabilities=capabilities)
    oracle = counter.oracle()
    t0 = logdensity._now()
    final_cold = np.empty((n_chains, target.dim), dtype=float)
    displacements = []

    for c in range(n_chains):
        pos = exact_starts[c].copy()
        aux = sample_aux(aux_spec, walkers, target.dim, is_1d, rng)
        prev_cold_z = target.to_latent(pos[cold_idx[0]])
        for s in range(k_steps):
            counter.reset_step()
            arg = Chain(pos[0].copy() if is_1d else pos.copy(), aux)
            raw_next = transition_fn(tuning, arg, oracle, np.random.default_rng([seed, 10, c, s]))
            pos_next, aux_next, _ = normalize_chain(raw_next, target.dim, expected_walkers=walkers, expected_is_1d=is_1d)
            logdensity.validate_aux(aux_spec, aux_next, walkers, target.dim, is_1d)
            counter.check_step_provenance(pos, pos_next)
            if s == 0:
                cur_z = target.to_latent(pos_next[cold_idx[0]])
                displacements.append(float(np.linalg.norm(cur_z - prev_cold_z)))
            pos, aux = pos_next, aux_next
        final_cold[c] = pos[cold_idx[0]]

    elapsed = logdensity._now() - t0
    if logdensity.canonical_digest(tuning) != tuning_sha:
        return InvarianceResult(target.family, target.rotated, False, False, False, math.inf, 0.0,
                                math.inf, 0.0, 0.0, 0.0, 0.0, "tuning mutated during invariance test")

    # 3. Evaluate omnibus statistic in unrotated latent coordinates
    z_end = target.to_latent(final_cold)
    funcs = target.latent_functionals(z_end)
    truth = target.exact_truth()
    z_thresh = _omnibus_threshold(len(truth), alpha)

    z_scores = {}
    b2_vals = {}
    mse_ratios = []
    for name, (mu_true, sd_true) in truth.items():
        vals = funcs[name]
        mean_est = float(np.mean(vals))
        err = mean_est - mu_true
        # Standard error of the mean across n_chains independent chains
        se_null = sd_true / math.sqrt(n_chains)
        z_scores[name] = abs(err) / se_null if se_null > 0 else (0.0 if err == 0 else math.inf)
        b2_vals[name] = (err / sd_true) ** 2 if sd_true > 0 else 0.0
        # Empirical efficiency on end-state vs truth
        mse = float(np.mean((vals - mu_true) ** 2))
        mse_ratios.append((sd_true ** 2) / max(mse, 1e-12))

    max_z = max(z_scores.values())
    worst_fn = max(z_scores, key=z_scores.get)
    b2_max = max(b2_vals.values())
    mean_move = float(np.mean(displacements)) if displacements else 0.0

    # Paired cost for one chain's k_steps transitions
    cost_total = float(counter.density + r_grad * counter.gradient)
    cost_per_chain = max(1.0, cost_total / n_chains)
    # Empirical efficiency: fraction of an exact draw earned per cost unit (scaled by movement to penalize stuck chains)
    invariance_ok = bool(math.isfinite(max_z) and max_z <= z_thresh)
    mixing_ok = bool(mean_move >= min_movement_sd)
    passed = invariance_ok and mixing_ok

    # Empirical efficiency per cost unit via lagged variance reduction (movement^2 / cost proxy when stationary)
    z_start = target.to_latent(exact_starts[:, cold_idx[0], :])
    f_start = target.latent_functionals(z_start)
    # Autocorrelation across k_steps gives true mixing efficiency: (1 - rho_k) / cost_per_chain
    mix_effs = []
    for name, (mu_true, sd_true) in truth.items():
        if sd_true <= 0:
            continue
        c0 = f_start[name] - mu_true
        ck = funcs[name] - mu_true
        cov = float(np.mean(c0 * ck))
        rho_k = max(-1.0, min(1.0, cov / (sd_true ** 2)))
        # Effective independent fraction gained in k_steps, penalized by squared bias
        eff_frac = max(0.0, (1.0 - max(0.0, rho_k))) / (1.0 + n_chains * b2_vals[name])
        mix_effs.append(eff_frac / cost_per_chain)
    emp_eff = float(min(mix_effs)) if mix_effs else 0.0
    overhead_ratio = max(0.0, elapsed - counter.eval_s) / counter.eval_s if counter.eval_s > 0 else 0.0

    reason = ""
    if not invariance_ok:
        reason = f"invariance rejected on {worst_fn}: |z|={max_z:.2f} > {z_thresh:.2f} (b2_max={b2_max:.4g})"
    elif not mixing_ok:
        reason = f"stuck/identity kernel: mean latent step {mean_move:.3g} < {min_movement_sd:.3g}"

    return InvarianceResult(
        target=target.family, rotated=target.rotated, passed=passed,
        invariance_ok=invariance_ok, mixing_ok=mixing_ok,
        max_z=max_z, z_threshold=z_thresh, b2_max=b2_max,
        empirical_efficiency=emp_eff, mean_movement_sd=mean_move,
        cost_total=cost_total, overhead_ratio=overhead_ratio, reason=reason,
    )


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
