"""Lane A1: heavy-tail-robust dense covariance metric with dispersion-gated fallback.

Implements `adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)` for Lane A1's fixed
`harness/judge.py::a1_transition` kernel. Builds on the E1 line (arXiv:2603.18845 /
`codex/cards/fisher-low-rank.md`) with a full-rank dense correction (`inv_m_chol`):
  - Robust diagonal: 50/50 log-blend of sample variance and (1.4826*MAD)^2 scale — consistent
    under Gaussian marginals, bounded under StudentT(nu<=2) / funnel excursion tails.
  - Dense correction: ridge-shrunk draw covariance on MAD-whitened coordinates, eigenvalues
    clipped to [0.05, 20] and re-symmetrized, shipped as a lower-triangular Cholesky factor.
    Full rank removes the rank cap that left stiff low-variance directions (rotated
    IllCondGaussian kappa=50 at d=20, hybrid_rosenbrock residual coords) at marginal scale.
  - Guards funnels by monitoring gradient-norm dispersion (q95 / median) and falling back to
    the robust diagonal when gradient norms span > 1 order of magnitude (alpha_global < 0.5).
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.70
PATH_LENGTH = 0.1
MAX_STEPS = 256
GAMMA, T0, KAPPA = 0.05, 10.0, 0.75


class StepSize:
    """Dual averaging of log step size with safe clamping."""

    def __init__(self, eps: float):
        self.mu, self.hbar, self.log_bar, self.t = math.log(10.0 * eps), 0.0, 0.0, 0
        self.eps = eps

    def update(self, accept: float) -> None:
        self.t += 1
        eta = 1.0 / (self.t + T0)
        self.hbar = (1.0 - eta) * self.hbar + eta * (TARGET_ACCEPT - accept)
        log_eps = max(-20.0, min(5.0, self.mu - math.sqrt(self.t) / GAMMA * self.hbar))
        w = self.t**-KAPPA
        self.log_bar = w * log_eps + (1.0 - w) * self.log_bar
        self.eps = math.exp(log_eps)

    def final(self) -> float:
        return math.exp(max(-20.0, min(5.0, self.log_bar)))


def _steps(eps: float, rng: np.random.Generator, max_steps: int = MAX_STEPS) -> int:
    n = max(1, min(max_steps, math.ceil(PATH_LENGTH / eps)))
    return int(rng.integers(max(1, n // 2), n + n // 2 + 1))


def _apply_metric(inv_d: np.ndarray, chol: np.ndarray | None,
                  rng: np.random.Generator):
    if chol is None:
        p = rng.standard_normal(inv_d.size) / np.sqrt(inv_d)

        def apply_inv_m(m: np.ndarray) -> np.ndarray:
            return inv_d * m

        def kinetic(m: np.ndarray) -> float:
            return 0.5 * float(np.sum(inv_d * m * m))

        return p, apply_inv_m, kinetic

    z = rng.standard_normal(inv_d.size)
    p = np.linalg.solve(chol.T, z)

    def apply_inv_m(m: np.ndarray) -> np.ndarray:
        return chol @ (chol.T @ m)

    def kinetic(m: np.ndarray) -> float:
        w = chol.T @ m
        return 0.5 * float(w @ w)

    return p, apply_inv_m, kinetic


def _warmup_hmc_step(grad, x: np.ndarray, lp: float, g: np.ndarray, eps: float, n: int,
                     inv_d: np.ndarray, chol: np.ndarray | None,
                     rng: np.random.Generator):
    p, apply_inv_m, kinetic = _apply_metric(inv_d, chol, rng)
    h0 = lp - kinetic(p)
    x1, g1, lp1 = x, g, lp
    p1 = p + 0.5 * eps * g1
    for i in range(n):
        x1 = x1 + eps * apply_inv_m(p1)
        lp1, g1 = grad(x1)
        if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
            return x, lp, g, 0.0
        if i < n - 1:
            p1 = p1 + eps * g1
    p1 = p1 + 0.5 * eps * g1
    h1 = lp1 - kinetic(p1)
    accept = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
    if rng.uniform() < accept:
        return x1, lp1, g1, accept
    return x, lp, g, accept


def _estimate_fisher_metric(xs: np.ndarray, gs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Robust-whitened dense ridge-shrunk draw-covariance metric with dispersion guard."""
    k, d = xs.shape
    xc = xs - xs.mean(axis=0, keepdims=True)
    shrink_w = k / (k + 3.0)
    var_x = shrink_w * np.mean(xc * xc, axis=0) + 1e-3 * (1.0 - shrink_w)
    var_x = np.clip(var_x, 1e-6, 1e6)

    # Per-coordinate non-Gaussian curvature / funnel guard:
    # On any Gaussian (aligned or rotated), Var(x_j) * Var(g_j) >= 1 and -Corr(x_j, g_j) == 1 / sqrt(Var(x_j)*Var(g_j)).
    # On curved valleys (Rosenbrock/banana) or funnel necks, Var(g_j) is inflated by orthogonal curvature noise,
    # making sqrt(Var(x_j) * Var(g_j)) * (-Corr(x_j, g_j)) << 1 (or disp_ratio large).
    g_norms = np.linalg.norm(gs, axis=1)
    disp_ratio = float(np.percentile(g_norms, 95.0)) / max(float(np.median(g_norms)), 1e-12)
    alpha_global = 1.0 if disp_ratio <= 6.0 else max(0.0, (6.0 / disp_ratio) ** 2)

    # Robust-blend diagonal: half weight on the (1.4826*MAD)^2 scale, which is
    # consistent for sd under Gaussian marginals but stays bounded under the
    # infinite-variance tails of StudentT(nu<=2) and funnel v-multiplicity,
    # where a single excursion dominates the 21-42-draw sample variance.
    mad = np.median(np.abs(xc - np.median(xc, axis=0)), axis=0)
    robust_var = np.clip((mad * 1.4826) ** 2, 1e-6, 1e6)
    log_inv_d = 0.5 * (np.log(var_x) + np.log(robust_var))
    inv_d = np.clip(np.exp(log_inv_d), 1e-6, 1e6)

    # Skip the dense correction if too few samples (need k >= 1.5*d for a stable whitened
    # covariance: at d=20 the [0.2w, 0.5w) window has k=21 ~ d draws and S_w is MP-noise),
    # or strong funnel dispersion detected
    if k < max(12, 1.5 * d) or d < 2 or alpha_global < 0.5:
        return inv_d, None

    sigma = np.sqrt(inv_d)
    ys = xc / sigma[None, :]

    # Dense ridge-shrunk whitened draw covariance: MAD normalization makes the correlation
    # estimate robust to heavy-tail excursions, the full-rank basis removes the rank cap that
    # leaves stiff low-variance directions (rotated IllCondGaussian kappa=50 at d=20,
    # hybrid_rosenbrock residual coords) at marginal scale, and eigenvalue clipping bounds
    # finite-sample noise on isotropic targets (S_w ~ I -> correction ~ I).
    ridge = min(0.35, (d + 2.0) / (k + d + 2.0))
    s_w = (ys.T @ ys) / k
    s_reg = (1.0 - ridge) * s_w + ridge * np.eye(d)
    w, v = np.linalg.eigh(s_reg)
    w = np.clip(w, 0.05, 20.0)
    # Spectral damping toward identity: k/(k+d) mutes finite-sample eigenvalue noise
    # (a stiff direction whose variance is overestimated mid-warmup overshoots and stalls
    # mixing), scaling ALL corrections down proportionally instead of dropping directions.
    gamma = k / (k + float(d))
    w = w**gamma
    s_reg = (v * w) @ v.T
    s_reg = 0.5 * (s_reg + s_reg.T)
    try:
        chol_w = np.linalg.cholesky(s_reg)
    except np.linalg.LinAlgError:
        return inv_d, None
    chol = sigma[:, None] * chol_w
    return inv_d, chol


def _find_finite_init(oracle, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    for _ in range(100):
        x = rng.uniform(-1.0, 1.0, size=oracle.dim)
        lp, g = oracle.log_density_gradient(x)
        if math.isfinite(lp) and np.all(np.isfinite(g)):
            return x, lp, g
    x = np.zeros(oracle.dim, dtype=float)
    lp, g = oracle.log_density_gradient(x)
    return x, lp, g


def adapt_metric(oracle, budget, rng):
    """Estimate robust dense-covariance metric + step size via dual averaging."""
    x, lp, g = _find_finite_init(oracle, rng)
    inv_d = np.ones(oracle.dim, dtype=float)
    chol: np.ndarray | None = None
    warmup = max(10, min(1000, int(0.7 * budget.draws)))
    updates = {int(0.5 * warmup): int(0.2 * warmup), int(0.75 * warmup): int(0.2 * warmup)}
    hist_x = np.empty((warmup, oracle.dim), dtype=float)
    hist_g = np.empty((warmup, oracle.dim), dtype=float)
    step = StepSize(0.2)
    for i in range(warmup):
        if i in updates:
            s0 = updates[i]
            inv_d, chol = _estimate_fisher_metric(hist_x[s0:i], hist_g[s0:i])
        x, lp, g, accept = _warmup_hmc_step(
            oracle.log_density_gradient, x, lp, g, step.eps, _steps(step.eps, rng, max_steps=12),
            inv_d, chol, rng,
        )
        step.update(accept)
        hist_x[i] = x
        hist_g[i] = g
    eps_final = float(min(5.0, max(1e-6, step.final())))
    if chol is None:
        return {"eps": eps_final, "inv_m": np.asarray(inv_d, dtype=float)}, x
    return {"eps": eps_final, "inv_m_chol": np.asarray(chol, dtype=float)}, x
