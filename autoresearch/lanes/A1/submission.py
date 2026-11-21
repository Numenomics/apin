"""Lane A1 Experiment E1: regularized low-rank Fisher geometric-mean metric with funnel guard.

Implements `adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)` for Lane A1's fixed
`harness/judge.py::a1_transition` kernel. Follows Seyboldt, Carlson & Carpenter
(arXiv:2603.18845 / `codex/cards/fisher-low-rank.md`):
  - Centers both draws and scores before whitening by sigma = sqrt(inv_m_diag)
    (correcting walnutpie W-63's uncentered / inv_diag / positive-only eigenvalue bugs).
  - Computes the exact Riemannian/Fisher matrix geometric mean A^{1/2}(A^{1/2} B A^{1/2})^{-1/2} A^{1/2}
    in the top-k whitened subspace with ridge shrinkage and log-eigenvalue soft-thresholding.
  - Guards funnels by monitoring gradient-norm dispersion (q95 / median) and smoothly falling
    back to variance-damped diagonal when gradient norms span > 1 order of magnitude.
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.8
PATH_LENGTH = 2.0
MAX_STEPS = 256
GAMMA, T0, KAPPA = 0.05, 10.0, 0.75
MAX_RANK = 10


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


def _apply_metric(inv_d: np.ndarray, u: np.ndarray | None, eigs: np.ndarray | None,
                  rng: np.random.Generator):
    if u is None or eigs is None:
        p = rng.standard_normal(inv_d.size) / np.sqrt(inv_d)

        def apply_inv_m(m: np.ndarray) -> np.ndarray:
            return inv_d * m

        def kinetic(m: np.ndarray) -> float:
            return 0.5 * float(np.sum(inv_d * m * m))

        return p, apply_inv_m, kinetic

    sqrt_d = np.sqrt(inv_d)
    z = rng.standard_normal(inv_d.size)
    inv_sqrt_delta = (1.0 / np.sqrt(eigs)) - 1.0
    p = (z + u @ (inv_sqrt_delta * (u.T @ z))) / sqrt_d
    eig_delta = eigs - 1.0

    def apply_inv_m(m: np.ndarray) -> np.ndarray:
        w = sqrt_d * m
        return sqrt_d * (w + u @ (eig_delta * (u.T @ w)))

    def kinetic(m: np.ndarray) -> float:
        return 0.5 * float(m @ apply_inv_m(m))

    return p, apply_inv_m, kinetic


def _warmup_hmc_step(grad, x: np.ndarray, lp: float, g: np.ndarray, eps: float, n: int,
                     inv_d: np.ndarray, u: np.ndarray | None, eigs: np.ndarray | None,
                     rng: np.random.Generator):
    p, apply_inv_m, kinetic = _apply_metric(inv_d, u, eigs, rng)
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


def _estimate_fisher_metric(xs: np.ndarray, gs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Centered, whitened low-rank Fisher geometric-mean metric with shrinkage & funnel guard."""
    k, d = xs.shape
    xc = xs - xs.mean(axis=0, keepdims=True)
    gc = gs - gs.mean(axis=0, keepdims=True)
    shrink_w = k / (k + 5.0)
    var_x = shrink_w * np.mean(xc * xc, axis=0) + 1e-3 * (1.0 - shrink_w)
    var_g = shrink_w * np.mean(gc * gc, axis=0) + 1e-3 * (1.0 - shrink_w)
    var_x = np.clip(var_x, 1e-6, 1e6)
    var_g = np.clip(var_g, 1e-6, 1e6)

    # Per-coordinate non-Gaussian curvature / funnel guard:
    # On any Gaussian (aligned or rotated), Var(x_j) * Var(g_j) >= 1 and -Corr(x_j, g_j) == 1 / sqrt(Var(x_j)*Var(g_j)).
    # On curved valleys (Rosenbrock/banana) or funnel necks, Var(g_j) is inflated by orthogonal curvature noise,
    # making sqrt(Var(x_j) * Var(g_j)) * (-Corr(x_j, g_j)) << 1 (or disp_ratio large).
    g_norms = np.linalg.norm(gs, axis=1)
    disp_ratio = float(np.percentile(g_norms, 95.0)) / max(float(np.median(g_norms)), 1e-12)
    alpha_global = 1.0 if disp_ratio <= 6.0 else max(0.0, (6.0 / disp_ratio) ** 2)

    cov_xg = -np.mean(xc * gc, axis=0)
    # Exact identity on any Gaussian: cov_xg == 1.0 (Stein's identity E[-x_j g_j] = 1).
    # When Var(g_j) spikes from nonlinear curvature, |cov_xg / sqrt(var_x * var_g)| drops well below 1/sqrt(vx*vg)
    # and var_x / (1/var_g) = var_x * var_g >> 1. Blend smoothly toward draw variance var_x when var_x * var_g is huge.
    vg_excess = np.maximum(1.0, var_x * var_g)
    stein_ok = np.clip(cov_xg, 0.0, 2.0) / np.maximum(1.0, cov_xg)
    alpha_coord = alpha_global * stein_ok / np.sqrt(np.maximum(1.0, vg_excess / 4.0))
    alpha_coord = np.clip(alpha_coord, 0.0, 1.0)

    log_inv_d = (1.0 - 0.5 * alpha_coord) * np.log(var_x) - (0.5 * alpha_coord) * np.log(var_g)
    inv_d = np.clip(np.exp(log_inv_d), 1e-6, 1e6)

    # Skip low-rank update if too few samples or strong funnel dispersion detected
    if k < 12 or d < 2 or alpha_global < 0.5:
        return inv_d, None, None

    sigma = np.sqrt(inv_d)
    ys = xc / sigma[None, :]
    ss = gc * sigma[None, :]

    # Orthonormal basis Q for top-r subspace of centered, whitened [Ys, Ss]
    r = min(d, MAX_RANK, max(1, k // 3))
    stacked = np.concatenate([ys.T, ss.T], axis=1) / math.sqrt(k)  # (d, 2k)
    q_full, _, _ = np.linalg.svd(stacked, full_matrices=False)
    q = q_full[:, :r]

    # Ridge-shrunk projected draw covariance A and score covariance B
    ridge = min(0.35, (r + 2.0) / (k + r + 2.0))
    ay = ys @ q  # (k, r)
    bs = ss @ q  # (k, r)
    a_mat = (1.0 - ridge) * ((ay.T @ ay) / k) + ridge * np.eye(r)
    b_mat = (1.0 - ridge) * ((bs.T @ bs) / k) + ridge * np.eye(r)
    a_mat = 0.5 * (a_mat + a_mat.T)
    b_mat = 0.5 * (b_mat + b_mat.T)

    # Exact Riemannian / Fisher geometric mean: A^{1/2} (A^{1/2} B A^{1/2})^{-1/2} A^{1/2}
    ea, va = np.linalg.eigh(a_mat)
    ea = np.clip(ea, 1e-4, 1e4)
    a_half = (va * np.sqrt(ea)[None, :]) @ va.T
    w_mat = a_half @ b_mat @ a_half
    w_mat = 0.5 * (w_mat + w_mat.T)
    ew, vw = np.linalg.eigh(w_mat)
    ew = np.clip(ew, 1e-4, 1e4)
    w_inv_half = (vw * (1.0 / np.sqrt(ew))[None, :]) @ vw.T
    m_sub_inv = a_half @ w_inv_half @ a_half
    m_sub_inv = 0.5 * (m_sub_inv + m_sub_inv.T)

    eigs, p_mat = np.linalg.eigh(m_sub_inv)
    u_all = q @ p_mat  # (d, r) orthonormal candidate modes

    # Directional Fisher-consistency gate: on linear/Gaussian correlations, -Corr(u^T y, u^T s) ~ 1
    # and Var(u^T y) * Var(u^T s) ~ 1. On curved ridges (banana/Rosenbrock), scores are nonlinear
    # in position so v_y * v_s >> 1 or |rho| << 1; damp those modes smoothly back toward 1.0.
    uy = ys @ u_all  # (k, r)
    us = ss @ u_all  # (k, r)
    vy = np.mean(uy * uy, axis=0) + 1e-12
    vs = np.mean(us * us, axis=0) + 1e-12
    cov_ys = -np.mean(uy * us, axis=0)
    rho = np.clip(cov_ys / np.sqrt(vy * vs), 0.0, 1.0)
    mismatch = np.maximum(1.0, np.sqrt(vy * vs))
    consistency = (rho ** 2) / mismatch

    # Soft-threshold gated log-eigenvalues toward 0 to suppress finite-sample noise
    log_e = np.log(np.clip(eigs, 0.05, 20.0)) * consistency
    tau = 0.25 * math.sqrt(r / max(k, 1))
    shrunk = np.sign(log_e) * np.maximum(0.0, np.abs(log_e) - tau)
    keep = np.abs(shrunk) > 0.15
    if not np.any(keep):
        return inv_d, None, None

    u_kept = u_all[:, keep]
    # Re-orthonormalize via QR for numerical exactness (U^T U = I)
    u_kept, _ = np.linalg.qr(u_kept, mode="reduced")
    eigs_kept = np.clip(np.exp(shrunk[keep]), 0.05, 20.0)
    return inv_d, u_kept, eigs_kept


def _find_finite_init(oracle, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    for _ in range(100):
        x = rng.uniform(-2.0, 2.0, size=oracle.dim)
        lp, g = oracle.log_density_gradient(x)
        if math.isfinite(lp) and np.all(np.isfinite(g)):
            return x, lp, g
    x = np.zeros(oracle.dim, dtype=float)
    lp, g = oracle.log_density_gradient(x)
    return x, lp, g


def adapt_metric(oracle, budget, rng):
    """Estimate regularized low-rank Fisher geometric-mean metric + step size via dual averaging."""
    x, lp, g = _find_finite_init(oracle, rng)
    inv_d = np.ones(oracle.dim, dtype=float)
    u: np.ndarray | None = None
    eigs: np.ndarray | None = None
    warmup = max(10, min(1000, int(budget.draws)))
    updates = {int(0.3 * warmup): int(0.1 * warmup), int(0.7 * warmup): int(0.3 * warmup)}
    hist_x = np.empty((warmup, oracle.dim), dtype=float)
    hist_g = np.empty((warmup, oracle.dim), dtype=float)
    step = StepSize(0.1)
    for i in range(warmup):
        if i in updates:
            s0 = updates[i]
            inv_d, u, eigs = _estimate_fisher_metric(hist_x[s0:i], hist_g[s0:i])
            step = StepSize(step.eps)
        w_cap = 24 if i < int(0.3 * warmup) else (48 if i < int(0.7 * warmup) else 96)
        x, lp, g, accept = _warmup_hmc_step(
            oracle.log_density_gradient, x, lp, g, step.eps, _steps(step.eps, rng, max_steps=w_cap),
            inv_d, u, eigs, rng,
        )
        step.update(accept)
        hist_x[i] = x
        hist_g[i] = g
    eps_final = float(min(5.0, max(1e-6, step.final())))
    if u is None or eigs is None:
        return {"eps": eps_final, "inv_m": np.asarray(inv_d, dtype=float)}, x
    return {
        "eps": eps_final,
        "inv_m_diag": np.asarray(inv_d, dtype=float),
        "low_rank_u": np.asarray(u, dtype=float),
        "low_rank_eigs": np.asarray(eigs, dtype=float),
    }, x
