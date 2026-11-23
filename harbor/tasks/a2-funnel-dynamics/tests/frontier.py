"""Lane A2: persistent-momentum generalized HMC (Horowitz, exact on joint law) + Barker fallback.

chain.aux = normalized momentum u ~ N(0,I) (aux_spec normal). Each transition:
partial refresh u' = cos_phi*u + sin_phi*z, reversible leapfrog at eps, MH on
joint Hamiltonian, momentum negation on rejection. Persistent momentum suppresses
random-walk backtracking along narrow Rosenbrock/funnel ridges at identical
gradient cost to fully-refreshed HMC. With prob P_BARKER an exact skew-symmetric
Barker gradient step (state-independent coin, aux untouched) targets heavy tails
and funnel walls. Step count jittered as baseline.
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.8
PATH_LENGTH = 2.0
MAX_STEPS = 256
GAMMA, T0, KAPPA = 0.05, 10.0, 0.75
COS_PHI_PERSIST = 0.9
SIN_PHI_PERSIST = math.sqrt(1.0 - COS_PHI_PERSIST * COS_PHI_PERSIST)
P_FULL_REFRESH = 0.5
P_BARKER = 0.3
BARKER_SIGMA_SCALE = 2.0
AUX_SPEC = {"kind": "normal"}


class StepSize:
    """Dual averaging of log step size."""

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


def _steps(eps: float, rng: np.random.Generator) -> int:
    n = max(1, min(MAX_STEPS, math.ceil(PATH_LENGTH / eps)))
    return int(rng.integers(max(1, n // 2), n + n // 2 + 1))


def _ghmc_update(grad, x: np.ndarray, u: np.ndarray, lp: float, g: np.ndarray,
                 eps: float, n: int, inv_m: np.ndarray, rng: np.random.Generator):
    """One exact GHMC update. Returns (x_next, u_next, accept_prob)."""
    s = np.sqrt(inv_m)
    if rng.uniform() < P_FULL_REFRESH:
        ur = rng.standard_normal(x.size)
    else:
        ur = COS_PHI_PERSIST * u + SIN_PHI_PERSIST * rng.standard_normal(x.size)
    h0 = lp - 0.5 * float(np.sum(ur * ur))
    p1 = ur / s + 0.5 * eps * g
    x1, g1, lp1 = x, g, lp
    ok = True
    for i in range(n):
        x1 = x1 + eps * inv_m * p1
        lp1, g1 = grad(x1)
        if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
            ok = False
            break
        if i < n - 1:
            p1 = p1 + eps * g1
    if not ok:
        return x, -ur, 0.0
    p1 = p1 + 0.5 * eps * g1
    u1 = p1 * s
    h1 = lp1 - 0.5 * float(np.sum(u1 * u1))
    accept = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
    if rng.uniform() < accept:
        return x1, u1, accept
    return x, -ur, accept


def _barker_step(grad, x: np.ndarray, lp: float, g: np.ndarray,
                 sigma: np.ndarray, rng: np.random.Generator):
    """Skew-symmetric Barker proposal; exact MH, leaves aux untouched."""
    logit = np.clip(sigma * g, -20.0, 20.0)
    p_plus = 1.0 / (1.0 + np.exp(-logit))
    plus = rng.uniform(size=x.size) < p_plus
    y = x + np.where(plus, sigma, -sigma)
    lp_y, g_y = grad(y)
    if not (math.isfinite(lp_y) and np.all(np.isfinite(g_y))):
        return x, lp, g, 0.0
    logit_y = np.clip(sigma * g_y, -20.0, 20.0)
    p_plus_y = 1.0 / (1.0 + np.exp(-logit_y))
    q_fwd = np.where(plus, p_plus, 1.0 - p_plus)
    q_rev = np.where(plus, 1.0 - p_plus_y, p_plus_y)
    log_a = (lp_y - lp) + float(np.sum(np.log(q_rev) - np.log(q_fwd)))
    a = math.exp(min(0.0, log_a)) if math.isfinite(log_a) else 0.0
    if rng.uniform() < a:
        return y, lp_y, g_y, a
    return x, lp, g, a


def _find_finite_init(oracle, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    for _ in range(100):
        x = rng.uniform(-1.0, 1.0, size=oracle.dim)
        lp, g = oracle.log_density_gradient(x)
        if math.isfinite(lp) and np.all(np.isfinite(g)):
            return x, lp, g
    x = np.zeros(oracle.dim, dtype=float)
    lp, g = oracle.log_density_gradient(x)
    return x, lp, g


def adapt(oracle, budget, rng):
    x, lp, g = _find_finite_init(oracle, rng)
    u = rng.standard_normal(oracle.dim)
    inv_m = np.ones(oracle.dim, dtype=float)
    warmup = max(10, min(1000, int(budget.draws)))
    updates = {int(0.3 * warmup): int(0.1 * warmup), int(0.7 * warmup): int(0.3 * warmup)}
    history = np.empty((warmup, oracle.dim), dtype=float)
    step = StepSize(0.1)
    for i in range(warmup):
        if i in updates:
            window = history[updates[i]:i]
            k = len(window)
            inv_m = np.clip((k / (k + 5.0)) * window.var(axis=0) + 1e-3 * (5.0 / (k + 5.0)), 1e-6, 1e6)
            step = StepSize(step.eps)
        x, u, accept = _ghmc_update(
            oracle.log_density_gradient, x, u, lp, g, step.eps, _steps(step.eps, rng), inv_m, rng
        )
        lp, g = oracle.log_density_gradient(x)
        step.update(accept)
        history[i] = x
    tuning = {"eps": float(step.final()), "inv_m": np.asarray(inv_m, dtype=float)}
    return tuning, (x, u), [1.0], dict(AUX_SPEC)


def transition(tuning, chain, oracle, rng):
    x = np.asarray(chain.position, dtype=float)
    u = np.asarray(chain.aux, dtype=float)
    if u.shape != x.shape:
        u = rng.standard_normal(x.size)
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, -u
    eps = float(tuning["eps"])
    inv_m = tuning["inv_m"]
    if rng.uniform() < P_BARKER:
        sigma = np.sqrt(inv_m) * eps * BARKER_SIGMA_SCALE
        x1, _, _, _ = _barker_step(oracle.log_density_gradient, x, lp, g, sigma, rng)
        return x1, u
    x1, u1, _ = _ghmc_update(oracle.log_density_gradient, x, u, lp, g, eps, _steps(eps, rng), inv_m, rng)
    return x1, u1
