"""Lane A3: persistent-momentum GHMC (Horowitz) with whitened aux momentum.

Momentum z ~ N(0,I) lives in chain.aux (aux_spec = normal). Each transition:
partial refresh z' = cos(theta) z + sin(theta) xi, then a short leapfrog
trajectory with exact MH on the total energy, flipping z on rejection so
generalized detailed balance holds for pi x N(0,I). Metric + step size adapt
as in the diag-HMC baseline (windowed variance, dual averaging).
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.8
N_STEPS = 1
COS_THETA = 0.8
GAMMA, T0, KAPPA = 0.05, 10.0, 0.75


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


def _ghmc_step(grad, x: np.ndarray, lp: float, g: np.ndarray, z: np.ndarray,
               eps: float, n: int, inv_m: np.ndarray, rng: np.random.Generator):
    """One partial-refresh + trajectory + MH/flip GHMC update. Returns (x, lp, g, z, accept)."""
    # partial momentum refreshment (whitened z stays N(0,I) exactly)
    sin_t = math.sqrt(1.0 - COS_THETA * COS_THETA)
    z = COS_THETA * z + sin_t * rng.standard_normal(z.shape)
    h0 = -lp + 0.5 * float(np.dot(z, z))
    p = z / np.sqrt(inv_m)
    x1, lp1, g1 = x, lp, g
    ok = True
    for _ in range(n):
        p = p + (0.5 * eps) * g1
        x1 = x1 + eps * inv_m * p
        lp1, g1 = grad(x1)
        if not math.isfinite(lp1):
            ok = False
            break
        p = p + (0.5 * eps) * g1
    if ok:
        z1 = p * np.sqrt(inv_m)
        h1 = -lp1 + 0.5 * float(np.dot(z1, z1))
        dh = h0 - h1
        accept = 1.0 if dh >= 0 else math.exp(dh)
        if not math.isfinite(accept):
            accept = 0.0
        if rng.uniform() < accept:
            return x1, lp1, g1, z1, accept
        return x, lp, g, -z, accept
    return x, lp, g, -z, 0.0


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
    z = rng.standard_normal(oracle.dim)
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
        x, lp, g, z, accept = _ghmc_step(
            oracle.log_density_gradient, x, lp, g, z, step.eps, N_STEPS, inv_m, rng
        )
        step.update(accept)
        history[i] = x
    tuning = {"eps": float(step.final()), "inv_m": np.asarray(inv_m, dtype=float)}
    return tuning, (x, z), [1.0], {"kind": "normal"}


def transition(tuning, chain, oracle, rng):
    x = chain.position
    z = np.asarray(chain.aux, dtype=float)
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, -z
    x1, _, _, z1, _ = _ghmc_step(
        oracle.log_density_gradient, x, lp, g, z, float(tuning["eps"]), N_STEPS, tuning["inv_m"], rng
    )
    return x1, z1
