"""Lane A1 baseline submission: diagonal metric estimation with dual averaging.

Implements the narrow Lane A1 contract (`autoresearch/lanes/A1/program.md`):
    adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)

The transition kernel is owned and frozen by `harness/judge.py` (`a1_transition`).
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.8
PATH_LENGTH = 2.0
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


def _steps(eps: float, rng: np.random.Generator) -> int:
    n = max(1, min(MAX_STEPS, math.ceil(PATH_LENGTH / eps)))
    return int(rng.integers(max(1, n // 2), n + n // 2 + 1))


def _warmup_hmc_step(grad, x: np.ndarray, lp: float, g: np.ndarray, eps: float, n: int,
                     inv_m: np.ndarray, rng: np.random.Generator):
    p = rng.standard_normal(x.size) / np.sqrt(inv_m)
    h0 = lp - 0.5 * float(np.sum(inv_m * p * p))
    x1, g1, lp1 = x, g, lp
    p1 = p + 0.5 * eps * g1
    for i in range(n):
        x1 = x1 + eps * inv_m * p1
        lp1, g1 = grad(x1)
        if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
            return x, lp, g, 0.0
        if i < n - 1:
            p1 = p1 + eps * g1
    p1 = p1 + 0.5 * eps * g1
    h1 = lp1 - 0.5 * float(np.sum(inv_m * p1 * p1))
    accept = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
    if rng.uniform() < accept:
        return x1, lp1, g1, accept
    return x, lp, g, accept


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
    """Estimate diagonal inverse mass matrix and step size via dual averaging."""
    x, lp, g = _find_finite_init(oracle, rng)
    inv_m = np.ones(oracle.dim, dtype=float)
    warmup = max(10, min(1000, int(budget.draws)))
    updates = {int(0.3 * warmup): int(0.1 * warmup), int(0.7 * warmup): int(0.3 * warmup)}
    history = np.empty((warmup, oracle.dim), dtype=float)
    step = StepSize(0.1)
    for i in range(warmup):
        if i in updates:
            window = history[updates[i]:i]
            k = len(window)
            inv_m = (k / (k + 5.0)) * window.var(axis=0) + 1e-3 * (5.0 / (k + 5.0))
            inv_m = np.clip(inv_m, 1e-6, 1e6)
            step = StepSize(step.eps)
        x, lp, g, accept = _warmup_hmc_step(
            oracle.log_density_gradient, x, lp, g, step.eps, _steps(step.eps, rng), inv_m, rng
        )
        step.update(accept)
        history[i] = x
    eps_final = float(min(5.0, max(1e-6, step.final())))
    return {"eps": eps_final, "inv_m": np.asarray(inv_m, dtype=float)}, x
