"""Plain HMC, an example submission for the Slice 2 adapt/transition kernel ABI (BENCH.md).

adapt(oracle, budget, rng) runs a warmup of `budget.draws` HMC transitions with dual averaging
(Hoffman & Gelman 2014) toward acceptance rate 0.8 and two diagonal-metric updates (at 30% and
70% of warmup), then freezes `tuning = {"eps": eps, "inv_m": inv_m}`.

transition(tuning, chain, oracle, rng) runs one stateless, reversible HMC transition with a
jittered leapfrog step count in [N/2, 3N/2], N = ceil(PATH_LENGTH / eps).
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.8
PATH_LENGTH = 2.0  # in units of metric-scaled posterior SD
MAX_STEPS = 256
GAMMA, T0, KAPPA = 0.05, 10.0, 0.75  # dual averaging constants from Stan


class StepSize:
    """Dual averaging of log step size."""

    def __init__(self, eps: float):
        self.mu, self.hbar, self.log_bar, self.t = math.log(10 * eps), 0.0, 0.0, 0
        self.eps = eps

    def update(self, accept: float) -> None:
        self.t += 1
        eta = 1 / (self.t + T0)
        self.hbar = (1 - eta) * self.hbar + eta * (TARGET_ACCEPT - accept)
        log_eps = self.mu - math.sqrt(self.t) / GAMMA * self.hbar
        w = self.t**-KAPPA
        self.log_bar = w * log_eps + (1 - w) * self.log_bar
        self.eps = math.exp(log_eps)

    def final(self) -> float:
        return math.exp(self.log_bar)


def _steps(eps: float, rng: np.random.Generator, jitter: bool = True) -> int:
    n = max(1, min(MAX_STEPS, math.ceil(PATH_LENGTH / eps)))
    return int(rng.integers(max(1, n // 2), n + n // 2 + 1)) if jitter else n


def _hmc_step(grad, x: np.ndarray, lp: float, g: np.ndarray, eps: float, n: int,
              inv_m: np.ndarray, rng: np.random.Generator):
    """One reversible HMC transition with n leapfrog steps."""
    p = rng.standard_normal(x.size) / np.sqrt(inv_m)
    h0 = lp - 0.5 * float(np.sum(inv_m * p * p))
    x1, g1, lp1 = x, g, lp
    p1 = p + 0.5 * eps * g1
    for i in range(n):
        x1 = x1 + eps * inv_m * p1
        lp1, g1 = grad(x1)
        if not math.isfinite(lp1):
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


def adapt(oracle, budget, rng):
    x, lp, g = _find_finite_init(oracle, rng)
    inv_m = np.ones(oracle.dim, dtype=float)
    warmup = max(10, int(budget.draws))
    updates = {int(0.3 * warmup): int(0.1 * warmup), int(0.7 * warmup): int(0.3 * warmup)}
    history = np.empty((warmup, oracle.dim), dtype=float)
    step = StepSize(0.1)
    for i in range(warmup):
        if i in updates:
            window = history[updates[i]:i]
            k = len(window)
            inv_m = (k / (k + 5)) * window.var(axis=0) + 1e-3 * (5 / (k + 5))
            step = StepSize(step.eps)
        x, lp, g, accept = _hmc_step(
            oracle.log_density_gradient, x, lp, g, step.eps, _steps(step.eps, rng), inv_m, rng
        )
        step.update(accept)
        history[i] = x
    tuning = {"eps": float(step.final()), "inv_m": np.asarray(inv_m, dtype=float)}
    return tuning, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, None
    eps = float(tuning["eps"])
    inv_m = tuning["inv_m"]
    x1, _, _, _ = _hmc_step(oracle.log_density_gradient, x, lp, g, eps, _steps(eps, rng), inv_m, rng)
    return x1, None
