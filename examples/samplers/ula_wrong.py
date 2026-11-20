"""A deliberately biased submission: unadjusted Langevin (ULA) with a large step size.

adapt() runs MALA warmup to tune step size h toward TARGET_ACCEPT and estimate a diagonal metric M,
freezing tuning = {"h": h, "inv_m": inv_m}.

transition() runs unadjusted Langevin (accepting every finite move without Metropolis correction),
so its stationary distribution differs from the target by an amount growing with h.
This example exists to verify that the benchmark gate catches biased kernels.
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.5


def _log_q(y: np.ndarray, x: np.ndarray, g: np.ndarray, h: float, inv_m: np.ndarray) -> float:
    diff = y - (x + 0.5 * h * inv_m * g)
    return -0.5 * float(np.sum(diff * diff / (h * inv_m)))


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
    h = 0.05
    warmup = max(20, min(1000, int(budget.draws)))
    history = np.empty((warmup, oracle.dim), dtype=float)
    for i in range(warmup):
        if i in (int(0.3 * warmup), int(0.7 * warmup)):
            window = history[max(0, i - warmup // 3):i]
            k = len(window)
            inv_m = (k / (k + 5)) * window.var(axis=0) + 1e-3 * (5 / (k + 5))
        y = x + 0.5 * h * inv_m * g + np.sqrt(h * inv_m) * rng.standard_normal(x.size)
        lpy, gy = oracle.log_density_gradient(y)
        if math.isfinite(lpy) and np.all(np.isfinite(gy)):
            log_a = (lpy + _log_q(x, y, gy, h, inv_m)) - (lp + _log_q(y, x, g, h, inv_m))
            acc = math.exp(min(0.0, log_a)) if math.isfinite(log_a) else 0.0
        else:
            acc = 0.0
        if rng.uniform() < acc:
            x, lp, g = y, lpy, gy
        h = float(np.clip(h * math.exp(0.05 * (acc - TARGET_ACCEPT)), 1e-6, 1.5))
        history[i] = x
    tuning = {"h": float(max(h, 0.25)), "inv_m": np.asarray(inv_m, dtype=float)}
    return tuning, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    h = float(tuning["h"])
    inv_m = tuning["inv_m"]
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, None
    y = x + 0.5 * h * inv_m * g + np.sqrt(h * inv_m) * rng.standard_normal(x.size)
    lpy = oracle.log_density(y)
    if math.isfinite(lpy):
        return y, None
    return x, None
