"""Lane A0 ground-up starter: minimal isotropic random-walk Metropolis (no priors).

Implements the full kernel-v2 ABI (`adapt` + `transition`) from a clean slate with zero
pre-baked HMC, NUTS, or Fisher boilerplate. Replace or evolve any part of `adapt()` and
`transition()` while preserving exact-draw invariance and bit-exact replay.
"""

from __future__ import annotations

import math

import numpy as np


def _find_finite_init(oracle, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    for _ in range(100):
        x = rng.uniform(-1.0, 1.0, size=oracle.dim)
        lp = oracle.log_density(x)
        if math.isfinite(lp):
            return x, lp
    x = np.zeros(oracle.dim, dtype=float)
    return x, oracle.log_density(x)


def adapt(oracle, budget, rng):
    """Minimal step-scale tuning for isotropic random-walk Metropolis."""
    x, lp = _find_finite_init(oracle, rng)
    log_scale = math.log(2.38 / math.sqrt(max(1, oracle.dim)))
    warmup = max(20, min(400, int(budget.draws)))
    for i in range(1, warmup + 1):
        scale = math.exp(max(-12.0, min(4.0, log_scale)))
        y = x + scale * rng.standard_normal(oracle.dim)
        lpy = oracle.log_density(y)
        acc = math.exp(min(0.0, lpy - lp)) if math.isfinite(lpy) else 0.0
        if rng.uniform() < acc:
            x, lp = y, lpy
        log_scale += (acc - 0.234) / math.sqrt(i + 10.0)
    scale = math.exp(max(-12.0, min(4.0, log_scale)))
    return {"scale": float(scale)}, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    """One reversible isotropic random-walk Metropolis transition."""
    x = chain.position
    lp = oracle.log_density(x)
    if not math.isfinite(lp):
        return x, None
    scale = float(tuning["scale"])
    y = x + scale * rng.standard_normal(x.size)
    lpy = oracle.log_density(y)
    if math.isfinite(lpy) and math.log(rng.uniform()) < min(0.0, lpy - lp):
        return y, None
    return x, None
