"""S3 Rewind Reference (FIXED): Metropolis-Adjusted Langevin (MALA) preserves exact invariance."""

from __future__ import annotations

import math
import numpy as np

requires = ["grad"]


def adapt(oracle, budget, rng):
    d = int(oracle.dim)
    x = np.zeros(d, dtype=np.float64)
    oracle.log_density_gradient(x)
    return {"eps": 0.35}, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, None
    eps = float(tuning["eps"])
    step2 = eps * eps
    mean_f = x + 0.5 * step2 * g
    x1 = mean_f + eps * rng.standard_normal(x.size)
    lp1, g1 = oracle.log_density_gradient(x1)
    if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
        return x, None
    mean_b = x1 + 0.5 * step2 * g1
    log_q_fwd = -0.5 * float(np.sum((x1 - mean_f) ** 2)) / step2
    log_q_rev = -0.5 * float(np.sum((x - mean_b) ** 2)) / step2
    log_a = (lp1 + log_q_rev) - (lp + log_q_fwd)
    if math.isfinite(log_a) and rng.uniform() < math.exp(min(0.0, log_a)):
        return x1, None
    return x, None
