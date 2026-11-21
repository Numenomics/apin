"""S3 Rewind Reference (FIXED): state-independent coin mixture preserves detailed balance."""

from __future__ import annotations

import math
import numpy as np

requires = ["grad"]


def adapt(oracle, budget, rng):
    d = int(oracle.dim)
    x = np.zeros(d, dtype=np.float64)
    oracle.log_density_gradient(x)
    return {"eps": 0.25, "rwm_sigma": 0.8}, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, None
    # FIX: state-independent coin guarantees exact mixture invariance
    if rng.uniform() < 0.35:
        prop = x + float(tuning["rwm_sigma"]) * rng.standard_normal(x.size)
        lp1 = oracle.log_density(prop)
        if math.isfinite(lp1) and rng.uniform() < math.exp(min(0.0, lp1 - lp)):
            return prop, None
        return x, None
    eps = float(tuning["eps"])
    p = rng.standard_normal(x.size)
    h0 = lp - 0.5 * float(p @ p)
    x1, lp1, g1 = x, lp, g
    p1 = p + 0.5 * eps * g1
    for i in range(4):
        x1 = x1 + eps * p1
        lp1, g1 = oracle.log_density_gradient(x1)
        if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
            return x, None
        if i < 3:
            p1 = p1 + eps * g1
    p1 = p1 + 0.5 * eps * g1
    h1 = lp1 - 0.5 * float(p1 @ p1)
    if math.isfinite(h1) and rng.uniform() < math.exp(min(0.0, h1 - h0)):
        return x1, None
    return x, None
