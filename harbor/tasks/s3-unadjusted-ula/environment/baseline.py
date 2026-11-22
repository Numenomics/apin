"""S3 Rewind Base (BUGGY): Unadjusted Langevin (ULA) without Metropolis-Hastings correction."""

from __future__ import annotations

import math
import numpy as np

requires = ["grad"]


def adapt(oracle, budget, rng):
    d = int(oracle.dim)
    x = np.zeros(d, dtype=np.float64)
    oracle.log_density_gradient(x)
    return {"eps": 1.65}, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    eps = float(tuning["eps"])
    # BUG: unadjusted ULA steps (always accepts!) have severe O(eps^2) stationary variance bias
    for _ in range(3):
        lp, g = oracle.log_density_gradient(x)
        if not (math.isfinite(lp) and np.all(np.isfinite(g))):
            return x, None
        x1 = x + 0.5 * eps * eps * g + eps * rng.standard_normal(x.size)
        lp1 = oracle.log_density(x1)
        if math.isfinite(lp1):
            x = x1
    return x, None
