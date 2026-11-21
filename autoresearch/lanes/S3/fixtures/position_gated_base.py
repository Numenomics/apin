"""S3 Rewind Base (BUGGY): radial position-dependent kernel gating breaks MH detailed balance."""

from __future__ import annotations

import math
import numpy as np

requires = ["grad"]


def adapt(oracle, budget, rng):
    d = int(oracle.dim)
    x = np.zeros(d, dtype=np.float64)
    lp, g = oracle.log_density_gradient(x)
    # Locate typical set center via short damped gradient ascent
    for _ in range(30):
        if not (math.isfinite(lp) and np.all(np.isfinite(g))):
            break
        gn = float(np.linalg.norm(g))
        step = g / max(1.0, gn)
        x_try = x + 0.5 * step
        lp_try, g_try = oracle.log_density_gradient(x_try)
        if math.isfinite(lp_try) and lp_try > lp:
            x, lp, g = x_try, lp_try, g_try
    return {"center": x.copy(), "r2_gate": float(d) * 3.0, "eps_outer": 0.4, "sigma_inner": 0.08}, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, None
    center = tuning["center"]
    # BUG: switching between fast HMC outside r2_gate and tiny RWM inside r2_gate
    # without accounting for reverse-kernel selection probability traps mass inside the core!
    if float(np.sum((x - center) ** 2)) > float(tuning["r2_gate"]):
        eps = float(tuning["eps_outer"])
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
    prop = x + float(tuning["sigma_inner"]) * rng.standard_normal(x.size)
    lp1 = float(oracle.log_density(prop))
    if math.isfinite(lp1) and rng.uniform() < math.exp(min(0.0, lp1 - lp)):
        return prop, None
    return x, None
