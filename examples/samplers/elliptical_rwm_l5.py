"""Track L5 density-only baseline sampler (`requires = []`, strictly `n_gradient == 0`).

Implements an adaptive preconditioned Random-Walk Metropolis kernel with randomized
step-size scale mixture (Roberts & Rosenthal optimal scaling 2.38/sqrt(d) + small/large
mixture scales to handle heavy tails and curved ridges without gradient evaluations).
"""

from __future__ import annotations

import math

import numpy as np

requires: list[str] = []


def adapt(oracle, budget, rng):
    d = int(oracle.dim)
    x = rng.normal(0.0, 0.5, size=d)
    lp = float(oracle.log_density(x))
    for _ in range(32):
        if math.isfinite(lp):
            break
        x = rng.normal(0.0, 0.25, size=d)
        lp = float(oracle.log_density(x))
    if not math.isfinite(lp):
        x = np.zeros(d, dtype=np.float64)
        lp = float(oracle.log_density(x))

    n_warm = max(40, min(int(budget.draws) * 2, 320))
    scale = 2.38 / math.sqrt(max(1, d))
    diag_std = np.ones(d, dtype=np.float64)
    samples = []
    acc_ema = 0.234

    for step in range(n_warm):
        prop = x + (scale * diag_std) * rng.standard_normal(d)
        lp_prop = float(oracle.log_density(prop))
        accepted = False
        if math.isfinite(lp_prop):
            a = math.exp(min(0.0, lp_prop - lp))
            if rng.uniform() < a:
                x, lp = prop, lp_prop
                accepted = True
        acc_ema = 0.92 * acc_ema + 0.08 * (1.0 if accepted else 0.0)
        scale *= math.exp(0.08 * (acc_ema - 0.234))
        scale = float(np.clip(scale, 1e-4, 50.0))
        if step >= n_warm // 4:
            samples.append(x.copy())
        if step == n_warm // 2 and len(samples) >= max(12, d + 2):
            arr = np.asarray(samples, dtype=np.float64)
            mad = 1.4826 * np.median(np.abs(arr - np.median(arr, axis=0)), axis=0)
            diag_std = np.clip(mad, 1e-3, 1e3)

    if len(samples) >= max(20, 2 * d):
        arr = np.asarray(samples[len(samples) // 3 :], dtype=np.float64)
        med = np.median(arr, axis=0)
        mad = 1.4826 * np.median(np.abs(arr - med), axis=0)
        mad = np.clip(mad, 1e-3, 1e3)
        z = np.clip((arr - med) / mad, -5.0, 5.0)
        cov = np.cov(z, rowvar=False) if d > 1 else np.array([[float(np.var(z))]])
        cov = 0.85 * np.atleast_2d(cov) + 0.15 * np.eye(d)
        try:
            chol_z = np.linalg.cholesky(cov)
            chol = mad[:, None] * chol_z
        except np.linalg.LinAlgError:
            chol = np.diag(mad)
    else:
        chol = np.diag(diag_std)

    tuning = {
        "scale": float(scale),
        "chol": np.tril(chol),
    }
    return tuning, (x, None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    x = chain.position
    lp = float(oracle.log_density(x))
    if not math.isfinite(lp):
        return x, None
    chol = tuning["chol"]
    base_scale = float(tuning["scale"])
    # State-independent 3-scale symmetric mixture preserves detailed balance identically
    u = float(rng.uniform())
    mult = 0.35 if u < 0.2 else (2.2 if u > 0.85 else 1.0)
    step_scale = base_scale * mult
    d = x.size
    for _ in range(4):
        prop = x + step_scale * (chol @ rng.standard_normal(d))
        lp_prop = float(oracle.log_density(prop))
        if math.isfinite(lp_prop):
            if rng.uniform() < math.exp(min(0.0, lp_prop - lp)):
                x, lp = prop, lp_prop
    return x, None
