"""A deliberately wrong submission: the unadjusted Langevin algorithm with a large step.

ULA moves x to x + h M grad log p(x) + sqrt(2 h M) z and never rejects, so
its draws come from a distribution that differs from the target by an amount
that grows with h. On a standard normal, for example, its stationary variance
is 1 / (1 - h / 2). This file exists so the tests and the score table can
show that the correctness gate catches such a sampler.

Warmup is MALA, the corrected version: each chain tunes h towards an
acceptance rate of TARGET_ACCEPT and estimates a diagonal metric M from its
warmup draws. The chains then share the smallest h and the average M, so that they all
target the same wrong distribution, and sampling accepts every move.
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.5


def sample(problem, budget, rng):
    grad = problem.log_density_gradient
    warm = [warmup(grad, x0, budget.draws, rng) for x0 in problem.initial_points]
    log_eps = float(np.min([w[3] for w in warm]))
    inv_m = np.mean([w[4] for w in warm], axis=0)
    out = np.empty((budget.chains, budget.draws, problem.dim))
    for c, (x, lp, g, _, _) in enumerate(warm):
        for i in range(budget.draws):
            x, lp, g, _ = langevin(grad, x, lp, g, math.exp(log_eps), inv_m, rng)
            out[c, i] = x
    return out


def langevin(grad, x, lp, g, eps, inv_m, rng):
    """One leapfrog step with fresh momentum, which is one Langevin step with h = eps^2 / 2.
    Returns the proposal and the probability with which MALA would accept it."""
    p = rng.standard_normal(x.size) / np.sqrt(inv_m)
    h0 = lp - 0.5 * np.sum(inv_m * p * p)
    p1 = p + 0.5 * eps * g
    x1 = x + eps * inv_m * p1
    lp1, g1 = grad(x1)
    p1 = p1 + 0.5 * eps * g1
    h1 = lp1 - 0.5 * np.sum(inv_m * p1 * p1)
    accept = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
    return x1, lp1, g1, accept


def warmup(grad, x0, n, rng):
    x = np.array(x0, dtype=float)
    lp, g = grad(x)
    inv_m = np.ones(x.size)
    log_eps = math.log(0.1)
    history = np.empty((n, x.size))
    for i in range(n):
        if i == n // 2:
            inv_m = history[n // 4:i].var(axis=0) + 1e-3
        x1, lp1, g1, accept = langevin(grad, x, lp, g, math.exp(log_eps), inv_m, rng)
        if rng.uniform() < accept:
            x, lp, g = x1, lp1, g1
        log_eps += 0.05 * (accept - TARGET_ACCEPT)  # Robbins-Monro on the acceptance rate
        history[i] = x
    return x, lp, g, log_eps, inv_m
