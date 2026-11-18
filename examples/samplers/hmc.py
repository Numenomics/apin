"""Plain HMC, an example submission for the logdensity adapter (BENCH.md).

Each chain runs a warmup of budget.draws iterations, then budget.draws
sampling iterations with a fixed step size and a diagonal metric.

Warmup: the step size follows dual averaging (Hoffman and Gelman 2014,
section 3.2) towards an acceptance rate of 0.8. The metric is set twice from
the variance of the warmup draws so far, at 30% and 70% of warmup, as in
Stan's windows but shorter. Dual averaging restarts after each update.

Sampling: the number of leapfrog steps is drawn uniformly from
[N/2, 3N/2], N = PATH_LENGTH / step size, so the path length is not fixed
(a fixed path length can be periodic on a Gaussian).
"""

from __future__ import annotations

import math

import numpy as np

TARGET_ACCEPT = 0.8
PATH_LENGTH = 2.0  # in units of the metric-scaled posterior standard deviation
MAX_STEPS = 256
GAMMA, T0, KAPPA = 0.05, 10.0, 0.75  # dual averaging constants from Stan


def sample(problem, budget, rng):
    out = np.empty((budget.chains, budget.draws, problem.dim))
    for c in range(budget.chains):
        out[c] = run_chain(problem.log_density_gradient, problem.initial_points[c], budget.draws, rng)
    return out


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


def transition(grad, x, lp, g, eps, n, inv_m, rng):
    """One HMC transition with n leapfrog steps. Returns the new state and the acceptance probability."""
    p = rng.standard_normal(x.size) / np.sqrt(inv_m)
    h0 = lp - 0.5 * np.sum(inv_m * p * p)
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
    h1 = lp1 - 0.5 * np.sum(inv_m * p1 * p1)
    accept = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
    if rng.uniform() < accept:
        return x1, lp1, g1, accept
    return x, lp, g, accept


def steps(eps: float, rng, jitter: bool = True) -> int:
    n = max(1, min(MAX_STEPS, math.ceil(PATH_LENGTH / eps)))
    return int(rng.integers(max(1, n // 2), n + n // 2 + 1)) if jitter else n


def run_chain(grad, x0, draws, rng):
    x = np.array(x0, dtype=float)
    lp, g = grad(x)
    inv_m = np.ones(x.size)
    warmup = draws
    updates = {int(0.3 * warmup): int(0.1 * warmup), int(0.7 * warmup): int(0.3 * warmup)}
    history = np.empty((warmup, x.size))
    step = StepSize(0.1)
    for i in range(warmup):
        if i in updates:
            window = history[updates[i]:i]
            k = len(window)
            inv_m = (k / (k + 5)) * window.var(axis=0) + 1e-3 * (5 / (k + 5))
            step = StepSize(step.eps)
        x, lp, g, accept = transition(grad, x, lp, g, step.eps, steps(step.eps, rng), inv_m, rng)
        step.update(accept)
        history[i] = x
    eps = step.final()
    out = np.empty((draws, x.size))
    for i in range(draws):
        x, lp, g, _ = transition(grad, x, lp, g, eps, steps(eps, rng), inv_m, rng)
        out[i] = x
    return out
