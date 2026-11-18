"""Rank-normalized split R-hat and ESS (Vehtari et al. 2021, Bayesian Analysis 16(2)).

A line-by-line port of the R package `posterior` 1.7.0 (`.ess`, `.rhat`,
`autocovariance`, `z_scale`, `.split_chains`, `.ess_quantile`). The tests check
it against values computed with that package.

Because it is a translation, this file carries posterior's licence:
BSD-3-Clause, copyright 2021 the posterior package authors, Stan Developers
and their Assignees, and the Trustees of Columbia University. The licence text
is the one in LICENSE at the top of this repository, with that copyright line.

Every function takes one variable as an array of shape (draws, chains).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

__all__ = ["ess_bulk", "ess_tail", "ess_mean", "mcse_mean", "rhat", "summarise"]


def _is_degenerate(x: np.ndarray) -> bool:
    return (not np.all(np.isfinite(x))) or abs(np.max(x) - np.min(x)) < np.finfo(float).eps


def _split_chains(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x
    half = n / 2
    return np.concatenate([x[: math.floor(half)], x[math.ceil(half + 1) - 1 :]], axis=1)


def _z_scale(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average", axis=None).reshape(x.shape)
    return ndtri((r - 3 / 8) / (r.size - 2 * 3 / 8 + 1))


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Autocovariance of each column of x (draws, chains), one FFT for all chains."""
    n = x.shape[0]
    varx = np.var(x, axis=0, ddof=1)
    m = _nextn(n)
    yc = np.zeros((2 * m, x.shape[1]))
    yc[:n] = x - x.mean(axis=0)
    ac = np.fft.ifft(np.abs(np.fft.fft(yc, axis=0)) ** 2, axis=0).real[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ac / ac[0] * varx * (n - 1) / n
    out[:, varx == 0] = 0.0
    return out


def _nextn(n: int) -> int:
    """Smallest integer >= n whose only prime factors are 2, 3 and 5 (R's nextn)."""
    k = n
    while True:
        r = k
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return k
        k += 1


def _ess(x: np.ndarray) -> float:
    niter, nchains = x.shape
    if niter < 3 or _is_degenerate(x):
        return math.nan
    acov = _autocovariance(x)
    am = np.concatenate([[math.nan], acov.mean(axis=1)])  # 1-based, as in R
    mean_var = am[1] * niter / (niter - 1)
    var_plus = mean_var * (niter - 1) / niter
    if nchains > 1:
        var_plus += np.var(x.mean(axis=0), ddof=1)
    rho = np.zeros(niter + 3)  # 1-based: rho[1] .. rho[niter]
    t = 0
    rho_even = 1.0
    rho[t + 1] = rho_even
    rho_odd = 1 - (mean_var - am[t + 2]) / var_plus
    rho[t + 2] = rho_odd
    while t < niter - 5 and not math.isnan(rho_even + rho_odd) and rho_even + rho_odd > 0:
        t += 2
        rho_even = 1 - (mean_var - am[t + 1]) / var_plus
        rho_odd = 1 - (mean_var - am[t + 2]) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
    max_t = t
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    t = 0
    while t <= max_t - 4:
        t += 2
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2
            rho[t + 2] = rho[t + 1]
    ess = nchains * niter
    # R's rho_hat_t[1:max_t] is rho[1] alone when max_t is 0.
    head = rho[1] if max_t == 0 else rho[1 : max_t + 1].sum()
    tau = -1 + 2 * head + rho[max_t + 1]
    tau = max(tau, 1 / math.log10(ess))
    return ess / tau


def _rhat_basic(x: np.ndarray) -> float:
    if _is_degenerate(x):
        return math.nan
    niter = x.shape[0]
    chain_mean = x.mean(axis=0)
    var_between = niter * np.var(chain_mean, ddof=1)
    var_within = np.var(x, axis=0, ddof=1).mean()
    return math.sqrt((var_between / var_within + niter - 1) / niter)


def ess_bulk(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if _is_degenerate(x):
        return math.nan
    return _ess(_z_scale(_split_chains(x)))


def ess_mean(x: np.ndarray) -> float:
    """ESS for the mean of x: split chains, no rank normalisation (posterior's `ess_mean`)."""
    x = np.asarray(x, dtype=float)
    if _is_degenerate(x):
        return math.nan
    return _ess(_split_chains(x))


def mcse_mean(x: np.ndarray) -> float:
    """Monte Carlo standard error of the mean of x (posterior's `mcse_mean`)."""
    x = np.asarray(x, dtype=float)
    e = ess_mean(x)
    return float(np.std(x, ddof=1) / math.sqrt(e)) if math.isfinite(e) and e > 0 else math.nan


def _ess_quantile(x: np.ndarray, prob: float) -> float:
    if _is_degenerate(x):
        return math.nan
    indicator = (x <= np.quantile(x, prob)).astype(float)
    return _ess(_split_chains(indicator))


def ess_tail(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    q05, q95 = _ess_quantile(x, 0.05), _ess_quantile(x, 0.95)
    if math.isnan(q05) or math.isnan(q95):
        return math.nan
    return min(q05, q95)


def rhat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if _is_degenerate(x):
        return math.nan
    bulk = _rhat_basic(_z_scale(_split_chains(x)))
    folded = np.abs(x - np.median(x))
    tail = _rhat_basic(_z_scale(_split_chains(folded)))
    if math.isnan(bulk) or math.isnan(tail):
        return math.nan
    return max(bulk, tail)


def summarise(draws: dict[str, np.ndarray]) -> dict:
    """Summary of a posterior: draws maps variable name to a (draws, chains) array.

    Variables whose ESS is undefined (constant or non-finite) are left out of
    the minima, as `posterior::summarise_draws` reports them as NA.
    """
    per = {}
    for name, x in draws.items():
        per[name] = (ess_bulk(x), ess_tail(x), rhat(x))
    bulk = {k: v[0] for k, v in per.items() if math.isfinite(v[0])}
    tails = [v[1] for v in per.values() if math.isfinite(v[1])]
    rhats = [v[2] for v in per.values() if math.isfinite(v[2])]
    positive = [b for b in bulk.values() if b > 0]
    worst = sorted(bulk, key=bulk.get)[:5]
    return {
        "ess_bulk_min": min(bulk.values()) if bulk else math.nan,
        "ess_bulk_geomean": math.exp(np.mean(np.log(positive))) if positive else math.nan,
        "ess_tail_min": min(tails) if tails else math.nan,
        "rhat_max": max(rhats) if rhats else math.nan,
        "n_variables": len(bulk),
        "worst_variables": worst,
    }
