"""Lane A4 starter baseline: diagonal affine warm-start transport T(z) = loc + scale * z.

Editable surface for Lane A4 (`harness/judge.py` owns the verified latent-space HMC kernel
and automatically checks round-trip invertibility + finite-difference Jacobian/log-det):
  - adapt_transport(oracle, budget, rng) -> (transport_tuning, x0)
  - forward(transport_tuning, z) -> (theta, log_abs_det_jac)
  - pullback_grad(transport_tuning, z, grad_theta) -> grad_z
  - inverse(transport_tuning, theta) -> z
"""

from __future__ import annotations

import math

import numpy as np

requires = ["grad"]


def forward(transport_tuning, z):
    z_arr = np.asarray(z, dtype=np.float64)
    loc = transport_tuning["loc"]
    scale = transport_tuning["scale"]
    theta = loc + scale * z_arr
    ldj = float(np.sum(np.log(scale)))
    return theta, ldj


def pullback_grad(transport_tuning, z, grad_theta):
    # log det J is constant in z for an affine map, so grad_z(ldj) = 0
    return transport_tuning["scale"] * np.asarray(grad_theta, dtype=np.float64)


def inverse(transport_tuning, theta):
    return (np.asarray(theta, dtype=np.float64) - transport_tuning["loc"]) / transport_tuning["scale"]


def _leapfrog_diag(oracle, x, lp, g, p, inv_m, eps, n_steps):
    x1, lp1, g1 = x, lp, g
    p1 = p + 0.5 * eps * g1
    for i in range(n_steps):
        x1 = x1 + eps * (inv_m * p1)
        lp1, g1 = oracle.log_density_gradient(x1)
        if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
            return None
        if i < n_steps - 1:
            p1 = p1 + eps * g1
    p1 = p1 + 0.5 * eps * g1
    return x1, lp1, g1, p1


def adapt_transport(oracle, budget, rng):
    d = int(oracle.dim)
    x = rng.normal(0.0, 0.5, size=d)
    lp, g = oracle.log_density_gradient(x)
    for _ in range(16):
        if math.isfinite(lp) and np.all(np.isfinite(g)):
            break
        x = rng.normal(0.0, 0.25, size=d)
        lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        x = np.zeros(d, dtype=np.float64)
        lp, g = oracle.log_density_gradient(x)

    n_warm = max(24, min(int(budget.draws), 80))
    eps = 0.25 / (max(1, d) ** 0.25)
    inv_m = np.ones(d, dtype=np.float64)
    samples = []

    for step in range(n_warm):
        p = rng.standard_normal(d) / np.sqrt(inv_m)
        h0 = lp - 0.5 * float(np.sum(inv_m * p * p))
        n_steps = max(2, min(10, int(round(1.5 / max(eps, 1e-3)))))
        res = _leapfrog_diag(oracle, x, lp, g, p, inv_m, eps, n_steps)
        acc = 0.0
        if res is not None:
            x1, lp1, g1, p1 = res
            h1 = lp1 - 0.5 * float(np.sum(inv_m * p1 * p1))
            if math.isfinite(h1):
                acc = math.exp(min(0.0, h1 - h0))
                if rng.uniform() < acc:
                    x, lp, g = x1, lp1, g1
        eps *= math.exp(0.15 * (acc - 0.7))
        eps = float(np.clip(eps, 1e-4, 2.0))
        if step >= n_warm // 3:
            samples.append(x.copy())
        if step == (2 * n_warm) // 3 and len(samples) >= 8:
            arr = np.asarray(samples, dtype=np.float64)
            mad = 1.4826 * np.median(np.abs(arr - np.median(arr, axis=0)), axis=0)
            inv_m = np.clip(mad * mad, 1e-6, 1e6)

    if len(samples) >= 6:
        arr = np.asarray(samples, dtype=np.float64)
        loc = np.median(arr, axis=0)
        scale = 1.4826 * np.median(np.abs(arr - loc), axis=0)
        scale = np.clip(scale, 1e-3, 1e3)
    else:
        loc = np.zeros(d, dtype=np.float64)
        scale = np.sqrt(inv_m)

    transport_tuning = {
        "eps": float(np.clip(eps, 0.04, 0.8)),
        "loc": np.asarray(loc, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
    }

    # Calibrate latent-space step size so dispersed latent starts never stall (0% acceptance).
    def _latent_eval(z_vec):
        th, ldj = forward(transport_tuning, z_vec)
        lp_th, g_th = oracle.log_density_gradient(th)
        if not (math.isfinite(lp_th) and math.isfinite(ldj) and np.all(np.isfinite(g_th))):
            return -math.inf, np.zeros(d, dtype=np.float64)
        return float(lp_th) + float(ldj), pullback_grad(transport_tuning, z_vec, g_th)

    eps_lat = float(np.clip(eps, 0.05, 0.7))
    for probe_sd in (1.0, 2.0):
        z0 = rng.normal(0.0, probe_sd, size=d)
        lp0, g0 = _latent_eval(z0)
        if not math.isfinite(lp0):
            continue
        for frac in (1.0, 0.5, 0.25, 0.125, 0.0625):
            e = max(0.03, eps_lat * frac)
            n_i = max(2, min(16, int(math.ceil(1.5 / e))))
            p = rng.standard_normal(d)
            h0 = lp0 - 0.5 * float(p @ p)
            z1, lp1, g1, p1 = z0, lp0, g0, p + 0.5 * e * g0
            ok = True
            for i in range(n_i):
                z1 = z1 + e * p1
                lp1, g1 = _latent_eval(z1)
                if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
                    ok = False
                    break
                if i < n_i - 1:
                    p1 = p1 + e * g1
            if ok:
                p1 = p1 + 0.5 * e * g1
                h1 = lp1 - 0.5 * float(p1 @ p1)
                if math.isfinite(h1) and math.exp(min(0.0, h1 - h0)) >= 0.4:
                    eps_lat = min(eps_lat, e)
                    break
        else:
            eps_lat = min(eps_lat, 0.04)

    transport_tuning["eps"] = float(np.clip(eps_lat, 0.03, 0.8))
    return transport_tuning, x
