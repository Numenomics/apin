"""S3 Rewind Reference (FIXED): Lane A4 nonlinear sinh-arcsinh transport with exact log-det-Jacobian."""

from __future__ import annotations

import numpy as np

requires = ["grad"]


def forward(transport_tuning, z):
    z_arr = np.asarray(z, dtype=np.float64)
    delta = transport_tuning["delta"]
    scale = transport_tuning["scale"]
    a = np.arcsinh(z_arr)
    u = delta * a
    y = np.sinh(u)
    theta = transport_tuning["loc"] + scale * y
    # d(theta_i)/d(z_i) = scale_i * delta_i * cosh(delta_i * arcsinh(z_i)) / sqrt(1 + z_i^2)
    ldj = float(np.sum(np.log(scale * delta) + np.log(np.cosh(u)) - 0.5 * np.log1p(z_arr * z_arr)))
    return theta, ldj


def pullback_grad(transport_tuning, z, grad_theta):
    z_arr = np.asarray(z, dtype=np.float64)
    delta = transport_tuning["delta"]
    scale = transport_tuning["scale"]
    a = np.arcsinh(z_arr)
    u = delta * a
    inv_r = 1.0 / np.sqrt(1.0 + z_arr * z_arr)
    dtheta_dz = scale * delta * np.cosh(u) * inv_r
    # d/dz [log cosh(delta * arcsinh(z)) - 0.5 * log(1 + z^2)]
    dldj_dz = delta * np.tanh(u) * inv_r - z_arr / (1.0 + z_arr * z_arr)
    return dtheta_dz * np.asarray(grad_theta, dtype=np.float64) + dldj_dz


def inverse(transport_tuning, theta):
    y = (np.asarray(theta, dtype=np.float64) - transport_tuning["loc"]) / transport_tuning["scale"]
    return np.sinh(np.arcsinh(y) / transport_tuning["delta"])


def adapt_transport(oracle, budget, rng):
    d = int(oracle.dim)
    x0 = np.zeros(d, dtype=np.float64)
    oracle.log_density_gradient(x0)
    return {
        "eps": 0.35,
        "loc": np.zeros(d, dtype=np.float64),
        "scale": np.ones(d, dtype=np.float64),
        "delta": np.full(d, 1.35, dtype=np.float64),
    }, x0
