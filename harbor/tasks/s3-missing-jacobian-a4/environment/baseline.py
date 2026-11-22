"""S3 Rewind Base (BUGGY): Lane A4 nonlinear sinh-arcsinh transport missing log-det-Jacobian."""

from __future__ import annotations

import numpy as np

requires = ["grad"]


def forward(transport_tuning, z):
    z_arr = np.asarray(z, dtype=np.float64)
    delta = transport_tuning["delta"]
    y = np.sinh(delta * np.arcsinh(z_arr))
    theta = transport_tuning["loc"] + transport_tuning["scale"] * y
    # BUG: forgot log|det J_T(z)| for nonlinear sinh(delta * arcsinh(z))!
    return theta, 0.0


def pullback_grad(transport_tuning, z, grad_theta):
    z_arr = np.asarray(z, dtype=np.float64)
    delta = transport_tuning["delta"]
    a = np.arcsinh(z_arr)
    dy_dz = delta * np.cosh(delta * a) / np.sqrt(1.0 + z_arr * z_arr)
    # BUG: also omitted grad_z(log_abs_det_jac)!
    return transport_tuning["scale"] * dy_dz * np.asarray(grad_theta, dtype=np.float64)


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
