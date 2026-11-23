# ruff: noqa: F821
# --- lane_a4_transport.py: lane A4 (the judge owns latent HMC; the submission owns the map) ------
# Appended to stan_core.py by scripts/sync_harbor_tests.py; not importable on its own.
#
# The map is T(z) = loc + scale * z with scale = sqrt(Stan's regularized window variance) and
# loc = the Welford mean of the same window. The A4 starting code is not this map: it fits loc and
# scale by median and MAD over its own fixed-path HMC warmup, so it could not be reused.
# Unit-metric leapfrog in z is the same integrator as leapfrog in theta with inverse metric
# scale**2, so Stan's step size carries over to the judge's latent kernel unchanged (loc does not
# affect the dynamics). What is not Stan: the judge's sampling kernel (jittered fixed-path HMC,
# `harness/judge.py::make_a4_transition`), as in lane A1. If no metric window closes (warmup under
# 20 draws) the map is the identity, as Stan's metric would be.
REFERENCE_NAME = "stan-windowed-diag-affine-2.37"
REFERENCE_LANES = ("A4",)


def forward(transport_tuning, z):
    z_arr = np.asarray(z, dtype=np.float64)
    scale = transport_tuning["scale"]
    return transport_tuning["loc"] + scale * z_arr, float(np.sum(np.log(scale)))


def pullback_grad(transport_tuning, z, grad_theta):
    return transport_tuning["scale"] * np.asarray(grad_theta, dtype=np.float64)


def inverse(transport_tuning, theta):
    return (np.asarray(theta, dtype=np.float64) - transport_tuning["loc"]) / transport_tuning["scale"]


def adapt_transport(oracle, budget, rng):
    """Stan's warmup; the affine map is fitted from its last metric window."""
    w = stan_warmup(oracle, int(budget.draws), rng)
    scale = np.sqrt(np.asarray(w["inv_m"], dtype=np.float64))
    loc = w["window_mean"] if w["window_mean"] is not None else np.zeros(oracle.dim)
    eps = float(min(10.0, max(1e-8, w["eps"])))  # the judge accepts eps in [1e-8, 10]
    tuning = {"eps": eps, "loc": np.asarray(loc, dtype=np.float64), "scale": scale}
    return tuning, np.asarray(w["q"], dtype=np.float64)
