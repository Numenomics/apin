# ruff: noqa: F821
# --- lane_a1_metric.py: lane A1 (the judge owns the HMC kernel; the submission owns adapt_metric) -
# Appended to stan_core.py by scripts/sync_harbor_tests.py; not importable on its own.
#
# What is Stan here: the whole warmup (NUTS with Stan's dual averaging and windowed diagonal
# variance estimator, as in stan_core.stan_warmup). What is not: the sampling kernel. Lane A1's
# judge runs its own jittered-path-length HMC (`harness/judge.py::a1_transition`, path length 2),
# so Stan's step size, tuned for NUTS's mean acceptance statistic of 0.8, is used by a different
# kernel. No other choice keeps the reference inside the lane's seam.
REFERENCE_NAME = "stan-windowed-diag-metric-2.37"
REFERENCE_LANES = ("A1",)


def adapt_metric(oracle, budget, rng):
    """Stan's warmup; returns its step size and regularized diagonal variance for the judge."""
    w = stan_warmup(oracle, int(budget.draws), rng)
    eps = float(min(10.0, max(1e-8, w["eps"])))  # the judge accepts eps in [1e-8, 10]
    return {"eps": eps, "inv_m": np.asarray(w["inv_m"], dtype=np.float64)}, np.asarray(w["q"], dtype=np.float64)
