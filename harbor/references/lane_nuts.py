# ruff: noqa: F821
# --- lane_nuts.py: lanes A0, A2, A3 (the submission owns the whole kernel) ---------------------
# Appended to stan_core.py by scripts/sync_harbor_tests.py; not importable on its own.
REFERENCE_NAME = "stan-nuts-diag-2.37"
REFERENCE_LANES = ("A0", "A2", "A3")


def adapt(oracle, budget, rng):
    """Stan's adaptive warmup for budget.draws iterations; tuning = adapted step size and metric."""
    w = stan_warmup(oracle, int(budget.draws), rng)
    tuning = {"eps": float(w["eps"]), "inv_m": np.asarray(w["inv_m"], dtype=np.float64)}
    return tuning, (np.asarray(w["q"], dtype=np.float64), None), [1.0], "none"


def transition(tuning, chain, oracle, rng):
    """One Stan NUTS transition (max depth 10) with the frozen step size and diagonal metric."""
    x = np.asarray(chain.position, dtype=np.float64)
    lp, g = oracle.log_density_gradient(x)
    g = np.asarray(g, dtype=np.float64)
    if not _finite(float(lp), g):
        return x, None
    z, _, _ = nuts_transition(x, float(lp), g, oracle, tuning["inv_m"], float(tuning["eps"]), rng)
    return np.asarray(z.q, dtype=np.float64), None
