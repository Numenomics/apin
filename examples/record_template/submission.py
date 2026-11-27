"""Minimal valid adjusted sampler template (Metropolis-adjusted random walk) for PR records.

Copy `examples/record_template/` to `records/<release>/<track>/<date>_<slug>/`,
replace this file with your verified sampler, update metadata/pins/invariance notes,
and run `uv run apin records validate records/<release>/<track>/<date>_<slug>`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def adapt(oracle: Any, budget: Any, rng: np.random.Generator):
    dim = int(oracle.dim)
    x = np.zeros(dim, dtype=np.float64)
    lp = float(oracle.log_density(x))
    if not math.isfinite(lp):
        for _ in range(16):
            cand = rng.normal(0.0, 0.5, size=dim)
            cand_lp = float(oracle.log_density(cand))
            if math.isfinite(cand_lp):
                x, lp = cand, cand_lp
                break
    step = 2.38 / math.sqrt(max(1, dim))
    tuning = {"step": float(step), "dim": dim}
    chain0 = (x, {"lp": float(lp)})
    betas = (1.0,)
    aux_spec = {"kind": "none"}
    return tuning, chain0, betas, aux_spec


def transition(tuning: dict[str, Any], chain: Any, oracle: Any, rng: np.random.Generator):
    x = np.asarray(chain.position, dtype=np.float64)
    step = float(tuning["step"])
    lp_cur = float(oracle.log_density(x))
    prop = x + step * rng.standard_normal(x.shape)
    lp_prop = float(oracle.log_density(prop))
    if math.isfinite(lp_prop) and (lp_prop >= lp_cur or math.log(float(rng.random())) < lp_prop - lp_cur):
        return prop, {"lp": lp_prop}
    return x, {"lp": lp_cur}
