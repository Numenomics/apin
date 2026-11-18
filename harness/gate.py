"""The correctness gate: does a run's posterior agree with the reference?

    check(draws, ref) -> Verdict

For every reference variable x and three moments f in {x, x^2, (x - mu_ref)^2}
the gate computes the standardised squared bias (Robnik, Cohn-Gordon and
Seljak 2025, eq. 15)

    b2 = (mean_run(f) - E_ref[f])^2 / Var_ref[f].

Both expectations are estimates. Under a correct sampler their difference is
roughly normal with variance s^2 = mcse_run^2 + mcse_ref^2. The gate fails a
run only when it is confident the bias is above the threshold: with z the
Bonferroni critical value for m = 3 * (number of variables) two-sided tests
at family-wise level ALPHA,

    b2_lower = max(0, |diff| - z * s)^2 / Var_ref[f]   and   fail if b2_lower > THRESHOLD.

A correct sampler fails only when some |diff| > z * s, which has probability
at most ALPHA per run however few draws it returns. PROTOCOL.md, "The
correctness gate", gives the reasons and tests/test_gate.py the check.

Before the bias test the gate fails a run with non-finite draws, a missing
reference variable, or a split R-hat above RHAT_BAD on any variable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtri

from . import ess
from .reference import Reference

THRESHOLD = 0.01  # b2 for the worst moment: |bias| of 0.1 reference standard deviations
ALPHA = 0.01  # designed false-fail rate per run for a correct sampler
RHAT_BAD = 1.05
MOMENTS = ("x", "x2", "var")


@dataclass
class Verdict:
    passed: bool
    reason: str  # "" when passed; else "nonfinite", "missing", "rhat" or "bias"
    worst_variable: str = ""
    worst_moment: str = ""
    b2: float = math.nan  # point estimate for the worst (variable, moment)
    b2_lower: float = math.nan  # its lower confidence bound; the gate statistic
    rhat_max: float = math.nan
    n_tests: int = 0

    def label(self) -> str:
        """Short text for the score table."""
        if self.passed:
            return "pass"
        if self.reason == "bias":
            return f"wrong: {self.worst_variable} {self.worst_moment} b2={self.b2:.3g}"
        if self.reason == "rhat":
            return f"wrong: R-hat {self.rhat_max:.3f} on {self.worst_variable}"
        return f"wrong: {self.reason} {self.worst_variable}".rstrip()

    def to_dict(self) -> dict:
        return asdict(self)


def critical_z(n_tests: int, alpha: float = ALPHA) -> float:
    return float(-ndtri(alpha / (2 * n_tests)))


def _mcse(f: np.ndarray) -> float:
    sd = float(np.std(f, ddof=1))
    if sd == 0:
        return 0.0
    e = ess.ess_mean(f)
    return sd / math.sqrt(e) if math.isfinite(e) and e > 0 else math.inf


def check(draws: dict[str, np.ndarray], ref: Reference, threshold: float = THRESHOLD, alpha: float = ALPHA,
          rhat_bad: float = RHAT_BAD) -> Verdict:
    """draws maps variable name -> (draws, chains) array of the run (diagnostics excluded)."""
    for name, x in draws.items():
        if not np.all(np.isfinite(x)):
            return Verdict(False, "nonfinite", name)
    missing = [n for n in ref.names if n not in draws]
    if missing:
        return Verdict(False, "missing", missing[0])

    rhats = {n: ess.rhat(x) for n, x in draws.items()} if math.isfinite(rhat_bad) else {}
    finite = {n: r for n, r in rhats.items() if math.isfinite(r)}
    worst_r = max(finite, key=finite.get) if finite else ""
    rhat_max = finite.get(worst_r, math.nan)
    if rhat_max > rhat_bad:
        return Verdict(False, "rhat", worst_r, rhat_max=float(rhat_max))

    st = ref.stats
    m = len(MOMENTS) * len(ref.names)
    z = critical_z(m, alpha)
    worst = (-1.0, -1.0, "", "")
    for i, name in enumerate(ref.names):
        x = np.asarray(draws[name], dtype=float)
        for moment, f, target, sd, se in (
            ("x", x, st["mean"][i], st["sd"][i], st["mcse_mean"][i]),
            ("x2", x**2, st["sq_mean"][i], st["sq_sd"][i], st["mcse_sq"][i]),
            ("var", (x - st["mean"][i]) ** 2, st["var"][i], st["var_sd"][i], st["mcse_var"][i]),
        ):
            diff = abs(float(f.mean()) - target)
            s = math.hypot(_mcse(f), se)
            b2 = diff**2 / sd**2
            lower = max(0.0, diff - z * s) ** 2 / sd**2 if math.isfinite(s) else 0.0
            if (lower, b2) > worst[:2]:
                worst = (lower, b2, name, moment)
    lower, b2, name, moment = float(worst[0]), float(worst[1]), worst[2], worst[3]
    ok = lower <= threshold
    return Verdict(ok, "" if ok else "bias", name, moment, b2, lower, float(rhat_max), m)
