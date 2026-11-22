"""The correctness gate: has a run shown that its posterior agrees with the reference?

    check(draws, ref, reps) -> Verdict        one repetition
    judge(verdicts, ref, reps) -> ModelVerdict  a model, from all its repetitions

BENCH.md states the rules this module implements; PROTOCOL.md, "The correctness gate", the
reasons and the error rates, and harness/gate_rates.py the simulation that measures them.

A repetition gets one of three verdicts.

  wrong     a hard failure (a missing or non-finite variable, a variable the run never moves,
            a reference constant the run does not reproduce, a function whose reference value
            is exact and which the run misses), or a checked quantity whose error is shown to
            be outside the tolerance.
  right     every checked quantity is shown to be within the tolerance.
  unproven  everything else: chains that have not mixed, a quantity whose standard error the
            run does not define, an error that is neither shown small nor shown large.

The checked quantities are the FUNCTIONS of every series of the reference (reference.py): the
mean, the mean of the square, and the fractions below the reference's 5% and 95% quantiles, of
each standardised variable, of each stored projection and of lp__. For quantity j

    e_j = (mean_run(f_j) - E_ref[f_j]) / SD_ref[f_j]
    s_j = sqrt(mcse_run(f_j)^2 + mcse_ref(f_j)^2) / SD_ref[f_j]

and, with tolerance TOLERANCE (in reference SDs of the function),

    wrong  if some |e_j| - z_w s_j > TOLERANCE, z_w the Bonferroni value for all quantities of
           all reps at family-wise level ALPHA_WRONG per model;
    right  if every |e_j| + z_r s_j < TOLERANCE, z_r = Phi^-1(1 - ALPHA_RIGHT).

`right` is an intersection-union test: a run whose true error on one quantity is at least the
tolerance is called right with probability at most ALPHA_RIGHT, however many quantities there
are. Both tests trust s_j, so they run only when the chains look mixed: rank-normalized split
R-hat (bulk and folded, whichever is defined) below RHAT_MAX and bulk and tail ESS at least
MIN_ESS on every series. Cross-chain copying is prevented by per-chain process isolation and checked via calibration in judge().

A model is right when every rep is right and the realized errors agree with the claimed
standard errors (judge, the calibration test), wrong when any rep is wrong, unproven otherwise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2

from . import ess
from .reference import FUNCTIONS, LP, Reference

STATUSES = ("right", "wrong", "unproven")
TOLERANCE = 0.1  # reference SDs of the function
# Default (dev/standard) significance levels
ALPHA_WRONG = 0.01  # false-wrong rate per model for a correct sampler whose chains have mixed
ALPHA_RIGHT = 0.005  # false-right rate per rep for a sampler whose error is at the tolerance (z = 2.576)
ALPHA_CALIBRATION = 0.005  # rate at which a correct, mixed model is refused for miscalibration
# Strict (verification) significance levels (`apin score --strict`)
STRICT_ALPHA_WRONG = 0.001
STRICT_ALPHA_RIGHT = 0.001  # z = 3.090 per rep
STRICT_ALPHA_CALIBRATION = 0.01
RHAT_MAX = 1.01
MIN_ESS = 400
EXACT = 1e-6  # agreement asked of a quantity whose reference value is exact
SPLIT_Z = 5.0  # in MCSEs of the more confident half: the reference's halves disagree beyond this


def resolve_alphas(
    *,
    strict: bool = False,
    alpha_right: float | None = None,
    alpha_wrong: float | None = None,
    alpha_calibration: float | None = None,
) -> tuple[float, float, float]:
    """Return `(alpha_right, alpha_wrong, alpha_calibration)` for default or `--strict` tier."""
    ar = (STRICT_ALPHA_RIGHT if strict else ALPHA_RIGHT) if alpha_right is None else float(alpha_right)
    aw = (STRICT_ALPHA_WRONG if strict else ALPHA_WRONG) if alpha_wrong is None else float(alpha_wrong)
    ac = (STRICT_ALPHA_CALIBRATION if strict else ALPHA_CALIBRATION) if alpha_calibration is None else float(alpha_calibration)
    return ar, aw, ac


@dataclass
class Verdict:
    status: str  # "right", "wrong" or "unproven"
    reason: str = ""  # "" when right
    series: str = ""  # the series that decided, or the worst one
    function: str = ""
    error: float = math.nan  # e_j of that quantity
    bound: float = math.nan  # its confidence bound on |e_j|: upper if right/unproven, lower if wrong
    rhat_max: float = math.nan
    ess: float = math.nan  # min over series of bulk and tail ESS: the ESS the score credits
    quantities: int = 0
    errors: list | None = None  # e_j of every quantity (NaN where the reference value is exact)
    mcse: list | None = None  # mcse_run(f_j) / SD_ref[f_j], inf where the run does not define it

    def label(self) -> str:
        """Short text for the score table."""
        if self.status == "right":
            return "right"
        if self.reason == "disputed":
            return (f"unproven: the reference's halves disagree on {self.series} {self.function} "
                    f"e={self.error:+.3g} bound={self.bound:.3g}")
        if self.reason in ("error", "exact", "unproven"):
            return f"{self.status}: {self.series} {self.function} e={self.error:+.3g} bound={self.bound:.3g}"
        if self.reason == "rhat":
            return f"unproven: R-hat {self.rhat_max:.3f} on {self.series}"
        if self.reason == "ess":
            return f"unproven: ESS {self.ess:.0f} on {self.series}"
        return f"{self.status}: {self.reason} {self.series}".rstrip()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


@dataclass
class ModelVerdict:
    status: str
    reason: str = ""
    calibration: float = math.nan  # realized / claimed error variance, pooled over reps
    calibration_p: float = math.nan


def _z_right(alpha_right: float = ALPHA_RIGHT) -> float:
    return float(ndtri(1 - alpha_right))


def z_wrong(quantities: int, reps: int, alpha_wrong: float = ALPHA_WRONG) -> float:
    return float(-ndtri(alpha_wrong / (2 * max(quantities, 1) * max(reps, 1))))


def disputed(ref: Reference) -> np.ndarray:
    """(len(series), len(FUNCTIONS)) True where the two halves of the reference disagree.

    They disagree on a quantity when their estimates differ by more than SPLIT_Z times the
    smaller of the two halves' MCSEs: one half's estimate is far outside what the other half
    claims as its own error. A half whose MCSE is 0 claims its estimate is exact, so against it
    any difference beyond the exact tolerance (EXACT, as for exact reference values) is a
    disagreement; estimates within that tolerance agree. The exact tolerance applies only then:
    with both MCSEs positive, however small, SPLIT_Z times the smaller decides. The combined
    MCSE, hypot of the two, does not do: a half that saw
    k rare excursions the other half missed differs from it by about sqrt(k) combined MCSEs,
    since its own MCSE grows with what it saw. A reference without halves cannot show that they
    agree, so every quantity is disputed."""
    target, _, _ = ref.targets()
    if not ref.halves:
        return np.ones(target.shape, dtype=bool)
    m, s = ref.halves["mean"], ref.halves["mcse"]
    smaller = np.minimum(s[0], s[1])
    slack = np.where(smaller > 0, SPLIT_Z * smaller, EXACT * np.maximum(1.0, np.abs(m).max(axis=0)))
    return np.abs(m[0] - m[1]) > slack


def check(
    draws: dict[str, np.ndarray],
    ref: Reference,
    reps: int = 1,
    tolerance: float = TOLERANCE,
    *,
    strict: bool = False,
    alpha_right: float | None = None,
    alpha_wrong: float | None = None,
) -> Verdict:
    """draws maps a name to a (draws, chains) array of the run; lp__ is needed when ref has it."""
    ar, aw, _ = resolve_alphas(strict=strict, alpha_right=alpha_right, alpha_wrong=alpha_wrong)
    need = list(ref.names) + list(ref.constants) + ([LP] if ref.has_lp else [])
    for n in need:
        if n not in draws:
            return Verdict("wrong", "missing", n)
    for n in need:
        if not np.all(np.isfinite(draws[n])):
            return Verdict("wrong", "nonfinite", n)
    for n, v in ref.constants.items():
        if np.max(np.abs(np.asarray(draws[n], dtype=float) - v)) > EXACT * max(1.0, abs(v)):
            return Verdict("wrong", "constant", n)
    for n in ref.names + ([LP] if ref.has_lp else []):
        if np.ptp(draws[n]) == 0:  # exact, before any arithmetic: a frozen coordinate is not a sample
            return Verdict("wrong", "frozen", n)

    s = ref.standardise(draws)
    target, sd, mcse_ref = ref.targets()
    n_series, n_fun = target.shape
    err = np.full((n_series, n_fun), math.nan)
    se = np.full((n_series, n_fun), math.nan)
    rhats, esses = np.empty(n_series), np.empty(n_series)
    split = disputed(ref)
    exact_disputed = None  # the first exact value missed where the halves disagree: unproven at most
    for i, x in enumerate(s):
        parts = [r for r in ess.rhat_parts(x) if not math.isnan(r)]
        rhats[i] = max(parts) if parts else math.inf
        b, t = ess.ess_bulk(x), ess.ess_tail(x)
        esses[i] = min(b, t) if math.isfinite(b) and math.isfinite(t) else math.nan
        for j, f in enumerate(ref.functions(x, i)):
            run = float(f.mean())
            if sd[i, j] == 0:  # the reference value is exact (S10): compare it directly
                if abs(run - target[i, j]) > EXACT * max(1.0, abs(target[i, j])):
                    if not split[i, j]:
                        return Verdict("wrong", "exact", ref.series[i], FUNCTIONS[j], run - target[i, j])
                    exact_disputed = exact_disputed or (i, j, run - target[i, j])
                continue
            err[i, j] = (run - target[i, j]) / sd[i, j]
            e = ess.ess_mean(f) if np.ptp(f) > 0 else math.nan
            mc = float(np.std(f, ddof=1)) / math.sqrt(e) if math.isfinite(e) and e > 0 else math.inf
            se[i, j] = math.inf if math.isinf(mc) else math.hypot(mc, mcse_ref[i, j]) / sd[i, j]

    checked = ~np.isnan(err)
    q = int(checked.sum())
    ess_min = float(np.nanmin(esses)) if not np.all(np.isnan(esses)) else math.nan
    base = dict(rhat_max=float(np.max(rhats)), ess=ess_min if not np.isnan(esses).any() else math.nan, quantities=q)
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = np.sqrt(np.maximum(se**2 - (mcse_ref / sd) ** 2, 0.0))
    base.update(errors=err.ravel().tolist(), mcse=mc.ravel().tolist())

    worst_r = int(np.argmax(rhats))
    if not rhats[worst_r] < RHAT_MAX:
        return Verdict("unproven", "rhat", ref.series[worst_r], **base)
    if np.isnan(esses).any() or ess_min < MIN_ESS:
        i = int(np.argmax(np.isnan(esses))) if np.isnan(esses).any() else int(np.nanargmin(esses))
        return Verdict("unproven", "ess", ref.series[i], **{**base, "ess": float(esses[i])})

    absval = np.abs(np.where(checked, err, 0.0))
    # a quantity the reference's halves disagree on can keep the run from right, not make it wrong
    lower = np.where(checked & ~split, absval - z_wrong(q, reps, alpha_wrong=aw) * se, -math.inf)
    i, j = np.unravel_index(int(np.argmax(lower)), lower.shape)
    if lower[i, j] > tolerance:
        return Verdict("wrong", "error", ref.series[i], FUNCTIONS[j], float(err[i, j]), float(lower[i, j]), **base)
    upper = np.where(checked, absval + _z_right(alpha_right=ar) * se, -math.inf)
    i, j = np.unravel_index(int(np.argmax(upper)), upper.shape)
    status = "right" if upper[i, j] < tolerance else "unproven"
    if status == "right" and exact_disputed:
        i, j, e = exact_disputed
        return Verdict("unproven", "disputed", ref.series[i], FUNCTIONS[j], float(e), math.nan, **base)
    reason = "" if status == "right" else "disputed" if split[i, j] else "unproven"
    return Verdict(status, reason, ref.series[i], FUNCTIONS[j],
                   float(err[i, j]), float(upper[i, j]), **base)


def judge(
    verdicts: list[Verdict | None],
    ref: Reference,
    reps: int,
    *,
    strict: bool = False,
    alpha_calibration: float | None = None,
) -> ModelVerdict:
    """The model's verdict from its reps; None stands for a rep that is missing or failed (wrong).

    A model is right only when every rep is right and it passes the calibration test (rule 2).
    """
    _, _, ac = resolve_alphas(strict=strict, alpha_calibration=alpha_calibration)
    if len(verdicts) < reps or any(v is None for v in verdicts):
        return ModelVerdict("wrong", "a rep is missing or failed")
    if any(v.status == "wrong" for v in verdicts):
        return ModelVerdict("wrong", next(v.label() for v in verdicts if v.status == "wrong"))
    if any(v.status == "unproven" for v in verdicts):
        return ModelVerdict("unproven", next(v.label() for v in verdicts if v.status == "unproven"))
    ratio, p = calibration(verdicts, ref)
    if not p >= ac:
        return ModelVerdict("unproven", f"calibration: realized error variance {ratio:.2f}x the claimed "
                            f"(p={p:.2g})", ratio, p)
    return ModelVerdict("right", "", ratio, p)


def calibration(verdicts: list[Verdict], ref: Reference) -> tuple[float, float]:
    """Do the realized errors agree with the claimed standard errors? (ratio, p-value).

    Two sums of squared standardised errors, over all quantities, each compared with its null by a
    scaled chi-square with the same mean and variance (Satterthwaite), the variance computed from
    tr(C^2), C the correlation of the quantities in the reference:

      against the reference   sum_r sum_j e_rj^2 / (mcse_run^2 + mcse_ref^2); the reference error
                              is shared by the reps, which the variance accounts for;
      between the reps        sum_j sum_r w_rj (e_rj - ebar_j)^2, w = 1 / mcse_run^2, which the
                              reference error cancels out of (needs two reps or more).

    The p-value is the smaller of the two, doubled; the ratio is realized over claimed variance of
    the between-rep sum (of the other when there is one rep). Copies of a chain, or chains that are
    otherwise dependent, give a ratio near the number of copies.
    """
    r = len(verdicts)
    if r == 0 or any(v.errors is None for v in verdicts):
        return math.nan, math.nan
    _, sd, mcse_ref = ref.targets()
    e = np.array([v.errors for v in verdicts], dtype=float)  # (reps, quantities)
    mc = np.array([v.mcse for v in verdicts], dtype=float)
    use = ~np.isnan(e).any(axis=0)
    e, mc = e[:, use], mc[:, use]
    if not np.all(np.isfinite(mc)) or not np.all(mc > 0):
        return math.nan, math.nan
    ref_se = (mcse_ref / sd).ravel()[use]
    j = e.shape[1]
    tr = ref.calibration["trC2"] * j / ref.calibration["functions"]  # if fewer quantities were checked

    claimed = mc**2 + ref_se**2
    w = float(np.mean(ref_se**2 / claimed))
    q_ref = float((e**2 / claimed).sum())
    p_ref = _sf(q_ref, r * j, 2 * tr * (r + r * (r - 1) * w**2))
    if r < 2:
        return q_ref / (r * j), min(1.0, p_ref)
    wt = 1 / mc**2
    ebar = (wt * e).sum(axis=0) / wt.sum(axis=0)
    q_rep = float((wt * (e - ebar) ** 2).sum())
    p_rep = _sf(q_rep, (r - 1) * j, 2 * (r - 1) * tr)
    return q_rep / ((r - 1) * j), min(1.0, 2 * min(p_ref, p_rep))


def _sf(q: float, mean: float, var: float) -> float:
    c, nu = var / (2 * mean), 2 * mean**2 / var
    return float(chi2.sf(q / c, nu))
