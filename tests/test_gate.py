"""The correctness gate: three verdicts, and the findings of the first review (S1-S4, S6, S10).

The target is a correlated normal in five variables with lp__ computed from the draws, as the
harness computes it. The reference is estimated from 10 chains x 1000 independent draws, as
posteriordb's are. harness/gate_rates.py measures the error rates at full scale.
"""

import math

import numpy as np
import pytest

from harness import gate, reference

MU = np.array([0.0, 3.0, -1.0, 10.0, 0.5])
SIGMA = np.array([1.0, 0.5, 2.0, 1.0, 0.1])
NAMES = [f"x.{i + 1}" for i in range(len(MU))]
CORR = np.array([[1.0, -0.888, 0.2, 0.0, 0.0],
                 [-0.888, 1.0, -0.1, 0.0, 0.0],
                 [0.2, -0.1, 1.0, 0.3, 0.0],
                 [0.0, 0.0, 0.3, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0]])
L = np.linalg.cholesky(CORR)


def with_lp(x: np.ndarray) -> dict:
    """x: (n, chains, 5) draws -> the run's dict, with lp__ as the harness would compute it."""
    e = np.linalg.solve(L, ((x - MU) / SIGMA)[..., None])[..., 0]
    return {**{name: x[:, :, i] for i, name in enumerate(NAMES)}, "lp__": -0.5 * (e**2).sum(axis=-1)}


def ar1(rng, n, chains, rho, latent=None):
    """Stationary AR(1) chains on the latent independent normals, mapped to the target."""
    e = rng.standard_normal((n, chains, len(MU))) if latent is None else latent
    z = np.empty_like(e)
    z[0] = e[0]
    for t in range(1, n):
        z[t] = rho * z[t - 1] + math.sqrt(1 - rho**2) * e[t]
    return MU + SIGMA * (z @ L.T)


@pytest.fixture(scope="module")
def ref():
    rng = np.random.default_rng(991)
    d = with_lp(ar1(rng, 1000, 10, 0.0))
    return reference.summarise({n: d[n] for n in NAMES}, lp=d["lp__"], model="sim", source="test")


def test_independent_draws_are_right(ref):
    v = gate.check(with_lp(ar1(np.random.default_rng(2), 5000, 4, 0.0)), ref)
    assert v.status == "right" and v.label() == "right", v.label()
    assert v.quantities == 4 * len(ref.series) and v.ess > 15000


def test_autocorrelated_correct_chains_are_never_wrong(ref):
    """S1: stationary AR(1) chains of length 1000. At rho 0.99 and 0.999 the chains have not mixed
    and the old gate called most of them wrong; they are unproven, and none is wrong."""
    rng = np.random.default_rng(1)
    for rho in (0.9, 0.99, 0.999):
        got = [gate.check(with_lp(ar1(rng, 1000, 4, rho)), ref, reps=1).status for _ in range(20)]
        assert "wrong" not in got, (rho, got)
        assert set(got) == {"unproven"}, (rho, got)  # ESS about 210, 20 and 2 of 4000: no proof possible


def test_a_shift_of_one_sd_is_wrong(ref):
    x = ar1(np.random.default_rng(3), 5000, 4, 0.0)
    x[..., 0] += SIGMA[0]
    v = gate.check(with_lp(x), ref)
    assert v.status == "wrong" and v.reason == "error"


def test_one_wild_draw_earns_no_credit(ref):
    """S2: the old gate passed a run with one draw at 1e6 because that draw inflated its own MCSE.
    The fractions below the reference quantiles are bounded, and the square is not proven."""
    x = ar1(np.random.default_rng(4), 5000, 4, 0.0)
    x[0, 0, 0] = 1e6
    assert gate.check(with_lp(x), ref).status != "right"
    x[..., 0] += SIGMA[0]  # with the shift as well, the bounded fractions show the error
    assert gate.check(with_lp(x), ref).status == "wrong"


def test_a_frozen_coordinate_is_wrong(ref):
    """S3: a variable held at one value gave an undefined MCSE, which the old gate read as a pass."""
    d = with_lp(ar1(np.random.default_rng(4), 1000, 4, 0.0))
    d["x.1"] = np.full((1000, 4), 0.9996422734533653)
    v = gate.check(d, ref)
    assert v.status == "wrong" and v.reason == "frozen" and v.series == "x.1"


def test_independent_marginals_are_wrong(ref):
    """S4: draws with the right marginals but no correlation (a diagonal approximation). The
    projections on the reference's eigenvectors and lp__ see it."""
    rng = np.random.default_rng(4)
    x = MU + SIGMA * rng.standard_normal((5000, 4, 5))
    v = gate.check(with_lp(x), ref)
    assert v.status == "wrong" and (v.series.startswith("proj.") or v.series == "lp__"), v.label()


def test_a_two_valued_variable_is_compared_exactly():
    """S10: the reference variable is +-1 in equal shares, so its square is exactly 1 and has no
    MCSE. Chains frozen at -2, -2, +2, +2 have the right mean; their square is 4."""
    x = np.tile([-1.0, 1.0], 5000).reshape(1000, 10)
    ref = reference.summarise({"x": x})
    run = np.column_stack([np.full(1000, -2.0)] * 2 + [np.full(1000, 2.0)] * 2)
    v = gate.check({"x": run}, ref)
    assert v.status == "wrong" and v.reason == "exact" and v.function == "square"


def test_copied_or_dependent_chains_fail_calibration(ref):
    """Slice 2 removes the _copies heuristic (process isolation prevents cross-chain copying by
    construction), and calibration in judge() still catches dependent chains across reps."""
    rng = np.random.default_rng(7)
    verdicts = []
    for _ in range(3):
        one = ar1(rng, 5000, 1, 0.0)
        x = np.concatenate([one + c * 1e-3 * SIGMA for c in range(4)], axis=1)
        verdicts.append(gate.check(with_lp(x), ref, reps=3))
    m = gate.judge(verdicts, ref, reps=3)
    assert m.status == "unproven" and m.calibration > 2.5, m


def test_correct_reps_pass_the_calibration(ref):
    rng = np.random.default_rng(8)
    verdicts = [gate.check(with_lp(ar1(rng, 5000, 4, 0.0)), ref, reps=3) for _ in range(3)]
    m = gate.judge(verdicts, ref, reps=3)
    assert m.status == "right" and 0.5 < m.calibration < 1.5, m


def test_model_verdict_from_reps(ref):
    right = gate.Verdict("right", quantities=44, errors=[0.0] * 44, mcse=[0.01] * 44)
    unproven = gate.Verdict("unproven", "rhat", "x.1", rhat_max=1.2)
    wrong = gate.Verdict("wrong", "missing", "x.2")
    assert gate.judge([right, right], ref, 2).status == "right"
    assert gate.judge([right], ref, 2).status == "wrong"  # a rep is missing
    assert gate.judge([right, None], ref, 2).status == "wrong"
    assert gate.judge([right, wrong], ref, 2).status == "wrong"
    assert gate.judge([right, unproven], ref, 2).status == "unproven"


def test_hard_failures(ref):
    rng = np.random.default_rng(9)
    d = with_lp(ar1(rng, 500, 4, 0.0))
    d["x.2"][5, 1] = np.nan
    assert gate.check(d, ref).reason == "nonfinite"
    d = with_lp(ar1(rng, 500, 4, 0.0))
    del d["lp__"]
    v = gate.check(d, ref)
    assert v.status == "wrong" and v.reason == "missing" and v.series == "lp__"


def test_a_reference_constant_must_be_reproduced():
    rng = np.random.default_rng(10)
    ref = reference.summarise({"a": rng.standard_normal((1000, 10)), "one": np.ones((1000, 10))})
    run = {"a": rng.standard_normal((1000, 4)), "one": np.ones((1000, 4))}
    assert gate.check(run, ref).status == "right"
    run["one"] = run["one"] + 1e-3
    assert gate.check(run, ref).reason == "constant"


def test_extra_run_variables_are_ignored(ref):
    d = dict(with_lp(ar1(np.random.default_rng(11), 5000, 4, 0.0)), extra=np.full((5000, 4), np.nan))
    assert gate.check(d, ref).status == "right"


def _excursions(rng, n, chains, at):
    """x = z + .002 w, y = z - .002 w, w normal with +200 added at the (draw, chain) pairs `at`:
    the projection x - y is w alone and shows the rare excursions; x + y never does."""
    z, w = rng.standard_normal((2, n, chains))
    for t, c in at:
        w[t, c] += 200
    return {"x": z + 0.002 * w, "y": z - 0.002 * w}


def test_a_quantity_the_reference_halves_disagree_on_cannot_be_wrong():
    """T5: the reference is summarised on chains 5-9, which saw none of the rare excursions that
    chains 0-4 saw, so its x - y projection claims a tiny MCSE. A correct run that sees them at
    their true rate was called wrong. The halves disagree on that projection, so it can keep the
    run from right but cannot make it wrong; quantities the halves agree on still can."""
    rng = np.random.default_rng(5)
    ref = reference.summarise(_excursions(rng, 1000, 10, [(100, 0), (500, 2), (900, 4)]))
    assert ref.halves["mean"].shape == ref.halves["mcse"].shape == (2, len(ref.series), 4)
    disputed = gate.disputed(ref)
    assert disputed.any() and not disputed[:2].any()  # x and y themselves: the halves agree
    run = _excursions(rng, 5000, 4, [(t, c) for c in range(4) for t in range(250, 5000, 2000)])
    v = gate.check(run, ref)
    assert v.status == "unproven" and v.reason == "disputed", v.label()
    assert "halves" in v.label()
    shifted = {k: x + (0.5 if k == "x" else 0.0) for k, x in run.items()}
    assert gate.check(shifted, ref).status == "wrong"


def test_a_reference_without_halves_cannot_make_a_run_wrong(ref):
    """A reference that does not keep its halves cannot show that they agree."""
    import dataclasses
    bare = dataclasses.replace(ref, halves={})
    assert gate.disputed(bare).all()
    d = with_lp(ar1(np.random.default_rng(6), 5000, 4, 0.0))
    d["x.1"] = d["x.1"] + 1.0
    assert gate.check(d, ref).status == "wrong" and gate.check(d, bare).status == "unproven"


def _rare_tens(rng, shape):
    """x = z + w/4, y = z - w/4, w = +-1 except a rare 10 that x - y shows."""
    z = rng.normal(size=shape)
    w = rng.choice([-1.0, 1.0], size=shape)
    return z, w


def test_an_exact_mismatch_the_halves_disagree_on_is_unproven():
    """U1: the reference's below_q95 of x - y is exactly 1 (no draw above its q95), but one half
    saw a draw of 10 that the other half did not: estimates 0.9996 and 1.0, MCSEs 0.00028 and 0.
    A correct run that sees the excursions at their true rate was called wrong by the exact
    branch. The halves disagree there, so the mismatch keeps the run from right and no more.
    Where the halves agree the exact check still makes the run wrong."""
    rng = np.random.default_rng(8)
    z, w = _rare_tens(rng, (500, 10))
    w[0, 0] = 10
    z, w = np.concatenate([z, -z]), np.tile(w, (2, 1))
    ref = reference.summarise({"x": z + 0.25 * w, "y": z - 0.25 * w})
    _, sd, _ = ref.targets()
    exact_disputed = gate.disputed(ref) & (sd == 0)
    assert exact_disputed[3, 3] and ref.series[3] == "proj.2"
    a, b = _rare_tens(rng, (5000, 4))
    b[rng.random(a.shape) < 0.0004] = 10
    v = gate.check({"x": a + 0.25 * b, "y": a - 0.25 * b}, ref, reps=5)
    assert v.status == "unproven" and v.reason == "disputed", v.label()
    assert v.series == "proj.2"
    shifted = gate.check({"x": a + 0.25 * b + 0.5, "y": a - 0.25 * b}, ref, reps=5)
    assert shifted.status == "wrong", shifted.label()


def test_a_zero_mcse_half_disagrees_with_any_difference_beyond_the_exact_tolerance():
    """A half whose MCSE is 0 claims its estimate is exact. Estimates that differ by more than
    EXACT disagree; ones that differ by less agree."""
    import dataclasses
    ref = reference.summarise({"x": np.random.default_rng(9).normal(size=(1000, 10))})
    m = np.zeros((2, len(ref.series), 4))
    s = np.zeros_like(m)
    m[1, 0, 0], m[1, 0, 1] = 1e-3, 1e-9
    d = gate.disputed(dataclasses.replace(ref, halves={"mean": m, "mcse": s}))
    assert d[0, 0] and not d[0, 1] and not d[0, 2]


def test_halves_with_tiny_positive_mcses_are_judged_by_their_mcses_alone():
    """V2: the exact tolerance was a floor under SPLIT_Z times the smaller MCSE, so halves with
    MCSEs of 1e-8 and estimates 5e-7 apart, more than 5 times either MCSE, agreed. The exact
    tolerance applies only when a half's MCSE is 0."""
    import dataclasses
    ref = reference.summarise({"x": np.random.default_rng(31).normal(size=(5000, 4))})
    m = np.zeros((2, len(ref.series), 4))
    s = np.full_like(m, 1e-8)
    m[1, 0, 0], m[1, 0, 1] = 5e-7, 4e-8
    d = gate.disputed(dataclasses.replace(ref, halves={"mean": m, "mcse": s}))
    assert d[0, 0] and not d[0, 1] and not d[0, 2]


def test_two_tier_significance_default_and_strict(ref):
    """Default tier uses ALPHA_RIGHT=0.005 (z=2.576); strict tier tightens to 0.001 (z=3.090) and cal=0.01."""
    ar_def, aw_def, ac_def = gate.resolve_alphas(strict=False)
    ar_str, aw_str, ac_str = gate.resolve_alphas(strict=True)
    assert (ar_def, aw_def, ac_def) == (0.005, 0.01, 0.005)
    assert (ar_str, aw_str, ac_str) == (0.001, 0.001, 0.01)
    assert gate._z_right(ar_str) > gate._z_right(ar_def) > 2.5

    # Run with exact independent draws gets strictly larger upper bound under strict=True
    draws = with_lp(ar1(np.random.default_rng(12), 5000, 4, 0.0))
    v_def = gate.check(draws, ref, reps=3, strict=False)
    v_str = gate.check(draws, ref, reps=3, strict=True)
    assert v_def.status == "right" and v_str.status == "right"
    assert v_str.bound > v_def.bound

    # Borderline tolerance: passes default, unproven under strict
    mid_tol = 0.5 * (v_def.bound + v_str.bound)
    assert gate.check(draws, ref, reps=3, tolerance=mid_tol, strict=False).status == "right"
    assert gate.check(draws, ref, reps=3, tolerance=mid_tol, strict=True).status == "unproven"
