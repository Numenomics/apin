"""Measure the gate's error rates by simulation (BENCH.md, rule 1; PROTOCOL.md, "The correctness gate").

    uv run python -m harness.gate_rates [--runs N] [--workers W] > results/tables/gate_rates.md

The target is a normal in five variables with correlations up to -0.888 (as b.22 and b.23 of
diamonds), with lp__ computed from the draws. Each run is 4 stationary AR(1) chains on the
latent independent normals, so every variable and every projection has autocorrelation rho at
lag 1 and ESS about 4 n (1 - rho) / (1 + rho). Each batch of runs draws a new reference from
10 chains x 1000 independent draws, so the rates include the reference's own error.

Three samplers per setting:
  correct       the target itself;
  error = tol   the mean of x.5 shifted by exactly the tolerance (in SDs): the hardest case
                for the false-right rate. x.5 is independent of the other variables, so no
                other quantity has a larger error. (A shift of x.1 by 0.1 SD is 0.23 SD on the
                projection of smallest variance, because x.1 and x.2 are correlated.)
  error = 2 tol the mean of x.5 shifted by twice the tolerance: the power to call it wrong.

Then the calibration test: groups of 5 correct reps (its false-refusal rate), and groups of 5
reps whose 4 chains are two pairs of copies 1e-3 SD apart (its power).
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import lfilter

from . import gate, reference

MU = np.array([0.0, 3.0, -1.0, 10.0, 0.5])
SIGMA = np.array([1.0, 0.5, 2.0, 1.0, 0.1])
NAMES = [f"x.{i + 1}" for i in range(len(MU))]
CORR = np.array([[1.0, -0.888, 0.2, 0.0, 0.0],
                 [-0.888, 1.0, -0.1, 0.0, 0.0],
                 [0.2, -0.1, 1.0, 0.3, 0.0],
                 [0.0, 0.0, 0.3, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0]])
L = np.linalg.cholesky(CORR)
SHIFTED = 4  # x.5, independent of the rest
SETTINGS = [(0.0, 1000), (0.0, 5000), (0.5, 1000), (0.5, 15000), (0.9, 1000), (0.9, 20000), (0.9, 50000),
            (0.99, 1000), (0.99, 50000), (0.999, 1000), (0.999, 50000)]


def with_lp(x: np.ndarray) -> dict:
    e = np.linalg.solve(L, ((x - MU) / SIGMA)[..., None])[..., 0]
    return {**{name: x[:, :, i] for i, name in enumerate(NAMES)}, "lp__": -0.5 * (e**2).sum(axis=-1)}


def ar1(rng, n, chains, rho) -> np.ndarray:
    """(n, chains, 5) stationary AR(1) draws of the target."""
    e = rng.standard_normal((n, chains, len(MU)))
    c = math.sqrt(1 - rho**2)
    z = lfilter([c], [1, -rho], e, axis=0, zi=((1 - c) * e[0])[None])[0]  # z_0 = e_0, z_t = rho z_t-1 + c e_t
    return MU + SIGMA * (z @ L.T)


def make_reference(rng) -> reference.Reference:
    d = with_lp(ar1(rng, 1000, 10, 0.0))
    return reference.summarise({n: d[n] for n in NAMES}, lp=d["lp__"], model="sim", source="sim")


def _batch(args) -> list[tuple]:
    seed, rho, n, runs = args
    rng = np.random.default_rng(seed)
    ref = make_reference(rng)
    out = []
    for _ in range(runs):
        x = ar1(rng, n, 4, rho)
        for case, shift in (("correct", 0.0), ("error = tol", 1.0), ("error = 2 tol", 2.0)):
            y = x.copy()
            y[..., SHIFTED] += shift * gate.TOLERANCE * SIGMA[SHIFTED]
            out.append((rho, n, case, gate.check(with_lp(y), ref, reps=1).status))
    return out


def _calibration(args) -> list[tuple]:
    seed, rho, n, groups = args
    rng = np.random.default_rng(seed)
    ref = make_reference(rng)
    out = []
    for _ in range(groups):
        for case in ("correct", "copies"):
            vs = []
            for _ in range(5):
                if case == "correct":
                    x = ar1(rng, n, 4, rho)
                else:
                    half = ar1(rng, n, 2, rho)
                    x = np.concatenate([half, half + 1e-3 * SIGMA], axis=1)
                vs.append(gate.check(with_lp(x), ref, reps=5))
            m = gate.judge(vs, ref, reps=5)
            out.append((rho, n, case, m.status, m.calibration))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200, help="runs per setting")
    ap.add_argument("--groups", type=int, default=100, help="groups of 5 reps per calibration setting")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=20, help="runs per reference")
    a = ap.parse_args(argv)
    jobs = [(1000 * k + 7 * i, rho, n, a.batch) for k, (rho, n) in enumerate(SETTINGS)
            for i in range(a.runs // a.batch)]
    cal_jobs = [(10**6 + 1000 * k + 7 * i, rho, n, 10) for k, (rho, n) in enumerate([(0.0, 5000), (0.5, 15000)])
                for i in range(a.groups // 10)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for b in ex.map(_batch, jobs) for r in b]
        cal = [r for b in ex.map(_calibration, cal_jobs) for r in b]

    print(f"Gate error rates by simulation: tolerance {gate.TOLERANCE} SD, alpha_wrong {gate.ALPHA_WRONG} per model, "
          f"alpha_right {gate.ALPHA_RIGHT} per rep, R-hat < {gate.RHAT_MAX}, ESS >= {gate.MIN_ESS}. "
          f"{a.runs} runs of 4 chains per row, a new reference every {a.batch} runs. One rep per verdict.\n")
    print("| rho | draws per chain | ESS about | sampler | right | unproven | wrong |")
    print("|---|---|---|---|---|---|---|")
    for rho, n in SETTINGS:
        for case in ("correct", "error = tol", "error = 2 tol"):
            got = [s for r, nn, c, s in res if r == rho and nn == n and c == case]
            share = {s: got.count(s) / len(got) for s in ("right", "unproven", "wrong")}
            e = 4 * n * (1 - rho) / (1 + rho)
            print(f"| {rho} | {n} | {e:.0f} | {case} | {share['right']:.3f} | {share['unproven']:.3f} | "
                  f"{share['wrong']:.3f} |")
    print(f"\nCalibration test, groups of 5 reps ({a.groups} groups per row):\n")
    print("| rho | draws per chain | sampler | right | unproven | wrong | median ratio |")
    print("|---|---|---|---|---|---|---|")
    for rho, n in [(0.0, 5000), (0.5, 15000)]:
        for case in ("correct", "copies"):
            got = [(s, c) for r, nn, k, s, c in cal if r == rho and nn == n and k == case]
            st = [s for s, _ in got]
            ratio = np.nanmedian([c for _, c in got]) if any(math.isfinite(c) for _, c in got) else math.nan
            print(f"| {rho} | {n} | {case} | {st.count('right') / len(st):.3f} | "
                  f"{st.count('unproven') / len(st):.3f} | {st.count('wrong') / len(st):.3f} | {ratio:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
