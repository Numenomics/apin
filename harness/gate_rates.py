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

Then the halves of the reference (gate.disputed): how often they disagree by chance on correct
reference draws, and the false-wrong rate of correct runs against references that missed rare
excursions, with the rule and without it (SPLIT_Z infinite).

With --null MODEL:GROUPS (repeatable) it prints only the calibration test's false-refusal rate
under an idealised Gaussian null with the correlations of a shipped reference (calibration_null),
to be appended to the table:

    uv run python -m harness.gate_rates --null eight_schools_noncentered:100000 \
        --null hier_2pl:20000 >> results/tables/gate_rates.md

That needs the reference's draws: the posteriordb zip or the kept chain CSVs in
build/reference/<model>, and the model's BridgeStan library for lp__. It samples nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
from scipy.signal import lfilter

from . import cmdstan, gate, models, reference

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


def _ar_latent(rng, n, chains, d, rho) -> np.ndarray:
    e = rng.standard_normal((n, chains, d))
    c = math.sqrt(1 - rho**2)
    return lfilter([c], [1, -rho], e, axis=0, zi=((1 - c) * e[0])[None])[0]


def _disputed_by_chance(args) -> tuple:
    """Share of quantities disputed on references of correct draws: 10 chains x 1000, AR(1) at rho."""
    seed, rho, refs = args
    rng = np.random.default_rng(seed)
    hit = total = 0
    for _ in range(refs):
        x = MU + SIGMA * (_ar_latent(rng, 1000, 10, len(MU), rho) @ L.T)
        d = with_lp(x)
        dis = gate.disputed(reference.summarise({n: d[n] for n in NAMES}, lp=d["lp__"]))
        hit, total = hit + int(dis.sum()), total + dis.size
    return rho, hit, total


def _excursions(args) -> tuple:
    """False wrongs of correct runs against references that may miss rare +200 excursions.

    "projection": x = z + .002 w, y = z - .002 w with the excursions in w (p = 0.0005), which
    only the x - y projection shows; 10-chain references. "variable": one variable with the
    excursions (p = 0.00035); 5-chain references. Runs are 4 chains of 5000 draws."""
    case, split_z, trials = args
    old, gate.SPLIT_Z = gate.SPLIT_Z, split_z
    try:
        rng = np.random.default_rng(547881 if case == "projection" else 787127)
        out = {"right": 0, "unproven": 0, "wrong": 0}
        for _ in range(trials):
            if case == "projection":
                draws = []
                for n, c in ((1000, 10), (5000, 4)):
                    z, w = rng.normal(size=(2, n, c))
                    w[rng.random(w.shape) < 0.0005] += 200
                    draws.append({"x": z + 0.002 * w, "y": z - 0.002 * w})
            else:
                draws = []
                for n, c in ((1000, 5), (5000, 4)):
                    x = rng.normal(size=(n, c))
                    x[rng.random(x.shape) < 0.00035] += 200
                    draws.append({"x": x})
            out[gate.check(draws[1], reference.summarise(draws[0])).status] += 1
        return case, split_z, out
    finally:
        gate.SPLIT_Z = old


RUN_ESS = 20000  # the ESS of every rep and every quantity in this null
NULL_SEEDS = {"eight_schools_noncentered": 913, "hier_2pl": 2716}


def reference_draws(name: str, most: int = 2000) -> dict:
    """An evenly thinned subset of at most `most` of the draws reference `name` was summarised from,
    with lp__ from BridgeStan, as the reference computes it. Reads the posteriordb zip or the kept
    chain CSVs in build/reference/<name>; samples nothing."""
    model = models.load()[name]
    ref = reference.load(name)
    if ref.source == "posteriordb":
        pdb = ref.meta["posteriordb_reference"]
        with zipfile.ZipFile(reference.PDB / "reference_posteriors" / "draws" / "draws" / f"{pdb}.json.zip") as z:
            chains = json.loads(z.read(f"{pdb}.json"))
        draws = {reference.stan_name(v): np.column_stack([c[v] for c in chains]) for v in chains[0]}
    else:
        csvs = sorted((reference.ROOT / "build" / "reference" / name).glob("ref_*.csv"),
                      key=lambda p: int(p.stem.split("_")[-1]))
        if len(csvs) != ref.meta["settings"]["chains"]:
            raise FileNotFoundError(f"{name}: {len(csvs)} chain CSVs kept, the reference used "
                                    f"{ref.meta['settings']['chains']}")
        files = [cmdstan.read_csv(p) for p in csvs]
        draws = {c: np.column_stack([f.column(c) for f in files]) for c in files[0].columns if not c.endswith("__")}
    n, chains = next(iter(draws.values())).shape
    take = np.unique(np.linspace(0, n - 1, max(1, most // chains)).astype(int))
    draws = {k: v[take] for k, v in draws.items()}
    if ref.has_lp:
        draws[reference.LP] = reference.log_density(model, draws, reference.ROOT / "build")
    return draws


def calibration_null(name: str, groups: int, chunk: int = 500) -> dict:
    """Groups of 5 reps under an idealised Gaussian null with the correlations of reference `name`.

    Every rep's error vector is Normal(0, run_se^2 C) plus the shared reference error, Normal(0,
    ref_se^2 C), with run_se = 1/sqrt(RUN_ESS), ref_se the reference's MCSE in SDs and C the
    correlation of the checked functions over the reference draws, its off-diagonal part scaled so
    that tr(C^2) is the reference's trC2 (the raw correlation of 2000 draws overstates it). The
    reps' claimed MCSEs are exact. Groups in which every rep is right go through gate.calibration;
    the rest are discarded, as the judge never calls the test on them."""
    ref = reference.load(name)
    s = ref.standardise(reference_draws(name))
    flat = s.reshape(len(s), -1)
    _, sd, mcse_ref = ref.targets()
    cols, use = [], np.zeros(sd.shape, dtype=bool)
    for i in range(len(flat)):
        for j, f in enumerate(ref.functions(flat[i], i)):
            if sd[i, j] > 0 and np.ptp(f) > 0:
                cols.append((f - f.mean()) / f.std())
                use[i, j] = True
    F = np.array(cols).T  # (draws, J)
    n, J = F.shape
    raw = (F.T @ F) / n
    target = ref.calibration["trC2"] * J / ref.calibration["functions"]
    a = math.sqrt(max(target - J, 0.0) / ((raw * raw).sum() - J))
    if a > 1:
        raise ValueError(f"{name}: the draws' correlations are weaker than trC2 says (scale {a:.3f})")
    run_se = 1 / math.sqrt(RUN_ESS)
    ref_se = (mcse_ref / sd)[use]
    tot = np.hypot(run_se, ref_se)
    rng = np.random.default_rng(NULL_SEEDS.get(name, 0))
    reps, tol, zr = 5, gate.TOLERANCE, gate._z_right()
    right = refused = done = 0
    full = np.full(sd.size, math.nan)
    mc = [run_se] * sd.size
    while done < groups:
        m = min(chunk, groups - done)
        z = rng.standard_normal((m, reps + 1, n))
        y = math.sqrt(a) * (z @ F) / math.sqrt(n) + math.sqrt(1 - a) * rng.standard_normal((m, reps + 1, J))
        e = y[:, :reps] * run_se - y[:, reps:] * ref_se
        ok = (np.abs(e) + zr * tot < tol).all(axis=(1, 2))
        for g in np.flatnonzero(ok):
            vs = []
            for r in range(reps):
                err = full.copy()
                err[use.ravel()] = e[g, r]
                vs.append(SimpleNamespace(errors=err.tolist(), mcse=mc))
            _, p = gate.calibration(vs, ref)
            refused += not p >= gate.ALPHA_CALIBRATION
        right += int(ok.sum())
        done += m
    return {"model": name, "functions": J, "trC2": target, "raw trC2": float((raw * raw).sum()), "scale": a,
            "groups": groups, "all right": right, "refused": refused}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200, help="runs per setting")
    ap.add_argument("--groups", type=int, default=100, help="groups of 5 reps per calibration setting")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=20, help="runs per reference")
    ap.add_argument("--null", action="append", default=[], metavar="MODEL:GROUPS",
                    help="print only the calibration null for these shipped references")
    a = ap.parse_args(argv)
    if a.null:
        _print_null([calibration_null(m, int(g)) for m, g in (x.split(":") for x in a.null)])
        return 0
    jobs = [(1000 * k + 7 * i, rho, n, a.batch) for k, (rho, n) in enumerate(SETTINGS)
            for i in range(a.runs // a.batch)]
    cal_jobs = [(10**6 + 1000 * k + 7 * i, rho, n, 10) for k, (rho, n) in enumerate([(0.0, 5000), (0.5, 15000)])
                for i in range(a.groups // 10)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for b in ex.map(_batch, jobs) for r in b]
        cal = [r for b in ex.map(_calibration, cal_jobs) for r in b]
        chance = list(ex.map(_disputed_by_chance, [(2 * 10**6 + k, rho, 200) for k, rho in enumerate((0.0, 0.5, 0.9))]))
        exc = list(ex.map(_excursions, [(c, z, n) for c, n in (("projection", 120), ("variable", 150))
                                        for z in (math.inf, gate.SPLIT_Z)]))

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
    print(f"\nHalves of the reference: quantities disputed by chance (SPLIT_Z {gate.SPLIT_Z}), "
          "200 references of 10 chains x 1000 draws per row:\n")
    print("| rho of the reference chains | disputed | of quantities | share |")
    print("|---|---|---|---|")
    for rho, hit, total in chance:
        print(f"| {rho} | {hit} | {total} | {hit / total:.4%} |")
    print("\nCorrect runs against references that may miss rare excursions:\n")
    print("| case | rule | right | unproven | wrong |")
    print("|---|---|---|---|---|")
    for case, z, out in exc:
        rule = "none" if math.isinf(z) else f"SPLIT_Z {z}"
        print(f"| {case} | {rule} | {out['right']} | {out['unproven']} | {out['wrong']} |")
    return 0



def _print_null(rows: list[dict]) -> None:
    print(f"\nCalibration test under an idealised Gaussian null with the correlations of a shipped reference "
          f"(calibration_null): groups of 5 reps, every rep's ESS {RUN_ESS}, cutoff p < {gate.ALPHA_CALIBRATION}. "
          "Only groups in which every rep is right are tested.\n")
    print("| reference | quantities | tr(C^2) | scale of the correlations | groups | all reps right | refused | share |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['functions']} | {r['trC2']:.6g} | {r['scale']:.3f} | {r['groups']} | "
              f"{r['all right']} | {r['refused']} | {r['refused'] / r['all right']:.3%} |")


if __name__ == "__main__":
    raise SystemExit(main())
