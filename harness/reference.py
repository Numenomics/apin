"""Reference posteriors: small summary statistics that the correctness gate reads.

A reference is a JSON file `models/reference/<model>.json`. It holds no draws. It holds the
summaries of a set of *series*, each a scalar function of a posterior draw:

  - every variable the reference varies, standardised: z = (x - loc) / scale;
  - a fixed set of projections u_k . z of the standardised variables (u_k are eigenvectors of
    the reference correlation matrix, the largest and the smallest), which depend on the joint
    distribution and not only on the marginals;
  - lp__, the log density of the draw in CmdStan's convention (BridgeStan with propto=True,
    jacobian=True, on the benchmark model), which also depends on the joint distribution.

For every series the gate checks four functions (FUNCTIONS): its mean, the mean of its square
and the fraction of draws below the reference's 5% and 95% quantiles. The two fractions are
bounded, so a single wild draw cannot hide a bias in them. SERIES_STATS lists what is stored
for each series; every number is computed with the rounded loc, scale and directions that are
stored, so the gate repeats the same arithmetic on a run.

The projections are chosen on the first half of the reference chains and summarised on the
second half, so that choosing the direction of smallest variance does not bias its variance.
Variables that are constant in the reference (the fixed entries of a Cholesky factor) are kept
in `constants` and the gate checks that a run reproduces them.

`calibration` holds J, the number of functions whose reference SD is positive, and tr(C^2) for
their correlation matrix C in the reference; the gate uses it to test whether a run's errors
are as large as its claimed standard errors say (BENCH.md, rule 2).

`models/manifest.json` says where each model's reference comes from (`reference.source`):
"posteriordb" (summarised from posteriordb's reference draws), "local" (built here with long
CmdStan runs) or "none".

  build(model, ...)   write models/reference/<model>.json from the right source
  load(name)          the Reference, or None when the model has none yet
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import cmdstan, ess
from .models import ROOT, Model

REF_DIR = ROOT / "models" / "reference"
PDB = ROOT / "upstream" / "posteriordb" / "posterior_database"
FUNCTIONS = ("mean", "square", "below_q05", "below_q95")
SERIES_STATS = ("loc", "scale", "m1", "sd1", "mcse1", "m2", "sd2", "mcse2", "q05", "q95",
                "p05", "p95", "mcse_p05", "mcse_p95")
MAX_BYTES = 1_000_000
LP = "lp__"

# Settings of the local build: posteriordb's own recipe for Stan references
# (doc/REFERENCE_POSTERIOR_DEFINITION.md: 10 chains, thin 10, 10 000 draws, no
# divergences, R-hat below 1.01), with 2000 instead of 10 000 warmup iterations and
# 20 chains instead of 10, so the half that summarises the projections has 10 chains.
LOCAL = {"chains": 20, "warmup": 2000, "samples": 10000, "thin": 10, "adapt_delta": 0.95, "seed": 4711}

# Parameters that a posteriordb reference leaves out but BridgeStan needs to compute lp__,
# derived from the ones it has.
DERIVED = {
    "eight_schools_noncentered": lambda d: {
        f"theta_trans.{k}": (d[f"theta.{k}"] - d["mu"]) / d["tau"] for k in range(1, 9)},
}


def _round(v, digits: int = 7):
    return np.vectorize(lambda x: float(f"{x:.{digits}g}"), otypes=[float])(np.asarray(v, dtype=float))


@dataclass
class Reference:
    model: str
    source: str
    names: list[str]  # the variables the reference varies, in order
    loc: np.ndarray  # (len(names),) mean of each variable
    scale: np.ndarray  # (len(names),) its SD
    directions: np.ndarray  # (K, len(names)) projections of the standardised variables
    series: list[str]  # names, then proj.1 .. proj.K, then lp__ if the reference has it
    stats: dict[str, np.ndarray]  # SERIES_STATS -> (len(series),)
    constants: dict[str, float] = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)  # {"functions": J, "trC2": tr(C^2)}
    meta: dict = field(default_factory=dict)

    @property
    def has_lp(self) -> bool:
        return LP in self.series

    def standardise(self, draws: dict[str, np.ndarray]) -> np.ndarray:
        """The series of a run: (len(series), draws, chains). draws maps a name to (draws, chains);
        it must hold every variable in `names`, and lp__ when the reference has it."""
        z = np.stack([(np.asarray(draws[n], dtype=float) - self.loc[i]) / self.scale[i]
                      for i, n in enumerate(self.names)])
        rows = [z]
        if len(self.directions):
            raw = np.einsum("kd,dnc->knc", self.directions, z)
            k = len(self.names)
            rows.append((raw - self.stats["loc"][k:k + len(raw), None, None])
                        / self.stats["scale"][k:k + len(raw), None, None])
        if self.has_lp:
            rows.append(((np.asarray(draws[LP], dtype=float) - self.stats["loc"][-1]) / self.stats["scale"][-1])[None])
        return np.concatenate(rows)

    def targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference mean, SD and MCSE of every function: three (len(series), len(FUNCTIONS)) arrays."""
        st = self.stats
        mean = np.column_stack([st["m1"], st["m2"], st["p05"], st["p95"]])
        sd = np.column_stack([st["sd1"], st["sd2"], np.sqrt(st["p05"] * (1 - st["p05"])),
                              np.sqrt(st["p95"] * (1 - st["p95"]))])
        mcse = np.column_stack([st["mcse1"], st["mcse2"], st["mcse_p05"], st["mcse_p95"]])
        return mean, sd, mcse

    def functions(self, s: np.ndarray, i: int) -> list[np.ndarray]:
        """The FUNCTIONS of series i at its standardised draws s."""
        return [s, s * s, (s <= self.stats["q05"][i]).astype(float), (s <= self.stats["q95"][i]).astype(float)]

    def to_json(self) -> str:
        def r(v, digits=7):
            return [float(x) for x in _round(v, digits).ravel()]

        return json.dumps({
            "model": self.model, "source": self.source, "meta": self.meta, "names": self.names,
            "constants": self.constants, "calibration": self.calibration, "series": self.series,
            "loc": r(self.loc), "scale": r(self.scale),
            "directions": [r(u, 4) for u in self.directions],
            **{k: r(self.stats[k]) for k in SERIES_STATS}}, separators=(",", ":"))


def _directions(z: np.ndarray) -> np.ndarray:
    """Eigenvectors of the correlation of z (d, draws, chains): all of them up to 20 variables,
    else the 10 (4 above 500 variables) of largest and of smallest eigenvalue. Rounded to 4
    significant digits, each with its largest entry positive."""
    d = len(z)
    if d < 2:
        return np.zeros((0, d))
    flat = z.reshape(d, -1)
    val, vec = np.linalg.eigh(np.corrcoef(flat))
    k = d if d <= 20 else (10 if d <= 500 else 4)
    pick = list(range(d)) if k == d else list(range(k)) + list(range(d - k, d))
    u = vec[:, pick[::-1]].T  # largest eigenvalue first
    u *= np.sign(u[np.arange(len(u)), np.abs(u).argmax(axis=1)])[:, None]
    return _round(u, 4)


def _halves(n_chains: int, n_draws: int) -> tuple[slice, slice, int]:
    """(axis, first half, second half): split by chains when there are at least two, else by draws."""
    if n_chains >= 2:
        h = n_chains // 2
        return slice(0, h), slice(h, None), 2
    h = n_draws // 2
    return slice(0, h), slice(h, None), 1


def _series_stats(s: np.ndarray, loc: float, scale: float) -> dict[str, float]:
    """SERIES_STATS of one standardised series s (draws, chains)."""
    q05, q95 = (float(v) for v in _round(np.quantile(s, [0.05, 0.95])))
    out = {"loc": loc, "scale": scale}
    for tag, f in (("1", s), ("2", s * s)):
        out["m" + tag] = float(f.mean())
        out["sd" + tag] = float(f.std(ddof=1))
        out["mcse" + tag] = _mcse(f)
    for tag, q in (("05", q05), ("95", q95)):
        ind = (s <= q).astype(float)
        out["p" + tag] = float(ind.mean())
        out["mcse_p" + tag] = _mcse(ind)
    out["q05"], out["q95"] = q05, q95
    return out


def _mcse(f: np.ndarray) -> float:
    return 0.0 if np.ptp(f) == 0 else ess.mcse_mean(f)


def summarise(draws: dict[str, np.ndarray], lp: np.ndarray | None = None, model: str = "", source: str = "",
              meta: dict | None = None) -> Reference:
    """The Reference for reference draws: draws maps name -> (draws, chains); lp is lp__ at the
    same draws, or None."""
    names, constants = [], {}
    for name, x in draws.items():
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError(f"reference draws of {name} are not all finite")
        if np.ptp(x) == 0:
            constants[name] = float(x.flat[0])
        else:
            names.append(name)
    x = np.stack([np.asarray(draws[n], dtype=float) for n in names])
    loc = _round(x.mean(axis=(1, 2)))
    scale = _round(x.std(axis=(1, 2), ddof=1))
    z = (x - loc[:, None, None]) / scale[:, None, None]
    first, second, axis = _halves(z.shape[2], z.shape[1])
    pick = (lambda a, h: a[:, :, h]) if axis == 2 else (lambda a, h: a[:, h, :])
    directions = _directions(pick(z, first))
    rows = [_series_stats(z[i], float(loc[i]), float(scale[i])) for i in range(len(names))]
    series = list(names)
    if len(directions):
        proj = np.einsum("kd,dnc->knc", directions, pick(z, second))
        for k, p in enumerate(proj):
            ploc, pscale = (float(v) for v in _round([p.mean(), p.std(ddof=1)]))
            rows.append(_series_stats((p - ploc) / pscale, ploc, pscale))
            series.append(f"proj.{k + 1}")
    if lp is not None:
        lp = np.asarray(lp, dtype=float)
        lloc, lscale = (float(v) for v in _round([lp.mean(), lp.std(ddof=1)]))
        rows.append(_series_stats((lp - lloc) / lscale, lloc, lscale))
        series.append(LP)
    stats = {k: _round([r[k] for r in rows]) for k in SERIES_STATS}
    ref = Reference(model, source, names, loc, scale, directions, series, stats, constants, {}, meta or {})
    ref.calibration = _calibration(ref, draws, lp)
    return ref


def _calibration(ref: Reference, draws: dict[str, np.ndarray], lp: np.ndarray | None, most: int = 2000) -> dict:
    """J and tr(C^2) of the functions with positive reference SD, C their correlation over draws.

    tr(C^2) = E[(f_a . f_b)^2] for independent draws a, b of the standardised function vector f,
    so it is estimated from the pairs of an evenly thinned subset of at most `most` draws."""
    full = {**{n: draws[n] for n in ref.names}, **({LP: lp} if lp is not None else {})}
    s = ref.standardise(full)
    flat = s.reshape(len(s), -1)
    keep = np.unique(np.linspace(0, flat.shape[1] - 1, min(most, flat.shape[1])).astype(int))
    flat = flat[:, keep]
    _, sd, _ = ref.targets()
    cols = []
    for i in range(len(flat)):
        for j, f in enumerate(ref.functions(flat[i], i)):
            if sd[i, j] > 0 and np.ptp(f) > 0:
                cols.append(((f - f.mean()) / f.std()).astype(np.float32))
    f = np.array(cols).T  # (draws, J)
    g = (f @ f.T).astype(float)
    n = len(g)
    tr = float(((g * g).sum() - (np.diag(g) ** 2).sum()) / (n * (n - 1)))
    return {"functions": int(f.shape[1]), "trC2": float(f"{tr:.6g}")}


def path(name: str) -> Path:
    return REF_DIR / f"{name}.json"


def load(name: str) -> Reference | None:
    p = path(name)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    a = lambda v: np.array(v, dtype=float)  # noqa: E731
    return Reference(d["model"], d["source"], d["names"], a(d["loc"]), a(d["scale"]),
                     a(d["directions"]).reshape(len(d["directions"]), len(d["names"])), d["series"],
                     {k: a(d[k]) for k in SERIES_STATS}, d["constants"], d["calibration"], d["meta"])


def stan_name(pdb_name: str) -> str:
    """posteriordb's `theta[1]`, `L[2,1]` -> the CSV style `theta.1`, `L.2.1`."""
    return re.sub(r"\[([^\]]*)\]", lambda m: "." + m.group(1).replace(",", "."), pdb_name).replace(" ", "")


def posteriordb_name(model: Model) -> str | None:
    """Name of posteriordb's reference posterior for this model, or None."""
    info = PDB / "posteriors" / f"{model.posterior}.json"
    if not info.is_file():
        raise FileNotFoundError(f"{info} missing; run `git submodule update --init upstream/posteriordb`")
    return json.loads(info.read_text()).get("reference_posterior_name")


def from_posteriordb(model: Model, build_dir: Path = ROOT / "build") -> Reference:
    ref = posteriordb_name(model)
    if not ref:
        raise LookupError(f"posteriordb has no reference posterior for {model.posterior}")
    with zipfile.ZipFile(PDB / "reference_posteriors" / "draws" / "draws" / f"{ref}.json.zip") as z:
        chains = json.loads(z.read(f"{ref}.json"))  # list of chains, each {variable: [draws]}
    draws = {stan_name(v): np.column_stack([c[v] for c in chains]) for v in chains[0]}
    info = json.loads((PDB / "reference_posteriors" / "draws" / "info" / f"{ref}.info.json").read_text())
    meta = {"posteriordb_reference": ref, "inference": info.get("inference"),
            "draws": int(next(iter(draws.values())).size), "chains": len(chains), "lp": LP_CONVENTION}
    return summarise(draws, log_density(model, draws, build_dir), model.name, "posteriordb", meta)


LP_CONVENTION = "BridgeStan log_density(propto=True, jacobian=True) on the benchmark model, as CmdStan's lp__"


def log_density(model: Model, draws: dict[str, np.ndarray], build_dir: Path, strict: bool = True) -> np.ndarray:
    """lp__ of the benchmark model at each draw (draws, chains), the way CmdStan writes it.

    Computed with BridgeStan from the constrained parameters, so a reference sampled from a
    reparameterisation (reference_stan) gets the benchmark model's lp__, and a run's lp__ does not
    depend on what its sampler wrote. With strict=False a draw that is non-finite, outside the
    support or rejected by the model gets NaN, which the gate reads as a hard failure."""
    import bridgestan

    from . import stan_library

    lib = stan_library.build(stan_library.tree(), model.stan_file, build_dir)
    sm = bridgestan.StanModel(str(lib), str(model.data_file()), warn=False, capture_stan_prints=True)
    full = {**draws, **(DERIVED[model.name](draws) if model.name in DERIVED else {})}
    names = list(sm.param_names())
    missing = [n for n in names if n not in full]
    if missing:
        raise KeyError(f"{model.name}: the reference draws lack {missing[:5]}, needed for lp__")
    x = np.stack([np.asarray(full[n], dtype=float) for n in names], axis=-1)
    out = np.empty(x.shape[:2])
    for i, j in np.ndindex(out.shape):
        try:
            if not np.all(np.isfinite(x[i, j])):
                raise ValueError("non-finite draw")
            out[i, j] = sm.log_density(sm.param_unconstrain(x[i, j]), propto=True, jacobian=True)
        except (RuntimeError, ValueError):
            if strict:
                raise
            out[i, j] = math.nan
    return out


def from_cmdstan(model: Model, root: Path, build_dir: Path, work_dir: Path, settings: dict | None = None,
                 log=print, rerun: bool = False) -> Reference:
    """Long CmdStan runs, summarised. Raises if the result fails posteriordb's checks.

    Reuses the chain CSVs in work_dir when all of them are there, complete and made with these
    settings (their CmdStan header says so), unless rerun. Any other CSV is sampled again."""
    s = settings_for(model, settings)
    work_dir.mkdir(parents=True, exist_ok=True)
    csvs = [work_dir / f"ref_{c}.csv" for c in range(s["chains"])]
    files = None if rerun else _complete(csvs, s)
    wall = None
    if files is None:
        exe = cmdstan.compile_model(root, model.reference_stan_file, build_dir)
        extra = [f"thin={s['thin']}", "adapt", f"delta={s['adapt_delta']}"]
        log(f"{model.name}: {s['chains']} chains x ({s['warmup']} warmup + {s['samples']} draws, thin {s['thin']})")
        wall = cmdstan.run_chains(lambda c: cmdstan.sample_argv(exe, model.data_file(), s["seed"], c, csvs[c],
                                                                s["warmup"], s["samples"], extra=extra), s["chains"])
        files = [cmdstan.read_csv(p) for p in csvs]
    else:
        log(f"{model.name}: reusing {len(csvs)} chain CSVs in {work_dir}")
    params = set(cmdstan.parameter_names(root, model.stan_file))
    cols = [c for c in files[0].columns if not c.endswith("__") and c.split(".")[0] in params]
    absent = params - {c.split(".")[0] for c in cols}
    if absent:
        raise RuntimeError(f"{model.reference_stan_file.name} does not output {sorted(absent)} of {model.name}")
    draws = {c: np.column_stack([f.column(c) for f in files]) for c in cols}
    divergences = int(sum(f.column("divergent__").sum() for f in files))
    varying = [x for x in draws.values() if np.ptp(x) > 0]
    rhat = max(ess.rhat(x) for x in varying)
    checks = {"rhat_max": rhat, "ess_bulk_min": min(ess.ess_bulk(x) for x in varying),
              "ess_tail_min": min(ess.ess_tail(x) for x in varying), "divergences": divergences}
    problems = []
    if divergences:
        problems.append(f"{divergences} divergent transitions")
    if not rhat < 1.01:
        problems.append(f"R-hat {rhat:.4f} >= 1.01")
    if problems:
        raise RuntimeError(f"{model.name}: reference run fails posteriordb's checks: {'; '.join(problems)}")
    meta = {"cmdstan": cmdstan.version(root), "stan_file": str(model.reference_stan_file.relative_to(ROOT)),
            "settings": s, "checks": checks, "wall_s": None if wall is None else round(wall, 1),
            "draws": int(next(iter(draws.values())).size), "variables": "parameters block", "lp": LP_CONVENTION}
    return summarise(draws, log_density(model, draws, build_dir), model.name, "local", meta)


def settings_for(model: Model, settings: dict | None = None) -> dict:
    """The CmdStan settings of a local reference: LOCAL, then the model's manifest settings
    (reference.settings), then `settings`."""
    return {**LOCAL, **model.reference_settings, **(settings or {})}


def _config(comments: list[str]) -> dict[str, str]:
    """The `key = value` lines of a CmdStan CSV header, without "(Default)"."""
    out = {}
    for c in comments:
        k, eq, v = c.lstrip("# ").partition(" = ")
        if eq:
            out.setdefault(k.strip(), v.replace("(Default)", "").strip())
    return out


def _made_with(f: cmdstan.StanCsv, s: dict, chain: int) -> bool:
    """Whether a chain CSV was made with settings `s` as chain `chain` (CmdStan's id is chain + 1)."""
    want = {"num_samples": s["samples"], "num_warmup": s["warmup"], "thin": s["thin"], "delta": s["adapt_delta"],
            "seed": s["seed"], "id": chain + 1}
    got = _config(f.comments)
    try:
        return all(k in got and float(got[k]) == float(v) for k, v in want.items())
    except ValueError:
        return False


def _complete(csvs: list[Path], s: dict) -> list | None:
    """The chain CSVs, when every one is there, readable, complete and made with settings `s`;
    otherwise None (a cache miss)."""
    files = []
    for c, p in enumerate(csvs):
        try:
            f = cmdstan.read_csv(p)
        except (OSError, ValueError, cmdstan.CmdStanError):
            return None
        if len(f.values) != s["samples"] // s["thin"] or not _made_with(f, s, c):
            return None
        files.append(f)
    return files


def build(model: Model, source: str, cmdstan_root: str | None = None, build_dir: Path = ROOT / "build",
          work_dir: Path | None = None, settings: dict | None = None, log=print, rerun: bool = False) -> Path:
    """Write models/reference/<model>.json from `source` ("posteriordb" or "local")."""
    if source == "posteriordb":
        ref = from_posteriordb(model, build_dir)
    elif source == "local":
        work = work_dir or build_dir / "reference" / model.name
        ref = from_cmdstan(model, cmdstan.locate(cmdstan_root), build_dir, work, settings, log, rerun)
    else:
        raise ValueError(f"{model.name}: reference source is {source!r}; nothing to build")
    text = ref.to_json()
    if len(text.encode()) > MAX_BYTES:
        raise RuntimeError(f"{model.name}: reference is {len(text.encode())} bytes, over the 1 MB limit")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    out = path(model.name)
    out.write_text(text + "\n")
    log(f"{model.name}: {len(ref.names)} variables, {len(ref.series)} series, {len(text.encode())} bytes "
        f"from {source} -> {out.relative_to(ROOT)}")
    return out
