"""Reference posteriors: small summary statistics that the correctness gate reads.

A reference is a JSON file `models/reference/<model>.json`. It holds no draws,
only, for each variable, the moments the gate checks and their Monte Carlo
standard errors (MCSE):

  mean, sd, mcse_mean          of x
  sq_mean, sq_sd, mcse_sq      of x^2
  var, var_sd, mcse_var        of (x - mean)^2

`models/manifest.json` says where each model's reference comes from
(`reference.source`): "posteriordb" (summarised from posteriordb's reference
draws), "local" (built here with long CmdStan runs) or "none".

  build(model, ...)   write models/reference/<model>.json from the right source
  load(name)          the Reference, or None when the model has none yet
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cmdstan, ess
from .models import ROOT, Model

REF_DIR = ROOT / "models" / "reference"
PDB = ROOT / "upstream" / "posteriordb" / "posterior_database"
STATS = ("mean", "sd", "mcse_mean", "sq_mean", "sq_sd", "mcse_sq", "var", "var_sd", "mcse_var")
MAX_BYTES = 1_000_000

# Settings of the local build: posteriordb's own recipe for Stan references
# (doc/REFERENCE_POSTERIOR_DEFINITION.md: 10 chains, thin 10, 10 000 draws, no
# divergences, R-hat below 1.01), with 2000 instead of 10 000 warmup iterations.
LOCAL = {"chains": 10, "warmup": 2000, "samples": 10000, "thin": 10, "adapt_delta": 0.95, "seed": 4711}


@dataclass
class Reference:
    model: str
    source: str
    names: list[str]
    stats: dict[str, np.ndarray]  # each STATS key -> array over names
    meta: dict

    def to_json(self) -> str:
        def r(v):
            return float(f"{v:.9g}")

        return json.dumps({"model": self.model, "source": self.source, "meta": self.meta, "names": self.names,
                           **{k: [r(v) for v in self.stats[k]] for k in STATS}}, indent=0)


def summarise(draws: dict[str, np.ndarray]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Per-variable moments and MCSEs. draws maps name -> (draws, chains).

    Constant variables (zero variance, such as the fixed entries of a
    Cholesky factor) are left out: no bias can be standardised by them.
    """
    names, cols = [], {k: [] for k in STATS}
    for name, x in draws.items():
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
            continue
        m = float(x.mean())
        dev = (x - m) ** 2
        vals = {"mean": m, "sd": float(x.std(ddof=1)), "mcse_mean": ess.mcse_mean(x),
                "sq_mean": float((x**2).mean()), "sq_sd": float((x**2).std(ddof=1)), "mcse_sq": ess.mcse_mean(x**2),
                "var": float(dev.mean()), "var_sd": float(dev.std(ddof=1)), "mcse_var": ess.mcse_mean(dev)}
        names.append(name)
        for k in STATS:
            cols[k].append(vals[k])
    return names, {k: np.array(v) for k, v in cols.items()}


def path(name: str) -> Path:
    return REF_DIR / f"{name}.json"


def load(name: str) -> Reference | None:
    p = path(name)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return Reference(d["model"], d["source"], d["names"], {k: np.array(d[k], dtype=float) for k in STATS}, d["meta"])


def stan_name(pdb_name: str) -> str:
    """posteriordb's `theta[1]`, `L[2,1]` -> the CSV style `theta.1`, `L.2.1`."""
    return re.sub(r"\[([^\]]*)\]", lambda m: "." + m.group(1).replace(",", "."), pdb_name).replace(" ", "")


def posteriordb_name(model: Model) -> str | None:
    """Name of posteriordb's reference posterior for this model, or None."""
    info = PDB / "posteriors" / f"{model.posterior}.json"
    if not info.is_file():
        raise FileNotFoundError(f"{info} missing; run `git submodule update --init upstream/posteriordb`")
    return json.loads(info.read_text()).get("reference_posterior_name")


def from_posteriordb(model: Model) -> Reference:
    ref = posteriordb_name(model)
    if not ref:
        raise LookupError(f"posteriordb has no reference posterior for {model.posterior}")
    with zipfile.ZipFile(PDB / "reference_posteriors" / "draws" / "draws" / f"{ref}.json.zip") as z:
        chains = json.loads(z.read(f"{ref}.json"))  # list of chains, each {variable: [draws]}
    draws = {stan_name(v): np.column_stack([c[v] for c in chains]) for v in chains[0]}
    names, stats = summarise(draws)
    info = json.loads((PDB / "reference_posteriors" / "draws" / "info" / f"{ref}.info.json").read_text())
    meta = {"posteriordb_reference": ref, "inference": info.get("inference"),
            "draws": int(draws[names[0]].size), "chains": len(chains)}
    return Reference(model.name, "posteriordb", names, stats, meta)


def from_cmdstan(model: Model, root: Path, build_dir: Path, work_dir: Path, settings: dict | None = None,
                 log=print) -> Reference:
    """Long CmdStan runs, summarised. Raises if the result fails posteriordb's checks."""
    s = {**LOCAL, **(settings or {})}
    exe = cmdstan.compile_model(root, model.reference_stan_file, build_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    csvs = [work_dir / f"ref_{c}.csv" for c in range(s["chains"])]
    extra = [f"thin={s['thin']}", "adapt", f"delta={s['adapt_delta']}"]
    log(f"{model.name}: {s['chains']} chains x ({s['warmup']} warmup + {s['samples']} draws, thin {s['thin']})")
    wall = cmdstan.run_chains(lambda c: cmdstan.sample_argv(exe, model.data_file(), s["seed"], c, csvs[c],
                                                            s["warmup"], s["samples"], extra=extra), s["chains"])
    files = [cmdstan.read_csv(p) for p in csvs]
    params = set(cmdstan.parameter_names(root, model.stan_file))
    cols = [c for c in files[0].columns if not c.endswith("__") and c.split(".")[0] in params]
    absent = params - {c.split(".")[0] for c in cols}
    if absent:
        raise RuntimeError(f"{model.reference_stan_file.name} does not output {sorted(absent)} of {model.name}")
    draws = {c: np.column_stack([f.column(c) for f in files]) for c in cols}
    divergences = int(sum(f.column("divergent__").sum() for f in files))
    rhat = max(ess.rhat(x) for x in draws.values() if np.ptp(x) > 0)
    bulk = min(ess.ess_bulk(x) for x in draws.values() if np.ptp(x) > 0)
    tail = min(ess.ess_tail(x) for x in draws.values() if np.ptp(x) > 0)
    checks = {"rhat_max": rhat, "ess_bulk_min": bulk, "ess_tail_min": tail, "divergences": divergences}
    problems = []
    if divergences:
        problems.append(f"{divergences} divergent transitions")
    if not rhat < 1.01:
        problems.append(f"R-hat {rhat:.4f} >= 1.01")
    if problems:
        raise RuntimeError(f"{model.name}: reference run fails posteriordb's checks: {'; '.join(problems)}")
    names, stats = summarise(draws)
    meta = {"cmdstan": cmdstan.version(root), "stan_file": str(model.reference_stan_file.relative_to(ROOT)),
            "settings": s, "checks": checks, "wall_s": round(wall, 1),
            "draws": int(draws[names[0]].size), "variables": "parameters block"}
    return Reference(model.name, "local", names, stats, meta)


def build(model: Model, source: str, cmdstan_root: str | None = None, build_dir: Path = ROOT / "build",
          work_dir: Path | None = None, settings: dict | None = None, log=print) -> Path:
    """Write models/reference/<model>.json from `source` ("posteriordb" or "local")."""
    if source == "posteriordb":
        ref = from_posteriordb(model)
    elif source == "local":
        work = work_dir or build_dir / "reference" / model.name
        ref = from_cmdstan(model, cmdstan.locate(cmdstan_root), build_dir, work, settings, log)
    else:
        raise ValueError(f"{model.name}: reference source is {source!r}; nothing to build")
    text = ref.to_json()
    if len(text.encode()) > MAX_BYTES:
        raise RuntimeError(f"{model.name}: reference is {len(text.encode())} bytes, over the 1 MB limit")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    out = path(model.name)
    out.write_text(text + "\n")
    log(f"{model.name}: {len(ref.names)} variables from {source} -> {out.relative_to(ROOT)}")
    return out
