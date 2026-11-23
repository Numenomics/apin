#!/usr/bin/env python3
"""Compare the Harbor Stan-style reference (harbor/references/stan_core.py) with real CmdStan.

Two Stan programs, an ill-conditioned diagonal Gaussian (D=10, standard deviations log-spaced from
0.1 to 10) and Neal's funnel (v ~ N(0, 3), nine x_i ~ N(0, exp(v/2))), are sampled by CmdStan and
by the Python reference with the same settings: 100 warmup draws (the Harbor budget), 1000
sampling draws, max tree depth 10, delta 0.8, 16 chains (--chains), from Stan's uniform(-2, 2) inits.

Warmup configurations:
  windows  `adapt init_buffer=15 window=75 term_buffer=10`: the windows CmdStan prints for 100
           warmup draws, passed explicitly so that CmdStan actually uses them. This is what the
           reference does (SHORT_WARMUP_RESTART = True).
  default  plain num_warmup=100. CmdStan 2.37 then never updates the metric (see stan_core.py);
           the reference reproduces this with SHORT_WARMUP_RESTART = False.
  long     num_warmup=1000 with CmdStan's default windows (75/25/50), the normal regime.

Statistics per model and configuration: adapted step size (median over chains), mean tree depth
and mean gradients (leapfrog steps) per sampling draw, and min over parameters of bulk ESS per
sampling gradient. CmdStan is not needed where the analysis runs:

    # where CmdStan is (here: m0pad, which has ~/.cmdstan/cmdstan-2.37.0)
    python3 scripts/compare_stan_reference.py run-cmdstan --cmdstan ~/.cmdstan/cmdstan-2.37.0 --out out/
    # here, with the CSVs copied back
    uv run python scripts/compare_stan_reference.py analyze --cmdstan-csv out/ --json results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAINS = 16
NUM_SAMPLES = 1000
NUM_WARMUP = 100
SEED = 20261123
D = 10
GAUSS_SD = np.logspace(-1.0, 1.0, D)

PROGRAMS = {
    "illcond_gauss": """
data { int<lower=1> D; vector<lower=0>[D] sd_vec; }
parameters { vector[D] x; }
model { x ~ normal(0, sd_vec); }
""",
    "funnel": """
data { int<lower=2> D; }
parameters { real v; vector[D - 1] x; }
model { v ~ normal(0, 3); x ~ normal(0, exp(v / 2)); }
""",
}
DATA = {
    "illcond_gauss": {"D": D, "sd_vec": [float(s) for s in GAUSS_SD]},
    "funnel": {"D": D},
}
CONFIGS = {
    "windows": {"warmup": NUM_WARMUP, "cmdstan": ["adapt", "init_buffer=15", "window=75", "term_buffer=10"],
                "restart": True},
    "default": {"warmup": NUM_WARMUP, "cmdstan": [], "restart": False},
    "long": {"warmup": 1000, "cmdstan": [], "restart": True},
}


# -------------------------------------------------------------------------------------- CmdStan

def run_cmdstan(cmdstan_dir: Path, out: Path) -> None:
    from harness import cmdstan

    root = cmdstan.locate(cmdstan_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"version": cmdstan.version(root), "runs": {}}
    for name, code in PROGRAMS.items():
        stan_file = out / f"{name}.stan"
        stan_file.write_text(code.lstrip(), encoding="utf-8")
        data_file = out / f"{name}.data.json"
        data_file.write_text(json.dumps(DATA[name]), encoding="utf-8")
        exe = cmdstan.compile_model(root, stan_file, out / "build")
        for cfg_name, cfg in CONFIGS.items():
            def argv(c, cfg=cfg, cfg_name=cfg_name, exe=exe, data_file=data_file, name=name):
                extra = ["algorithm=hmc", "engine=nuts", "max_depth=10", "metric=diag_e"]
                return cmdstan.sample_argv(exe, data_file, SEED, c, out / f"{name}_{cfg_name}_{c + 1}.csv",
                                           cfg["warmup"], NUM_SAMPLES, extra=[*cfg["cmdstan"], *extra])
            wall = cmdstan.run_chains(argv, CHAINS)
            meta["runs"][f"{name}_{cfg_name}"] = {"wall_s": round(wall, 2)}
            print(f"{name} {cfg_name}: {wall:.1f}s", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _cmdstan_stats(csv_dir: Path, name: str, cfg_name: str) -> dict:
    from harness import cmdstan

    eps, depth, nlf, draws = [], [], [], []
    inv_metric = []
    for c in range(CHAINS):
        s = cmdstan.read_csv(csv_dir / f"{name}_{cfg_name}_{c + 1}.csv")
        col = {h: i for i, h in enumerate(s.columns)}
        eps.append(float(s.values[0, col["stepsize__"]]))
        depth.append(s.values[:, col["treedepth__"]])
        nlf.append(s.values[:, col["n_leapfrog__"]])
        params = [h for h in s.columns if not h.endswith("__")]
        draws.append(s.values[:, [col[h] for h in params]])
        for i, line in enumerate(s.comments):
            if "Diagonal elements of inverse mass matrix" in line:
                inv_metric.append([float(v) for v in s.comments[i + 1].lstrip("# ").split(",")])
                break
    return _stats(np.array(eps), np.stack(depth), np.stack(nlf), np.stack(draws), np.array(inv_metric))


# ----------------------------------------------------------------------------- Python reference

class _Oracle:
    def __init__(self, name: str):
        self.name = name
        self.dim = D

    def log_density_gradient(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.name == "illcond_gauss":
            return float(-0.5 * np.sum((x / GAUSS_SD) ** 2)), -x / GAUSS_SD**2
        v, xs = x[0], x[1:]
        if not -700.0 < v < 700.0:  # exp overflows; the harness Counter also maps this to -inf
            return -math.inf, np.full(D, math.nan)
        s2 = math.exp(v)
        lp = -v * v / 18.0 - 0.5 * float(np.sum(xs * xs)) / s2 - 0.5 * (D - 1) * v
        g = np.empty(D)
        g[0] = -v / 9.0 + 0.5 * float(np.sum(xs * xs)) / s2 - 0.5 * (D - 1)
        g[1:] = -xs / s2
        return lp, g


def _load_reference():
    path = ROOT / "harbor" / "tasks" / "a0-ground-up" / "tests" / "reference.py"
    spec = importlib.util.spec_from_file_location("apin_stan_reference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apin_stan_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


def _python_stats(name: str, cfg_name: str) -> dict:
    ref = _load_reference()
    cfg = CONFIGS[cfg_name]
    ref.SHORT_WARMUP_RESTART = cfg["restart"]
    oracle = _Oracle(name)
    eps, depth, nlf, draws, inv_metric = [], [], [], [], []
    for c in range(CHAINS):
        rng = np.random.default_rng([SEED, c])
        w = ref.stan_warmup(oracle, cfg["warmup"], rng)
        q, lp, g = w["q"], w["lp"], w["g"]
        d_c, n_c, x_c = [], [], []
        for _ in range(NUM_SAMPLES):
            z, _, traj = ref.nuts_transition(q, lp, g, oracle, w["inv_m"], w["eps"], rng)
            q, lp, g = z.q, z.lp, z.g
            d_c.append(traj.depth)
            n_c.append(traj.n_leapfrog)
            x_c.append(q)
        eps.append(w["eps"])
        inv_metric.append(w["inv_m"])
        depth.append(d_c)
        nlf.append(n_c)
        draws.append(x_c)
    return _stats(np.array(eps), np.array(depth, float), np.array(nlf, float), np.array(draws), np.array(inv_metric))


# --------------------------------------------------------------------------------------- shared

def _stats(eps, depth, nlf, draws, inv_metric) -> dict:
    """draws: (chains, n, params)."""
    from harness import ess

    total_grad = float(np.sum(nlf))
    ess_bulk = [ess.ess_bulk(draws[:, :, j].T) for j in range(draws.shape[2])]
    return {
        "stepsize_median": float(np.median(eps)),
        "stepsize_range": [float(np.min(eps)), float(np.max(eps))],
        "mean_treedepth": float(np.mean(depth)),
        "grad_per_draw": float(np.mean(nlf)),
        "min_ess_bulk": float(np.min(ess_bulk)),
        "min_ess_per_grad": float(np.min(ess_bulk)) / total_grad,
        "inv_metric_median": [round(float(v), 4) for v in np.median(inv_metric, axis=0)] if len(inv_metric) else None,
    }


def analyze(csv_dir: Path | None) -> dict:
    out: dict = {"chains": CHAINS, "num_samples": NUM_SAMPLES, "seed": SEED, "results": {}}
    if csv_dir is not None and (csv_dir / "meta.json").is_file():
        out["cmdstan_version"] = json.loads((csv_dir / "meta.json").read_text())["version"]
    for name in PROGRAMS:
        for cfg_name in CONFIGS:
            key = f"{name}_{cfg_name}"
            t0 = time.perf_counter()
            row = {"python": _python_stats(name, cfg_name)}
            row["python"]["wall_s"] = round(time.perf_counter() - t0, 1)
            if csv_dir is not None:
                row["cmdstan"] = _cmdstan_stats(csv_dir, name, cfg_name)
            out["results"][key] = row
            print(key, json.dumps(row), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("run-cmdstan")
    rc.add_argument("--cmdstan", type=Path, default=None)
    rc.add_argument("--out", type=Path, required=True)
    rc.add_argument("--chains", type=int, default=CHAINS)
    an = sub.add_parser("analyze")
    an.add_argument("--cmdstan-csv", type=Path, default=None)
    an.add_argument("--json", type=Path, default=None)
    an.add_argument("--chains", type=int, default=CHAINS)
    a = ap.parse_args(argv)
    globals()["CHAINS"] = a.chains  # read by the run and stats helpers
    if a.cmd == "run-cmdstan":
        run_cmdstan(a.cmdstan, a.out)
        return 0
    res = analyze(a.cmdstan_csv)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
