"""Score finished runs: ESS, R-hat and cost, per run, per model and per label.

Outputs (in the scores directory):
  per_run.csv    one row per rep
  per_model.csv  median over reps, per label and model
  summary.csv    per label: geometric means over models, R-hat failures
  ratios.csv     with --baseline: per-model ratios of ESS/s against the baseline label

Two wall conventions are reported (PROTOCOL.md):
  ess_per_s_wall   ESS_bulk_min / wall_s (chains in parallel, batch wall clock)
  ess_per_s_chain  ESS_bulk_min / sum of per-chain sampler time
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from . import cmdstan, ess

RHAT_BAD = 1.01


def chain_draws(rep_dir: Path) -> dict[str, np.ndarray]:
    """Variables as (draws, chains) arrays. Ragged chains are cut to the shortest."""
    files = sorted(rep_dir.glob("chain_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"no chain CSVs in {rep_dir}")
    csvs = [cmdstan.read_csv(f) for f in files]
    cols = [c for c in csvs[0].columns if not c.endswith("__")]
    n = min(f.values.shape[0] for f in csvs)
    return {c: np.column_stack([f.values[:n, f.columns.index(c)] for f in csvs]) for c in cols}


def score_rep(rep_dir: Path) -> dict:
    cache = rep_dir / "ess.json"
    if cache.exists() and cache.stat().st_mtime >= (rep_dir / "DONE").stat().st_mtime:
        summary = json.loads(cache.read_text())
    else:
        summary = ess.summarise(chain_draws(rep_dir))
        cache.write_text(json.dumps(summary, indent=1))
    rec = json.loads((rep_dir / "run.json").read_text())
    t = rec["timing"]
    chain_sum = sum(t["chain_s"])
    row = {
        "label": rec["label"], "sampler": rec["sampler"], "model": rec["model"], "rep": rec["rep"],
        "wall_s": t["wall_s"], "chain_s_sum": chain_sum,
        "grads_sampling": t["grads_sampling"], "grads_warmup": t["grads_warmup"], "divergences": t["divergences"],
        **{k: summary[k] for k in ("ess_bulk_min", "ess_bulk_geomean", "ess_tail_min", "rhat_max", "n_variables")},
        "worst_variable": (summary["worst_variables"] or [""])[0],
    }
    eb = row["ess_bulk_min"]
    row["ess_per_s_wall"] = eb / t["wall_s"] if t["wall_s"] else math.nan
    row["ess_per_s_chain"] = eb / chain_sum if chain_sum else math.nan
    row["ess_per_grad"] = eb / t["grads_sampling"] if t["grads_sampling"] else math.nan
    return row


NUMERIC = ("wall_s", "chain_s_sum", "grads_sampling", "divergences", "ess_bulk_min", "ess_bulk_geomean",
           "ess_tail_min", "rhat_max", "ess_per_s_wall", "ess_per_s_chain", "ess_per_grad")


def _median(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else math.nan


def _geomean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else math.nan


def per_model(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["label"], r["model"]), []).append(r)
    out = []
    for (label, model), g in sorted(groups.items()):
        out.append({"label": label, "model": model, "reps": len(g), **{k: _median([r[k] for r in g]) for k in NUMERIC}})
    return out


def summary(models: list[dict]) -> list[dict]:
    out = []
    for label in sorted({m["label"] for m in models}):
        g = [m for m in models if m["label"] == label]
        out.append({
            "label": label, "n_models": len(g),
            "geo_ess_per_s_wall": _geomean([m["ess_per_s_wall"] for m in g]),
            "geo_ess_per_s_chain": _geomean([m["ess_per_s_chain"] for m in g]),
            "geo_ess_per_grad": _geomean([m["ess_per_grad"] for m in g]),
            "geo_ess_bulk_min": _geomean([m["ess_bulk_min"] for m in g]),
            "n_models_rhat_bad": sum(1 for m in g if m["rhat_max"] > RHAT_BAD),
        })
    return out


def ratios(models: list[dict], baseline: str) -> list[dict]:
    base = {m["model"]: m for m in models if m["label"] == baseline}
    if not base:
        raise KeyError(f"baseline label {baseline!r} has no scored runs")
    out = []
    for m in models:
        if m["label"] == baseline or m["model"] not in base:
            continue
        b = base[m["model"]]
        out.append({"label": m["label"], "baseline": baseline, "model": m["model"],
                    **{f"ratio_{k}": (m[k] / b[k] if b[k] else math.nan)
                       for k in ("ess_per_s_wall", "ess_per_s_chain", "ess_per_grad", "ess_bulk_min", "wall_s")}})
    return out


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score(runs_dir: Path, out_dir: Path, labels: list[str] | None = None, baseline: str | None = None,
          log=print) -> dict:
    reps = sorted(p.parent for p in runs_dir.glob("*/*/rep*/DONE"))
    if labels:
        reps = [r for r in reps if r.parent.parent.name in labels]
    if not reps:
        raise FileNotFoundError(f"no finished runs under {runs_dir}")
    rows = [score_rep(r) for r in reps]
    models = per_model(rows)
    summ = summary(models)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "per_run.csv", rows)
    _write(out_dir / "per_model.csv", models)
    _write(out_dir / "summary.csv", summ)
    result = {"per_run": rows, "per_model": models, "summary": summ}
    if baseline:
        rat = ratios(models, baseline)
        _write(out_dir / "ratios.csv", rat)
        result["ratios"] = rat
        for label in sorted({r["label"] for r in rat}):
            g = [r["ratio_ess_per_s_wall"] for r in rat if r["label"] == label]
            log(f"{label} vs {baseline}: geomean ESS/s ratio {_geomean(g):.3f} over {len(g)} models")
    for s in summ:
        log(f"{s['label']}: {s['n_models']} models, geo ESS/s {s['geo_ess_per_s_wall']:.3g} (wall) "
            f"{s['geo_ess_per_s_chain']:.3g} (chain sum), geo ESS/grad {s['geo_ess_per_grad']:.3g}, "
            f"R-hat > {RHAT_BAD}: {s['n_models_rhat_bad']}")
    return result
