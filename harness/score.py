"""Score finished runs over a declared grid of models and reps (BENCH.md, rules 1, 2 and 4).

    score(runs_dir, out_dir, grid, baseline) -> per_run, per_model, ranking, table

The grid is every model with a reference posterior and its budget (models/manifest.json), times
`reps` reps whose seeds follow from the base seed. A rep that is missing, failed, or was run with
another seed or budget is wrong. A model is right only when every rep is right and the reps pass
the calibration test (gate.judge); only a right model earns speed credit.

Outputs, in out_dir:
  per_run.csv    one row per rep of the grid: verdict, credited ESS, evaluations, cost
  per_model.csv  one row per label and model: verdict and mean cost over the reps
  ranking.csv    per label: models right, then speed against the baseline
  table.md       the ranking and the per-model table

Cost: `ess` is gate.Verdict.ess, the smallest bulk or tail ESS over every series the gate checks
(variables, projections and lp__). ess_per_eval divides it by every density and gradient
evaluation, warmup included; ess_per_s by the sum of the chains' seconds (one core).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from . import cmdstan, gate, grid as grid_mod, models as models_mod, reference
from .reference import LP, Reference

BOOTSTRAP = 2000


@dataclass(frozen=True)
class Entry:
    """One model of the grid: its reference, its budget, and lp__ of constrained draws."""

    model: str
    ref: Reference
    chains: int
    warmup: int
    draws: int
    lp: Callable[[dict], np.ndarray]


@dataclass(frozen=True)
class Grid:
    entries: dict[str, Entry]
    reps: int
    base_seed: int = grid_mod.BASE_SEED


def declared(spec: str = "all", reps: int = 5, base_seed: int = grid_mod.BASE_SEED,
             build_dir: Path = models_mod.ROOT / "build") -> Grid:
    """The benchmark's grid: the selected models that have a reference, at their manifest budget."""
    entries = {}
    for m in models_mod.select(spec):
        ref = reference.load(m.name)
        if ref is None:
            continue  # no reference, so not in the grid (BENCH.md, "Models without a reference")
        entries[m.name] = Entry(m.name, ref, m.chains, m.warmup, m.draws,
                                lambda d, m=m: reference.log_density(m, d, build_dir, strict=False))
    return Grid(entries, reps, base_seed)


def chain_draws(rep_dir: Path, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """The named columns (default: all but the sampler's `__` columns) as (draws, chains) arrays.
    A name missing from a chain is left out. Ragged chains are cut to the shortest."""
    files = sorted(rep_dir.glob("chain_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"no chain CSVs in {rep_dir}")
    csvs = [cmdstan.read_csv(f) for f in files]
    n = min(f.values.shape[0] for f in csvs)
    if names is None:
        names = [c for c in csvs[0].columns if not c.endswith("__")]
    return {c: np.column_stack([f.values[:n, f.columns.index(c)] for f in csvs])
            for c in names if all(c in f.columns for f in csvs)}


def _key(entry: Entry, reps: int) -> dict:
    """What a cached verdict depends on besides the draws."""
    consts = {k: getattr(gate, k) for k in ("TOLERANCE", "ALPHA_WRONG", "ALPHA_RIGHT", "RHAT_MAX", "MIN_ESS",
                                            "EXACT", "COPY_DISTANCE", "COPY_FRACTION", "COPY_SERIES")}
    return {"reps": reps, "reference": hashlib.sha256(entry.ref.to_json().encode()).hexdigest(), "gate": consts}


def verdict(rep_dir: Path, entry: Entry, reps: int) -> gate.Verdict:
    """The gate's verdict on a finished rep, cached in rep_dir/gate.json. lp__ is computed by the
    harness from the constrained draws; a column of that name written by the sampler is ignored."""
    key = _key(entry, reps)
    cache = rep_dir / "gate.json"
    if cache.is_file():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return gate.Verdict.from_dict(c["verdict"])
    ref = entry.ref
    draws = chain_draws(rep_dir, list(ref.names) + list(ref.constants))
    if ref.has_lp and all(n in draws for n in ref.names):
        draws[LP] = entry.lp(draws)
    v = gate.check(draws, ref, reps=reps)
    cache.write_text(json.dumps({"key": key, "verdict": v.to_dict()}))
    return v


def evaluation_counts(t: dict) -> tuple[int | None, int | None, int | None]:
    """(all evaluations, density calls, gradient calls), None where unknown."""
    extra = t.get("extra") or {}
    total = t.get("evaluations")
    if total is None and t.get("grads_sampling") is not None and t.get("grads_warmup") is not None:
        total = t["grads_sampling"] + t["grads_warmup"]
    dens, grad = extra.get("density_evals"), extra.get("gradient_evals")
    if total is not None and dens is None and grad is None and t.get("evaluations") is None:
        dens, grad = 0, total  # a gradient-only sampler (CmdStan NUTS)
    return total, dens, grad


def score_rep(label_dir: Path, entry: Entry, rep: int, g: Grid) -> dict:
    d = label_dir / entry.model / f"rep{rep}"
    row = {"label": label_dir.name, "model": entry.model, "rep": rep, "status": "wrong", "reason": "",
           "identity": "", "ess": math.nan, "evaluations": None, "density_evals": None, "gradient_evals": None,
           "seconds": math.nan, "ess_per_eval": math.nan, "ess_per_s": math.nan, "verdict": None}

    def wrong(reason):
        row.update(reason=reason, verdict=gate.Verdict("wrong", reason))
        return row

    if (d / "FAILED").is_file():
        err = (d / "FAILED").read_text().strip().splitlines()
        return wrong(f"rep{rep} failed: {err[-1][:120] if err else ''}")
    if not (d / "DONE").is_file():
        return wrong(f"rep{rep} missing")
    rec = json.loads((d / "run.json").read_text())
    row["identity"] = identity_digest(rec.get("identity", {}))
    want_seed = grid_mod.seed_for(rep, g.base_seed)
    if rec.get("seed") != want_seed:
        return wrong(f"rep{rep} seed {rec.get('seed')}, the grid's is {want_seed}")
    got = (rec.get("chains"), rec.get("warmup"), rec.get("draws"))
    want = (entry.chains, entry.warmup, entry.draws)
    if got != want:
        return wrong(f"rep{rep} budget {got} (chains, warmup, draws), the grid's is {want}")
    v = verdict(d, entry, g.reps)
    t = rec["timing"]
    total, dens, grad = evaluation_counts(t)
    secs = float(sum(t["chain_s"]))
    row.update(status=v.status, reason=v.label(), ess=v.ess, evaluations=total, density_evals=dens,
               gradient_evals=grad, seconds=secs, verdict=v)
    if v.status == "right":
        row["ess_per_eval"] = v.ess / total if total else math.nan
        row["ess_per_s"] = v.ess / secs if secs > 0 else math.nan
    return row


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def per_model(rows: list[dict], g: Grid) -> list[dict]:
    out = []
    for (label, model) in dict.fromkeys((r["label"], r["model"]) for r in rows):
        reps = sorted((r for r in rows if r["label"] == label and r["model"] == model), key=lambda r: r["rep"])
        m = gate.judge([r["verdict"] for r in reps], g.entries[model].ref, g.reps)
        right = m.status == "right"
        out.append({"label": label, "model": model, "status": m.status, "reason": m.reason,
                    "calibration": m.calibration, "calibration_p": m.calibration_p,
                    "eval": [r["ess_per_eval"] for r in reps] if right else [],
                    "sec": [r["ess_per_s"] for r in reps] if right else [],
                    "ess_per_eval": _mean([r["ess_per_eval"] for r in reps]) if right else math.nan,
                    "ess_per_s": _mean([r["ess_per_s"] for r in reps]) if right else math.nan})
    return out


def _mean(v) -> float:
    return float(np.mean(v)) if v and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) else math.nan


def rank(models: list[dict], baseline: str | None, names: list[str], seed: int = 0) -> list[dict]:
    """Labels ranked by models right (over every model of the grid), then by speed against the
    baseline: the geometric mean over the models where both are right of the ratio of mean cost.
    The 95 % interval resamples the reps of the label and of the baseline. A right model whose
    cost is unavailable leaves the speed undefined (S12)."""
    by = {(m["label"], m["model"]): m for m in models}
    labels = list(dict.fromkeys(m["label"] for m in models))
    out = []
    for lab in labels:
        right = [n for n in names if by.get((lab, n), {}).get("status") == "right"]
        row = {"label": lab, "right": len(right), "of": len(names), "note": ""}
        for unit in ("eval", "sec"):
            key = "speed_eval" if unit == "eval" else "speed_s"
            row.update({key: math.nan, f"{key}_lo": math.nan, f"{key}_hi": math.nan})
            if baseline is None:
                continue
            both = [n for n in right if by.get((baseline, n), {}).get("status") == "right"]
            row["compared"] = len(both)
            pairs = [(by[(lab, n)][unit], by[(baseline, n)][unit]) for n in both]
            bad = [n for n, (a, b) in zip(both, pairs) if not _finite(a) or not _finite(b)]
            if bad:
                if unit == "eval":
                    row["note"] = f"cost unavailable on {', '.join(bad)}"
                continue
            if not pairs:
                continue
            row[key] = _geo([np.mean(a) / np.mean(b) for a, b in pairs])
            row[f"{key}_lo"], row[f"{key}_hi"] = _bootstrap(pairs, seed)
        out.append(row)
    out.sort(key=lambda r: (-r["right"], -(r["speed_eval"] if math.isfinite(r["speed_eval"]) else -math.inf)))
    for i, r in enumerate(out):
        r["rank"] = i + 1
    return out


def _finite(v) -> bool:
    return len(v) > 0 and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in v)


def _geo(v) -> float:
    return float(math.exp(np.mean(np.log(v))))


def _bootstrap(pairs, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    logs = np.zeros(BOOTSTRAP)
    for a, b in pairs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ma = a[rng.integers(0, len(a), (BOOTSTRAP, len(a)))].mean(axis=1)
        mb = b[rng.integers(0, len(b), (BOOTSTRAP, len(b)))].mean(axis=1)
        logs += np.log(ma / mb)
    lo, hi = np.percentile(logs / len(pairs), [2.5, 97.5])
    return float(math.exp(lo)), float(math.exp(hi))


def table(ranking: list[dict], models: list[dict], baseline: str | None, names: list[str]) -> str:
    labels = [r["label"] for r in ranking]
    lines = ["| rank | label | models right | speed per evaluation [95 %] | speed per second [95 %] | note |",
             "|---|---|---|---|---|---|"]
    for r in ranking:
        lines.append(f"| {r['rank']} | {r['label']} | {r['right']} of {r['of']} | {_speed(r, 'speed_eval')} | "
                     f"{_speed(r, 'speed_s')} | {r['note']} |")
    lines += ["", f"Rank by models right, then by speed: the geometric mean, over the models where the label "
              f"and {baseline or 'the baseline'} are both right, of the ratio of ESS per evaluation (per second "
              f"of one core in the next column).", ""]
    by = {(m["label"], m["model"]): m for m in models}
    lines += ["| model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for n in names:
        cells = []
        for lab in labels:
            m = by.get((lab, n))
            if m is None:
                cells.append("")
            elif m["status"] == "right":
                cells.append(f"right: {m['ess_per_eval']:.3g}/eval, {m['ess_per_s']:.3g}/s")
            else:
                cells.append(f"{m['status']}: {m['reason']}" if not m["reason"].startswith(m["status"])
                             else m["reason"])
        lines.append(f"| {n} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    lines += ["", "Cells: mean over reps of credited ESS (the smallest bulk or tail ESS over every checked "
              "series) per density or gradient evaluation, warmup included, and per second of chain time; "
              "or the verdict and the reason."]
    return "\n".join(lines) + "\n"


def _speed(r, key) -> str:
    if not math.isfinite(r[key]):
        return ""
    return f"{r[key]:.3g} [{r[key + '_lo']:.3g}, {r[key + '_hi']:.3g}]"


def _write(path: Path, rows: list[dict], drop=("verdict", "eval", "sec")) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r if k not in drop))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k not in drop} for r in rows)


def score(runs_dir: Path, out_dir: Path, g: Grid, baseline: str | None = None, labels: list[str] | None = None,
          log=print) -> dict:
    """Score every label under runs_dir over the grid g. A label whose reps record different
    sampler identities (another submission, other options) is refused."""
    label_dirs = sorted(p for p in Path(runs_dir).iterdir() if p.is_dir() and (not labels or p.name in labels))
    if baseline and baseline not in [p.name for p in label_dirs]:
        raise KeyError(f"baseline label {baseline!r} has no runs under {runs_dir}")
    names = list(g.entries)
    rows = []
    for ld in label_dirs:
        lab = [score_rep(ld, g.entries[n], r, g) for n in names for r in range(g.reps)]
        ids = {r["identity"] for r in lab if r["identity"]}
        if len(ids) > 1:
            raise ValueError(f"label {ld.name!r} mixes {len(ids)} sampler identities (submission or options); "
                             "give each its own label")
        rows += lab
    models = per_model(rows, g)
    ranking = rank(models, baseline, names)
    tab = table(ranking, models, baseline, names)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "per_run.csv", rows)
    _write(out_dir / "per_model.csv", models)
    _write(out_dir / "ranking.csv", ranking)
    (out_dir / "table.md").write_text(tab)
    log(tab)
    return {"per_run": rows, "per_model": models, "ranking": ranking, "table": tab}
