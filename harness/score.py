"""Score finished runs: the correctness gate, ESS, R-hat and cost, per run, per model and per label.

Outputs (in the scores directory):
  per_run.csv    one row per rep, with the gate verdict
  per_model.csv  median over the reps that passed, per label and model
  summary.csv    per label: geometric means over models, R-hat failures, wrong models
  ratios.csv     with --baseline: per-model ratios against the baseline label
  table.md       the per-model table, with "wrong" and the worst parameter for a model that failed

A rep that fails the gate (harness/gate.py), or that raised (FAILED), gets no
speed score: its cost columns are NaN. A model is "wrong" for a label when
more than half of its reps are wrong or failed.

Cost conventions (PROTOCOL.md):
  ess_per_s_wall   ESS_bulk_min / wall_s (chains in parallel, batch wall clock)
  ess_per_s_chain  ESS_bulk_min / sum of per-chain sampler time
  ess_per_grad     ESS_bulk_min / sampling-phase gradients (the old campaign's number)
  ess_per_eval     min(bulk, tail ESS) / every density and gradient evaluation, warmup
                   included (the benchmark's number, BENCH.md)
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from . import cmdstan, ess, gate, reference

RHAT_BAD = 1.01
PENALTY = 1e-3  # relative score of a wrong or failed model in the benchmark (BENCH.md)
COST = ("ess_per_s_wall", "ess_per_s_chain", "ess_per_grad", "ess_per_eval")


def chain_draws(rep_dir: Path) -> dict[str, np.ndarray]:
    """Variables as (draws, chains) arrays. Ragged chains are cut to the shortest."""
    files = sorted(rep_dir.glob("chain_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"no chain CSVs in {rep_dir}")
    csvs = [cmdstan.read_csv(f) for f in files]
    cols = [c for c in csvs[0].columns if not c.endswith("__")]
    n = min(f.values.shape[0] for f in csvs)
    return {c: np.column_stack([f.values[:n, f.columns.index(c)] for f in csvs]) for c in cols}


def evaluations(t: dict) -> int | None:
    """Every density and gradient evaluation of a run, warmup included, or None if unknown."""
    if t.get("evaluations") is not None:
        return t["evaluations"]
    if t.get("grads_sampling") is not None and t.get("grads_warmup") is not None:
        return t["grads_sampling"] + t["grads_warmup"]
    return None


def _cached(path: Path, newer_than: list[Path], compute):
    stamp = max(p.stat().st_mtime for p in newer_than if p.exists())
    if path.exists() and path.stat().st_mtime >= stamp:
        return json.loads(path.read_text())
    value = compute()
    path.write_text(json.dumps(value, indent=1))
    return value


def score_rep(rep_dir: Path) -> dict:
    rec = json.loads((rep_dir / "run.json").read_text())
    model = rec["model"]
    draws = None

    def load():
        nonlocal draws
        if draws is None:
            draws = chain_draws(rep_dir)
        return draws

    summary = _cached(rep_dir / "ess.json", [rep_dir / "DONE"], lambda: ess.summarise(load()))
    ref = reference.load(model)
    if ref is None:
        verdict = {"status": "no reference", "label": "no reference"}
    else:
        def judge():
            v = gate.check(load(), ref)
            return {"status": "pass" if v.passed else "wrong", "label": v.label(), **v.to_dict()}

        verdict = _cached(rep_dir / "gate.json", [rep_dir / "DONE", reference.path(model)], judge)
    t = rec["timing"]
    chain_sum = sum(t["chain_s"])
    evals = evaluations(t)
    row = {
        "label": rec["label"], "sampler": rec["sampler"], "model": model, "rep": rec["rep"],
        "gate": verdict["status"], "gate_detail": verdict["label"],
        "gate_variable": verdict.get("worst_variable", ""), "gate_moment": verdict.get("worst_moment", ""),
        "gate_b2": verdict.get("b2", math.nan), "gate_b2_lower": verdict.get("b2_lower", math.nan),
        "wall_s": t["wall_s"], "chain_s_sum": chain_sum,
        "grads_sampling": t["grads_sampling"], "grads_warmup": t["grads_warmup"], "evaluations": evals,
        "divergences": t["divergences"],
        **{k: summary[k] for k in ("ess_bulk_min", "ess_bulk_geomean", "ess_tail_min", "rhat_max", "n_variables")},
        "worst_variable": (summary["worst_variables"] or [""])[0],
    }
    eb = row["ess_bulk_min"]
    e_min = min(eb, row["ess_tail_min"])
    row["ess_per_s_wall"] = eb / t["wall_s"] if t["wall_s"] else math.nan
    row["ess_per_s_chain"] = eb / chain_sum if chain_sum else math.nan
    row["ess_per_grad"] = eb / t["grads_sampling"] if t["grads_sampling"] else math.nan
    row["ess_per_eval"] = e_min / evals if evals else math.nan
    if row["gate"] == "wrong":
        for k in COST:
            row[k] = math.nan
    return row


def failed_row(rep_dir: Path) -> dict:
    label, model = rep_dir.parent.parent.name, rep_dir.parent.name
    err = (rep_dir / "FAILED").read_text().strip().splitlines()
    return {"label": label, "sampler": "", "model": model, "rep": int(rep_dir.name[3:]), "gate": "failed",
            "gate_detail": f"failed: {err[-1][:120] if err else ''}", **{k: math.nan for k in NUMERIC}}


NUMERIC = ("wall_s", "chain_s_sum", "grads_sampling", "evaluations", "divergences", "ess_bulk_min",
           "ess_bulk_geomean", "ess_tail_min", "rhat_max", "ess_per_s_wall", "ess_per_s_chain", "ess_per_grad",
           "ess_per_eval")


def _median(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else math.nan


def _geomean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else math.nan


def _ok(r) -> bool:
    return r.get("gate", "pass") in ("pass", "no reference")


def per_model(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["label"], r["model"]), []).append(r)
    out = []
    for (label, model), g in sorted(groups.items()):
        good = [r for r in g if _ok(r)]
        bad = [r for r in g if not _ok(r)]
        if 2 * len(bad) > len(g):
            worst = max(bad, key=lambda r: (r["gate"] == "wrong", r.get("gate_b2", 0) or 0))
            status, detail = "wrong", worst["gate_detail"]
        else:
            status = good[0].get("gate", "pass")
            detail = status
        out.append({"label": label, "model": model, "reps": len(g), "reps_wrong": len(bad), "gate": status,
                    "gate_detail": detail, **{k: _median([r.get(k) for r in good]) for k in NUMERIC}})
        if status == "wrong":
            out[-1].update({k: math.nan for k in COST})
    return out


def summary(models: list[dict]) -> list[dict]:
    out = []
    for label in sorted({m["label"] for m in models}):
        g = [m for m in models if m["label"] == label]
        out.append({
            "label": label, "n_models": len(g),
            "n_models_wrong": sum(1 for m in g if m.get("gate") == "wrong"),
            "geo_ess_per_s_wall": _geomean([m["ess_per_s_wall"] for m in g]),
            "geo_ess_per_s_chain": _geomean([m["ess_per_s_chain"] for m in g]),
            "geo_ess_per_grad": _geomean([m["ess_per_grad"] for m in g]),
            "geo_ess_per_eval": _geomean([m.get("ess_per_eval", math.nan) for m in g]),
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
        out.append({"label": m["label"], "baseline": baseline, "model": m["model"], "gate": m.get("gate", "pass"),
                    **{f"ratio_{k}": (m[k] / b[k] if b.get(k) else math.nan)
                       for k in ("ess_per_s_wall", "ess_per_s_chain", "ess_per_grad", "ess_per_eval",
                                 "ess_bulk_min", "wall_s") if k in m}})
    return out


def bench_score(rows: list[dict], models: list[dict], label: str, baseline: str) -> dict:
    """BENCH.md's score for one label: per rep, ess_per_eval relative to the baseline's per-model
    median, PENALTY for a wrong or failed rep; median over reps; geometric mean over models.

    A model on which the baseline itself is wrong, or has no finite cost, has nothing to compare
    against; it is left out of the score and listed under "excluded"."""
    base, excluded = {}, set()
    for m in models:
        if m["label"] == baseline:
            b = m.get("ess_per_eval", math.nan)
            if m.get("gate") == "wrong" or not (isinstance(b, (int, float)) and math.isfinite(b) and b > 0):
                excluded.add(m["model"])
            else:
                base[m["model"]] = b
    per = {}
    for r in rows:
        if r["label"] != label or r["model"] not in base:
            continue
        v = r.get("ess_per_eval", math.nan)
        rel = v / base[r["model"]] if _ok(r) and math.isfinite(v) and v > 0 else PENALTY
        per.setdefault(r["model"], []).append(rel)
    med = {m: statistics.median(v) for m, v in per.items()}
    lo, hi = _bootstrap(per)
    return {"label": label, "baseline": baseline, "n_models": len(med),
            "score": _geomean(list(med.values())), "score_lo": lo, "score_hi": hi, "per_model": med,
            "excluded": ",".join(sorted(excluded & {r["model"] for r in rows if r["label"] == label}))}


def _bootstrap(per: dict[str, list[float]], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the score from resampling each model's reps with replacement. It shows
    rep-to-rep noise only; the baseline's medians and the choice of models are held fixed."""
    if not per:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    logs = []
    for _ in range(n):
        meds = [np.median(rng.choice(v, size=len(v))) for v in per.values()]
        logs.append(np.mean(np.log(meds)))
    lo, hi = np.percentile(logs, [2.5, 97.5])
    return math.exp(lo), math.exp(hi)


def table(models: list[dict], labels: list[str]) -> str:
    """Markdown: one row per model, one column per label (ESS per evaluation and per chain-second)."""
    by = {(m["label"], m["model"]): m for m in models}
    names = sorted({m["model"] for m in models})
    lines = ["| model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name in names:
        cells = []
        for lab in labels:
            m = by.get((lab, name))
            if m is None:
                cells.append("")
            elif m["gate"] == "wrong":
                cells.append(m["gate_detail"])
            else:
                note = "" if m["gate"] == "pass" else f" ({m['gate']})"
                cells.append(f"{m['ess_per_eval']:.3g}/eval, {m['ess_per_s_chain']:.3g}/s{note}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Cells: min(bulk, tail) ESS per density or gradient evaluation, warmup included; "
              "bulk ESS per second of chain time. \"wrong\" names the gate's worst variable and moment."]
    return "\n".join(lines) + "\n"


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r if k != "per_model"))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k != "per_model"} for r in rows)


def score(runs_dir: Path, out_dir: Path, labels: list[str] | None = None, baseline: str | None = None,
          log=print) -> dict:
    done = sorted(p.parent for p in runs_dir.glob("*/*/rep*/DONE"))
    failed = sorted(p.parent for p in runs_dir.glob("*/*/rep*/FAILED"))
    if labels:
        done = [r for r in done if r.parent.parent.name in labels]
        failed = [r for r in failed if r.parent.parent.name in labels]
    if not done:
        raise FileNotFoundError(f"no finished runs under {runs_dir}")
    rows = [score_rep(r) for r in done] + [failed_row(r) for r in failed]
    models = per_model(rows)
    summ = summary(models)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "per_run.csv", rows)
    _write(out_dir / "per_model.csv", models)
    _write(out_dir / "summary.csv", summ)
    all_labels = sorted({m["label"] for m in models}, key=lambda x: (x != baseline, x))
    tab = table(models, all_labels)
    (out_dir / "table.md").write_text(tab)
    result = {"per_run": rows, "per_model": models, "summary": summ, "table": tab}
    if baseline:
        rat = ratios(models, baseline)
        _write(out_dir / "ratios.csv", rat)
        result["ratios"] = rat
        result["bench"] = [bench_score(rows, models, lab, baseline) for lab in all_labels if lab != baseline]
        _write(out_dir / "bench.csv", result["bench"])
        for b in result["bench"]:
            log(f"{b['label']} vs {baseline}: benchmark score {b['score']:.3g} "
                f"[{b['score_lo']:.3g}, {b['score_hi']:.3g}] over {b['n_models']} models "
                f"(geomean of ESS-per-evaluation ratios, {PENALTY:g} for a wrong or failed model)"
                + (f"; left out because {baseline} is wrong there: {b['excluded']}" if b["excluded"] else ""))
        for label in sorted({r["label"] for r in rat}):
            g = [r["ratio_ess_per_s_wall"] for r in rat if r["label"] == label]
            log(f"{label} vs {baseline}: geomean ESS/s ratio {_geomean(g):.3f} over {len(g)} models")
    log(tab)
    for s in summ:
        log(f"{s['label']}: {s['n_models']} models, {s['n_models_wrong']} wrong, geo ESS/s "
            f"{s['geo_ess_per_s_wall']:.3g} (wall) {s['geo_ess_per_s_chain']:.3g} (chain sum), "
            f"geo ESS/grad {s['geo_ess_per_grad']:.3g}, geo ESS/eval {s['geo_ess_per_eval']:.3g}, "
            f"R-hat > {RHAT_BAD}: {s['n_models_rhat_bad']}")
    return result
