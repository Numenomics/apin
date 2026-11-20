"""Score finished runs over a declared grid of models and reps (BENCH.md, rules 1, 2 and 4).

    score(runs_dir, out_dir, grid, baseline) -> per_run, per_model, ranking, table

Headline efficiency metric (Slice 2):
  cost = n_density + r_grad * n_gradient + r_hvp * n_hvp + r_hess * n_hess
  ess_per_cost = credited_ess / cost

where r_m is measured once per model through BridgeStan at reference draws (models/costs.json),
floored at 1.0, and frozen for a benchmark release. Missing r_m is an explicit error, never
silently defaulted to 1.0. 1:1 ess_per_eval, ess_per_s, and sampler overhead_ratio
(overhead_s / eval_s) are reported beside ess_per_cost.

Multi-walker or tempered runs (walkers > 1 or any beta < 1.0) are recorded by the harness in
Slice 2 and marked unscored until Slice 4 adds exact multi-point and tempered invariance testing.

Claim-wording rule: passing a model means "passed the 0.1-SD gate at this budget", never
"is correct" or "exact".
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import cmdstan, cost as cost_mod, gate, grid as grid_mod, models as models_mod, records, reference
from .reference import LP, Reference

BOOTSTRAP = 2000
CHECKPOINT_DRAWS = (500, 1000, 2000, 5000, 10000, 20000)


@dataclass(frozen=True)
class Entry:
    """One model of the grid: its reference, budget, cost weights, hashes, and lp__ evaluator."""

    model: str
    ref: Reference
    chains: int
    warmup: int
    draws: int
    lp: Callable[[dict], np.ndarray]
    cost_weights: dict[str, float] | None = None
    stan_sha256: str = ""
    data_sha256: str = ""
    bridgestan_key: str = ""

    @property
    def max_evaluations(self) -> int:
        return grid_mod.max_evaluations(self.chains, self.warmup, self.draws)

    @property
    def max_seconds(self) -> float:
        return grid_mod.MAX_SECONDS


@dataclass(frozen=True)
class Grid:
    entries: dict[str, Entry]
    reps: int
    base_seed: int = grid_mod.BASE_SEED


def declared(spec: str = "all", reps: int = 5, base_seed: int = grid_mod.BASE_SEED,
             build_dir: Path = models_mod.ROOT / "build",
             costs_path: Path = cost_mod.COSTS_PATH) -> Grid:
    """The benchmark's grid: selected models with a reference, at manifest budgets and frozen r_m."""
    costs = cost_mod.load(costs_path)
    entries = {}
    for m in models_mod.select(spec):
        ref = reference.load(m.name)
        if ref is None:
            continue
        weights = cost_mod.weights_for(m.name, costs=costs, path=costs_path)
        stan_sha = hashlib.sha256(m.stan_file.read_bytes()).hexdigest() if m.stan_file.is_file() else ""
        data_sha = m.data_sha256 or ""
        bs_key = str((costs.get(m.name) or {}).get("bridgestan_key") or "")
        entries[m.name] = Entry(
            m.name, ref, m.chains, m.warmup, m.draws,
            lambda d, m=m: reference.log_density(m, d, build_dir, strict=False),
            cost_weights=weights, stan_sha256=stan_sha, data_sha256=data_sha, bridgestan_key=bs_key,
        )
    return Grid(entries, reps, base_seed)


def chain_files(rep_dir: Path) -> list[Path]:
    def _sort_key(p: Path) -> tuple[int, int]:
        parts = p.stem.split("_")
        c = int(parts[1])
        w = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("w") else 0
        return (c, w)

    return sorted(rep_dir.glob("chain_*.csv"), key=_sort_key)


def chain_draws(rep_dir: Path, names: list[str] | None = None, csvs: list | None = None) -> dict[str, np.ndarray]:
    if csvs is None:
        files = chain_files(rep_dir)
        if not files:
            raise FileNotFoundError(f"no chain CSVs in {rep_dir}")
        csvs = [cmdstan.read_csv(f) for f in files]
    n = min(f.values.shape[0] for f in csvs)
    if names is None:
        names = [c for c in csvs[0].columns if not c.endswith("__")]
    return {c: np.column_stack([f.values[:n, f.columns.index(c)] for f in csvs])
            for c in names if all(c in f.columns for f in csvs)}


GATE_SOURCE = hashlib.sha256(b"".join(Path(m.__file__).read_bytes()
                                      for m in (gate, gate.ess, reference, cmdstan, cost_mod))
                             + Path(__file__).read_bytes()).hexdigest()


def _key(entry: Entry, reps: int, bridgestan_key: str = "") -> dict:
    """Expanded cache key: hashes Stan model, data, reference+halves, BridgeStan build, and gate code."""
    halves = entry.ref.halves_json() if entry.ref.halves else ""
    return {
        "reps": reps,
        "reference": hashlib.sha256((entry.ref.to_json() + halves).encode()).hexdigest(),
        "stan_sha256": entry.stan_sha256,
        "data_sha256": entry.data_sha256,
        "bridgestan_key": bridgestan_key or entry.bridgestan_key,
        "gate": GATE_SOURCE,
    }


def read_rep(rep_dir: Path, chains: int, draws: int, names: list[str] | None = None) -> tuple[list, dict[str, str]]:
    files = chain_files(rep_dir)
    if len(files) != chains:
        raise ValueError(f"{len(files)} chains, the budget is {chains}")
    csvs, hashes = [], {}
    for f in files:
        try:
            data = f.read_bytes()
            csvs.append(cmdstan.read_csv(f, columns=names, data=data))
        except (OSError, ValueError, cmdstan.CmdStanError) as e:
            raise ValueError(f"{f.name} unreadable: {e}") from None
        hashes[f.name] = hashlib.sha256(data).hexdigest()
        if len(csvs[-1].values) != draws:
            raise ValueError(f"{f.name} has {len(csvs[-1].values)} draws, the budget is {draws}")
    return csvs, hashes


def _first_passing_checkpoint(draws: dict[str, np.ndarray], ref: Reference, total_draws: int, reps: int) -> int:
    """Earliest prefix checkpoint in CHECKPOINT_DRAWS that passes the 0.1-SD gate (right)."""
    for k in CHECKPOINT_DRAWS:
        if k >= total_draws:
            break
        sub = {name: arr[:k] for name, arr in draws.items()}
        if gate.check(sub, ref, reps=reps).status == "right":
            return k
    return total_draws


def verdict_with_checkpoint(rep_dir: Path, entry: Entry, reps: int, bridgestan_key: str = "") -> tuple[gate.Verdict, int | None]:
    ref = entry.ref
    names = list(ref.names) + list(ref.constants)
    try:
        csvs, hashes = read_rep(rep_dir, entry.chains, entry.draws, names)
    except ValueError as e:
        return gate.Verdict("wrong", str(e)), None
    key = {**_key(entry, reps, bridgestan_key=bridgestan_key), "chains": hashes}
    cache = rep_dir / "gate.json"
    try:
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            v_cached = gate.Verdict.from_dict(c["verdict"])
            fp = c.get("first_pass_draws")
            if fp is None and v_cached.status == "right":
                fp = entry.draws
            return v_cached, fp
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    draws = chain_draws(rep_dir, names, csvs)
    if ref.has_lp and all(n in draws for n in ref.names):
        draws[LP] = entry.lp(draws)
    v = gate.check(draws, ref, reps=reps)
    fp = _first_passing_checkpoint(draws, ref, entry.draws, reps) if v.status == "right" else None
    cache.write_text(json.dumps({"key": key, "verdict": v.to_dict(), "first_pass_draws": fp}))
    return v, fp


def verdict(rep_dir: Path, entry: Entry, reps: int, bridgestan_key: str = "") -> gate.Verdict:
    v, _ = verdict_with_checkpoint(rep_dir, entry, reps, bridgestan_key=bridgestan_key)
    return v


def evaluation_counts(t: dict) -> tuple[int | None, int | None, int | None]:
    """(all evaluations, density calls, gradient calls), None where unknown."""
    total, dens, grad, _, _ = extended_evaluation_counts(t)
    return total, dens, grad


def extended_evaluation_counts(t: dict) -> tuple[int | None, int | None, int | None, int, int]:
    """(all evaluations, density calls, gradient calls, hvp calls, hess calls), None where unknown."""
    if t.get("uncounted"):
        return None, None, None, 0, 0
    extra = t.get("extra") or {}
    total = t.get("evaluations")
    if total is None and t.get("grads_sampling") is not None and t.get("grads_warmup") is not None:
        total = t["grads_sampling"] + t["grads_warmup"]
    dens, grad = extra.get("density_evals"), extra.get("gradient_evals")
    hvp = int(extra.get("hvp_evals") or 0)
    hess = int(extra.get("hess_evals") or 0)
    if total is not None and dens is None and grad is None and t.get("evaluations") is None:
        dens, grad = 0, total  # gradient-only sampler (CmdStan NUTS)
    return total, dens, grad, hvp, hess


METERED_KEYS = ("density_evals", "gradient_evals", "max_evaluations", "max_seconds")
SPLIT_COUNT_KEYS = (
    "hvp_evals", "hess_evals",
    "adapt_density_evals", "adapt_gradient_evals", "adapt_hvp_evals", "adapt_hess_evals",
    "transition_density_evals", "transition_gradient_evals", "transition_hvp_evals", "transition_hess_evals",
)


def metered(rec: dict) -> bool:
    t = rec.get("timing") or {}
    extra = t.get("extra") or {}
    return (rec.get("sampler") == "logdensity" or t.get("evaluations") is not None
            or any(extra.get(k) is not None for k in METERED_KEYS))


COUNT_KEYS = ("evaluations", "grads_sampling", "grads_warmup")


def count_problem(rec: dict, entry: Entry) -> str | None:
    t = rec["timing"]
    extra = t.get("extra") or {}
    check_ints = [(k, t.get(k)) for k in COUNT_KEYS] + [(k, extra.get(k)) for k in (*METERED_KEYS[:2], *SPLIT_COUNT_KEYS)]
    for name, v in check_ints:
        if v is not None and (type(v) is not int or v < 0):
            return f"recorded {name} {v!r}, not a non-negative integer"
    if metered(rec):
        limits = (entry.max_evaluations, entry.max_seconds)
        recorded = (extra.get("max_evaluations"), extra.get("max_seconds"))
        if recorded != limits:
            return f"limits {recorded} (evaluations, s), the grid's are {limits}"
        if t.get("uncounted"):
            return f"was metered by the harness, so it may not be uncounted ({t['uncounted']!r})"
        total, dens, grad = t.get("evaluations"), extra.get("density_evals"), extra.get("gradient_evals")
        hvp = extra.get("hvp_evals", 0) or 0
        hess = extra.get("hess_evals", 0) or 0
        if None in (total, dens, grad) or total != dens + grad + hvp + hess:
            return (f"counted {total} evaluations, not its {dens} density + {grad} gradient "
                    f"+ {hvp} hvp + {hess} hess calls")
        for kind, full in (("density_evals", dens), ("gradient_evals", grad), ("hvp_evals", hvp), ("hess_evals", hess)):
            a_k, t_k = extra.get(f"adapt_{kind}"), extra.get(f"transition_{kind}")
            if a_k is not None or t_k is not None:
                if a_k is None or t_k is None or a_k + t_k != full:
                    return f"adapt/transition split for {kind} ({a_k}, {t_k}) does not sum to {full}"
    else:
        present = [t[k] for k in ("grads_sampling", "grads_warmup") if t.get(k) is not None]
        total = sum(present) if present else None
    if total is not None and total > entry.max_evaluations:
        return f"used {total} evaluations, the limit is {entry.max_evaluations}"
    return None


def score_rep(label_dir: Path, entry: Entry, rep: int, g: Grid, digest: str | None = None) -> dict:
    d = label_dir / entry.model / f"rep{rep}"
    row = {
        "label": label_dir.name, "model": entry.model, "rep": rep, "status": "wrong", "reason": "",
        "identity": "", "ess": math.nan, "evaluations": None, "density_evals": None, "gradient_evals": None,
        "hvp_evals": 0, "hess_evals": 0, "cost": math.nan, "seconds": math.nan,
        "first_pass_draws": None, "cost_to_gate": math.nan,
        "ess_per_cost": math.nan, "ess_per_eval": math.nan, "ess_per_s": math.nan,
        "overhead_ratio": math.nan, "verdict": None,
    }

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
    for what, got, want in (("model", rec.get("model"), entry.model), ("rep", rec.get("rep"), rep),
                            ("base seed", rec.get("base_seed"), g.base_seed)):
        if got != want:
            return wrong(f"rep{rep} recorded {what} {got!r}, the grid cell's is {want!r}")
    if digest is not None and (rec.get("identity") or {}).get("submission_sha256") != digest:
        return wrong(f"rep{rep} submission digest {(rec.get('identity') or {}).get('submission_sha256')}, "
                     f"not the submission file's {digest}")
    want_seed = grid_mod.seed_for(rep, g.base_seed)
    if rec.get("seed") != want_seed:
        return wrong(f"rep{rep} seed {rec.get('seed')}, the grid's is {want_seed}")
    got = (rec.get("chains"), rec.get("warmup"), rec.get("draws"))
    want = (entry.chains, entry.warmup, entry.draws)
    if got != want:
        return wrong(f"rep{rep} budget {got} (chains, warmup, draws), the grid's is {want}")
    t = rec["timing"]
    extra = t.get("extra") or {}
    # Multi-walker / tempered entries require invariance certification before scoring (Slice 4)
    walkers = int(extra.get("walkers") or 1)
    tempered = bool(extra.get("tempered"))
    if (walkers > 1 or tempered) and not bool(extra.get("invariance_verified")):
        reason = "unscored: multi-walker/tempered scoring deferred to Slice 4 (invariance_verified not set)"
        row.update(status="unproven", reason=reason, verdict=gate.Verdict("unproven", reason))
        return row
    problem = count_problem(rec, entry)
    if problem:
        return wrong(f"rep{rep} {problem}")
    total, dens, grad, hvp, hess = extended_evaluation_counts(t)
    if t.get("wall_s") is not None and t["wall_s"] > entry.max_seconds:
        return wrong(f"rep{rep} took {t['wall_s']:.0f} s, the limit is {entry.max_seconds:.0f} s")

    # Compute target-calibrated cost (raises explicitly if r_m is missing!)
    rep_cost = math.nan
    if total is not None and dens is not None and grad is not None:
        rep_cost = cost_mod.compute_cost(dens, grad, entry.cost_weights, hvp=hvp, hess=hess)

    bs_key = str(extra.get("bridgestan_key") or "")
    v, first_pass = verdict_with_checkpoint(d, entry, g.reps, bridgestan_key=bs_key)
    secs = float(sum(t["chain_s"]))
    overhead = extra.get("overhead_ratio")
    cost_to_gate = math.nan
    if v.status == "right" and first_pass is not None and math.isfinite(rep_cost) and entry.draws > 0:
        frac = min(1.0, float(first_pass) / float(entry.draws))
        if extra.get("adapt_density_evals") is not None and extra.get("transition_density_evals") is not None:
            adapt_c = cost_mod.compute_cost(
                int(extra.get("adapt_density_evals") or 0),
                int(extra.get("adapt_gradient_evals") or 0),
                entry.cost_weights,
                hvp=int(extra.get("adapt_hvp_evals") or 0),
                hess=int(extra.get("adapt_hess_evals") or 0),
            )
            trans_c = max(0.0, rep_cost - adapt_c)
            cost_to_gate = adapt_c + frac * trans_c
        else:
            cost_to_gate = frac * rep_cost
    row.update(
        status=v.status, reason=v.label(), ess=v.ess, evaluations=total, density_evals=dens,
        gradient_evals=grad, hvp_evals=hvp, hess_evals=hess, cost=rep_cost, seconds=secs,
        first_pass_draws=first_pass, cost_to_gate=cost_to_gate,
        overhead_ratio=float(overhead) if overhead is not None and math.isfinite(overhead) else math.nan,
        verdict=v,
    )
    if v.status == "right":
        row["ess_per_cost"] = v.ess / rep_cost if math.isfinite(rep_cost) and rep_cost > 0 else math.nan
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
        oh = [r["overhead_ratio"] for r in reps if isinstance(r.get("overhead_ratio"), (int, float))
              and math.isfinite(r["overhead_ratio"])]
        out.append({
            "label": label, "model": model, "status": m.status, "reason": m.reason,
            "calibration": m.calibration, "calibration_p": m.calibration_p,
            "cost_u": [r["ess_per_cost"] for r in reps] if right else [],
            "eval": [r["ess_per_eval"] for r in reps] if right else [],
            "sec": [r["ess_per_s"] for r in reps] if right else [],
            "gate_eff": [1.0 / r["cost_to_gate"] if isinstance(r.get("cost_to_gate"), (int, float))
                         and math.isfinite(r["cost_to_gate"]) and r["cost_to_gate"] > 0 else math.nan
                         for r in reps] if right else [],
            "ess_per_cost": _mean([r["ess_per_cost"] for r in reps]) if right else math.nan,
            "ess_per_eval": _mean([r["ess_per_eval"] for r in reps]) if right else math.nan,
            "ess_per_s": _mean([r["ess_per_s"] for r in reps]) if right else math.nan,
            "cost_to_gate": _mean([r["cost_to_gate"] for r in reps]) if right else math.nan,
            "overhead_ratio": float(np.mean(oh)) if oh else math.nan,
        })
    return out


def _mean(v) -> float:
    return float(np.mean(v)) if v and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) else math.nan


def rank(models: list[dict], baseline: str | None, names: list[str], seed: int = 0) -> list[dict]:
    """Rank by models passing the 0.1-SD gate, then by ess_per_cost ratio vs baseline (headline),
    with 1:1 ess_per_eval and ess_per_s reported alongside."""
    by = {(m["label"], m["model"]): m for m in models}
    labels = list(dict.fromkeys(m["label"] for m in models))
    out = []
    for lab in labels:
        right = [n for n in names if by.get((lab, n), {}).get("status") == "right"]
        oh_vals = [by[(lab, n)]["overhead_ratio"] for n in names
                   if (lab, n) in by and math.isfinite(by[(lab, n)].get("overhead_ratio", math.nan))]
        row = {
            "label": lab, "right": len(right), "of": len(names),
            "overhead_ratio": float(np.median(oh_vals)) if oh_vals else math.nan, "note": "",
        }
        for unit, key in (("cost_u", "speed_cost"), ("eval", "speed_eval"), ("sec", "speed_s"), ("gate_eff", "gate_speed")):
            row.update({key: math.nan, f"{key}_lo": math.nan, f"{key}_hi": math.nan, f"{key}_p": math.nan})
            if baseline is None:
                continue
            both = [n for n in right if by.get((baseline, n), {}).get("status") == "right"]
            row["compared"] = len(both)
            if any(unit not in by[(lab, n)] or unit not in by[(baseline, n)] for n in both):
                continue
            pairs = [(by[(lab, n)][unit], by[(baseline, n)][unit]) for n in both]
            bad = [n for n, (a, b) in zip(both, pairs) if not _finite(a) or not _finite(b)]
            if bad:
                if unit in ("cost_u", "eval") and not row["note"]:
                    row["note"] = f"cost unavailable on {', '.join(bad)}"
                continue
            if not pairs:
                continue
            if lab == baseline:
                row[key] = 1.0
                continue
            boot = records.paired_hierarchical_bootstrap(pairs, n_boot=BOOTSTRAP, seed=seed, margin=1.0)
            row[key] = boot["ratio"]
            row[f"{key}_lo"], row[f"{key}_hi"] = boot["ci_lo"], boot["ci_hi"]
            row[f"{key}_p"] = boot["p_value"]
        out.append(row)
    out.sort(key=lambda r: (-r["right"], -(r["speed_cost"] if math.isfinite(r["speed_cost"]) else (r["speed_eval"] if math.isfinite(r["speed_eval"]) else -math.inf))))
    for i, r in enumerate(out):
        r["rank"] = i + 1
    return out


def _finite(v) -> bool:
    return len(v) > 0 and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in v)


def _geo(v) -> float:
    return float(math.exp(np.mean(np.log(v))))


def _bootstrap(pairs, seed: int) -> tuple[float, float]:
    boot = records.paired_hierarchical_bootstrap(pairs, n_boot=BOOTSTRAP, seed=seed, margin=1.0)
    return boot["ci_lo"], boot["ci_hi"]


def table(ranking: list[dict], models: list[dict], baseline: str | None, names: list[str]) -> str:
    labels = [r["label"] for r in ranking]
    lines = [
        "| rank | label | passed 0.1-SD gate | ESS/cost vs baseline [95 %] | ESS/eval (1:1) [95 %] | ESS/s [95 %] | overhead/eval | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ranking:
        oh = f"{r['overhead_ratio']:.2g}x" if math.isfinite(r.get("overhead_ratio", math.nan)) else ""
        lines.append(
            f"| {r['rank']} | {r['label']} | {r['right']} of {r['of']} | {_speed(r, 'speed_cost')} | "
            f"{_speed(r, 'speed_eval')} | {_speed(r, 'speed_s')} | {oh} | {r['note']} |"
        )
    lines += [
        "",
        f"Ranked by models passing the 0.1-SD bench gate, then by relative ESS per cost unit "
        f"(cost = n_density + r_m * n_gradient, calibrated per model via BridgeStan) against "
        f"{baseline or 'the baseline'} on models where both passed the gate. Passing the gate means "
        f"no checked functional exceeded 0.1 reference SDs at this budget; it is not a proof of exactness.",
        "",
    ]
    by = {(m["label"], m["model"]): m for m in models}
    lines += ["| model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for n in names:
        cells = []
        for lab in labels:
            m = by.get((lab, n))
            if m is None:
                cells.append("")
            elif m["status"] == "right":
                cells.append(f"passed: {m['ess_per_cost']:.3g}/cost, {m['ess_per_eval']:.3g}/eval, {m['ess_per_s']:.3g}/s")
            else:
                cells.append(f"{m['status']}: {m['reason']}" if not m["reason"].startswith(m["status"])
                             else m["reason"])
        lines.append(f"| {n} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    lines += [
        "",
        "Cells: mean over reps of credited ESS per calibrated cost unit, per 1:1 evaluation (warmup included), "
        "and per chain-second; or the gate status and reason.",
    ]
    return "\n".join(lines) + "\n"


def _speed(r, key) -> str:
    if not math.isfinite(r[key]):
        return ""
    if not math.isfinite(r[key + "_lo"]):
        return f"{r[key]:.3g}"
    return f"{r[key]:.3g} [{r[key + '_lo']:.3g}, {r[key + '_hi']:.3g}]"


def _write(path: Path, rows: list[dict], drop=("verdict", "cost_u", "eval", "sec", "gate_eff")) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r if k not in drop))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k not in drop} for r in rows)


def score(runs_dir: Path, out_dir: Path, g: Grid, baseline: str | None = None, labels: list[str] | None = None,
          log=print, submissions: dict[str, Path] | None = None) -> dict:
    digests = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in (submissions or {}).items()}
    label_dirs = sorted(p for p in Path(runs_dir).iterdir() if p.is_dir() and (not labels or p.name in labels))
    unknown = sorted(set(digests) - {ld.name for ld in label_dirs})
    if unknown:
        raise ValueError(f"--submission names {', '.join(map(repr, unknown))}, which is not among the scored labels "
                         f"({', '.join(ld.name for ld in label_dirs) or 'none'}); its digest would not be checked")
    if baseline and baseline not in [p.name for p in label_dirs]:
        raise KeyError(f"baseline label {baseline!r} has no runs under {runs_dir}")
    names = list(g.entries)
    rows = []
    for ld in label_dirs:
        lab = [score_rep(ld, g.entries[n], r, g, digests.get(ld.name)) for n in names for r in range(g.reps)]
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
