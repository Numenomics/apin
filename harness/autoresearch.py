"""Slice 5 Autoresearch runner: Karpathy 3-file contract + paired MAP-Elites archive.

Hardened post-review & post-dogfood:
  - Read-only `harness/judge.py` owns Lane A1's fixed transition kernel (`adapt_metric` seam).
  - Snapshots `submission.py` at trial start so mid-eval edits never race with active trials.
  - Pre-eval novelty filter rejects both exact-file duplicates and AST-equivalent
    (comment/whitespace-only) re-submissions against `ledger.jsonl` (with `--dry-run` pre-check).
  - Per-`--run-id` isolated `incumbent.py` and `archive/` (concurrent run IDs never collide).
  - Built-in FIFO file-locked evaluation queue (`harness/queue.py`) serializes CPU-bound evals
    across parallel agents while letting reasoning and editing overlap freely.
  - Tier 0 & Tier 1 run untrusted submissions inside isolated out-of-process brokered workers
    (`harness/ipc.py`) with `adapt()` cost amortized to a realistic sampling horizon
    (`horizon_steps=500`, ~1:5 warmup:sampling) so warmup-starving MALA hacks cannot game Tier 1.
  - Multi-seed `--repeats` paired evaluation reduces coin-flip single-seed variance.
  - Simplicity regularizer credits fewer LOC, fewer magic constants, or >=15% lower overhead
    within `±NOISE_MARGIN` (aligned with `program.md` Rule 2.5).
  - Records full trial config (`dim`, `seed`, `repeats`, `k_steps`, `adapt_draws`, `horizon_steps`,
    `magic_constants`, `queue_wait_s`) in both `ledger.jsonl` and `results.tsv`.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import flywheel, invariance, ipc, judge, logdensity, queue, shapes
from .models import ROOT

NOISE_MARGIN = 0.02  # +/- 2% tie band where simplicity regularizer breaks ties
_TRIVIAL_CONSTS = frozenset({-1, 0, 1, 2, -1.0, 0.0, 1.0, 2.0})


def _judge_digest() -> str:
    return judge.judge_digest()


def _submission_stats(path: Path) -> tuple[str, str, int, int]:
    """Return (raw_sha256[:16], ast_normalized_sha256[:16], loc, magic_constants)."""
    text = path.read_text()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    magic_constants = 0
    try:
        tree = ast.parse(text)
        norm = ast.dump(tree, include_attributes=False)
        ast_sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value not in _TRIVIAL_CONSTS:
                    magic_constants += 1
    except SyntaxError:
        ast_sha = sha
    loc = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    return sha, ast_sha, loc, magic_constants


def _load_ledger(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_results_tsv(ledger_path: Path, tsv_path: Path) -> None:
    rows = _load_ledger(ledger_path)
    cols = [
        "trial_id", "timestamp", "candidate_sha", "loc", "magic_constants",
        "dim", "dims", "seed", "repeats", "k_steps", "adapt_draws", "horizon_steps",
        "tier0_ok", "tier1_ok", "incumbent_ok", "soft_min_eff", "incumbent_soft_min", "ratio_vs_incumbent",
        "short_horizon_soft_min", "long_horizon_soft_min", "median_adapt_cost",
        "b2_max", "overhead_ratio", "queue_wait_s", "decision", "archive_cells",
        "codex_refs", "hypothesis",
    ]
    with open(tsv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {
                **r,
                "dims": ",".join(str(x) for x in (r.get("dims") or [])),
                "archive_cells": ",".join(r.get("archive_cells") or []),
                "codex_refs": ",".join(r.get("codex_refs") or []),
            }
            w.writerow(flat)


def _append_row(ledger_path: Path, tsv_path: Path, row: dict, *, run_id: str | None = None,
                parent_id: str | None = None, parents: list[dict] | None = None,
                source: str | None = None) -> dict:
    """Append to the ledger, then best-effort post the trial (with its source and parents) to Flywheel."""
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    _write_results_tsv(ledger_path, tsv_path)
    if run_id is not None:
        flywheel.post_trial_row(row, run_id=run_id, parent_id=parent_id, parents=parents, source=source)
    return row


def _eval_tier0(
    sub_path: Path,
    seed: int,
    work_dir: Path,
    *,
    lane: str | None = None,
    horizon_steps: int = 500,
) -> tuple[bool, dict]:
    """Tier 0 fast reject: AST lint, lane seam, isolated ABI + replay, calibrated invariance smoke."""
    try:
        logdensity.check_imports(sub_path.read_text())
        judge.resolve_lane_submission(sub_path, lane)
    except Exception as e:
        return False, {"stage": "load_lint", "error": str(e)}

    # 1. Isolated 3-process ABI + frozen tuning + 5-step replay on a fresh rotated Gaussian (d=4)
    g4 = shapes.IllCondGaussian(dim=4, seed=seed + 11, rotated=True, kappa=25.0)
    try:
        _, summary = ipc.execute_run_isolated(
            g4, sub_path, logdensity.Budget(1, 20, 200_000, 30.0),
            run_seed=seed + 13, work_dir=work_dir / "t0_abi", replay_count=5, lane=lane,
        )
    except Exception as e:
        return False, {"stage": "abi_replay", "error": str(e)}

    # 2. Isolated exact-draw invariance + dispersed-start mixing smoke on rotated Gaussian & small funnel
    smoke_targets = [
        shapes.IllCondGaussian(dim=6, seed=seed + 17, rotated=True, kappa=25.0),
        shapes.NealsFunnel(dim=4, seed=seed + 19, rotated=False, v_sd=2.0),
    ]
    checks = []
    for tgt in smoke_targets:
        res = invariance.check_target_invariance(
            tgt, sub_path, seed=seed + 23, n_chains=48, k_steps=12, adapt_draws=80,
            alpha=0.0005, lane=lane, isolated=True, horizon_steps=horizon_steps,
        )
        checks.append({
            "target": tgt.family, "passed": res.passed, "max_z": round(res.max_z, 3),
            "z_thresh": round(res.z_threshold, 3), "b2_max": round(res.b2_max, 5),
            "move": round(res.mean_movement_sd, 4), "adapt_cost": round(res.adapt_cost, 1),
            "reason": res.reason,
        })
        if not res.passed:
            return False, {"stage": "invariance_smoke", "checks": checks, "error": res.reason}

    return True, {"stage": "passed", "tuning_bytes": summary["adapt"]["tuning_bytes"], "checks": checks}


def _build_tier1_suite(dim: int, seed: int, dims: tuple[int, ...] | None = None) -> list[shapes.AnalyticTarget]:
    """Fresh aligned + rotated Tier 1 suite, optionally with a higher-dim scaling ladder."""
    ladder = tuple(dims) if dims else (int(dim),)
    d0 = ladder[0]
    targets: list[shapes.AnalyticTarget] = [
        shapes.IllCondGaussian(d0, seed + 1, rotated=False, kappa=50.0),
        shapes.IllCondGaussian(d0, seed + 2, rotated=True, kappa=50.0),
        shapes.NealsFunnel(d0, seed + 3, rotated=True, v_sd=2.0),
        shapes.Banana(d0, seed + 4, rotated=True),
        shapes.HybridRosenbrock(d0, seed + 5, rotated=False, b=5.0),
        shapes.StudentT(d0, seed + 6, rotated=False, nu=1.5),
        shapes.StudentT(d0, seed + 7, rotated=True, nu=3.0),
        shapes.GaussianMixture(d0, seed + 8, rotated=True, separation=2.5),
    ]
    for idx, d_hi in enumerate(ladder[1:], start=1):
        off = 20 * idx
        targets.extend([
            shapes.IllCondGaussian(d_hi, seed + off + 1, rotated=True, kappa=50.0),
            shapes.NealsFunnel(d_hi, seed + off + 2, rotated=True, v_sd=2.0),
            shapes.Banana(d_hi, seed + off + 3, rotated=True),
            shapes.HybridRosenbrock(d_hi, seed + off + 4, rotated=False, b=5.0),
            shapes.StudentT(d_hi, seed + off + 5, rotated=True, nu=3.0),
        ])
    return targets


def _eval_tier1_suite(
    sub_path: Path,
    suite: list[shapes.AnalyticTarget],
    seed: int,
    *,
    n_chains: int = 64,
    k_steps: int = 20,
    adapt_draws: int = 120,
    lane: str | None = None,
    horizon_steps: int = 500,
) -> dict:
    """Evaluate a submission across a fresh aligned + rotated analytic suite via isolated workers."""
    per_target = []
    by_class: dict[str, list[float]] = {}
    by_class_short: dict[str, list[float]] = {}
    by_class_long: dict[str, list[float]] = {}
    all_ok = True
    b2_worst = 0.0
    oh_vals = []
    adapt_vals = []
    for i, tgt in enumerate(suite):
        res = invariance.check_target_invariance(
            tgt, sub_path, seed=seed + 100 * i, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, isolated=True, horizon_steps=horizon_steps,
        )
        # Dual-horizon geometric mean (H_short=150 & H_long=1000) rewards genuine fast warmup
        # without letting warmup starvation hijack steady-state sampling quality.
        eff_val = res.dual_horizon_eff if res.dual_horizon_eff > 0 else res.empirical_efficiency
        per_target.append({
            "family": tgt.family, "dim": tgt.dim, "rotated": tgt.rotated, "passed": res.passed,
            "emp_eff": eff_val, "short_horizon_eff": res.short_horizon_eff,
            "long_horizon_eff": res.long_horizon_eff, "b2_max": res.b2_max,
            "max_z": res.max_z, "z_threshold": res.z_threshold,
            "adapt_cost": res.adapt_cost, "adapt_cost_amortized": res.adapt_cost_amortized,
            "cost_per_chain": res.cost_per_chain, "overhead_ratio": res.overhead_ratio,
            "reason": res.reason,
        })
        if not res.passed:
            all_ok = False
        b2_worst = max(b2_worst, res.b2_max)
        oh_vals.append(res.overhead_ratio)
        adapt_vals.append(res.adapt_cost)
        if not tgt.multimodal:
            by_class.setdefault(tgt.family, []).append(eff_val)
            by_class_short.setdefault(tgt.family, []).append(res.short_horizon_eff)
            by_class_long.setdefault(tgt.family, []).append(res.long_horizon_eff)

    class_geo = {
        c: float(np.exp(np.mean(np.log(np.maximum(vs, 1e-18))))) for c, vs in by_class.items()
    }
    class_short = {
        c: float(np.exp(np.mean(np.log(np.maximum(vs, 1e-18))))) for c, vs in by_class_short.items()
    }
    class_long = {
        c: float(np.exp(np.mean(np.log(np.maximum(vs, 1e-18))))) for c, vs in by_class_long.items()
    }
    soft_min = invariance.soft_min_power_mean(list(class_geo.values()), p=-4.0) if all_ok else 0.0
    short_sm = invariance.soft_min_power_mean(list(class_short.values()), p=-4.0) if all_ok else 0.0
    long_sm = invariance.soft_min_power_mean(list(class_long.values()), p=-4.0) if all_ok else 0.0
    geomean = float(np.exp(np.mean(np.log(np.maximum(list(class_geo.values()), 1e-18))))) if class_geo else 0.0
    return {
        "passed": all_ok,
        "soft_min_eff": soft_min,
        "short_horizon_soft_min": short_sm,
        "long_horizon_soft_min": long_sm,
        "median_adapt_cost": float(np.median(adapt_vals)) if adapt_vals else 0.0,
        "geomean_eff": geomean,
        "b2_max": b2_worst,
        "overhead_ratio": float(np.median(oh_vals)) if oh_vals else 0.0,
        "by_class": class_geo,
        "per_target": per_target,
    }


def _eval_tier1_multi(
    sub_path: Path,
    suites: list[tuple[int, list[shapes.AnalyticTarget]]],
    *,
    n_chains: int,
    k_steps: int,
    adapt_draws: int,
    lane: str | None,
    horizon_steps: int,
) -> dict:
    """Evaluate across one or more freshly seeded suites and aggregate via geometric mean."""
    if len(suites) == 1:
        rseed, suite = suites[0]
        return _eval_tier1_suite(
            sub_path, suite, seed=rseed + 1000, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
        )
    runs = [
        _eval_tier1_suite(
            sub_path, suite, seed=rseed + 1000, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
        )
        for rseed, suite in suites
    ]
    all_ok = all(r["passed"] for r in runs)
    classes = sorted({c for r in runs for c in r["by_class"]})
    by_class = {
        c: float(np.exp(np.mean(np.log(np.maximum([r["by_class"][c] for r in runs if c in r["by_class"]], 1e-18)))))
        for c in classes
    }
    soft_min = (
        float(np.exp(np.mean(np.log(np.maximum([r["soft_min_eff"] for r in runs], 1e-18)))))
        if all_ok else 0.0
    )
    short_sm = (
        float(np.exp(np.mean(np.log(np.maximum([r["short_horizon_soft_min"] for r in runs], 1e-18)))))
        if all_ok else 0.0
    )
    long_sm = (
        float(np.exp(np.mean(np.log(np.maximum([r["long_horizon_soft_min"] for r in runs], 1e-18)))))
        if all_ok else 0.0
    )
    geomean = float(np.exp(np.mean(np.log(np.maximum(list(by_class.values()), 1e-18))))) if by_class else 0.0
    return {
        "passed": all_ok,
        "soft_min_eff": soft_min,
        "short_horizon_soft_min": short_sm,
        "long_horizon_soft_min": long_sm,
        "median_adapt_cost": float(np.median([r["median_adapt_cost"] for r in runs])),
        "geomean_eff": geomean,
        "b2_max": max(r["b2_max"] for r in runs),
        "overhead_ratio": float(np.median([r["overhead_ratio"] for r in runs])),
        "by_class": by_class,
        "per_repeat": runs,
        "per_target": [t for r in runs for t in r["per_target"]],
    }


def _eval_class_paired_multi(
    specialist_path: Path,
    suites: list[tuple[int, list[shapes.AnalyticTarget]]],
    cls_name: str,
    *,
    n_chains: int,
    k_steps: int,
    adapt_draws: int,
    lane: str | None,
    horizon_steps: int,
) -> tuple[bool, float]:
    """Evaluate an archived MAP-Elites specialist on the exact same class targets & seeds."""
    vals = []
    for rseed, suite in suites:
        for i, tgt in enumerate(suite):
            if tgt.family != cls_name:
                continue
            res = invariance.check_target_invariance(
                tgt, specialist_path, seed=rseed + 1000 + 100 * i, n_chains=n_chains, k_steps=k_steps,
                adapt_draws=adapt_draws, lane=lane, isolated=True, horizon_steps=horizon_steps,
            )
            if not res.passed:
                return False, 0.0
            vals.append(res.dual_horizon_eff if res.dual_horizon_eff > 0 else res.empirical_efficiency)
    if not vals:
        return False, 0.0
    return True, float(np.exp(np.mean(np.log(np.maximum(vals, 1e-18)))))


def _overhead_bucket(ratio: float) -> str:
    if ratio < 1.0:
        return "oh_low"
    if ratio < 5.0:
        return "oh_med"
    return "oh_high"


def _is_simpler(cand_loc: int, inc_loc: int, cand_consts: int, inc_consts: int,
                cand_oh: float, inc_oh: float) -> bool:
    """Rule 2.5 simplicity regularizer: fewer LOC, or (loc <= inc) with fewer magic constants or >=15% lower overhead."""
    if cand_loc < inc_loc:
        return True
    if cand_loc <= inc_loc and cand_consts < inc_consts:
        return True
    if cand_loc <= inc_loc and inc_oh > 0.0 and cand_oh <= 0.85 * inc_oh:
        return True
    return False


def evaluate_lane_trial(
    lane: str,
    hypothesis: str,
    *,
    codex_refs: list[str] | None = None,
    seed: int | None = None,
    dim: int = 6,
    dims: tuple[int, ...] | None = None,
    n_chains: int = 48,
    k_steps: int = 16,
    adapt_draws: int = 100,
    horizon_steps: int = 500,
    repeats: int = 1,
    autoresearch_dir: Path = ROOT / "autoresearch",
    run_id: str = "default",
    allow_duplicate: bool = False,
    submission_path: Path | None = None,
    dry_run: bool = False,
    use_queue: bool = True,
    combines: list[str] | None = None,
    fork_of: list[str] | None = None,
) -> dict:
    """Run one complete Karpathy-ratchet + paired MAP-Elites autoresearch trial on `lane`.

    ``combines`` / ``fork_of`` name Flywheel trial ids whose mechanism this candidate reuses
    or whose source it started from; they are posted as extra parents next to the incumbent.
    """
    lane_dir = autoresearch_dir / "lanes" / lane
    sub_path = Path(submission_path) if submission_path is not None else (lane_dir / "submission.py")
    if not sub_path.is_file():
        raise FileNotFoundError(f"missing editable submission at {sub_path}")

    run_dir = lane_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    tsv_path = run_dir / "results.tsv"
    incumbent_path = run_dir / "incumbent.py"
    archive_dir = run_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    prior = _load_ledger(ledger_path)
    last_kept = next((r for r in reversed(prior) if r.get("decision") in flywheel.KEPT_DECISIONS), None)
    parent_node_id = (
        f"{lane}:{run_id}:{last_kept['trial_id']}" if last_kept and last_kept.get("trial_id") else None
    )
    parent_links = flywheel.parent_links(parent_node_id, combines=combines, fork_of=fork_of)
    extra_parents = [p for p in parent_links if p["relation"] != "incumbent"]
    trial_id = f"T{len(prior) + 1:04d}"
    seed = int(seed if seed is not None else (20261120 + len(prior) * 97))
    repeats = max(1, int(repeats))
    t0 = time.perf_counter()

    # Snapshot submission.py before any read/eval so concurrent mid-eval edits cannot race
    raw_text = sub_path.read_text()
    if dry_run:
        snap_path = sub_path
        cand_sha, cand_ast_sha, cand_loc, cand_consts = _submission_stats(sub_path)
    else:
        trial_work = run_dir / "trials" / trial_id
        trial_work.mkdir(parents=True, exist_ok=True)
        snap_path = trial_work / "submission_snapshot.py"
        snap_path.write_text(raw_text)
        cand_sha, cand_ast_sha, cand_loc, cand_consts = _submission_stats(snap_path)

    dim_ladder = tuple(int(x) for x in dims) if dims else (int(dim),)
    base_meta = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "candidate_sha": cand_sha,
        "ast_sha": cand_ast_sha,
        "loc": cand_loc,
        "magic_constants": cand_consts,
        "judge_digest": _judge_digest(),
        "dim": dim_ladder[0],
        "dims": list(dim_ladder),
        "seed": seed,
        "repeats": repeats,
        "k_steps": int(k_steps),
        "adapt_draws": int(adapt_draws),
        "horizon_steps": int(horizon_steps),
        "hypothesis": hypothesis,
        "codex_refs": list(codex_refs or []),
    }
    if extra_parents:
        base_meta["parents"] = extra_parents

    # Pre-eval novelty filter: reject exact duplicate OR AST-equivalent (comment/whitespace-only) diff
    is_dup = any(
        r.get("candidate_sha") == cand_sha or (r.get("ast_sha") and r.get("ast_sha") == cand_ast_sha)
        for r in prior
    )

    if dry_run:
        lint_error = None
        try:
            logdensity.check_imports(raw_text)
            judge.resolve_lane_submission(snap_path, lane)
        except Exception as e:
            lint_error = str(e)
        return {
            **base_meta,
            "dry_run": True,
            "is_duplicate": is_dup,
            "lint_ok": lint_error is None,
            "lint_error": lint_error,
            "tier0_ok": (not is_dup or allow_duplicate) and lint_error is None,
            "tier1_ok": False,
            "decision": (
                "dry_run_duplicate" if (is_dup and not allow_duplicate)
                else ("dry_run_lint_error" if lint_error is not None else "dry_run_ok")
            ),
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    if not allow_duplicate and is_dup:
        row = {
            **base_meta,
            "tier0_ok": False, "tier1_ok": False, "queue_wait_s": 0.0,
            "decision": "discard_duplicate",
            "wall_s": round(time.perf_counter() - t0, 3),
        }
        return _append_row(ledger_path, tsv_path, row, run_id=run_id, parent_id=parent_node_id,
                           parents=parent_links, source=raw_text)

    with queue.eval_slot(f"autoresearch:{lane}/{run_id}/{trial_id}", enabled=use_queue) as queue_wait_s:
        t0_ok, t0_info = _eval_tier0(
            snap_path, seed=seed, work_dir=trial_work, lane=lane, horizon_steps=horizon_steps,
        )
        if not t0_ok:
            row = {
                **base_meta,
                "tier0_ok": False, "tier0_info": t0_info, "tier1_ok": False,
                "queue_wait_s": round(queue_wait_s, 3),
                "decision": "discard_tier0",
                "wall_s": round(time.perf_counter() - t0, 3),
            }
            return _append_row(ledger_path, tsv_path, row, run_id=run_id, parent_id=parent_node_id,
                           parents=parent_links, source=raw_text)

        # Tier 1: paired fresh suites for candidate AND incumbent on identical fresh seeds
        suites = [(seed + r * 7919, _build_tier1_suite(dim_ladder[0], seed + r * 7919, dims=dim_ladder)) for r in range(repeats)]
        try:
            cand_res = _eval_tier1_multi(
                snap_path, suites, n_chains=n_chains, k_steps=k_steps,
                adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
            )
        except Exception as e:
            row = {
                **base_meta,
                "tier0_ok": True, "tier1_ok": False, "queue_wait_s": round(queue_wait_s, 3),
                "decision": "discard_error",
                "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.perf_counter() - t0, 3),
            }
            return _append_row(ledger_path, tsv_path, row, run_id=run_id, parent_id=parent_node_id,
                           parents=parent_links, source=raw_text)

        inc_sha = None
        inc_ok: bool | None = None
        if not incumbent_path.is_file():
            inc_res = None
            ratio = 1.0 if cand_res["passed"] else 0.0
            decision = "keep_baseline" if cand_res["passed"] else "discard_tier1"
        else:
            inc_sha, _, inc_loc, inc_consts = _submission_stats(incumbent_path)
            inc_res = _eval_tier1_multi(
                incumbent_path, suites, n_chains=n_chains, k_steps=k_steps,
                adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
            )
            inc_ok = bool(inc_res["passed"])
            inc_sm = inc_res["soft_min_eff"] if inc_ok else 0.0
            cand_sm = cand_res["soft_min_eff"]
            ratio = (cand_sm / inc_sm) if inc_sm > 0 else (math.inf if cand_sm > 0 else 0.0)
            if not cand_res["passed"]:
                decision = "discard_tier1"
            elif not inc_ok:
                decision = "keep_incumbent_failed"
            elif ratio >= 1.0 + NOISE_MARGIN:
                decision = "keep"
            elif ratio >= 1.0 - NOISE_MARGIN and _is_simpler(
                cand_loc, inc_loc, cand_consts, inc_consts,
                cand_res["overhead_ratio"], inc_res["overhead_ratio"],
            ):
                decision = "keep_simpler"
            else:
                decision = "discard_no_gain"

        # Paired MAP-Elites specialist archive update (same fresh seeds for candidate vs existing specialist!)
        archived_cells: list[str] = []
        if cand_res["passed"]:
            oh_tag = _overhead_bucket(cand_res["overhead_ratio"])
            for cls_name, cls_score in cand_res["by_class"].items():
                cell_key = f"{cls_name}__{oh_tag}"
                cell_dir = archive_dir / cell_key
                meta_file = cell_dir / "best.json"
                spec_file = cell_dir / "submission.py"
                should_store = False
                if not meta_file.is_file() or not spec_file.is_file():
                    should_store = True
                else:
                    prev_meta = json.loads(meta_file.read_text())
                    prev_sha = prev_meta.get("candidate_sha")
                    if prev_sha == cand_sha:
                        should_store = False
                    elif inc_res is not None and prev_sha == inc_sha and cls_name in inc_res["by_class"]:
                        prev_paired = inc_res["by_class"][cls_name]
                        should_store = cls_score > prev_paired * (1.0 + NOISE_MARGIN)
                    else:
                        prev_ok, prev_paired = _eval_class_paired_multi(
                            spec_file, suites, cls_name, n_chains=n_chains, k_steps=k_steps,
                            adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
                        )
                        should_store = (not prev_ok) or (cls_score > prev_paired * (1.0 + NOISE_MARGIN))
                if should_store:
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snap_path, spec_file)
                    meta_file.write_text(json.dumps({
                        "trial_id": trial_id, "candidate_sha": cand_sha, "score": cls_score,
                        "soft_min_eff": cand_res["soft_min_eff"], "hypothesis": hypothesis,
                    }, indent=1))
                    archived_cells.append(cell_key)
            if decision == "discard_no_gain" and archived_cells:
                decision = "archive_only"

        if decision in ("keep", "keep_simpler", "keep_baseline", "keep_incumbent_failed"):
            shutil.copy2(snap_path, incumbent_path)

    (trial_work / "metrics.json").write_text(
        json.dumps({"tier0": t0_info, "candidate": cand_res, "incumbent": inc_res}, indent=1)
    )
    shutil.copy2(snap_path, trial_work / "submission.py")

    row = {
        **base_meta,
        "tier0_ok": True,
        "tier1_ok": cand_res["passed"],
        "incumbent_ok": inc_ok,
        "soft_min_eff": round(cand_res["soft_min_eff"], 6),
        "incumbent_soft_min": round(inc_res["soft_min_eff"], 6) if inc_res else round(cand_res["soft_min_eff"], 6),
        "short_horizon_soft_min": round(cand_res.get("short_horizon_soft_min", 0.0), 6),
        "long_horizon_soft_min": round(cand_res.get("long_horizon_soft_min", 0.0), 6),
        "median_adapt_cost": round(cand_res.get("median_adapt_cost", 0.0), 1),
        "ratio_vs_incumbent": round(ratio, 4) if math.isfinite(ratio) else None,
        "b2_max": round(cand_res["b2_max"], 5),
        "overhead_ratio": round(cand_res["overhead_ratio"], 3),
        "queue_wait_s": round(queue_wait_s, 3),
        "archive_cells": archived_cells,
        "by_class": {k: round(v, 6) for k, v in cand_res["by_class"].items()},
        "decision": decision,
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    return _append_row(ledger_path, tsv_path, row, run_id=run_id, parent_id=parent_node_id,
                           parents=parent_links, source=raw_text)


REPLICATION_TOLERANCE = 0.25  # a rerun passes at >= 75% of the claimed soft_min_eff


def rerun_for_verification(
    source_path: Path,
    lane: str,
    *,
    claimed_soft_min: float | None,
    dims: tuple[int, ...] = (6, 20),
    seed: int | None = None,
    repeats: int = 2,
    n_chains: int = 48,
    k_steps: int = 16,
    adapt_draws: int = 100,
    horizon_steps: int = 500,
    tolerance: float = REPLICATION_TOLERANCE,
    work_dir: Path | None = None,
    use_queue: bool = True,
) -> dict:
    """Independently rerun a fetched trial source through Tier 0 + Tier 1 on fresh seeds.

    Passes when Tier 0 and Tier 1 pass and the reproduced soft_min_eff reaches
    ``(1 - tolerance)`` of the claimed value. Returns ``{passed, metrics}`` for
    ``POST /api/verify/:id/result``. Nothing is written to any ledger.
    """
    import secrets

    seed = int(seed if seed is not None else 30_000_000 + secrets.randbelow(10_000_000))
    dims = tuple(int(d) for d in dims) or (6,)
    work = work_dir or (source_path.parent / "verify_work")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    metrics: dict = {
        "seed": seed, "dims": list(dims), "repeats": int(repeats), "judge_digest": _judge_digest(),
        "claimed_soft_min_eff": claimed_soft_min, "tolerance": tolerance,
    }
    with queue.eval_slot(f"verify:{lane}/{source_path.name}", enabled=use_queue):
        t0_ok, t0_info = _eval_tier0(source_path, seed=seed, work_dir=work, lane=lane, horizon_steps=horizon_steps)
        metrics["tier0_ok"] = bool(t0_ok)
        if not t0_ok:
            metrics["tier0_info"] = t0_info
            metrics["wall_s"] = round(time.perf_counter() - t0, 3)
            return {"passed": False, "metrics": metrics, "reason": f"tier0 failed: {t0_info.get('stage')}"}
        suites = [(seed + r * 7919, _build_tier1_suite(dims[0], seed + r * 7919, dims=dims)) for r in range(max(1, int(repeats)))]
        res = _eval_tier1_multi(
            source_path, suites, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
        )
    soft = float(res["soft_min_eff"])
    ratio = (soft / claimed_soft_min) if claimed_soft_min else None
    metrics.update({
        "tier1_ok": bool(res["passed"]),
        "soft_min_eff": round(soft, 6),
        "ratio_vs_claimed": round(ratio, 4) if ratio is not None and math.isfinite(ratio) else None,
        "overhead_ratio": round(float(res["overhead_ratio"]), 3),
        "b2_max": round(float(res["b2_max"]), 5),
        "wall_s": round(time.perf_counter() - t0, 3),
    })
    passed = bool(res["passed"]) and (ratio is None or ratio >= 1.0 - tolerance)
    reason = "tier1 failed" if not res["passed"] else (
        f"reproduced {soft:.4g} vs claimed {claimed_soft_min:.4g}" if claimed_soft_min else f"reproduced {soft:.4g}"
    )
    return {"passed": passed, "metrics": metrics, "reason": reason}


def format_evaluator_json(
    row: dict,
    *,
    run_id: str = "default",
    autoresearch_dir: Path = ROOT / "autoresearch",
) -> dict:
    """Format an autoresearch trial row as an immutable evaluator JSON object.

    Follows the `{valid, score, metrics, evidence}` contract used by
    `Auto-research-agent` (`RSI-top` / `Fork-and-Flush`) so `apin autoresearch eval --json`
    works directly as a drop-in evaluator. Missing or non-finite observations are omitted
    from `metrics` rather than replaced with sentinels.
    """
    is_dry = bool(row.get("dry_run"))
    valid = (row.get("decision") == "dry_run_ok") if is_dry else bool(row.get("tier0_ok") and row.get("tier1_ok"))
    raw_score = row.get("soft_min_eff")
    score_val = float(raw_score) if (valid and isinstance(raw_score, (int, float)) and math.isfinite(raw_score)) else 0.0

    metrics: dict[str, float | int] = {}
    for key in (
        "soft_min_eff",
        "ratio_vs_incumbent",
        "short_horizon_soft_min",
        "long_horizon_soft_min",
        "median_adapt_cost",
        "overhead_ratio",
        "b2_max",
        "loc",
        "magic_constants",
        "wall_s",
    ):
        val = row.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            metrics[key] = val
    for cls_name, cls_val in sorted((row.get("by_class") or {}).items()):
        if isinstance(cls_val, (int, float)) and not isinstance(cls_val, bool) and math.isfinite(cls_val):
            metrics[f"class_{cls_name}"] = float(cls_val)

    lane = str(row.get("lane") or "A1")
    trial_id = str(row.get("trial_id") or "")
    evidence: list[str] = []
    run_dir = autoresearch_dir / "lanes" / lane / "runs" / run_id
    for cand in (
        run_dir / "ledger.jsonl",
        run_dir / "results.tsv",
        run_dir / "trials" / trial_id / "metrics.json" if trial_id else None,
    ):
        if cand is not None and cand.is_file():
            try:
                evidence.append(str(cand.relative_to(ROOT)))
            except ValueError:
                evidence.append(str(cand))

    return {
        "valid": valid,
        "score": score_val,
        "metrics": metrics,
        "evidence": evidence,
        "decision": row.get("decision"),
        "trial_id": trial_id,
        "lane": lane,
    }


def _extract_list(resp: object, *keys: str) -> list[dict]:
    if isinstance(resp, list):
        return [x for x in resp if isinstance(x, dict)]
    if isinstance(resp, dict):
        for k in keys:
            v = resp.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
            if isinstance(v, dict):
                return [val for _, val in sorted(v.items()) if isinstance(val, dict)]
    return []


def _brief_notes(lane: str | None, flywheel_url: str, limit: int) -> list[str]:
    """Open hypotheses and the most recent insights for a lane, from the live Flywheel."""
    where = f"on {lane}" if lane else "(all lanes)"
    try:
        hyp = _extract_list(flywheel.list_notes(lane=lane, kind="hypothesis", state="open", limit=limit,
                                                url=flywheel_url, timeout=5.0), "notes")
        ins = _extract_list(flywheel.list_notes(lane=lane, kind="insight", limit=limit,
                                                url=flywheel_url, timeout=5.0), "notes")
    except flywheel.FlywheelError as exc:
        return [f"## Hypotheses and Insights {where}", f"- (unavailable: {exc})", ""]
    ins = [n for n in ins if n.get("state") != "retracted"]

    def fmt(n: dict) -> str:
        tag = "" if lane else f"[{n.get('lane') or '-'}] "
        body = " ".join(str(n.get("body") or "").split())
        body = f" — {body[:160]}{'…' if len(body) > 160 else ''}" if body else ""
        links = n.get("node_ids") or []
        linked = f" (trials: {', '.join(f'`{x}`' for x in links[:4])})" if links else ""
        state = f" [{n.get('state')}]" if n.get("state") not in (None, "open") else ""
        return f"- {tag}`{n.get('id', '?')}` by `{n.get('author', '?')}`{state}: **{n.get('title', '')}**{body}{linked}"

    lines = [f"## Open Hypotheses {where} (test one, or add yours: `apin autoresearch note add --kind hypothesis`)"]
    lines += [fmt(n) for n in hyp] or ["- (none)"]
    lines += ["", f"## Recent Insights {where}"]
    lines += [fmt(n) for n in ins] or ["- (none)"]
    lines.append("")
    return lines


def build_research_brief(
    lane: str = "all",
    *,
    run_id: str | None = None,
    autoresearch_dir: Path = ROOT / "autoresearch",
    flywheel_url: str | None = flywheel.DEFAULT_PUBLIC_URL,
    offline: bool = False,
    max_failures: int = 8,
    stall_window: int = 5,
) -> str:
    """Assemble a compact, zero-synthesis verbatim brief for starting or flushing a research session.

    Combines live public Flywheel frontier/claims/dead-ends (read-only, no token needed)
    with local lane `NEGATIVE_MEMORY.md`, run `RESEARCH_HANDOFF.md`, and stall/bottleneck
    diagnostics from `ledger.jsonl`. Pass `lane='all'` for a cross-lane overview.
    """
    all_lanes = lane.strip().lower() == "all"
    lines: list[str] = [f"# Autoresearch Brief — {'All Lanes' if all_lanes else f'Lane {lane}'}", ""]
    lane_dir = (autoresearch_dir / "lanes" / lane) if not all_lanes else None

    # 1. Live Flywheel snapshot (public read-only; clean offline fallback)
    if not offline and flywheel_url:
        try:
            fr_raw = flywheel.get_frontier(url=flywheel_url, token="", timeout=5.0)
            nodes_raw = flywheel.get_nodes(lane=None if all_lanes else lane, limit=40, url=flywheel_url, token="", timeout=5.0)
            claims_raw = flywheel.get_claims(url=flywheel_url, token="", timeout=5.0)
            frontier = _extract_list(fr_raw, "frontier", "lanes", "items")
            nodes = _extract_list(nodes_raw, "nodes", "items")
            claims = _extract_list(claims_raw, "claims", "items")

            local_digest = _judge_digest()
            lines.append(f"## Live Flywheel Frontier (`{flywheel_url}` | local judge_digest=`{local_digest}`)")
            if frontier:
                for row in frontier:
                    rlane = row.get("lane") or "?"
                    sm = row.get("soft_min_eff")
                    sm_s = f"{float(sm):.4g}" if isinstance(sm, (int, float)) else "n/a"
                    ver = "verified" if row.get("verified") else "unverified"
                    rdims = row.get("dims")
                    dims_s = ",".join(str(x) for x in rdims) if isinstance(rdims, list) and rdims else "?"
                    rdigest = ""
                    mj = row.get("metrics_json")
                    if isinstance(mj, str) and mj:
                        try:
                            rdigest = str(json.loads(mj).get("judge_digest") or "")[:8]
                        except ValueError:
                            rdigest = ""
                    elif isinstance(mj, dict):
                        rdigest = str(mj.get("judge_digest") or "")[:8]
                    compat = ""
                    if (rdigest and not local_digest.startswith(rdigest)) or dims_s != "6,20":
                        compat = " [diff dims/judge — compare vs local incumbent]"
                    title = str(row.get("title") or row.get("hypothesis") or row.get("id") or "")[:80]
                    tag = " <-- THIS LANE" if rlane == lane else ""
                    lines.append(
                        f"- **{rlane}** ({ver}, soft_min={sm_s}, dims={dims_s}, judge={rdigest or '?'}): "
                        f"`{row.get('id', '?')}` — {title}{compat}{tag}"
                    )
            else:
                lines.append("- (no frontier nodes yet)")
            lines.append("")

            active_claims = [c for c in claims if (c.get("status") or "active") == "active"]
            lines.append("## Active Claims (avoid overlap)")
            if active_claims:
                for c in active_claims[:10]:
                    lines.append(f"- [{c.get('lane', '?')}] `{c.get('task_id', '?')}` by `{c.get('agent', '?')}` — {c.get('summary', '')}")
            else:
                lines.append("- (none)")
            lines.append("")

            discards = [n for n in nodes if str(n.get("decision") or "").startswith("discard")]
            lines.append(f"## Recent Discarded Trials on {lane} (do not repeat)")
            if discards:
                for n in discards[:max_failures]:
                    hyp = str(n.get("hypothesis") or n.get("title") or "")[:100]
                    lines.append(f"- `{n.get('id', '?')}` [{n.get('decision')}]: {hyp}")
            else:
                lines.append("- (none recorded on Flywheel)")
            lines.append("")
        except flywheel.FlywheelError as exc:
            lines += [f"## Live Flywheel Snapshot", f"- (unreachable: {exc})", ""]
        else:
            lines += _brief_notes(None if all_lanes else lane, flywheel_url, max_failures)

    # 2. Local NEGATIVE_MEMORY.md
    if lane_dir is not None:
        neg_path = lane_dir / "NEGATIVE_MEMORY.md"
        if neg_path.is_file():
            neg_lines = [ln for ln in neg_path.read_text().splitlines() if ln.strip()]
            if neg_lines:
                lines.append(f"## Local Negative Memory (`{neg_path.name}`)")
                lines.extend(neg_lines[:20])
                if len(neg_lines) > 20:
                    lines.append(f"... ({len(neg_lines) - 20} more lines in `{neg_path}`)")
                lines.append("")

    # 3. Local run ledger + stall / bottleneck diagnostics + RESEARCH_HANDOFF.md
    run_dir = (lane_dir / "runs" / run_id) if (lane_dir is not None and run_id) else None
    if run_dir is not None and run_dir.is_dir():
        ledger = _load_ledger(run_dir / "ledger.jsonl")
        lines.append(f"## Local Run `{run_id}` ({len(ledger)} trials)")
        if ledger:
            kept = [r for r in ledger if r.get("decision") in flywheel.KEPT_DECISIONS]
            tail_non_kept = 0
            for r in reversed(ledger):
                if r.get("decision") in flywheel.KEPT_DECISIONS:
                    break
                tail_non_kept += 1
            best = kept[-1] if kept else ledger[-1]
            lines.append(
                f"- Incumbent: `{best.get('trial_id')}` decision={best.get('decision')} "
                f"soft_min={best.get('soft_min_eff')} short={best.get('short_horizon_soft_min')} "
                f"long={best.get('long_horizon_soft_min')} adapt_cost={best.get('median_adapt_cost')} "
                f"overhead={best.get('overhead_ratio')}x"
            )
            by_cls = best.get("by_class") or {}
            if by_cls:
                ranked_cls = sorted(
                    ((k, float(v)) for k, v in by_cls.items() if isinstance(v, (int, float)) and math.isfinite(v)),
                    key=lambda kv: kv[1],
                )
                if ranked_cls:
                    weakest = ", ".join(f"{k}={v:.4g}" for k, v in ranked_cls[:3])
                    lines.append(f"- Weakest target classes (attack these first): {weakest}")
            if tail_non_kept >= stall_window:
                lines.append(
                    f"- **STALL DETECTED ({tail_non_kept} consecutive non-kept trials):** "
                    f"Consider a Fork-and-Flush reset — start a fresh context from `incumbent.py` "
                    f"and target a distinct mechanism or cross-lane composition instead of micro-tweaks."
                )
        handoff = run_dir / "RESEARCH_HANDOFF.md"
        if handoff.is_file():
            h_lines = handoff.read_text().splitlines()
            lines.append(f"### Local Handoff (`{handoff.name}`)")
            lines.extend(h_lines[:25])
            if len(h_lines) > 25:
                lines.append(f"... ({len(h_lines) - 25} more lines in `{handoff}`)")
        lines.append("")

    lines += [
        "## Action Checklist",
        "- Prefer **composing** compatible verified ideas across lanes before inventing complex heuristics.",
        "- Verify edits with `uv run apin autoresearch eval --lane " + lane + " --run-id <RUN_ID> --repeats 2 --hypothesis \"...\"`.",
        "- Freeze `submission.py` before running full real-Stan verification (`apin run logdensity` + `apin score --submission`).",
    ]
    return "\n".join(lines) + "\n"
