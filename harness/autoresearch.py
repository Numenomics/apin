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

from . import invariance, ipc, judge, logdensity, queue, shapes
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
        "dim", "seed", "repeats", "k_steps", "adapt_draws", "horizon_steps",
        "tier0_ok", "tier1_ok", "soft_min_eff", "incumbent_soft_min", "ratio_vs_incumbent",
        "b2_max", "overhead_ratio", "queue_wait_s", "decision", "archive_cells",
        "codex_refs", "hypothesis",
    ]
    with open(tsv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {
                **r,
                "archive_cells": ",".join(r.get("archive_cells") or []),
                "codex_refs": ",".join(r.get("codex_refs") or []),
            }
            w.writerow(flat)


def _append_row(ledger_path: Path, tsv_path: Path, row: dict) -> dict:
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    _write_results_tsv(ledger_path, tsv_path)
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
            alpha=0.001, lane=lane, isolated=True, horizon_steps=horizon_steps,
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


def _build_tier1_suite(dim: int, seed: int) -> list[shapes.AnalyticTarget]:
    """Fresh aligned + rotated Tier 1 suite covering all non-multimodal shape classes."""
    return [
        shapes.IllCondGaussian(dim, seed + 1, rotated=False, kappa=50.0),
        shapes.IllCondGaussian(dim, seed + 2, rotated=True, kappa=50.0),
        shapes.NealsFunnel(dim, seed + 3, rotated=True, v_sd=2.0),
        shapes.Banana(dim, seed + 4, rotated=True),
        shapes.HybridRosenbrock(dim, seed + 5, rotated=False, b=5.0),
        shapes.StudentT(dim, seed + 6, rotated=False, nu=1.5),
        shapes.StudentT(dim, seed + 7, rotated=True, nu=3.0),
        shapes.GaussianMixture(dim, seed + 8, rotated=True, separation=2.5),
    ]


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
    all_ok = True
    b2_worst = 0.0
    oh_vals = []
    for i, tgt in enumerate(suite):
        res = invariance.check_target_invariance(
            tgt, sub_path, seed=seed + 100 * i, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, isolated=True, horizon_steps=horizon_steps,
        )
        per_target.append({
            "family": tgt.family, "rotated": tgt.rotated, "passed": res.passed,
            "emp_eff": res.empirical_efficiency, "b2_max": res.b2_max,
            "max_z": res.max_z, "z_threshold": res.z_threshold,
            "adapt_cost": res.adapt_cost, "adapt_cost_amortized": res.adapt_cost_amortized,
            "cost_per_chain": res.cost_per_chain, "overhead_ratio": res.overhead_ratio,
            "reason": res.reason,
        })
        if not res.passed:
            all_ok = False
        b2_worst = max(b2_worst, res.b2_max)
        oh_vals.append(res.overhead_ratio)
        if not tgt.multimodal:
            by_class.setdefault(tgt.family, []).append(res.empirical_efficiency)

    class_geo = {
        c: float(np.exp(np.mean(np.log(np.maximum(vs, 1e-18))))) for c, vs in by_class.items()
    }
    soft_min = invariance.soft_min_power_mean(list(class_geo.values()), p=-4.0) if all_ok else 0.0
    geomean = float(np.exp(np.mean(np.log(np.maximum(list(class_geo.values()), 1e-18))))) if class_geo else 0.0
    return {
        "passed": all_ok,
        "soft_min_eff": soft_min,
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
    geomean = float(np.exp(np.mean(np.log(np.maximum(list(by_class.values()), 1e-18))))) if by_class else 0.0
    return {
        "passed": all_ok,
        "soft_min_eff": soft_min,
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
            vals.append(res.empirical_efficiency)
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
) -> dict:
    """Run one complete Karpathy-ratchet + paired MAP-Elites autoresearch trial on `lane`."""
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

    base_meta = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "candidate_sha": cand_sha,
        "ast_sha": cand_ast_sha,
        "loc": cand_loc,
        "magic_constants": cand_consts,
        "judge_digest": _judge_digest(),
        "dim": int(dim),
        "seed": seed,
        "repeats": repeats,
        "k_steps": int(k_steps),
        "adapt_draws": int(adapt_draws),
        "horizon_steps": int(horizon_steps),
        "hypothesis": hypothesis,
        "codex_refs": list(codex_refs or []),
    }

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
        return _append_row(ledger_path, tsv_path, row)

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
            return _append_row(ledger_path, tsv_path, row)

        # Tier 1: paired fresh suites for candidate AND incumbent on identical fresh seeds
        suites = [(seed + r * 7919, _build_tier1_suite(dim, seed + r * 7919)) for r in range(repeats)]
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
            return _append_row(ledger_path, tsv_path, row)

        inc_sha = None
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
            inc_sm = inc_res["soft_min_eff"]
            cand_sm = cand_res["soft_min_eff"]
            ratio = (cand_sm / inc_sm) if inc_sm > 0 else (math.inf if cand_sm > 0 else 0.0)
            if not cand_res["passed"]:
                decision = "discard_tier1"
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

        if decision in ("keep", "keep_simpler", "keep_baseline"):
            shutil.copy2(snap_path, incumbent_path)

    (trial_work / "metrics.json").write_text(
        json.dumps({"tier0": t0_info, "candidate": cand_res, "incumbent": inc_res}, indent=1)
    )
    shutil.copy2(snap_path, trial_work / "submission.py")

    row = {
        **base_meta,
        "tier0_ok": True,
        "tier1_ok": cand_res["passed"],
        "soft_min_eff": round(cand_res["soft_min_eff"], 6),
        "incumbent_soft_min": round(inc_res["soft_min_eff"], 6) if inc_res else round(cand_res["soft_min_eff"], 6),
        "ratio_vs_incumbent": round(ratio, 4) if math.isfinite(ratio) else None,
        "b2_max": round(cand_res["b2_max"], 5),
        "overhead_ratio": round(cand_res["overhead_ratio"], 3),
        "queue_wait_s": round(queue_wait_s, 3),
        "archive_cells": archived_cells,
        "decision": decision,
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    return _append_row(ledger_path, tsv_path, row)
