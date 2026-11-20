"""Slice 5 Autoresearch MVP runner: Karpathy 3-file contract + MAP-Elites specialist archive.

Combines:
  - Pre-eval novelty filter (diff SHA-256 dedupe against ledger.jsonl + AST import lint)
  - Tier 0 fast reject (< 30 s): ABI check, frozen-tuning SHA-256, replay determinism, and
    coarse exact-draw invariance + dispersed-start mixing on fresh rotated Gaussian & funnel
  - Tier 1 paired dev eval: candidate vs lane incumbent on IDENTICAL fresh seeds across
    aligned + rotated analytic shapes (gaussian_illcond, funnel, banana, student_t, mixture)
  - Noise-aware ratchet + MAP-Elites specialist archive (`archive/<cell>/`)
  - Append-only `ledger.jsonl` + auto-generated flat `results.tsv`
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import invariance, logdensity, shapes
from .models import ROOT

NOISE_MARGIN = 0.02  # +/- 2% tie band where simplicity regularizer breaks ties


def _judge_digest() -> str:
    h = hashlib.sha256()
    for mod in (logdensity, shapes, invariance):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def _submission_stats(path: Path) -> tuple[str, int]:
    text = path.read_text()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    loc = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    return sha, loc


def _load_ledger(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_results_tsv(ledger_path: Path, tsv_path: Path) -> None:
    rows = _load_ledger(ledger_path)
    cols = [
        "trial_id", "timestamp", "candidate_sha", "loc", "tier0_ok", "tier1_ok",
        "soft_min_eff", "incumbent_soft_min", "ratio_vs_incumbent", "b2_max",
        "overhead_ratio", "decision", "archive_cells", "codex_refs", "hypothesis",
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


def _eval_tier0(sub_path: Path, seed: int, work_dir: Path) -> tuple[bool, dict]:
    """Tier 0 fast reject: AST lint, ABI + replay determinism, coarse invariance smoke on Gaussian & funnel."""
    try:
        logdensity.check_imports(sub_path.read_text())
        logdensity.load_submission(sub_path)
    except Exception as e:
        return False, {"stage": "load_lint", "error": str(e)}

    # 1. ABI + frozen tuning + 5-step replay on a fresh rotated Gaussian (d=4)
    g4 = shapes.IllCondGaussian(dim=4, seed=seed + 11, rotated=True, kappa=25.0)
    try:
        _, summary = logdensity.execute_run(
            g4, sub_path, logdensity.Budget(1, 20, 200_000, 30.0),
            run_seed=seed + 13, work_dir=work_dir / "t0_abi", replay_count=5,
        )
    except Exception as e:
        return False, {"stage": "abi_replay", "error": str(e)}

    # 2. Coarse exact-draw invariance smoke on rotated Gaussian (d=6) and small Neal's funnel (d=4)
    smoke_targets = [
        shapes.IllCondGaussian(dim=6, seed=seed + 17, rotated=True, kappa=25.0),
        shapes.NealsFunnel(dim=4, seed=seed + 19, rotated=False, v_sd=2.0),
    ]
    checks = []
    for tgt in smoke_targets:
        res = invariance.check_target_invariance(
            tgt, sub_path, seed=seed + 23, n_chains=48, k_steps=12, adapt_draws=80, alpha=0.001
        )
        checks.append({
            "target": tgt.family, "passed": res.passed, "max_z": round(res.max_z, 3),
            "b2_max": round(res.b2_max, 5), "move": round(res.mean_movement_sd, 4), "reason": res.reason,
        })
        if not res.passed:
            return False, {"stage": "invariance_smoke", "checks": checks, "error": res.reason}

    return True, {"stage": "passed", "tuning_bytes": summary["adapt"]["tuning_bytes"], "checks": checks}


def _eval_tier1_suite(sub_path: Path, suite: list[shapes.AnalyticTarget], seed: int,
                      n_chains: int = 64, k_steps: int = 20, adapt_draws: int = 120) -> dict:
    """Evaluate a submission across a fresh aligned + rotated analytic suite."""
    per_target = []
    by_class: dict[str, list[float]] = {}
    all_ok = True
    b2_worst = 0.0
    oh_vals = []
    for i, tgt in enumerate(suite):
        res = invariance.check_target_invariance(
            tgt, sub_path, seed=seed + 100 * i, n_chains=n_chains, k_steps=k_steps, adapt_draws=adapt_draws
        )
        per_target.append({
            "family": tgt.family, "rotated": tgt.rotated, "passed": res.passed,
            "emp_eff": res.empirical_efficiency, "b2_max": res.b2_max,
            "max_z": res.max_z, "overhead_ratio": res.overhead_ratio, "reason": res.reason,
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


def _overhead_bucket(ratio: float) -> str:
    if ratio < 1.0:
        return "oh_low"
    if ratio < 5.0:
        return "oh_med"
    return "oh_high"


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
    autoresearch_dir: Path = ROOT / "autoresearch",
    run_id: str = "default",
    allow_duplicate: bool = False,
) -> dict:
    """Run one complete Karpathy-ratchet + MAP-Elites autoresearch trial on `lane`."""
    lane_dir = autoresearch_dir / "lanes" / lane
    sub_path = lane_dir / "submission.py"
    if not sub_path.is_file():
        raise FileNotFoundError(f"missing editable submission at {sub_path}")

    run_dir = lane_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    tsv_path = run_dir / "results.tsv"
    incumbent_path = lane_dir / "incumbent.py"
    archive_dir = lane_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    prior = _load_ledger(ledger_path)
    trial_id = f"T{len(prior) + 1:04d}"
    cand_sha, cand_loc = _submission_stats(sub_path)
    seed = int(seed if seed is not None else (20261120 + len(prior) * 97))
    t0 = time.perf_counter()

    # Pre-eval novelty filter: reject exact duplicate diff
    if not allow_duplicate and any(r.get("candidate_sha") == cand_sha for r in prior):
        row = {
            "trial_id": trial_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "lane": lane, "candidate_sha": cand_sha, "loc": cand_loc,
            "judge_digest": _judge_digest(), "seed": seed,
            "hypothesis": hypothesis, "codex_refs": list(codex_refs or []),
            "tier0_ok": False, "tier1_ok": False, "decision": "discard_duplicate",
            "wall_s": round(time.perf_counter() - t0, 3),
        }
        with open(ledger_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        _write_results_tsv(ledger_path, tsv_path)
        return row

    trial_work = run_dir / "trials" / trial_id
    trial_work.mkdir(parents=True, exist_ok=True)
    t0_ok, t0_info = _eval_tier0(sub_path, seed=seed, work_dir=trial_work)
    if not t0_ok:
        row = {
            "trial_id": trial_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "lane": lane, "candidate_sha": cand_sha, "loc": cand_loc,
            "judge_digest": _judge_digest(), "seed": seed,
            "hypothesis": hypothesis, "codex_refs": list(codex_refs or []),
            "tier0_ok": False, "tier0_info": t0_info, "tier1_ok": False,
            "decision": "discard_tier0", "wall_s": round(time.perf_counter() - t0, 3),
        }
        with open(ledger_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        _write_results_tsv(ledger_path, tsv_path)
        return row

    # Tier 1: paired fresh suite for candidate AND incumbent on identical fresh seeds
    suite = [
        shapes.IllCondGaussian(dim, seed + 1, rotated=False, kappa=50.0),
        shapes.IllCondGaussian(dim, seed + 2, rotated=True, kappa=50.0),
        shapes.Banana(dim, seed + 3, rotated=False),
        shapes.StudentT(dim, seed + 4, rotated=False, nu=3.0),
        shapes.GaussianMixture(dim, seed + 5, rotated=False, separation=2.5),
    ]
    cand_res = _eval_tier1_suite(sub_path, suite, seed=seed + 1000, n_chains=n_chains,
                                 k_steps=k_steps, adapt_draws=adapt_draws)

    # First valid trial establishes incumbent automatically
    if not incumbent_path.is_file():
        inc_res = None
        inc_loc = cand_loc
        ratio = 1.0 if cand_res["passed"] else 0.0
        decision = "keep_baseline" if cand_res["passed"] else "discard_tier1"
    else:
        _, inc_loc = _submission_stats(incumbent_path)
        inc_res = _eval_tier1_suite(incumbent_path, suite, seed=seed + 1000, n_chains=n_chains,
                                    k_steps=k_steps, adapt_draws=adapt_draws)
        inc_sm = inc_res["soft_min_eff"]
        cand_sm = cand_res["soft_min_eff"]
        ratio = (cand_sm / inc_sm) if inc_sm > 0 else (math.inf if cand_sm > 0 else 0.0)
        if not cand_res["passed"]:
            decision = "discard_tier1"
        elif ratio >= 1.0 + NOISE_MARGIN:
            decision = "keep"
        elif ratio >= 1.0 - NOISE_MARGIN and cand_loc < inc_loc:
            decision = "keep_simpler"
        else:
            decision = "discard_no_gain"

    # Update MAP-Elites specialist archive for any gate-passing candidate
    archived_cells: list[str] = []
    if cand_res["passed"]:
        oh_tag = _overhead_bucket(cand_res["overhead_ratio"])
        for cls_name, cls_score in cand_res["by_class"].items():
            cell_key = f"{cls_name}__{oh_tag}"
            cell_dir = archive_dir / cell_key
            meta_file = cell_dir / "best.json"
            prev_best = json.loads(meta_file.read_text())["score"] if meta_file.is_file() else -1.0
            if cls_score > prev_best:
                cell_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sub_path, cell_dir / "submission.py")
                meta_file.write_text(json.dumps({
                    "trial_id": trial_id, "candidate_sha": cand_sha, "score": cls_score,
                    "soft_min_eff": cand_res["soft_min_eff"], "hypothesis": hypothesis,
                }, indent=1))
                archived_cells.append(cell_key)
        if decision == "discard_no_gain" and archived_cells:
            decision = "archive_only"

    if decision in ("keep", "keep_simpler", "keep_baseline"):
        shutil.copy2(sub_path, incumbent_path)

    (trial_work / "metrics.json").write_text(json.dumps({"tier0": t0_info, "candidate": cand_res, "incumbent": inc_res}, indent=1))
    shutil.copy2(sub_path, trial_work / "submission.py")

    row = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "candidate_sha": cand_sha,
        "loc": cand_loc,
        "judge_digest": _judge_digest(),
        "seed": seed,
        "hypothesis": hypothesis,
        "codex_refs": list(codex_refs or []),
        "tier0_ok": True,
        "tier1_ok": cand_res["passed"],
        "soft_min_eff": round(cand_res["soft_min_eff"], 6),
        "incumbent_soft_min": round(inc_res["soft_min_eff"], 6) if inc_res else round(cand_res["soft_min_eff"], 6),
        "ratio_vs_incumbent": round(ratio, 4) if math.isfinite(ratio) else None,
        "b2_max": round(cand_res["b2_max"], 5),
        "overhead_ratio": round(cand_res["overhead_ratio"], 3),
        "archive_cells": archived_cells,
        "decision": decision,
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    _write_results_tsv(ledger_path, tsv_path)
    return row
