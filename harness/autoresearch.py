"""The lane trial loop: `apin autoresearch eval`.

A trial is: pre-checks (lint, duplicate filter, snapshot, evaluation queue), then one run of the
grader (harness/grade.py) on the candidate at the standard public configuration, then a decision
from the candidate's speed-up over the lane's starting code.

Each lane directory holds its frozen starting code (`start.py`) and the frozen public-seed score
of that code (`yardstick.json`, written by scripts/freeze_harbor_references.py). The speed-up is
candidate soft-min / yardstick soft-min. It is deterministic for a given file and comparable across
runs and agents, because every trial is scored on the same seeds and configuration.

Decisions (KEPT_DECISIONS in harness/flywheel.py are the kept ones):
  keep_baseline    the first graded trial of a run (or of a run under a new yardstick)
  keep             speed-up >= IMPROVEMENT_MARGIN x the best speed-up kept so far in this run
  keep_simpler     within TIE_BAND of that best and simpler than the current incumbent
  archive_only     no overall gain, but a new best score for one target class (MAP-Elites cell)
  discard_no_gain  anything else that passed every check
  discard_tier0 / discard_tier1   failed a check (Tier 1 after the confirmation recheck)
  discard_duplicate / discard_error
  exploratory      passed every check at a non-standard configuration: never kept, no speed-up

Every run keeps its own `incumbent.py`, `archive/`, `ledger.jsonl` and `results.tsv` under
`autoresearch/lanes/<LANE>/runs/<run-id>/`.
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

from . import flywheel, grade, judge, logdensity, queue
from .grade import IMPROVEMENT_MARGIN
from .models import ROOT

# Tie band for the simplicity rule. At the standard configuration, copies of one sampler that
# differ only in their random numbers score within 2.5% of each other (p95 1.1%, the A/A
# calibration in harbor/README.md). A candidate within 2% of the best is therefore
# indistinguishable from it, and the simpler file wins.
TIE_BAND = 0.02
START_FILE = "start.py"
YARDSTICK_FILE = "yardstick.json"
# A rerun (`apin autoresearch verify-next`) uses the same grader on the same seeds, so it should
# reproduce the posted speed-up exactly. The tolerance only absorbs floating-point differences
# between machines.
REPLICATION_TOLERANCE = 1e-3
_TRIVIAL_CONSTS = frozenset({-1, 0, 1, 2, -1.0, 0.0, 1.0, 2.0})


class YardstickError(RuntimeError):
    """The lane has no yardstick that matches today's judge, configuration and starting code."""


def _sig(v, digits: int = 6) -> float | None:
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{float(v):.{digits}g}")


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


TSV_COLUMNS = (
    "trial_id", "timestamp", "candidate_sha", "loc", "magic_constants",
    "dims", "seed", "repeats", "k_steps", "adapt_draws", "horizon_steps",
    "tier0_ok", "tier1_ok", "speedup", "yardstick", "best_speedup", "ratio_vs_best",
    "soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min", "median_adapt_cost",
    "b2_max", "overhead_ratio", "queue_wait_s", "decision", "archive_cells",
    "codex_refs", "hypothesis",
)


def _write_results_tsv(ledger_path: Path, tsv_path: Path) -> None:
    rows = _load_ledger(ledger_path)
    with open(tsv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
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


# ---------------------------------------------------------------------------------------------
# Yardstick
# ---------------------------------------------------------------------------------------------

def load_yardstick(lane: str, *, autoresearch_dir: Path = ROOT / "autoresearch", digest: str | None = None) -> dict:
    """The lane's frozen public-seed score of its starting code. Raises YardstickError unless it
    matches today's judge digest, the standard configuration and the bytes of `start.py`."""
    lane_dir = Path(autoresearch_dir) / "lanes" / lane
    path = lane_dir / YARDSTICK_FILE
    fix = "Refreeze it: uv run python scripts/freeze_harbor_references.py --public-only --lanes " + lane
    if not path.is_file():
        raise YardstickError(f"lane {lane} has no {YARDSTICK_FILE}. {fix}")
    try:
        yd = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise YardstickError(f"cannot read {path}: {e}") from e
    digest = digest or judge.judge_digest()
    start = lane_dir / START_FILE
    problems = []
    if yd.get("schema") != grade.YARDSTICK_SCHEMA or yd.get("lane") != lane or yd.get("role") != "start":
        problems.append("wrong schema or lane")
    if yd.get("judge_digest") != digest:
        problems.append(f"judge digest {yd.get('judge_digest')} is not today's {digest}")
    if yd.get("config") != grade.config():
        problems.append("its configuration is not the standard one")
    if start.is_file() and grade.sha256(start) != yd.get("sha256"):
        problems.append(f"{START_FILE} changed after it was frozen")
    if yd.get("id") != grade.yardstick_id(yd):
        problems.append("its id does not match its contents")
    if not grade.soft_min(yd):
        problems.append("the starting code failed a check when it was frozen")
    if problems:
        raise YardstickError(f"the yardstick of lane {lane} is stale: {'; '.join(problems)}. {fix}")
    return yd


def _yardstick_score(yd: dict) -> dict:
    """A frozen yardstick read as a grader result, for a file byte-identical to start.py."""
    out = {k: yd[k] for k in grade.FROZEN_KEYS if k in yd}
    out["wall_s"] = 0.0
    return out


# ---------------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------------

def _overhead_bucket(ratio: float) -> str:
    if ratio < 1.0:
        return "oh_low"
    if ratio < 5.0:
        return "oh_med"
    return "oh_high"


def _is_simpler(cand_loc: int, inc_loc: int, cand_consts: int, inc_consts: int,
                cand_oh: float, inc_oh: float) -> bool:
    """Simplicity rule: fewer LOC, or (loc <= inc) with fewer magic constants or >=15% lower overhead."""
    if cand_loc < inc_loc:
        return True
    if cand_loc <= inc_loc and cand_consts < inc_consts:
        return True
    if cand_loc <= inc_loc and inc_oh > 0.0 and cand_oh <= 0.85 * inc_oh:
        return True
    return False


def decide(speedup: float, best: float | None, simpler: bool) -> str:
    """Decision for a graded trial that passed every check, given the best speed-up kept so far
    in this run (None for the run's first graded trial)."""
    if best is None:
        return "keep_baseline"
    if speedup >= IMPROVEMENT_MARGIN * best:
        return "keep"
    if speedup >= (1.0 - TIE_BAND) * best and simpler:
        return "keep_simpler"
    return "discard_no_gain"


def _run_state(prior: list[dict], yardstick: str | None) -> tuple[dict | None, dict | None]:
    """(best kept row, incumbent row) among this run's kept rows graded against `yardstick`."""
    kept = [r for r in prior if r.get("decision") in flywheel.KEPT_DECISIONS and yardstick is not None
            and r.get("yardstick") == yardstick and isinstance(r.get("speedup"), (int, float))]
    if not kept:
        return None, None
    best = kept[0]
    for r in kept[1:]:
        if r["speedup"] > best["speedup"]:
            best = r
    return best, kept[-1]


def _update_archive(archive_dir: Path, snap_path: Path, score: dict, *, trial_id: str, cand_sha: str,
                    yardstick: str, speedup: float, hypothesis: str) -> list[str]:
    """MAP-Elites cells (target class x overhead bucket): store the candidate where it beats the
    cell's specialist by IMPROVEMENT_MARGIN. Every graded trial is scored on the same seeds, so the
    stored class scores compare directly and the specialist is never rescored."""
    cells = []
    oh_tag = _overhead_bucket(float(score.get("overhead_ratio") or 0.0))
    for cls_name, cls_score in (score.get("by_class") or {}).items():
        cell_dir = archive_dir / f"{cls_name}__{oh_tag}"
        meta_file, spec_file = cell_dir / "best.json", cell_dir / "submission.py"
        store = True
        if meta_file.is_file() and spec_file.is_file():
            prev = json.loads(meta_file.read_text())
            if prev.get("candidate_sha") == cand_sha:
                store = False
            elif prev.get("yardstick") == yardstick and isinstance(prev.get("score"), (int, float)):
                store = cls_score >= IMPROVEMENT_MARGIN * float(prev["score"])
        if store:
            cell_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap_path, spec_file)
            meta_file.write_text(json.dumps({
                "trial_id": trial_id, "candidate_sha": cand_sha, "score": cls_score, "yardstick": yardstick,
                "speedup": speedup, "hypothesis": hypothesis,
            }, indent=1))
            cells.append(cell_dir.name)
    return cells


# ---------------------------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------------------------

def evaluate_lane_trial(
    lane: str,
    hypothesis: str,
    *,
    cfg: dict | None = None,
    codex_refs: list[str] | None = None,
    autoresearch_dir: Path = ROOT / "autoresearch",
    run_id: str = "default",
    allow_duplicate: bool = False,
    submission_path: Path | None = None,
    dry_run: bool = False,
    use_queue: bool = True,
    combines: list[str] | None = None,
    fork_of: list[str] | None = None,
) -> dict:
    """Run one trial of `lane` and return its ledger row.

    `cfg` defaults to the standard public configuration (grade.config()). Any other configuration
    makes the trial exploratory: it is graded and recorded, but it gets no speed-up, is never kept
    and never becomes the incumbent. ``combines`` / ``fork_of`` name Flywheel trial ids whose
    mechanism this candidate reuses or whose source it started from. They are posted as extra
    parents next to the incumbent.
    """
    cfg = grade.config() if cfg is None else dict(cfg)
    exploratory = not grade.is_standard(cfg)
    lane_dir = Path(autoresearch_dir) / "lanes" / lane
    sub_path = Path(submission_path) if submission_path is not None else (lane_dir / "submission.py")
    if not sub_path.is_file():
        raise FileNotFoundError(f"missing editable submission at {sub_path}")

    run_dir = lane_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    tsv_path = run_dir / "results.tsv"
    incumbent_path = run_dir / "incumbent.py"
    archive_dir = run_dir / "archive"

    prior = _load_ledger(ledger_path)
    last_kept = next((r for r in reversed(prior) if r.get("decision") in flywheel.KEPT_DECISIONS), None)
    parent_node_id = (
        f"{lane}:{run_id}:{last_kept['trial_id']}" if last_kept and last_kept.get("trial_id") else None
    )
    parent_links = flywheel.parent_links(parent_node_id, combines=combines, fork_of=fork_of)
    extra_parents = [p for p in parent_links if p["relation"] != "incumbent"]
    trial_id = f"T{len(prior) + 1:04d}"
    digest = judge.judge_digest()
    t0 = time.perf_counter()

    # Snapshot submission.py before any read or eval, so a concurrent edit cannot race the trial.
    raw = sub_path.read_bytes()
    raw_text = raw.decode("utf-8", errors="replace")
    if dry_run:
        snap_path = sub_path
    else:
        trial_work = run_dir / "trials" / trial_id
        trial_work.mkdir(parents=True, exist_ok=True)
        snap_path = trial_work / "submission_snapshot.py"
        snap_path.write_bytes(raw)
    cand_sha, cand_ast_sha, cand_loc, cand_consts = _submission_stats(snap_path)

    base_meta = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "candidate_sha": cand_sha,
        "ast_sha": cand_ast_sha,
        "loc": cand_loc,
        "magic_constants": cand_consts,
        "judge_digest": digest,
        "dims": list(cfg["dims"]),
        "seed": cfg["seed"],
        "repeats": cfg["repeats"],
        "n_chains": cfg["n_chains"],
        "k_steps": cfg["k_steps"],
        "adapt_draws": cfg["adapt_draws"],
        "horizon_steps": cfg["horizon_steps"],
        "exploratory": exploratory,
        "yardstick": None,
        "speedup": None,
        "hypothesis": hypothesis,
        "codex_refs": list(codex_refs or []),
    }
    if extra_parents:
        base_meta["parents"] = extra_parents

    # Novelty filter: reject an exact duplicate or an AST-equivalent (comment or whitespace) change.
    # An exploratory trial does not count against a standard one: the report of an exploratory
    # trial tells the agent to grade the same file again at the standard setup.
    is_dup = any(
        (r.get("candidate_sha") == cand_sha or (r.get("ast_sha") and r.get("ast_sha") == cand_ast_sha))
        and (exploratory or not r.get("exploratory"))
        for r in prior
    )

    if dry_run:
        lint_error = None
        try:
            logdensity.check_imports(raw_text)
            grade.load_check(snap_path, lane)
        except Exception as e:
            lint_error = str(e)
        return {
            **base_meta,
            "dry_run": True,
            "is_duplicate": is_dup,
            "lint_ok": lint_error is None,
            "lint_error": lint_error,
            "decision": (
                "dry_run_duplicate" if (is_dup and not allow_duplicate)
                else ("dry_run_lint_error" if lint_error is not None else "dry_run_ok")
            ),
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    def finish(row: dict) -> dict:
        row["wall_s"] = round(time.perf_counter() - t0, 3)
        return _append_row(ledger_path, tsv_path, row, run_id=run_id, parent_id=parent_node_id,
                           parents=parent_links, source=raw_text)

    if not allow_duplicate and is_dup:
        return finish({**base_meta, "tier0_ok": False, "tier1_ok": False, "queue_wait_s": 0.0,
                       "decision": "discard_duplicate"})

    yd = None if exploratory else load_yardstick(lane, autoresearch_dir=autoresearch_dir, digest=digest)
    best_row, inc_row = _run_state(prior, yd["id"] if yd else None)
    best = float(best_row["speedup"]) if best_row else None
    if yd is not None:
        base_meta.update(yardstick=yd["id"], best_speedup=best,
                         best_trial=best_row.get("trial_id") if best_row else None)

    with queue.eval_slot(f"autoresearch:{lane}/{run_id}/{trial_id}", enabled=use_queue) as queue_wait_s:
        if yd is not None and grade.sha256(snap_path) == yd["sha256"]:
            # The lane's own starting code: its score at this configuration is the yardstick.
            score, graded_from = _yardstick_score(yd), "yardstick"
        else:
            graded_from = "grader"
            try:
                score = grade.score_file(snap_path, lane, cfg, trial_work / "grade")
            except Exception as e:
                return finish({**base_meta, "tier0_ok": False, "tier1_ok": False,
                               "queue_wait_s": round(queue_wait_s, 3), "decision": "discard_error",
                               "error": f"{type(e).__name__}: {e}"})

        sp = grade.speedup(score, yd) if yd is not None else None
        if not score.get("tier0_ok"):
            decision = "discard_tier0"
        elif not score.get("valid") or score.get("unpaired") or (not exploratory and sp is None):
            decision = "discard_tier1"
        elif exploratory:
            decision = "exploratory"
        else:
            simpler = inc_row is not None and _is_simpler(
                cand_loc, int(inc_row.get("loc") or 0), cand_consts, int(inc_row.get("magic_constants") or 0),
                float(score.get("overhead_ratio") or 0.0), float(inc_row.get("overhead_ratio") or 0.0))
            decision = decide(sp, best, simpler)

        archived: list[str] = []
        if sp is not None:
            archived = _update_archive(archive_dir, snap_path, score, trial_id=trial_id, cand_sha=cand_sha,
                                       yardstick=yd["id"], speedup=sp, hypothesis=hypothesis)
            if decision == "discard_no_gain" and archived:
                decision = "archive_only"
        if decision in flywheel.KEPT_DECISIONS:
            shutil.copy2(snap_path, incumbent_path)

    (trial_work / "metrics.json").write_text(json.dumps(
        {"config": cfg, "yardstick": yd["id"] if yd else None, "graded_from": graded_from, "score": score},
        indent=1, default=str))
    shutil.copy2(snap_path, trial_work / "submission.py")

    row = {
        **base_meta,
        "tier0_ok": bool(score.get("tier0_ok")),
        "tier1_ok": bool(score.get("valid")) and not score.get("unpaired"),
        "confirmed": bool((score.get("confirmation") or {}).get("ran") and score["confirmation"].get("passed")),
        "graded_from": graded_from,
        "speedup": _sig(sp),
        "ratio_vs_best": _sig(sp / best) if (sp is not None and best) else None,
        "soft_min_eff": _sig(score.get("soft_min_eff", 0.0)),
        "short_horizon_soft_min": _sig(score.get("short_horizon_soft_min", 0.0)),
        "long_horizon_soft_min": _sig(score.get("long_horizon_soft_min", 0.0)),
        "median_adapt_cost": round(float(score.get("median_adapt_cost") or 0.0), 1),
        "b2_max": round(float(score.get("b2_max") or 0.0), 5),
        "overhead_ratio": round(float(score.get("overhead_ratio") or 0.0), 3),
        "by_class": {k: _sig(v) for k, v in (score.get("by_class") or {}).items()},
        "queue_wait_s": round(queue_wait_s, 3),
        "archive_cells": archived,
        "decision": decision,
    }
    if decision in flywheel.KEPT_DECISIONS:
        row["best_speedup_after"] = max(row["speedup"], best or 0.0)
    if decision.startswith("discard_tier"):
        row["error"] = score.get("error") or (
            "passed only on the confirmation seed; the public seed left a target without an efficiency, "
            "so there is no score to compare" if score.get("unpaired") else "failed")
        row["failed_targets"] = score.get("failed_targets", [])
        if not score.get("tier0_ok"):
            row["tier0_info"] = score.get("tier0_info")
    return finish(row)


DUPLICATE_HINT = ("The same code was already tried in this run (comments and whitespace do not count), so it "
                  "was not graded again. Its earlier row in results.tsv has the outcome. "
                  "Pass --allow-duplicate to grade it anyway.")


def trial_report(row: dict, run_id: str) -> list[str]:
    """The lines `apin autoresearch eval` prints for one trial row."""
    dims = ",".join(str(x) for x in (row.get("dims") or []))
    lines = [f"{row['trial_id']} [{row['lane']}] decision={row['decision']} speedup={row.get('speedup')} "
             f"tier0={row.get('tier0_ok')} tier1={row.get('tier1_ok')} dims={dims} repeats={row.get('repeats')} "
             f"seed={row.get('seed')} yardstick={row.get('yardstick')} qwait={row.get('queue_wait_s')}s "
             f"wall={row.get('wall_s')}s"]
    if not row.get("exploratory"):
        sp = row.get("speedup")
        lines.append(f"METRIC speedup={sp if sp is not None else 0}")
    for key in ("soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min", "median_adapt_cost",
                "overhead_ratio", "b2_max"):
        if row.get(key) is not None:
            lines.append(f"METRIC {key}={row[key]}")
    for cls_name, cls_val in sorted((row.get("by_class") or {}).items()):
        lines.append(f"CLASS_METRIC {cls_name}={cls_val}")
    if row.get("error"):
        lines.append(f"Failed a check: {row['error']}")
    if row.get("decision") == "discard_duplicate":
        lines.append(DUPLICATE_HINT)

    kept = row.get("decision") in flywheel.KEPT_DECISIONS
    if kept:
        best, best_trial = row.get("best_speedup_after"), (
            row["trial_id"] if row.get("best_speedup_after") == row.get("speedup") else row.get("best_trial"))
    else:
        best, best_trial = row.get("best_speedup"), row.get("best_trial")
    best_txt = (f"Best so far in run {run_id}: {best}x the starting code ({best_trial})." if best is not None
                else f"Run {run_id} has no kept trial yet.")
    if row.get("exploratory"):
        lines.append(f"EXPLORATORY: dims={dims} repeats={row.get('repeats')} seed={row.get('seed')} is not the "
                     "standard configuration. This trial has no speed-up and was not kept. "
                     "Run without --dims, --dim, --repeats, --seed and --horizon-steps to grade it.")
    elif kept:
        lines.append(f"{best_txt} This trial was kept ({row['decision']}).")
    else:
        lines.append(f"{best_txt} This trial was not kept ({row['decision']}).")
    return lines


def rerun_for_verification(
    source_path: Path,
    lane: str,
    *,
    claimed_speedup: float | None,
    claimed_yardstick: str | None,
    tolerance: float = REPLICATION_TOLERANCE,
    work_dir: Path | None = None,
    use_queue: bool = True,
    autoresearch_dir: Path = ROOT / "autoresearch",
) -> dict:
    """Rerun a fetched trial through the grader on the standard public configuration.

    The rerun is the same computation as the trial it checks, so it should reproduce the posted
    speed-up to within `tolerance` (relative). Returns ``{comparable, passed, metrics, reason}``.
    `comparable` is False when the trial was graded against another yardstick (an older judge or
    starting code) or posted no speed-up. Such a rerun says nothing about the trial. Nothing is
    written to any ledger.
    """
    yd = load_yardstick(lane, autoresearch_dir=autoresearch_dir)
    metrics: dict = {"yardstick": yd["id"], "judge_digest": yd["judge_digest"], "config": grade.config(),
                     "claimed_speedup": claimed_speedup, "claimed_yardstick": claimed_yardstick,
                     "tolerance": tolerance}
    if claimed_yardstick != yd["id"] or not isinstance(claimed_speedup, (int, float)):
        why = ("it posted no speed-up" if claimed_yardstick is None or claimed_speedup is None
               else f"it was graded against yardstick {claimed_yardstick}, this checkout has {yd['id']}")
        return {"comparable": False, "passed": False, "metrics": metrics, "reason": f"not comparable: {why}"}
    work = work_dir or (Path(source_path).parent / "verify_work")
    work.mkdir(parents=True, exist_ok=True)
    with queue.eval_slot(f"verify:{lane}/{Path(source_path).name}", enabled=use_queue):
        score = grade.score_file(Path(source_path), lane, grade.config(), work)
    sp = grade.speedup(score, yd)
    metrics.update({
        "tier0_ok": bool(score.get("tier0_ok")), "tier1_ok": bool(score.get("valid")),
        "speedup": _sig(sp), "soft_min_eff": _sig(score.get("soft_min_eff")),
        "overhead_ratio": _sig(score.get("overhead_ratio")), "b2_max": _sig(score.get("b2_max")),
        "wall_s": score.get("wall_s"),
    })
    if sp is None:
        return {"comparable": True, "passed": False, "metrics": metrics,
                "reason": f"failed a check on rerun: {score.get('error') or 'no score'}"}
    rel = abs(sp / float(claimed_speedup) - 1.0)
    metrics["relative_difference"] = _sig(rel, 3)
    passed = rel <= tolerance
    reason = (f"reproduced {sp:.6g}x (posted {claimed_speedup:.6g}x)" if passed
              else f"mismatch: rerun {sp:.6g}x, posted {claimed_speedup:.6g}x ({rel:.2%} apart)")
    return {"comparable": True, "passed": passed, "metrics": metrics, "reason": reason}


def format_evaluator_json(
    row: dict,
    *,
    run_id: str = "default",
    autoresearch_dir: Path = ROOT / "autoresearch",
) -> dict:
    """Format a trial row as one `{valid, score, metrics, evidence}` evaluator object.

    `score` is the speed-up over the lane's starting code (0.0 when the trial failed a check or
    was exploratory). Missing or non-finite observations are left out of `metrics`.
    """
    is_dry = bool(row.get("dry_run"))
    valid = (row.get("decision") == "dry_run_ok") if is_dry else bool(row.get("tier0_ok") and row.get("tier1_ok"))
    sp = row.get("speedup")
    score_val = float(sp) if (valid and isinstance(sp, (int, float)) and math.isfinite(sp)) else 0.0

    metrics: dict[str, float | int] = {}
    for key in (
        "speedup",
        "soft_min_eff",
        "short_horizon_soft_min",
        "long_horizon_soft_min",
        "median_adapt_cost",
        "overhead_ratio",
        "b2_max",
        "best_speedup",
        "ratio_vs_best",
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
        "kept": row.get("decision") in flywheel.KEPT_DECISIONS,
        "exploratory": bool(row.get("exploratory")),
        "yardstick": row.get("yardstick"),
        "trial_id": trial_id,
        "lane": lane,
    }


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


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


def _local_yardstick_id(lane: str | None, autoresearch_dir: Path) -> str | None:
    if not lane:
        return None
    try:
        return json.loads((Path(autoresearch_dir) / "lanes" / lane / YARDSTICK_FILE).read_text()).get("id")
    except (OSError, ValueError):
        return None


def _node_by_class(node: dict) -> dict[str, float]:
    """Extract numeric `by_class` map from a Flywheel node's `metrics` or `metrics_json`."""
    metrics = node.get("metrics")
    if not isinstance(metrics, dict) and isinstance(node.get("metrics_json"), str):
        try:
            metrics = json.loads(node["metrics_json"])
        except ValueError:
            metrics = None
    by_cls = metrics.get("by_class") if isinstance(metrics, dict) else None
    if not isinstance(by_cls, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in by_cls.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }


def _weakest_classes(by_cls: dict[str, float], limit: int = 3) -> str:
    ranked = sorted(
        ((k, float(v)) for k, v in by_cls.items() if isinstance(v, (int, float)) and math.isfinite(v)),
        key=lambda kv: kv[1],
    )
    return ", ".join(f"{k}={v:.4g}" for k, v in ranked[:limit])


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

    Combines live public Flywheel frontier/kept-trials/claims/dead-ends (read-only, no token needed)
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

            local_digest = judge.judge_digest()
            local_yd = _local_yardstick_id(None if all_lanes else lane, autoresearch_dir)
            lines.append(f"## Live Flywheel Frontier (`{flywheel_url}` | local judge_digest=`{local_digest}`)")
            lane_frontier: dict | None = None
            if frontier:
                for row in frontier:
                    rlane = row.get("lane") or "?"
                    if rlane == lane:
                        lane_frontier = row
                    sp = row.get("speedup")
                    sp_s = f"{float(sp):.4g}x" if isinstance(sp, (int, float)) else "n/a"
                    ver = "verified" if row.get("verified") else "unverified"
                    ryd = row.get("yardstick")
                    mine = local_yd if not all_lanes else _local_yardstick_id(rlane, autoresearch_dir)
                    compat = " [other yardstick: not comparable with your trials]" if ryd and mine and ryd != mine else ""
                    title = str(row.get("title") or row.get("hypothesis") or row.get("id") or "")[:80]
                    tag = " <-- THIS LANE" if rlane == lane else ""
                    lines.append(
                        f"- **{rlane}** ({ver}, speedup={sp_s}, yardstick={ryd or '?'}): "
                        f"`{row.get('id', '?')}` — {title}{compat}{tag}"
                    )
            else:
                lines.append("- (no frontier nodes yet)")
            lines.append("")

            if not all_lanes and lane not in ("S2", "S3"):
                lines.append(f"## Starting Point on {lane} (build on the live frontier)")
                ryd = lane_frontier.get("yardstick") if lane_frontier else None
                compatible = bool(lane_frontier and (not ryd or not local_yd or ryd == local_yd))
                if compatible and lane_frontier and lane_frontier.get("id"):
                    fid = str(lane_frontier["id"])
                    sp = lane_frontier.get("speedup")
                    sp_s = f"{float(sp):.4g}x" if isinstance(sp, (int, float)) else "n/a"
                    who = lane_frontier.get("author") or lane_frontier.get("agent_model") or "unknown"
                    hyp = _clip(str(lane_frontier.get("hypothesis") or lane_frontier.get("title") or ""), 240)
                    lines.append(f"- Current compatible frontier: `{fid}` ({sp_s}) by `{who}` — {hyp}")
                    weak = _weakest_classes(_node_by_class(lane_frontier))
                    if weak:
                        lines.append(f"- Frontier weakest target classes (attack these first): {weak}")
                    local_sub = lane_dir / "submission.py" if lane_dir is not None else None
                    local_sha = _submission_stats(local_sub)[0] if (local_sub and local_sub.is_file()) else None
                    f_sha = lane_frontier.get("code_sha")
                    if f_sha and local_sha and f_sha == local_sha:
                        lines.append(
                            f"- Local `submission.py` already matches `{fid}` (sha `{local_sha}`). "
                            f"Pass `--fork-of {fid}` on your first edit if starting a new run."
                        )
                    else:
                        lines.append(
                            f"- **Pick up from `{fid}` instead of starting from scratch** (unless testing a "
                            f"deliberately clean-slate architecture):"
                        )
                        lines.append(
                            f"  - `uv run apin autoresearch fetch --node {fid} "
                            f"--out autoresearch/lanes/{lane}/submission.py --force`"
                        )
                        lines.append(
                            f"  - Then pass `--fork-of {fid}` on your first `eval` in this run so lineage is recorded."
                        )
                else:
                    lines.append(
                        f"- No compatible live frontier posted for {lane} yet; start from "
                        f"`autoresearch/lanes/{lane}/submission.py`."
                    )
                lines.append("")

            kept_nodes = [
                n for n in nodes
                if (n.get("decision") in flywheel.KEPT_DECISIONS or n.get("decision") == "archive_only")
                and (not local_yd or not n.get("yardstick") or n.get("yardstick") == local_yd)
            ]
            kept_nodes.sort(
                key=lambda n: (
                    float(n["speedup"]) if isinstance(n.get("speedup"), (int, float)) else -1.0,
                    str(n.get("created_at") or ""),
                ),
                reverse=True,
            )
            lines.append(f"## Top Recent Kept Trials on {lane} (build on or combine these)")
            if kept_nodes:
                for n in kept_nodes[:6]:
                    sp = n.get("speedup")
                    sp_s = f"{float(sp):.4g}x" if isinstance(sp, (int, float)) else "n/a"
                    who = n.get("author") or n.get("agent_model") or "?"
                    src_tag = " [source]" if n.get("has_source") else ""
                    hyp = _clip(str(n.get("hypothesis") or n.get("title") or ""), 240)
                    lines.append(f"- `{n.get('id', '?')}` [{n.get('decision')}, {sp_s}, by `{who}`]{src_tag}: {hyp}")
            else:
                lines.append("- (none recorded on Flywheel)")
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
                    hyp = _clip(str(n.get("hypothesis") or n.get("title") or ""), 300)
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
            neg_text = neg_path.read_text().strip()
            if neg_text:
                lines.append(f"## Local Negative Memory (`{neg_path.name}`)")
                lines.extend(neg_text.splitlines())
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
            graded = [r for r in kept if isinstance(r.get("speedup"), (int, float))]
            best = max(graded, key=lambda r: r["speedup"]) if graded else (kept[-1] if kept else ledger[-1])
            lines.append(
                f"- Best kept: `{best.get('trial_id')}` decision={best.get('decision')} "
                f"speedup={best.get('speedup')}x soft_min={best.get('soft_min_eff')} "
                f"short={best.get('short_horizon_soft_min')} long={best.get('long_horizon_soft_min')} "
                f"adapt_cost={best.get('median_adapt_cost')} overhead={best.get('overhead_ratio')}x"
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

    if lane == "S2":
        lines += [
            "## Action Checklist (Lane S2 — Systems)",
            "- Read `autoresearch/lanes/S2/program.md` and `upstream/STATUS.md` before editing.",
            "- Initialize the target upstream submodule first if empty (e.g. `git submodule update --init upstream/math`).",
            "- Score each clean-rebuilt change with `uv run apin autoresearch eval-s2 --upstream <name> --allowed-path <prefix> ...` (use `--dry-run` first).",
            "- Share reusable findings and hypotheses on Flywheel via `uv run apin autoresearch note add --kind insight --lane S2 --title \"...\" --body \"...\"`.",
        ]
    elif lane == "S3":
        lines += [
            "## Action Checklist (Lane S3 — Rewind Calibration)",
            "- Read `autoresearch/lanes/S3/program.md` and `autoresearch/lanes/S3/tasks.json`.",
            "- Verify calibration with `uv run apin autoresearch eval-s3 --mode base` and `--mode reference`.",
            "- Grade custom fixes with `uv run apin autoresearch eval-s3 --mode custom --submission <task_id>=<path/to/fix.py>`.",
            "- Share findings, new rewind bug candidates, or AUARC results on Flywheel via `uv run apin autoresearch note add --kind insight --lane S3 --title \"...\" --body \"...\"`.",
        ]
    else:
        lines += [
            "## Action Checklist",
            "- Start from the live Flywheel frontier when one exists (`fetch --node <id>` + `--fork-of <id>`), and prefer **composing** compatible verified ideas across lanes (`--combines <id>`) before inventing complex heuristics.",
            "- Score each edit with `uv run apin autoresearch eval --lane " + lane + " --run-id <RUN_ID> --hypothesis \"...\"`. "
            "It prints the speed-up over the starting code. A difference under 5% is noise.",
            "- Record reusable learnings via `uv run apin autoresearch note add --kind insight --lane " + lane + " --title \"...\" --body \"...\"`.",
            "- Freeze `submission.py` before running full real-Stan verification (`apin run logdensity` + `apin score --submission`).",
        ]
    return "\n".join(lines) + "\n"
