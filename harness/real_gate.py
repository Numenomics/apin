"""The real-posterior gate: step 8 of auto-research.md as one call.

    load() -> dict                      autoresearch/real_gate.json, checked
    run(submission, lane) -> dict       run and score a sampler file on the gate's models
    summarize(gate, rows, ...) -> dict  the per-model verdicts and counts, from `apin score` rows
    flywheel_fields(result) -> dict     what a verifier posts to the Flywheel, or None

A trial counts as its lane's frontier only when an independent verifier ran this gate on its
source and it passed the 0.1-SD bench gate on every required model, under the current gate `id`
(the Flywheel checks the id: REAL_GATE_ID in flywheel/src/index.ts). The required models are those
the Stan NUTS reference passes at this budget. The reported models are run and shown, but not
required. To change either list, re-measure the reference at the step 8 budget and bump `id`
(the `reference` note in the JSON file says how).

`run` is `apin run logdensity` followed by `apin score --submission`: the same adapter, grid,
budgets, seeds, queue and gate statistics (harness/grid.py, harness/score.py, harness/gate.py).

Claim wording: a model "passed the 0.1-SD bench gate at this budget", never "is correct".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from . import grid, models as models_mod, queue, sampler, score
from .models import ROOT

GATE_PATH = ROOT / "autoresearch" / "real_gate.json"
DEFAULT_WORK = ROOT / "runs" / "real_gate"
# Lanes whose transition kernel the judge supplies (judge.resolve_lane_submission). Their files
# hold only the adapt code, so the adapter must be told the lane (`--opt lane=` in step 8).
JUDGE_KERNEL_LANES = ("A1", "A4")
_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def load(path: Path = GATE_PATH) -> dict:
    """The gate definition: `id`, `required` and `reported` models, `reps` and the `reference` note."""
    raw = json.loads(Path(path).read_text())
    gid, reps = raw.get("id"), raw.get("reps")
    required, reported = list(raw.get("required") or []), list(raw.get("reported") or [])
    names = required + reported
    if not isinstance(gid, str) or not _ID.fullmatch(gid):
        raise ValueError(f"{path}: `id` must be 1-64 letters, digits, `.`, `_` or `-`, got {gid!r}")
    if not required or len(set(names)) != len(names) or not all(isinstance(m, str) and m for m in names):
        raise ValueError(f"{path}: `required` must be non-empty and no model may be listed twice")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        raise ValueError(f"{path}: `reps` must be a positive integer, got {reps!r}")
    return {"id": gid, "required": required, "reported": reported, "reps": reps,
            "reference": str(raw.get("reference") or "")}


def valid_label(label: str) -> bool:
    """A label is a directory name: 1-64 letters, digits, `.`, `_` or `-`, not `.` or `..`."""
    return bool(_ID.fullmatch(label)) and label.strip(".") != ""


def default_label(submission: Path, lane: str | None) -> str:
    """A label named after the file's content, so a rerun of the same file resumes its reps."""
    sha = hashlib.sha256(Path(submission).read_bytes()).hexdigest()[:12]
    return f"gate-{lane}-{sha}" if lane in JUDGE_KERNEL_LANES else f"gate-{sha}"


def run(
    submission: Path,
    lane: str | None = None,
    *,
    models: list[str] | None = None,
    reps: int | None = None,
    label: str | None = None,
    work_dir: Path | None = None,
    use_queue: bool = True,
    gate: dict | None = None,
    log=print,
) -> dict:
    """Run step 8 for one sampler file and return `summarize`'s result.

    `models` runs a subset (or other models) for a quick check: the result is then partial and
    never `ok`. So is a run with other `reps` than the gate's. Runs and scores go under
    `work_dir` (default runs/real_gate/<label>/), and a finished rep is reused when its file,
    seed and budget match (harness/grid.py).
    """
    gate = gate or load()
    selected = list(dict.fromkeys(models)) if models else gate["required"] + gate["reported"]
    reps = gate["reps"] if reps is None else int(reps)
    submission = Path(submission).expanduser().resolve()
    label = label or default_label(submission, lane)
    if not valid_label(label):
        raise ValueError(f"label {label!r} is not a plain directory name")
    work =Path(work_dir) if work_dir else DEFAULT_WORK / label
    options = {"sampler": str(submission)}
    if lane in JUDGE_KERNEL_LANES:
        options["lane"] = str(lane)
    spec = ",".join(selected)
    s_obj = sampler.get("logdensity", sampler.Settings(build_dir=ROOT / "build", options=options))
    with queue.eval_slot(f"gate:{label}:{spec}", enabled=use_queue):
        grid.run_grid(s_obj, models_mod.select(spec), reps, work / "runs", label, log=log)
    g = score.declared(spec, reps, grid.BASE_SEED, ROOT / "build")
    scored = score.score(work / "runs", work / "scores", g, labels=[label], submissions={label: submission},
                         log=lambda *_: None)
    result = summarize(gate, scored["per_model"], selected=selected, reps=reps)
    result.update(lane=lane, submission=str(submission), label=label, work_dir=str(work),
                  submission_sha256=hashlib.sha256(submission.read_bytes()).hexdigest())
    return result


def _finite(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None


def summarize(gate: dict, rows: list[dict], *, selected: list[str] | None = None, reps: int | None = None) -> dict:
    """Count the verdicts of `apin score`'s per-model rows (score.per_model) against the gate.

    A model passed when its status is `right`. A selected model without a row has no reference in
    this checkout and did not pass. `required_total` is always the gate's full required count, so a
    subset can never report n of n. `ok` (every required model passed, on the full gate) is what
    a frontier needs.
    """
    full = gate["required"] + gate["reported"]
    selected = list(selected or full)
    reps = gate["reps"] if reps is None else reps
    by_model = {r["model"]: r for r in rows}
    per_model = []
    for m in selected:
        r = by_model.get(m)
        status = r["status"] if r else "missing"
        passed = status == "right"
        per_model.append({
            "model": m,
            "required": m in gate["required"],
            "status": status,
            "passed": passed,
            "reason": (r.get("reason") or "") if r else "not scored: no reference posterior here (apin reference build)",
            "ess_per_cost": _finite(r.get("ess_per_cost")) if r and passed else None,
        })
    required = [p for p in per_model if p["required"]]
    reported = [p for p in per_model if p["model"] in gate["reported"]]
    partial = set(selected) != set(full) or reps != gate["reps"]
    required_passed = sum(p["passed"] for p in required)
    ok = not partial and required_passed == len(gate["required"])
    costs = [p["ess_per_cost"] for p in required]
    ess = (math.exp(sum(math.log(c) for c in costs) / len(costs))
           if ok and costs and all(c is not None for c in costs) else None)
    return {
        "gate_id": gate["id"],
        "models": per_model,
        "required_passed": required_passed,
        "required_total": len(gate["required"]),
        "required_run": len(required),
        "reported_passed": sum(p["passed"] for p in reported),
        "reported_total": len(gate["reported"]),
        "reps": reps,
        "partial": partial,
        "ok": ok,
        # Geometric mean over the required models, as the bench ranks: only when all passed.
        "real_ess_per_cost": ess,
    }


def summary_line(result: dict) -> str:
    """The one sentence to report. It never says correct, exact, unbiased or proven."""
    line = (f"{result['required_passed']} of {result['required_total']} required models passed "
            f"the 0.1-SD bench gate at this budget")
    if result["partial"]:
        line += (f" (partial run: {result['required_run']} of {result['required_total']} required models, "
                 f"{result['reps']} reps; it cannot qualify a frontier)")
    return line + f" [gate {result['gate_id']}]"


def table(result: dict) -> str:
    """A plain per-model table: model, role, verdict, ESS per cost and the gate's reason."""
    verdicts = {"right": "passed", "wrong": "failed", "unproven": "unproven", "missing": "not scored"}
    lines = [f"{'model':28s} {'role':9s} {'verdict':11s} {'ess/cost':>10s}  reason"]
    for p in result["models"]:
        ess = f"{p['ess_per_cost']:.3g}" if p["ess_per_cost"] is not None else "-"
        role = "required" if p["required"] else "reported"
        lines.append(f"{p['model']:28s} {role:9s} {verdicts.get(p['status'], p['status']):11s} {ess:>10s}  "
                     f"{' '.join(str(p['reason']).split())[:160]}")
    return "\n".join(lines)


def flywheel_fields(result: dict) -> dict | None:
    """The fields a verifier posts with its result, or None for a partial run (never posted)."""
    if result.get("partial"):
        return None
    fields = {"real_stan_passed": int(result["required_passed"]), "real_stan_total": int(result["required_total"]),
              "real_gate": result["gate_id"]}
    if result.get("real_ess_per_cost") is not None:
        fields["real_ess_per_cost"] = float(result["real_ess_per_cost"])
    return fields


def metrics_summary(result: dict) -> dict:
    """A compact record for the verification result's metrics: the gate id and each model's status."""
    return {
        "id": result["gate_id"],
        "ok": bool(result["ok"]),
        "required_passed": result["required_passed"],
        "required_total": result["required_total"],
        "models": {p["model"]: p["status"] for p in result["models"]},
        "submission_sha256": result.get("submission_sha256"),
    }
