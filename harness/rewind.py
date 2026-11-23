"""Lane S3 — Rewind Calibration & Agent Grading (AUARC).

Implements lane S3 (`autoresearch/lanes/S3/program.md`):
  - Evaluates candidate agent patches against pre-registered rewind tasks where the
    base snapshot fails a hidden invariance/efficiency/correctness check and the
    reference fix passes.
  - Computes AUARC (Area Under Accuracy–Resource Curve) across a graded task suite
    so different agent models, prompts, and harness configurations can be benchmarked
    quantitatively.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import invariance, shapes
from .models import ROOT

DEFAULT_TASKS_PATH = ROOT / "autoresearch" / "lanes" / "S3" / "tasks.json"


@dataclass(frozen=True)
class RewindTask:
    task_id: str
    title: str
    kind: str  # "invariance_fix" | "efficiency_fix" | "upstream_rewind"
    lane: str | None
    base_submission: str
    reference_submission: str | None
    target_family: str
    dim: int
    rotated: bool
    seed: int
    n_chains: int = 48
    k_steps: int = 16
    adapt_draws: int = 100
    min_efficiency: float | None = None
    target_kwargs: dict[str, Any] | None = None
    resource_budget: float = 1.0


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> list[RewindTask]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[RewindTask] = []
    for t in data.get("tasks", []):
        out.append(
            RewindTask(
                task_id=str(t["id"]),
                title=str(t["title"]),
                kind=str(t["kind"]),
                lane=t.get("lane"),
                base_submission=str(t["base_submission"]),
                reference_submission=t.get("reference_submission"),
                target_family=str(t["target_family"]),
                dim=int(t["dim"]),
                rotated=bool(t.get("rotated", True)),
                seed=int(t.get("seed", 20261121)),
                n_chains=int(t.get("n_chains", 48)),
                k_steps=int(t.get("k_steps", 16)),
                adapt_draws=int(t.get("adapt_draws", 100)),
                min_efficiency=float(t["min_efficiency"]) if t.get("min_efficiency") is not None else None,
                target_kwargs=dict(t.get("target_kwargs") or {}),
                resource_budget=float(t.get("resource_budget", 1.0)),
            )
        )
    return out


def _make_target(task: RewindTask) -> shapes.AnalyticTarget:
    kw = dict(task.target_kwargs or {})
    fam = task.target_family
    if fam == "gaussian_illcond":
        return shapes.IllCondGaussian(task.dim, task.seed, rotated=task.rotated, **kw)
    if fam == "funnel":
        return shapes.NealsFunnel(task.dim, task.seed, rotated=task.rotated, **kw)
    if fam == "banana":
        return shapes.Banana(task.dim, task.seed, rotated=task.rotated, **kw)
    if fam == "hybrid_rosenbrock":
        return shapes.HybridRosenbrock(task.dim, task.seed, rotated=task.rotated, **kw)
    if fam == "student_t":
        return shapes.StudentT(task.dim, task.seed, rotated=task.rotated, **kw)
    if fam == "mixture_crossable":
        return shapes.GaussianMixture(task.dim, task.seed, rotated=task.rotated, **kw)
    raise ValueError(f"unknown S3 target_family: {fam!r}")


def evaluate_rewind_task(
    task: RewindTask,
    submission_path: Path,
    *,
    resource_used: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one submission file against a RewindTask's hidden invariance + efficiency check."""
    t0 = time.perf_counter()
    sub = Path(submission_path)
    if not sub.is_absolute():
        sub = (ROOT / sub).resolve()
    tgt = _make_target(task)
    try:
        res = invariance.check_target_invariance(
            tgt, sub, seed=task.seed + 17, n_chains=task.n_chains,
            k_steps=task.k_steps, adapt_draws=task.adapt_draws,
            lane=task.lane, isolated=True,
        )
        eff = res.dual_horizon_eff if res.dual_horizon_eff > 0 else res.empirical_efficiency
        eff_ok = True if task.min_efficiency is None else (eff >= task.min_efficiency)
        passed = bool(res.passed and eff_ok)
        reason = res.reason if not res.passed else (
            "passed" if eff_ok else f"efficiency {eff:.3e} < min_efficiency {task.min_efficiency:.3e}"
        )
        return {
            "task_id": task.task_id,
            "title": task.title,
            "kind": task.kind,
            "submission": str(sub),
            "passed": passed,
            "invariance_passed": bool(res.passed),
            "emp_eff": round(float(eff), 8),
            "min_efficiency": task.min_efficiency,
            "max_z": round(float(res.max_z), 3),
            "z_threshold": round(float(res.z_threshold), 3),
            "b2_max": round(float(res.b2_max), 6),
            "resource_used": float(resource_used),
            "reason": reason,
            "wall_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        return {
            "task_id": task.task_id,
            "title": task.title,
            "kind": task.kind,
            "submission": str(sub),
            "passed": False,
            "invariance_passed": False,
            "emp_eff": 0.0,
            "min_efficiency": task.min_efficiency,
            "resource_used": float(resource_used),
            "reason": f"{type(e).__name__}: {e}",
            "wall_s": round(time.perf_counter() - t0, 3),
        }


def compute_auarc(results: list[dict[str, Any]], max_resource: float | None = None) -> float:
    """Compute Area Under Accuracy–Resource Curve (AUARC) in [0, 1].

    Each entry needs `passed: bool` and `resource_used: float >= 0`.
    Tasks solved earlier (lower resource) contribute proportionally more area under the
    step-function cumulative solve-rate curve over [0, R_max].
    """
    n = len(results)
    if n == 0:
        return 0.0
    r_max = float(max_resource) if max_resource is not None else max(
        max((float(r.get("resource_used", 0.0)) for r in results), default=1.0), 1.0
    )
    if r_max <= 0.0:
        return float(sum(1 for r in results if r.get("passed")) / n)
    solved_times = sorted(
        min(max(float(r.get("resource_used", 0.0)), 0.0), r_max)
        for r in results if r.get("passed")
    )
    # Exact integral of step function (1/n) * sum_i 1[t >= t_i] over [0, r_max], divided by r_max:
    # = (1/n) * sum_i (r_max - t_i) / r_max
    area = sum((r_max - t) / r_max for t in solved_times) / float(n)
    return round(float(area), 6)


def evaluate_s3_suite(
    submissions_by_task: dict[str, Path] | None = None,
    *,
    mode: str = "reference",
    tasks_path: Path = DEFAULT_TASKS_PATH,
    resources_by_task: dict[str, float] | None = None,
    max_resource: float = 1.0,
    run_id: str = "calibration",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full Lane S3 rewind suite (`mode='base'`, `'reference'`, or explicit `submissions_by_task`)."""
    tasks = load_tasks(tasks_path)
    results = []
    for task in tasks:
        if submissions_by_task and task.task_id in submissions_by_task:
            sub = Path(submissions_by_task[task.task_id])
        elif mode == "base":
            sub = Path(task.base_submission)
        elif mode == "reference" and task.reference_submission:
            sub = Path(task.reference_submission)
        else:
            continue
        r_used = float((resources_by_task or {}).get(task.task_id, 0.0))
        results.append(evaluate_rewind_task(task, sub, resource_used=r_used))

    n_passed = sum(1 for r in results if r["passed"])
    auarc = compute_auarc(results, max_resource=max_resource)
    report = {
        "lane": "S3",
        "run_id": run_id,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": n_passed,
        "total": len(results),
        "pass_rate": round(n_passed / len(results), 4) if results else 0.0,
        "auarc": auarc,
        "results": results,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"s3_{mode}.json").write_text(json.dumps(report, indent=2))
    return report
