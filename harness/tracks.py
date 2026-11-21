"""Benchmark tracks L2 (wall-clock + machine fingerprint), L3 (Haar rotation & extreme scale stress),
L4 (unadjusted kernels: measured bias & cost to b^2_max < 0.01, never pooled with L1),
and L5 (density-only view of L1, strictly enforcing n_gradient == 0).

Implements ARCHITECTURE.md §1.5 & LEADERBOARD.md §2.3:
  - No cross-tier pooling: L1, L2, L3, L4, and L5 publish separate reports/tables.
  - Claim-wording hygiene: passing a check means "passed the 0.1-SD bench gate / invariance check
    at this budget" — never "is correct", "exact", "unbiased", or "proven".
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import invariance, ipc, judge, logdensity, shapes
from .logdensity import Budget
from .models import ROOT


def machine_fingerprint() -> dict[str, Any]:
    """Capture host hardware and toolchain fingerprint for Track L2 (wall-clock) reproducibility."""
    cpu_model = platform.processor() or ""
    if not cpu_model and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": cpu_model or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }


def _check_unadjusted_target(
    target: shapes.AnalyticTarget,
    submission_path: Path,
    *,
    seed: int,
    n_chains: int = 48,
    checkpoints: tuple[int, ...] = (5, 10, 20, 40),
    adapt_draws: int = 100,
    lane: str | None = None,
    bias_threshold: float = 0.01,
) -> dict[str, Any]:
    """Run an unadjusted kernel on one analytic target through isolated IPC workers.

    Records invariance z-score and b^2_max as *measured bias* (never hard-fails on bias),
    and measures cost units required to reach b^2_max < bias_threshold from dispersed starts.
    """
    max_k = max(checkpoints)
    work_dir = ROOT / ".cache" / "track_l4" / f"{target.family}_{target.dim}_{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run adapt once in isolated worker
    adapt_budget = Budget(1, adapt_draws, 200_000, 60.0)
    adapt_meta = ipc.run_adapt_phase_isolated(
        target, submission_path, adapt_budget, seed + 1, work_dir, lane=lane,
    )
    adapt_cost = float(adapt_meta["density_evals"] + 2.0 * adapt_meta["gradient_evals"])

    # 2. Check exact-draw invariance & record measured bias (do not reject on invariance failure)
    inv_res = invariance.check_target_invariance(
        target, submission_path, seed=seed + 10, n_chains=n_chains, k_steps=min(20, max_k),
        adapt_draws=adapt_draws, lane=lane, isolated=True,
    )

    # 3. Run dispersed-start chains up to max_k and compute b^2_max at each checkpoint
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    x_starts = target.dispersed_starts(n_chains, rng)
    truth = target.exact_truth()
    walkers = int(adapt_meta["walkers"])
    is_1d = bool(adapt_meta["is_1d"])
    aux_spec = adapt_meta["aux_spec"]
    tuning = logdensity.deserialize_tree(work_dir / "tuning")

    counter = logdensity.Counter(target, max_evaluations=2_000_000, max_seconds=120.0)
    worker = ipc.IsolatedTransitionSession(
        counter, submission_path, tuning, adapt_meta["tuning_sha256"],
        walkers, is_1d, aux_spec, lane=lane,
    )
    checkpoint_set = set(checkpoints)
    checkpoint_stats: list[dict[str, Any]] = []
    cost_to_thresh: float | None = None
    try:
        states = []
        for c in range(n_chains):
            pos = x_starts[c : c + 1] if walkers == 1 else np.repeat(x_starts[c : c + 1], walkers, axis=0)
            aux = invariance.sample_aux(aux_spec, walkers, target.dim, is_1d, rng)
            states.append([pos, aux])
        for step in range(1, max_k + 1):
            for c in range(n_chains):
                pos_next, aux_next = worker.step(states[c][0], states[c][1], [seed, 9001, c, step], check_provenance=True)
                states[c][0] = pos_next
                states[c][1] = aux_next
            if step in checkpoint_set:
                xs = np.array([s[0][0] for s in states], dtype=float)
                z = target.to_latent(xs)
                funcs = target.latent_functionals(z)
                b2_max = 0.0
                for fname, (mu, sd) in truth.items():
                    if sd <= 0:
                        continue
                    err = float(np.mean(funcs[fname]) - mu) / sd
                    b2_val = (err * err) if (math.isfinite(err) and abs(err) < 1e150) else math.inf
                    b2_max = max(b2_max, b2_val)
                step_cost_per_chain = float(counter.density + 2.0 * counter.gradient) / float(n_chains)
                total_cost = adapt_cost + step_cost_per_chain
                checkpoint_stats.append({
                    "step": step,
                    "b2_max": round(b2_max, 6),
                    "cost_per_chain": round(total_cost, 2),
                })
                if cost_to_thresh is None and b2_max < bias_threshold:
                    cost_to_thresh = total_cost
    finally:
        worker.close()

    return {
        "family": target.family,
        "dim": target.dim,
        "rotated": target.rotated,
        "invariance_max_z": round(float(inv_res.max_z), 3),
        "invariance_z_threshold": round(float(inv_res.z_threshold), 3),
        "invariance_rejected": not bool(inv_res.passed),
        "measured_b2_max": round(float(inv_res.b2_max), 6),
        "empirical_efficiency": round(float(inv_res.empirical_efficiency), 8),
        "adapt_cost": round(adapt_cost, 1),
        "cost_to_b2_threshold": round(cost_to_thresh, 2) if cost_to_thresh is not None else None,
        "reached_threshold": cost_to_thresh is not None,
        "checkpoints": checkpoint_stats,
        "overhead_ratio": round(float(inv_res.overhead_ratio), 3),
    }


def run_track_l3(
    submission_path: Path,
    *,
    dims: tuple[int, ...] = (6, 16),
    scales: tuple[float, ...] = (1e-2, 1.0, 1e2),
    seed: int = 20261121,
    n_chains: int = 48,
    k_steps: int = 16,
    adapt_draws: int = 100,
    lane: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Track L3 — Full Haar rotation & extreme scale-stress sweep (1e-2 .. 1e2)."""
    sub_path = Path(submission_path).resolve()
    t0 = time.perf_counter()
    by_scale: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    all_passed = True
    worst_b2 = 0.0

    for s_idx, scale in enumerate(scales):
        s_key = f"{scale:g}"
        by_scale[s_key] = []
        for d_idx, dim in enumerate(dims):
            base = seed + 1000 * s_idx + 100 * d_idx
            targets = [
                shapes.IllCondGaussian(dim, base + 1, rotated=True, scale_stress=scale, kappa=100.0),
                shapes.NealsFunnel(dim, base + 2, rotated=True, scale_stress=scale, v_sd=2.0),
                shapes.Banana(dim, base + 3, rotated=True, scale_stress=scale),
                shapes.HybridRosenbrock(dim, base + 4, rotated=True, scale_stress=scale, b=5.0),
                shapes.StudentT(dim, base + 5, rotated=True, scale_stress=scale, nu=3.0),
            ]
            for t_idx, tgt in enumerate(targets):
                res = invariance.check_target_invariance(
                    tgt, sub_path, seed=base + 10 + t_idx, n_chains=n_chains,
                    k_steps=k_steps, adapt_draws=adapt_draws, lane=lane, isolated=True,
                )
                eff = res.dual_horizon_eff if res.dual_horizon_eff > 0 else res.empirical_efficiency
                if not res.passed:
                    all_passed = False
                worst_b2 = max(worst_b2, float(res.b2_max))
                by_scale[s_key].append(eff)
                rows.append({
                    "scale_stress": scale,
                    "dim": dim,
                    "family": tgt.family,
                    "rotated": True,
                    "passed": res.passed,
                    "emp_eff": round(float(eff), 8),
                    "b2_max": round(float(res.b2_max), 6),
                    "max_z": round(float(res.max_z), 3),
                    "z_threshold": round(float(res.z_threshold), 3),
                    "adapt_cost": round(float(res.adapt_cost), 1),
                    "overhead_ratio": round(float(res.overhead_ratio), 3),
                    "reason": res.reason,
                })

    scale_soft_min = {
        k: invariance.soft_min_power_mean(vs, p=-4.0) if vs else 0.0
        for k, vs in by_scale.items()
    }
    base_eff = scale_soft_min.get("1", 0.0)
    extreme_vals = [v for k, v in scale_soft_min.items() if k != "1"]
    scale_inv_ratio = (min(extreme_vals) / base_eff) if (base_eff > 0 and extreme_vals) else 0.0
    overall_soft_min = (
        invariance.soft_min_power_mean(list(scale_soft_min.values()), p=-4.0) if all_passed else 0.0
    )

    report = {
        "track": "L3",
        "description": "Full Haar rotation & extreme scale-stress robustness (1e-2 to 1e2)",
        "submission": str(sub_path),
        "lane": lane,
        "dims": list(dims),
        "scales": list(scales),
        "passed_all_invariance": all_passed,
        "soft_min_eff": round(overall_soft_min, 8),
        "scale_soft_min": {k: round(v, 8) for k, v in scale_soft_min.items()},
        "scale_invariance_ratio": round(scale_inv_ratio, 4),
        "worst_b2_max": round(worst_b2, 6),
        "wall_s": round(time.perf_counter() - t0, 2),
        "machine": machine_fingerprint(),
        "targets": rows,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "track_L3.json").write_text(json.dumps(report, indent=2))
        (out_dir / "track_L3.md").write_text(format_track_markdown(report))
    return report


def run_track_l4(
    submission_path: Path,
    *,
    dims: tuple[int, ...] = (6,),
    seed: int = 20261121,
    n_chains: int = 48,
    checkpoints: tuple[int, ...] = (5, 10, 20, 40),
    adapt_draws: int = 100,
    bias_threshold: float = 0.01,
    lane: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Track L4 — Unadjusted kernels (ULA, unadjusted MCLMC, Foster ULD).

    Never pooled with Track L1. Records invariance rejection as measured bias and measures
    cost units to reach b^2_max < bias_threshold.
    """
    sub_path = Path(submission_path).resolve()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for d_idx, dim in enumerate(dims):
        base = seed + 200 * d_idx
        targets = [
            shapes.IllCondGaussian(dim, base + 1, rotated=True, kappa=50.0),
            shapes.NealsFunnel(dim, base + 2, rotated=True, v_sd=2.0),
            shapes.Banana(dim, base + 3, rotated=True),
            shapes.HybridRosenbrock(dim, base + 4, rotated=False, b=5.0),
            shapes.StudentT(dim, base + 5, rotated=True, nu=3.0),
        ]
        for t_idx, tgt in enumerate(targets):
            rows.append(
                _check_unadjusted_target(
                    tgt, sub_path, seed=base + 10 + t_idx, n_chains=n_chains,
                    checkpoints=checkpoints, adapt_draws=adapt_draws, lane=lane,
                    bias_threshold=bias_threshold,
                )
            )

    reached = [r["cost_to_b2_threshold"] for r in rows if r["cost_to_b2_threshold"] is not None]
    median_cost_to_b2 = float(np.median(reached)) if reached else None
    worst_b2 = max((r["measured_b2_max"] for r in rows), default=0.0)
    report = {
        "track": "L4",
        "pool_policy": "UNADJUSTED_ONLY — never ranked or pooled against Track L1/L2/L3/L5",
        "submission": str(sub_path),
        "lane": lane,
        "dims": list(dims),
        "bias_threshold": bias_threshold,
        "targets_reaching_threshold": len(reached),
        "targets_total": len(rows),
        "median_cost_to_b2_threshold": round(median_cost_to_b2, 2) if median_cost_to_b2 is not None else None,
        "worst_measured_b2_max": round(worst_b2, 6),
        "wall_s": round(time.perf_counter() - t0, 2),
        "machine": machine_fingerprint(),
        "targets": rows,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "track_L4.json").write_text(json.dumps(report, indent=2))
        (out_dir / "track_L4.md").write_text(format_track_markdown(report))
    return report


def run_track_l5(
    submission_path: Path,
    *,
    dims: tuple[int, ...] = (6, 12),
    seed: int = 20261121,
    n_chains: int = 48,
    k_steps: int = 20,
    adapt_draws: int = 120,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Track L5 — Density-only view of L1, strictly enforcing n_gradient == 0."""
    sub_path = Path(submission_path).resolve()
    t0 = time.perf_counter()

    # Enforce density-only capability via isolated ABI check first (capabilities=set())
    g_check = shapes.IllCondGaussian(dim=4, seed=seed + 3, rotated=True, kappa=25.0)
    work_dir = ROOT / ".cache" / "track_l5" / f"capcheck_{seed}"
    try:
        _, abi_sum = ipc.execute_run_isolated(
            g_check, sub_path, Budget(1, 20, 100_000, 30.0),
            run_seed=seed + 5, work_dir=work_dir, capabilities={"density_only"}, replay_count=5,
        )
    except Exception as e:
        report = {
            "track": "L5",
            "description": "Density-only view of L1 (n_gradient == 0 strictly enforced)",
            "submission": str(sub_path),
            "passed_density_only": False,
            "error": f"{type(e).__name__}: {e}",
            "soft_min_eff": 0.0,
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "track_L5.json").write_text(json.dumps(report, indent=2))
            (out_dir / "track_L5.md").write_text(format_track_markdown(report))
        return report

    if int(abi_sum.get("gradient_evals", 0)) > 0:
        raise logdensity.CapabilityError(
            f"Track L5 requires n_gradient == 0, got {abi_sum['gradient_evals']} gradient evaluations"
        )

    rows: list[dict[str, Any]] = []
    by_class: dict[str, list[float]] = {}
    all_passed = True
    worst_b2 = 0.0
    total_grads = 0

    for d_idx, dim in enumerate(dims):
        base = seed + 300 * d_idx
        targets = [
            shapes.IllCondGaussian(dim, base + 1, rotated=False, kappa=25.0),
            shapes.IllCondGaussian(dim, base + 2, rotated=True, kappa=25.0),
            shapes.Banana(dim, base + 3, rotated=True),
            shapes.StudentT(dim, base + 4, rotated=True, nu=3.0),
            shapes.GaussianMixture(dim, base + 5, rotated=True, separation=2.5),
        ]
        for t_idx, tgt in enumerate(targets):
            res = invariance.check_target_invariance(
                tgt, sub_path, seed=base + 10 + t_idx, n_chains=n_chains,
                k_steps=k_steps, adapt_draws=adapt_draws, isolated=True,
                capabilities={"density_only"},
            )
            eff = res.dual_horizon_eff if res.dual_horizon_eff > 0 else res.empirical_efficiency
            if not res.passed:
                all_passed = False
            worst_b2 = max(worst_b2, float(res.b2_max))
            if not tgt.multimodal:
                by_class.setdefault(tgt.family, []).append(eff)
            rows.append({
                "dim": dim,
                "family": tgt.family,
                "rotated": tgt.rotated,
                "passed": res.passed,
                "emp_eff": round(float(eff), 8),
                "b2_max": round(float(res.b2_max), 6),
                "max_z": round(float(res.max_z), 3),
                "z_threshold": round(float(res.z_threshold), 3),
                "adapt_cost": round(float(res.adapt_cost), 1),
                "overhead_ratio": round(float(res.overhead_ratio), 3),
                "reason": res.reason,
            })

    class_geo = {
        c: float(np.exp(np.mean(np.log(np.maximum(vs, 1e-18))))) for c, vs in by_class.items()
    }
    soft_min = invariance.soft_min_power_mean(list(class_geo.values()), p=-4.0) if all_passed else 0.0
    report = {
        "track": "L5",
        "description": "Density-only view of L1 (n_gradient == 0 strictly enforced)",
        "submission": str(sub_path),
        "dims": list(dims),
        "passed_density_only": True,
        "gradient_evals": total_grads,
        "passed_all_invariance": all_passed,
        "soft_min_eff": round(soft_min, 8),
        "by_class": {k: round(v, 8) for k, v in class_geo.items()},
        "worst_b2_max": round(worst_b2, 6),
        "wall_s": round(time.perf_counter() - t0, 2),
        "machine": machine_fingerprint(),
        "targets": rows,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "track_L5.json").write_text(json.dumps(report, indent=2))
        (out_dir / "track_L5.md").write_text(format_track_markdown(report))
    return report


def format_track_markdown(report: dict[str, Any]) -> str:
    """Render a Track L3/L4/L5 JSON report into a compact GitHub-flavoured Markdown summary."""
    track = report.get("track", "?")
    lines = [f"# Track {track} Report", ""]
    if report.get("pool_policy"):
        lines.append(f"> **Policy:** {report['pool_policy']}")
        lines.append("")
    lines.append(f"- **Submission:** `{report.get('submission')}`")
    if track == "L3":
        lines.append(f"- **Passed all invariance checks:** `{report.get('passed_all_invariance')}`")
        lines.append(f"- **Overall soft-min efficiency:** `{report.get('soft_min_eff')}`")
        lines.append(f"- **Scale-invariance ratio (extreme / unit):** `{report.get('scale_invariance_ratio')}`")
        lines.append(f"- **Worst b2_max:** `{report.get('worst_b2_max')}`")
        lines.append("")
        lines.append("| scale | dim | family | passed | emp_eff | b2_max | max_z / thresh |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in report.get("targets", []):
            lines.append(
                f"| {r['scale_stress']:g} | {r['dim']} | {r['family']} | {r['passed']} | "
                f"{r['emp_eff']:.3g} | {r['b2_max']:.4g} | {r['max_z']:.2f}/{r['z_threshold']:.2f} |"
            )
    elif track == "L4":
        lines.append(
            f"- **Targets reaching b^2_max < {report.get('bias_threshold')}:** "
            f"`{report.get('targets_reaching_threshold')} / {report.get('targets_total')}`"
        )
        lines.append(f"- **Median cost to threshold:** `{report.get('median_cost_to_b2_threshold')}`")
        lines.append(f"- **Worst measured b^2_max:** `{report.get('worst_measured_b2_max')}`")
        lines.append("")
        lines.append("| dim | family | invariance_rejected | measured_b2_max | cost_to_b2_thresh | emp_eff |")
        lines.append("|---|---|---|---|---|---|")
        for r in report.get("targets", []):
            lines.append(
                f"| {r['dim']} | {r['family']} | {r['invariance_rejected']} | "
                f"{r['measured_b2_max']:.4g} | {r['cost_to_b2_threshold']} | {r['empirical_efficiency']:.3g} |"
            )
    elif track == "L5":
        lines.append(f"- **Density-only enforced (`n_gradient == 0`):** `{report.get('passed_density_only')}`")
        if report.get("error"):
            lines.append(f"- **Error:** `{report['error']}`")
        else:
            lines.append(f"- **Passed all invariance checks:** `{report.get('passed_all_invariance')}`")
            lines.append(f"- **Soft-min efficiency:** `{report.get('soft_min_eff')}`")
            lines.append("")
            lines.append("| dim | family | rotated | passed | emp_eff | b2_max |")
            lines.append("|---|---|---|---|---|---|")
            for r in report.get("targets", []):
                lines.append(
                    f"| {r['dim']} | {r['family']} | {r['rotated']} | {r['passed']} | "
                    f"{r['emp_eff']:.3g} | {r['b2_max']:.4g} |"
                )
    return "\n".join(lines) + "\n"
