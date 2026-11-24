"""The grader: one score for one sampler file, used by every caller that ranks a sampler.

A trial's result is its speed-up: the file's soft-min efficiency at the standard configuration
divided by the frozen score of the lane's starting code at the same configuration. Three callers
use this module and differ only in the seed:

  1. the lane loop (`apin autoresearch eval`, harness/autoresearch.py), on the public seed,
  2. an agent's own check inside a benchmark container, which is the same command,
  3. the benchmark verifier (harbor/verify.py), on a hidden seed, after the agent has finished.

`score_file` runs Tier 0 (lint, lane seam, isolated replay, invariance smoke test), then Tier 1
(`repeats` separately seeded suites at `dims`), then, only after a purely statistical Tier 1
rejection, one confirmation pass on the configuration's confirmation seed. For a given file and
configuration the result is deterministic.

The hidden verifier seeds are not in this module. They live in harbor/verify.py, which never
reaches an agent container. Every constant here can change a score, so this file is part of the
judge digest (harness/judge.py::judge_digest).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import invariance, ipc, logdensity, shapes
from .models import ROOT

# Public seeds. The lane loop and an agent's own check score on these. They are disjoint from
# every hidden verifier seed (harbor/verify.py), so tuning a sampler to them does not carry over to
# the final grade. A seed s uses suites seeded up to about s + 60,000.
PUBLIC_SEED = 15800004
PUBLIC_CONFIRMATION_SEED = 60352169

# The standard configuration. REPEATS comes from the A/A calibration in harbor/README.md
# (scripts/calibrate_harbor_noise.py): at 8 repeats, copies of one sampler that differ only in their
# random numbers scored at most 1.025x each other (p95 1.011, 26 copies over lanes A0 to A4). At 4
# repeats they reached 1.23x, at 2 repeats 1.39x.
REPEATS = 8
DIMS = (6, 20)
N_CHAINS = 48
K_STEPS = 16
ADAPT_DRAWS = 100
HORIZON_STEPS = 500

# A speed-up counts as a gain only when it beats the comparison by this factor: twice the largest
# A/A difference at 8 repeats.
IMPROVEMENT_MARGIN = 1.05

# The Tier 1 invariance and rank tests reject a correct sampler by chance now and then (about one
# file in eight at 8 repeats). A file whose only Tier 1 failures are such rejections is scored once
# more on the confirmation seed and is invalid only if that run fails too.
STATISTICAL_REJECTIONS = ("invariance rejected on", "rank test rejected on")


def config(*, seed: int = PUBLIC_SEED, confirmation_seed: int | None = PUBLIC_CONFIRMATION_SEED,
           dims: tuple[int, ...] = DIMS, repeats: int = REPEATS, n_chains: int = N_CHAINS,
           k_steps: int = K_STEPS, adapt_draws: int = ADAPT_DRAWS,
           horizon_steps: int = HORIZON_STEPS) -> dict:
    """A full scoring configuration. With no arguments it is the standard public one. It is part
    of every frozen key, so two scores are comparable only when their configurations are equal."""
    return {
        "seed": int(seed), "dims": [int(d) for d in dims], "repeats": int(max(1, repeats)),
        "n_chains": int(n_chains), "k_steps": int(k_steps), "adapt_draws": int(adapt_draws),
        "horizon_steps": int(horizon_steps),
        "confirmation_seed": None if confirmation_seed is None else int(confirmation_seed),
    }


def is_standard(cfg: dict) -> bool:
    """True for the standard public configuration, the only one whose trials get a speed-up."""
    return cfg == config()


# ---------------------------------------------------------------------------------------------
# Tier 0
# ---------------------------------------------------------------------------------------------

_LOAD_CHECK = (
    "import sys; from pathlib import Path; from harness import judge, sandbox; "
    "p = Path(sys.argv[1]); sandbox.confine_worker(p); judge.resolve_lane_submission(p, sys.argv[2] or None)"
)


def load_check(sub_path: Path, lane: str | None) -> None:
    """Check that a submission imports and exposes its lane's seam, in a confined child process.

    A submission is untrusted code: a benchmark candidate, or another contributor's trial being
    rerun. It must never be imported in this process, which computes the statistics and writes
    the verdict, so the import happens in a child confined like the sampling workers."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", _LOAD_CHECK, str(Path(sub_path).resolve()), lane or ""],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("submission did not finish importing within 120 s") from e
    if res.returncode != 0:
        lines = [ln for ln in res.stderr.strip().splitlines() if ln.strip()]
        raise RuntimeError(lines[-1] if lines else f"submission failed to load (exit {res.returncode})")


def eval_tier0(sub_path: Path, seed: int, work_dir: Path, *, lane: str | None = None,
               horizon_steps: int = HORIZON_STEPS) -> tuple[bool, dict]:
    """Tier 0 fast reject: AST lint, lane seam, isolated ABI + replay, calibrated invariance smoke."""
    try:
        logdensity.check_imports(Path(sub_path).read_text())
        load_check(sub_path, lane)
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


# ---------------------------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------------------------

def build_suite(seed: int, dims: tuple[int, ...]) -> list[shapes.AnalyticTarget]:
    """Fresh aligned + rotated Tier 1 suite at dims[0], plus a scaling ladder for later dims."""
    ladder = tuple(int(d) for d in dims)
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


def _geo(vals) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(vals, dtype=float), 1e-18)))))


def eval_suite(sub_path: Path, suite: list[shapes.AnalyticTarget], seed: int, *, n_chains: int,
               k_steps: int, adapt_draws: int, lane: str | None, horizon_steps: int) -> dict:
    """Evaluate a submission on one suite via isolated workers. A failed check zeroes the soft-mins."""
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

    class_geo = {c: _geo(vs) for c, vs in by_class.items()}
    class_short = {c: _geo(vs) for c, vs in by_class_short.items()}
    class_long = {c: _geo(vs) for c, vs in by_class_long.items()}
    soft_min = invariance.soft_min_power_mean(list(class_geo.values()), p=-4.0) if all_ok else 0.0
    short_sm = invariance.soft_min_power_mean(list(class_short.values()), p=-4.0) if all_ok else 0.0
    long_sm = invariance.soft_min_power_mean(list(class_long.values()), p=-4.0) if all_ok else 0.0
    return {
        "passed": all_ok,
        "soft_min_eff": soft_min,
        "short_horizon_soft_min": short_sm,
        "long_horizon_soft_min": long_sm,
        "median_adapt_cost": float(np.median(adapt_vals)) if adapt_vals else 0.0,
        "geomean_eff": _geo(list(class_geo.values())) if class_geo else 0.0,
        "b2_max": b2_worst,
        "overhead_ratio": float(np.median(oh_vals)) if oh_vals else 0.0,
        "by_class": class_geo,
        "per_target": per_target,
    }


def eval_tier1(sub_path: Path, suites: list[tuple[int, list[shapes.AnalyticTarget]]], *, n_chains: int,
               k_steps: int, adapt_draws: int, lane: str | None, horizon_steps: int) -> dict:
    """Evaluate on one or more separately seeded suites and aggregate by geometric mean."""
    kw = dict(n_chains=n_chains, k_steps=k_steps, adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps)
    if len(suites) == 1:
        rseed, suite = suites[0]
        return eval_suite(sub_path, suite, seed=rseed + 1000, **kw)
    runs = [eval_suite(sub_path, suite, seed=rseed + 1000, **kw) for rseed, suite in suites]
    all_ok = all(r["passed"] for r in runs)
    classes = sorted({c for r in runs for c in r["by_class"]})
    by_class = {c: _geo([r["by_class"][c] for r in runs if c in r["by_class"]]) for c in classes}
    return {
        "passed": all_ok,
        "soft_min_eff": _geo([r["soft_min_eff"] for r in runs]) if all_ok else 0.0,
        "short_horizon_soft_min": _geo([r["short_horizon_soft_min"] for r in runs]) if all_ok else 0.0,
        "long_horizon_soft_min": _geo([r["long_horizon_soft_min"] for r in runs]) if all_ok else 0.0,
        "median_adapt_cost": float(np.median([r["median_adapt_cost"] for r in runs])),
        "geomean_eff": _geo(list(by_class.values())) if by_class else 0.0,
        "b2_max": max(r["b2_max"] for r in runs),
        "overhead_ratio": float(np.median([r["overhead_ratio"] for r in runs])),
        "by_class": by_class,
        "per_repeat": runs,
        "per_target": [t for r in runs for t in r["per_target"]],
    }


def tier1(path: Path, lane: str, cfg: dict, seed: int) -> tuple[dict, list]:
    """One Tier 1 pass: cfg["repeats"] suites seeded seed, seed + 7919, ..."""
    dims = tuple(cfg["dims"])
    suites = [(seed + r * 7919, build_suite(seed + r * 7919, dims)) for r in range(cfg["repeats"])]
    res = eval_tier1(path, suites, n_chains=cfg["n_chains"], k_steps=cfg["k_steps"],
                     adapt_draws=cfg["adapt_draws"], lane=lane, horizon_steps=cfg["horizon_steps"])
    return res, suites


def scores_from_targets(res: dict, suites: list) -> dict:
    """Soft-min, horizon soft-mins and class scores rebuilt from the per-target efficiencies with
    the formulas of `eval_suite` and `eval_tier1`, but without zeroing a suite that has a failed
    check. For a run whose checks all pass this equals the judge's numbers (score_file checks that
    on every such run)."""
    runs = res.get("per_repeat") or [res]
    reps = []
    complete = True
    for run, (_, suite) in zip(runs, suites, strict=True):
        cls: dict[str, list[float]] = {}
        short: dict[str, list[float]] = {}
        long_: dict[str, list[float]] = {}
        for tgt, p in zip(suite, run["per_target"], strict=True):
            if tgt.multimodal:
                continue
            effs = (p["emp_eff"], p["short_horizon_eff"], p["long_horizon_eff"])
            complete = complete and all(isinstance(e, (int, float)) and math.isfinite(e) and e > 0 for e in effs)
            cls.setdefault(tgt.family, []).append(p["emp_eff"])
            short.setdefault(tgt.family, []).append(p["short_horizon_eff"])
            long_.setdefault(tgt.family, []).append(p["long_horizon_eff"])
        by_class = {c: _geo(v) for c, v in cls.items()}
        reps.append({
            "soft_min_eff": invariance.soft_min_power_mean(list(by_class.values()), p=-4.0),
            "short_horizon_soft_min": invariance.soft_min_power_mean([_geo(v) for v in short.values()], p=-4.0),
            "long_horizon_soft_min": invariance.soft_min_power_mean([_geo(v) for v in long_.values()], p=-4.0),
            "by_class": by_class,
        })
    if len(reps) == 1:
        agg = dict(reps[0])
    else:
        agg = {k: _geo([r[k] for r in reps])
               for k in ("soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min")}
        classes = sorted({c for r in reps for c in r["by_class"]})
        agg["by_class"] = {c: _geo([r["by_class"][c] for r in reps if c in r["by_class"]]) for c in classes}
    agg["per_repeat_soft_min"] = [float(r["soft_min_eff"]) for r in reps]
    agg["per_repeat_by_class"] = [{k: float(v) for k, v in r["by_class"].items()} for r in reps]
    agg["complete"] = complete  # every target reported a finite, positive efficiency
    return agg


def _failures(res: dict) -> list[str]:
    return [f"{p['family']}(d={p.get('dim')}, rot={p.get('rotated')}): {p.get('reason')}"
            for p in (res.get("per_target") or []) if not p.get("passed")]


def only_statistical(res: dict) -> bool:
    """True when every failed check of a Tier 1 result is a statistical rejection."""
    # The judge joins every failed gate of a target with "; ": all of them must be statistical.
    reasons = [part for p in (res.get("per_target") or []) if not p.get("passed")
               for part in (str(p.get("reason") or "").split("; ") or [""])]
    return bool(reasons) and all(r.startswith(STATISTICAL_REJECTIONS) for r in reasons)


def _check_rebuild(res: dict, rebuilt: dict) -> None:
    for k in ("soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min"):
        a, b = float(res[k]), float(rebuilt[k])
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-300):
            raise RuntimeError(f"rebuilt {k} {b!r} differs from the judge's {a!r}; scores_from_targets is stale")


def score_file(path: Path, lane: str, cfg: dict, work_dir: Path, *, confirm: bool = True) -> dict:
    """Grade one file: Tier 0, then Tier 1 on cfg["seed"], then (only after a purely statistical
    Tier 1 rejection) one confirmation Tier 1 pass on cfg["confirmation_seed"]. Returns a plain
    dict. `valid` is the pass flag and `confirmation` says whether a confirmation ran and how it
    ended.

    Tier 0 failures, non-statistical Tier 1 failures (stuck kernel, adapt-cost cap, exceptions,
    timeouts inside adapt() or transition()) and a missing file are invalid at once. After a
    confirmed pass the efficiency numbers are the first run's, rebuilt from the per-target
    efficiencies (`scores_from_targets`), so they stay comparable with frozen scores on cfg["seed"].
    If the first run left no usable efficiency, the file is valid but `unpaired`, with no score."""
    t0 = time.perf_counter()
    path = Path(path)
    out: dict = {"valid": False, "tier0_ok": False, "tier1_ok": False, "confirmation": {"ran": False}}
    if not path.is_file():
        out.update(error=f"missing file: {path}", wall_s=0.0)
        return out
    seed = cfg["seed"]
    work_dir.mkdir(parents=True, exist_ok=True)
    t0_ok, t0_info = eval_tier0(path, seed=seed, work_dir=work_dir / "t0", lane=lane,
                                horizon_steps=cfg["horizon_steps"])
    out["tier0_info"] = t0_info
    if not t0_ok:
        out.update(error=f"tier0 {t0_info.get('stage')}: {t0_info.get('error')}",
                   wall_s=round(time.perf_counter() - t0, 1))
        return out
    out["tier0_ok"] = True

    res, suites = tier1(path, lane, cfg, seed)
    first = scores_from_targets(res, suites)
    if res["passed"]:
        _check_rebuild(res, first)
    failed = _failures(res)
    out.update(
        tier1_ok=bool(res["passed"]),
        valid=bool(res["passed"]),
        b2_max=float(res["b2_max"]),
        overhead_ratio=float(res["overhead_ratio"]),
        median_adapt_cost=float(res["median_adapt_cost"]),
        failed_targets=failed[:8],
        per_target=res.get("per_target") or [],
        **{k: v for k, v in first.items() if k != "complete"},
    )
    if not res["passed"]:
        conf_seed = cfg.get("confirmation_seed")
        if confirm and conf_seed is not None and only_statistical(res):
            t_conf = time.perf_counter()
            res2, _ = tier1(path, lane, cfg, int(conf_seed))
            conf = {"ran": True, "passed": bool(res2["passed"]), "seed": int(conf_seed),
                    "failed_targets_first_run": failed[:8], "failed_targets": _failures(res2)[:8],
                    "wall_s": round(time.perf_counter() - t_conf, 1)}
            if res2["passed"]:
                if first["complete"] and first["soft_min_eff"] > 0.0 and math.isfinite(first["soft_min_eff"]):
                    conf["efficiency_source"] = "first_run_paired"
                else:
                    # The first run left no usable efficiency. A score from the confirmation seed is
                    # not comparable with frozen scores on the primary seed, so none is reported.
                    conf["efficiency_source"] = "none"
                    out.update(unpaired=True, soft_min_eff=0.0, short_horizon_soft_min=0.0, long_horizon_soft_min=0.0)
                out.update(valid=True, tier1_ok=True, failed_targets=[])
            else:
                out["failed_targets"] = conf["failed_targets"]
            out["confirmation"] = conf
        else:
            out["confirmation"] = {"ran": False, "reason": (
                "a Tier 1 failure that is not a statistical rejection" if confirm else "confirmation disabled")}
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    if not out["valid"]:
        # A pass with a failed check has no soft-min, and a failed repeat scores 0
        # (scripts/calibrate_harbor_noise.py reads failed repeats this way).
        runs = res.get("per_repeat") or [res]
        out["per_repeat_soft_min"] = [v if run["passed"] else 0.0
                                      for v, run in zip(out["per_repeat_soft_min"], runs, strict=True)]
        out.update(soft_min_eff=0.0, short_horizon_soft_min=0.0, long_horizon_soft_min=0.0)
        where = (f" on seed {seed} and the confirmation seed {out['confirmation']['seed']}"
                 if out["confirmation"].get("ran") else "")
        out["error"] = f"tier1{where}: " + ("; ".join(out["failed_targets"]) or "failed")
    return out


# ---------------------------------------------------------------------------------------------
# Frozen scores and speed-up
# ---------------------------------------------------------------------------------------------

FROZEN_KEYS = ("valid", "tier0_ok", "tier1_ok", "unpaired", "soft_min_eff", "short_horizon_soft_min",
               "long_horizon_soft_min", "median_adapt_cost", "b2_max", "overhead_ratio", "by_class",
               "per_repeat_soft_min", "per_repeat_by_class", "failed_targets", "error", "confirmation", "wall_s")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_entry(lane: str, role: str, path: Path, cfg: dict, digest: str, score: dict) -> dict:
    """The frozen record of one file's score. Its key is (lane, role, sha256, judge_digest, config)."""
    entry = {"lane": lane, "role": role, "file": Path(path).name, "sha256": sha256(path),
             "judge_digest": digest, "config": cfg}
    entry.update({k: score[k] for k in FROZEN_KEYS if k in score})
    return entry


def frozen_match(entry: dict | None, lane: str, role: str, sha: str, cfg: dict, digest: str) -> bool:
    """A frozen score stands in for a live one only when every part of its key matches."""
    return bool(entry) and (
        entry.get("lane") == lane and entry.get("role") == role and entry.get("sha256") == sha
        and entry.get("judge_digest") == digest and entry.get("config") == cfg
    )


def soft_min(score: dict | None) -> float | None:
    """The usable soft-min of a score or frozen entry, or None when it failed a check."""
    if not score or not score.get("valid") or score.get("unpaired"):
        return None
    v = score.get("soft_min_eff")
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None


def speedup(score: dict | None, start: dict | None) -> float | None:
    """Speed-up of a scored file over the frozen starting-code score of its lane, on the same
    configuration: candidate soft-min / start soft-min. None when either failed a check."""
    c, s = soft_min(score), soft_min(start)
    return c / s if c is not None and s is not None else None


# A yardstick is the frozen score of a lane's starting code (role "start") at the standard public
# configuration. It carries a short id that changes whenever its key or its score changes, so a
# speed-up names the yardstick it was measured against.
YARDSTICK_SCHEMA = 1
_YARDSTICK_ID_FIELDS = ("lane", "sha256", "judge_digest", "config", "valid", "soft_min_eff",
                        "short_horizon_soft_min", "long_horizon_soft_min", "by_class")


def yardstick_id(entry: dict) -> str:
    body = json.dumps({k: entry.get(k) for k in _YARDSTICK_ID_FIELDS}, sort_keys=True, separators=(",", ":"))
    return f"{entry.get('lane')}-{hashlib.sha256(body.encode()).hexdigest()[:12]}"


def yardstick_entry(lane: str, path: Path, cfg: dict, digest: str, score: dict) -> dict:
    entry = {"schema": YARDSTICK_SCHEMA, **frozen_entry(lane, "start", path, cfg, digest, score)}
    entry["id"] = yardstick_id(entry)
    return entry
