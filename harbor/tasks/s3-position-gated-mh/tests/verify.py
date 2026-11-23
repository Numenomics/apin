#!/usr/bin/env python3
"""Canonical Harbor verifier for apin Agent Benchmark tasks.

harbor/verify.py is the single source; scripts/sync_harbor_tests.py copies it verbatim to every
harbor/tasks/*/tests/verify.py. Writes Harbor-standard reward.txt, reward.json (numbers only) and
details.json to /logs/verifier (or --out-dir).

  - `rewind` mode: reward 1.0 if the hidden invariance/Jacobian check passes, else 0.0;
    outcome `solved` or `invalid`.
  - `optimize` mode: only the candidate is scored at run time, on the hidden verifier seeds. Its
    soft-min efficiency is divided by three frozen denominators read from tests/frozen_scores.json:
    the starting code (tests/baseline.py), the Stan-style reference (tests/reference.py) and the
    lane's frontier incumbent at freeze time (tests/frontier.py). A denominator whose frozen key
    (lane, judge digest, scoring config, file SHA-256) does not match is scored live instead.
    reward = ref_speedup = candidate / reference (0 if the candidate fails a check).
    A Tier 1 failure made only of statistical rejections is confirmed on CONFIRMATION_SEED before
    the candidate is called invalid (see score_file).

Isolation: the candidate file is the only input taken from the agent. It is copied into a private
temporary directory before scoring. The harness is imported from the image (/app), and the
denominator files and frozen scores come from the tests directory, which Harbor uploads. Run it
with `python -I` (tests/test.sh does) so no environment variable or working-directory module can
change what is imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _find_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "harness" / "judge.py").is_file():
            return parent
    if Path("/app/harness/judge.py").is_file():
        return Path("/app")
    return Path.cwd()


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from harness import autoresearch, invariance, judge, rewind

# Verifier seed, rotated every release. It is public once a release ships (it is in this file);
# during a run the agent cannot read it because tests/ is uploaded only after the agent phase and
# the agent's network is sealed. bench-v0.2.0 retired 88421077.
HIDDEN_VERIFIER_SEED = 39389868

# Scoring configuration. VERIFIER_REPEATS and IMPROVEMENT_MARGIN come from the A/A calibration in
# harbor/README.md (scripts/calibrate_harbor_noise.py, harbor/calibration/summary.json): at 8
# repeats, copies of the starting code that differ only in their random numbers scored at most
# 1.025x the frozen starting code (p95 1.011, 26 copies over lanes A0-A4); at 4 repeats they reached
# 1.23x, at 2 repeats 1.39x.
VERIFIER_REPEATS = 8
IMPROVEMENT_MARGIN = 1.05

# Confirmation of statistical rejections. The Tier 1 invariance and rank tests reject a correct
# sampler by chance now and then (about one file in eight at 8 repeats, harbor/README.md). A
# candidate whose only Tier 1 failures are such rejections is scored once more on this seed with
# the same configuration and is invalid only if that run fails too. Disjoint from every seed the
# verifier seed's suites use; rotated with it.
CONFIRMATION_SEED = 78210474
STATISTICAL_REJECTIONS = ("invariance rejected on", "rank test rejected on")
VERIFIER_DIMS = (6, 20)
N_CHAINS = 48
K_STEPS = 16
ADAPT_DRAWS = 100
HORIZON_STEPS = 500

# Lane -> Harbor task directory of the optimize tasks.
LANE_TASKS = {
    "A0": "a0-ground-up",
    "A1": "a1-preconditioning",
    "A2": "a2-funnel-dynamics",
    "A3": "a3-integrator-trajectory",
    "A4": "a4-nonlinear-transport",
}
# Denominator role -> file in the task's tests directory.
DENOMINATOR_FILES = {"start": "baseline.py", "reference": "reference.py", "frontier": "frontier.py"}
FROZEN_FILE = "frozen_scores.json"
FROZEN_SCHEMA = 1

OUTCOMES = ("invalid", "unscorable", "no_gain", "improved", "solved")

# Accept either Harbor task directory slugs (hyphenated) or internal tasks.json IDs (underscored).
REWIND_TASK_ALIASES = {
    "s3-position-gated-mh": "s3_position_gated_mh_bias",
    "s3-missing-jacobian-a4": "s3_missing_jacobian_transport",
    "s3-unadjusted-ula": "s3_unadjusted_ula_bias",
}


def _default_out_dir() -> Path:
    harbor_dir = Path("/logs/verifier")
    if harbor_dir.parent.is_dir() and os.access(harbor_dir.parent, os.W_OK):
        return harbor_dir
    return Path.cwd() / "verifier_out"


def _sig(v: float | None, digits: int = 6) -> float | None:
    """Round to significant digits (soft-min values are ~1e-4, so fixed decimals lose them)."""
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{float(v):.{digits}g}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_TRIVIAL_CONSTS = frozenset({-1, 0, 1, 2})  # equal floats (-1.0 ... 2.0) match too


def _solution_stats(submission: Path) -> dict:
    """Measure code size and structural complexity of the submitted Python file."""
    if not submission.is_file():
        return {}
    raw = submission.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    loc = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    magic_constants = 0
    ast_nodes = 0
    cyclomatic = 1
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            ast_nodes += 1
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value not in _TRIVIAL_CONSTS:
                    magic_constants += 1
            elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)):
                cyclomatic += 1
            elif isinstance(node, ast.BoolOp):
                cyclomatic += max(0, len(node.values) - 1)
            elif isinstance(node, ast.comprehension):
                cyclomatic += 1 + len(node.ifs)
    except SyntaxError:
        pass
    return {
        "loc": loc,
        "bytes": len(raw),
        "magic_constants": magic_constants,
        "ast_nodes": ast_nodes,
        "cyclomatic_complexity": cyclomatic,
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
    }


def _archive_submission(submission: Path, out_dir: Path) -> None:
    if submission.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "submission.py"
        if submission.resolve() != dest.resolve():
            shutil.copy2(submission, dest)


# reward.json holds numbers only (Harbor's parser rejects anything else); details.json holds all.
_NUMERIC_KEYS = (
    "passed", "solved", "improved", "unscorable",
    "ref_speedup", "start_speedup", "frontier_speedup", "gap_closed",
    "soft_min_eff", "start_soft_min", "reference_soft_min", "frontier_soft_min",
    "short_horizon_soft_min", "long_horizon_soft_min", "b2_max", "overhead_ratio",
    "confirmed", "improvement_margin", "repeats", "candidate_wall_s", "emp_eff", "max_z",
)


def _write_rewards(out_dir: Path, reward: float, payload: dict, *, submission: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_reward = float(reward) if (math.isfinite(reward) and reward > 0.0) else 0.0
    extra = dict(payload)
    if submission is not None and submission.is_file():
        extra.setdefault("solution", _solution_stats(submission))
        _archive_submission(submission, out_dir)
    details = {"reward": round(clean_reward, 6), **extra}
    numeric_rewards: dict[str, float | int] = {"reward": round(clean_reward, 6)}
    for k in _NUMERIC_KEYS:
        v = extra.get(k)
        if isinstance(v, bool):
            numeric_rewards[k] = int(v)
        elif isinstance(v, (int, float)) and math.isfinite(v):
            numeric_rewards[k] = v if isinstance(v, int) else float(v)
    sol = extra.get("solution") if isinstance(extra.get("solution"), dict) else {}
    for sk, nk in (("loc", "solution_loc"), ("cyclomatic_complexity", "solution_cyclomatic"),
                   ("magic_constants", "solution_magic_constants"), ("ast_nodes", "solution_ast_nodes")):
        sv = sol.get(sk)
        if isinstance(sv, (int, float)) and not isinstance(sv, bool) and math.isfinite(sv):
            numeric_rewards[nk] = int(sv)

    (out_dir / "reward.txt").write_text(f"{clean_reward:.6f}\n", encoding="utf-8")
    (out_dir / "reward.json").write_text(json.dumps(numeric_rewards, indent=2) + "\n", encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(details, indent=2))


# ---------------------------------------------------------------------------------------------
# Rewind tasks
# ---------------------------------------------------------------------------------------------

def verify_rewind(task_id: str, submission: Path, out_dir: Path, tasks_path: Path) -> int:
    tasks = {t.task_id: t for t in rewind.load_tasks(tasks_path)}
    resolved_id = REWIND_TASK_ALIASES.get(task_id, task_id)
    if resolved_id not in tasks:
        valid = sorted(set(tasks) | set(REWIND_TASK_ALIASES))
        _write_rewards(out_dir, 0.0, {"passed": 0, "solved": 0, "outcome": "invalid",
                                      "error": f"unknown task_id {task_id!r}; valid: {valid}"})
        return 1
    res = rewind.evaluate_rewind_task(tasks[resolved_id], submission)
    passed = 1 if res["passed"] else 0
    _write_rewards(
        out_dir, float(passed),
        {"passed": passed, "solved": passed, "outcome": "solved" if passed else "invalid", **res},
        submission=submission,
    )
    return 0 if passed else 1


# ---------------------------------------------------------------------------------------------
# Optimize tasks
# ---------------------------------------------------------------------------------------------

def scoring_config(*, seed: int = HIDDEN_VERIFIER_SEED, dims: tuple[int, ...] = VERIFIER_DIMS,
                   repeats: int = VERIFIER_REPEATS, n_chains: int = N_CHAINS, k_steps: int = K_STEPS,
                   adapt_draws: int = ADAPT_DRAWS, horizon_steps: int = HORIZON_STEPS,
                   confirmation_seed: int = CONFIRMATION_SEED) -> dict:
    """The full scoring configuration; part of every frozen key."""
    return {
        "seed": int(seed), "dims": [int(d) for d in dims], "repeats": int(max(1, repeats)),
        "n_chains": int(n_chains), "k_steps": int(k_steps), "adapt_draws": int(adapt_draws),
        "horizon_steps": int(horizon_steps), "confirmation_seed": int(confirmation_seed),
    }


def _tier1(path: Path, lane: str, cfg: dict, seed: int) -> tuple[dict, list]:
    """One Tier 1 pass (cfg["repeats"] separately seeded suites) via the judge."""
    dims = tuple(cfg["dims"])
    suites = [
        (seed + r * 7919, autoresearch._build_tier1_suite(dims[0], seed + r * 7919, dims=dims))
        for r in range(cfg["repeats"])
    ]
    res = autoresearch._eval_tier1_multi(
        path, suites, n_chains=cfg["n_chains"], k_steps=cfg["k_steps"],
        adapt_draws=cfg["adapt_draws"], lane=lane, horizon_steps=cfg["horizon_steps"],
    )
    return res, suites


def _geo(vals) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(vals, dtype=float), 1e-18)))))


def scores_from_targets(res: dict, suites: list) -> dict:
    """Soft-min, horizon soft-mins and class scores rebuilt from the judge's per-target
    efficiencies with the judge's own formulas (`harness/autoresearch.py::_eval_tier1_suite` and
    `_eval_tier1_multi`), but without zeroing a suite that has a failed check. For a run whose
    checks all pass this equals the judge's numbers (score_file checks that on every such run)."""
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


def _only_statistical(res: dict) -> bool:
    reasons = [str(p.get("reason") or "") for p in (res.get("per_target") or []) if not p.get("passed")]
    return bool(reasons) and all(r.startswith(STATISTICAL_REJECTIONS) for r in reasons)


def _check_rebuild(res: dict, rebuilt: dict) -> None:
    for k in ("soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min"):
        a, b = float(res[k]), float(rebuilt[k])
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-300):
            raise RuntimeError(f"rebuilt {k} {b!r} differs from the judge's {a!r}; scores_from_targets is stale")


def score_file(path: Path, lane: str, cfg: dict, work_dir: Path, *, confirm: bool = True) -> dict:
    """Tier 0, then Tier 1 on the verifier seeds, then (only after a purely statistical Tier 1
    rejection) one confirmation Tier 1 pass on cfg["confirmation_seed"]. Returns a plain dict;
    `valid` is the pass flag and `confirmation` says whether a confirmation ran and how it ended.

    Tier 0 failures, non-statistical Tier 1 failures (stuck kernel, adapt-cost cap, exceptions,
    timeouts inside adapt() or transition()) and a missing file are invalid at once. After a
    confirmed pass the efficiency numbers are the first run's, rebuilt from the judge's per-target
    efficiencies (`scores_from_targets`), so they stay paired with the frozen denominators."""
    t0 = time.perf_counter()
    path = Path(path)
    out: dict = {"valid": False, "tier0_ok": False, "tier1_ok": False, "confirmation": {"ran": False}}
    if not path.is_file():
        out.update(error=f"missing file: {path}", wall_s=0.0)
        return out
    seed = cfg["seed"]
    work_dir.mkdir(parents=True, exist_ok=True)
    t0_ok, t0_info = autoresearch._eval_tier0(
        path, seed=seed, work_dir=work_dir / "t0", lane=lane, horizon_steps=cfg["horizon_steps"],
    )
    out["tier0_info"] = t0_info
    if not t0_ok:
        out.update(error=f"tier0 {t0_info.get('stage')}: {t0_info.get('error')}",
                   wall_s=round(time.perf_counter() - t0, 1))
        return out
    out["tier0_ok"] = True

    res, suites = _tier1(path, lane, cfg, seed)
    first = scores_from_targets(res, suites)
    if res["passed"]:
        _check_rebuild(res, first)
    failed = _failures(res)
    out.update(
        tier1_ok=bool(res["passed"]),
        valid=bool(res["passed"]),
        b2_max=float(res["b2_max"]),
        overhead_ratio=float(res["overhead_ratio"]),
        failed_targets=failed[:8],
        **{k: v for k, v in first.items() if k != "complete"},
    )
    if not res["passed"]:
        conf_seed = cfg.get("confirmation_seed")
        if confirm and conf_seed is not None and _only_statistical(res):
            t_conf = time.perf_counter()
            res2, suites2 = _tier1(path, lane, cfg, int(conf_seed))
            conf = {"ran": True, "passed": bool(res2["passed"]), "seed": int(conf_seed),
                    "failed_targets_first_run": failed[:8], "failed_targets": _failures(res2)[:8],
                    "wall_s": round(time.perf_counter() - t_conf, 1)}
            if res2["passed"]:
                if first["complete"] and first["soft_min_eff"] > 0.0 and math.isfinite(first["soft_min_eff"]):
                    conf["efficiency_source"] = "first_run_paired"
                else:
                    # The first run left no usable efficiency. A score from the confirmation seed
                    # is not paired with the frozen denominators, so no ratio is reported.
                    conf["efficiency_source"] = "none"
                    out["unpaired"] = True
                out.update(valid=True, tier1_ok=True, failed_targets=[])
            else:
                out["failed_targets"] = conf["failed_targets"]
            out["confirmation"] = conf
        else:
            out["confirmation"] = {"ran": False, "reason": (
                "a Tier 1 failure that is not a statistical rejection" if confirm else "confirmation disabled")}
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    if not out["valid"]:
        # The judge's convention: a pass with a failed check has no soft-min, and a failed repeat
        # scores 0 (scripts/calibrate_harbor_noise.py reads failed repeats this way).
        runs = res.get("per_repeat") or [res]
        out["per_repeat_soft_min"] = [v if run["passed"] else 0.0
                                      for v, run in zip(out["per_repeat_soft_min"], runs, strict=True)]
        out.update(soft_min_eff=0.0, short_horizon_soft_min=0.0, long_horizon_soft_min=0.0)
        where = (f" on the verifier seed and the confirmation seed {out['confirmation']['seed']}"
                 if out["confirmation"].get("ran") else "")
        out["error"] = f"tier1{where}: " + ("; ".join(out["failed_targets"]) or "failed")
    return out


def frozen_entry(lane: str, role: str, path: Path, cfg: dict, digest: str, score: dict) -> dict:
    """The frozen record of one denominator (written by scripts/freeze_harbor_references.py)."""
    keep = ("valid", "tier0_ok", "tier1_ok", "soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min",
            "b2_max", "overhead_ratio", "by_class", "per_repeat_soft_min", "per_repeat_by_class",
            "failed_targets", "error", "confirmation", "wall_s")
    entry = {"lane": lane, "role": role, "file": Path(path).name, "sha256": _sha256(path),
             "judge_digest": digest, "config": cfg}
    entry.update({k: score[k] for k in keep if k in score})
    return entry


def _frozen_match(entry: dict | None, lane: str, role: str, sha: str, cfg: dict, digest: str) -> bool:
    return bool(entry) and (
        entry.get("lane") == lane and entry.get("role") == role and entry.get("sha256") == sha
        and entry.get("judge_digest") == digest and entry.get("config") == cfg
    )


def load_frozen(tests_dir: Path) -> dict:
    p = Path(tests_dir) / FROZEN_FILE
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if data.get("schema") == FROZEN_SCHEMA else {}


def reference_name(path: Path) -> str | None:
    """REFERENCE_NAME from a reference file, read without executing it."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REFERENCE_NAME"
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            return node.value.value
    return None


def denominator(role: str, lane: str, path: Path, cfg: dict, digest: str, frozen: dict,
                work_dir: Path) -> dict:
    """Frozen score when the key matches, else a live score. Adds `source` and `sha256`."""
    path = Path(path)
    if not path.is_file():
        return {"valid": False, "source": "missing", "file": path.name, "error": f"missing {role} file {path.name}"}
    sha = _sha256(path)
    entry = (frozen.get("scores") or {}).get(role)
    if _frozen_match(entry, lane, role, sha, cfg, digest):
        out = dict(entry)
        out["source"] = "frozen"
        return out
    print(f"[verify] {role}: no matching frozen score; scoring {path.name} live", file=sys.stderr, flush=True)
    out = frozen_entry(lane, role, path, cfg, digest, score_file(path, lane, cfg, work_dir / role))
    out["source"] = "live"
    return out


def classify(candidate_valid: bool, candidate_sm: float, start: dict, reference: dict,
             frontier: dict | None = None, margin: float = IMPROVEMENT_MARGIN) -> dict:
    """Outcome rule. Denominators are dicts with `valid` and `soft_min_eff`.

    invalid     the candidate failed a Tier 0 or Tier 1 check (reward 0)
    unscorable  the candidate passed but the starting code or the reference failed a check on
                these seeds, so there is no ratio to report (reward 0, never a substituted 1x)
    no_gain     start_speedup < margin
    improved    start_speedup >= margin, ref_speedup < 1
    solved      start_speedup >= margin, ref_speedup >= 1
    """
    def sm(d):
        if not d or not d.get("valid"):
            return None
        v = d.get("soft_min_eff")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None

    s, r, f = sm(start), sm(reference), sm(frontier)
    c = float(candidate_sm) if candidate_valid and math.isfinite(candidate_sm) and candidate_sm > 0 else 0.0
    out = {"start_speedup": None, "ref_speedup": None, "frontier_speedup": None, "gap_closed": None}
    if not candidate_valid or c <= 0.0:
        out.update(outcome="invalid", reward=0.0,
                   start_speedup=0.0 if s else None, ref_speedup=0.0 if r else None,
                   frontier_speedup=0.0 if f else None)
        return out
    out["start_speedup"] = c / s if s else None
    out["ref_speedup"] = c / r if r else None
    out["frontier_speedup"] = c / f if f else None
    if s and f and f > s:
        out["gap_closed"] = math.log(c / s) / math.log(f / s)
    if s is None or r is None:
        bad = [name for name, v in (("starting code", s), ("reference", r)) if v is None]
        out.update(outcome="unscorable", reward=0.0,
                   unscorable_reason=" and ".join(bad) + " failed a check on the verifier seeds")
        return out
    if out["start_speedup"] < margin:
        outcome = "no_gain"
    elif out["ref_speedup"] < 1.0:
        outcome = "improved"
    else:
        outcome = "solved"
    out.update(outcome=outcome, reward=out["ref_speedup"])
    return out


def _tests_dir_for(lane: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    if (here / DENOMINATOR_FILES["start"]).is_file():
        return here
    return ROOT / "harbor" / "tasks" / LANE_TASKS[lane] / "tests"


def _summary(d: dict) -> dict:
    keys = ("valid", "source", "file", "sha256", "soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min",
            "by_class", "per_repeat_soft_min", "b2_max", "confirmation", "error", "wall_s", "name")
    out = {k: d[k] for k in keys if k in d}
    for k in ("soft_min_eff", "short_horizon_soft_min", "long_horizon_soft_min"):
        if k in out:
            out[k] = _sig(out[k])
    if "by_class" in out:
        out["by_class"] = {k: _sig(v) for k, v in out["by_class"].items()}
    if "per_repeat_soft_min" in out:
        out["per_repeat_soft_min"] = [_sig(v) for v in out["per_repeat_soft_min"]]
    if "sha256" in out:
        out["sha256"] = out["sha256"][:16]
    return out


def verify_optimize(lane: str, submission: Path, out_dir: Path, *, tests_dir: Path | None = None,
                    overrides: dict[str, Path] | None = None, cfg: dict | None = None,
                    margin: float = IMPROVEMENT_MARGIN) -> int:
    t_start = time.perf_counter()
    cfg = cfg or scoring_config()
    lane = lane.upper()
    tests_dir = _tests_dir_for(lane, tests_dir)
    files = {role: tests_dir / name for role, name in DENOMINATOR_FILES.items()}
    files.update({k: Path(v) for k, v in (overrides or {}).items() if v is not None})
    sub = Path(submission)
    digest = judge.judge_digest()
    base_payload = {"lane": lane, "task": LANE_TASKS.get(lane), "config": cfg, "repeats": cfg["repeats"],
                    "improvement_margin": margin, "judge_digest": digest}
    if not sub.is_file():
        _write_rewards(out_dir, 0.0, {**base_payload, "passed": 0, "solved": 0, "improved": 0, "unscorable": 0,
                                      "confirmed": 0, "confirmation": {"ran": False},
                                      "outcome": "invalid", "error": f"missing submission: {sub}"})
        return 1

    with tempfile.TemporaryDirectory(prefix="apin_harbor_verify_") as tmp:
        work = Path(tmp)
        cand_path = work / "candidate" / "submission.py"
        cand_path.parent.mkdir()
        shutil.copyfile(sub, cand_path)  # the only bytes taken from the agent

        t_cand = time.perf_counter()
        cand = score_file(cand_path, lane, cfg, work / "cand")
        cand_wall = time.perf_counter() - t_cand

        frozen = load_frozen(tests_dir)
        if frozen and frozen.get("lane") not in (None, lane):
            frozen = {}
        dens = {role: denominator(role, lane, files[role], cfg, digest, frozen, work) for role in DENOMINATOR_FILES}

    ref_name = reference_name(files["reference"]) if files["reference"].is_file() else None
    dens["reference"]["name"] = ref_name
    if cand.get("unpaired"):
        cls = {"outcome": "unscorable", "reward": 0.0, "start_speedup": None, "ref_speedup": None,
               "frontier_speedup": None, "gap_closed": None,
               "unscorable_reason": "the candidate passed its confirmation run, but its first run left no "
                                    "usable efficiency to compare with the frozen denominators"}
    else:
        cls = classify(cand["valid"], cand.get("soft_min_eff", 0.0), dens["start"], dens["reference"],
                       dens["frontier"], margin=margin)
    outcome = cls["outcome"]
    payload = {
        **base_payload,
        "outcome": outcome,
        "passed": 1 if cand["valid"] else 0,
        "solved": 1 if outcome == "solved" else 0,
        "improved": 1 if outcome == "improved" else 0,
        "unscorable": 1 if outcome == "unscorable" else 0,
        "ref_speedup": _sig(cls["ref_speedup"]),
        "start_speedup": _sig(cls["start_speedup"]),
        "frontier_speedup": _sig(cls["frontier_speedup"]),
        "gap_closed": _sig(cls["gap_closed"], 4),
        "soft_min_eff": _sig(cand.get("soft_min_eff", 0.0)) if cand["valid"] else 0.0,
        "start_soft_min": _sig(dens["start"].get("soft_min_eff")) if dens["start"].get("valid") else None,
        "reference_soft_min": _sig(dens["reference"].get("soft_min_eff")) if dens["reference"].get("valid") else None,
        "frontier_soft_min": _sig(dens["frontier"].get("soft_min_eff")) if dens["frontier"].get("valid") else None,
        "tier0_ok": cand["tier0_ok"],
        "tier1_ok": cand["tier1_ok"],
        "confirmation": cand.get("confirmation", {"ran": False}),
        "confirmed": 1 if (cand.get("confirmation") or {}).get("ran") and cand["confirmation"].get("passed") else 0,
        "reference": {"name": ref_name, "file": files["reference"].name,
                      "source": dens["reference"].get("source"), "valid": bool(dens["reference"].get("valid"))},
        "denominators": {role: _summary(d) for role, d in dens.items()},
        "candidate_wall_s": round(cand_wall, 1),
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    if "unscorable_reason" in cls:
        payload["unscorable_reason"] = cls["unscorable_reason"]
    for k in ("short_horizon_soft_min", "long_horizon_soft_min"):
        if k in cand:
            payload[k] = _sig(cand[k])
    for k in ("b2_max", "overhead_ratio"):
        if k in cand:
            payload[k] = round(cand[k], 5)
    if "by_class" in cand:
        payload["by_class"] = {k: _sig(v) for k, v in cand["by_class"].items()}
    if "per_repeat_soft_min" in cand:
        payload["per_repeat_soft_min"] = [_sig(v) for v in cand["per_repeat_soft_min"]]
    if not cand["valid"]:
        payload["error"] = cand.get("error")
        payload["failed_targets"] = cand.get("failed_targets", [])
        if not cand["tier0_ok"]:
            payload["tier0_info"] = cand.get("tier0_info")
    _write_rewards(out_dir, float(cls["reward"]), payload, submission=sub)
    return {"invalid": 1, "unscorable": 2}.get(outcome, 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harbor/verify.py")
    sub = ap.add_subparsers(dest="mode", required=True)

    rw = sub.add_parser("rewind")
    rw.add_argument("--task-id", required=True)
    rw.add_argument("--submission", type=Path, required=True)
    rw.add_argument("--tasks", type=Path, default=rewind.DEFAULT_TASKS_PATH)
    rw.add_argument("--out-dir", type=Path, default=None)

    op = sub.add_parser("optimize")
    op.add_argument("--lane", required=True, choices=sorted(LANE_TASKS))
    op.add_argument("--submission", type=Path, required=True)
    op.add_argument("--tests-dir", type=Path, default=None,
                    help="directory holding baseline.py, reference.py, frontier.py and frozen_scores.json "
                         "(default: next to this file, else harbor/tasks/<task>/tests)")
    op.add_argument("--baseline", type=Path, default=None, help="override tests/baseline.py")
    op.add_argument("--reference", type=Path, default=None, help="override tests/reference.py")
    op.add_argument("--frontier", type=Path, default=None, help="override tests/frontier.py")
    op.add_argument("--dims", default=",".join(str(d) for d in VERIFIER_DIMS))
    op.add_argument("--repeats", type=int, default=VERIFIER_REPEATS)
    op.add_argument("--seed", type=int, default=HIDDEN_VERIFIER_SEED)
    op.add_argument("--n-chains", type=int, default=N_CHAINS)
    op.add_argument("--k-steps", type=int, default=K_STEPS)
    op.add_argument("--adapt-draws", type=int, default=ADAPT_DRAWS)
    op.add_argument("--horizon-steps", type=int, default=HORIZON_STEPS)
    op.add_argument("--out-dir", type=Path, default=None)

    a = ap.parse_args(argv)
    out_dir = a.out_dir if a.out_dir is not None else _default_out_dir()
    if a.mode == "rewind":
        return verify_rewind(a.task_id, a.submission, out_dir, a.tasks)
    cfg = scoring_config(
        seed=a.seed, dims=tuple(int(x.strip()) for x in a.dims.split(",") if x.strip()), repeats=a.repeats,
        n_chains=a.n_chains, k_steps=a.k_steps, adapt_draws=a.adapt_draws, horizon_steps=a.horizon_steps,
    )
    overrides = {"start": a.baseline, "reference": a.reference, "frontier": a.frontier}
    return verify_optimize(a.lane, a.submission, out_dir, tests_dir=a.tests_dir, overrides=overrides, cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
