#!/usr/bin/env python3
"""Harbor verifier for the apin agent benchmark: the grader (harness/grade.py) on hidden seeds.

harbor/verify.py is the single source. scripts/sync_harbor_tests.py copies it verbatim to every
harbor/tasks/*/tests/verify.py. It writes Harbor-standard reward.txt, reward.json (numbers only)
and details.json to /logs/verifier (or --out-dir).

The candidate is graded with harness/grade.py::score_file at the standard configuration, on the
hidden verifier seed instead of the public one. Its soft-min efficiency is divided by three frozen
denominators read from tests/frozen_scores.json, all scored the same way: the starting code
(tests/baseline.py), the Stan-style reference (tests/reference.py) and the lane's frontier at
freeze time (tests/frontier.py). A denominator whose frozen key (lane, judge digest, scoring config,
file SHA-256) does not match is scored live instead.

    start_speedup = candidate / starting code     (the leaderboard number)
    reward = ref_speedup = candidate / reference  (0 if the candidate fails a check)

Isolation: the candidate file is the only input taken from the agent. It is copied into a private
temporary directory before scoring. The harness is imported from the image (/app), and the
denominator files and frozen scores come from the tests directory, which Harbor uploads only to the
verifier container. Run it with `python -I` (tests/test.sh does) so no environment variable or
working-directory module can change what is imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _find_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "harness" / "grade.py").is_file():
            return parent
    if Path("/app/harness/grade.py").is_file():
        return Path("/app")
    return Path.cwd()


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import grade, judge, sandbox

# Hidden seeds, rotated every release. A seed is public once its release ships (it is in this
# file). During a run the agent cannot read it: tests/ reaches only the verifier container, after
# the agent phase, and this file never enters the agent image. Both must stay disjoint from the
# public seeds in harness/grade.py.
# Retired: 88421077 (bench-v0.1), 39389868 and confirmation 78210474 (bench-v0.2).
# Drawn for bench-v0.3 and never shipped, because the A0 starting code got stuck on hybrid
# Rosenbrock d=20 there and an invalid starting code leaves no ratio: 43547104 / 64244275,
# 33259997 / 10459241, 28495072 / 91346270.
HIDDEN_VERIFIER_SEED = 72432996
HIDDEN_CONFIRMATION_SEED = 75713498
IMPROVEMENT_MARGIN = grade.IMPROVEMENT_MARGIN

# Lane -> Harbor task directory.
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
FROZEN_SCHEMA = 2

OUTCOMES = ("invalid", "unscorable", "no_gain", "improved", "solved")


def scoring_config(**overrides) -> dict:
    """The grader's standard configuration on the hidden seeds. Part of every frozen key."""
    overrides.setdefault("seed", HIDDEN_VERIFIER_SEED)
    overrides.setdefault("confirmation_seed", HIDDEN_CONFIRMATION_SEED)
    return grade.config(**overrides)


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
    sha = grade.sha256(path)
    entry = (frozen.get("scores") or {}).get(role)
    if grade.frozen_match(entry, lane, role, sha, cfg, digest):
        out = dict(entry)
        out["source"] = "frozen"
        return out
    print(f"[verify] {role}: no matching frozen score; scoring {path.name} live", file=sys.stderr, flush=True)
    out = grade.frozen_entry(lane, role, path, cfg, digest, grade.score_file(path, lane, cfg, work_dir / role))
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
    s, r, f = grade.soft_min(start), grade.soft_min(reference), grade.soft_min(frontier)
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
        cand = grade.score_file(cand_path, lane, cfg, work / "cand")
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


_CONFINEMENT_PROBE = (
    "import sys; from pathlib import Path; from harness import sandbox; "
    "sandbox.confine_worker(Path(sys.argv[1]), mode='strict')"
)


def require_confinement() -> None:
    """Stop before any grading when this machine cannot confine agent code.

    Without this probe the failure would surface inside the candidate's load check and read as
    an invalid submission with reward 0. A verifier that cannot confine is a setup failure, so
    it exits without writing a reward and Harbor records an error, not a result."""
    res = subprocess.run([sys.executable, "-c", _CONFINEMENT_PROBE, str(Path(__file__).resolve())],
                         capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    if res.returncode != 0:
        lines = [ln for ln in res.stderr.strip().splitlines() if ln.strip()]
        raise SystemExit("verifier setup failure, no reward written: agent code cannot be confined here "
                         f"({lines[-1] if lines else f'exit {res.returncode}'})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harbor/verify.py")
    sub = ap.add_subparsers(dest="mode", required=True)

    op = sub.add_parser("optimize")
    op.add_argument("--lane", required=True, choices=sorted(LANE_TASKS))
    op.add_argument("--submission", type=Path, required=True)
    op.add_argument("--tests-dir", type=Path, default=None,
                    help="directory holding baseline.py, reference.py, frontier.py and frozen_scores.json "
                         "(default: next to this file, else harbor/tasks/<task>/tests)")
    op.add_argument("--baseline", type=Path, default=None, help="override tests/baseline.py")
    op.add_argument("--reference", type=Path, default=None, help="override tests/reference.py")
    op.add_argument("--frontier", type=Path, default=None, help="override tests/frontier.py")
    op.add_argument("--dims", default=",".join(str(d) for d in grade.DIMS))
    op.add_argument("--repeats", type=int, default=grade.REPEATS)
    op.add_argument("--seed", type=int, default=HIDDEN_VERIFIER_SEED)
    op.add_argument("--confirmation-seed", type=int, default=HIDDEN_CONFIRMATION_SEED)
    op.add_argument("--n-chains", type=int, default=grade.N_CHAINS)
    op.add_argument("--k-steps", type=int, default=grade.K_STEPS)
    op.add_argument("--adapt-draws", type=int, default=grade.ADAPT_DRAWS)
    op.add_argument("--horizon-steps", type=int, default=grade.HORIZON_STEPS)
    op.add_argument("--out-dir", type=Path, default=None)

    a = ap.parse_args(argv)
    out_dir = a.out_dir if a.out_dir is not None else _default_out_dir()
    cfg = scoring_config(
        seed=a.seed, confirmation_seed=a.confirmation_seed,
        dims=tuple(int(x.strip()) for x in a.dims.split(",") if x.strip()), repeats=a.repeats,
        n_chains=a.n_chains, k_steps=a.k_steps, adapt_draws=a.adapt_draws, horizon_steps=a.horizon_steps,
    )
    overrides = {"start": a.baseline, "reference": a.reference, "frontier": a.frontier}
    # The verifier holds the hidden seeds and the frozen scores, and it writes the reward. Agent
    # code must not run here unless both confinement layers are active, so a layer that fails to
    # install fails the run instead of being skipped.
    os.environ[sandbox.ENV_SANDBOX] = "strict"
    require_confinement()
    return verify_optimize(a.lane, a.submission, out_dir, tests_dir=a.tests_dir, overrides=overrides, cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
