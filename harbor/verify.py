#!/usr/bin/env python3
"""Canonical Harbor verifier for apin Agent Benchmark tasks.

Writes Harbor-standard `/logs/verifier/reward.txt` and `/logs/verifier/reward.json`:
  - `rewind` mode: binary reward (`1.0` if hidden invariance/Jacobian check passes, else `0.0`).
  - `optimize` mode: continuous speedup reward (`cand_soft_min / base_soft_min` on hidden
    held-out verifier seeds across paired dual-horizon suites; `0.0` on any Tier 0/1 failure).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import autoresearch, rewind  # noqa: E402

# Hidden verifier seed offset (disjoint from default autoresearch dev seed 20261120)
HIDDEN_VERIFIER_SEED = 88421077


def _write_rewards(out_dir: Path, reward: float, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_reward = float(reward) if (math.isfinite(reward) and reward > 0.0) else 0.0
    out = {"reward": round(clean_reward, 6), **payload}
    (out_dir / "reward.txt").write_text(f"{clean_reward:.6f}\n", encoding="utf-8")
    (out_dir / "reward.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


def verify_rewind(task_id: str, submission: Path, out_dir: Path, tasks_path: Path) -> int:
    tasks = {t.task_id: t for t in rewind.load_tasks(tasks_path)}
    if task_id not in tasks:
        _write_rewards(out_dir, 0.0, {"passed": 0, "error": f"unknown task_id {task_id!r}"})
        return 1
    res = rewind.evaluate_rewind_task(tasks[task_id], submission)
    passed = 1 if res["passed"] else 0
    _write_rewards(out_dir, float(passed), {"passed": passed, **res})
    return 0 if passed else 1


def verify_optimize(
    lane: str,
    submission: Path,
    baseline: Path,
    out_dir: Path,
    *,
    dims: tuple[int, ...] = (6, 20),
    repeats: int = 2,
    seed: int = HIDDEN_VERIFIER_SEED,
    n_chains: int = 48,
    k_steps: int = 16,
    adapt_draws: int = 100,
    horizon_steps: int = 500,
) -> int:
    sub = Path(submission).resolve()
    base = Path(baseline).resolve()
    if not sub.is_file():
        _write_rewards(out_dir, 0.0, {"passed": 0, "error": f"missing submission: {sub}"})
        return 1

    with tempfile.TemporaryDirectory(prefix="apin_harbor_verify_") as tmp:
        work_dir = Path(tmp)
        t0_ok, t0_info = autoresearch._eval_tier0(
            sub, seed=seed, work_dir=work_dir / "t0", lane=lane, horizon_steps=horizon_steps,
        )
        if not t0_ok:
            _write_rewards(
                out_dir, 0.0,
                {"passed": 0, "tier0_ok": False, "tier0_info": t0_info, "lane": lane},
            )
            return 1

        suites = [
            (seed + r * 7919, autoresearch._build_tier1_suite(dims[0], seed + r * 7919, dims=dims))
            for r in range(max(1, repeats))
        ]
        cand = autoresearch._eval_tier1_multi(
            sub, suites, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
        )
        if not cand["passed"]:
            _write_rewards(
                out_dir, 0.0,
                {
                    "passed": 0,
                    "tier0_ok": True,
                    "tier1_ok": False,
                    "lane": lane,
                    "b2_max": round(float(cand["b2_max"]), 5),
                    "by_class": cand["by_class"],
                },
            )
            return 1

        base_res = autoresearch._eval_tier1_multi(
            base, suites, n_chains=n_chains, k_steps=k_steps,
            adapt_draws=adapt_draws, lane=lane, horizon_steps=horizon_steps,
        )
        base_sm = float(base_res["soft_min_eff"]) if base_res["passed"] else 0.0
        cand_sm = float(cand["soft_min_eff"])
        speedup = (cand_sm / base_sm) if base_sm > 0.0 else (1.0 if cand_sm > 0.0 else 0.0)
        _write_rewards(
            out_dir,
            speedup,
            {
                "passed": 1,
                "tier0_ok": True,
                "tier1_ok": True,
                "lane": lane,
                "dims": list(dims),
                "repeats": repeats,
                "speedup": round(speedup, 4),
                "soft_min_eff": round(cand_sm, 6),
                "baseline_soft_min": round(base_sm, 6),
                "short_horizon_soft_min": round(float(cand["short_horizon_soft_min"]), 6),
                "long_horizon_soft_min": round(float(cand["long_horizon_soft_min"]), 6),
                "b2_max": round(float(cand["b2_max"]), 5),
                "overhead_ratio": round(float(cand["overhead_ratio"]), 3),
                "by_class": {k: round(float(v), 6) for k, v in cand["by_class"].items()},
            },
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harbor/verify.py")
    sub = ap.add_subparsers(dest="mode", required=True)

    rw = sub.add_parser("rewind")
    rw.add_argument("--task-id", required=True)
    rw.add_argument("--submission", type=Path, required=True)
    rw.add_argument("--tasks", type=Path, default=rewind.DEFAULT_TASKS_PATH)
    rw.add_argument("--out-dir", type=Path, default=Path("/logs/verifier"))

    op = sub.add_parser("optimize")
    op.add_argument("--lane", required=True)
    op.add_argument("--submission", type=Path, required=True)
    op.add_argument("--baseline", type=Path, required=True)
    op.add_argument("--dims", default="6,20")
    op.add_argument("--repeats", type=int, default=2)
    op.add_argument("--seed", type=int, default=HIDDEN_VERIFIER_SEED)
    op.add_argument("--n-chains", type=int, default=48)
    op.add_argument("--k-steps", type=int, default=16)
    op.add_argument("--adapt-draws", type=int, default=100)
    op.add_argument("--horizon-steps", type=int, default=500)
    op.add_argument("--out-dir", type=Path, default=Path("/logs/verifier"))

    a = ap.parse_args(argv)
    if a.mode == "rewind":
        return verify_rewind(a.task_id, a.submission, a.out_dir, a.tasks)
    dims = tuple(int(x.strip()) for x in a.dims.split(",") if x.strip())
    return verify_optimize(
        a.lane, a.submission, a.baseline, a.out_dir,
        dims=dims, repeats=a.repeats, seed=a.seed,
        n_chains=a.n_chains, k_steps=a.k_steps,
        adapt_draws=a.adapt_draws, horizon_steps=a.horizon_steps,
    )


if __name__ == "__main__":
    sys.exit(main())
