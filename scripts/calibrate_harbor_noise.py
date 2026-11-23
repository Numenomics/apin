#!/usr/bin/env python3
"""A/A noise calibration for the Harbor optimize verifier.

For every optimize lane, scores the starting code (tests/baseline.py) and N copies of it that
differ only in random numbers: copy k draws k extra uniforms from `rng` at the top of its adapt
entry point (adapt / adapt_metric / adapt_transport) and, in lanes A0, A2 and A3, of transition(),
so warmup and sampling follow different random paths while the algorithm is unchanged. Optionally also scores tests/reference.py and
tests/frontier.py. Every file is scored once at --max-repeats; the score at R repeats is the
geometric mean of the first R per-repeat soft-mins, which is exactly what the verifier computes at
repeats=R (each repeat is an independent, separately seeded suite).

Output: one JSON per scored file under --out-dir (so an interrupted run resumes), plus
summary.json with the A/A ratio table (copy soft-min / original soft-min) per lane and repeats.

    uv run python scripts/calibrate_harbor_noise.py --copies 8 --workers 6 \
        --out-dir harbor/calibration
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINTS = ("adapt", "adapt_metric", "adapt_transport")


def _load_verify():
    spec = importlib.util.spec_from_file_location("apin_harbor_verify", ROOT / "harbor" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apin_harbor_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def rng_copy(source: str, k: int) -> str:
    """Insert `rng.random(k)` as the first statement (after the docstring) of the adapt entry point
    and of transition() when the file defines one (lanes A0, A2, A3). Lanes A1 and A4 use the
    judge's transition, whose per-step random numbers no submission can change."""
    tree = ast.parse(source)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (*ENTRY_POINTS, "transition")]
    if not any(f.name in ENTRY_POINTS for f in fns):
        raise ValueError("no adapt entry point found")
    lines = source.splitlines(keepends=True)
    for fn in sorted(fns, key=lambda f: f.lineno, reverse=True):
        rng_arg = fn.args.args[-1].arg
        first = fn.body[0]
        has_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        insert_after = first.end_lineno if has_doc else first.lineno - 1
        anchor = fn.body[1] if has_doc and len(fn.body) > 1 else first
        stmt = f"{' ' * anchor.col_offset}{rng_arg}.random({int(k)})  # A/A copy {k}: RNG offset only\n"
        lines.insert(insert_after, stmt)
    return "".join(lines)


def _job(args: tuple[str, str, str, dict, str]) -> dict:
    lane, label, path, cfg, out_json = args
    verify = _load_verify()
    t0 = time.perf_counter()
    work = Path(out_json).with_suffix(".work")
    score = verify.score_file(Path(path), lane, cfg, work)
    rec = {"lane": lane, "label": label, "file": path, "config": cfg, "wall_s": round(time.perf_counter() - t0, 1),
           **{k: score.get(k) for k in ("valid", "tier0_ok", "tier1_ok", "soft_min_eff", "per_repeat_soft_min",
                                         "per_repeat_by_class", "by_class", "error", "failed_targets")}}
    Path(out_json).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def _geo(vals: list[float]) -> float:
    vals = [max(float(v), 1e-18) for v in vals]
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def _prefix(rec: dict, R: int) -> float | None:
    """Soft-min at R repeats, or None if the file fails a check within the first R repeats (a
    failed repeat has soft-min 0; a Tier 0 failure has no repeats at all)."""
    vals = rec.get("per_repeat_soft_min") or []
    if not rec.get("tier0_ok") or len(vals) < R or min(vals[:R]) <= 0.0:
        return None
    return _geo(vals[:R])


def summarize(recs: list[dict], repeats_list: tuple[int, ...]) -> dict:
    out: dict = {"lanes": {}, "pooled": {}}
    pooled: dict[int, list[float]] = {r: [] for r in repeats_list}
    pooled_invalid: dict[int, list[int]] = {r: [0, 0] for r in repeats_list}
    seen_starts: dict[tuple, str] = {}
    for lane in sorted({r["lane"] for r in recs}):
        lane_recs = {r["label"]: r for r in recs if r["lane"] == lane}
        orig = lane_recs.get("start")
        # Lanes A2 and A3 start from the same code (only the docstring differs) and score
        # identically; count such a lane's copies once in the pooled numbers.
        key = tuple(orig.get("per_repeat_soft_min") or []) if orig else ()
        duplicate_of = seen_starts.get(key) if key else None
        if key and duplicate_of is None:
            seen_starts[key] = lane
        copies = {k: v for k, v in lane_recs.items() if k.startswith("copy")}
        lane_out: dict = {"files": {}, "duplicate_of": duplicate_of}
        for label, r in sorted(lane_recs.items()):
            lane_out["files"][label] = {
                "valid_r_max": r["valid"], "wall_s": r["wall_s"], "error": r.get("error"),
                **{f"soft_min_r{R}": _prefix(r, R) for R in repeats_list},
            }
        for R in repeats_list:
            base = _prefix(orig, R) if orig else None
            vals = {k: _prefix(r, R) for k, r in copies.items()}
            n_invalid = sum(v is None for v in vals.values())
            if duplicate_of is None:
                pooled_invalid[R][0] += n_invalid
                pooled_invalid[R][1] += len(vals)
            row: dict = {"copies": len(vals), "invalid_copies": n_invalid, "start_valid": base is not None}
            ratios = [v / base for v in vals.values() if v is not None] if base else []
            if ratios:
                arr = np.asarray(ratios)
                row.update({
                    "n": len(ratios), "min": float(arr.min()), "median": float(np.median(arr)),
                    "max": float(arr.max()), "p95": float(np.percentile(arr, 95)),
                    "sd_log": float(np.std(np.log(arr), ddof=1)) if len(arr) > 1 else None,
                    "ratios": [round(float(x), 4) for x in ratios],
                })
                if duplicate_of is None:
                    pooled[R].extend(ratios)
            lane_out[f"r{R}"] = row
        out["lanes"][lane] = lane_out
    for R, vals in pooled.items():
        row = {"invalid_copies": pooled_invalid[R][0], "copies": pooled_invalid[R][1]}
        if vals:
            arr = np.asarray(vals)
            row.update({
                "n": len(vals), "min": float(arr.min()), "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)), "max": float(arr.max()),
                "sd_log": float(np.std(np.log(arr), ddof=1)) if len(arr) > 1 else None,
            })
        out["pooled"][f"r{R}"] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", default="A0,A1,A2,A3,A4")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--max-repeats", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--extra", default="reference,frontier",
                    help="also score these tests/ files (comma list of reference, frontier; empty for none)")
    ap.add_argument("--seed", type=int, default=None, help="default: the verifier seed")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "harbor" / "calibration")
    a = ap.parse_args(argv)

    verify = _load_verify()
    cfg = verify.scoring_config(repeats=a.max_repeats, **({"seed": a.seed} if a.seed is not None else {}))
    out_dir = a.out_dir
    copies_dir = out_dir / "copies"
    copies_dir.mkdir(parents=True, exist_ok=True)
    extras = [x for x in a.extra.split(",") if x]
    jobs = []
    for lane in [x.strip().upper() for x in a.lanes.split(",") if x.strip()]:
        tests = ROOT / "harbor" / "tasks" / verify.LANE_TASKS[lane] / "tests"
        start = tests / verify.DENOMINATOR_FILES["start"]
        files = [("start", start)]
        src = start.read_text(encoding="utf-8")
        for k in range(1, a.copies + 1):
            p = copies_dir / f"{lane}_copy{k:02d}.py"
            p.write_text(rng_copy(src, k), encoding="utf-8")
            files.append((f"copy{k:02d}", p))
        for role in extras:
            p = tests / verify.DENOMINATOR_FILES[role]
            if p.is_file():
                files.append((role, p))
        for label, p in files:
            out_json = out_dir / f"{lane}_{label}.json"
            if out_json.is_file():
                prev = json.loads(out_json.read_text())
                if prev.get("config") == cfg:
                    continue
            jobs.append((lane, label, str(p), cfg, str(out_json)))

    print(f"{len(jobs)} files to score with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            lane, label = futs[fut][0], futs[fut][1]
            try:
                r = fut.result()
                print(f"{lane} {label}: valid={r['valid']} soft_min={r.get('soft_min_eff')} {r['wall_s']}s", flush=True)
            except Exception as e:  # noqa: BLE001 - keep the other jobs going
                print(f"{lane} {label}: FAILED {type(e).__name__}: {e}", flush=True)

    recs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("A*_*.json"))]
    recs = [r for r in recs if r.get("config") == cfg]
    repeats_list = tuple(R for R in (1, 2, 4, 8, 16) if R <= a.max_repeats)
    summary = {"config": cfg, "copies": a.copies, **summarize(recs, repeats_list)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for R in repeats_list:
        p = summary["pooled"].get(f"r{R}") or {}
        if "p95" in p:
            print(f"repeats={R}: pooled A/A ratio min {p['min']:.3f} median {p['median']:.3f} "
                  f"p95 {p['p95']:.3f} max {p['max']:.3f} (n={p['n']}); "
                  f"{p['invalid_copies']}/{p['copies']} copies invalid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
