#!/usr/bin/env python3
"""Freeze the verifier's denominators for every Harbor optimize task.

For each lane this script
  1. copies the lane's current incumbent autoresearch/lanes/<LANE>/submission.py to
     harbor/tasks/<task>/tests/frontier.py (skip with --keep-frontier),
  2. scores tests/baseline.py (starting code), tests/reference.py (Stan-style reference) and
     tests/frontier.py with harbor/verify.py::score_file at the verifier's seed, dims, repeats and
     budgets, and
  3. writes tests/frozen_scores.json, keyed by lane, judge digest, scoring config and each file's
     SHA-256. The verifier uses a frozen score only when every part of the key matches, and scores
     the file live otherwise, so a stale file costs time, never correctness.

Rerun it whenever harness/ (the judge digest), a denominator file, or a scoring constant in
harbor/verify.py changes, and after rotating HIDDEN_VERIFIER_SEED.

    uv run python scripts/freeze_harbor_references.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_verify():
    spec = importlib.util.spec_from_file_location("apin_harbor_verify", ROOT / "harbor" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apin_harbor_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def _score(args: tuple[str, str, str, dict, str]) -> tuple[str, str, dict]:
    lane, role, path, cfg, work = args
    verify = _load_verify()
    digest = verify.judge.judge_digest()
    score = verify.score_file(Path(path), lane, cfg, Path(work))
    return lane, role, verify.frozen_entry(lane, role, Path(path), cfg, digest, score)


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path)], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", default="A0,A1,A2,A3,A4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep-frontier", action="store_true", help="do not refresh tests/frontier.py")
    ap.add_argument("--work-dir", type=Path, default=None, help="scratch directory (default: a temp dir)")
    a = ap.parse_args(argv)

    verify = _load_verify()
    cfg = verify.scoring_config()
    digest = verify.judge.judge_digest()
    lanes = [x.strip().upper() for x in a.lanes.split(",") if x.strip()]
    import tempfile
    tmp = tempfile.TemporaryDirectory(prefix="apin_freeze_") if a.work_dir is None else None
    work = Path(tmp.name) if tmp else a.work_dir

    frontier_src: dict[str, dict] = {}
    jobs = []
    for lane in lanes:
        tests = ROOT / "harbor" / "tasks" / verify.LANE_TASKS[lane] / "tests"
        src = ROOT / "autoresearch" / "lanes" / lane / "submission.py"
        dest = tests / verify.DENOMINATOR_FILES["frontier"]
        if not a.keep_frontier:
            shutil.copyfile(src, dest)
        frontier_src[lane] = {"source": str(src.relative_to(ROOT)), "git_commit": _git_commit(src),
                              "copied_from_source": not a.keep_frontier}
        for role, name in verify.DENOMINATOR_FILES.items():
            jobs.append((lane, role, str(tests / name), cfg, str(work / lane / role)))

    print(f"judge digest {digest}; config {cfg}; {len(jobs)} files with {a.workers} workers", flush=True)
    results: dict[str, dict[str, dict]] = {lane: {} for lane in lanes}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for fut in as_completed([ex.submit(_score, j) for j in jobs]):
            lane, role, entry = fut.result()
            if entry["judge_digest"] != digest:
                raise SystemExit(f"judge digest changed during the run ({entry['judge_digest']} != {digest}); rerun")
            results[lane][role] = entry
            print(f"{lane} {role}: valid={entry['valid']} soft_min={entry.get('soft_min_eff')} "
                  f"{entry.get('wall_s')}s {entry.get('error') or ''}", flush=True)

    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    for lane in lanes:
        tests = ROOT / "harbor" / "tasks" / verify.LANE_TASKS[lane] / "tests"
        ref_name = verify.reference_name(tests / verify.DENOMINATOR_FILES["reference"])
        results[lane]["reference"]["name"] = ref_name
        results[lane]["frontier"].update(frontier_src[lane])
        doc = {
            "schema": verify.FROZEN_SCHEMA, "lane": lane, "task": verify.LANE_TASKS[lane],
            "judge_digest": digest, "config": cfg, "frozen_at": stamp, "reference_name": ref_name,
            "scores": results[lane],
        }
        (tests / verify.FROZEN_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {(tests / verify.FROZEN_FILE).relative_to(ROOT)}")
    print(f"done in {time.perf_counter() - t0:.0f}s")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
