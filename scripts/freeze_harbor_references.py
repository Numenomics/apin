#!/usr/bin/env python3
"""Freeze every score that a speed-up is divided by: the lane yardsticks and the benchmark's
hidden denominators.

For each lane this script
  1. copies the lane's current frontier autoresearch/lanes/<LANE>/submission.py to
     harbor/tasks/<task>/tests/frontier.py (skip with --keep-frontier),
  2. scores, with harness/grade.py::score_file at the standard configuration,
       - public: autoresearch/lanes/<LANE>/start.py on the public seeds, written to
         autoresearch/lanes/<LANE>/yardstick.json (the lane loop's denominator, shipped to agents),
       - hidden: tests/baseline.py (the same bytes as start.py), tests/reference.py and
         tests/frontier.py on the hidden verifier seeds of harbor/verify.py, written to
         tests/frozen_scores.json (never shipped to agents),
  3. keys every score by lane, judge digest, scoring config and the file's SHA-256.

A file scored for several lanes with the same seam (A0, A2 and A3 run the whole sampler; A1 and A4
have their own seams) is scored once. Rerun this after any change to a module in the judge digest
(harness/judge.py::judge_digest), to start.py, to a reference, or to a seed. Run
scripts/sync_harbor_tests.py first so tests/baseline.py matches start.py.

    uv run python scripts/freeze_harbor_references.py --workers 4
    uv run python scripts/freeze_harbor_references.py --public-only --lanes A2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import grade, judge  # noqa: E402

LANES_DIR = ROOT / "autoresearch" / "lanes"


def _load_verify():
    spec = importlib.util.spec_from_file_location("apin_harbor_verify", ROOT / "harbor" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apin_harbor_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def _seam(lane: str) -> str:
    """Lanes whose files are scored identically (judge.resolve_lane_submission, ipc)."""
    return lane if lane in ("A1", "A4") else "full"


def _score(args: tuple[str, str, dict, str]) -> tuple[tuple, dict]:
    lane, path, cfg, work = args
    return (args[0], args[1], json.dumps(cfg, sort_keys=True)), grade.score_file(Path(path), lane, cfg, Path(work))


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
    ap.add_argument("--workers", type=int, default=4)
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--public-only", action="store_true", help="only the lane yardsticks")
    which.add_argument("--hidden-only", action="store_true", help="only the benchmark's hidden denominators")
    ap.add_argument("--keep-frontier", action="store_true", help="do not refresh tests/frontier.py")
    ap.add_argument("--work-dir", type=Path, default=None, help="scratch directory (default: a temp dir)")
    a = ap.parse_args(argv)

    verify = _load_verify()
    hidden_cfg, public_cfg = verify.scoring_config(), grade.config()
    if hidden_cfg["seed"] == public_cfg["seed"] or hidden_cfg["confirmation_seed"] == public_cfg["confirmation_seed"]:
        raise SystemExit("the hidden verifier seeds must differ from the public seeds")
    digest = judge.judge_digest()
    lanes = [x.strip().upper() for x in a.lanes.split(",") if x.strip()]
    tmp = tempfile.TemporaryDirectory(prefix="apin_freeze_") if a.work_dir is None else None
    work = Path(tmp.name) if tmp else a.work_dir

    # (lane, role, path, cfg) to freeze; role "yardstick" is the public score of start.py.
    wanted: list[tuple[str, str, Path, dict]] = []
    frontier_src: dict[str, dict] = {}
    for lane in lanes:
        start = LANES_DIR / lane / "start.py"
        tests = ROOT / "harbor" / "tasks" / verify.LANE_TASKS[lane] / "tests"
        if not a.hidden_only:
            wanted.append((lane, "yardstick", start, public_cfg))
        if a.public_only:
            continue
        if (tests / verify.DENOMINATOR_FILES["start"]).read_bytes() != start.read_bytes():
            raise SystemExit(f"{tests / 'baseline.py'} differs from {start}: run scripts/sync_harbor_tests.py")
        src = LANES_DIR / lane / "submission.py"
        if not a.keep_frontier:
            shutil.copyfile(src, tests / verify.DENOMINATOR_FILES["frontier"])
        frontier_src[lane] = {"source": str(src.relative_to(ROOT)), "git_commit": _git_commit(src),
                              "copied_from_source": not a.keep_frontier}
        for role, name in verify.DENOMINATOR_FILES.items():
            wanted.append((lane, role, tests / name, hidden_cfg))

    jobs: dict[tuple, tuple] = {}
    for lane, _role, path, cfg in wanted:
        key = (_seam(lane), grade.sha256(path), json.dumps(cfg, sort_keys=True))
        jobs.setdefault(key, (lane, str(path), cfg, str(work / f"job{len(jobs):02d}")))
    print(f"judge digest {digest}; {len(wanted)} scores from {len(jobs)} grader runs with {a.workers} workers",
          flush=True)

    scores: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(_score, job): key for key, job in jobs.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            _, score = fut.result()
            scores[key] = score
            lane, path = jobs[key][0], Path(jobs[key][1])
            print(f"{lane} {path.relative_to(ROOT)} seed={json.loads(key[2])['seed']}: valid={score['valid']} "
                  f"soft_min={score.get('soft_min_eff')} {score.get('wall_s')}s {score.get('error') or ''}", flush=True)
    if judge.judge_digest() != digest:
        raise SystemExit("the judge digest changed during the run; rerun")

    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    hidden: dict[str, dict] = {lane: {} for lane in lanes}
    for lane, role, path, cfg in wanted:
        score = scores[(_seam(lane), grade.sha256(path), json.dumps(cfg, sort_keys=True))]
        if role == "yardstick":
            entry = grade.yardstick_entry(lane, path, cfg, digest, score)
            entry["frozen_at"] = stamp
            out = LANES_DIR / lane / "yardstick.json"
            out.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)} ({entry['id']})")
        else:
            hidden[lane][role] = grade.frozen_entry(lane, role, path, cfg, digest, score)
    for lane in lanes if not a.public_only else []:
        tests = ROOT / "harbor" / "tasks" / verify.LANE_TASKS[lane] / "tests"
        ref_name = verify.reference_name(tests / verify.DENOMINATOR_FILES["reference"])
        hidden[lane]["reference"]["name"] = ref_name
        hidden[lane]["frontier"].update(frontier_src[lane])
        doc = {
            "schema": verify.FROZEN_SCHEMA, "lane": lane, "task": verify.LANE_TASKS[lane],
            "judge_digest": digest, "config": hidden_cfg, "frozen_at": stamp, "reference_name": ref_name,
            "scores": hidden[lane],
        }
        (tests / verify.FROZEN_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {(tests / verify.FROZEN_FILE).relative_to(ROOT)}")
    print(f"done in {time.perf_counter() - t0:.0f}s")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
