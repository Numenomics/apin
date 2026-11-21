"""Lightweight FIFO file-locked evaluation queue for concurrent autoresearch agents.

Serializes CPU-heavy `apin autoresearch eval` and `apin run` executions on a single host so
parallel agents can think, read, and edit concurrently without overlapping benchmark runs or
distorting wall-clock / overhead measurements.

Bypass with `enabled=False`, `--no-queue`, or `APIN_NO_QUEUE=1` (also bypassed automatically
under `PYTEST_CURRENT_TEST` unless `APIN_FORCE_QUEUE=1` is set).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import ROOT

DEFAULT_LOCK_PATH = ROOT / ".apin_eval_queue.lock"
DEFAULT_STATUS_PATH = ROOT / ".apin_eval_queue.json"


def queue_enabled(flag_enabled: bool = True) -> bool:
    if not flag_enabled:
        return False
    if os.environ.get("APIN_NO_QUEUE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("APIN_FORCE_QUEUE", "") != "1":
        return False
    return True


def _read_holder(status_path: Path) -> str:
    try:
        info = json.loads(status_path.read_text())
        return f"pid={info.get('pid')} job={info.get('job')} since={info.get('started_at')}"
    except Exception:
        return "another active apin job"


@contextlib.contextmanager
def eval_slot(
    job: str,
    *,
    enabled: bool = True,
    lock_path: Path | None = None,
    status_path: Path | None = None,
) -> Iterator[float]:
    """Acquire an exclusive evaluation slot and yield the seconds waited."""
    if not queue_enabled(enabled):
        yield 0.0
        return

    lpath = Path(os.environ.get("APIN_QUEUE_LOCK") or (lock_path or DEFAULT_LOCK_PATH))
    spath = status_path or lpath.with_suffix(".json")
    lpath.parent.mkdir(parents=True, exist_ok=True)

    waited = 0.0
    fh = open(lpath, "a+")
    try:
        t0 = time.perf_counter()
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_holder(spath)
            print(f"[apin-queue] waiting for eval slot ({holder})...", file=sys.stderr, flush=True)
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            waited = time.perf_counter() - t0
            print(f"[apin-queue] acquired eval slot for {job} after {waited:.1f}s", file=sys.stderr, flush=True)

        try:
            spath.write_text(json.dumps({
                "pid": os.getpid(),
                "job": job,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }))
        except OSError:
            pass

        yield waited
    finally:
        try:
            if spath.is_file():
                info = json.loads(spath.read_text())
                if info.get("pid") == os.getpid():
                    spath.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
