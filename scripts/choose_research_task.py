#!/usr/bin/env python3
"""Pick an open autoresearch task card from autoresearch/task_catalog.json.

Inspired by CrocSwap/integer-mult-bounds' task picker: routes donated-inference and
multi-agent runs to distinct lanes/bottlenecks without manual coordination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "autoresearch" / "task_catalog.json"


def load_tasks(lane: str | None = None) -> list[dict]:
    data = json.loads(CATALOG.read_text())
    tasks = list(data.get("tasks") or [])
    if lane:
        tasks = [t for t in tasks if t.get("lane", "").upper() == lane.upper()]
    return tasks


def pick_task(tasks: list[dict], seed: str | None = None) -> dict:
    if not tasks:
        raise SystemExit("no matching tasks in autoresearch/task_catalog.json")
    weights = [max(1, int(t.get("priority", 1))) for t in tasks]
    rng = random.Random(
        int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16) if seed is not None else None
    )
    return rng.choices(tasks, weights=weights, k=1)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Choose an autoresearch task card for apin.")
    ap.add_argument("--lane", help="Filter by lane (A0, A1, A2, A3)")
    ap.add_argument("--list", action="store_true", help="List all matching task cards")
    ap.add_argument("--seed", help="Deterministic selection seed (e.g. agent name or run_id)")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    tasks = load_tasks(args.lane)
    if args.list:
        if args.json:
            print(json.dumps(tasks, indent=2))
            return 0
        for t in sorted(tasks, key=lambda x: (-int(x.get("priority", 0)), x["id"])):
            print(f"[p={t['priority']:2d}] {t['id']:30s} lane={t['lane']}  {t['title']}")
        return 0

    chosen = pick_task(tasks, seed=args.seed)
    if args.json:
        print(json.dumps(chosen, indent=2))
        return 0
    print(f"Task: {chosen['id']} (lane {chosen['lane']}, priority {chosen['priority']})")
    print(f"Title: {chosen['title']}")
    print(f"Starting file: {chosen['starting_file']}")
    if chosen.get("codex_refs"):
        print(f"Codex refs: {', '.join(chosen['codex_refs'])}")
    print(f"Goal: {chosen['goal']}")
    print(f"Bottlenecks: {'; '.join(chosen.get('bottlenecks') or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
