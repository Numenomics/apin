#!/usr/bin/env python3
"""Stage a sealed build context for the Harbor benchmark base image and build it.

The base image holds only what the five Harbor tasks need to run `apin autoresearch eval` and
`apin invariance` offline: the harness package (with the grader and its public seeds), the lane
documents the instructions name, each lane's starting code and public yardstick, the data the
harness loads, and the locked dependencies. Everything else in the repository stays out, in
particular evolved lane submissions, example samplers, tests, results, the verifier with its
hidden seeds, the hidden frozen scores, and .git.

Every lane's `submission.py` in the image is that lane's frozen starting code,
autoresearch/lanes/<LANE>/start.py, which is also in the image next to its yardstick.json. Task
images (FROM the base) then copy their own starting file.

    uv run python scripts/build_harbor_base.py            # stage + build base image
    uv run python scripts/build_harbor_base.py --tasks    # also build the 5 task images
    uv run python scripts/build_harbor_base.py --stage-only build/harbor-context
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "harbor" / "tasks"
BASE_DOCKERFILE = ROOT / "harbor" / "base.Dockerfile"

RELEASE = "bench-v0.3.0"
BASE_IMAGE = "apin-harbor-base"

# Copied verbatim (files or whole directories), relative to the repository root.
ALLOWLIST = (
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    "harness",
    "models",
    "data",
    "codex",
    "autoresearch/RULINGS.md",
)
# Copied for every lane that has a benchmark task: its documents, its starting code and its
# public yardstick (the lane loop divides by it). Never its hidden frozen scores.
LANE_FILES = ("program.md", "NEGATIVE_MEMORY.md", "start.py", "yardstick.json")
# Never copied, even from inside an allowlisted directory.
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "archive", "incumbent.py", ".DS_Store")


def task_configs() -> dict[str, dict]:
    """Parsed task.toml of every Harbor task, keyed by task directory name."""
    return {
        p.parent.name: tomllib.loads(p.read_text())
        for p in sorted(TASKS_DIR.glob("*/task.toml"))
    }


def task_lanes() -> list[str]:
    """Lanes that have a benchmark task."""
    lanes = [cfg["metadata"]["lane"] for cfg in task_configs().values()]
    if len(set(lanes)) != len(lanes):
        raise SystemExit("two tasks claim the same lane")
    return sorted(lanes)


def stage(dest: Path) -> Path:
    """Write a clean build context to `dest` (replaced if it exists) and return it."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in ALLOWLIST:
        src = ROOT / rel
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=IGNORE)
        else:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
    for lane in task_lanes():
        lane_src = ROOT / "autoresearch" / "lanes" / lane
        lane_dst = dest / "autoresearch" / "lanes" / lane
        lane_dst.mkdir(parents=True)
        for name in LANE_FILES:
            if (lane_src / name).is_file():
                shutil.copy2(lane_src / name, lane_dst / name)
        shutil.copy2(lane_src / "start.py", lane_dst / "submission.py")
    shutil.copy2(BASE_DOCKERFILE, dest / "Dockerfile")
    return dest


def _docker(*args: str) -> None:
    print("+ docker", " ".join(args), flush=True)
    subprocess.run(["docker", *args], check=True)


def build_base(context: Path) -> str:
    tag = f"{BASE_IMAGE}:{RELEASE}"
    _docker("build", "--build-arg", f"APIN_BENCH_RELEASE={RELEASE}",
            "-t", tag, "-t", f"{BASE_IMAGE}:latest", str(context))
    return tag


def build_tasks() -> list[str]:
    tags = []
    for name in task_configs():
        tag = f"apin-harbor-task-{name}:{RELEASE}"
        _docker("build", "-t", tag, str(TASKS_DIR / name / "environment"))
        tags.append(tag)
    return tags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context", type=Path, default=ROOT / "build" / "harbor-context",
                    help="where to stage the build context (default: build/harbor-context)")
    ap.add_argument("--stage-only", action="store_true", help="stage the context, do not build")
    ap.add_argument("--tasks", action="store_true",
                    help=f"also build every task image as apin-harbor-task-<task>:{RELEASE}")
    a = ap.parse_args(argv)
    ctx = stage(a.context)
    print(f"staged {sum(1 for p in ctx.rglob('*') if p.is_file())} files in {ctx}")
    if a.stage_only:
        return 0
    print(f"built {build_base(ctx)}")
    if a.tasks:
        for tag in build_tasks():
            print(f"built {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
