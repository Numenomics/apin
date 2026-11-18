"""Build a Stan model into a BridgeStan shared library, cached by source and tree.

Used by the walnutpie and logdensity adapters. The BridgeStan source tree is
the first of: an explicit path, $BRIDGESTAN, and the tree the `bridgestan`
Python package keeps for its own version (it downloads that tree on first
use). bridgestan.compile_model caches by path only and can hand back a stale
library, so this calls BridgeStan's makefile directly with its own cache key.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


def tree(path: str | os.PathLike | None = None, fallback_to_package: bool = True) -> Path:
    raw = path or os.environ.get("BRIDGESTAN")
    if raw:
        root = Path(raw).expanduser().resolve()
    elif fallback_to_package:
        from bridgestan.compile import get_bridgestan_path

        root = Path(get_bridgestan_path()).resolve()
    else:
        raise RuntimeError("no BridgeStan tree: pass --opt bridgestan=PATH or set BRIDGESTAN")
    if not (root / "Makefile").is_file():
        raise RuntimeError(f"{root} is not a BridgeStan source tree")
    return root


def build(root: Path, stan_file: Path, build_dir: Path, make_args: list[str] | None = None) -> Path:
    """Return <build_dir>/bs-<model>-<key>/<model>_model.so, building it if needed."""
    src = stan_file.read_bytes()
    make_args = list(make_args or [])
    key = hashlib.sha256(b"\0".join([src, str(root).encode(), " ".join(make_args).encode()])).hexdigest()[:16]
    bdir = (build_dir / f"bs-{stan_file.stem}-{key}").resolve()
    so = bdir / f"{stan_file.stem}_model.so"
    if so.is_file():
        return so
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / stan_file.name).write_bytes(src)
    r = subprocess.run(["make", "-C", str(root), *make_args, str(so)], capture_output=True, text=True,
                       env={**os.environ, "MAKEFLAGS": "-j4"})
    if r.returncode != 0 or not so.is_file():
        raise RuntimeError(f"BridgeStan build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return so
