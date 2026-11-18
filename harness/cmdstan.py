"""Find, build and run CmdStan, and read its output files.

The CmdStan tree comes from `--cmdstan`, else the CMDSTAN environment
variable. It can be a release install or the patched build in
`upstream/cmdstan` (see PROTOCOL.md).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class CmdStanError(RuntimeError):
    pass


def locate(path: str | os.PathLike | None = None) -> Path:
    raw = path or os.environ.get("CMDSTAN")
    if not raw:
        raise CmdStanError("no CmdStan tree: pass --cmdstan or set CMDSTAN")
    root = Path(raw).expanduser().resolve()
    if not (root / "makefile").is_file():
        raise CmdStanError(f"{root} has no makefile; it is not a CmdStan tree")
    if not (root / "bin" / "stanc").is_file():
        raise CmdStanError(f"{root}/bin/stanc is missing; run `make build` in {root} first")
    return root


def version(root: Path) -> str:
    out = subprocess.run([str(root / "bin" / "stanc"), "--version"], capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip()


def compile_model(
    root: Path,
    stan_file: Path,
    build_root: Path,
    stancflags: str = "",
    cxxflags: str = "",
    jobs: int = 4,
) -> Path:
    """Build an executable for stan_file. Builds are cached under build_root,
    keyed by model source, flags and CmdStan tree, so a patched tree never
    reuses a binary from another tree."""
    src = stan_file.read_bytes()
    key = hashlib.sha256(
        b"\0".join([src, stancflags.encode(), cxxflags.encode(), str(root).encode(), version(root).encode()])
    ).hexdigest()[:16]
    bdir = (build_root / f"{stan_file.stem}-{key}").resolve()
    exe = bdir / stan_file.stem
    if exe.is_file():
        return exe
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / stan_file.name).write_bytes(src)
    env = dict(os.environ)
    env["MAKEFLAGS"] = f"-j{jobs}"
    cmd = ["make", "-C", str(root), str(exe)]
    if stancflags:
        cmd.append(f"STANCFLAGS={stancflags}")
    if cxxflags:
        env["CXXFLAGS"] = f"{cxxflags} {env.get('CXXFLAGS', '')}".strip()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not exe.is_file():
        raise CmdStanError(f"build of {stan_file.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return exe


def parameter_names(root: Path, stan_file: Path) -> list[str]:
    """Names of the variables in the parameters block, from `stanc --info`."""
    import json

    r = subprocess.run([str(root / "bin" / "stanc"), "--info", str(stan_file)], capture_output=True, text=True)
    if r.returncode != 0:
        raise CmdStanError(f"stanc --info failed: {r.stderr[:500]}")
    return list(json.loads(r.stdout).get("parameters", {}))


@dataclass
class StanCsv:
    columns: list[str]
    values: np.ndarray  # (rows, columns), post-warmup draws only
    warmup_s: float | None = None
    sampling_s: float | None = None
    total_s: float | None = None
    comments: list[str] = field(default_factory=list)
    warmup_values: np.ndarray | None = None  # rows written before "Adaptation terminated" (save_warmup=1)

    def column(self, name: str) -> np.ndarray | None:
        try:
            return self.values[:, self.columns.index(name)]
        except ValueError:
            return None

    def warmup_column(self, name: str) -> np.ndarray | None:
        if self.warmup_values is None or name not in self.columns:
            return None
        return self.warmup_values[:, self.columns.index(name)]

    @property
    def config(self) -> dict[str, str]:
        """The `key = value` lines of the CmdStan header, without "(Default)"; the first of a repeated key."""
        out = {}
        for c in self.comments:
            k, eq, v = c.lstrip("# ").partition(" = ")
            if eq:
                out.setdefault(k.strip(), v.replace("(Default)", "").strip())
        return out


_ELAPSED = re.compile(r"#\s*Elapsed Time:\s*([\d.eE+-]+) seconds \(Warm-up\)")
_SAMPLING = re.compile(r"#\s*([\d.eE+-]+) seconds \(Sampling\)")
_TOTAL = re.compile(r"#\s*([\d.eE+-]+) seconds \(Total\)")


def read_csv(path: Path, columns: list[str] | None = None, data: bytes | None = None) -> StanCsv:
    """Read a CmdStan output CSV (or any CSV with a header row and '#' comments).

    With save_warmup=1, CmdStan writes the warmup draws first and then a
    comment "# Adaptation terminated". Rows before that comment go to
    `warmup_values`; `values` holds only the draws after it. A file without
    the comment (save_warmup=0, or not from CmdStan) has no warmup rows. A row without a line end
    or with a different number of fields than the header raises CmdStanError: the file was cut short.
    With `columns`, only those of them in the file are kept, in file order. With `data`, those
    bytes are parsed as the file's content and the file is not opened (the scorer hashes the
    bytes it parses).
    """
    comments, header, rows, warm, keep = [], None, [], None, None
    with (open(path) if data is None else io.TextIOWrapper(io.BytesIO(data))) as fh:
        for line in fh:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
                if warm is None and "Adaptation terminated" in line:
                    warm, rows = rows, []
                continue
            if not line.strip():
                continue
            if not line.endswith("\n"):
                raise CmdStanError(f"{path.name}: the last row is cut short (no line end)")
            fields = line.strip().split(",")
            if header is None:
                header = fields
                wanted = set(header if columns is None else columns)
                keep = [i for i, h in enumerate(header) if h in wanted]
            elif len(fields) != len(header):
                raise CmdStanError(f"{path.name}: a row has {len(fields)} fields, the header {len(header)}")
            else:
                rows.append(np.array([fields[i] for i in keep], dtype=float))
    if header is None:
        raise CmdStanError(f"{path} has no header row")
    def mat(r):
        return np.array(r, dtype=float).reshape(len(r), len(keep))

    out = StanCsv([header[i] for i in keep], mat(rows), comments=comments, warmup_values=mat(warm) if warm is not None else None)
    for c in comments:
        if m := _ELAPSED.match(c):
            out.warmup_s = float(m.group(1))
        elif m := _SAMPLING.match(c):
            out.sampling_s = float(m.group(1))
        elif m := _TOTAL.match(c):
            out.total_s = float(m.group(1))
    return out


def run_chains(argv_for_chain, chains: int) -> float:
    """Start one process per chain at once and wait for all. argv_for_chain(c)
    gives chain c's command. Returns the wall time of the whole batch."""
    t0 = time.perf_counter()
    procs = [subprocess.Popen(argv_for_chain(c), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
             for c in range(chains)]
    errors = []
    for c, p in enumerate(procs):
        _, err = p.communicate()
        if p.returncode != 0:
            errors.append(f"chain {c} exited {p.returncode}: {err[-1500:]}")
    wall = time.perf_counter() - t0
    if errors:
        raise CmdStanError("\n".join(errors))
    return wall


def sample_argv(exe: Path, data: Path | None, seed: int, chain: int, out_csv: Path, warmup: int, draws: int,
                init: Path | None = None, extra: list[str] | None = None) -> list[str]:
    """CmdStan NUTS command for one chain. All chains share the seed and differ
    by id, so their RNG streams are distinct and reproducible. Warmup draws are
    saved so that their gradient count is known; `read_csv` keeps them apart."""
    argv = [str(exe), f"id={chain + 1}"]
    if data is not None:
        argv += ["data", f"file={data}"]
    argv += ["random", f"seed={seed}"]
    if init is not None:
        argv += [f"init={init}"]
    argv += ["output", f"file={out_csv}", "method=sample", f"num_warmup={warmup}", f"num_samples={draws}",
             "save_warmup=1"]
    return argv + list(extra or [])
