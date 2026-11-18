"""The core model set and its data.

Models live in models/<name>.stan and are listed in models/manifest.json.
Data files whose licence allows it are shipped in data/. The rest are fetched
from posteriordb at a pinned commit, checked against a sha256 and cached in
.cache/data/ (see data/SOURCES.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTERIORDB_COMMIT = "28f8d3d6e975315f42aa274a8399f21e07a43b30"
POSTERIORDB_RAW = "https://raw.githubusercontent.com/stan-dev/posteriordb/{commit}/posterior_database/data/data/{name}.json.zip"


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    posterior: str
    data_name: str
    data_sha256: str
    data_shipped: bool
    reference: str = "none"  # "posteriordb", "local" or "none" (harness/reference.py)
    reference_stan: str | None = None  # a local reference is built from this file under models/ if set
    bench: str = ""  # "public" or "held-out" (BENCH.md)
    chains: int = 4  # the benchmark budget of one rep (BENCH.md, "The budget")
    warmup: int = 1000
    draws: int = 1000  # per chain; raised per model where NUTS cannot be proven right with fewer

    @property
    def stan_file(self) -> Path:
        return ROOT / "models" / f"{self.name}.stan"

    @property
    def reference_stan_file(self) -> Path:
        """The model a local reference is sampled from: the model itself, or an equivalent
        reparameterisation with the same posterior and the same parameter names."""
        return ROOT / "models" / self.reference_stan if self.reference_stan else self.stan_file

    def data_file(self, fetch: bool = True) -> Path:
        """Path to the JSON data, fetching it first when it is not shipped."""
        if self.data_shipped:
            return ROOT / "data" / f"{self.data_name}.json"
        path = cache_dir() / f"{self.data_name}.json"
        if not path.is_file():
            if not fetch:
                raise FileNotFoundError(f"{path} missing; run `apin fetch-data`")
            _fetch(self.data_name, self.data_sha256, path)
        return path


def cache_dir() -> Path:
    return Path(os.environ.get("APIN_DATA_CACHE", ROOT / ".cache" / "data"))


def _fetch(name: str, sha256: str, dest: Path) -> None:
    local = ROOT / "upstream" / "posteriordb" / "posterior_database" / "data" / "data" / f"{name}.json.zip"
    if local.is_file():  # submodule checked out: no network needed
        blob = local.read_bytes()
    else:
        url = POSTERIORDB_RAW.format(commit=POSTERIORDB_COMMIT, name=name)
        with urllib.request.urlopen(url, timeout=60) as resp:
            blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        payload = z.read(f"{name}.json")
    got = hashlib.sha256(payload).hexdigest()
    if got != sha256:
        raise RuntimeError(f"{name}: sha256 {got} does not match the manifest ({sha256})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)


def load() -> dict[str, Model]:
    entries = json.loads((ROOT / "models" / "manifest.json").read_text())
    return {
        e["model"]: Model(e["model"], e["family"], e["posterior"], e["data"], e["data_sha256"], e["data_shipped"],
                          e.get("reference", {}).get("source", "none"), e.get("reference", {}).get("stan"),
                          e.get("bench", ""), **e.get("budget", {}))
        for e in entries
    }


def select(spec: str | None) -> list[Model]:
    """'all', a family name, a benchmark split ('public' or 'held-out'), or a comma list of model names."""
    allm = load()
    if not spec or spec == "all":
        return list(allm.values())
    families = {m.family for m in allm.values()}
    if spec in families:
        return [m for m in allm.values() if m.family == spec]
    if spec in ("public", "held-out"):
        return [m for m in allm.values() if m.bench == spec]
    out = []
    for name in spec.split(","):
        name = name.strip()
        if name not in allm:
            raise KeyError(f"unknown model {name!r}; `apin list` shows the set")
        out.append(allm[name])
    return out
