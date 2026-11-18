"""walnutpie's stan_cli (flatironinstitute/walnutpie, built from upstream/walnutpie).

stan_cli samples one chain per process from a BridgeStan model library, so
this adapter builds the .so with BridgeStan and starts one process per chain.
Chain c gets seed + c.

Settings (option, else environment variable):
  cli=PATH         stan_cli binary            (WALNUTPIE_CLI; default
                                               upstream/walnutpie/build/examples/stan_cli)
  bridgestan=PATH  BridgeStan source tree     (BRIDGESTAN)
  flags=...        extra stan_cli flags, space separated (e.g. "--metric-window 50")
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from .. import cmdstan
from ..models import ROOT
from ..sampler import RunRequest, Sampler, Timing

STANZA = re.compile(
    r"total time: ([\d.eE+-]+)s?\s*\n"
    r"logp_grad time: ([\d.eE+-]+)s?\s*\n"
    r"logp_grad fraction: ([\d.eE+-]+)\s*\n"
    r"\s*logp_grad calls: (\d+)\s*\n"
    r"\s*time per call: ([\d.eE+-]+)s?\s*\n"
)


def parse_stanzas(text: str) -> list[dict]:
    """stan_cli prints a five-line timing stanza twice: warmup, then sampling."""
    keys = ("total_s", "grad_s", "grad_fraction", "grad_calls", "s_per_grad")
    return [{k: (int(v) if k == "grad_calls" else float(v)) for k, v in zip(keys, m.groups())}
            for m in STANZA.finditer(text)]


class Adapter(Sampler):
    name = "walnutpie"

    @property
    def cli(self) -> Path:
        raw = self.opt("cli") or os.environ.get("WALNUTPIE_CLI") or ROOT / "upstream/walnutpie/build/examples/stan_cli"
        return Path(raw).expanduser().resolve()

    @property
    def bridgestan(self) -> Path:
        raw = self.opt("bridgestan") or os.environ.get("BRIDGESTAN")
        if not raw:
            raise RuntimeError("no BridgeStan tree: pass --opt bridgestan=PATH or set BRIDGESTAN")
        return Path(raw).expanduser().resolve()

    def check(self) -> None:
        if not self.cli.is_file():
            raise RuntimeError(f"stan_cli not found at {self.cli}; build upstream/walnutpie (PROTOCOL.md)")
        if not (self.bridgestan / "Makefile").is_file():
            raise RuntimeError(f"{self.bridgestan} is not a BridgeStan source tree")

    def describe(self) -> dict:
        return {"cli": str(self.cli), "bridgestan": str(self.bridgestan), "flags": self.opt("flags", "")}

    def build(self, stan_file: Path) -> Path:
        """Build <name>_model.so with BridgeStan's makefile directly, keyed by
        source and BridgeStan tree (bridgestan.compile_model caches by path
        only and can hand back a stale library)."""
        src = stan_file.read_bytes()
        key = hashlib.sha256(src + b"\0" + str(self.bridgestan).encode()).hexdigest()[:16]
        bdir = (self.settings.build_dir / f"bs-{stan_file.stem}-{key}").resolve()
        so = bdir / f"{stan_file.stem}_model.so"
        if so.is_file():
            return so
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / stan_file.name).write_bytes(src)
        r = subprocess.run(["make", "-C", str(self.bridgestan), str(so)], capture_output=True, text=True,
                           env={**os.environ, "MAKEFLAGS": "-j4"})
        if r.returncode != 0 or not so.is_file():
            raise RuntimeError(f"BridgeStan build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return so

    def argv(self, so: Path, req: RunRequest, c: int) -> list[str]:
        return [str(self.cli), str(so), str(req.data_file), "--seed", str(req.seed + c),
                "--warmup", str(req.warmup), "--samples", str(req.draws),
                "--output", str(req.out_dir / f"chain_{c}.csv")] + (self.opt("flags") or "").split()

    def run(self, req: RunRequest) -> Timing:
        self.check()
        so = self.build(req.model.stan_file)
        logs = [req.out_dir / f"chain_{c}.log" for c in range(req.chains)]

        def argv(c):
            # stan_cli prints its timing stanzas to stdout; send them to the chain log.
            return ["sh", "-c", 'exec "$@" > "$0" 2>&1', str(logs[c]), *self.argv(so, req, c)]

        wall = cmdstan.run_chains(argv, req.chains)
        warm, samp, chain, gw, gs = [], [], [], 0, 0
        for log in logs:
            blocks = parse_stanzas(log.read_text())
            if len(blocks) < 2:
                raise RuntimeError(f"{log}: expected two timing stanzas, found {len(blocks)}")
            w, s = blocks[0], blocks[1]
            warm.append(w["total_s"])
            samp.append(s["total_s"])
            chain.append(w["total_s"] + s["total_s"])
            gw += w["grad_calls"]
            gs += s["grad_calls"]
        return Timing(wall_s=wall, chain_s=chain, warmup_s=warm, sampling_s=samp,
                      grads_sampling=gs, grads_warmup=gw, divergences=None)
