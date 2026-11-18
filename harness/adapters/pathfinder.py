"""CmdStan NUTS started from Pathfinder draws.

Runs CmdStan's multi-path Pathfinder once per run, picks one approximate draw
per chain, and starts each NUTS chain there. Wall time includes Pathfinder.

The seed for Pathfinder and for the draw choice is the run seed, so each rep
gets its own inits. The original campaign ran Pathfinder once per model at
seed 20260819 and reused those inits for every rep (PROTOCOL.md).

Options: as for the cmdstan adapter, plus
  pathfinder_args=...   extra pathfinder arguments (e.g. "num_paths=8")
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import time

from .. import cmdstan
from ..sampler import RunRequest, Timing
from .cmdstan import Adapter as CmdStanAdapter


class Adapter(CmdStanAdapter):
    name = "pathfinder"

    def run(self, req: RunRequest) -> Timing:
        exe = self.build(req)
        pf_csv = req.out_dir / "pathfinder.csv"
        argv = [str(exe), "data", f"file={req.data_file}", "random", f"seed={req.seed}",
                "output", f"file={pf_csv}", "method=pathfinder"] + (self.opt("pathfinder_args") or "").split()
        t0 = time.perf_counter()
        r = subprocess.run(argv, capture_output=True, text=True)
        pf_s = time.perf_counter() - t0
        if r.returncode != 0:
            raise cmdstan.CmdStanError(f"pathfinder failed: {r.stderr[-1500:]}")
        params = cmdstan.parameter_names(self.root, req.model.stan_file)
        inits = pathfinder_inits(cmdstan.read_csv(pf_csv), params, req.seed, req.chains, req.out_dir)
        timing = super().run(req, inits=inits)
        timing.wall_s += pf_s
        timing.extra["pathfinder_s"] = pf_s
        return timing


def pathfinder_inits(pf: cmdstan.StanCsv, params: list[str], seed: int, chains: int, out_dir) -> list:
    lp = pf.column("lp__")
    rows = [i for i in range(pf.values.shape[0]) if lp is None or math.isfinite(lp[i])]
    if not rows:
        raise cmdstan.CmdStanError("pathfinder returned no finite draws")
    paths = []
    for c in range(chains):
        row = pf.values[random.Random(f"{seed}-{c}").choice(rows)]
        init = unflatten(dict(zip(pf.columns, row)), params)
        path = out_dir / f"init_{c}.json"
        path.write_text(json.dumps(init))
        paths.append(path)
    return paths


def unflatten(flat: dict[str, float], names: list[str]) -> dict:
    """Turn CmdStan's flat columns (a, b.1, c.2.3) back into nested JSON values
    for the named variables."""
    out = {}
    for name in names:
        if name in flat:
            out[name] = flat[name]
            continue
        cells = {}
        prefix = name + "."
        for col, v in flat.items():
            if col.startswith(prefix):
                cells[tuple(int(i) for i in col[len(prefix):].split("."))] = v
        if not cells:
            continue  # zero-size parameter
        dims = [max(ix[k] for ix in cells) for k in range(len(next(iter(cells))))]
        out[name] = _nest(cells, dims, ())
    return out


def _nest(cells, dims, prefix):
    if len(prefix) == len(dims):
        return cells[prefix]
    return [_nest(cells, dims, prefix + (i,)) for i in range(1, dims[len(prefix)] + 1)]
