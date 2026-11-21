"""Standalone CmdStan multi-path Pathfinder (no MCMC / no Markov-chain exactness guarantee).

Runs CmdStan's `method=pathfinder` directly for each chain slot (`id = 1..chains`,
each producing `num_psis_draws = draws` draws from `num_paths` independent L-BFGS
paths with PSIS resampling). Exists as a fast approximate-inference reference point
for how the 0.1-SD correctness gate and calibration test behave on a method without
Markov chain invariance guarantees.

CmdStan does not report total Pathfinder gradient evaluations in the CSV, so runs are
marked `uncounted` (no per-evaluation speed; `ess_per_s` is still reported when a
model passes the gate).

Options (via --opt key=value):
  num_paths=4          number of independent L-BFGS paths per chain (default: CmdStan's 4)
  pathfinder_args=...  extra CmdStan pathfinder subarguments
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import cmdstan
from ..sampler import RunRequest, Timing
from .cmdstan import Adapter as CmdStanAdapter

_TOTAL_RE = re.compile(r"#\s*([\d.eE+-]+)\s+seconds\s+\(Total\)")
_PF_RE = re.compile(r"#\s*Elapsed Time:\s*([\d.eE+-]+)\s+seconds\s+\(Pathfinders\)")
_PSIS_RE = re.compile(r"#\s*([\d.eE+-]+)\s+seconds\s+\(PSIS\)")


def parse_pathfinder_time(csv_path: Path) -> tuple[float | None, float | None, float | None]:
    """Return (pathfinders_s, psis_s, total_s) from CmdStan's Pathfinder CSV footer."""
    text = csv_path.read_text(errors="replace")
    pf = _PF_RE.search(text)
    psis = _PSIS_RE.search(text)
    tot = _TOTAL_RE.search(text)
    return (
        float(pf.group(1)) if pf else None,
        float(psis.group(1)) if psis else None,
        float(tot.group(1)) if tot else None,
    )


class Adapter(CmdStanAdapter):
    name = "pathfinder_only"

    def describe(self) -> dict:
        base = super().describe()
        base.update({
            "method": "pathfinder_only",
            "num_paths": self.opt("num_paths", "4"),
            "pathfinder_args": self.opt("pathfinder_args", ""),
        })
        return base

    def run(self, req: RunRequest) -> Timing:
        exe = self.build(req)
        num_paths = self.opt("num_paths", "4")
        extra_args = (self.opt("pathfinder_args") or "").split()
        # Ensure each path draws enough samples so PSIS resampling covers req.draws cleanly
        num_draws = max(1000, int(req.draws))

        def argv(c: int) -> list[str]:
            return [
                str(exe),
                f"id={c + 1}",
                "data", f"file={req.data_file}",
                "random", f"seed={req.seed}",
                "output", f"file={req.out_dir / f'chain_{c}.csv'}",
                "method=pathfinder",
                f"num_paths={num_paths}",
                f"num_draws={num_draws}",
                f"num_psis_draws={req.draws}",
                *extra_args,
            ]

        wall = cmdstan.run_chains(argv, req.chains)
        chain_s: list[float] = []
        pf_s: list[float | None] = []
        psis_s: list[float | None] = []
        for c in range(req.chains):
            p_t, s_t, tot_t = parse_pathfinder_time(req.out_dir / f"chain_{c}.csv")
            pf_s.append(p_t)
            psis_s.append(s_t)
            chain_s.append(tot_t if tot_t is not None else wall / req.chains)
        return Timing(
            wall_s=wall,
            chain_s=chain_s,
            warmup_s=pf_s,
            sampling_s=psis_s,
            grads_sampling=None,
            grads_warmup=None,
            divergences=None,
            uncounted="standalone pathfinder does not report total gradient evaluations in CSV",
            extra={"method": "pathfinder_only", "num_paths": int(num_paths)},
        )
