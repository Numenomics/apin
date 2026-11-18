"""CmdStan NUTS.

Options:
  stancflags=...   passed to stanc (e.g. --O1)
  cxxflags=...     prepended to CXXFLAGS for the model build
  args=...         extra sample arguments, space separated (e.g. "adapt delta=0.95")
"""

from __future__ import annotations

import re

from .. import cmdstan
from ..sampler import RunRequest, Sampler, Timing


class Adapter(Sampler):
    name = "cmdstan"

    @property
    def root(self):
        return cmdstan.locate(self.settings.cmdstan)

    def check(self) -> None:
        self.root

    def describe(self) -> dict:
        return {"cmdstan": str(self.root), "stanc": cmdstan.version(self.root),
                "stancflags": self.opt("stancflags", ""), "cxxflags": self.opt("cxxflags", ""),
                "args": self.opt("args", "")}

    def build(self, req: RunRequest):
        return cmdstan.compile_model(self.root, req.model.stan_file, self.settings.build_dir,
                                     self.opt("stancflags", ""), self.opt("cxxflags", ""))

    def run(self, req: RunRequest, inits: list | None = None) -> Timing:
        exe = self.build(req)
        extra = (self.opt("args") or "").split()
        csvs = [req.out_dir / f"chain_{c}.csv" for c in range(req.chains)]
        wall = cmdstan.run_chains(
            lambda c: cmdstan.sample_argv(exe, req.data_file, req.seed, c, csvs[c], req.warmup, req.draws,
                                          init=inits[c] if inits else None, extra=extra),
            req.chains,
        )
        return timing_from_csvs(csvs, wall)


def timing_from_csvs(csvs, wall: float) -> Timing:
    """Times and leapfrog counts from CmdStan's CSVs. The count is complete
    only when every transition was written (thin = 1) with its n_leapfrog__;
    otherwise `uncounted` says why and the run earns no per-evaluation score."""
    warm, samp, chain, grads, grads_w, div, uncounted = [], [], [], 0, 0, 0, []
    for path in csvs:
        f = cmdstan.read_csv(path)
        warm.append(f.warmup_s)
        samp.append(f.sampling_s)
        chain.append((f.warmup_s or 0.0) + (f.sampling_s or 0.0))
        thin = _thin(f.comments)
        if thin != 1:
            uncounted.append(f"{path.name}: thin = {thin}, so most transitions are not written")
        # read_csv keeps the warmup rows (save_warmup=1) out of `values`.
        lf = f.column("n_leapfrog__")
        if lf is None:
            uncounted.append(f"{path.name}: no n_leapfrog__ column")
            grads = grads_w = None
        if grads is not None:
            grads += int(lf.sum())
        if grads_w is not None and f.warmup_values is not None:
            grads_w += int(f.warmup_values[:, f.columns.index("n_leapfrog__")].sum())
        else:
            grads_w = None  # warmup not saved: unknown
        d = f.column("divergent__")
        div += int(d.sum()) if d is not None else 0
    return Timing(wall_s=wall, chain_s=chain, warmup_s=warm, sampling_s=samp,
                  grads_sampling=grads, grads_warmup=grads_w, divergences=div, uncounted="; ".join(uncounted))


def _thin(comments: list[str]) -> int:
    for c in comments:
        m = re.match(r"#\s*thin\s*=\s*(\d+)", c)
        if m:
            return int(m.group(1))
    return 1
