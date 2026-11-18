"""CmdStan NUTS.

Options:
  stancflags=...   passed to stanc (e.g. --O1)
  cxxflags=...     prepended to CXXFLAGS for the model build
  args=...         extra sample arguments, space separated (e.g. "adapt delta=0.95")
"""

from __future__ import annotations

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
    warm, samp, chain, grads, div = [], [], [], 0, 0
    for path in csvs:
        f = cmdstan.read_csv(path)
        warm.append(f.warmup_s)
        samp.append(f.sampling_s)
        chain.append((f.warmup_s or 0.0) + (f.sampling_s or 0.0))
        # save_warmup=0, so every row is a sampling draw.
        lf = f.column("n_leapfrog__")
        grads += int(lf.sum()) if lf is not None else 0
        d = f.column("divergent__")
        div += int(d.sum()) if d is not None else 0
    return Timing(wall_s=wall, chain_s=chain, warmup_s=warm, sampling_s=samp,
                  grads_sampling=grads, grads_warmup=None, divergences=div)
