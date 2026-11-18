"""A user-supplied Python sampler, run against the model's log density through BridgeStan.

The submission is one file defining `sample(problem, budget, rng) -> draws`
(harness/logdensity.py, PROTOCOL.md "The log-density seam", BENCH.md). It runs
in a child process with a hard timeout. The harness counts its evaluations,
enforces the budget, maps its draws to the constrained space and writes the
usual chain CSVs.

The request's warmup is not passed to the submission, which tunes as it likes
inside the budget; it only sets the default evaluation limit.

Settings (option, else environment variable):
  sampler=PATH       the submission file                        (required)
  max_evals=N        evaluation limit, all chains               (default: chains * (warmup + draws) * 1023,
                                                                 the most NUTS may spend at tree depth 10)
  max_seconds=S      wall-time limit of sample()                (default 1800)
  bridgestan=PATH    BridgeStan source tree                     (BRIDGESTAN; default: the bridgestan
                                                                 package's own tree)

Timing: wall_s is the time inside sample(). chain_s is wall_s / chains for
each chain, because the chains share one process. evaluations counts density
plus gradient calls; extra holds the two counts apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .. import stan_library
from ..models import ROOT
from ..sampler import RunRequest, Sampler, Timing

TREEDEPTH_CEILING = 1023
DEFAULT_SECONDS = 1800.0


class Adapter(Sampler):
    name = "logdensity"

    @property
    def submission(self) -> Path:
        raw = self.opt("sampler")
        if not raw:
            raise RuntimeError("no submission: pass --opt sampler=PATH (see examples/samplers/)")
        return Path(raw).expanduser().resolve()

    def check(self) -> None:
        import bridgestan  # noqa: F401  (a dependency; fail early if the environment is broken)

        from ..logdensity import check_imports

        if not self.submission.is_file():
            raise RuntimeError(f"submission not found: {self.submission}")
        check_imports(self.submission.read_text())
        stan_library.tree(self.opt("bridgestan"))

    def describe(self) -> dict:
        src = self.submission.read_bytes()
        return {"submission": self.submission.name, "submission_sha256": hashlib.sha256(src).hexdigest(),
                "bridgestan_tree": stan_library.tree(self.opt("bridgestan")).name,
                "max_evals": self.opt("max_evals"), "max_seconds": self.opt("max_seconds")}

    def budget(self, req: RunRequest) -> tuple[int, float]:
        evals = int(self.opt("max_evals") or req.chains * (req.warmup + req.draws) * TREEDEPTH_CEILING)
        return evals, float(self.opt("max_seconds") or DEFAULT_SECONDS)

    def run(self, req: RunRequest) -> Timing:
        so = stan_library.build(stan_library.tree(self.opt("bridgestan")), req.model.stan_file,
                                self.settings.build_dir)
        max_evals, max_seconds = self.budget(req)
        out = req.out_dir.resolve()
        spec = {"library": str(so), "data": str(Path(req.data_file).resolve()), "submission": str(self.submission),
                "seed": req.seed, "chains": req.chains, "draws": req.draws, "max_evaluations": max_evals,
                "max_seconds": max_seconds, "out_dir": str(out)}
        (out / "spec.json").write_text(json.dumps(spec, indent=1))
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            r = subprocess.run([sys.executable, "-m", "harness.logdensity", str(out / "spec.json")], cwd=ROOT,
                               env=env, capture_output=True, text=True, timeout=max_seconds + 300)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"submission killed after {e.timeout:.0f} s") from e
        (out / "submission.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"submission failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
        res = json.loads((out / "evaluations.json").read_text())
        wall = res["seconds"]
        return Timing(wall_s=wall, chain_s=[wall / req.chains] * req.chains, warmup_s=[None] * req.chains,
                      sampling_s=[None] * req.chains, grads_sampling=None, grads_warmup=None, divergences=None,
                      evaluations=res["evaluations"],
                      extra={"density_evals": res["density_evals"], "gradient_evals": res["gradient_evals"],
                             "max_evaluations": max_evals, "max_seconds": max_seconds,
                             "constrain_s": res["constrain_s"], "dim": res["dim"], "chain_s_is_wall": True})
