"""Run one sampler over models x reps.

Layout of a run directory (the runner output schema, see README):

    runs/<label>/<model>/rep<r>/
        chain_<c>.csv    draws, one per chain (adapter writes these)
        run.json         request, sampler identity and Timing
        DONE | FAILED    marker; FAILED holds the error

A rep with DONE is reused on rerun, so an interrupted grid resumes, but only when its seed, base
seed, sampler identity (for a submission, its sha256) and budget match the request; otherwise it is
run again (BENCH.md, rule 4).
"""

from __future__ import annotations

import json
import platform
import shutil
import traceback
from pathlib import Path

from . import models as models_mod
from .sampler import RunRequest, Sampler

BASE_SEED = 20260819
TREEDEPTH_CEILING = 1023  # leapfrog steps in a full NUTS tree of depth 10 (CmdStan's max_depth); the cap is a choice
MAX_SECONDS = 1800.0  # wall time of one rep


def max_evaluations(chains: int, warmup: int, draws: int) -> int:
    """The evaluation limit of one rep: every transition at the tree-depth ceiling (BENCH.md, "The budget")."""
    return chains * (warmup + draws) * TREEDEPTH_CEILING


def seed_for(rep: int, base: int = BASE_SEED) -> int:
    return base + 1000 * rep


def run_grid(sampler: Sampler, models: list[models_mod.Model], reps: int, runs_dir: Path, label: str | None = None,
             chains: int | None = None, warmup: int | None = None, draws: int | None = None, log=print,
             base_seed: int = BASE_SEED) -> list[Path]:
    """chains, warmup and draws default to each model's budget in models/manifest.json."""
    sampler.check()
    label = label or sampler.name
    identity = sampler.describe()
    done = []
    for model in models:
        data = model.data_file()
        budget = {"chains": chains or model.chains, "warmup": model.warmup if warmup is None else warmup,
                  "draws": draws or model.draws}
        for rep in range(reps):
            d = runs_dir / label / model.name / f"rep{rep}"
            key = {"seed": seed_for(rep, base_seed), "base_seed": base_seed, "identity": identity, **budget}
            if (d / "DONE").exists():
                old = json.loads((d / "run.json").read_text())
                changed = [k for k in key if old.get(k) != key[k]]
                if not changed:
                    done.append(d)
                    continue
                log(f"[{label}] {model.name} rep{rep}: rerun, {', '.join(changed)} changed")
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            req = RunRequest(model=model, data_file=data, seed=key["seed"], out_dir=d, **budget)
            try:
                timing = sampler.run(req)
            except Exception:
                (d / "FAILED").write_text(traceback.format_exc())
                log(f"[{label}] {model.name} rep{rep}: FAILED (see {d / 'FAILED'})")
                continue
            record = {"label": label, "sampler": sampler.name, "model": model.name, "rep": rep, **key,
                      "host": platform.node(), "timing": timing.to_dict()}
            (d / "run.json").write_text(json.dumps(record, indent=1))
            (d / "DONE").touch()
            done.append(d)
            log(f"[{label}] {model.name} rep{rep}: wall {timing.wall_s:.2f}s, "
                f"chains {sum(timing.chain_s):.2f}s, grads {timing.grads_sampling}"
                + (f", evaluations {timing.evaluations}" if timing.evaluations is not None else ""))
    return done
