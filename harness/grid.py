"""Run one sampler over models x reps.

Layout of a run directory (the runner output schema, see README):

    runs/<label>/<model>/rep<r>/
        chain_<c>.csv    draws, one per chain (adapter writes these)
        run.json         request, sampler identity and Timing
        DONE | FAILED    marker; FAILED holds the error

A rep with DONE is skipped on rerun, so an interrupted grid resumes.
"""

from __future__ import annotations

import json
import platform
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path

from . import models as models_mod
from .sampler import RunRequest, Sampler

BASE_SEED = 20260819


def seed_for(rep: int, base: int = BASE_SEED) -> int:
    return base + 1000 * rep


def run_grid(sampler: Sampler, models: list[models_mod.Model], reps: int, runs_dir: Path, label: str | None = None,
             chains: int = 4, warmup: int = 1000, draws: int = 1000, log=print, base_seed: int = BASE_SEED
             ) -> list[Path]:
    sampler.check()
    label = label or sampler.name
    identity = sampler.describe()
    done = []
    for model in models:
        data = model.data_file()
        for rep in range(reps):
            d = runs_dir / label / model.name / f"rep{rep}"
            if (d / "DONE").exists():
                done.append(d)
                continue
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            req = RunRequest(model=model, data_file=data, seed=seed_for(rep, base_seed), chains=chains, warmup=warmup,
                             draws=draws, out_dir=d)
            try:
                timing = sampler.run(req)
            except Exception:
                (d / "FAILED").write_text(traceback.format_exc())
                log(f"[{label}] {model.name} rep{rep}: FAILED (see {d / 'FAILED'})")
                continue
            record = {
                "label": label, "sampler": sampler.name, "model": model.name, "rep": rep, "seed": req.seed,
                "chains": chains, "warmup": warmup, "draws": draws, "identity": identity,
                "host": platform.node(), "timing": timing.to_dict(),
            }
            (d / "run.json").write_text(json.dumps(record, indent=1))
            (d / "DONE").touch()
            done.append(d)
            log(f"[{label}] {model.name} rep{rep}: wall {timing.wall_s:.2f}s, "
                f"chains {sum(timing.chain_s):.2f}s, grads {timing.grads_sampling}"
                + (f", evaluations {timing.evaluations}" if timing.evaluations is not None else ""))
    return done
