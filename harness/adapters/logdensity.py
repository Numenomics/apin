"""Adapter: run a Python submission through the Slice 2 adapt/transition kernel ABI.

For each independent run c in 0 .. chains-1, the adapter executes three isolated child
processes with OMP_NUM_THREADS=1 under a hard wall-clock deadline:
  1. adapt   -> validates (tuning, chain0, betas, aux_spec) and serializes canonical bundle
  2. sample  -> brand-new clean process loads tuning, runs transition() with per-step RNGs,
                checks per-step finite provenance and frozen-tuning SHA-256
  3. replay  -> brand-new clean process replays ~20 steps + a short contiguous window unbilled
                and asserts bit-exact output equality

Draws are recorded as (runs, walkers, draws, dim) in unconstrained.npz, and cold walkers
(beta == 1.0) are mapped to constrained space and written as chain_<c>.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import grid, logdensity, stan_library
from ..models import ROOT
from ..sampler import RunRequest, Sampler, Timing


def _run_phase(phase: str, spec: dict, spec_path: Path, env: dict[str, str], timeout_s: float) -> str:
    spec_path.write_text(json.dumps(spec, indent=1))
    try:
        r = subprocess.run(
            [sys.executable, "-m", "harness.logdensity", phase, str(spec_path)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=max(1.0, timeout_s),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"submission {phase} killed after {e.timeout:.1f} s") from e
    log = f"=== {phase} ===\n{r.stdout}{r.stderr}\n"
    if r.returncode != 0:
        raise RuntimeError(f"submission {phase} failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
    return log


class Adapter(Sampler):
    name = "logdensity"

    @property
    def submission(self) -> Path:
        raw = self.opt("sampler")
        if not raw:
            raise RuntimeError("no submission: pass --opt sampler=PATH (see examples/samplers/)")
        return Path(raw).expanduser().resolve()

    def capabilities(self) -> list[str]:
        raw = self.opt("capabilities") or ""
        return sorted({c.strip() for c in raw.split(",") if c.strip()})

    def check(self) -> None:
        import bridgestan  # noqa: F401

        if not self.submission.is_file():
            raise RuntimeError(f"submission not found: {self.submission}")
        logdensity.check_imports(self.submission.read_text())
        stan_library.tree(self.opt("bridgestan"))

    def describe(self) -> dict:
        src = self.submission.read_bytes()
        return {
            "abi": "kernel-v2",
            "submission": self.submission.name,
            "submission_sha256": hashlib.sha256(src).hexdigest(),
            "bridgestan_tree": stan_library.tree(self.opt("bridgestan")).name,
            "capabilities": self.capabilities(),
            "max_evals": self.opt("max_evals"),
            "max_seconds": self.opt("max_seconds"),
        }

    def budget(self, req: RunRequest) -> tuple[int, float]:
        evals = int(self.opt("max_evals") or grid.max_evaluations(req.chains, req.warmup, req.draws))
        return evals, float(self.opt("max_seconds") or grid.MAX_SECONDS)

    def run(self, req: RunRequest) -> Timing:
        so = stan_library.build(
            stan_library.tree(self.opt("bridgestan")), req.model.stan_file, self.settings.build_dir
        )
        max_evals, max_seconds = self.budget(req)
        per_run_evals = max(1, max_evals // req.chains)
        out = req.out_dir.resolve()
        work = out / "_work"
        work.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        caps = self.capabilities()
        replay_steps = int(self.opt("replay_steps") or logdensity.REPLAY_STEPS)
        lane = self.opt("lane") or None

        rep_t0 = time.perf_counter()
        logs: list[str] = []
        chain_draws: list[np.ndarray] = []
        chain_s: list[float] = []
        warmup_s: list[float | None] = []
        sampling_s: list[float | None] = []
        tuning_shas: list[str] = []
        betas_ref: list[float] | None = None
        dim = 0
        adapt_dens = adapt_grad = adapt_hvp = adapt_hess = 0
        trans_dens = trans_grad = trans_hvp = trans_hess = 0
        total_eval_s = 0.0
        replayed_singles = 0
        replayed_window = 0

        try:
            for c in range(req.chains):
                elapsed = time.perf_counter() - rep_t0
                remaining = max_seconds - elapsed
                if remaining <= 0:
                    raise RuntimeError(f"rep wall-clock deadline ({max_seconds:.1f} s) exceeded before chain {c}")
                run_seed = int(np.random.SeedSequence([req.seed, c]).generate_state(1)[0])
                bdir = work / f"chain_{c}"
                bdir.mkdir(parents=True, exist_ok=True)
                base_spec = {
                    "library": str(so),
                    "data": str(Path(req.data_file).resolve()),
                    "submission": str(self.submission),
                    "seed": req.seed + c,
                    "run_seed": run_seed,
                    "chains": req.chains,
                    "draws": req.draws,
                    "max_evaluations": per_run_evals,
                    "max_seconds": remaining,
                    "bundle_dir": str(bdir),
                    "capabilities": caps,
                    "replay_steps": replay_steps,
                    "lane": lane,
                }
                logs.append(_run_phase("adapt", base_spec, bdir / "spec_adapt.json", env, remaining + 30.0))
                adapt_meta = json.loads((bdir / "adapt_meta.json").read_text())

                rem_after_adapt = max_seconds - (time.perf_counter() - rep_t0)
                if rem_after_adapt <= 0:
                    raise RuntimeError(f"rep wall-clock deadline ({max_seconds:.1f} s) exceeded after adapt {c}")
                logs.append(_run_phase("sample", base_spec, bdir / "spec_sample.json", env, rem_after_adapt + 30.0))
                sample_meta = json.loads((bdir / "sample_meta.json").read_text())

                # Unbilled fresh-process bit-exact replay
                logs.append(_run_phase("replay", base_spec, bdir / "spec_replay.json", env, max(60.0, remaining)))
                replay_meta = json.loads((bdir / "replay_meta.json").read_text())

                if betas_ref is None:
                    betas_ref = list(adapt_meta["betas"])
                    dim = int(adapt_meta["dim"])
                elif list(adapt_meta["betas"]) != betas_ref:
                    raise RuntimeError(f"chain {c} returned betas {adapt_meta['betas']}, expected {betas_ref}")

                draws_c = np.load(bdir / "draws.npy")
                chain_draws.append(draws_c)
                a_s, s_s = float(adapt_meta["seconds"]), float(sample_meta["seconds"])
                warmup_s.append(a_s)
                sampling_s.append(s_s)
                chain_s.append(a_s + s_s)
                tuning_shas.append(adapt_meta["tuning_sha256"])
                adapt_dens += int(adapt_meta["density_evals"])
                adapt_grad += int(adapt_meta["gradient_evals"])
                adapt_hvp += int(adapt_meta["hvp_evals"])
                adapt_hess += int(adapt_meta["hess_evals"])
                trans_dens += int(sample_meta["density_evals"])
                trans_grad += int(sample_meta["gradient_evals"])
                trans_hvp += int(sample_meta["hvp_evals"])
                trans_hess += int(sample_meta["hess_evals"])
                total_eval_s += float(adapt_meta["eval_s"]) + float(sample_meta["eval_s"])
                replayed_singles += int(replay_meta["replayed_singles"])
                replayed_window += int(replay_meta["replayed_window"])

            all_draws = np.stack(chain_draws, axis=0)  # (runs, walkers, draws, dim)
            draws_file = out / "unconstrained.npz"
            np.savez_compressed(draws_file, draws=all_draws, betas=np.asarray(betas_ref, dtype=float))

            constrain_spec = {
                "library": str(so),
                "data": str(Path(req.data_file).resolve()),
                "seed": req.seed,
                "bundle_dir": str(work),
                "draws_file": str(draws_file),
                "out_dir": str(out),
                "result_file": str(work / "constrain_meta.json"),
            }
            logs.append(_run_phase("constrain", constrain_spec, work / "spec_constrain.json", env, max(120.0, max_seconds)))
            constrain_s = float(json.loads((work / "constrain_meta.json").read_text())["constrain_s"])
        finally:
            (out / "submission.log").write_text("".join(logs))
            shutil.rmtree(work, ignore_errors=True)

        wall_s = float(sum(chain_s))
        dens_total = adapt_dens + trans_dens
        grad_total = adapt_grad + trans_grad
        hvp_total = adapt_hvp + trans_hvp
        hess_total = adapt_hess + trans_hess
        evals_total = dens_total + grad_total + hvp_total + hess_total
        overhead_s = max(0.0, wall_s - total_eval_s)
        overhead_ratio = (overhead_s / total_eval_s) if total_eval_s > 0 else None
        walkers = int(all_draws.shape[1])
        tempered = any(b < 1.0 for b in (betas_ref or [1.0]))

        return Timing(
            wall_s=wall_s,
            chain_s=chain_s,
            warmup_s=warmup_s,
            sampling_s=sampling_s,
            grads_sampling=trans_grad,
            grads_warmup=adapt_grad,
            divergences=None,
            evaluations=evals_total,
            extra={
                "abi": "kernel-v2",
                "density_evals": dens_total,
                "gradient_evals": grad_total,
                "hvp_evals": hvp_total,
                "hess_evals": hess_total,
                "adapt_density_evals": adapt_dens,
                "adapt_gradient_evals": adapt_grad,
                "adapt_hvp_evals": adapt_hvp,
                "adapt_hess_evals": adapt_hess,
                "transition_density_evals": trans_dens,
                "transition_gradient_evals": trans_grad,
                "transition_hvp_evals": trans_hvp,
                "transition_hess_evals": trans_hess,
                "max_evaluations": max_evals,
                "max_seconds": max_seconds,
                "eval_s": total_eval_s,
                "overhead_s": overhead_s,
                "overhead_ratio": overhead_ratio,
                "constrain_s": constrain_s,
                "dim": dim,
                "walkers": walkers,
                "betas": betas_ref,
                "tempered": tempered,
                "tuning_sha256": tuning_shas,
                "replay_singles": replayed_singles,
                "replay_window": replayed_window,
                "bridgestan_key": so.parent.name,
            },
        )
