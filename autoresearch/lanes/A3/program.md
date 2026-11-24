# Autoresearch Lane A3: Integrators, Momentum & Trajectories (`program.md`)

> **Karpathy-style 3-file contract for `Numenomics/apin`.**
> - **Read-only judge:** `harness/judge.py`, `harness/invariance.py`, `harness/ipc.py`, `harness/autoresearch.py` (never edit)
> - **Single writable file:** `autoresearch/lanes/A3/submission.py`
> - **Contract:** Full kernel-v2 ABI `adapt(oracle, budget, rng) -> (tuning, chain0, betas, aux_spec)` and `transition(tuning, chain, oracle, rng) -> chain_next` — focus on **symplectic / isokinetic integrators, persistent momentum / auxiliary velocity, and trajectory length / termination rules**.

## 1. Objective

Design and evolve the **integrator, momentum-refreshment law, and trajectory-termination rule** in `transition()` (plus supporting tuning in `adapt()`) to maximize horizon-amortized soft-min empirical efficiency ($p = -4$) across all non-multimodal shape classes while preserving exact-draw invariance (including joint $(\pi \otimes \text{aux\_law})$ stationarity!) and low Python overhead.

## 2. Hard Rules (Non-Negotiable)

1. **Edit ONLY `autoresearch/lanes/A3/submission.py`** (and append to `autoresearch/lanes/A3/NEGATIVE_MEMORY.md` only for $\ge 2$-trial clean falsifications). Never modify `harness/`, `data/`, `models/`, or any other lane.
2. **Respect `autoresearch/RULINGS.md` strictly:** `requires = ["grad"]` only; no unpriced `hvp`/`hess`, no inspecting `oracle` internals, no target fingerprinting.
3. **Exact-draw invariance + `aux_spec`:** If you carry persistent momentum (`"gaussian"`), unit-sphere microcanonical velocity (`"unit_sphere"`), or a direction/parity bit (`"rademacher"`), you MUST declare its exact stationary law in `aux_spec` and preserve $\pi \otimes \text{aux\_law}$ (including momentum flip on rejection for GHMC/MAMS so detailed balance holds!).
4. **Simplicity rule:** A candidate within 2% of the run's best speed-up is kept as `keep_simpler` when it has fewer lines, fewer magic constants or at least 15% lower `overhead_ratio` than the incumbent.
5. **Never stop early** while trial/wall-clock budget remains.

## 3. Research Loop & Priority Hypotheses

1. **Read context:** `autoresearch/lanes/A3/NEGATIVE_MEMORY.md`, `autoresearch/RULINGS.md`, `codex/INDEX.yaml`, `codex/cards/mams-microcanonical.md`, and `codex/cards/surrogate-and-ess.md`.
2. **Priority ideas to test:**
   - **Higher-order symplectic integrators:** McLachlan 2-stage (`velocity_verlet` with optimal $\lambda \approx 0.1931833275$) or Blanes–Casas–Sanz-Serna 2-/3-stage force-gradient-free palindromic splitting — higher accuracy per gradient allows larger effective step size $\epsilon$!
   - **Persistent-momentum GHMC (Horowitz / Horowitz–Kennedy):** Carry momentum $p \sim \mathcal{N}(0, M)$ in `chain.aux` (`aux_spec = "gaussian"` — store whitened $z \sim \mathcal{N}(0, I)$ in `aux`!), partially refresh $z' = \cos\theta \, z + \sin\theta \, \xi$, run short trajectories (even 1–4 steps!), and **flip momentum on rejection** (exact generalized detailed balance).
   - **Metropolis-Adjusted Microcanonical Sampler (MAMS, `codex/cards/mams-microcanonical.md`):** Carry unit-sphere velocity $u \in S^{d-1}$ (`aux_spec = "unit_sphere"`), use the exact microcanonical isokinetic integrator + MH acceptance ratio with exact energy/Jacobian adjustment + partial velocity refresh on $S^{d-1}$.
   - **Dynamic trajectory length (NUTS / multinomial U-turn / randomized exponential path length / ChEES-tuned path length):** Adapt trajectory length $L$ during `adapt()` (e.g. ChEES or SJD maximization) or use exact reversible multinomial NUTS doubling inside `transition()`.
   - **Surrogate-trajectory HMC (`codex/cards/surrogate-and-ess.md`):** Use a frozen quadratic/low-rank surrogate learned during `adapt()` for inner leapfrog proposals with exact target MH correction at the endpoints.
3. **Evaluate:**
   ```bash
   apin autoresearch eval --lane A3 --run-id <run_id> --codex-ref mams-microcanonical --hypothesis "..."
   ```
   It prints `METRIC speedup=<x>`, the speed-up over the lane's starting code (`start.py`) on the public seeds, and keeps the change when it beats the run's best by at least 5%. Use `--dry-run` first to lint/check duplicates without burning a ledger row. On discard, restore from `autoresearch/lanes/A3/runs/<run_id>/incumbent.py` (do **not** run `git checkout`/`git reset` while other agents share the repo).
4. **Real-posterior verification:**
   ```bash
   apin run logdensity --label a3_cand --opt sampler=autoresearch/lanes/A3/submission.py --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3
   ```
