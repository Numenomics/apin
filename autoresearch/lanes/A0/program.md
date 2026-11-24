# Autoresearch Lane A0: Ground-Up MCMC Sampler (No Priors) (`program.md`)

> **Karpathy-style 3-file contract for `Numenomics/apin` — full sampler from scratch.**
> - **Read-only judge:** `harness/judge.py`, `harness/invariance.py`, `harness/ipc.py`, `harness/autoresearch.py` (never edit)
> - **Single writable file:** `autoresearch/lanes/A0/submission.py`
> - **Contract:** Implement both `adapt(oracle, budget, rng) -> (tuning, chain0, betas, aux_spec)` and `transition(tuning, chain, oracle, rng) -> chain_next` from scratch.

## 1. Objective

Design and evolve an adjusted Markov chain Monte Carlo sampler **from the ground up** — starting from a bare isotropic random-walk Metropolis stub with **zero pre-baked HMC, NUTS, or mass-matrix boilerplate** — that passes exact-draw invariance, dispersed-start mixing, and the 0.1-SD real-posterior bench gate while maximizing **horizon-amortized empirical efficiency** (`Var_pi(f) / MSE` per cost unit, soft-min $p = -4$ across all non-multimodal shape classes) and generalizing cleanly to higher dimensions and real Stan posteriors.

## 2. Hard Rules (Non-Negotiable)

1. **Edit ONLY `autoresearch/lanes/A0/submission.py`** (and append to `autoresearch/lanes/A0/NEGATIVE_MEMORY.md` only for $\ge 2$-trial clean falsifications). Never modify `harness/`, `data/`, `models/`, or any other lane.
2. **Respect `autoresearch/RULINGS.md` strictly:** `requires = ["grad"]` (or density-only) — no unpriced `hvp`/`hess`, no inspecting `oracle` internals, no target fingerprinting, no external sampler libraries (`cmdstan`, `nutpie`, `blackjax`, `pymc`, etc.). Only Python stdlib + `numpy` + `scipy`.
3. **Freeze everything after `adapt()`:** Return canonical finite `(tuning, chain0, betas, aux_spec)`. `transition()` runs in a fresh child process with derived per-step RNGs and bit-exact replay — no module globals or caches survive.
4. **Declared `aux_spec`:** If you use auxiliary Markov state (e.g. persistent momentum `"gaussian"`, unit-sphere velocity `"unit_sphere"`, discrete parity `"rademacher"`, or `{"kind": "product", "parts": [...]}`), declare its exact stationary law in `aux_spec` so the invariance suite checks joint $(\pi \otimes \text{aux\_law})$ stationarity and independence.
5. **Simplicity rule:** A candidate within 2% of the run's best speed-up is kept as `keep_simpler` when it has fewer lines, fewer magic constants or at least 15% lower `overhead_ratio` than the incumbent.
6. **Never stop early** while trial/wall-clock budget remains.

## 3. Research Loop

1. Use `apin autoresearch eval --lane A0 --run-id <run_id> --hypothesis "..."`. It prints `METRIC speedup=<x>`, the speed-up over the lane's starting code (`start.py`) on the public seeds, and keeps the change when it beats the run's best by at least 5%. Run `--dry-run` first to lint and check for duplicates without writing a ledger row.
2. Inspect `autoresearch/lanes/A0/runs/<run_id>/trials/<trial_id>/metrics.json` after each trial (`by_class`, `per_target`, `adapt_cost`, `b2_max`, `overhead_ratio`).
3. On discard, restore from `autoresearch/lanes/A0/runs/<run_id>/incumbent.py` (`cp autoresearch/lanes/A0/runs/<run_id>/incumbent.py autoresearch/lanes/A0/submission.py`). Do **not** run `git checkout` or `git reset` if other agents share the workspace.
4. Once you have a strong incumbent, look at higher dimensions with `--dim 12` or `--dim 16` and `--allow-duplicate`. Such a trial is exploratory: it gets no speed-up and is never kept. Then check real Stan posteriors:
   ```bash
   apin run logdensity --label a0_cand --opt sampler=autoresearch/lanes/A0/submission.py --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3
   apin score --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3 --baseline cmdstan --labels cmdstan,a0_cand
   ```
