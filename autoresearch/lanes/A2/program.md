# Autoresearch Lane A2: Multiscale & Funnel Dynamics (`program.md`)

> **Karpathy-style 3-file contract for `Numenomics/apin`.**
> - **Read-only judge:** `harness/judge.py`, `harness/invariance.py`, `harness/ipc.py`, `harness/autoresearch.py` (never edit)
> - **Single writable file:** `autoresearch/lanes/A2/submission.py`
> - **Contract:** Full kernel-v2 ABI `adapt(oracle, budget, rng) -> (tuning, chain0, betas, aux_spec)` and `transition(tuning, chain, oracle, rng) -> chain_next` — focus on **local step-size / multiscale / delayed-rejection transition dynamics** that survive funnels, bananas, and multiscale Rosenbrock ridges without harming rotated Gaussians.

## 1. Objective

Improve `transition()` (and matching warmup parameters in `adapt()`) to eliminate multiscale/funnel failure modes (`NealsFunnel`, `Banana`, `HybridRosenbrock`, `StudentT(nu=1.5)`, and real Stan funnels like `eight_schools_centered` and `pilots`) while maximizing horizon-amortized soft-min empirical efficiency ($p = -4$) across all non-multimodal shape classes and preserving exact-draw invariance.

## 2. Hard Rules (Non-Negotiable)

1. **Edit ONLY `autoresearch/lanes/A2/submission.py`** (and append to `autoresearch/lanes/A2/NEGATIVE_MEMORY.md` only for $\ge 2$-trial clean falsifications). Never modify `harness/`, `data/`, `models/`, or any other lane.
2. **Respect `autoresearch/RULINGS.md` strictly:** `requires = ["grad"]` only; no unpriced `hvp`/`hess`, no inspecting `oracle` internals, no target fingerprinting.
3. **Preserve exact-draw invariance:** Every local step-size or delayed-rejection rule MUST be involutively / detailed-balance exact (check `codex/cards/walnuts-drghmc.md` for WALNUTS reversible micro-step halving and DRGHMC delayed-rejection acceptance ratios). Unadjusted step-size halving without the reverse-path check will be caught immediately by the Tier 0/1 invariance suite!
4. **Declared `aux_spec`:** If using persistent momentum (DRGHMC) or auxiliary state, declare its exact stationary law (`"gaussian"`, `"unit_sphere"`, `"rademacher"`, or `{"kind": "product", "parts": [...]}`).
5. **Simplicity rule:** A candidate within 2% of the run's best speed-up is kept as `keep_simpler` when it has fewer lines, fewer magic constants or at least 15% lower `overhead_ratio` than the incumbent.
6. **Never stop early** while trial/wall-clock budget remains.

## 3. Research Loop & Priority Hypotheses

1. **Read context:** `autoresearch/lanes/A2/NEGATIVE_MEMORY.md`, `autoresearch/RULINGS.md`, `codex/INDEX.yaml`, and `codex/cards/walnuts-drghmc.md`.
2. **Priority ideas to test:**
   - **Delayed-Rejection HMC / DRGHMC:** On rejection of a macro-step trajectory at step size $\epsilon$, propose a secondary finer-step trajectory at $\epsilon / 2$ (or $\epsilon / 4$) with exact Tierney–Mira delayed-rejection acceptance probability $\alpha_2$ (and partial momentum refreshment via `aux_spec = "gaussian"`).
   - **Reversible WALNUTS-style micro-step halving:** Adaptive local micro-step halving inside macro steps with exact forward/backward energy-error reversibility check (guaranteeing detailed balance in funnel necks).
   - **Barker / skew-symmetric robust gradient proposals** (Livingstone & Zanella) as a fallback or primary multiscale kernel on heavy-tailed `StudentT(nu=1.5)` and steep funnel walls where unbounded gradients explode leapfrog.
   - **GIST (Gibbs Self-Tuning) / randomized step-size mixture:** Exact randomized step-size distribution $\epsilon \sim p(\epsilon)$ (e.g. two-scale log-uniform mixture of coarse + fine step sizes) satisfying detailed balance trivially while exploring both funnel mouths and necks!
3. **Evaluate:**
   ```bash
   apin autoresearch eval --lane A2 --run-id <run_id> --codex-ref walnuts-drghmc --hypothesis "..."
   ```
   It prints `METRIC speedup=<x>`, the speed-up over the lane's starting code (`start.py`) on the public seeds, and keeps the change when it beats the run's best by at least 5%. Use `--dry-run` first to lint/check duplicates without burning a ledger row. On discard, restore from `autoresearch/lanes/A2/runs/<run_id>/incumbent.py` (do **not** run `git checkout`/`git reset` while other agents share the repo).
4. **Real-posterior verification (especially `eight_schools_centered`!):**
   ```bash
   apin run logdensity --label a2_cand --opt sampler=autoresearch/lanes/A2/submission.py --models blr,eight_schools_noncentered,eight_schools_centered,gp_regr,garch11 --reps 3
   ```
