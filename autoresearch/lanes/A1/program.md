# Autoresearch Lane A1: Preconditioning & Metric Estimation (`program.md`)

> **Karpathy-style 3-file contract for `Numenomics/apin` — unlocked immediately after Slice 2.**
> - **Read-only judge:** `harness/judge.py` (never edit; digest recorded on every trial)
> - **Single writable file:** `autoresearch/lanes/A1/submission.py` (implements `adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)`)
> - **Fixed transition kernel:** Verified adjusted NUTS / HMC kernel provided by the harness (NOT editable in Lane A1). Note that `x0` returned by `adapt_metric` is ABI-checked for finite provenance during `adapt()`, while Tier 1 mixing scores start from fresh dispersed initial states so `adapt_metric` is judged on the metric & step size it learns rather than warm-start placement (warm-start transport belongs in Lane A4).

## 1. Objective

Discover an `adapt()` preconditioning and metric-estimation strategy that minimizes **total billed cost** (`adapt` + `transition` combined) to reach the 0.1-SD accuracy gate relative to CmdStan diagonal NUTS across correlated, rotated, and axis-aligned posteriors—**without** harming isotropic targets, short-warmup budgets, or funnels.

## 2. Hard Rules (Non-Negotiable)

1. **Edit ONLY `autoresearch/lanes/A1/submission.py`.** Never modify `harness/`, `data/`, `models/`, or `results.tsv`.
2. **Respect the capability allowlist:** `requires = ["grad"]` only. Do not call unpriced `hvp`/`hess` or inspect `oracle` internals, filenames, or model metadata (`RULINGS.md` R-001..R-006).
3. **Freeze everything after `adapt()`:** Return a canonical, finite `metric_tuning` tree (`dict`/`list`/`ndarray`/scalars) within the 16 MB state cap. `transition()` runs in a clean child process—no module globals or caches survive.
4. **Never repeat falsified ideas without a stated new condition:** Read `NEGATIVE_MEMORY.md` before every proposal.
5. **Simplicity rule:** A candidate within 2% of the run's best speed-up is kept as `keep_simpler` when it has fewer lines, fewer magic constants or at least 15% lower `overhead_ratio` than the incumbent.
6. **Never stop** until the trial/token/wall-clock budget is exhausted; on crash or Tier 0 fail, log the row, reset the worktree, and move immediately to the next hypothesis.

## 3. Research Loop (Per Trial)

1. **Read context:**
   - `NEGATIVE_MEMORY.md` (mandatory!)
   - Last 20 rows of `results.tsv` + current `archive/` cells
   - `codex/INDEX.yaml` (Axes 1 and 7; start with `fisher-low-rank`, Pathfinder Woodbury, shrinkage/regularization, Batch-and-Match VI)
2. **Formulate hypothesis (structured header in commit message & ledger):**
   - `hypothesis`: one falsifiable sentence
   - `codex_refs`: card IDs cited
   - `assumption`: what geometry/regime this improves and why it will not harm funnels or isotropic targets
   - `falsification_test`: which check or model will refute it if wrong
3. **Edit `submission.py`** and run `apin autoresearch eval --lane A1 --run-id <run_id> --hypothesis "..."` (use `--dry-run` first to lint and check for duplicates without writing a ledger row). It prints `METRIC speedup=<x>`, the speed-up over the lane's starting code (`start.py`) on the public seeds, and keeps the change when it beats the run's best by at least 5%. The runner snapshots `submission.py` at start, acquires the host FIFO evaluation lock (`harness/queue.py`), isolates `incumbent.py` and `archive/` per `--run-id`, amortizes `adapt_cost` to a 500-step sampling horizon so warmup starvation cannot game Tier 1, and grades the snapshot once with `harness/grade.py` at the standard configuration (dims 6 and 20, 8 repeats).
4. **Ratchet & Archive decision (enforced by runner):**
   - **Fail Tier 0 or any Tier 1 accuracy/replay gate:** restore from `autoresearch/lanes/A1/runs/<run_id>/incumbent.py` (or `git checkout -- autoresearch/lanes/A1/submission.py` if working solo on a branch), keeping the immutable `ledger.jsonl` row. If a general mechanism fails cleanly across $\ge 2$ disciplined variants, append a scoped entry to `NEGATIVE_MEMORY.md`.
   - **Pass + speed-up at least 1.05 times the run's best (or within 2% of it and simpler):** runner promotes the snapshot to `runs/<run_id>/incumbent.py`.
   - **Pass + new best in a MAP-Elites cell:** runner saves patch to `runs/<run_id>/archive/<cell>/` automatically.
5. **Real-posterior verification (recommended before declaring a breakthrough):**
   ```bash
   apin run logdensity --label a1_cand --opt sampler=autoresearch/lanes/A1/submission.py --opt lane=A1 --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3
   ```

## 4. Priority Starting Hypotheses

1. **Settle E1 first:** Replicate `nuts-rs` low-rank Fisher shrinkage/eigenvalue flooring (`codex/cards/fisher-low-rank.md`) and diagnose the exact regularization gap vs. `apin` W-62..W-66.
2. **Funnel-safe Fisher damping:** Cap gradient-variance domination when gradient norms span $> 2$ orders of magnitude (preventing neck collapse on `eight_schools_centered`).
3. **Pathfinder Woodbury → short Fisher polish:** Use L-BFGS curvature from multi-path warmup to cut `adapt()` evaluations while preserving correlated-posterior mixing.
4. **Adaptive rank selection:** Shrink rank $k \to 0$ (pure diagonal) automatically when the top eigenvalues of the whitened Fisher sketch fail a Marchenko–Pastur / cross-window signal threshold.
