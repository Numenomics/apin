# Build an adjusted posterior sampler from a 43-LOC isotropic RWM stub

Starting from the 43-line isotropic Random-Walk Metropolis stub in `/app/autoresearch/lanes/A0/submission.py`, design and evolve a gradient-capable adjusted MCMC/PDMP sampler from scratch that maximizes dual-horizon (`H=150/1000`) empirical efficiency across `dims=(6,20)` while passing all exact-draw invariance, declared `aux_spec`, dispersed-start mixing, and bit-exact replay checks.

> **Scoring note:** `soft_min_eff` is a **$p = -4$ power mean** over per-class geometric means, so the **weakest target class dominates**. Check `CLASS_METRIC` after each trial and lift the worst class without regressing it.

## Rules
- Read `/app/autoresearch/lanes/A0/program.md`, `/app/autoresearch/RULINGS.md`, and `/app/codex/INDEX.yaml`.
- Edit **ONLY** `/app/autoresearch/lanes/A0/submission.py`. Never modify `harness/`, `models/`, or `data/`.
- Iterate locally using:
  ```bash
  uv run apin autoresearch eval --lane A0 --run-id harbor_dev --dims 6,20 --repeats 2 --no-queue --hypothesis "<one falsifiable sentence>"
  ```
- Final grading runs your `/app/autoresearch/lanes/A0/submission.py` against the frozen baseline on **hidden held-out seeds**; reward = `cand_soft_min / baseline_soft_min` (or `0.0` on any invariance/replay failure).
