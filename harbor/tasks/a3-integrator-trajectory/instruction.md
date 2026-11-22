# Optimize symplectic/isokinetic integrators, momentum, trajectories, or PDMPs

Optimize `adapt()` and `transition()` in `/app/autoresearch/lanes/A3/submission.py` using higher-order symplectic splitting, persistent-momentum GHMC, MAMS (`unit_sphere` aux_spec), dynamic trajectories, or thinning-free PDMP mixtures (Boomerang/Zig-Zag/BPS) to maximize dual-horizon `soft_min_eff` across `dims=(6,20)` while passing exact-draw invariance.

> **Scoring note:** `soft_min_eff` is a **$p = -4$ power mean** over per-class geometric means, so the **weakest target class dominates**. Check `CLASS_METRIC` after each trial and lift the worst class without regressing it.

## Rules
- Read `/app/autoresearch/lanes/A3/program.md`, `/app/autoresearch/RULINGS.md`, and `/app/codex/INDEX.yaml`.
- Edit **ONLY** `/app/autoresearch/lanes/A3/submission.py`. Never modify `harness/`, `models/`, or `data/`.
- Iterate locally using:
  ```bash
  uv run apin autoresearch eval --lane A3 --run-id harbor_dev --dims 6,20 --repeats 2 --no-queue --hypothesis "<one falsifiable sentence>"
  ```
- Final grading runs your `/app/autoresearch/lanes/A3/submission.py` against the frozen baseline on **hidden held-out seeds**; reward = `cand_soft_min / baseline_soft_min` (or `0.0` on any invariance/replay failure).
