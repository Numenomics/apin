# Optimize multiscale and funnel-surviving transition dynamics

Optimize `adapt()` and `transition()` in `/app/autoresearch/lanes/A2/submission.py` to survive multiscale funnels, bananas, hybrid Rosenbrock ridges, and heavy tails across `dims=(6,20)` (e.g., reversible WALNUTS micro-step halving, DRGHMC delayed rejection, GHMC+Barker) while preserving exact-draw global-balance invariance and maximizing dual-horizon `soft_min_eff`.

> **Scoring note:** `soft_min_eff` is a **$p = -4$ power mean** over per-class geometric means, so the **weakest target class dominates**. Check `CLASS_METRIC` after each trial and lift the worst class without regressing it.

## Rules
- Read `/app/autoresearch/lanes/A2/program.md`, `/app/autoresearch/RULINGS.md`, and `/app/codex/INDEX.yaml`.
- Edit **ONLY** `/app/autoresearch/lanes/A2/submission.py`. Never modify `harness/`, `models/`, or `data/`.
- Iterate locally using:
  ```bash
  uv run apin autoresearch eval --lane A2 --run-id harbor_dev --dims 6,20 --repeats 2 --no-queue --hypothesis "<one falsifiable sentence>"
  ```
- Final grading runs your `/app/autoresearch/lanes/A2/submission.py` against the frozen baseline on **hidden held-out seeds**; reward = `cand_soft_min / baseline_soft_min` (or `0.0` on any invariance/replay failure).
