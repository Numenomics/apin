# Optimize adapt_metric() preconditioning under fixed HMC judge

Optimize `adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)` in `/app/autoresearch/lanes/A1/submission.py` (the transition kernel is fixed and owned by `harness/judge.py`). Maximize dual-horizon (`H=150/1000`) soft-min efficiency across `dims=(6,20)` on rotated Gaussians, funnels, bananas, hybrid Rosenbrock, and heavy-tailed Student-t(nu=1.5) without inflating warmup cost.

> **Scoring note:** `soft_min_eff` is a **$p = -4$ power mean** over per-class geometric means, so the **weakest target class dominates**. Check `CLASS_METRIC` after each trial and lift the worst class without regressing it.

## Rules
- Read `/app/autoresearch/lanes/A1/program.md`, `/app/autoresearch/RULINGS.md`, and `/app/codex/INDEX.yaml`.
- Edit **ONLY** `/app/autoresearch/lanes/A1/submission.py`. Never modify `harness/`, `models/`, or `data/`.
- Iterate locally using (`--no-queue` is safe inside an isolated Harbor container; omit it on shared hosts):
  ```bash
  uv run apin autoresearch eval --lane A1 --run-id harbor_dev --dims 6,20 --repeats 2 --no-queue --hypothesis "<one falsifiable sentence>"
  ```
- Final grading runs your `/app/autoresearch/lanes/A1/submission.py` against the frozen baseline (`environment/baseline.py`) on **hidden held-out seeds** via `harbor/verify.py optimize`; reward = `cand_soft_min / baseline_soft_min` (or `0.0` on any invariance/replay failure).
