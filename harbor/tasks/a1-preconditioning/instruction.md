# Optimize adapt_metric() preconditioning under fixed HMC judge

Optimize `adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)` in `/app/autoresearch/lanes/A1/submission.py` (the transition kernel is fixed and owned by `harness/judge.py`). Maximize dual-horizon (`H=150/1000`) soft-min efficiency across `dims=(6,20)` on rotated Gaussians, funnels, bananas, hybrid Rosenbrock, and heavy-tailed Student-t(nu=1.5) without inflating warmup cost.

## Rules
- Read `/app/autoresearch/lanes/A1/program.md`, `/app/autoresearch/RULINGS.md`, and `/app/codex/INDEX.yaml`.
- Edit **ONLY** `/app/autoresearch/lanes/A1/submission.py`. Never modify `harness/`, `models/`, or `data/`.
- Iterate locally using:
  ```bash
  uv run apin autoresearch eval --lane A1 --run-id harbor_dev --dims 6,20 --repeats 2 --no-queue --hypothesis "<one falsifiable sentence>"
  ```
- Final grading runs your `/app/autoresearch/lanes/A1/submission.py` against the frozen baseline on **hidden held-out seeds**; reward = `cand_soft_min / baseline_soft_min` (or `0.0` on any invariance/replay failure).
