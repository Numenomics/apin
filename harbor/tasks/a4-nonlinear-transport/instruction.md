# Optimize warm-start & diffeomorphic nonlinear transport under fixed latent HMC judge

Optimize `adapt_transport()`, `forward()`, `pullback_grad()`, and `inverse()` in `/app/autoresearch/lanes/A4/submission.py` (latent-space HMC and automatic finite-difference Jacobian + round-trip inverse verification are owned by `harness/judge.py`). Evolve a fast diffeomorphic transport map (e.g. whitening, Sinh-Arcsinh tail warps, triangular funnel/banana unwarping) maximizing dual-horizon `soft_min_eff` across `dims=(6,20)`.

> **Scoring note:** `soft_min_eff` is a **$p = -4$ power mean** over per-class geometric means, so the **weakest target class dominates**. Check `CLASS_METRIC` after each trial and lift the worst class without regressing it.

## Rules
- Read `/app/autoresearch/lanes/A4/program.md`, `/app/autoresearch/RULINGS.md`, and `/app/codex/INDEX.yaml`.
- Edit **ONLY** `/app/autoresearch/lanes/A4/submission.py`. Never modify `harness/`, `models/`, or `data/`.
- Iterate locally using:
  ```bash
  uv run apin autoresearch eval --lane A4 --run-id harbor_dev --dims 6,20 --repeats 2 --no-queue --hypothesis "<one falsifiable sentence>"
  ```
- Final grading runs your `/app/autoresearch/lanes/A4/submission.py` against the frozen baseline on **hidden held-out seeds**; reward = `cand_soft_min / baseline_soft_min` (or `0.0` on any invariance/replay failure).
