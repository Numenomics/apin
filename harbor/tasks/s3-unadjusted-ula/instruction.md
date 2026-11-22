# Add Metropolis-Hastings adjustment to biased unadjusted Langevin (ULA) kernel

The kernel in `/app/submission.py` implements unadjusted Langevin (ULA), which has severe O(eps^2) stationary variance bias and fails global-balance invariance.

Convert `/app/submission.py` into an adjusted kernel (e.g., MALA or HMC) that preserves the target distribution and mixes well on rotated ill-conditioned Gaussians.

Verify locally with:
```bash
uv run apin invariance --sampler /app/submission.py --dim 4
```
Do not edit `harness/` or `models/`. Only edit `/app/submission.py`.
