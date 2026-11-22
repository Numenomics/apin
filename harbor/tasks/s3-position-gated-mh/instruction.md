# Fix position-dependent kernel gating MH detailed-balance bias

The sampler in `/app/submission.py` (when running on a host checkout outside Docker, edit your copy of `environment/baseline.py`) fails exact-draw invariance because it switches between HMC and RWM based on the current radial position without correcting for reverse kernel selection probability. Diagnose and fix it so it preserves the target distribution and mixes well on rotated ill-conditioned Gaussians.

Quick dev smoke check (fast, looser than final hidden grading):
```bash
uv run apin invariance --sampler /app/submission.py --dim 4
```
Final grading runs on hidden held-out seeds after you finish. Do not edit `harness/` or `models/`. Only edit `/app/submission.py`.
