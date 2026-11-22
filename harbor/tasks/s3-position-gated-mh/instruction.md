# Fix position-dependent kernel gating MH detailed-balance bias

The sampler in `/app/submission.py` fails exact-draw invariance because it switches between HMC and RWM based on the current radial position without correcting for reverse kernel selection probability. Diagnose and fix `/app/submission.py` so it preserves the target distribution and mixes well on rotated ill-conditioned Gaussians.

Verify locally with:
```bash
uv run apin invariance --sampler /app/submission.py --dim 4
```
Do not edit `harness/` or `models/`. Only edit `/app/submission.py`.
