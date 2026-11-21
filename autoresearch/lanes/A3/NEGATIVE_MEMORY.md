# Negative Result Memory (`autoresearch/lanes/A3/NEGATIVE_MEMORY.md`)

Pre-seeded from `autoresearch/lanes/A1/NEGATIVE_MEMORY.md` (`NM-001`..`NM-005`):
- **NM-001:** Naive single-chain windowed warmup early-exit degrades marginal models.
- **NM-003:** Naive late-warmup truncation harms steady-state sampling unless replaced by a faster-converging estimator.
- **NM-004:** ML step-size schedules (Adam / Prodigy / D-Adaptation) oscillate and underperform dual averaging.

Append new Lane A3 falsifications (`NM-A3-001`, ...) after $\ge 2$ disciplined trials fail cleanly.

- **NM-A3-001 (2026-11-21, run `glm53_pdmp_1h`, trials T0003/T0004):** Exact state-independent
  coin mixtures (HMC $p\in\{0.6,0.75\}$ | adjusted Boomerang-orbit mini-steps with a
  shrunk-full-covariance reference) did not beat the diag-HMC incumbent on dual-horizon
  soft-min at dims=(6,20), repeats=2: ratios 0.9296 and 0.8746 respectively (both passed
  exact-draw global balance, b2_max <= 0.29).  The orbit branch wins mixture_crossable 1.49x
  and ties funnel/gaussian/rosenbrock but loses banana/student_t at d=20; the p=-4 soft-min
  stays pinned to hybrid_rosenbrock (~1.7e-4) where neither kernel breaks ESS starvation at
  k_steps=16.  Related one-off finding (not a 2-trial falsification, but exactness-critical):
  a *position-dependent* kernel gate (whitened-radius routing HMC vs orbit) is NOT exact -
  it broke MH reverse-proposal pairing and drifted the lp functional (z=5.21, reproducible);
   kernel selection must be state-independent or priced into the MH ratios.
- **NM-A3-002 (2026-11-21, run `glm53_a3_symp_1h`, trials T0007/T0006):** Raising GHMC partial-refresh
  persistence above cos θ = 0.8 fails cleanly on dual-horizon soft-min at dims=(6,20), repeats=2:
  α=0.9 at n=1 → 0.844× vs incumbent; α=0.9 at n=3 → 0.709× (vs α=0.8's 0.694× at n=3, i.e. no
  rescue).  α=0.8 is the interior optimum (0.5 → 4.63× baseline, 0.8 → 5.94× baseline).  Beyond
  0.8, aux-z autocorrelation time (~1/(1−α²) transitions) slows warmup exploration and aux mixing
  faster than the correlated-travel boost √((1+α)/(1−α)) helps.

- **NM-A3-003 (2026-11-21, run `glm53_a3_symp_1h`, trials T0005/T0006):** GHMC with n=3-step
  trajectories raises absolute dispersed-start ESS on every class (gaussian geo ess 0.095→0.76,
  hybrid_rosenbrock 0.092→0.115, funnel 0.67→0.80) but loses the p=-4 soft-min anyway
  (0.694×/0.709×): the 2× transition cost (4 vs 2 grads, cost_per_chain 70.5→147.6) flips the
  stiff/curved classes (hr, funnel, banana) net-negative, and the soft-min pins on
  hr+gaussian_illcond jointly.  Trajectory length > 1 is a cost-arbitrage loss for any kernel
  family in this horizon/amortization regime (short horizon pays adapt ×20/150; per-draw cost
  dominates).  Longer trajectories only pay if ESS gain per extra gradient exceeds ~2×.

