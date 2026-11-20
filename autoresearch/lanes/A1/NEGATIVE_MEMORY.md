# Negative Result Memory (`NEGATIVE_MEMORY.md`)

Append-only ledger of **falsified hypotheses** injected into every autoresearch agent prompt before proposal generation. Pre-seeded from `apin`'s prior campaign (`REPORT.md` §6 and `HANDOFF.md`, experiments W-1..W-77).

**Rule:** An agent may NOT re-propose a falsified mechanism unless its hypothesis explicitly names the `NM-*` ID and states a concrete, untested structural change that invalidates the prior refutation's scope.

---

## NM-001 · Naive Warmup Early-Exit via Windowed Convergence Statistics
- **Prior trials:** `W-21`, `W-25`, `W-28`, `W-37` (CmdStan / Stan adaptation)
- **What was tried:** Terminating Stan's metric/step-size warmup early once windowed parameter variance, mass-matrix drift, or acceptance statistics crossed a stability threshold.
- **Observed failure:** Refuted across 4 independent gate designs. No windowed statistic reliably separated easy models from marginal ones; premature exit severely degraded ESS on `arma11` (−33%), `lsat` (−40%), and `hier_2pl` (−58%).
- **Retry condition:** Only if using a qualitatively different convergence criterion backed by multi-chain pooled diagnostics (e.g., LAPS equipartition bias proxy or split-$\hat{R}$ across $K \ge 8$ parallel chains) and tested specifically on `arma11`, `lsat`, and `hier_2pl`.

## NM-002 · Unscreened Forced Low-Rank Fisher Metric in `walnutpie`
- **Prior trials:** `W-62`, `W-63`, `W-64`, `W-65`, `W-66` (`walnutpie`)
- **What was tried:** Unconditionally fitting a low-rank Fisher metric during warmup, and screening activation using `window_cross_ratio`.
- **Observed failure:** Catastrophic regression at grid scale (0.037× geometric mean; 8/15 models harmed). The `window_cross_ratio` gate was *inverted* relative to actual benefit (turned low-rank on where it hurt and off where it helped).
- **Retry condition:** Allowed **only** under **Experiment E1** (diagnosing the exact algorithmic difference vs. `pymc-devs/nuts-rs` arXiv:2603.18845 — e.g., eigenvalue shrinkage, damping, Fisher-divergence estimator, or funnel-neck clipping) with explicit screening tested on both rotated Gaussians and `eight_schools_centered`.

## NM-003 · Two-Phase Warmup Truncation & Adapt-Freeze Schedules
- **Prior trials:** `W-45`, `W-74`, `W-77`
- **What was tried:** Cutting late adaptation windows short or freezing metric adaptation earlier to save warmup gradient evaluations.
- **Observed failure:** Harmed 7 benchmark models; late warmup adaptation was measurably productive for final sampling efficiency.
- **Retry condition:** Only if replacing late warmup with a *faster-converging* estimator (e.g., Pathfinder Woodbury warm-start or Batch-and-Match Fisher VI) rather than merely truncating standard Welford variance estimation.

## NM-004 · ML Step-Size Schedules (Adam / Prodigy / D-Adaptation) for HMC Step Size
- **Prior trials:** `W-64`, `W-73` (`walnutpie` Adam step-size sweeps) + literature audit (`upstream/CANDIDATES.md` §5)
- **What was tried:** Tuning HMC step size $\epsilon$ with Adam-style adaptive learning rates instead of dual averaging.
- **Observed failure:** Zero net gain over dual averaging defaults; step-size oscillation harms acceptance stability.
- **Retry condition:** Do not sweep Adam for HMC step size. Only test structurally different step-size controllers (e.g., `PIDController` inside `adapt()` with strict damping) if motivated by a measured dual-averaging failure mode.

## NM-005 · `-march=native` in Mixed CmdStan / Stan-Math Builds
- **Prior trials:** `W-27` (Lane S2 systems campaign)
- **What was tried:** Enabling `-march=native` across partial/mixed Stan C++ translation units.
- **Observed failure:** Hard ABI corruption and gradient miscompilation across mixed Eigen/Stan-Math compilation units.
- **Retry condition:** **Hard ban** on mixed builds. Any ISA/SIMD flag experiment in Lane S2 must rebuild *all* translation units cleanly from scratch, pass finite-difference gradient checks, and match baseline parity rules.
