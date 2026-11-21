# Negative Result Memory (`autoresearch/lanes/A0/NEGATIVE_MEMORY.md`)

Ground-up Lane A0 starts with a clean slate (no pre-seeded algorithmic priors). Append scoped entries (`NM-A0-001`, ...) only after a mechanism fails cleanly across $\ge 2$ disciplined trials.

## NM-A0-001 — Dual averaging on jittered-length HMC warmup (falsified)

**Mechanism:** Hoffman–Gelman dual averaging (`log_eps = mu - sqrt(t)/gamma · h_bar`, `gamma=0.05`) driving the step size of a *doubling-jitter* leapfrog warmup (L ∈ {1,2,4,…} with growth probability).

**Falsification (2 instrumented designs, T0003-era and T0004-era warmups; per-iteration traces on IllCondGaussian d=6 rot κ=50, NealsFunnel d=6 rot, HybridRosenbrock d=20):** mixing over trajectory lengths makes the acceptance statistic bimodal — short trajectories accept (acc≈1) while same-eps long trajectories diverge (acc=0) — so no single eps achieves a target acceptance and DA limit-cycles (acceptance flips 0↔1 across iterations while eps swings 10×) with `sqrt(t)/gamma ≈ 17·h_bar` amplifying any persistent acceptance deficit until `log_eps_bar` collapses to the `e^-14` floor (observed final eps 8.3e-7 = floor on multiple targets/seeds; frozen-kernel signature: late-step latent displacement ≈ 1e-6 ≪ 1e-3 mixing bar).

**Fix that worked:** plain Robbins–Monro, `log_eps += (acc − δ)/sqrt(i+10)` (bounded per-iteration movement, no amplified oscillation), fixed L=3 during warmup so the acceptance signal is smooth, jitter only in the final frozen kernel, plus a measured ring-sweep of eps under the final metric (T0005/T0006 kept). Note δ=0.55–0.65 still froze curved-ridge classes; δ=0.8 did not.
