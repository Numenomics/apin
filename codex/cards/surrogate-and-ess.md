# Card: `surrogate-and-ess` — Surrogate-Trajectory HMC & Transport Elliptical Slice

- **Axis / Lanes / Phase:** Axes 4 & 5 (Phase-Space Dynamics / Gradient-Free) · Lanes `A3`, `A0`, Track `L5` · Phase `both`
- **Requires:** `["grad"]` (Surrogate HMC) or `["density_only"]` (Elliptical Slice) · **Aux state:** `none`
- **Source & Licence:** `blackjax` (`blackjax/mcmc/elliptical_slice.py`, Apache-2.0 — `clean_room_from_paper`); Rasmussen (2003); Murray et al. (2010); Cabezas & Nemeth (2023)

## Mathematical Idea

Replaces classical Split HMC (which suffers from period-$\pi$ resonance and stiff residual forces on heavy tails) with two exact mechanisms that exploit a fitted Gaussian or transport surrogate without corrupting the target:

1. **Surrogate-Trajectory HMC (Rasmussen 2003 / Delayed Acceptance, Track L1):** Fit a fast surrogate $\hat{U}(x)$ (Gaussian from Pathfinder/Fisher or degree-2 `MParT` map) during `adapt()` and freeze it. In `transition()`, integrate a long HMC trajectory entirely on the **cheap surrogate $\hat{U}$** (zero true target gradient calls along the trajectory!), then perform **one exact Metropolis-Hastings step** on the true Hamiltonian $H(x, p) = U(x) + K(p)$ at the endpoint—paying only **1 true density evaluation** (`n_density = 1, n_gradient = 0` inside `transition`!).
2. **Transport Elliptical Slice Sampling (Track L5, density-only):** Fit a Gaussian/triangular map $T$ in `adapt()` (from density-only optimization or warm-start draws), then run exact **Elliptical Slice Sampling** on the residual $\log \pi(T^{-1}(z)) + \log|\det J| - \log \mathcal{N}(z; 0, I)$—zero step-size tuning and exact invariance.

## Invariance & Failure Modes

- **Invariant:** Both kernels are **strictly reversible** (Gandy–Scott valid) provided the surrogate $\hat{U}$ / map $T$ is frozen in `tuning` before `transition()` and the trajectory integrator on $\hat{U}$ is symmetric and volume-preserving.
- **Degeneracy risk:** If the surrogate poorly approximates the true target (heavy tails, sharp funnel neck), MH acceptance collapses toward 0! Track acceptance rate and pair with a fallback local kernel (mixture of surrogate step + standard NUTS/Barker step) so ergodicity never stalls.

## Falsification & Verification Tests

1. **Near-Gaussian efficiency (`tier0_gaussian_rot`):** Should achieve dramatic `cost` reduction on correlated/high-$d$ near-Gaussian posteriors (`blr`, `gp_regr`) because surrogate trajectories spend zero target gradients.
2. **Heavy-tail / mismatch guard (`tier2_heavy_tails`):** Verify acceptance does not collapse on Student-t ($\nu = 3$) or funnel targets when paired with a local fallback move.
