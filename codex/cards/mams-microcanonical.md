# Card: `mams-microcanonical` — Metropolis-Adjusted Microcanonical Sampler (MAMS)

- **Axis / Lanes / Phase:** Axis 4 (Integrators & Dynamics) · Lanes `A3`, `A0` · Phase `transition`
- **Requires:** `["grad"]` · **Aux state:** `unit_sphere` (unit velocity $u \in \mathbb{S}^{d-1}$)
- **Source & Licence:** `blackjax` (`blackjax/mcmc/adjusted_mclmc_dynamic.py`, Apache-2.0 — `clean_room_from_paper`); `nuts-rs` (MIT); Robnik et al. (2022/2025)

## Mathematical Idea

Microcanonical Langevin Monte Carlo (MCLMC) replaces Gaussian kinetic energy with an **isokinetic** unit-norm velocity $u \in \mathbb{S}^{d-1}$ evolving under the microcanonical ODE:
$$\dot{x} = u, \qquad \dot{u} = \frac{1}{d-1}(I - u u^\top)\nabla \log \pi(x)$$
discretized via symmetric isokinetic splitting (`mclachlan` / `omelyan` minimal-norm), with periodic partial velocity refresh on $\mathbb{S}^{d-1}$.

Because unadjusted MCLMC retains $O(h^2)$ discretization bias (Track **L4 only**), **MAMS** integrates a randomized trajectory length $L$ with symmetric isokinetic steps and applies an **exact Metropolis-Hastings acceptance step** using the conserved microcanonical energy change $\Delta E$ (plus volume correction on the isokinetic manifold), yielding an **adjusted Track L1** kernel with strong high-dimensional scaling.

## Invariance & Failure Modes

- **Invariant:** Lifted non-reversible kernel on $\pi(x) \otimes \operatorname{Unif}(\mathbb{S}^{d-1})$. Store $u$ in `chain.aux` with `aux_spec = unit_sphere`. Flip $u \mapsto -u$ on rejection if using persistent velocity, and verify via **joint/marginal exact-draw stationarity** (NOT Gandy–Scott reversibility).
- **L1 vs L4 boundary:** Never submit unadjusted MCLMC to L1—any missing MH volume term will fail Tier 0/2 invariance.
- **Multiscale weakness:** Fixed-step isokinetic integration still struggles in steep funnels unless paired with Axis 2 transport (`mpart-monotone-transport`) or Axis 1 preconditioning.

## Falsification & Verification Tests

1. **Joint stationarity (`tier0_joint_stationarity`):** Initialize $(x_0, u_0) \sim \pi \otimes \operatorname{Unif}(\mathbb{S}^{d-1})$; verify zero drift in latent position statistics and $|u| = 1$ invariant.
2. **Dimension scaling (`tier2_high_dim_ladder`):** Measure empirical efficiency per cost unit across $d \in \{10, 50, 250, 1000\}$ vs. CmdStan NUTS.
