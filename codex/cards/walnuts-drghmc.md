# Card: `walnuts-drghmc` — Within-Orbit Adaptive & Delayed-Rejection HMC

- **Axis / Lanes / Phase:** Axis 3 (Multiscale / Funnel Dynamics) · Lanes `A2`, `A0` · Phase `transition`
- **Requires:** `["grad"]` · **Aux state:** `none` (WALNUTS / DR-HMC) or `normal_momentum` (DRGHMC)
- **Source & Licence:** `walnutpie` / `bob-carpenter/walnuts` / `giladturok/drghmc` / `flatironinstitute/bayes-kit` (all MIT — `direct_port_with_attribution`)

## Mathematical Idea

Standard HMC/NUTS fixes a single step size $h$ across phase space, forcing either divergences in a funnel neck or tiny $h$ everywhere else. Two exact remedies adapt step size *locally inside a transition*:

1. **WALNUTS (Within-Orbit Adaptive Leapfrog NUTS):** At each leapfrog macro-step, refine $h \to h / 2^m$ until energy error $|\Delta H| \le \theta$, then check the **reverse step-size condition** from the endpoint so forward and backward trajectories select the same refinement level $m$ (rejecting mismatched paths to preserve detailed balance).
2. **DR-HMC / DRGHMC (Delayed Rejection Generalized HMC):** Propose with large step $h_1$; if rejected, make a second-stage proposal with $h_2 = h_1 / a$ (and optional partial momentum refresh $p' = \cos\phi\, p + \sin\phi\, z$ stored in `chain.aux`), adjusting the stage-$k$ Metropolis acceptance ratio by the probability of failing earlier stages in reverse.

## Invariance & Failure Modes

- **Invariant:** WALNUTS and DR-HMC are reversible (Gandy–Scott valid). **DRGHMC with partial momentum refresh is lifted/non-reversible** (`aux_spec = normal_momentum`) and requires joint stationarity testing on $\pi \otimes \mathcal{N}(0, M)$ + momentum negation on rejection!
- **Critical trap (Council R2):** Swapping leapfrog leaves for higher-order McLachlan/Omelyan inside WALNUTS invalidates the standard backward step-size check unless the reversible balance condition is re-derived for the composite leaf.
- **Funnel coupling:** Global Fisher preconditioning is neck-dominated; pair local step adaptivity with Axis 2 transport (not just global scaling).

## Falsification & Verification Tests

1. **Latent-neck invariance (`tier2_funnel_neck_invariance`):** Exact-draw stationarity on Neal's funnel ($d = 10, 50$) measuring unrotated quantile indicators $v < -6$ at Tier 2 MDB.
2. **Dispersed-start mixing:** Must eliminate divergences on `eight_schools_centered` and hybrid Rosenbrock while keeping Gaussian cost overhead $< 15\%$.
