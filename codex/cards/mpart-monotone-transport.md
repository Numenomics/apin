# Card: `mpart-monotone-transport` — Monotone Linearized-Hermite Transport Maps

- **Axis / Lanes / Phase:** Axis 2 (Nonlinear Geometry Neutralization) · Lanes `A4`, `A0` · Phase `adapt`
- **Requires:** `["grad"]` · **Aux state:** `none`
- **Source & Licence:** `MeasureTransport/MParT` (BSD-3-Clause — `direct_port_with_attribution`); `flowjax` (MIT) for spline alternative

## Mathematical Idea

Fit a lower-triangular Knothe–Rosenblatt map $T: \mathbb{R}^d \to \mathbb{R}^d$, where component $T_k(x_{1:k})$ is strictly increasing in $x_k$, so the Jacobian is lower triangular and $\log |\det \nabla T(x)| = \sum_{k=1}^d \log \partial_{x_k} T_k(x_{1:k})$ is exact and $O(d)$.

Unlike neural normalizing flows (noisy SGD, slow warmup), low-degree monotone polynomial/Hermite maps (`MParT`) have **convex or quasi-convex** KL / Fisher-divergence objectives over warmup draws + gradients (`FisherLoss`). Sampling then runs an ordinary adjusted kernel (NUTS or MAMS) on the pulled-back density $\tilde{\pi}(z) = \pi(T^{-1}(z)) |\det \nabla T^{-1}(z)|$ (or pushforward), mapping draws back via $T$.

## Invariance & Failure Modes

- **Mandatory tail guard (Council R3):** Pure polynomial bases diverge as $|x| \to \infty$, exploding Jacobians and destroying heavy-tailed targets (`Student-t`). **Require `MParT`'s linearized-Hermite basis** (linear extrapolation outside the bulk) or rational-quadratic splines (`flowjax`) that become linear in the tails!
- **Invariant:** Map parameters MUST freeze in `tuning` before `transition()`. Every evaluation in `transition()` must include the exact $\log|\det J|$ term and its gradient (via chain rule).
- **Warmup billing:** All `adapt()` gradient calls used to fit $T$ count toward total `cost`. Keep degree low ($p \in \{2, 3\}$) and sparse/banded coupling for $d > 20$.

## Falsification & Verification Tests

1. **Missing-Jacobian detector (`tier0_missing_jacobian_check`):** Exact-draw invariance on 2D/10D banana and funnel catches any sign error or omitted Jacobian term immediately.
2. **Heavy-tail survival (`tier2_student_t_tails`):** Must pass Student-t ($\nu \in \{1.5, 3, 5\}$) quantile checks without tail blow-up or overflow.
3. **Net cost gain:** Total billed (`adapt` + `transition`) cost to gate on banana, Rosenbrock, and Neal's funnel must beat stock NUTS.
