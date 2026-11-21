# Technique Card: `pdmp-boomerang-zigzag`

- **Axis:** 3 (Non-Reversible Continuous-Time PDMP Dynamics)
- **Lanes:** `A3`, `A2`, `A0` | **Phase:** `adapt` + `transition` | **Requires:** `grad`
- **Source:** `mschauer/ZigZagBoomerang.jl` (`src/boomerang.jl`, MIT, SHA `691afe2200fa99d762b58f5c5fd7894cc9342294`)
- **References:** Bierkens, Grazzi, Kamatani, Roberts & Schauer (ICML 2020, *The Boomerang Sampler*); Bierkens, Fearnhead & Roberts (2019, *Zig-Zag*); Corbella, Spencer & Roberts (2022, *Automatic Zig-Zag*); Sutton & Fearnhead (2023, *Concave-Convex PDMP*).

## Core Idea
Piecewise Deterministic Markov Processes (PDMPs) evolve $(x_t, v_t)$ along deterministic trajectories interrupted by inhomogeneous Poisson reflection events at rate $\lambda(t) = \max(0, \langle v_t, -\nabla \log \pi(x_t) \rangle)$:
1. **Boomerang Sampler:** Elliptical orbits around a frozen Gaussian reference $(\mu, \Sigma)$ learned in `adapt()`:
   $x_t - \mu = (x_0 - \mu)\cos t + v_0 \sin t$, $v_t = -(x_0 - \mu)\sin t + v_0 \cos t$, with stationary velocity $v \sim \mathcal{N}(0, \Sigma)$ (`aux_spec = {"kind": "normal", "diag": ...}` or whitened `{"kind": "normal"}`). Reflections use the residual score $\nabla U_{\text{res}}(x) = -\nabla\log\pi(x) - \Sigma^{-1}(x-\mu)$ — so on an exact Gaussian target, event rate is **zero**!
2. **Zig-Zag / Bouncy Particle Sampler (BPS):** Straight-line flow $x_t = x_0 + t v_0$ with coordinate-wise flips (`v_i -> -v_i`) or hyperplane reflection (`v -> v - 2 (v·g / ||g||^2) g`) and periodic refreshment (`aux_spec = {"kind": "unit_sphere"}` for BPS or `{"kind": "normal"}` for Gaussian-velocity BPS).

## Black-Box Implementation Challenges (The Research Target!)
- Classical PDMPs assume an analytic Poisson upper bound $\bar{\lambda}(t) \ge \lambda(t)$ for exact thinning. On a black-box `oracle.log_density_gradient`, you must either:
  - Build a tight local/automatic upper bound (Corbella–Spencer–Roberts local optimization / Brent / secant bound over short horizon $\Delta t$, retrying on bound violation with exact restart), **or**
  - Use a **Metropolis-adjusted discrete-time Boomerang / randomized-orbit HMC** step (exact elliptical flow + MH/delayed-rejection reflection) that guarantees exact global balance with zero thinning-bound risk!
- Carry the velocity $v$ in `chain.aux` and declare its exact stationary law in `aux_spec`. Note: `invariance.py` checks **global balance** ($\pi \otimes \text{aux\_law}$ stationarity), so exact non-reversible PDMPs pass without needing detailed balance!

## Known Failure Modes & Validation
- **Bound violation bias:** If numerical thinning underestimates $\bar{\lambda}(t)$, `invariance.py` catches the drift immediately (`max_z` / `b2_max` — route unadjusted approximate PDMPs to Track L4 only).
- **Zero-refreshment reducibility:** BPS without occasional velocity refreshment can be reducible on spherical targets — include nonzero refreshment rate $\lambda_{\text{ref}} > 0$.
