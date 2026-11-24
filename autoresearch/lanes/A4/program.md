# Autoresearch Lane A4: Warm-Start & Nonlinear Transport (`program.md`)

> **Karpathy-style 3-file contract for `Numenomics/apin`.**
> - **Read-only judge:** `harness/judge.py` (owns the verified unit-mass reversible HMC kernel `a4_transition` in latent $z$-space + automatic finite-difference Jacobian & round-trip inverse checks!)
> - **Single writable file:** `autoresearch/lanes/A4/submission.py`
> - **Contract:** Implement ONLY the 4 diffeomorphic transport functions:
>   - `adapt_transport(oracle, budget, rng) -> (transport_tuning, x0)`
>   - `forward(transport_tuning, z) -> (theta, log_abs_det_jac)`
>   - `pullback_grad(transport_tuning, z, grad_theta) -> grad_z`
>   - `inverse(transport_tuning, theta) -> z`
>   You may NOT define `adapt()` or `transition()`.

## 1. Objective

Learn a fast, frozen, invertible nonlinear transport map $T: z \mapsto \theta$ (and warm-start $x_0$) during `adapt_transport()` so that the pullback density
$$\log \tilde\pi(z) = \log \pi(T(z)) + \log|\det J_T(z)|$$
is close to an isotropic standard normal $\mathcal{N}(0, I_d)$ — **un-funneling** `NealsFunnel` / `eight_schools_centered`, straightening `Banana` / `HybridRosenbrock`, taming heavy tails (`StudentT`), and whitening rotated `IllCondGaussian` — while keeping `adapt_cost` low on both short ($H=150$) and long ($H=1000$) horizons.

## 2. Hard Rules (Non-Negotiable)

1. **Edit ONLY `autoresearch/lanes/A4/submission.py`** (plus `NEGATIVE_MEMORY.md` for $\ge 2$-trial clean falsifications).
2. **Respect `autoresearch/RULINGS.md`:** `requires = ["grad"]` only; no unpriced `hvp`/`hess`, no inspecting `oracle` internals, no target fingerprinting.
3. **Exact diffeomorphism verified automatically:** `harness/judge.py` checks round-trip invertibility (`T(T^{-1}(theta)) == theta`, `T^{-1}(T(z)) == z`), central finite-difference $\log|\det J_T(z)|$, and central finite-difference `pullback_grad(z, v) = J_T(z)^T v + \nabla_z \log|\det J_T(z)|` automatically in Tier 0 / `adapt`.
4. **Canonical `transport_tuning`:** Must include `"eps"` (float in `[1e-8, 10.0]`) and finite numpy/JSON-serializable parameters (no lambdas/objects).
5. **Simplicity rule:** A candidate within 2% of the run's best speed-up is kept as `keep_simpler` when it has fewer lines, fewer magic constants or at least 15% lower `overhead_ratio` than the incumbent.

## 3. Priority Starting Hypotheses

1. **Pathfinder / L-BFGS warm-start + Cholesky affine whitening:** Replace naive warmup with fast L-BFGS mode + inverse-Hessian approximation + short Fisher/MAD polish to slash `adapt_cost` while whitening rotated Gaussians.
2. **Diagonal Sinh-Arcsinh (SAS) / tail-power warp:** Elementwise bijection $y_i = \sinh(\delta_i \operatorname{arcsinh}(z_i) + \gamma_i)$ to Gaussianize heavy-tailed `StudentT(nu=1.5)` and skewed marginals cleanly with exact $O(d)$ Jacobian & gradient.
3. **Triangular affine-coupling / funnel-unwarping map:** Detect scale-dependent coupling ($\log\sigma_j(z_0) = a_j z_0 + b_j$) via regression of log-residuals on the top principal axis, directly un-funneling `NealsFunnel` and `eight_schools_centered` with closed-form triangular Jacobian!
4. **Quadratic triangular straightening (Banana / Rosenbrock):** Lower-triangular quadratic shift $y_j = \exp(s_j(z_{<j})) z_j + m_j(z_{<j})$ fitted from warmup draws.

## 4. Evaluation

```bash
uv run apin autoresearch eval --lane A4 --run-id <run_id> --hypothesis "..."   # prints the speed-up over start.py
# Real Stan posteriors (especially eight_schools_centered!):
uv run apin run logdensity --label a4_cand --opt sampler=autoresearch/lanes/A4/submission.py --opt lane=A4 --models blr,eight_schools_noncentered,eight_schools_centered,gp_regr,garch11 --reps 3
```
