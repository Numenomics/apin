# Draft Issue — `flatironinstitute/walnutpie`

> **IMPORTANT PRE-FLIGHT CHECK BEFORE FILING:**
> 1. This crash was observed on fork commit `sims1253/walnutpie@43b6435` (`exp/safe-adapt-defaults`). **Please reproduce against upstream `flatironinstitute/walnutpie` `main` first** before opening an issue upstream!
> 2. **Do NOT file the W-63 `low_rank_metric.hpp` defects upstream:** git history confirms `include/walnutpie/low_rank_metric.hpp` (`5302ed8`, `2f97cd6`) lives only on `sims1253/walnutpie`, not in `flatironinstitute/walnutpie`. (Fix those directly on the fork or in a future PR if upstreaming low-rank adaptation.)

---

**Suggested Title:** `WalnutsSampler` throws uncaught `std::invalid_argument("macro_time must be in (0, inf).")` (`SIGABRT`) after repeated `logp_grad` NaNs during warmup on `arma11`

## Summary

On `posteriordb`'s `arma11` model (BridgeStan 2.9.0, 4 chains, 1000 warmup + 5000 draws, `--metric-window 50`, no Pathfinder inits), warmup adaptation can collapse the step size / mass matrix when `logp_grad` repeatedly throws domain/NaN exceptions:

- On `rep2`, `chain_3` logged thousands of `normal_lpdf: Random variable is -nan` exceptions during warmup and then aborted with `SIGABRT` when constructing the post-warmup `WalnutsSampler`:
  ```text
  terminate called after throwing an instance of 'std::invalid_argument'
    what():  macro_time must be in (0, inf).
  ```
- Even on `rep2`, `chain_0` (which did not abort), adaptation converged to a degenerate diagonal mass matrix (`[3.33e-51, 5.88e-50, 3.88e-51, 6.51e-49]`) and stuck chain (`mu: stddev = 9.26e-16`).

## Where it triggers

In `include/walnutpie/walnuts.hpp` (constructor of `WalnutsSampler`):
```cpp
detail::validate_positive(inv_mass, "inv_mass");
detail::validate_positive(macro_time, "macro_time");
```
When warmup adaptation drives `macro_time` to `0.0` (underflow) or `NaN` after repeated invalid `logp_grad` evaluations, the adapted `macro_time` is passed directly into `WalnutsSampler` without clamping/guarding or returning a clean adaptation failure status.

## Reproducer (verify on upstream `main` `stan_cli`)

```bash
# Build arma11_model.so via BridgeStan 2.9.0 from posteriordb arma11.stan
# Run chain 3 of rep 2 (seed = 20262819 + 3 = 20262822, or sweep seeds 20260819..20265819)
./stan_cli \
  --model arma11_model.so \
  --data arma11.data.json \
  --num-warmup 1000 \
  --num-draws 5000 \
  --metric-window 50 \
  --seed 20262822 \
  --output chain_3.csv
```

## Suggested Fix Direction

1. Clamp adapted `macro_time` and diagonal `inv_mass` to safe finite positive bounds (e.g. `[1e-12, 1e6]` for `macro_time`, `[1e-12, 1e12]` for `inv_mass`) before freezing/constructing `WalnutsSampler`, and/or
2. Guard the dual-averaging / step-size update when a warmup trajectory fails with non-finite `logp_grad` so `log_step_bar` cannot underflow to `-inf` / `NaN` (and catch `std::invalid_argument` in `stan_cli` to exit cleanly with a diagnostic message rather than `SIGABRT`).

---

## Fork-Only Note (for `sims1253/walnutpie`, NOT upstream)

For fixing `sims1253/walnutpie`'s own `include/walnutpie/low_rank_metric.hpp` (lines 95–111, W-63):
1. Center `Y` and `S` (`Yc = Y - mean`, `Sc = S - mean`) before stacking.
2. Standardize by $\sigma = \sqrt{\text{diag}}$ (`Ys = diag^{-1/2} Yc`, `Ss = diag^{+1/2} Sc`), not `diag^{-1}` and `diag`.
3. Compute the exact symmetric geometric mean $A^{1/2}(A^{1/2} B A^{1/2})^{-1/2} A^{1/2}$ in the whitened subspace so stiff directions can shrink eigenvalues ($\lambda < 1$) instead of positive-only inflation (`c >= 0`).
