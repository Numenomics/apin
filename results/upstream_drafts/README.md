# Upstream Issue Drafts (Manual Owner Submission Only)

> **Standing rule (`upstream/STATUS.md`):** Agents never post to `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr`. Every file in this directory is a local draft prepared for the owner to review, edit into their own voice, verify against current upstream `main`/`develop` where noted, and submit manually.
>
> **Stan AI contribution policy (May 2026):** For any issue or PR sent to `stan-dev/*`, disclose any AI-assisted analysis/benchmarking per the Stan project policy and author the submission personally.

## Draft Index

| File | Target Repo | Topic | Status / Pre-flight Check |
|---|---|---|---|
| [01_nutpie_lowrank_overhead.md](01_nutpie_lowrank_overhead.md) | `pymc-devs/nutpie` / `nuts-rs` | `adaptation="low_rank"` adds 3.2–5.2× wall-clock cost per gradient on `radon` (`D=176`) and `bym2_offset_only` (`D=3842`) | Verified on `nutpie 0.16.11` (PyPI release, unpatched) |
| [02_nutpie_chain_trapping.md](02_nutpie_chain_trapping.md) | `pymc-devs/nutpie` / `nuts-rs` | Single-chain treedepth-10 / 100%-divergence trapping on `arma11`, `lotka_volterra`, and `bym2_offset_only` (plus `low_rank` funnel bias on `eight_schools_centered` & `pilots`) | Verified on `nutpie 0.16.11` (PyPI release, unpatched) |
| [03_cmdstan_pathfinder_diagnostics.md](03_cmdstan_pathfinder_diagnostics.md) | `stan-dev/cmdstan` | Pathfinder CSV omits total log-density/gradient evaluation counts; boundary-saturated draws (`rho=1.0`, `sigma=1.6e-125`) on `bym2_offset_only` | Verified on stock CmdStan 2.39.0 |
| [04_walnutpie_arma11_macro_time_abort.md](04_walnutpie_arma11_macro_time_abort.md) | `flatironinstitute/walnutpie` | Unguarded `macro_time`/mass collapse after repeated `logp_grad` NaNs throws uncaught `std::invalid_argument` (`SIGABRT`) on `arma11` | **Verify on upstream `main` first** (reproduced on fork commit `43b6435`; note W-63 `low_rank_metric.hpp` bugs are fork-only) |
| [05_stan_math_eigen_cluster_adjoint.md](05_stan_math_eigen_cluster_adjoint.md) | `stan-dev/math` | Reverse-mode `eigenvectors_sym` / `eigendecompose_sym` adjoint blows up / returns `NaN` on repeated or rounding-degenerate eigenvalues (`kronecker_gp`) | Re-check closed PR `#3367` discussion & current `develop` before filing |
