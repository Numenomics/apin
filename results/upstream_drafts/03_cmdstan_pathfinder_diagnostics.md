# Draft Issue — `stan-dev/cmdstan` (or `stan-dev/stan`)

> **Reminder (Stan AI contribution policy, May 2026):** Edit into your own words and include an AI-assistance disclosure if submitting based on this benchmark analysis.

**Suggested Title:** Pathfinder: (1) report total log-density/gradient evaluations in CSV diagnostics, and (2) boundary-saturated draws (`rho=1.0`, `sigma=1.6e-125`) fail `param_unconstrain` on `bym2_offset_only`

*(Note: consider splitting these into two focused issues if preferred — (1) is an enhancement request for `cmdstan`/`stan`, (2) is a numerical edge case in Pathfinder's constrained output / PSIS resampling.)*

---

## Issue A: Pathfinder CSV output omits total log-density / gradient evaluation counts

### Problem
CmdStan NUTS writes `n_leapfrog__` per draw (and step-size adaptation diagnostics), enabling reproducible evaluation-cost comparisons (`ESS / gradient_eval`) across samplers and algorithms. By contrast, `method=pathfinder` writes `lp_approx__`, `lp__`, and `Elapsed Time: ... (Total)`, but does **not** report the total number of log-density or gradient evaluations performed during L-BFGS optimization, ELBO estimation, and importance weighting.

### Suggestion
Emit summary comments in the Pathfinder CSV footer (alongside `# Elapsed Time:`), e.g.:
```text
# Log density evaluations: <N>
# Gradient evaluations: <M>
```
(or per-path counts in single-path CSVs plus the sum in the multi-path output CSV). This would make Pathfinder directly comparable to HMC/NUTS in evaluation-cost benchmarks without needing custom instrumented builds.

---

## Issue B: Multi-path Pathfinder emits boundary-saturated draws on `bym2_offset_only` whose `param_unconstrain` / BridgeStan `log_density` is `NaN`

### Observation (CmdStan 2.39.0)
Running multi-path Pathfinder on `posteriordb`'s `bym2_offset_only` (`D = 3842`, default settings: `num_paths=4`, `num_draws=5000`, `num_psis_draws=5000`, seed `20260819`, chain `id=4`):

```bash
./bym2_offset_only method=pathfinder num_paths=4 num_draws=5000 num_psis_draws=5000 \
  data file=bym2_offset_only.data.json random seed=20260819 id=4 output file=pf_chain4.csv
```

The output CSV reports a finite `lp__` (`-2242.4665`) for all 5,000 draws, **but 27 PSIS-resampled draws have `rho = 1.0` (exact upper bound of `real<lower=0, upper=1>`) and `sigma = 1.6058408e-125`**. When those constrained draws are passed back through `param_unconstrain` (e.g. BridgeStan `2.9.0` `log_density` or Stan's own `generate_quantities` / unconstrained evaluation), `logit(1.0) = +inf` yields `NaN` log-density.

### Why it matters
Downstream tools that re-evaluate Pathfinder draws (such as PSIS diagnostics, BridgeStan verification, or using Pathfinder draws to initialize NUTS chains) silently receive boundary-saturated draws that cannot be unconstrained back to finite $\mathbb{R}^D$, even though Pathfinder's own CSV logged a finite `lp__` computed prior to the constrained-space rounding to `1.0`.

## Environment
- CmdStan `2.39.0` (stock release)
- BridgeStan `2.9.0`
- Model: `posteriordb` `bym2_offset_only`
