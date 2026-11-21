# Clean-Machine Head-to-Head & Experiment E1 Record (2026-11-21)

**Harness commit:** `11cf21b` (`Numenomics/apin`, hardened post-review: out-of-process brokered Unix-socket IPC, calibrated omnibus + Gandy–Scott rank invariance, locked Lane A1 `adapt_metric` seam in `harness/judge.py`, paired hierarchical bootstrap, and checkpoint `cost_to_gate`).
**Machine:** 12-core Linux host, quiet (load average `< 0.35` before launch), sequential arms, `<= 4` chains per rep, default `BASE_SEED = 20260819`, exact manifest budgets (`4 × 1000` warmup + `5000` draws; `20 000` for `accel_gp`).
**Claim wording:** Passing a model means *passed the 0.1-SD bench gate at this budget*, never *exact* or *proven*.

---

## 1. Clean-Machine Head-to-Head: CmdStan NUTS vs. `nutpie` vs. `walnutpie` (16 Models × 5 Reps)

Arms compared sequentially on the same 16 scored models (covering all 5 families, including both `bym2_offset_only` and `diamonds`):
- **`cmdstan`**: CmdStan 2.39.0 NUTS (default settings)
- **`nutpie_diag`**: `nutpie 0.16.11` (`nuts-rs`, `adaptation=diag`, `save_warmup=True`)
- **`nutpie_lowrank`**: `nutpie 0.16.11` (`nuts-rs`, `adaptation=low_rank`, `save_warmup=True`)
- **`walnutpie_mw50`**: `walnutpie` `stan_cli` (`43b6435`, `--metric-window 50`, no Pathfinder inits, BridgeStan 2.9.0)

### Aggregate Ranking (Paired Hierarchical Bootstrap vs. `cmdstan`, 2000 reps)

| Rank | Sampler | Passed 0.1-SD Gate | ESS/cost vs. CmdStan [95% CI] (`p`) | Gate cost-speed [95% CI] (`p`) | ESS/s vs. CmdStan [95% CI] (`p`) |
|---|---|---:|---:|---:|---:|
| 1 | `cmdstan` | **14 / 16** | `1.00` | `1.00` | `1.00` |
| 2 | `nutpie_lowrank` | **10 / 16** | **5.51×** `[2.67, 13.63]` (`p=0.000`) | **5.42×** `[2.80, 12.38]` (`p=0.000`) | **3.87×** `[1.64, 10.82]` (`p=0.000`) |
| 3 | `nutpie_diag` | **10 / 16** | **1.59×** `[1.38, 1.83]` (`p=0.000`) | **1.54×** `[1.18, 1.91]` (`p=0.000`) | **1.42×** `[1.19, 1.70]` (`p=0.000`) |
| 4 | `walnutpie_mw50` | **5 / 16** | **0.82×** `[0.60, 1.03]` (`p=0.942`) | **1.04×** `[0.70, 1.43]` (`p=0.419`) | **0.97×** `[0.65, 1.43]` (`p=0.554`) |

### Per-Model Results (`ESS/cost`, `ESS/s`, `cost_to_gate`, or Gate Verdict)

| Model | `cmdstan` | `nutpie_lowrank` | `nutpie_diag` | `walnutpie_mw50` |
|---|---|---|---|---|
| `arma11` | **pass** `0.0224/cost`, `6922/s`, `109k` gate | `unproven`: R-hat 1.525 (`mu`) | `unproven`: R-hat 1.525 (`sigma`) | `wrong`: rep2 `SIGABRT` (-6) |
| `blr` | **pass** `0.00390/cost`, `5349/s`, `410k` gate | **pass** `0.0334/cost`, `21456/s`, `80k` gate (**8.6×**) | **pass** `0.00842/cost`, `7064/s`, `180k` gate | `unproven`: R-hat 4.141 (`sigma`) |
| `bym2_offset_only` | **pass** `0.000233/cost`, `2.86/s`, `5.51M` gate | `unproven`: R-hat 1.527 (`theta.936`) | `unproven`: R-hat 1.594 (`theta.231`) | `unproven`: R-hat 3.637 (`phi.1420`) |
| `diamonds` | **pass** `0.000165/cost`, `4.91/s`, `13.0M` gate | **pass** `0.0237/cost`, `709/s`, `142k` gate (**144×**) | **pass** `0.000176/cost`, `5.04/s`, `8.40M` gate | `unproven`: R-hat 2.553 (`b.2`) |
| `dogs_hierarchical` | **pass** `0.0231/cost`, `647/s`, `62k` gate | **pass** `0.0531/cost`, `1700/s`, `26k` gate (**2.3×**) | **pass** `0.0523/cost`, `1690/s`, `37k` gate | **pass** `0.0248/cost`, `927/s`, `50k` gate |
| `eight_schools_centered` | `wrong`: `lp__` mean `e=-0.274` | `wrong`: `lp__` mean `e=-0.298` | `unproven`: ESS 155 (`tau`) | `unproven`: R-hat 1.033 (`lp__`) |
| `eight_schools_noncentered` | **pass** `0.0177/cost`, `18548/s`, `145k` gate | `unproven`: `tau` sq `e=+0.0985` | **pass** `0.0227/cost`, `16622/s`, `148k` gate | **pass** `0.0189/cost`, `31649/s`, `160k` gate |
| `garch11` | **pass** `0.0103/cost`, `1100/s`, `172k` gate | **pass** `0.0236/cost`, `2333/s`, `50k` gate (**2.3×**) | **pass** `0.0176/cost`, `1775/s`, `187k` gate | **pass** `0.00469/cost`, `501/s`, `404k` gate |
| `gp_regr` | **pass** `0.0401/cost`, `11308/s`, `53k` gate | **pass** `0.0633/cost`, `13218/s`, `34k` gate (**1.6×**) | **pass** `0.0612/cost`, `13151/s`, `40k` gate | **pass** `0.0357/cost`, `10596/s`, `53k` gate |
| `kidscore_momiq` | **pass** `0.00327/cost`, `1268/s`, `569k` gate | **pass** `0.0442/cost`, `12528/s`, `49k` gate (**13.5×**) | `unproven`: `beta.2` mean `e=+0.0953` | `unproven`: `proj.1` sq `e=-0.0603` |
| `logmesquite_logvash` | **pass** `0.00256/cost`, `1835/s`, `715k` gate | **pass** `0.0331/cost`, `15337/s`, `73k` gate (**12.9×**) | **pass** `0.00309/cost`, `2320/s`, `854k` gate | `unproven`: R-hat 1.013 (`beta.3`) |
| `lotka_volterra` | **pass** `0.00168/cost`, `66.8/s`, `1.77M` gate | `unproven`: R-hat 1.229 (`theta.3`) | `wrong`: rep2 took `3154 s > 1800 s` | `unproven`: R-hat 1.027 (`theta.1`) |
| `low_dim_gauss_mix` | **pass** `0.0181/cost`, `262/s`, `130k` gate | **pass** `0.0347/cost`, `448/s`, `50k` gate (**1.9×**) | **pass** `0.0327/cost`, `438/s`, `64k` gate | `unproven`: R-hat 1.016 (`proj.5`) |
| `pilots` | `unproven`: R-hat 1.092 (`sigma_a`) | `wrong`: `lp__` `q95` `e=+0.204` | `unproven`: R-hat 1.103 (`b.4`) | `unproven`: R-hat 3.079 (`a.4`) |
| `radon_variable_intercept_slope_noncentered` | **pass** `0.00159/cost`, `109/s`, `1.14M` gate | **pass** `0.00295/cost`, `38.1/s`, `703k` gate (**1.9×** cost, **0.35×** wall) | **pass** `0.00280/cost`, `186/s`, `812k` gate | `unproven`: `proj.2` sq `e=+0.0689` |
| `wells_dist100_model` | **pass** `0.0281/cost`, `602/s`, `55k` gate | **pass** `0.115/cost`, `2417/s`, `14k` gate (**4.1×**) | **pass** `0.0446/cost`, `968/s`, `36k` gate | **pass** `0.0227/cost`, `499/s`, `48k` gate |

### Key Conclusions from the Head-to-Head
1. **Standalone `walnutpie` (`43b6435`, `--metric-window 50`, without Pathfinder inits) does not hold parity with CmdStan NUTS or `nutpie`:** It passed only **5 / 16** models (vs. 14 / 16 CmdStan and 10 / 16 `nutpie`), crashed with `SIGABRT` on `arma11` rep2, failed to mix on `blr`, `diamonds`, and `bym2_offset_only`, and was `0.82× [0.60, 1.03]` in ESS/cost (`0.97× [0.65, 1.43]` ESS/s) on the 5 shared passing models.
2. **`nutpie_lowrank` confirms massive correlated-posterior gains—and sharp open failure modes:** On the 10 shared passing models it reached **5.51×** ESS/cost (**144×** on `diamonds`, **13.5×** on `kidscore_momiq`, **12.9×** on `logmesquite_logvash`, **8.6×** on `blr`), but failed to pass `bym2_offset_only`, `arma11`, `lotka_volterra`, `eight_schools_noncentered`, and showed funnel bias (`wrong`) on both `eight_schools_centered` and `pilots`, plus a **2.9× wall-clock slowdown** on high-dim `radon_variable_intercept_slope_noncentered` (`38.1/s` vs. `109/s` CmdStan, `186/s` `nutpie_diag`).
3. **Quiet-machine `accel_gp` reconciliation (`BENCH.md:271`):** All 5 reps of `cmdstan` on `accel_gp` (`4 × 20 000` draws) completed in **324–385 s** (zero timeouts vs. `2945 s > 1800 s` under load in `BENCH.md`). Its clean-machine verdict is **`unproven: proj.17 square e=+0.0583 bound=0.144`**, confirming the earlier `wrong` verdict was solely load-induced timeout while the underlying rare excursion on `proj.17` remains.

---

## 2. Experiment E1: Diagnosing `walnutpie` W-63 vs. `nuts-rs` Low-Rank Fisher + First Lane A1 Result

### Root Cause of `walnutpie` W-63 (`0.037×` Regression vs. `nutpie_lowrank` `5.51×` Win)
Inspection of `upstream/walnutpie/include/walnutpie/low_rank_metric.hpp` (lines 95–111) reveals three concrete mathematical defects in `walnutpie`'s implementation of Seyboldt, Carlson & Carpenter (arXiv:2603.18845):
1. **Uncentered draws `Y` and scores `S`:** `stacked << Ys, Ss` uses raw uncentered draws, so the leading singular vector tracks the posterior mean $\mu \neq 0$ rather than the principal correlation axis of $\operatorname{Cov}(x)$.
2. **Wrong standardization power (`diag` instead of $\sqrt{\text{diag}}$):** `diag` is the inverse mass matrix ($\approx \sigma^2$), so multiplying by `diag.cwiseInverse()` and `diag` squares the scale distortion instead of whitening by $\sigma^{-1}$ and $\sigma$.
3. **Positive-only eigenvalue updates (`c >= 0`):** Computing `c = max(0, sv - sqrt(2K)) / denom` can only *inflate* `inv_mass` (`1 + c > 1`) and never distinguishes whether a large singular value came from `Y` (wide direction $\to \lambda > 1$) or from `S` (stiff direction $\to \lambda < 1$). Along a stiff correlated ridge where `S` spikes, `walnutpie` *inflated* `inv_mass` in the stiffest direction—forcing step-size collapse.

### Analytic Lane A1 Result (`e1_v1`, Locked `harness/judge.py::a1_transition`, Billed Warmup, Out-of-Process IPC)
Comparing `T0001` (diagonal dual-averaging baseline) against `T0005` (centered, $\sqrt{\text{diag}}$-whitened exact Riemannian matrix geometric mean $A^{1/2}(A^{1/2} B A^{1/2})^{-1/2} A^{1/2}$ with per-coordinate Stein-identity damping, directional draw–score linearity gating, eigenvalue soft-thresholding, and staged warmup leapfrog caps `24 -> 48 -> 96`) across **5 fresh paired seeds (`101..505`)** on the 8-target Tier 1 suite (`dim=6`):
- **Invariance pass rate:** **40 / 40 targets passed** for both arms.
- **Soft-min (`p = -4`) 5-seed geomean ratio (`T0005 / T0001`):** **`2.209×`** (won 5/5 seeds: `1.703×`, `3.205×`, `2.362×`, `1.392×`, `2.929×`).
- **Per-class 5-seed geomean ratios:** `hybrid_rosenbrock` **`2.231×`**, `student_t` **`1.270×`**, `gaussian_illcond` **`1.067×`** (`+2.64×` on rotated instance), `funnel` **`1.039×`**, `mixture_crossable` **`0.646×`**, `banana` **`0.469×`** (global low-rank still over-fits curved banana arms and bimodal separation axes).
- **Overall class geomean (paired hierarchical bootstrap):** `0.997× [0.627, 1.551]` (`p = 0.493`).

### Real-Posterior Lane A1 Validation (5 Public Stan Models × 3 Reps, Identical Locked `a1_transition` Kernel)
Both `a1_diag` (`T0001`) and `a1_fisher_lr` (`T0005`) executed via `apin run logdensity` (out-of-process brokered IPC, billed warmup + sampling, identical transition code) and scored with `apin score --submission ...`:
- **Passed 0.1-SD gate:** **4 / 5** for both (`blr`, `eight_schools_noncentered`, `gp_regr`, `garch11`); both failed `eight_schools_centered` (`a1_fisher_lr`: `wrong: lp__ mean e=-0.335`; `a1_diag`: `unproven: R-hat 1.017`).
- **Checkpoint `cost_to_gate` speedup (`gate_speed`):** **`1.96× [1.01, 5.05]` (`p = 0.0175`)**.
- **Final `ESS/cost` speedup:** **`1.61× [0.993, 3.44]` (`p = 0.035`)** (`1.32× [0.928, 2.32]` ESS/s; median Python overhead `4.9×` vs. `3.8×`).

| Model | `a1_diag` (`ESS/cost`, `cost_to_gate`) | `a1_fisher_lr` (`ESS/cost`, `cost_to_gate`) | `ESS/cost` Ratio | `cost_to_gate` Ratio |
|---|---|---|---:|---:|
| `blr` | `0.00285/cost`, `1,358k` gate | **`0.0145/cost`**, **`175k`** gate | **5.08×** | **7.77×** |
| `garch11` | `0.0146/cost`, `218k` gate | **`0.0196/cost`**, **`124k`** gate | **1.35×** | **1.76×** |
| `eight_schools_noncentered` | `0.0214/cost`, `160k` gate | **`0.0221/cost`**, **`115k`** gate | **1.03×** | **1.39×** |
| `gp_regr` | `0.0375/cost`, `77k` gate | `0.0361/cost`, `77k` gate | 0.96× | 1.00× |
| `eight_schools_centered` | `unproven`: R-hat 1.017 (`tau`) | `wrong`: `lp__` mean `e=-0.335` | — | — |
