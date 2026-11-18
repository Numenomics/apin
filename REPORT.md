# Report: the 2026 campaign on Stan sampler wall time

This report covers the work from Phase 0 to W-133 (August and September 2026). It replaces the old `stan/results/FINAL_REPORT.md`, which stopped at W-56. Each headline number names the file under `results/` that holds it. Numbers that exist only in the archived worklog are listed in section 11.

## 1. How the work was done

The author planned the campaign, chose each experiment and made every decision recorded here. AI coding agents wrote most of the code, ran the measurements and drafted the records, under the author's direction and review. The fork pull requests were prepared the same way. Nothing was sent upstream by an agent; the owner sends upstream personally (`upstream/STATUS.md`). For this repository, the ESS and R-hat code was rewritten and checked against the R package posterior, and a two-model CmdStan grid and the test suite were rerun. No other number in this report was remeasured.

## 2. Question, metric and machine

The question was where the wall time of Stan sampling goes, and how much of it a patch to the samplers, the compiler or the math library can remove without changing the posterior.

The metric is bulk ESS of the worst variable (`ess_bulk_min`) per second, median over 3 reps, geometric mean over the 21 models of the core set, 4 chains of 1000 warmup and 1000 draws. The machine was a Ryzen 9 5900X (AVX2, no AVX-512), with at most 4 cores in use. PROTOCOL.md has the details and the two wall conventions. Up to W-56 the wall is the batch wall clock. W-109 and W-133 use the sum of per-chain sampler times.

## 3. Headline results

| result | value | trace |
|---|---|---|
| CmdStan 2.39.0 default on the core set | 4 of 21 models with R-hat > 1.01 | records/PHASE0_SUMMARY.md, tables/summary_variants.csv (`default`) |
| nutpie 0.16 against CmdStan | 1.21× faster wall, 2.6× cheaper per gradient, 0.98× ESS/s | records/PHASE0_SUMMARY.md |
| walnutpie 0.0.1 at its defaults | 17 of 21 models with R-hat > 1.01 | records/PHASE0_SUMMARY.md, tables/summary_variants.csv (`walnut`) |
| walnutpie after the adaptation fixes (batch 50, Pathfinder inits, mass clamp, step heuristic, window chop 50) | 8 of 21 models with R-hat > 1.01 | tables/summary_variants.csv (`w7_full_1000`) |
| `-march=native` | 0.88× ESS/s (1.13× slower) | tables/summary_variants.csv (`clang_native` against `default`) |
| walnutpie with threaded chains and endpoint-gradient reuse against stock walnutpie, sequential (W-36, 10 models) | 0.341 wall, 2.93× faster, draws bit-identical | records/session_benchmark_w36.md |
| eigenvector adjoint fix (W-40), kronecker_gp, healthy rep | bulk ESS 29.1 to 411.4, R-hat 1.13 to 1.02 | records/cluster_adjoint_w40.md, gate (d) |
| full stack against the recommended walnutpie default (W-109, E/S, 21 models) | 1.485× ESS/s | records/everything_ess_w109.md, records/everything_v2_w133.md §4a |
| full stack plus the gathered primitives (W-133, E′/S, 21 models) | 1.746× ESS/s (band 1.728 to 1.761) | records/everything_v2_w133.md §4 |
| the four models a gathered primitive applies to (W-133) | 3.19× (hier_2pl 5.54, radon_var 3.90, radon_pp 2.65, bym2 1.80) | records/everything_v2_w133.md §4a |
| the other 17 models (W-133) | 1.515×, unchanged by construction | records/everything_v2_w133.md §4a |

Read the W-109 and W-133 ratios with care. Both arms are walnutpie. S is walnutpie with Pathfinder inits and `--metric-window 50` on the stock model library. E adds a per-model sampler posture and a patched math library. No arm in W-109 or W-133 is CmdStan, so neither number says how the stack compares with CmdStan NUTS.

## 4. Findings to W-56

These are the old report's findings, kept with their corrections.

### 4.1 Where the time goes (Phase 0, W-29)

- The model gradient is 76 to 97 percent of steady wall on data-heavy models. Small or stiff models are bound by sampler overhead (pilots: memcpy 21 percent).
- In the profiled walnutpie runs, the gradient is 81.6 to 99.4 percent of instructions (records/hotspot_atlas_w29.md). The tape and arena cost a fixed 8 to 17 percent of the gradient.
- Validity checks cost at most 2.2 percent. Last-level cache misses are near zero; there is no memory wall at these sizes.
- `--Oexperimental` in stanc3: 3 of 21 models did not compile and 1 compiled wrong (records/PHASE0_SUMMARY.md). W-39 later found that all 21 compile with the newer stanc3; the vectorize pass applies to none of them.
- The widely reported `-march=native` crashes come from CmdStan reusing objects from a build with other flags. Fork PR cmdstan#1 adds a guard (records/march_native_w35.md).

### 4.2 walnutpie adaptation (W-1 to W-19)

At its defaults walnutpie froze most chains at their initial point. The gradient-seeded mass matrix was about 1e6, so the step was throttled, every trajectory diverged and the transition returned the start point. A mass clamp, a Stan-style step-size heuristic and Pathfinder inits removed the freeze. Batching the mass observations (stride 50) and chopping the Fisher window halved the failures. Shrinkage, clipping, drift guards, power-mean combination and stall resets all made things worse. More warmup did not help: stock walnutpie at 4000 iterations was worse than the fixed version at 1000.

A low-rank Fisher metric with an exact operator was built and property-tested. On the full core set it was indistinguishable from the diagonal metric (fail counts 14, 16 and 14 for the three arms, worklog W-17 only). Rep-to-rep variance on hier_2pl was larger than the difference between arms. Funnels stayed unsolved: they need a different sampler, not a different metric.

### 4.3 Sampler wins (W-23 to W-42)

- Endpoint-gradient reuse (W-23): one gradient call per transition removed, draws bit-identical.
- Parallel chains in one process (W-30): bit-identical to sequential runs, and 1.03× faster than four separate processes (worklog only).
- Cross-chain early exit was turned off by default (W-31). With it on, one model fell from 519 to 24 ESS (records/UPSTREAM_SUMMARY.md).
- W-36 measured the composed result against stock walnutpie: 2.93× faster wall on 10 models, of which 2.77× is parallelism and 1.056× is gradient reuse. Per-gradient cost rose 10 to 25 percent under 4-way concurrency (records/session_benchmark_w36.md).
- Warmup early exit was closed after four separate tests (W-21, W-25, W-28, W-37). The late gains of warmup are invisible to every cheap window statistic.
- W-41 to W-43: a guard that fails fast on non-finite initial values (8.2 s to 0.16 s per failed run; records/UPSTREAM_SUMMARY.md), a clamp at freeze time, and a repaired step-size search (blr bulk ESS 5 to 9 rose to 779 at 100 warmup iterations; records/blr_pin_w43.md).

### 4.4 Gradient cost and stan-math (W-32 to W-56)

- kronecker_gp (W-32): `eigendecompose_sym` in place of the separate eigenvalue and eigenvector calls, −19.4 percent instructions per gradient, bit-identical (records/eigh_reuse_w32.md). The stanc3 rewrite that does this automatically is fork stanc3#1, now open upstream as stan-dev/stanc3#1677.
- hier_2pl (W-34): element-wise plumbing is 40.4 percent of the gradient. A GEMM form of the model is −28.2 percent (records/hier2pl_plumbing_w34.md). This led to the gathered primitives of W-108 onward.
- `square()` in stan-math called `std::pow(x, 2)` (W-33): `x * x` is −9.1 percent on gp_regr. Upstream fixed it (stan-dev/math#3373).
- The eigenvector adjoint (W-40): the reverse-mode adjoint of `eigenvectors_sym` divides by eigenvalue gaps and is ill-conditioned when eigenvalues are close. The fork patch uses the minimum-norm adjoint with a gap guard; the well-separated path is byte-identical. Finite-difference error fell from 30 to 52 percent to about 1e-6. All kronecker_gp quality numbers before W-40 were measured with the wrong gradient. This bug is still on stan-math develop (`upstream/STATUS.md`).
- The log1p kernel ceiling (W-46): a fused AVX2 and FMA kernel is −22.8 percent instructions on hier_2pl. The work found the sign bug in the `bernoulli_logit` partials, fixed upstream by stan-dev/math#3397.
- Struct-of-arrays arena (W-47, W-53): a vertical slice was bit-identical at −7.7 percent of tape time.
- A general element-wise fusion in stanc3 (W-48) was built and rejected: stock code already makes one callback per operation, and the cost is the per-element gather.
- Speculative parallelism inside a chain (W-49) cannot win: the best case on 4 cores is 2.43×, against 2.77× for 4 chains.

## 5. Findings W-57 to W-133

### 5.1 The struct-of-arrays and kernel patches (W-57 to W-107)

- W-57 and W-58 batched the arena records of the reverse element-wise operations (fork math#5): hier_2pl −15.95 percent total and −17.06 percent gradient instructions, bit-exact (records/soa_batch1_w57.md, soa_batch2_w58.md). W-92 confirmed the win under clang: −16.3 and −17.4 percent, draws identical across both compilers (records/soa_clang_w92.md).
- W-102 removed a redundant gather in stan's indexing (fork stan#2): −3.63 percent (records/gather_elim_w102.md).
- W-103 put the log1p kernel into stan-math (fork math#6): hier_2pl −26.93 percent instructions per gradient (records/log1p_kernel_w103.md).
- W-105b built model libraries with `-mavx2 -mfma` through a BridgeStan option (fork bridgestan#1): −11.5 percent (hier_2pl) to −67.6 percent (diamonds) gradient instructions. Instruction savings are larger than wall savings: diamonds −67.6 percent instructions gave −20.7 percent wall (records/avx2_builds_w105b.md).

### 5.2 Sampler posture and the ridge guard (W-63 to W-125)

These were measured on the 10-model W-36 grid with walnutpie, not on the 21-model core set.

- W-74, Pathfinder inits on every model: +81.8 percent geometric-mean ESS at 5.1 percent wall cost. The pre-registered adoption gate still failed, on two drops in `ess_bulk_min` and two new aborts on accel_gp (records/pf_init_w74.md).
- W-88, a ridge guard that escapes a locked metric: ESS 330.1 to 519.6, +57.4 percent, no model worse (records/ridge_guard_w88.md).
- W-93, both together: ESS 329.9 to 1094.9, +231.9 percent (records/combined_posture_w93.md).
- W-110 measured the ridge guard at fixed budget on the four ridge-locked models: ESS rose 1.45 to 2.21× but wall rose 4.4 to 14.7×, so ESS/s was 0.150× (records/ridge_composed_w110.md). The guard is a floor for models that otherwise fail, not a speed-up. W-125 set its budget at `clamp(16F/5, 16, 128)` iterations with a floor of 64 (records/ridge_budget_w125.md).

### 5.3 Gathered primitives (W-108 to W-132)

A gathered primitive is one stan-math function for a likelihood whose terms index into parameter vectors, for example `bernoulli_logit(a[jj] .* (theta[ii] - b[jj]))` in an IRT model. The stock code builds one tape node per element; the primitive computes the value and all partials in one pass. The primitives from W-108 to W-113, W-126 and W-130 give md5-identical draws on the model they apply to and leave the compiler output byte-identical elsewhere. W-122 and W-124 were measured as instructions per element of the likelihood interior on a census model, not per gradient of a core model; W-124's draws were not required to be identical.

| W | primitive (fork PR) | model | instructions | trace |
|---|---|---|---|---|
| W-108, W-108.1 | `bernoulli_logit_lpmf_gathered` (math#14, then #17) | hier_2pl | −40.9% | records/gathered_glm_w108.md, mapfix_w1081.md |
| W-112, W-112.2 | `normal_lpdf_gathered` (math#15) | radon_pp, radon_var | −65.5%, −66.4% | records/normal_gathered_w112.md, throwset_fix_w1122.md |
| W-113 | `dot_self_gathered_diff`, the ICAR prior (math#16) | bym2_offset_only | −17.0% | records/icar_gathered_w113.md |
| W-118 | fused normal interior (math#15) | radon_pp | −35.4% gradient | records/fused_interior_w118.md |
| W-122 | fused `poisson_log_glm` interior (math#19) | census posture, N = 12573 | −21.0% per element | records/poisglm_fused_w122.md |
| W-124 | stanc3 hoists the constant `lgamma` term of Poisson likelihoods (stanc3#9) | census posture | −48.4% of the interior | records/consthoist_w124.md |
| W-126 | `pcm_lpdf_gathered` (math#23) | gpcm_latent_reg_irt (not in the core set) | −88.3%, 9.7× wall | records/pcm_gathered_w126.md |
| W-130 | gathered additive transformed parameters (math#22) | election88_full (not in the core set) | −67.5%, 4.1× wall | records/tpvari_w130.md |

stanc3 emits these primitives automatically at `--O1` (W-115 registry, fork stanc3#8; W-131 and W-132 add the additive and PCM rows, stanc3#11 and #12). W-118's fused interior was never measured at wall and is not in the W-133 table.

W-128 is a different class. stanc3 can rewrite a plain normal regression into `normal_id_glm_lpdf` at the default optimisation level (fork stanc3#10). The result is analytically equal but not bitwise equal: the log density agrees to about 2 ulp and the draws differ. On kidscore_momiq it removed 82.5 percent of the likelihood's instructions and 56.3 percent of the whole run; on logmesquite_logvash, 66.2 percent of the likelihood (records/glm_default_emit_w128.md). Because the draws change, it is not in the W-133 ratio. Upstream closed the matching issue as completed, which may or may not cover these models (`upstream/STATUS.md`, Unverified).

### 5.4 The final benchmarks (W-109, W-133)

W-109 ran 21 models × 2 arms × 3 reps × 4 chains (540 cells, no failures). S is walnutpie at default flags with Pathfinder inits and `--metric-window 50` on the stock model library. E adds the per-model MM2 posture (13 models on, 8 off) and a model library with math#5, the W-102 gather fix, the W-103 kernel and `-mavx2 -mfma`. E/S is 1.485× ESS/s: 1.467× from ESS and 0.982× from wall (records/everything_ess_w109.md).

W-133 ran no new samples. It replaced the E cells of the four models that a gathered primitive applies to with their measured E′ cells, whose draws are md5-identical to E. The result is 1.746× (band 1.728 to 1.761, depending on which of three hier_2pl wall readings is used). Absolute geometric-mean ESS/s rose from 38.81 (S) to 57.61 (E) to 67.75 (E′). The whole gain from E to E′ is wall time (records/everything_v2_w133.md).

## 6. Negative results

Each of these was measured and recorded so that nobody repeats it.

| W | what | result | trace |
|---|---|---|---|
| W-45 | warmup on a subsample of the data | the mass matrix fits a different posterior | records/subsampled_warmup_w45.md |
| W-54 | init buffer and soft gradient clipping from the Stan forums | both rejected | records/warmup_shields_w54.md |
| W-63 | low-rank metric (algorithm 1); metric window 100 | no gain; both rejected | records/lowrank_ess_w63.md, metric_window_w63.md |
| W-64 | other step-size optimizers | Adam stays | records/step_optimizer_w64.md |
| W-66 | low-rank threshold | closed | records/lowrank_threshold_w66.md |
| W-70 | metric windows 250 and 500 | no window passed the adoption gate | records/metric_window_w70.md |
| W-73 | Adam hyper-parameter sweep | no gain | records/adam_sweep_w73.md |
| W-74 | warmup of 400 or 700 iterations with Pathfinder inits | no-go on the pre-registered gates | records/warmup_truncation_w74.md |
| W-76 | tree depth cap | 0 of 15 cells passed | records/depthcap_w76.md |
| W-77 | freezing adaptation at 400 iterations | sampling gradients 2.22× more expensive at 0.53× ESS | records/adaptfreeze_w77.md |
| W-104 | batched domain checks in `normal_lpdf` | stopped at the audit, no code change | records/batched_checks_w104.md |
| W-110 | ridge guard as an ESS/s lever | 0.150× | records/ridge_composed_w110.md |
| W-129 | additive emission, first form | its bit-identity claim failed at the build gate; W-131 replaced it | records/additive_emit_w129.md |
| W-38 | looser error discipline, growing the micro-step count | rejected with mechanism | records/error_discipline_w38e2.md, grow_m_w38e4.md |
| W-48 | general element-wise fusion in stanc3 | rejected with mechanism | records/stanc3_fusion_w48.md |
| W-49 | speculative parallelism inside a chain | 2.43× ceiling, below 2.77× | records/speculative_w49.md |

W-62, W-69, W-70 (DEER), W-71R, W-72, W-78 and W-83 were also negative, but their numbers are only in the worklog (section 11).

## 7. Retractions and corrections

Each of these replaced an earlier claim. The original record keeps its text with the correction added.

- W-16: "fold 11.75× against rec" is retracted. The step-size search drew from `std::rand()`, seeded by the clock, so fixed-seed runs were not reproducible. The clean rerun gave fold 35 against rec 33. All numbers after W-16 come from the repaired binary.
- ESS per gradient against CmdStan: first reported as 0.06×, corrected to 0.31×. The walnutpie denominator had included warmup, and CmdStan's `n_leapfrog_sampling` counted only the last 500 of 1000 draws. The old `aggregate.py` was fixed; the new harness counts sampling-phase gradients only.
- W-27: "`-march=native` miscompiles kronecker_gp" is retracted by W-35 and W-40. The divergence between builds is permitted floating-point reordering, amplified by the ill-conditioned eigenvector adjoint.
- W-33: "x*x is bit-identical to pow(x, 2) (glibc)" is corrected by W-50: glibc's `pow(x, 2)` is 1 ulp off in about 0.08 percent of cases. The patch is still correct; the bit-identity claim was not.
- Every kronecker_gp quality number before W-40 was measured with a gradient that was 30 to 50 percent wrong.
- W-63: the screen claim was corrected in the record.
- W-74: an over-adaptation reading was corrected.
- W-75: the bym2 anomaly was retracted.
- W-101: the reported Pathfinder regressions were rep noise.
- W-71 is invalid as run. Its flags were parsed and ignored by one dispatch path.
- W-105 was aborted: the finite-difference tripwire fired on both arms.
- W-107 resolved an omission in a W-96 script.
- W-108.1 redefined the root cause of the W-108 mismatch as the FMA contraction schedule.
- W-112.1 found that the W-112 mismatch was not FMA but the set of thrown checks (fixed in W-112.2).
- W-118 carries an erratum on its band arithmetic.
- W-115: two of the three recorded md5 sums could not be reproduced at `--O1`. The cause was the optimisation level, not the patch.

## 8. Where sources disagree

- The old FINAL_REPORT section 6 (W-27) says `-march=native` miscompiles kronecker_gp. Its section 7 and the W-35 and W-40 records retract this. I believe W-40: the finite-difference and cross-build evidence in records/cluster_adjoint_w40.md shows the adjoint, not the compiler.
- records/UPSTREAM_SUMMARY.md says the `square()` patch is bit-identical under glibc. records/errno_flags_w50.md says `pow(x, 2)` is 1 ulp off in about 0.08 percent of cases. I believe W-50, which measured it.
- The W-109 heading expected 2.5 to 6× for the composed stack, and records/DECISION_DELTA_2026-08-29.md projected "composed E/S 2 to 3×". The measurement was 1.485×. The measurement stands; the projection was wrong.
- The ridge budget appears as `16×F/5` in DECISION_DELTA and as `clamp(16F/5, 16, 128)` with a floor of 64 in W-125 and W-133. These agree; W-125 is the complete form.

## 9. The sampler seam

### 9.1 The old runners

The old repository had one runner script per sampler. They shared a directory convention and a CSV of timings, and nothing else.

| runner | took | wrote | hard-wired |
|---|---|---|---|
| `run_grid.py` (CmdStan) | a variant name from the `VARIANTS` table in `compile_variant.py`, models from `harness/core_manifest.json` | `runs/<variant>/<model>/rep<r>/chain_<c>.csv`, `rows.csv`, `DONE` | the CmdStan path |
| `run_walnutpie.py` | a `--tag`, hand-wired flags plus `--extra-flags`, init files `chain_<c>.txt` | `chain_<c>.log` with timing stanzas, `rows.csv` with divergences and tree-depth hits set to -1 | the `stan_cli` path, and the BridgeStan path through `compile_bridgestan.py` |
| `run_nutpie.py` | models only | `runs/nutpie/...`, `rows.csv` with no warmup or sampling split | nothing; nutpie reports no elapsed time |
| `run_pathfinder.py` | models only | `init_<c>.json` per model, reused by every rep | the CmdStan path |

`rows.csv` had `warmup_s`, `sampling_s`, `n_leapfrog_total`, `n_leapfrog_sampling`, `divergences`, `treedepth_hits`, `accept_mean`, `stepsize_final`, `wall_batch_s` and `seed`. `n_leapfrog_sampling` counted the last 500 draws only (section 7). `compute_ess.py` walked the `DONE` markers, called `Rscript ess.R` and cached a JSON per rep. `aggregate.py` globbed every `rows.csv`, computed `ess_bulk_min / wall_batch_s`, took medians over reps and geometric means over models.

The runners differed in how they set the seed (CmdStan chain ids against walnutpie `seed + c`), in how they reported unknown values (-1 against missing), in where inits came from, and in what timing they could see. A new sampler needed a new runner script, a new entry in the aggregation globs and a new hard-wired path.

### 9.2 The contract now

The harness makes the contract explicit, and the change was small enough to do here:

- One interface, `harness/sampler.py`. An adapter subclasses `Sampler`, sets `name`, and implements `run(RunRequest) -> Timing`. `check()` and `describe()` are optional.
- One place that lists the runners: every module in `harness/adapters/` that defines `Adapter` is registered. `apin list` prints them.
- One output schema, written by `harness/grid.py` and documented in PROTOCOL.md: `chain_<c>.csv`, `run.json`, `DONE` or `FAILED`. Unknown values are null.
- One seed rule and one budget, set by the grid, not by the adapter.

To add a sampler, write one file in `harness/adapters/` that writes the chain CSVs and returns a `Timing`. Nothing else changes.

### 9.3 What the seam still assumes

- Models are Stan programs. A sampler written in Julia, Rust or JAX reaches them through BridgeStan's bindings, which compile the same `.stan` file into a library with a log density and gradient. A sampler that needs its own model language needs a model translation as well, and the core set has none.
- `Settings.cmdstan` is a Stan-specific field in the generic settings. It is there because two of the four adapters need it. Other adapters take their paths through `--opt` and environment variables.
- The score was ESS and cost only when this report was written. The correctness gate (PROTOCOL.md, "The correctness gate") now compares every rep with a reference posterior, and a sampler that is fast and biased is marked wrong.

## 10. Open questions

- Funnels (eight_schools_centered and similar) still fail with walnutpie at the default budget. The metric is not the cause (W-17, W-19).
- The per-gradient cost gap between CmdStan's services and a minimal BridgeStan driver was measured in Phase 0 and never closed.
- No benchmark compares the final walnutpie stack with CmdStan NUTS on the core set under one protocol. W-109 and W-133 compare walnutpie arms only.
- The W-118 fused interior was never measured at wall.

## 11. Numbers that cannot be traced to results/

These appear in the old report or the worklog, and no file in `results/` holds them. They are in the archive (ARCHIVE.md) or nowhere.

- Phase 0: "CmdStan services carry 2.1 to 5.0× more instructions per gradient than a minimal BridgeStan driver" and "model gradient is 76 to 97 percent of steady wall". The callgrind data were in `results/profile/` (49 MB, archived).
- Session 1: "geo ESS_min 26 to 294 (11×)" and the per-model freeze fixes (blr 4 to 406, arma11 3 to 1381, lsat 7 to 736). The R-hat counts trace to tables/summary_variants.csv; these ESS figures do not.
- Session 2: the W-16 retraction figures (fold 35/1.084, rec 33/1.089), the W-17 fail counts (14, 16, 14), the 0.31× ESS-per-gradient correction and the W-19 basis ablation (32, 30, 38, 35).
- W-30: "3.2× against sequential four-chain runs". W-36 traces; W-30 does not.
- W-109 and W-133 per-cell data: `scratch/w109/w109_results.json` and the W-133 cell JSONs were never committed. The tables in the two records are the only copy.
- Worklog only: W-61, W-62, W-65, W-69 to W-72, W-75, W-78, W-83, W-85, W-86, W-89, W-91 (the error cap; E+ 1.508× on 3 models is quoted in the W-133 record), W-94 to W-101, and part of W-106.
