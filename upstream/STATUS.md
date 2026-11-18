# Upstream ledger

This file records every upstream that apin carries, every patch made against one, and where each patch stands. Update it in the same commit that moves a submodule pin or changes the state of a patch.

## The rule on sending things upstream

Agents never post to the stan-dev, flatironinstitute or cmdstanr repositories: no issues, comments, pull requests or reviews. The owner sends things upstream personally. The Stan project's AI contribution policy (May 2026) also forbids agent-authored contributions. An agent may prepare a patch on the owner's fork, write the pull request text into the fork pull request, and update this ledger. Reading public issues and pull requests is allowed.

## Upstreams

Each upstream is a git submodule at `upstream/<name>/` with build notes in `upstream/<name>.md`. The pin is the commit that the recorded measurements used. The fork is where patches live; `origin` of each submodule is the fork.

| name | kind | upstream | fork | branch | pinned commit | licence | build notes |
|---|---|---|---|---|---|---|---|
| cmdstan | C++ driver (make) | stan-dev/cmdstan | sims1253/cmdstan | none (see note 1) | 6380837b29564e6f45045b1fa24f12a1ca52c7f7 | BSD-3-Clause | [cmdstan.md](cmdstan.md) |
| stan | C++ library | stan-dev/stan | sims1253/stan | none (develop commit) | d13c50c0fe7e4a4c387b727b81246cee2541a320 | BSD-3-Clause | [stan.md](stan.md) |
| math | C++ library | stan-dev/math | sims1253/math | none (see note 1) | e355b1453592d8758ed412d9e074642b74dbfc41 | BSD-3-Clause | [math.md](math.md) |
| stanc3 | OCaml compiler (dune) | stan-dev/stanc3 | sims1253/stanc3 | gathered-pcm-emit | d2081e6704bd03d90a9ba33e1e3bd0dc7084315a | BSD-3-Clause | [stanc3.md](stanc3.md) |
| bridgestan | C/C++ model library with C, Python, Julia, R and Rust bindings | roualdes/bridgestan | sims1253/bridgestan | isa-option | 16e7b3e80c0137d7be3475786e3c4c63db73ba14 | BSD-3-Clause (code), CC BY 4.0 (docs) | [bridgestan.md](bridgestan.md) |
| walnutpie | C++ sampler (CMake) | flatironinstitute/walnutpie | sims1253/walnutpie | exp/safe-adapt-defaults | 43b6435cef883b04af29be4031fb7d36bfcc024b | MIT | [walnutpie.md](walnutpie.md) |
| posteriordb | models, data, reference posteriors | stan-dev/posteriordb | none | master | 28f8d3d6e975315f42aa274a8399f21e07a43b30 | BSD-3-Clause; per-model `licence` field (all 21 core models: BSD3); data files carry no licence field | [posteriordb.md](posteriordb.md) |

nutpie is a Python dependency (the `nutpie` extra in `pyproject.toml`, MIT), not a submodule, because apin carries no patch to it.

Note 1. The math pin e355b14 and the cmdstan pin 6380837 are the commits the campaign measured. Their fork branches (math `gathered-pcm`, cmdstan `nindan/mixed-build-guard`) were rebased later, so neither commit is on a branch now. GitHub keeps unreferenced commits only for a while. The owner should push a tag for each pin to the fork (for example `apin/math-w133` and `apin/cmdstan-w133`) so that `git submodule update` keeps working.

Note 2. Dropped from the old repository: tinystan (pinned at db27f82, never used by any runner) and the stray gitlink `stan/external/math_dev_w126`, which pointed at the same math commit as `upstream/math` and so is replaced by it.

## Patch states

- merged upstream: the change is on the upstream default branch. The PR column names the upstream pull request.
- fixed upstream: upstream fixed the same defect with its own change. The fork patch can be closed.
- open upstream: an upstream pull request exists and is open.
- local only: the patch exists on the fork and has not been sent upstream.
- superseded: a later fork patch replaces it.
- abandoned: closed without merging and not replaced.
- unverified: upstream closed the related issue or pull request as completed, but nobody has confirmed that the closing change covers this patch.

The record column points at the measurement under `results/records/`. "PR body" means the fork pull request holds the only evidence. "worklog W-n" means the measurement is in the archived worklog only (see ARCHIVE.md).

## Patches

### math (sims1253/math)

| fork PR | branch | W | state | upstream | record |
|---|---|---|---|---|---|
| #1 | eigen-cluster-aware-adjoint | W-40 | local only; the upstream PR was closed unmerged | stan-dev/math#3367 closed; the bug is still on develop at 0329c4cb68 | cluster_adjoint_w40.md |
| #2 | square-pow-to-mul | W-33, W-50 | fixed upstream; fork PR closed | stan-dev/math#3373, commit 0c82cc96 | gp_micro_w33.md, errno_flags_w50.md |
| #3 | bernoulli-logit-partials-sign | W-46 | fixed upstream | stan-dev/math#3397 (merged 2026-09-15) | log1p_ceiling_w46.md |
| #4 | bernoulli-logit-glm-partials-sign | W-46 | fixed upstream | stan-dev/math#3397 (merged 2026-09-15) | log1p_ceiling_w46.md |
| #5 | soa-eltwise-batch-records | W-53, W-57..W-59, W-85, W-92 | unverified | stan-dev/math#3391 closed as completed 2026-09-09, no linked commit | soa_var_w53.md, soa_batch1_w57.md, soa_batch2_w58.md, soa_clang_w92.md |
| #6 | log1p-kernel-avx2 | W-46, W-103, W-107 | local only (draft) | | log1p_kernel_w103.md, kernel_ilp_w107.md |
| #14 | gathered-glm | W-108 | superseded by #17 | | gathered_glm_w108.md |
| #15 | gathered-normal-fmafix | W-112, W-112.1, W-112.2, W-118 | local only | | normal_gathered_w112.md, throwset_fix_w1122.md, fused_interior_w118.md |
| #16 | gathered-icar | W-113 | local only | | icar_gathered_w113.md |
| #17 | gathered-glm-mapfix | W-108.1, W-127 | local only | | mapfix_w1081.md, additive_gathered_w127.md |
| #18 | glm-edge-cleanup | W-120 | local only | | glm_edge_cleanup_w120.md |
| #19 | poisglm-fused | W-122 | local only | | poisglm_fused_w122.md |
| #21 | nb2-plain-rebuilt | W-123 | local only | | nb2_rebuild_w123.md |
| #22 | gathered-additive-tpvari | W-130 | local only | | tpvari_w130.md |
| #23 | gathered-pcm | W-126 | local only | | pcm_gathered_w126.md |

### stanc3 (sims1253/stanc3)

| fork PR | branch | W | state | upstream | record |
|---|---|---|---|---|---|
| #1 | fuse-eigendecompose-pair | W-32, W-39 | open upstream | stan-dev/stanc3#1677, opened by a maintainer and crediting the fork | eigh_reuse_w32.md, stanc3_w39.md |
| #7 | gathered-glm-emit | W-108 | superseded by #8 | | gathered_glm_emit_w108i2.md |
| #8 | gathered-registry | W-115 | local only | | gathered_registry_w115.md |
| #9 | const-hoist | W-124 | local only | | consthoist_w124.md |
| #10 | glm-default-emit | W-119, W-128 | unverified | stan-dev/stanc3#1697 closed as completed 2026-09-23 | glm_lane_w119.md, glm_default_emit_w128.md |
| #11 | gathered-additive-tp-emit | W-131 | local only | | additive_emit_w131.md |
| #12 | gathered-pcm-emit | W-132 | local only (this is the pinned branch) | | pcm_emit_w132.md |
| #25 | fix/categorical-inv-logit-rewrite | | fixed upstream | issue stan-dev/stanc3#1684, fixed by #1689 (merged 2026-08-31) | PR body |
| #26 | fix/partial-eval-short-circuit-rejected | | fixed upstream | issue #1685, fixed by #1687 (merged 2026-08-31) | PR body |
| #27 | fix/dce-observable-effects | | abandoned | issue #1686 closed as not planned; related #1746 completed | PR body |

### stan (sims1253/stan)

| fork PR | branch | W | state | upstream | record |
|---|---|---|---|---|---|
| #1 | scratch-hoist-base-nuts | W-24 | local only | | UPSTREAM_SUMMARY.md |
| #2 | gather-elimination | W-102 | local only (draft) | | gather_elim_w102.md |

### cmdstan (sims1253/cmdstan)

| fork PR | branch | W | state | upstream | record |
|---|---|---|---|---|---|
| #1 | nindan/mixed-build-guard | Phase 0, W-27 | local only | | march_native_w35.md, PHASE0_SUMMARY.md |

### bridgestan (sims1253/bridgestan)

| fork PR | branch | W | state | upstream | record |
|---|---|---|---|---|---|
| #1 | isa-option | W-105, W-105b | local only | | avx2_builds_w105.md, avx2_builds_w105b.md |

### walnutpie (sims1253/walnutpie)

| fork PR | branch | W | state | upstream | record |
|---|---|---|---|---|---|
| #1 | dev/pluggable-step-optimizers | W-1 | local only | | tables/w1_optimizers.csv |
| #2 | dev/adaptation-hardening | session 1 | local only (negative result) | | REPORT.md §2 |
| #3 | dev/mass-matrix-shrinkage | session 1 | local only (negative result) | | REPORT.md §2 |
| #4 | dev/init-robustness | session 1 | local only | | OVERNIGHT_2026-08-25_SUMMARY.md |
| #5 | dev/research-optimizer-notes | session 1 | local only (notes) | | PR body |
| #6 | upstream/adapt-with-stats | session 2 | local only | | PR body |
| #7 | robustness/init-guard | W-42 | local only | | init_guard_w42.md |
| #8 | robustness/freeze-clamp | W-41 | local only | | freeze_clamp_w41.md |
| #9 | robustness/step-heuristic-fix | W-43 | local only | | blr_pin_w43.md |
| #10 | rob/nan-alpha-guard | W-64 | superseded by #23 | | step_optimizer_w64.md |
| #11 | exp/warmup-trace | W-65 | local only | | worklog W-65 |
| #12 | fix/online-moments-lazy-delta | W-65, W-75 | merged upstream | flatironinstitute/walnutpie#101 | worklog W-75 |
| #13 | robustness/auto-screen-gates-full-operator | W-65 | local only (draft) | | lowrank_screened_w65.md |
| #14 | fix/multi-wrapper-dispatch | W-71 | local only (draft) | | worklog W-71 |
| #15 | robustness/sampler-depth-diagnostics | | local only (draft) | | PR body |
| #16 | fix/temporal-guard-chains1 | W-75 | local only (draft) | | worklog W-75 |
| #17 | robustness/init-eval-guard | W-78 | local only (draft) | | worklog W-78 |
| #18 | rob/init-screen | W-77 | local only (draft) | | worklog W-77 |
| #19 | tests/property-hunt | W-84 | local only (draft) | | worklog W-84 |
| #20 | robustness/mm2-guard | W-82, W-84 | local only (draft) | | mm2_guard_w82.md, mm2_domain_w84.md |
| #21 | fix/threads-so-guard | W-89 | local only (draft) | | worklog W-89 |
| #22 | exp/ridge-guard | W-86, W-88 | superseded by #24 | | ridge_guard_w88.md |
| #23 | fix/nonfinite-alpha-guard | | merged upstream | flatironinstitute/walnutpie#104 | PR body |
| #24 | feat/ridge-guard | W-102, W-110, W-125 | local only (draft) | | ridge_composed_w110.md, ridge_budget_w125.md |
| #25 | exp/cap-census-v2 | W-97 | local only (draft) | | worklog W-97 |
| #33 | fix/chain-view-checked-indices | | merged upstream | flatironinstitute/walnutpie#103 | PR body |
| #34 | perf/deterministic-endpoint-reuse | | merged upstream | flatironinstitute/walnutpie#105 | PR body |
| #35 | fix/validate-gradient-mass-init | | local only | | PR body |
| #36 | cleanup/pin-googletest-tested | | abandoned | | PR body |
| #37 | docs/refresh-contributor-cli-help | | local only | | PR body |
| #38 | fix/native-stan-thread-precondition | | local only | | PR body |
| #39 | fix/config-value-storage | | merged upstream | flatironinstitute/walnutpie#109 | PR body |
| #41 | fix/opt-in-early-exit | W-31 | local only | | session_benchmark_w36.md |
| #42 | fix/completed-trajectory-depth | | merged upstream | flatironinstitute/walnutpie#107 | PR body |
| #50 | fix/reject-degenerate-mass-init-count | | merged upstream | flatironinstitute/walnutpie#106 | PR body |
| #51 | fix/python-compat-and-output | | abandoned | | PR body |
| #52 | fix/example-output-defects | | local only | | PR body |
| #55 | fix/abs-step-convergence-deviation | | merged upstream | flatironinstitute/walnutpie#108 | PR body |
| #57 | perf/alloc-free-trajectory | | open upstream | flatironinstitute/walnutpie#112 | PR body |
| #58 | perf/fused-leapfrog-prologue | | local only | | PR body |
| #59 | perf/rng-bit-transforms | | local only | | PR body |
| #60 | wip/dense-metric | | local only | | PR body |
| #61 | fix/adapt-step-absolute-error | | local only | | PR body |
| #62 | perf/dense-metric-warmup | | local only | | PR body |
| #63 | fix/dense-agreement-calibration | | local only | | PR body |
| #64 | fix/dense-min-obs-default | | local only | | PR body |
| #65 | fix/dense-min-corr | | local only | | PR body |

flatironinstitute/walnutpie#95 was opened by mistake and closed; it carries no patch.

### Counts by state

| state | math | stanc3 | stan | cmdstan | bridgestan | walnutpie | total |
|---|---|---|---|---|---|---|---|
| merged upstream | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| fixed upstream | 3 | 2 | 0 | 0 | 0 | 0 | 5 |
| open upstream | 0 | 1 | 0 | 0 | 0 | 1 | 2 |
| local only | 10 | 4 | 2 | 1 | 1 | 34 | 52 |
| superseded | 1 | 1 | 0 | 0 | 0 | 2 | 4 |
| abandoned | 0 | 1 | 0 | 0 | 0 | 2 | 3 |
| unverified | 1 | 1 | 0 | 0 | 0 | 0 | 2 |
| total | 15 | 10 | 2 | 1 | 1 | 47 | 76 |

The table counts math #1 as local only: the upstream pull request was closed and the defect is still on develop.

## The two math bugs

The campaign found two defects in stan-math's gradients.

The bernoulli_logit sign bug (fork math#3 for `bernoulli_logit_lpmf`, math#4 for `bernoulli_logit_glm_lpmf`, found in W-46) is fixed upstream. stan-dev/math#3397, written by a maintainer from scratch after the fork pull request, merged on 2026-09-15 and covers both functions. A human must close fork math#3 and math#4 with a link to #3397. Nothing needs to be sent.

The eigenvector adjoint bug (fork math#1, found in W-40) is not fixed upstream. The reverse-mode adjoint of `eigenvectors_sym` and `eigendecompose_sym` is ill-conditioned when eigenvalues are close or equal. The upstream pull request stan-dev/math#3367 was closed without a merge, and develop at 0329c4cb68 still has the old adjoint. To send the fix, the owner must:

1. Read why #3367 was closed and decide whether a new pull request or an issue is the right form.
2. Rebase `eigen-cluster-aware-adjoint` on current develop and rerun the unit tests and the stan-math checklist (`make test-headers`, cpplint, doxygen, the targeted rev tests).
3. Rerun the W-40 evidence on the rebased branch: the finite-difference check and the kronecker_gp cross-build divergence (see `results/records/cluster_adjoint_w40.md`).
4. Write the pull request or issue in their own words, disclose the AI assistance as the Stan policy requires, and send it personally.

## Unverified

Two fork patches have an upstream issue or pull request that was closed as completed with no change that clearly covers the patch. Treat both as unverified until someone checks.

| fork PR | upstream | what is unknown | how to check |
|---|---|---|---|
| math#5 soa-eltwise-batch-records | stan-dev/math#3391, closed as completed 2026-09-09 after a sceptical review; no linked commit | whether any upstream change batches the arena records for the reverse eltwise ops | search develop for batched record allocation in `subtract`, `add`, `elt_multiply` and `elt_divide`; rerun `soa_batch1_w57.md`'s hier_2pl callgrind comparison against develop |
| stanc3#10 glm-default-emit | stan-dev/stanc3#1697, closed as completed 2026-09-23; the maintainer pointed at the GLM recognition in `Partial_evaluator.ml` lines 533-549 at 500c027 | whether that recognition covers the normal-regression forms measured in W-128 (kidscore_momiq, logmesquite_logvash) at the default optimisation level | compile both models with the upstream stanc3 at default level and look for `normal_id_glm_lpdf` in the generated C++ |

## Reports without a patch

The owner filed these issues upstream. None has a fork patch.

| repository | issue | state |
|---|---|---|
| stan-dev/math | #3379, #3380, #3385 | completed |
| stan-dev/math | #3381 | open |
| stan-dev/stan | #3420, #3421, #3424, #3426 | completed |
| stan-dev/stan | #3425 | open |
| stan-dev/stanc3 | #1695 | fixed by #1702 |
| stan-dev/stanc3 | #1696 | fixed by #1701 |
| roualdes/bridgestan | #336 (unsafe default `.so` under threaded evaluation) | closed as documented behaviour |

## Candidates

Upstreams that apin may carry later. A row moves to the Upstreams table when it gets a submodule and build notes. Leave the table empty until a candidate is chosen.

| name | kind | upstream | fork | branch | pinned commit | licence | build notes |
|---|---|---|---|---|---|---|---|
