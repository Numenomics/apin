# Upstream candidates

Research note, 18 November 2026. Question: which projects should apin add as upstreams next, in what order, and what technique does each hold that Stan's NUTS lacks?

Everything below was read on 18 November 2026. Licences were read from each repository's licence file in a shallow clone of the default branch. Last-commit dates are the commit date of that branch's head. Release dates come from the GitHub releases API or from PyPI, as marked. Paper numbers were read from the arXiv HTML or abstract page through a page-to-text tool. Ratios marked "computed" are mine, from the paper's own table. The section "Not verified" lists what I could not check.

## 1. Summary

Add these five, in this order:

1. **nutpie with nuts-rs** (Rust, MIT). One codebase already holds NUTS, low-rank Fisher-divergence mass matrices, a normalising-flow transform hook, WALNUTS and an experimental MCLMC. It reaches Stan models through the BridgeStan Rust crate. apin uses nutpie today but cannot patch it.
2. **BlackJAX** (Python/JAX, Apache-2.0). It is the widest catalogue of post-NUTS methods: MAMS (adjusted MCLMC), LAPS, MEADS, ChEES, GIST, nutpie's low-rank adaptation, Pathfinder adaptation and SMC.
3. **TensorFlow Probability, for the Inference Gym** (Apache-2.0). It supplies test problems with ground-truth means and variances and standard errors. apin cannot score correctness without something like it. The same checkout carries SNAPER-HMC and ChEES.
4. **The Julia stack: AdvancedHMC.jl, Pathfinder.jl and StanLogDensityProblems.jl** (all MIT). It is a second, independent NUTS implementation. It has a Pathfinder-seeded Woodbury metric and a nutpie-style adaptor, and Stan models reach it through one small bridge package.
5. **bayes-kit and the WALNUTS prototype** (Python and C++, MIT). These are the reference implementations of delayed-rejection HMC (DR-HMC and DRGHMC) and of WALNUTS. They target funnels, apin's main open problem (REPORT.md §10).

Three surprises:

- The idea of fitting a mass matrix from gradients by Fisher divergence has spread through the ecosystems: nutpie wrote it, BlackJAX ports it, AdvancedHMC ships `NutpieVar`, and walnutpie's adaptation is a continuous form of it. Its authors report a median 4× gain over Stan with the low-rank form. apin's own tests of a low-rank Fisher metric in walnutpie (W-17, W-63) found no gain. That conflict is the first thing to settle.
- nuts-rs has become a multi-sampler library: it now has WALNUTS and MCLMC modules as well as NUTS. It is a better base for a combined Stan-compatible sampler than any project that would have to be built from scratch.
- Several of the obvious sources are GPL-3.0 or carry no licence file: the MCLMC reference code, zeus, pocomc, the posteriordb Python package on PyPI and the MAMS/LAPS benchmark repository.

## 2. The baseline: what Stan's NUTS already does

CmdStan 2.39.0 (released 19 May 2026, per its RELEASE-NOTES.txt) samples with multinomial NUTS and biased progressive sampling. The metric is diagonal by default (`diag_e`), or dense (`dense_e`). It is adapted in doubling windows after an initial buffer, with a terminal buffer for the step size (`stan/src/stan/mcmc/windowed_adaptation.hpp`). The step size is set by Nesterov dual averaging to a target acceptance of 0.8. The variance estimate uses draws only, never gradients. CmdStan also has Pathfinder as a separate method (multi-path, PSIS resampling; options added in v2.34.0), but the sampler does not take Pathfinder's covariance as a metric on its own. A user can pass one with `metric_file`.

apin already carries walnutpie, which adds within-orbit step sizes (WALNUTS) and a continuous nutpie-style adaptation (its README). apin measured some of these ideas already. Pathfinder inits gave +81.8 percent geometric-mean ESS for walnutpie (W-74). A low-rank metric gave no gain (W-17, W-63). Funnels stay unsolved with walnutpie (REPORT.md §4.2, §10).

## 3. Candidate table

"Stan route" says how a `.stan` file reaches the sampler. BridgeStan has C, Python, Julia, R and Rust bindings (upstream/STATUS.md).

| candidate | URL | licence (file read) | language | Stan route | last release | last commit |
|---|---|---|---|---|---|---|
| nutpie | https://github.com/pymc-devs/nutpie | MIT | Rust core, Python | built in (`compile_stan_model`, bundles BridgeStan) | v0.16.11, GitHub 2026-06-18; PyPI upload 2026-06-30 | 2026-11-05 |
| nuts-rs | https://github.com/pymc-devs/nuts-rs | MIT | Rust | BridgeStan Rust crate (nutpie's `stan` feature uses `bridgestan = 2.7.0`) | v0.19.0, GitHub 2026-10-02; CHANGELOG lists 0.19.1 on 2026-10-15 | 2026-11-05 |
| BlackJAX | https://github.com/blackjax-devs/blackjax | **Apache-2.0** | Python/JAX | BridgeStan Python under `jax.pure_callback` (one point per call) | 1.7.1, 2026-10-02 | 2026-11-11 |
| NumPyro | https://github.com/pyro-ppl/numpyro | **Apache-2.0** | Python/JAX | same as BlackJAX; its NUTS needs a JAX potential | 0.22.0, 2026-09-18 | 2026-11-17 |
| TensorFlow Probability (incl. Inference Gym) | https://github.com/tensorflow/probability (`spinoffs/inference_gym`) | **Apache-2.0** | Python, TF/JAX | none; Gym models are TFP code | TFP v0.25.0, 2024-11-08; inference-gym 0.0.5 on PyPI, 2025-01-24; tfp-nightly 2026-11-17 | 2026-11-09 (repository head) |
| AdvancedHMC.jl | https://github.com/TuringLang/AdvancedHMC.jl | MIT | Julia | LogDensityProblems interface, via StanLogDensityProblems.jl over BridgeStan.jl | v0.8.8, 2026-10-17 | 2026-11-08 |
| StanLogDensityProblems.jl | https://github.com/sethaxen/StanLogDensityProblems.jl | MIT | Julia | is the bridge | not checked | 2026-10-26 |
| LogDensityProblems.jl | https://github.com/tpapp/LogDensityProblems.jl | MIT | Julia | interface only | v2.2.1, 2026-10-20 | 2026-10-20 |
| Pathfinder.jl | https://github.com/mlcolab/Pathfinder.jl | MIT | Julia | LogDensityProblems | v0.10.9, 2026-10-24 | 2026-10-26 |
| AdvancedVI.jl | https://github.com/TuringLang/AdvancedVI.jl | MIT | Julia | LogDensityProblems | v0.7.0, 2026-06-07 | 2026-09-30 |
| DynamicHMC.jl | https://github.com/tpapp/DynamicHMC.jl | MIT (Expat) | Julia | LogDensityProblems | v3.6.2, 2026-10-20 | 2026-10-29 |
| DynamicPPL.jl, Turing.jl | https://github.com/TuringLang/DynamicPPL.jl, https://github.com/TuringLang/Turing.jl | MIT | Julia | none (own model language) | v0.42.14 (2026-09-21), v0.49.0 (2026-09-13) | 2026-11-10, 2026-10-15 |
| Enzyme.jl, Mooncake.jl | https://github.com/EnzymeAD/Enzyme.jl, https://github.com/chalk-lab/Mooncake.jl | MIT | Julia (Enzyme is LLVM-level) | not needed: Stan Math supplies the gradient | v0.13.221 (2026-11-17), v0.5.70 (2026-11-10) | 2026-11-17, 2026-11-17 |
| PyMC | https://github.com/pymc-devs/pymc | **Apache-2.0** | Python | none (own model language) | v6.4.0, 2026-11-01 | 2026-11-14 |
| walnutpie (already in) | https://github.com/flatironinstitute/walnutpie (the old name `flatironinstitute/walnuts` redirects here) | MIT | C++ header-only, Python | BridgeStan (`load_stan.hpp`) | walnutpie 0.0.3 on PyPI, 2026-08-26 | 2026-10-01 |
| WALNUTS prototype | https://github.com/bob-carpenter/walnuts | MIT | Python and C++ (`WALNUTSpy`, `walnuts_cpp`) | none built in | none | 2025-11-25 |
| bayes-kit | https://github.com/flatironinstitute/bayes-kit | MIT (code), CC BY 4.0 (docs) | Python | BridgeStan Python (a plain log-density protocol) | not checked | 2023-10-30 |
| drghmc | https://github.com/giladturok/drghmc | MIT | Python | BridgeStan Python | none | 2025-03-10 |
| flowMC | https://github.com/GW-JAX-Team/flowMC (moved from `kazewong/flowMC`, whose last commit is 2025-10-28) | MIT | Python/JAX | `pure_callback`; flow training wants batched densities | v0.6.5, 2026-10-26 | 2026-10-29 |
| flowjax | https://github.com/danielward27/flowjax | MIT | Python/JAX | library of flows, used by nutpie's flow adaptation | v19.2.1, 2026-11-07 | 2026-11-07 |
| MicroCanonicalHMC (Python) | https://github.com/JakobRobnik/MicroCanonicalHMC | **GPL-3.0** | Python/JAX | `pure_callback` | not checked | 2025-12-09 |
| MicroCanonicalHMC.jl | https://github.com/JaimeRZP/MicroCanonicalHMC.jl | MIT | Julia | LogDensityProblems | 0.1.7, 2024-09-17 | 2024-09-17 |
| sampler-benchmarks (MAMS and LAPS papers) | https://github.com/reubenharry/sampler-benchmarks | **no licence file** (all rights reserved by default) | Python/JAX | none | none | 2026-05-14 |
| mici | https://github.com/matt-graham/mici | MIT | Python | BridgeStan Python, which has Hessians | v0.4.1, 2025-12-09 | 2026-06-01 |
| emcee | https://github.com/dfm/emcee | MIT | Python | BridgeStan Python (log density only) | v3.1.6, 2024-04-19 | 2026-11-12 |
| zeus | https://github.com/minaskar/zeus | **GPL-3.0** | Python | BridgeStan Python | not checked | 2024-02-18 |
| pocomc | https://github.com/minaskar/pocomc | **GPL-3.0** | Python | BridgeStan Python | not checked | 2024-09-20 |
| optax | https://github.com/google-deepmind/optax | **Apache-2.0** | Python/JAX | not a sampler | v0.2.9, 2026-11-11 | 2026-11-16 |
| schedule_free | https://github.com/facebookresearch/schedule_free | **Apache-2.0** | Python (PyTorch) | not a sampler | PyPI 1.4.1, 2025-03-24 | 2026-07-28 |
| prodigy | https://github.com/konstmish/prodigy | MIT | Python (PyTorch) | not a sampler | PyPI 1.1.2, 2025-01-16 | 2025-01-16 |
| dadaptation | https://github.com/facebookresearch/dadaptation | MIT | Python (PyTorch) | not a sampler | PyPI 3.2, 2023-11-27 | 2025-01-22 |
| kfac-jax | https://github.com/google-deepmind/kfac-jax | **Apache-2.0** | Python/JAX | not a sampler | PyPI 0.0.8, 2026-02-25 | 2026-10-07 |
| optimizers (Distributed Shampoo) | https://github.com/facebookresearch/optimizers | BSD-3-Clause | Python (PyTorch) | not a sampler | not checked | 2026-10-23 |

Licence flags. Nothing proposed for the shortlist is copyleft. Apache-2.0 (BlackJAX, NumPyro, TFP, PyMC, optax, kfac-jax, schedule_free) is permissive, but it is not BSD-3-Clause. Code translated line by line from it into a Stan patch would carry Apache terms (LICENSING.md, "Translated code"). Reimplement those methods from the paper. GPL-3.0 projects (MicroCanonicalHMC Python, zeus, pocomc) should not be submodules whose code is read for porting. Use the MIT or Apache reimplementations instead (BlackJAX for MCLMC). sampler-benchmarks has no licence, so it can be cited and read but not vendored. The posteriordb Python package on PyPI declares GPL (PyPI metadata, version 0.2.0, 2020). apin uses the database submodule, which is BSD-3-Clause, not that package. Keep it that way.

## 4. Candidates in detail

Each entry gives the techniques worth carrying over, the evidence, and the wiring. "Authors" means the method's own authors measured it. "Independent" means someone else did.

### 4.1 nutpie and nuts-rs

Techniques:

1. **Diagonal mass matrix from draws and gradients.** The scale is set from the variances of the draws and of the gradients together, which is the diagonal Fisher-divergence minimiser. Source: `nuts-rs/src/transform/diagonal.rs` ("Diagonal (per-dimension) affine transformation estimated from draw and gradient variance"). Paper: Seyboldt, Carlson and Carpenter, "Preconditioning Hamiltonian Monte Carlo by minimizing Fisher Divergence", arXiv:2603.18845 (v1, 19 March 2026).
2. **Low-rank plus diagonal mass matrix by Fisher divergence.** M⁻¹ = diag(σ)(I + U(Λ − I)Uᵀ)diag(σ) at O(dk) cost per operation. Source: `nuts-rs/src/transform/low_rank.rs`; options `adaptation="low_rank"`, `mass_matrix_eigval_cutoff`, `mass_matrix_gamma` (nutpie `docs/sampling-options.qmd`). Same paper, Algorithm 1.
3. **Normalising-flow transformation adapted in warmup ("Fisher HMC").** A flow is trained during warmup with a Fisher loss on draws and gradients (`nutpie/python/nutpie/transform_adapter.py`, class `FisherLoss`). It uses flowjax (`nnflow` extra in `pyproject.toml`). The docs call it experimental (`docs/nf-adapt.qmd`), and the docs' own example is a 100-dimensional funnel.

nuts-rs also has `src/walnuts.rs` (WALNUTS leaves, citing the walnutpie reference), `src/mclmc.rs` (unadjusted MCLMC, marked "highly experimental … may not return correct posteriors"), Adam and dual-averaging step-size modules (`src/stepsize/`), and an `external_adapt_strategy.rs` hook.

Evidence:

- Authors, arXiv:2603.18845 §4: 4 chains, 1000 warmup, 1000 draws, target acceptance 0.8, compared with "Stan's default NUTS implementation" on 114 posteriordb models that passed a filter (no divergences, ESS above 200 for all parameters). The diagonal Fisher estimator needed "a median ~0.75x the number of gradient evaluations as Stan for an effective sample", which gave a "~1.3x median speedup" in wall time. The low-rank estimator gave "a median ~4x speedup in both effective sample size per gradient evaluation and ESS per second". The paper reports no dense experiments and no flow numbers ("we plan to report them separately").
- Independent, apin Phase 0: nutpie 0.16 (diagonal) against CmdStan gave 1.21× faster wall, 2.6× cheaper per gradient and 0.98× ESS/s on the 21 core models (REPORT.md §3). That is not the paper's set or filter, but it is a weaker result than the paper's 1.3×.
- Independent, apin W-17 and W-63: a low-rank Fisher metric built into walnutpie was indistinguishable from diagonal (REPORT.md §4.2, §6). That was apin's own implementation inside a different sampler, not nutpie's `low_rank` mode.

Wiring: the nutpie adapter exists. Pass `adaptation` and the low-rank options through `--opt` (a few lines), and make nutpie and nuts-rs submodules with a fork branch so that patches can be measured.

### 4.2 BlackJAX

Techniques (all in `blackjax/` at commit e8db80f, 2026-11-11):

1. **MAMS, the Metropolis-adjusted microcanonical sampler.** Microcanonical dynamics with an MH correction and momentum refreshed only between steps (`mcmc/adjusted_mclmc_dynamic.py`, tuning in `adaptation/adjusted_mclmc_adaptation.py`). Paper: Robnik, Cohn-Gordon and Seljak, arXiv:2503.01707 (v2, 17 May 2025). The unadjusted parent is MCLMC: arXiv:2212.08549 and arXiv:2303.18221, `mcmc/mclmc.py`.
2. **LAPS, the late adjusted parallel sampler.** An ensemble of unadjusted MCLMC chains for burn-in, then MAMS, with step size set by an equipartition bias proxy and adaptation statistics pooled across chains (`adaptation/laps.py`, `adaptation/laps_burn_in.py`, `eca.py`). Paper: Robnik and Seljak, "Faster parallel MCMC: Metropolis adjustment is best served warm", arXiv:2601.16696 (23 January 2026).
3. **Ensemble trajectory-length and GHMC tuning.** ChEES-HMC (Hoffman, Radul and Sountsov, AISTATS 2021, PMLR 130:3907-3915; `adaptation/chees_adaptation.py`) and MEADS (Hoffman and Sountsov, AISTATS 2022, PMLR 151:7799-7813; `adaptation/meads_adaptation.py`).

Also present: GIST step-size and trajectory-length kernels (`mcmc/composed/`, citing Bou-Rabee, Carpenter and Marsden, arXiv:2404.15253), a port of nutpie's low-rank adaptation (`adaptation/low_rank_adaptation.py`, "following the nutpie reference implementation"), MCLMC with low-rank preconditioning (`adaptation/mclmc_lrd_adaptation.py`), Pathfinder adaptation, Stan-schedule window adaptation (`adaptation/staged_adaptation.py`), SMC, nested sampling and Laplace-marginal HMC.

Evidence (all from the authors, none independent):

- MAMS, arXiv:2503.01707 Table 1. Gradient calls to reach a worst-case second-moment error b²max < 0.01, against BlackJAX NUTS with window adaptation (not Stan). Ratios are NUTS calls ÷ MAMS calls, computed from the table: Gaussian 19,652 / 3,249 (6.0×), Banana 95,519 / 14,078 (6.8×), Bimodal 1.5×, Rosenbrock 1.7×, Cauchy 1.6×, Brownian 2.2×, German Credit 1.6×, ItemResp 1.7×, StochVol 843,768 / 430,088 (2.0×). On Funnel, NUTS did not converge within 10⁸ calls and MAMS needed 2,346,899, at a target acceptance of 0.99. The problems are mostly from the Inference Gym. No posteriordb model is used.
- LAPS, arXiv:2601.16696 Table 1. Gradient calls per chain to b²max = 0.01 with 4096 chains on an A100: NUTS / ChEES / MEADS / LAPS = Banana 320 / 264 / 390 / 17; Gaussian 9846 / 5138 / 5520 / 308; German Credit 2716 / 1168 / 610 / 300; Brownian 2159 / 600 / 410 / 206; IRT 1162 / 537 / 790 / 185; SV 4173 / 3080 / 2860 / 1325. Pathfinder converged only on two problems. The authors state that in total gradients the sequential methods are usually cheaper. The gain is wall time on parallel hardware. LAPS needs about 128 to 256 chains or more, and the authors list heavy tails and multimodality as unsolved.
- ChEES (abstract): it "typically yields larger effective sample sizes per gradient evaluation than NUTS does". MEADS: no numeric NUTS comparison on the abstract page.

Wiring: a BlackJAX adapter wraps BridgeStan's `log_density_gradient` in `jax.pure_callback` and runs 4 chains. The many-chain methods then run one model call per chain per step on the CPU. ESS per gradient is valid, but the wall time is not comparable to vectorised JAX models.

### 4.3 TensorFlow Probability and the Inference Gym

Techniques:

1. **Problems with ground truth.** Each target ships means and variances of named transformations, with standard errors where they are Monte Carlo estimates (`spinoffs/inference_gym/README.md`). Targets include banana, ill-conditioned Gaussian, Neal's funnel, German credit (logistic, probit, sparse logistic), item response theory, stochastic volatility on S&P 500 (full and small), radon contextual effects (Indiana, Minnesota), eight schools, Brownian motion with missing observations, the Lorenz bridge and a log-Gaussian Cox process (`inference_gym/targets/`, `targets/ground_truth/`).
2. **SNAPER-HMC.** Trajectory length adapted along the direction of largest variance (`tensorflow_probability/python/experimental/mcmc/snaper_hmc.py`; Sountsov and Hoffman, arXiv:2110.11576).
3. **Gradient-based trajectory-length adaptation (ChEES) and sharded many-chain kernels** (`experimental/mcmc/gradient_based_trajectory_length_adaptation.py`, `sharded.py`).

Evidence: the MAMS and LAPS papers above use the Gym as their benchmark. I did not read SNAPER's numbers.

Wiring: the Gym is not Stan. Each target used by apin needs a `.stan` translation, checked against the Gym's log density at random points. Then its ground truth feeds a new bias score in `apin score`. TFP's own releases are stale (v0.25.0, November 2024); pin a commit.

### 4.4 The Julia stack: AdvancedHMC.jl, Pathfinder.jl, StanLogDensityProblems.jl

Techniques:

1. **Pathfinder-seeded Woodbury metric.** `RankUpdateEuclideanMetric` and `WoodburyFactorization` in `AdvancedHMC.jl/src/metric.jl` hold a diagonal-plus-low-rank metric of the form Pathfinder's L-BFGS inverse Hessian produces (Zhang, Carpenter, Gelman and Vehtari, JMLR 23(306), 2022).
2. **Pathfinder to start, or replace, metric adaptation.** Pathfinder.jl's `docs/src/examples/initializing-hmc.md` lists three uses: inits only, starting the metric adaptation from the Pathfinder covariance (replacing the first window), or using it as the metric with no adaptation.
3. **Score-based Gaussian VI to fit a dense metric.** AdvancedVI.jl's batch-and-match (`src/algorithms/fisherminbatchmatch.jl`; Cai et al., ICML 2024) minimises a covariance-weighted Fisher divergence. That is the same objective family as nutpie's estimator, but solved as VI.

AdvancedHMC also has `NutpieVar` (`src/adaptation/massmatrix.jl`: Welford estimators of draw and gradient variance) and Riemannian HMC (`src/riemannian/`).

Evidence: Pathfinder's paper reports draws comparable to short dynamic-HMC runs at "one to two orders of magnitude fewer log density and gradient evaluations" (authors, JMLR abstract). That is a claim about approximation quality at low cost, not about NUTS efficiency after warmup. apin's W-74 (independent) found Pathfinder inits +81.8 percent geometric-mean ESS for walnutpie, at 5.1 percent wall cost. I found no measurement of Pathfinder-seeded metrics against Stan.

Wiring: one Julia script, called from a Python adapter, loads the BridgeStan library through StanLogDensityProblems.jl, runs AdvancedHMC (or DynamicHMC) and writes the chain CSVs.

### 4.5 bayes-kit, drghmc and the WALNUTS prototype

Techniques:

1. **Delayed-rejection HMC.** After a rejection, retry with geometrically smaller steps, with a retry probability that depends on the first proposal's acceptance (Modi, Barnett and Carpenter, Bayesian Analysis, doi:10.1214/23-BA1360, arXiv:2110.00610; `bayes_kit/drghmc.py`).
2. **DR generalised HMC.** Delayed rejection with partial momentum refresh (AISTATS 2025, arXiv:2406.02741; repository `giladturok/drghmc`).
3. **WALNUTS.** Within-orbit dyadic step-size choice under an energy-error bound (Bou-Rabee, Carpenter, Kleppe and Liu, arXiv:2506.18746, JMLR 27(113), 2026; Python and C++ prototypes in `bob-carpenter/walnuts`). apin's walnutpie is the production form.

Evidence:

- DR-HMC, authors (abstract): "up to five-fold performance gains over optimally-tuned HMC, as measured by effective sample size per gradient evaluation". The comparison is against HMC, not NUTS.
- WALNUTS, authors §4: on a high-dimensional Gaussian with a warm start, WALNUTS-D and NUTS are "comparable" in ESS per 1000 gradients. On the funnel and the Stock-Watson model, the evidence is QQ-plots, trace plots and energy errors at roughly equal cost, not an ESS ratio. The funnel iteration counts in the text and the figure captions do not agree.
- Independent: apin found walnutpie still fails on funnels at the default budget (REPORT.md §10).

Wiring: bayes-kit samplers take a model object with `log_density_gradient`, which BridgeStan's Python `StanModel` provides. A thin adapter runs 4 chains and writes CSVs.

### 4.6 Candidates outside the shortlist

- **NumPyro.** Iterative NUTS on JAX (arXiv:1912.11554) and NeuTra reparameterisation (`numpyro/infer/reparam.py`, `NeuTraReparam`; Hoffman et al., arXiv:1903.03704). It has HMCECS and ensemble samplers. Its NUTS is the same algorithm as Stan's, and NeuTra is covered better by nutpie's flow mode.
- **DynamicHMC.jl.** A clean Stan-style NUTS with initial step-size search and dual averaging (`src/mcmc.jl`). It is useful as Pathfinder.jl's documented partner, but holds no new technique.
- **flowMC.** Local MALA or HMC steps plus global proposals from a normalising flow trained on the chains (arXiv:2107.08001). Its target is multimodal astrophysics posteriors. It is relevant to `low_dim_gauss_mix` only. I read no numbers.
- **mici.** Manifold and constrained HMC, with Riemannian metrics that need Hessians. BridgeStan exposes Hessians. It is expensive per step and has no evidence against NUTS on posteriordb that I found.
- **PyMC samplers.** PyMC's default is NUTS (and it can call nutpie). DEMetropolisZ and SMC do not use gradients. Nothing there for a Stan-compatible gradient sampler.
- **emcee and zeus.** Affine-invariant ensemble samplers degrade in high dimension (Huijser, Goodman and Brewer, arXiv:1509.02230). zeus is GPL-3.0. They are a useful weak baseline for the language-model benchmark, not a source of ideas.

## 5. Machine-learning optimisers

The mass matrix must stay fixed while the sampler draws, or adaptation must vanish, or the chain loses its stationary distribution. The target is a distribution, not a point. Step size is tuned to an acceptance rate, not to a loss. Those three facts decide what carries over.

| idea | source | carries over? | why |
|---|---|---|---|
| Adam's diagonal second moment of gradients | Kingma and Ba, arXiv:1412.6980 | Already in, as an idea | For a Gaussian, the gradient variance is the inverse posterior variance, which nutpie's estimator already uses (combined with draw variance) in a principled form. Adam's exponential moving average is a weaker version of a windowed estimate. apin's W-64 and W-73 kept Adam for walnutpie's step size and found no gain from sweeping it. |
| Schedule-free averaging | Defazio et al., "The Road Less Scheduled", arXiv:2405.15682 | Mostly no | Stan's dual averaging already averages iterates (it returns x̄). Schedule-free removes the need to know the stopping time. Stan's warmup length is known in advance, so there is little to gain. |
| D-Adaptation, Prodigy | arXiv:2301.07733, arXiv:2306.06101 | No | They estimate the distance to the optimum to set a learning rate. Step-size tuning in HMC has a direct target (acceptance rate or energy error) with no such distance. |
| K-FAC | Martens and Grosse, arXiv:1503.05671 | Maybe, as a structure | The idea that transfers is a Kronecker-structured metric for matrix-shaped parameters (person × item, group × coefficient, the two GP axes in `kronecker_gp`). The factors would be fitted from draws and scores by Fisher divergence, not from a network's activations. Nobody has measured this for HMC that I found. |
| Shampoo | Gupta, Koren and Singer, arXiv:1802.09568 | Maybe, as K-FAC | It preconditions with Kronecker factors built from gradient outer products. Gradient outer products are half of the Fisher estimator. Its matrix-root updates are fine during warmup and must freeze afterwards. |

Plainly: none of these offers a step-size method better than what is in apin, and only the Kronecker structure is a new metric idea. Treat it as a research experiment (§7, E8), not a reason to add an optimiser library as an upstream.

## 6. Benchmarks and problem sets

| set | what it gives | reference posteriors | notes |
|---|---|---|---|
| posteriordb (already in) | 147 posteriors at pin 28f8d3d, Stan code and data | 46 reference draw sets at the pin; 11 of the 21 core models have one: eight_schools_noncentered, blr, kidscore_momiq, logmesquite_logvash, diamonds, gp_regr, accel_gp, garch11, lotka_volterra, low_dim_gauss_mix, arma11 | No hierarchical-family model and no funnel in the core set has reference draws. |
| Inference Gym | about 20 targets with ground-truth means and variances and standard errors | yes, per target | TFP code, not Stan. Overlaps posteriordb in eight schools and radon. Holds what posteriordb lacks: an exact ill-conditioned Gaussian, funnel, banana and large IRT and SV. |
| sampler-benchmarks (MAMS, LAPS) | the b²max bias metric (worst-case squared second-moment error, normalised by variance) and the scripts | uses Gym ground truth | No licence file. Read and cite; do not vendor. |
| nutpie paper set | 114 posteriordb models after a filter (arXiv:2603.18845; the per-model list was not read) | via posteriordb | The filter favours models where all samplers converge. |

For the language-model benchmark, the scoring cannot be ESS per second alone: REPORT.md §9.3 notes that "a sampler that is fast and biased scores well". The b²max metric against known moments is what the MAMS and LAPS authors use. It needs ground truth, which today exists for 11 core models plus whatever is translated from the Gym.

## 7. Recommendation

### 7.1 Shortlist

**1. nutpie and nuts-rs.** This ranks first because one Rust codebase already holds the three adaptation ideas with the best published evidence against Stan (diagonal Fisher, low-rank Fisher, flow transform), plus WALNUTS and MCLMC. It is MIT and reaches Stan models natively through BridgeStan. apin already runs nutpie but cannot change it: a submodule and fork branch turn it from a black box into a patchable upstream. It is also the best seed for the long-term Stan-compatible combined sampler. Adding a technique to nuts-rs is cheaper than building a new driver, and apin's harness already scores it. The first job is to replicate, or refute, the paper's median 4× for low-rank on apin's 21 models, because apin's own low-rank test disagrees.

**2. BlackJAX.** This ranks second because it is the reference implementation of most post-NUTS methods: MAMS, LAPS, MEADS, ChEES, GIST, Pathfinder adaptation and SMC. MAMS has the largest per-gradient gains reported against NUTS on standard problems (1.5 to 6.8× in the authors' Table 1). The cost is that JAX cannot vectorise a BridgeStan model, so many-chain wall-time results will not transfer. Measure ESS per gradient, and port what wins into nuts-rs or Stan from the paper (Apache-2.0 code should not be translated into BSD upstreams).

**3. TFP, for the Inference Gym.** This ranks third, above the Julia stack, because it fixes a hole in apin's own score, not because it holds a sampler. Without ground truth, the planned language-model benchmark would reward biased samplers, and the MCLMC family is exactly where bias is the open question (nuts-rs flags its own MCLMC as possibly wrong). Pin one commit. Translate about six Gym targets to Stan (ill-conditioned Gaussian, funnel, banana, IRT, SV small, German credit), check the log densities against the Gym, and add a bias column to `apin score`. That makes MAMS and LAPS comparable with their papers.

**4. AdvancedHMC.jl, Pathfinder.jl and StanLogDensityProblems.jl.** This ranks fourth because it gives apin a second NUTS implementation in a different language, so that algorithm effects can be separated from implementation effects (CmdStan, nutpie and walnutpie share BridgeStan or Stan Math, but their drivers differ). It also has the most complete set of Pathfinder-to-metric options. The techniques are not new in kind. The Woodbury metric seeded from Pathfinder is the one idea Stan lacks, and it can be tried in CmdStan through `metric_file` before any Julia is wired in. It meets the owner's aim of covering Turing's ecosystem at low cost, since the Stan bridge exists.

**5. bayes-kit with drghmc, and the WALNUTS prototype.** This ranks fifth because funnels are apin's main open failure, and these are the small, readable reference implementations of the two step-size-adaptive answers to them: delayed rejection and within-orbit step sizes. bayes-kit's last commit is from 2023, so treat it as a frozen reference, not an active upstream. The value is in a direct comparison on `eight_schools_centered` and the Gym funnel against walnutpie, GIST (BlackJAX) and nutpie's flow mode.

### 7.2 Rejected for now

| candidate | reason |
|---|---|
| DynamicPPL.jl, Turing.jl | They are model languages. Stan models never pass through them. |
| Enzyme.jl, Mooncake.jl | Stan Math supplies the gradient through BridgeStan. A Julia AD library only matters for models written in Julia. |
| LogDensityProblems.jl as its own upstream | It is an interface and comes in as a dependency of the Julia adapter. |
| NumPyro | Same NUTS as Stan. NeuTra is superseded for apin by nutpie's flow mode. Keep it as a reference only. |
| TFP samplers as a separate upstream | SNAPER and ChEES are reachable from the same TFP pin as the Gym. ChEES is in BlackJAX. TFP's releases stopped in November 2024. |
| PyMC | Its gradient sampler is NUTS or nutpie. Its other samplers do not use gradients. |
| emcee, zeus | They scale poorly with dimension. zeus is GPL-3.0. emcee is kept only as a weak baseline for the language-model benchmark. |
| pocomc | GPL-3.0. Its flow-preconditioned SMC is covered in spirit by nutpie's flows and BlackJAX SMC. |
| MicroCanonicalHMC (Python) | GPL-3.0. Use BlackJAX's MCLMC and MAMS instead. |
| MicroCanonicalHMC.jl | MIT, but its last release is from September 2024. BlackJAX is the maintained source. |
| flowMC | It targets multimodal problems that matter to one core model. Reconsider if the core set gains multimodal posteriors. |
| mici | Riemannian HMC needs Hessians at every step. No evidence against NUTS on Stan-like models was found. |
| optax, schedule_free, prodigy, dadaptation, kfac-jax, Shampoo | Not samplers. See §5: the one transferable idea (Kronecker metric structure) is an experiment, not an upstream. |

## 8. Techniques to try in a combined sampler

Expected gains are hypotheses for pre-registration (PROTOCOL.md, "Measuring a patch"), not results.

| # | experiment | expected gain | models it should help | cost |
|---|---|---|---|---|
| E1 | **Replicate nutpie's low-rank claim.** Run nutpie `adaptation="diag"` and `"low_rank"` (cutoff 2, γ 1e-5, as in the paper) against CmdStan on the 21 core models. Report ESS per gradient as well as ESS/s. | Paper: median 4×. apin W-17 and W-63: about 1×. Either answer changes the plan. | correlated posteriors: diamonds, blr, hier_2pl, kronecker_gp, gp_regr, radon pair | Low: an `--opt` passthrough in the nutpie adapter and one grid (about a day of machine time). |
| E2 | **Fisher-divergence metric inside Stan's windowed warmup.** Collect gradients alongside draws in `windowed_adaptation`, fit the diagonal (and then low-rank) Fisher estimator at each window end, and keep dual averaging and multinomial NUTS unchanged. | Paper's diagonal result: about 0.75× gradients per effective sample, about 1.3× wall. Low-rank, if E1 confirms it. | GLM and hierarchical families | Medium: a patch to stan's adaptation classes and a new metric type for low rank. It is a BSD-to-BSD port from nutpie (MIT), credited. |
| E3 | **Pathfinder-seeded metric in CmdStan.** Run multi-path Pathfinder, build the dense covariance (or its diagonal) of the best normal approximation, and pass it as `metric_file` with inits. Then shorten or skip the first slow windows. | Warmup cost down. W-74 already showed +81.8 percent ESS from inits alone in walnutpie. | models with long warmup: hier_2pl, radon pair, kronecker_gp | Low: no Stan patch. It is a variant of the existing pathfinder adapter. |
| E4 | **MAMS on Stan models.** Add a BlackJAX MAMS adapter over BridgeStan and compare ESS per gradient (and b²max where ground truth exists) with CmdStan. | Authors: 1.5 to 6.8× fewer gradients than NUTS on Gym problems. Stan models are untested. | smooth, high-dimensional posteriors: hier_2pl (IRT-like), radon, eight_schools_noncentered, garch11 | Medium: an adapter with `pure_callback`, 4 sequential chains. The wall time will be pessimistic. |
| E5 | **Funnel shoot-out.** On eight_schools_centered and a Stan translation of the Gym funnel, compare walnutpie, DR-HMC (bayes-kit), GIST step size (BlackJAX), nutpie flow adaptation and MAMS at acceptance 0.99. | Count of R-hat failures down from today's. MAMS converged on the Gym funnel where BlackJAX NUTS did not (authors). | stiff/funnel family | Medium: three thin adapters. Flow mode needs the JAX extra and has a compile overhead. |
| E6 | **Bias score from ground truth.** Add b²max (worst-case squared second-moment error over parameters, normalised by variance) against posteriordb reference draws for the 11 core models that have them, and against Gym ground truth. | No speed gain. It stops biased samplers, unadjusted MCLMC included, from scoring well. It is a prerequisite for the language-model benchmark. | all | Low to medium: harness work only. |
| E7 | **Low-rank metric for WALNUTS in nuts-rs.** nuts-rs has both WALNUTS and the low-rank transform. Turn both on together and measure against walnutpie's diagonal. | If E1 shows low-rank wins with NUTS, it should carry over. The step-size adaptivity and the metric address different problems (local scale against global correlation). | hierarchical plus GP | Low once nutpie and nuts-rs are submodules: it may be a configuration, not code. Check that the two compose in `nuts-rs/src/`. |
| E8 | **Kronecker-factored metric.** For parameters that BridgeStan names as matrices, fit A ⊗ B by Fisher divergence from draws and gradients (the K-FAC and Shampoo structure) and use it in nuts-rs's transform hook. | Unknown, and it may be nil. It is a dense-quality metric at near-diagonal cost when the posterior has that structure. | kronecker_gp, hier_2pl, models with matrix-shaped random effects | High: new estimator, shape metadata, property tests. Run it only after E1 and E2. |
| E9 | **Ensemble warmup on the CPU.** LAPS-style pooled adaptation over 32 to 64 short chains on 12 cores, then 4 chains continue sampling. | Uncertain on the CPU: the authors' gain is per-chain wall on 4096 GPU chains, and total gradients rise. | models where warmup dominates | High. Defer until a batched Stan log density exists. |

The three I would run first are E1, E2 and E4. E1 is cheap and settles the conflict between the nutpie paper and apin's own low-rank results. E2 is the most direct route to a better Stan: it changes only the estimator inside a warmup that Stan users already trust. E4 tests the method with the largest published per-gradient gains on Stan models for the first time. E6 should be built alongside them, because E4 and every MCLMC-family result need a bias check to mean anything.

## 9. Not verified

- The nutpie paper's per-model list, exclusions and per-model results. I read the abstract and the experiments section only, and could not confirm whether "effective sample" means minimum ESS or another aggregate.
- The flow-adaptation paper ("If only my posterior were normal: Introducing Fisher HMC", linked from nutpie's `docs/nf-adapt.qmd`). I did not open it, and nutpie's docs report no numbers.
- The MAMS table was read through a page-to-text tool. The tool flagged that the paper's text says the worst case is "a factor of two" while two table ratios are below 2 (Bimodal 1.5×, Rosenbrock 1.7×). I did not resolve this in the PDF.
- The GIST paper's experimental numbers. I read the abstract and BlackJAX's code only.
- The WALNUTS paper's funnel and Stock-Watson results beyond the qualitative summary. The paper has no ESS ratio there.
- The DRGHMC (AISTATS 2025) numbers. A search snippet mentioned a gain on Neal's funnel, but I did not read the paper, so no number is quoted.
- SNAPER-HMC's, ChEES's and MEADS's numeric results, and Pathfinder's numbers beyond the abstract.
- Whether nuts-rs's low-rank transform and its WALNUTS mode can be combined in one run (E7). I read only the module headers.
- That BridgeStan.jl works with StanLogDensityProblems.jl at BridgeStan 2.9.0 (PyPI, 2026-07-06). I read only its repository date and licence.
- Releases for StanLogDensityProblems.jl, bayes-kit, zeus, pocomc, MicroCanonicalHMC (Python) and Distributed Shampoo. I did not check them.
- The inference-gym target count is from the directory listing, not from a count of registered models.
- The Inference Gym's ground-truth standard errors are taken from its README. I did not open individual ground-truth files.
- The posteriordb reference-draw count (46) and the 11 core models with references come from the submodule's directory listing at 28f8d3d, matched against `models/manifest.json` by posterior name. I did not check that each draw file is complete.
- Which NUTS settings the nutpie paper used for Stan beyond "Stan's default NUTS implementation", and whether through CmdStan.
- That apin's Phase 0 nutpie result (0.98× ESS/s) and the paper's 1.3× measure the same thing. They differ in model set, filter and version, so they are not a contradiction in the strict sense.

## 10. Sources

Repositories (cloned 18 November 2026 at the default-branch head): the URLs in §3. Files cited by path are in those clones.

Papers:

- Seyboldt, Carlson, Carpenter. Preconditioning Hamiltonian Monte Carlo by minimizing Fisher Divergence. arXiv:2603.18845, 2026. https://arxiv.org/abs/2603.18845
- Robnik, Cohn-Gordon, Seljak. Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo. arXiv:2503.01707, 2025. https://arxiv.org/abs/2503.01707
- Robnik, Seljak. Faster parallel MCMC: Metropolis adjustment is best served warm. arXiv:2601.16696, 2026. https://arxiv.org/abs/2601.16696
- Microcanonical Hamiltonian Monte Carlo, arXiv:2212.08549, and Microcanonical Langevin Monte Carlo, arXiv:2303.18221 (IDs verified; author lists not checked).
- Hoffman, Radul, Sountsov. An adaptive-MCMC scheme for setting trajectory lengths in HMC. AISTATS 2021, PMLR 130:3907-3915. https://proceedings.mlr.press/v130/hoffman21a.html
- Hoffman, Sountsov. Tuning-free generalized Hamiltonian Monte Carlo. AISTATS 2022, PMLR 151:7799-7813. https://proceedings.mlr.press/v151/hoffman22a.html
- Sountsov, Hoffman. Focusing on difficult directions for learning HMC trajectory lengths. arXiv:2110.11576.
- Bou-Rabee, Carpenter, Marsden. GIST: Gibbs self-tuning for locally adaptive Hamiltonian Monte Carlo. arXiv:2404.15253 (v4, 24 June 2025).
- Bou-Rabee, Carpenter, Kleppe, Liu. The within-orbit adaptive leapfrog no-U-turn sampler. arXiv:2506.18746; JMLR 27(113), 2026.
- Modi, Barnett, Carpenter. Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions. arXiv:2110.00610; Bayesian Analysis, doi:10.1214/23-BA1360.
- Sampling from multiscale densities with delayed rejection generalized HMC. AISTATS 2025, arXiv:2406.02741 (not read).
- Zhang, Carpenter, Gelman, Vehtari. Pathfinder: Parallel quasi-Newton variational inference. JMLR 23(306), 2022. https://jmlr.org/papers/v23/21-0889.html
- Cai et al. Batch and match: black-box variational inference with a score-based divergence. ICML 2024 (cited in AdvancedVI docs, not read).
- Hoffman et al. NeuTra-lizing bad geometry in HMC via neural transport. arXiv:1903.03704 (not read).
- Phan, Pradhan, Jankowiak. Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv:1912.11554.
- Huijser, Goodman, Brewer. Properties of the affine invariant ensemble sampler in high dimensions. arXiv:1509.02230.
- Margossian et al. Nested R̂: assessing the convergence of MCMC when running many short chains. arXiv:2110.13017.
- Kingma, Ba. Adam. arXiv:1412.6980. Defazio et al. The Road Less Scheduled. arXiv:2405.15682. Defazio, Mishchenko. Learning-rate-free learning by D-Adaptation. arXiv:2301.07733. Mishchenko, Defazio. Prodigy. arXiv:2306.06101. Martens, Grosse. K-FAC. arXiv:1503.05671. Gupta, Koren, Singer. Shampoo. arXiv:1802.09568.
- Hoffman, Gelman. The No-U-Turn Sampler. arXiv:1111.4246.

apin's own evidence: REPORT.md §3, §4.2, §6, §9.3 and §10, and upstream/STATUS.md, in this repository.
