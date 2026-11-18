# The sampler benchmark

This file defines a benchmark for MCMC samplers written in Python. A submission gets a log density and its gradient, and nothing else about the model. It returns draws. The score is effective samples per evaluation of the log density, relative to Stan's NUTS, over a fixed set of models. A rep that fails the correctness gate scores almost nothing.

PROTOCOL.md defines the measurement (the grid, ESS, the gate). This file defines the task, the rules and the score. Where a rule is a judgement call, the file says what it recommends and gives one alternative.

## The task

Write one Python file that defines

```python
def sample(problem: Problem, budget: Budget, rng: numpy.random.Generator) -> numpy.ndarray:
    ...  # unconstrained draws of shape (budget.chains, budget.draws, problem.dim)
```

`Problem` and `Budget` are in `harness/logdensity.py`. PROTOCOL.md, "The log-density seam", gives the full contract. In short:

- `problem.log_density(theta)` and `problem.log_density_gradient(theta)` evaluate the model on the unconstrained space, with the Jacobian and all constants. A point the model rejects gives −inf.
- `problem.initial_points` has one starting point per chain, uniform on (−2, 2).
- Every returned draw must be a point the submission evaluated. A chain must return its draws in the order it visited them, and no point may appear in two chains.

Run a submission on the public models with

```bash
uv run apin run --sampler logdensity --label mine --opt sampler=path/to/mine.py --models public --reps 3
uv run apin run --sampler cmdstan --label cmdstan --models public --reps 3
uv run apin score --baseline cmdstan
```

`examples/samplers/hmc.py` is a complete submission of about 100 lines. `examples/samplers/ula_wrong.py` is wrong on purpose and shows what the gate does with a fast, biased sampler.

## Rules

A submission may import numpy, scipy and the standard modules math, typing, dataclasses, functools, itertools, collections and `__future__`. It may not name `open`, `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `input`, `globals` or `vars`. `apin run` checks both before it starts.

A submission must not:

- read or write files, use the network, start processes or threads, or keep any state between runs;
- identify the model, from its dimension or by probing the density, in order to look up settings tuned for it;
- reach the model by any route other than the two functions in `problem`.

The import check is a lint. It does not stop a submission that sets out to break these rules (`getattr` on the builtins gets past it). An official round therefore reads every submission by hand, and runs it under the limits below. A real sandbox is not built yet.

## The budget

Each rep gives the submission:

| | value | why |
|---|---|---|
| chains | 4 | as CmdStan; R-hat needs several chains |
| draws | 1000 per chain | as CmdStan |
| evaluations | 4 × 2000 × 1023 = 8 184 000 | the most NUTS can spend with 1000 warmup and 1000 sampling iterations at tree depth 10 |
| time | 1800 s on one core | `OMP_NUM_THREADS=1` and the BLAS equivalents; the process is killed 300 s later |

The evaluation limit does not bind for a sensible sampler. On the small core models NUTS spends 35 000 to 100 000 gradients in its sampling phase, 1 to 3 % of the limit with warmup. Only a model that drives NUTS to its maximum tree depth on every iteration comes close. The limit is there to stop a submission that thins by a factor of thousands. The time limit stops a submission that spends its time in numpy between evaluations.

**Recommendation:** density and gradient calls count the same. Most samplers that use the density alone (random-walk Metropolis, slice sampling) are not competitive per evaluation anyway, and the cost of a gradient in reverse mode is a small multiple of the density. *Alternative:* charge a gradient 2 to 4 density calls, measured per model with BridgeStan. This favours gradient-free samplers on cheap models.

## The score

For each held-out model m and each rep r:

1. If the rep failed (an exception, the budget exceeded, a provenance error) or the gate marks it wrong, its relative score is PENALTY = 0.001.
2. Otherwise its relative score is `ess_per_eval` of the rep divided by the median `ess_per_eval` of CmdStan's NUTS on m. `ess_per_eval` is min(bulk ESS, tail ESS) over all variables, divided by every evaluation, warmup included (PROTOCOL.md, "The score").

The model's score is the median over reps. The benchmark score is the geometric mean over models. `apin score --baseline cmdstan` writes it to `bench.csv` and prints it with a bootstrap 95 % interval, from 2000 resamples of the reps within each model.

A score of 1 means "as efficient as NUTS per evaluation". A score of 2 means half the evaluations for the same ESS.

**The penalty.** One wrong model out of 11 multiplies the score by 0.001^(1/11) ≈ 0.53. A submission that is twice as fast as NUTS everywhere and wrong on one model scores about 1.07. Two wrong models give 0.29 times the score. The penalty is finite so that one failure does not hide the rest of the result. *Alternative:* score a wrong model as 0 and a submission with any wrong model as 0. This is simpler, but it makes the score useless for a sampler in development.

**Models where NUTS is wrong.** When CmdStan's own reps are wrong on a model, the model has no baseline and is left out. `bench.csv` lists it under "excluded". This can happen on eight_schools_centered, where NUTS has divergences. *Alternative:* keep the model and score any correct submission there at a fixed bonus. That rewards the samplers that solve the hard case, but the bonus would be arbitrary.

**Wrong reps.** The gate judges each rep alone, and its power depends on the rep's ESS. A sampler with a small bias can pass in some reps and fail in others. The median over reps then forgives one wrong rep in three, or two in five. The demo run below shows this twice: the wrong ULA passed one rep of garch11, and HMC failed one rep of eight_schools_centered on the mean of tau. Both models still came out right, wrong for ULA and correct for HMC. *Alternative:* a model whose reps include any wrong one scores PENALTY. The gate's false-fail rate is at most 1 % per rep, so a correct sampler would lose a model about once in 20 models of 5 reps. That rule is stricter on samplers that are nearly right, and it would have marked HMC wrong on eight_schools_centered.

**Models without a reference.** The gate cannot check a model that has no reference posterior, so a wrong sampler would score there. Such a model is left out and listed under "excluded". Today this is kronecker_gp, so the held-out score covers 10 models. *Alternative:* score it with the gate's other checks only (R-hat and finiteness). That keeps the model but lets a sampler that is consistently biased through.

**Why per evaluation.** ESS per evaluation does not depend on the machine, on load, or on how well the submission uses numpy. A submission in pure Python competes with C++ NUTS. *Alternative:* `ess_per_s_chain`, ESS per second of one core. That measures what a user waits for, but it mostly measures how well the overhead was vectorised. Both columns are in `table.md`.

**Why min(bulk, tail).** Bulk ESS alone rewards a sampler that mixes the centre and never visits the tails.

## The split

The 21 core models are split into 10 public and 11 held-out models. Each family is in both halves.

| family | public | held-out |
|---|---|---|
| easy/small | eight_schools_noncentered, blr | kidscore_momiq, lsat_model |
| GLM | diamonds | logmesquite_logvash, wells_dist100_model |
| hierarchical | radon_partially_pooled_noncentered, pilots | radon_variable_intercept_slope_noncentered, dogs_hierarchical, hier_2pl |
| stiff/funnel | eight_schools_centered, garch11, low_dim_gauss_mix | lotka_volterra, arma11 |
| GP/spatial | gp_regr, bym2_offset_only | accel_gp, kronecker_gp |

The reasons:

- Each public model shows a hard feature of its family. diamonds has strong correlations, pilots and eight_schools_centered have funnels, low_dim_gauss_mix is a two-component mixture, bym2_offset_only has a sparse spatial prior. A submission tuned on the public set has seen each difficulty.
- The two radon models share data but differ in structure. One goes in each half, so a submission cannot learn the radon data from the public half and score it twice.
- The held-out half has the one ODE model (lotka_volterra), the one item-response model (hier_2pl) and the one Kronecker GP. These are the models least like anything in the public half, so they test whether a sampler generalises.
- `models/manifest.json` records the split (`"bench"`). `--models public` and `--models held-out` select it.

The repository is public, so the held-out models are not secret. Anyone can run them. The split states the intent: tune on the public half and report on the held-out half. A held-out set that is really hidden needs a private manifest, or the same models with data that is not published. That is not built yet.

**Recommendation:** report the held-out score as the result and the public score next to it. A large gap between the two shows overfitting. *Alternative:* score all 21 models. That gives a smaller interval but no check on overfitting.

## Seeds and reps

Rep r uses seed BASE + 1000·r for the initial points and for `rng`. During development BASE is 20260819, the default. An official run uses a BASE that is kept secret until the run is done, passed with `apin run --base-seed`. A submission therefore cannot be tuned to the initial points it will see.

An official result uses 5 reps. Development runs use 3, the grid default. The model score is a median, so one unlucky rep in 5 does not change it. The bootstrap interval shows how much the score would move with other reps. Two submissions whose intervals overlap are not ranked.

## A demo run

One run on 5 public and held-out models, 3 reps, on a loaded 12-core machine: CmdStan 2.39 NUTS, `examples/samplers/hmc.py`, and `examples/samplers/ula_wrong.py`. Cells are the median ESS per evaluation over reps, or the gate's reason when the model is wrong.

| model | CmdStan NUTS | HMC example | ULA (wrong) |
|---|---|---|---|
| blr | 0.00594 | 0.00485 | wrong: R-hat 1.585 on sigma |
| eight_schools_centered | wrong: R-hat 1.060 on tau | 0.00206 | wrong: R-hat 1.708 on theta.2 |
| eight_schools_noncentered | 0.0238 | 0.0407 | wrong: R-hat 1.598 on tau |
| garch11 | 0.0198 | 0.0309 | wrong: alpha1 var b² = 0.0538 |
| kidscore_momiq | 0.00457 | 0.00631 | wrong: R-hat 3.090 on beta.1 |

Benchmark score against CmdStan, over the 4 models where CmdStan is right: HMC 1.32 [1.16, 1.38], ULA 0.001 [0.001, 0.0056]. eight_schools_centered is left out because NUTS fails 2 of its 3 reps on R-hat.

HMC scores above NUTS per evaluation on these small models. A likely reason is that NUTS doubles its trajectory until it turns back, which overshoots on simple targets; this run does not test that. NUTS is still faster per second, since it is compiled and the example is not (`ess_per_s_chain` 159 against 301, geometric means over the 5 models).

## Ways to game the score

| attack | defence |
|---|---|
| Return draws from a stored reference, or from a Laplace or variational fit, without paying for them. | Every draw must be a point the submission evaluated (provenance check). File access is forbidden; the references are not in the problem. |
| Shuffle the draws within a chain, which hides autocorrelation and inflates ESS. | Draws must keep the order in which they were first evaluated. |
| Copy one chain four times, so that R-hat is 1 and ESS counts four times. | No point may appear in two chains. |
| Spend evaluations in warmup and report only the sampling phase. | Every call counts, warmup included. |
| Thin a long chain by a large factor. | Allowed, and counted: thinning costs evaluations and ESS per evaluation shows it. The evaluation limit bounds it. |
| Be fast and biased. | The correctness gate (PROTOCOL.md). A wrong rep scores 0.001. |
| Be biased by less than the gate can see at low ESS. | The gate's power grows with ESS: at ESS 400 it catches about 0.27 sd, at high ESS close to 0.1 sd. A sampler with a small bias and a high ESS is checked tightly. A sampler with a low ESS scores low anyway. |
| Get stuck in one mode in every chain. | R-hat above 1.05 fails the rep if chains disagree. If every chain is stuck in the same mode, the bias test catches the missing mass at the gate's power. |
| Identify the model by `problem.dim` and look up tuned step sizes. | Forbidden. It works on any model whose dimension is known, which is all 21. The held-out split helps only if it is private. An official round reads the code. |
| Tune to the initial points or the seed. | A secret base seed in official runs. |
| Use more cores. | One process, `OMP_NUM_THREADS=1`, threads forbidden. The score is per evaluation, so extra cores would only help against the time limit. |
| Do work outside the counted functions, such as finite differences on a surrogate. | Allowed if it only uses points that were evaluated. The time limit bounds it. A surrogate that cuts real evaluations is a better sampler, not an exploit. |
| Overfit to the public models. | Report the held-out score. |
| Get past the import lint. | Not prevented in code. An official round reads the code and runs it in a container without network or file access. |

## Not built yet

- A sandbox. The child process has the rights of the user who runs it.
- Models that are not Stan programs. The seam works through BridgeStan, so every model must be a `.stan` file. A JAX or PyTorch log density would need a second backend behind `Problem`.
- GPU samplers, and samplers that evaluate many points in one call. `log_density_gradient` takes one point. A batched call, counted per point, would be the natural extension.
- A private held-out set (see "The split").
