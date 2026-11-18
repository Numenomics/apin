# The sampler benchmark

This file defines a benchmark for MCMC samplers written in Python. A submission gets a log density and its gradient, and nothing else about the model. It returns draws. A model counts for the submission only when the draws prove that its posterior agrees with a reference. Among submissions that prove the same number of models, the faster one per evaluation ranks higher.

PROTOCOL.md defines the measurement: the grid, ESS, the references and the gate's statistics. This file defines the task, the rules and the score. The benchmark will be hill-climbed, by people and by programs scored on it. A way to score without sampling well is a defect of the same rank as a wrong number, and the rules below are written to close them.

## The rules

**Rule 1. Credit needs proof.** Each repetition (rep) of a model gets one of three verdicts.

- `right`: every checked quantity is shown to be within the tolerance. The upper 95 % confidence bound of its error is below the tolerance.
- `wrong`: the data show an error outside the tolerance, or the run failed hard. A hard failure is a non-finite draw, a missing variable, a variable that never moves, a provenance error or a budget overrun.
- `unproven`: everything else. This includes chains that have not mixed and a quantity whose standard error the run does not define.

Only `right` earns speed credit. A model is right when every rep is right, wrong when any rep is wrong, and unproven otherwise.

**Rule 2. Speed is credited for the accuracy the run achieved, not the accuracy it claims.** The gate trusts the run's Monte Carlo standard errors (MCSEs) only after it has checked them. Over all variables, functions and reps of a model, the realized errors divided by the claimed MCSEs must look like draws from a unit normal. Copies of a chain, dependent chains and an approximation that hides correlation all claim more ESS than they have. This check catches them only when the excess is large enough for the data to show it, and it misses a single perturbed copy or a shuffled chain (see "Ways to game the score"). A cheaper check for chains that are near copies of each other comes first. It is a heuristic and is described as one. Neither check can rule out dependent chains under `sample()`, so results under this interface are provisional, and no official round runs on it.

**Rule 3. The posterior is joint.** The gate checks every variable, the log density of the draws, and a fixed set of projections stored with each reference. For each of these it checks the mean, the mean of the square, and the fractions of draws below the reference's 5 % and 95 % quantiles. A fraction is bounded, so a single extreme draw cannot hide an error in it. These are a stated, finite list of functions. A run can differ from the posterior in any function outside the list and still be right.

**Rule 4. Missing is failing.** The score is computed over a declared grid of models and reps. A missing or failed rep makes its model wrong. A saved rep is reused, and scored, only when its seed, its submission and its budget match the grid.

**Rule 5. A submission sees the interface and nothing else, and everything it does is charged.** The interface is `sample(problem, budget, rng)`. Every call is counted. The clock runs from loading the submission to converting what it returns. The interface cannot show that the returned draws are a Markov chain's or that the chains are independent, so results under it are provisional, and no official round runs on it.

## How the code meets the rules

### Rule 1: the gate

`harness/gate.py` implements the gate. For each checked quantity j of a rep it computes

    e_j = (run mean − reference mean) / reference SD
    s_j = sqrt(MCSE_run² + MCSE_ref²) / reference SD

The quantities are the four functions (mean, square, below the 5 % quantile, below the 95 % quantile) of each standardised variable, each stored projection and lp__.

The verdict is decided in this order.

1. A missing or non-finite variable, a variable that never moves, or a reference constant that the run does not reproduce: `wrong`.
2. A function whose reference value is exact, such as the fraction below the 5 % quantile of a two-valued variable, must match to 1e-6. Its MCSE is not defined, so it is checked directly. A miss is `wrong` when the two halves of the reference agree on the function (see below) and `unproven` when they disagree: a reference value can be exact only because the reference never saw a rare draw, and the halves show it when one half saw such a draw.
3. Rank-normalized split R-hat at or above 1.01, bulk or tail ESS below 400 on any series, or two chains that are near copies: `unproven`. The standard errors are not trusted until the chains have mixed.
4. Some |e_j| − z_w·s_j above the tolerance, on a quantity the two halves of the reference agree on: `wrong`. z_w is the Bonferroni value for all quantities of all reps, at a family-wise rate of 1 % per model. A quantity the halves disagree on can keep the rep from `right` in step 5 but cannot make it `wrong`.
5. Every |e_j| + 1.645·s_j below the tolerance: `right`.
6. Otherwise `unproven`.

The tolerance is **0.1 reference SDs** of each function. The right test is an intersection–union test. It measures a run's distance from the stored reference, not from the truth. A rep whose distance from the reference on one quantity equals the tolerance is called right at most 5 % of the time, however many quantities there are. This 5 % is nominal. It rests on three assumptions: the run's MCSE estimates the run's error, the error is close to normal, and the reference is redrawn, so the rate is over rebuilt references (the simulations draw a new reference for every batch). The first fails for rare large jumps that the run never saw: its draws carry no sign of them, so its MCSE does not include them. For example, take 100 rebuilt references of one standard normal variable (10 chains × 1000 draws) and runs of 4 × 5000 draws from a sampler that jumps, with probability 1e-6 per draw, far enough that its true mean is exactly the tolerance from the reference. The gate called 98 of the 100 runs right, and all 98 had missed the jump. Against one fixed reference, the gate resolves the truth only to the tolerance plus that reference's own error. For example, a 5-chain reference of one standard normal variable whose mean was 0.038 SD off (MCSE 0.0142) called 95 of 100 runs right whose mean was 0.099 SD from the truth, because they were only 0.061 SD from the reference.

A reference's MCSE is an estimate of its error, not the error it has. The table gives each shipped reference's largest MCSE over its checked quantities, in reference SDs (`mcse / sd` from `Reference.targets()`, where sd > 0).

| reference | source | largest MCSE [SD] | on |
|---|---|---|---|
| accel_gp | posteriordb | 0.0153 | proj.2 square |
| arma11 | posteriordb | 0.0149 | proj.1 mean |
| blr | posteriordb | 0.0150 | proj.3 below_q05 |
| bym2_offset_only | local | 0.0119 | proj.1 mean |
| diamonds | posteriordb | 0.0151 | proj.14 square |
| dogs_hierarchical | local | 0.0103 | proj.2 below_q95 |
| eight_schools_centered | posteriordb | 0.0150 | proj.4 below_q95 |
| eight_schools_noncentered | posteriordb | 0.0150 | proj.4 below_q95 |
| garch11 | posteriordb | 0.0152 | proj.2 mean |
| gp_regr | posteriordb | 0.0151 | proj.1 below_q05 |
| hier_2pl | local | 0.0117 | proj.1 mean |
| kidscore_momiq | posteriordb | 0.0153 | proj.1 below_q05 |
| logmesquite_logvash | posteriordb | 0.0151 | proj.6 below_q05 |
| lotka_volterra | posteriordb | 0.0146 | proj.8 mean |
| low_dim_gauss_mix | posteriordb | 0.0146 | proj.4 mean |
| lsat_model | local | 0.0103 | proj.1 below_q95 |
| pilots | local | 0.0107 | proj.5 mean |
| radon_partially_pooled_noncentered | local | 0.0107 | proj.2 mean |
| radon_variable_intercept_slope_noncentered | local | 0.0106 | proj.5 below_q95 |
| wells_dist100_model | local | 0.0103 | proj.1 mean |

At the largest of these, 0.015 SD, a reference error of two MCSEs moves the tolerance measured against the truth to anywhere from about 0.07 to 0.13 SD on that quantity.

The reference's MCSE comes from its own draws. A reference that missed a rare excursion claims an MCSE far below its true error, and then a correct run that does see the excursion is called wrong. The reference therefore keeps each half's estimate and MCSE of every function (`models/reference/halves/<model>.json`). For the variables and lp__ the halves are the two halves of the chains. For the projections they are the half that chose the directions and the half that summarises them. The halves disagree on a quantity when their estimates differ by more than 5 times the smaller of their two MCSEs (`gate.SPLIT_Z`). A half whose MCSE is 0 claims an exact estimate, so against it any difference above 1e-6 is a disagreement. The smaller MCSE is used and not the combined one: a half that saw k excursions which the other half missed differs from it by only about √k combined MCSEs, because its own MCSE grows with what it saw. On references of correct draws of the five-variable target, 10 chains × 1000 draws, the halves disagree by chance on 0.10 % of quantities with independent draws, 0.19 % with AR(1) draws at ρ = 0.5 and 0.61 % at ρ = 0.9. These rates and the trials below are in `results/tables/gate_rates.md`.

What the rule covers and what it does not:

- It covers a reference whose summarising half missed rare excursions that the other half saw. In the case found in review (a projection that shows an excursion of p = 0.0005 per draw, 10-chain references) a correct run was called wrong in 5 of 120 trials. With the rule it is wrong in 0 of 120 (106 right, 14 unproven).
- It does not cover a reference whose two halves both missed the excursions. On one variable with excursions of p = 0.00035 and 5-chain references, a correct run is still called wrong in 3 of 150 trials, with or without the rule, and all three references saw no excursion in either half.
- For the projections the halves are not alike even when the reference is right. The half that chose the directions sees its own sample eigenvalues as the variances along them, and in high dimension those are far from the truth. On the larger references most of the square and tail quantities of the projections are disputed for this reason (see rule 3). They can still keep a run from `right`. A sampler that is wrong only along those directions is unproven, not wrong. Both earn nothing.

The evidence for these rates is a simulation, `harness/gate_rates.py`, with its output in `results/tables/gate_rates.md`. It uses a correlated normal in five variables and stationary AR(1) chains at ρ = 0, 0.5, 0.9, 0.99 and 0.999. The reference is redrawn for every batch, so the rates include the reference's own error.

The table below is a summary of `results/tables/gate_rates.md`: 100 simulated runs of 4 chains per row, one rep per verdict. "ESS" is the ESS of each variable, 4n(1 − ρ)/(1 + ρ).

| ρ | draws per chain | ESS | correct: right / unproven / wrong | error = tol: right | error = 2·tol: wrong |
|---|---|---|---|---|---|
| 0 | 1000 | 4000 | 0.90 / 0.10 / 0 | 0.04 | 0.94 |
| 0 | 5000 | 20 000 | 1.00 / 0 / 0 | 0 | 1.00 |
| 0.5 | 1000 | 1333 | 0.38 / 0.62 / 0 | 0.02 | 0.49 |
| 0.9 | 1000 | 211 | 0 / 1.00 / 0 | 0 | 0 |
| 0.9 | 20 000 | 4211 | 1.00 / 0 / 0 | 0.01 | 0.94 |
| 0.9 | 50 000 | 10 526 | 1.00 / 0 / 0 | 0 | 1.00 |
| 0.99 | 1000 | 20 | 0 / 1.00 / 0 | 0 | 0 |
| 0.99 | 50 000 | 1005 | 0.20 / 0.80 / 0 | 0.01 | 0.33 |
| 0.999 | 1000 | 2 | 0 / 1.00 / 0 | 0 | 0 |
| 0.999 | 50 000 | 100 | 0 / 1.00 / 0 | 0 | 0 |

What it shows:

- **False wrong.** No correct run was called wrong, at any ρ, in 1100 runs. The design rate is at most 1 % per model. It was measured on Gaussian AR(1) chains, against references whose MCSE is accurate, and it holds only for chains that have mixed. Chains that have not mixed, as at ρ = 0.99 and 0.999 with short runs, are unproven, not wrong. A reference whose MCSE is too small breaks the rate. The halves rule above removes the case where one half saw what the other missed, and not the case where the whole reference missed it.
- **False right.** A run whose error is exactly the tolerance was called right in 0 to 4 % of runs. The nominal rate is at most 5 % per rep, for any degree of mixing, because a run that has not mixed is never right. It does not hold for a sampler with rare jumps its runs do not show (see the tolerance above).
- **Power.** A bias of twice the tolerance is called wrong in 94 % of runs at an ESS of 4000, and in all runs at 10 000 and above. At low ESS the same bias is unproven. It earns nothing either way.
- **What a correct sampler needs.** A rep is proven right in 90 % of runs at an ESS of 4000 per variable, and in all runs at 20 000. A model needs all 5 reps right, so 90 % per rep gives about 59 % per model. The budgets below are set from NUTS's ESS on each model, not from this table.

The calibration test (rule 2), 100 groups of 5 reps:

| ρ | draws per chain | sampler | model right | median ratio of realized to claimed variance |
|---|---|---|---|---|
| 0 | 5000 | correct | 1.00 | 0.96 |
| 0 | 5000 | two pairs of copies, 1e-3 SD apart | 0.25 | 1.96 |
| 0.5 | 15 000 | correct | 1.00 | 1.02 |
| 0.5 | 15 000 | two pairs of copies, 1e-3 SD apart | 0.21 | 2.01 |

The calibration test catches chains that claim twice their real ESS in 75 to 79 % of models on this five-variable target. That is not enough to stop a submission that copies its chains on a small model. Its power grows with the number of quantities and reps, so it is higher on the larger models.

The threshold is a trade. On the same target, in 200 groups of 5 reps:

| threshold p | correct model refused | copies caught |
|---|---|---|
| 0.001 (in use) | 0.5 % | 70 % |
| 0.01 | 2 % | 88 % |
| 0.05 | 4.5 % | 96 % |

The threshold of 0.001 is nominal: the χ² approximation of the two sums does not hold it. Under an idealised Gaussian null, in which every rep's errors are exactly normal with the reference's correlations and exactly the claimed MCSE (`gate_rates.calibration_null`, rates in `results/tables/gate_rates.md`), correct models whose reps were all right were refused in 477 of 99 951 groups (0.48 %) with the correlations of eight_schools_noncentered (84 quantities), and in 31 of 20 000 (0.16 %) with those of hier_2pl (2716 quantities). The review found 913 of 199 903 (0.46 %) on eight_schools_noncentered. The null for hier_2pl scales its correlations by 0.59 to match the reference's tr(C²), because the raw correlation of 2000 draws overstates it, so its rate is rougher. The threshold is not retuned.

The rank counts models right before it looks at speed. Copying the chains gains at most a factor of 2 in the tie-break on one model. The 70 % above was measured on one construction, two pairs of copies 1e-3 SD apart among four chains, and holds only for it. One perturbed copy among four chains, 1e-3 SD away, earned credit in 24 of 30 groups in the review and 27 of 30 in a rerun, so a submission can gain in the tie-break by copying. A higher threshold would cost honest submissions a model in 2 to 5 % of models. *Alternative:* p = 0.01, if the review of submissions finds that the tie-break is what people compete on.

The bound on the error shrinks as 1/√ESS. Stock NUTS at 4 × 1000 draws reaches an ESS of 700 to 1300 on its worst series on the core models, and its bounds were 0.09 to 0.15: most of its reps came out unproven. At 4 × 5000 its ESS is 4000 to 11 000 and its bounds are 0.05 to 0.08. The budgets are therefore raised per model (see "The budget"), and the tolerance stays at 0.1. *Alternative:* a tolerance of 0.2 would prove most models at 4 × 1000. A bias of 0.2 SD in a mean is large enough to change a conclusion, so this file does not recommend it.

### Rule 2: claimed against achieved accuracy

`gate.judge` runs the calibration test on a model whose reps are all right. It forms two sums of squared standardised errors over all quantities:

- against the reference: Σ e² / (MCSE_run² + MCSE_ref²) over reps and quantities. The reference error is shared by all reps, and the null distribution accounts for that.
- between the reps: the weighted spread of each quantity's errors around their mean over reps. The reference error cancels out of this sum.

Each sum is compared with a scaled χ² of the same mean and variance. The variance comes from the correlation of the quantities in the reference. If the smaller p-value, doubled, is below 0.001, the model is unproven, and the reason gives the ratio of the realized to the claimed variance. Two copies of every chain give a ratio near 2.

The near-copy check in `gate._copies` is the cheap heuristic. Two chains are flagged when more than 1 % of their draws lie within 1e-6 reference SDs of a draw in the other chain. It catches lazy copying and nothing more. A submission can perturb its copies past any fixed threshold. The calibration test is the check that matters.

Cross-chain adaptation stays legal. Chains may share a mass matrix or a step size, and ensemble methods may move chains together. The rule asks only that the MCSEs, which assume independent chains, are right. A method whose chains are dependent enough to break that is not credited for the ESS it does not have.

The credited ESS is the smallest bulk or tail ESS over every series the gate checks. It is not over the variables alone, so an approximation that samples each variable well but misses their correlation gets no more credit than its worst projection earns.

### Rule 3: what the reference stores

Each reference (`models/reference/<model>.json`, built by `harness/reference.py`) holds no draws. It holds summaries of a set of series:

- every variable, standardised;
- projections of the standardised variables on eigenvectors of the reference correlation matrix: all of them up to 20 variables, else the 10 of largest and the 10 of smallest eigenvalue (4 and 4 above 500 variables). The smallest directions are where an approximation that ignores correlation goes wrong. They are chosen on one half of the reference chains and summarised on the other, so the choice does not bias them. Local references run 20 chains, so each half has 10. With 10 chains, the summary half of radon_vis and bym2_offset_only carried errors of 3 to 5 of its own MCSEs on one projection, and correct NUTS runs came out unproven against it. The posteriordb references cannot be grown and keep halves of 5 chains (see "Not built yet");
- lp__.

For each series it stores the expected value of each function, its SD and its MCSE, and the 5 % and 95 % quantiles. Beside it, `models/reference/halves/<model>.json` holds each half's estimate and MCSE of every function, for the halves rule of rule 1.

The halves of the shipped references disagree on 155 of 25 860 quantities. On the variables and lp__ there are 18, close to the chance rate: bym2_offset_only 9, lsat_model 3, hier_2pl 2, radon_partially_pooled_noncentered 2, accel_gp 1 (zgp_1.19 mean) and logmesquite_logvash 1 (beta.6 below_q95). On the projections there are 137, all square and tail quantities and none of the means. Of each model's square and tail quantities of the projections they are: radon_partially_pooled_noncentered 50 of 60, radon_variable_intercept_slope_noncentered 31 of 60, lsat_model 23 of 24, bym2_offset_only 21 of 24, hier_2pl 9 of 24 and accel_gp 3 of 60. accel_gp's are proj.6 square, proj.7 square and proj.7 below_q05; proj.7 square is the 0.10 SD difference between halves that the review found. Most of the projection disputes in the large models come from the choice of directions (rule 1, "What the rule covers"), not from a reference error. An unbiased second estimate along those directions needs draws that neither chose them nor summarise them: a third set of chains, which means sampling the local references again, and which posteriordb's 10 chains do not allow. Variables that are constant in the reference, such as the fixed entries of a Cholesky factor, are stored apart, and a run must reproduce them.

The checked functions are a finite list, and a run can be wrong outside it and still be right. An example from the review: the target is a normal in six variables, three independent blocks of two correlated variables. The run keeps only the half of it where the product of three latent coordinates, one from each block, is positive. Every block, and every pair of blocks, has the same distribution as under the target, and so has lp__. The checked functions of the projections that mix all three blocks change by too little to show. All 5 reps were right and the calibration test passed (p = 0.64), with half of the posterior missing.

lp__ is computed by the harness from the constrained draws with BridgeStan. A column the sampler wrote is ignored. A draw outside the support, or one the model rejects, gives a NaN log density, and the rep is wrong.

### Rule 4: the declared grid

`score.declared` builds the grid: every selected model that has a reference, at the budget in `models/manifest.json`, with the declared number of reps and base seed. `score.score` walks this grid, not the run directories. A rep that is not there is wrong ("repN missing"). A rep whose record names another model, rep, seed, base seed or budget than its grid cell is wrong. The submission digest is checked less: the scorer refuses a label whose reps record more than one sampler identity (the identity includes the submission's sha256), so a label scores one submission, but by itself it does not know which one. `apin score --submission LABEL=PATH` gives it the file; then every rep of that label must record that file's sha256, or it is wrong. So is a rep whose chain files are fewer than the budget's chains, hold fewer draws than its draws, or are cut short: a last row without a line end or a row with the wrong number of fields. The grid also carries the evaluation and time limits of each model (see "The budget"). A rep that used more evaluations than the limit, or took longer than the time limit, is wrong. A rep the harness metered (the logdensity adapter, or any rep that records a harness count or limit) must record both limits, equal to the grid's, and a total equal to its density plus gradient calls; otherwise it is wrong. CmdStan is not metered by the harness, so its reps record no limits and are checked on their usage only. The verdict cache in `gate.json` holds only for the chain files it was computed from: its key includes the sha256 of every chain file, taken of the bytes the gate parsed (each file is read once), and the checks for missing and short files run on every score. A verdict cache that does not load is recomputed.

`grid.run_grid` resumes a rep only when the saved seed, base seed, submission identity and budget all match. Otherwise it deletes the rep and runs it again.

A label whose reps come from more than one submission is refused.

### Rule 5: the seam and its limits

`harness/logdensity.py` hands the submission two plain functions and a copy of its budget. The limits the harness enforces and the shape it expects are copied before the submission runs.

- Every call to `log_density` or `log_density_gradient` is charged before it is evaluated, at a rejected point too.
- A call past the limit raises BudgetExceeded, and the run fails even if the submission catches it.
- The clock starts before the submission file is loaded and stops after the return value is converted to an array. An overrun fails the run at the end, even if no call came after it.
- Every returned draw must be a point the submission evaluated with a finite log density (provenance). Points must keep the order in which they were first evaluated, and no point may appear in two chains. That is all provenance verifies. It does not verify that the order is a Markov chain's: a submission can evaluate its draws again in any order, at points one float step away, and return them in that order. Nor does it verify that the chains are independent.

This protects the counts from a submission that makes a mistake. It does not protect them from one that sets out to cheat. In-process Python cannot stop determined introspection. A submission can reach the counter through closure cells, the garbage collector or the stack. An official round therefore reads every submission before it trusts the result. Process isolation is listed under "Not built yet".

## The task

Write one Python file that defines

```python
def sample(problem: Problem, budget: Budget, rng: numpy.random.Generator) -> numpy.ndarray:
    ...  # unconstrained draws of shape (budget.chains, budget.draws, problem.dim)
```

`Problem` and `Budget` are in `harness/logdensity.py`. PROTOCOL.md, "The log-density seam", gives the full contract. In short:

- `problem.log_density(theta)` and `problem.log_density_gradient(theta)` evaluate the model on the unconstrained space, with the Jacobian and all constants. A point the model rejects gives −inf.
- `problem.initial_points` has one starting point per chain, uniform on (−2, 2).
- Every returned draw must be a point the submission evaluated with a finite log density.

Run a submission on the public models with

```bash
uv run apin run --sampler logdensity --label mine --opt sampler=path/to/mine.py --models public --reps 5
uv run apin run --sampler cmdstan --label cmdstan --models public --reps 5
uv run apin score --baseline cmdstan --models public --reps 5
```

`examples/samplers/hmc.py` is a complete submission of about 100 lines. `examples/samplers/ula_wrong.py` is wrong on purpose and shows what the gate does with a fast, biased sampler.

## What a submission may do

A submission may import numpy, scipy and the standard modules math, typing, dataclasses, functools, itertools, collections and `__future__`. It may not name `open`, `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `input`, `globals` or `vars`. `apin run` checks both before it starts.

A submission must not:

- read or write files, use the network, start processes or threads, or keep any state between runs;
- identify the model, from its dimension or by probing the density, in order to look up settings tuned for it;
- reach the model, the counter or the harness by any route other than the two functions in `problem`.

The import check is a lint. `getattr` on the builtins gets past it. It is there to catch mistakes, and the reading of the code is the real check.

## The budget

Each rep gives the submission the budget of its model in `models/manifest.json`:

| | value | why |
|---|---|---|
| chains | 4 | R-hat needs several chains |
| warmup | 1000 | as CmdStan; it sets the evaluation limit |
| draws | 5000 per chain; 10 000 for lsat_model, 20 000 for accel_gp | enough for stock NUTS to be proven right at a tolerance of 0.1 where a budget can do it (see below) |
| evaluations | 4 × (warmup + draws) × 1023 | a chosen cap: 1023 is the number of leapfrog steps in a full tree of depth 10 (see below) |
| time | 1800 s on one core | `OMP_NUM_THREADS=1` and the BLAS equivalents; the process is killed 300 s later |

A density call and a gradient call count the same. Both counts are recorded (`density_evals`, `gradient_evals`).

The evaluation cap is a choice, not the most NUTS can spend. A NUTS transition that reaches depth 10 makes 1024 gradients (its 1023 leapfrog steps and one at its start), before any step-size search, so a NUTS chain that hits the depth limit on every iteration would exceed the cap. The evaluation limit stops a submission that thins by a factor of thousands. The time limit stops one that spends its time in numpy between evaluations.

### Stock NUTS at the bench budget

CmdStan 2.39 NUTS with default settings, 5 reps per model, at the budgets above:

| model | draws per chain | verdict | reps right | largest bound | smallest ESS | note |
|---|---|---|---|---|---|---|
| eight_schools_noncentered | 5000 | right | 5 of 5 | 0.083 | 7621 |  |
| blr | 5000 | right | 5 of 5 | 0.079 | 3696 |  |
| kidscore_momiq | 5000 | right | 5 of 5 | 0.072 | 4978 |  |
| lsat_model | 10000 | right | 5 of 5 | 0.065 | 7269 |  |
| logmesquite_logvash | 5000 | right | 5 of 5 | 0.080 | 6275 |  |
| wells_dist100_model | 5000 | right | 5 of 5 | 0.062 | 4987 |  |
| diamonds | 5000 | right | 5 of 5 | 0.083 | 5654 |  |
| radon_partially_pooled_noncentered | 5000 | right | 5 of 5 | 0.080 | 1232 |  |
| radon_variable_intercept_slope_noncentered | 5000 | right | 5 of 5 | 0.070 | 3282 |  |
| dogs_hierarchical | 5000 | right | 5 of 5 | 0.055 | 6830 |  |
| pilots | 5000 | unproven | 0 of 5 | – | 32 | unproven: R-hat 1.092 on sigma_a |
| hier_2pl | 5000 | right | 5 of 5 | 0.076 | 3089 |  |
| gp_regr | 5000 | right | 5 of 5 | 0.063 | 9797 |  |
| accel_gp | 20000 | wrong | 2 of 5 | – | – | wrong: rep0 took 2945 s, the limit is 1800 s |
| bym2_offset_only | 5000 | right | 5 of 5 | 0.077 | 1094 |  |
| eight_schools_centered | 5000 | wrong | 0 of 5 | – | 43 | wrong: lp__ mean e=-0.274 bound=0.16 |
| garch11 | 5000 | right | 5 of 5 | 0.081 | 7072 |  |
| lotka_volterra | 5000 | right | 5 of 5 | 0.065 | 4511 |  |
| low_dim_gauss_mix | 5000 | right | 5 of 5 | 0.064 | 9337 |  |
| arma11 | 5000 | right | 5 of 5 | 0.066 | 9893 |  |

The runs behind this table were made on a machine under load, so their times are not reported here. The verdict depends on timing in one place: a rep over the time limit is wrong. accel_gp's rep0 took 2945 s against the limit of 1800 s, so accel_gp is wrong, and the bound and ESS columns are blank because rep0 was not checked. With the time limit lifted, its verdict is unproven, with 3 reps of 5 right, a largest bound of 0.151 and a smallest ESS of 15 333. Its other reps took 340 to 1204 s.

- 17 of 20 models are right. "Largest bound" is the largest upper bound on |e| over the reps' quantities; it must be below 0.1. "Smallest ESS" is the smallest credited ESS over the reps.
- eight_schools_centered is wrong. In one rep the mean of lp__ is 0.27 SD off, with a lower bound of 0.16 on the error. This is the known bias of NUTS on the centred funnel, and more draws only make it clearer.
- pilots is unproven. sigma_a has R-hat 1.092 and an ESS of 32 in 4 × 5000 draws: the chains do not cross the funnel. ESS grows about linearly with the draws, so 400 would take some 60 000 draws per chain. The budget stays at 5000.
- Without the time limit, accel_gp is unproven at 4 × 20 000 draws. Three reps are right. In the other two, the run's own MCSE of the square of proj.17 is 0.05 SD, an ESS of about 390 for that function, while the bulk and tail ESS of the series is 21 000. That points to rare long excursions along this direction. At 4 × 5000, one rep of five failed the same way: one chain went to 25 SD on proj.16. Proving it would take about 4 × 80 000 draws.
- radon_variable_intercept_slope_noncentered and bym2_offset_only were unproven against references built on 10 chains. The summary half of those references was off by 3 to 5 of its own MCSEs on one projection (see rule 3). Against the 20-chain references both are right at 4 × 5000.
- lsat_model runs 4 × 10 000 draws. The budget was raised when its reference still had 10 chains. 4 × 5000 has not been retried against the 20-chain reference.
- At 4 × 1000 draws, the old default, most reps of most models were unproven: their bounds were 0.09 to 0.15.

## The score

1. Each rep gets a verdict (rule 1). A failed or missing rep is wrong.
2. Each model gets a verdict from its reps: right if all are right and the calibration test passes, wrong if any is wrong, unproven otherwise.
3. The rank is by the number of models right, out of every model in the grid. That includes models where NUTS is wrong or unproven. A sampler that proves a model NUTS cannot prove gains a full point.
4. Ties are broken by speed: the geometric mean, over the models where both the submission and the baseline are right, of the ratio of the submission's ESS per evaluation to the baseline's. The ratio for a model is the ratio of the means over reps.
5. A 95 % bootstrap interval comes with the speed. It resamples the reps of the submission and of the baseline, 2000 times. The baseline's own row is 1 with no interval. The interval is conditional on the set of models where both labels are right, so two labels tied on models right can have their speeds taken over different sets of models.

There is no penalty factor and no median over reps. A model omitted counts as wrong, so it never scores better than an attempt.

The speed is per evaluation first. ESS per evaluation does not depend on the machine, on load, or on how well the submission uses numpy, so a submission in pure Python competes with C++ NUTS. ESS per second is printed next to it, from the sum of the chains' own times. It measures what a user waits for, but mostly how well the overhead was vectorised.

CmdStan's count is `n_leapfrog__`, plus one gradient at the start of every transition, plus the gradients of its step-size searches, which the adapter rebuilds from the CSV (PROTOCOL.md, "The log-density seam"). CmdStan's initialisation attempts are not counted: 2 to 200 evaluations per chain. At the bench budget a chain spends at least 2 × (1000 + 5000) = 12 000 on the rest, so the share left out is under 1.7 %. A run that cannot count all its evaluations has no ESS per evaluation. This is the case for CmdStan with `thin` above 1 or a step-size trace that does not rebuild (`stepsize_jitter` above 0), for the Pathfinder adapter (CmdStan does not report Pathfinder's gradients) and for nutpie (no warmup count). The adapter says why in `Timing.uncounted`. Such a label has no per-evaluation speed, and the table says on which models.

`apin score` writes `per_run.csv`, `per_model.csv`, `ranking.csv` and `table.md`. The table prints the rank and, next to it, the per-model table with each model's verdict and speed.

**Why min(bulk, tail).** Bulk ESS alone rewards a sampler that mixes the centre and never visits the tails.

**Models without a reference.** The gate cannot check a model that has no reference, so such a model is not in the grid. Today this is kronecker_gp. It is not scored and has no rows.

## The split

The 21 core models are split into 10 public and 11 held-out models. Each family is in both halves.

| family | public | held-out |
|---|---|---|
| easy/small | eight_schools_noncentered, blr | kidscore_momiq, lsat_model |
| GLM | diamonds | logmesquite_logvash, wells_dist100_model |
| hierarchical | radon_partially_pooled_noncentered, pilots | radon_variable_intercept_slope_noncentered, dogs_hierarchical, hier_2pl |
| stiff/funnel | eight_schools_centered, garch11, low_dim_gauss_mix | lotka_volterra, arma11 |
| GP/spatial | gp_regr, bym2_offset_only | accel_gp, kronecker_gp (no reference, not scored) |

The reasons:

- Each public model shows a hard feature of its family. diamonds has strong correlations, pilots and eight_schools_centered have funnels, low_dim_gauss_mix is a two-component mixture, bym2_offset_only has a sparse spatial prior. A submission tuned on the public set has seen each difficulty.
- The two radon models share data but differ in structure. One goes in each half, so a submission cannot learn the radon data from the public half and score it twice.
- The held-out half has the one ODE model (lotka_volterra) and the one item-response model (hier_2pl). These are the models least like anything in the public half, so they test whether a sampler generalises.
- `models/manifest.json` records the split (`"bench"`). `--models public` and `--models held-out` select it.

The held-out models are not secret. The repository is public and anyone can run them. The split states the intent: tune on the public half and report on the held-out half. A secret held-out set is not built yet.

**Recommendation:** report the held-out rank as the result and the public rank next to it. A large gap between the two shows overfitting.

## Seeds and reps

Rep r uses seed BASE + 1000·r for the initial points and for `rng`. During development BASE is 20260819, the default. An official run uses a BASE that is kept secret until the run is done, passed with `--base-seed` to both `apin run` and `apin score`.

An official result uses 5 reps. Every rep must be right for the model to be right, so a sampler that is right only some of the time is not credited.

## A demo run

One run on 5 public and held-out models, 5 reps each, at the bench budget (4 × 5000 draws), on a loaded 12-core machine: CmdStan 2.39 NUTS, `examples/samplers/hmc.py` and `examples/samplers/ula_wrong.py`. The machine was running other jobs, so the speeds per second were measured under load and are not a record. The speeds per evaluation do not depend on load.

| rank | label | models right | speed per evaluation [95 %] | speed per second [95 %] |
|---|---|---|---|---|
| 1 | CmdStan NUTS | 4 of 5 | 1 | 1 |
| 2 | HMC example | 3 of 5 | 0.538 [0.52, 0.559] | 0.155 [0.147, 0.163] |
| 3 | ULA (wrong on purpose) | 0 of 5 | | |

| model | CmdStan NUTS | HMC example | ULA |
|---|---|---|---|
| blr | right: 0.0107/eval, 1230/s | right: 0.00319/eval, 54.7/s | unproven: R-hat 1.361 on sigma |
| eight_schools_centered | wrong: lp__ mean e = −0.274, bound 0.16 | unproven: R-hat 1.013 on lp__ | unproven: R-hat 1.035 on proj.1 |
| eight_schools_noncentered | right: 0.0342/eval, 3390/s | unproven: proj.2 square e = +0.075, bound 0.131 | wrong: non-finite lp__ |
| garch11 | right: 0.0253/eval, 382/s | right: 0.0229/eval, 178/s | wrong: lp__ square e = +1.13, bound 0.837 |
| kidscore_momiq | right: 0.0085/eval, 361/s | right: 0.00491/eval, 64.4/s | unproven: R-hat 2.612 on beta.1 |

Cells are the mean over reps of the credited ESS per evaluation and per second of chain time, or the model's verdict and the reason from the first rep that decided it.

What the run shows:

- NUTS is wrong on eight_schools_centered. One of its five reps passed R-hat and ESS and still showed the funnel's bias in lp__, 0.27 SD with a lower bound of 0.16. The others had not mixed. This is the known failure of NUTS on the centred funnel, and the gate sees it.
- ULA earns nothing. Where its chains mix, the gate calls it wrong. Where they do not, it is unproven.
- The HMC example ranks below NUTS. It proves fewer models and is about half as fast per evaluation where both are right. With the old score, which credited the bulk ESS of the variables alone at 4 × 1000 draws, it scored 1.32 times NUTS on four of these models, with NUTS's cost counted as `n_leapfrog__` alone. A likely reason for the change is that the credited ESS now includes the projections on the correlated directions and lp__, and the example uses a diagonal metric, which mixes slowly along them. This run does not test that.

## Ways to game the score

"Prevented" says whether the harness stops the attack. "Not with this interface" means `sample()` gives the harness no way to stop it; results under it are provisional (rules 2 and 5).

| attack | defence | prevented |
|---|---|---|
| Return draws from a stored reference, or from a Laplace or variational fit, without paying for them. | Every draw must be a point the submission evaluated with a finite density (provenance). File access is forbidden; the references are not in the problem. | yes |
| Return points the model rejected. | A rejected point has no finite density, so it fails provenance. Rejected calls are still charged. | yes |
| Copy one chain four times. | No point may appear in two chains. | yes |
| Sample each variable well and ignore the correlations. | The projections and lp__ are checked (rule 3). The credited ESS is the worst over them. | yes |
| Hide a bias behind one extreme draw that inflates the MCSE. | The quantile exceedance fractions are bounded and still show the bias (rule 3). | yes |
| Spend evaluations in warmup and report only the sampling phase. | Every call counts, warmup included. | yes |
| Thin a long chain by a large factor. | Allowed, and counted: thinning costs evaluations. The evaluation limit bounds it. | yes |
| Be fast and biased. | The gate calls the bias wrong once the data show it. A small bias the data cannot show leaves the model unproven, which earns nothing. | yes |
| Get stuck so the chains do not mix. | R-hat or ESS makes the rep unproven. It earns nothing. | yes |
| Get stuck in the same mode in every chain. | Caught when the missing mass moves a checked function: the gate calls it wrong once the data show the shift. A mode whose absence leaves every checked function unchanged is not caught (rule 3). | partly |
| Skip a hard model. | A missing model is wrong (rule 4). | yes |
| Keep a lucky rep from an old seed or an old version. | A rep is reused only when its seed, submission and budget match. | yes |
| Change the budget or the shape the harness checks. | The harness copies both before the submission runs. | yes |
| Keep computing after the last call. | The clock runs to the end of the conversion. | yes |
| Identify the model by `problem.dim` and look up tuned step sizes. | Forbidden. Not prevented in code; an official round reads the code. | no: an official round reads the code |
| Tune to the initial points or the seed. | A secret base seed in official runs. | only in an official round |
| Use more cores. | One process, `OMP_NUM_THREADS=1`, threads forbidden. The score is per evaluation, so extra cores help only against the time limit. | partly: threads are forbidden by rule, and an official round reads the code |
| Reach the counter or the model through introspection. | Not prevented in code (rule 5). An official round reads the code. | no: an official round reads the code |
| Overfit to the public models. | Report the held-out rank. | partly: the held-out models are public (see "Not built yet") |
| Shuffle the draws within a chain, which hides autocorrelation and inflates ESS. | Draws must keep the order in which they were first evaluated, but a submission can evaluate its draws again in shuffled order, one float step away (`np.nextafter`), and return them. The calibration test refused 3 of 4 such groups of an AR(1) chain at ρ = 0.8. The fourth was right, at p = 0.0017, and its credited ESS per call rose from 0.111 to 0.473. | no: not with this interface |
| Copy a chain with a small perturbation. | The near-copy check catches copies within 1e-6 SD. A copy of one chain of four, 1e-3 SD away, earned credit in 24 of 30 groups in the review and 27 of 30 in a rerun: the calibration test does not see one copy. | no: not with this interface |
| Run chains that are not independent. | The calibration test catches dependence only when the realized errors are much larger than the claimed MCSEs (rule 2). | no: not with this interface |

## Not built yet

- Process isolation. The submission runs in a child process with the rights of the user who runs it. It shares that process with the counter, so it can reach the counter by introspection. A real sandbox would run the submission in a separate process, without file or network access, and serve the density over a pipe.
- A secret held-out set. The held-out models are in the public repository.
- Models that are not Stan programs. The seam works through BridgeStan, so every model must be a `.stan` file.
- A replicate check on the references. The halves rule compares two halves of one build, and nothing compares two independent builds of a reference. An error that both halves share is found only when a correct sampler fails against it, as on radon_vis and bym2_offset_only.
- Larger posteriordb references. The posteriordb draws are 10 chains of 1000, so their projections are summarised on 5 chains. An error of a few of their MCSEs in one projection is about 0.05 SD on these models. That is enough to make a correct run unproven, and too small to make it wrong at a tolerance of 0.1.
- Tail ESS at a tie. When many draws share the maximum or minimum, as in a normal clipped at a bound, the tail ESS is undefined (NaN), so the rep is unproven however well it mixed.
- GPU samplers, and samplers that evaluate many points in one call. `log_density_gradient` takes one point.
