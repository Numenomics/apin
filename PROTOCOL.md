# Measurement protocol

This file says how apin measures a sampler or a patch, what a runner must write, and the mistakes that cost the most time in the past. `apin run` and `apin score` implement the protocol. Where they leave a choice to you, this file says what to choose.

## The grid

The core set is the 21 posteriordb models in `models/manifest.json`. Each has a family label (easy/small 4, GLM 3, hierarchical 5, stiff/funnel 5, GP/spatial 4). The set was fixed in the first phase of the old campaign (`results/records/PHASE0_SUMMARY.md`) and has not changed since. `apin list` prints them with their family. Pass a family name to `--models` to run one family.

Each cell is one sampler, one model and one rep:

- 4 chains, 1000 warmup iterations, 1000 draws per chain. CmdStan saves its warmup draws so that their gradient count is known. `read_csv` keeps them apart, and they are never scored.
- 3 reps. Rep r uses seed 20260819 + 1000·r. Chains inside a rep get the chain id or seed offset that their sampler uses (CmdStan `id=c+1`; walnutpie `seed+c`).
- At most 4 cores in use for the whole grid. The old campaign ran on a 12-core Ryzen 9 5900X and kept to 4 cores so that memory bandwidth and turbo states were comparable between arms. A rep's 4 chains run in parallel. Run one grid at a time.

A rep that finishes gets a `DONE` marker; a rep that raises gets `FAILED` with the traceback. A rerun skips reps with `DONE`, so an interrupted grid resumes where it stopped.

## The score

`apin score` computes, for every rep, the rank-normalized bulk and tail ESS and split R-hat of every variable (Vehtari et al. 2021; `harness/ess.py` reproduces the R package posterior 1.7.0 to 10 significant digits). Columns whose names end in `__` are sampler diagnostics and are not scored. Chains of unequal length are cut to the shortest before scoring.

Before any cost, every rep passes through the correctness gate (next section). A rep that fails it is "wrong" and gets no speed numbers. A model is wrong for a label when more than half of its reps are wrong or failed. The score table then shows "wrong", the worst variable and moment, and that moment's b² in place of the speed.

The quality number of a rep is `ess_bulk_min`, the smallest bulk ESS over all variables. It is a pessimistic number with a large variance between reps, so the per-model value is the median over the reps that passed and the summary over models is a geometric mean of those medians. Apart from the gate, the summary counts the models whose median R-hat is above 1.01.

Cost is reported four ways:

| column | meaning |
|---|---|
| `ess_per_s_wall` | `ess_bulk_min` divided by the wall time of the rep, chains in parallel. This is what a user waits for. |
| `ess_per_s_chain` | `ess_bulk_min` divided by the sum of each chain's own sampler time (warmup plus sampling). This removes the effect of parallel dispatch and machine load. W-109 and W-133 used this convention ("stanza sum"). |
| `ess_per_grad` | `ess_bulk_min` divided by the gradient evaluations in the sampling phase. It does not depend on the machine. |
| `ess_per_eval` | min(`ess_bulk_min`, `ess_tail_min`) divided by every density and gradient evaluation, warmup included. It does not depend on the machine and does not let a sampler hide work in warmup. BENCH.md scores this one. It needs the warmup count: CmdStan's `n_leapfrog__` over the saved warmup rows, or the logdensity adapter's own count. |

Never compare numbers that use different conventions. A report states which one it uses. With `--baseline LABEL`, `apin score` writes per-model ratios against that label and prints their geometric mean.

`apin score` also writes `table.md`: one row per model, one column per label, with ESS per evaluation and ESS per second of chain time, or the gate's reason for "wrong". With `--baseline` it writes `bench.csv`, the benchmark score of BENCH.md.

## The correctness gate

ESS measures how fast a sampler forgets where it was. It says nothing about whether it converges to the right distribution. A sampler that is fast and wrong would win on ESS/s. So `apin score` compares each rep with a reference posterior and fails it when the two disagree. The code is `harness/gate.py`; the references are `models/reference/<model>.json`.

### References

A reference holds summary statistics only, never draws: for every variable of the parameters block, the mean and sd of x, of x², and of (x − mean)², and the Monte Carlo standard error (MCSE) of each mean. The largest, bym2_offset_only, is 516 KB; the limit is 1 MB.

- 12 models use posteriordb's reference draws (`posterior_database/reference_posteriors/`). posteriordb builds them with Stan: 10 chains, thin 10, 10 000 draws, no divergences and R-hat below 1.01 (`doc/REFERENCE_POSTERIOR_DEFINITION.md` in posteriordb). eight_schools_centered uses the reference that posteriordb lists for it, the one of the non-centred model, which has the same posterior.
- The other 9 have no posteriordb reference. For 8 of them, `apin reference build` built one with posteriordb's recipe in CmdStan: 10 chains, 2000 warmup and 10 000 sampling iterations each, thin 10, adapt delta 0.95. It refuses a result with a divergence or an R-hat of 1.01 or more. A model can name an equivalent reparameterisation to sample from instead (`reference.stan` in the manifest). pilots does this: NUTS diverges on its centred funnel, so its reference comes from `models/reference/pilots.stan`, the same model non-centred.
- kronecker_gp has no reference yet. Its build did not finish in 1 hour 50 minutes on a loaded machine. Its reps get the verdict "no reference": they are scored for speed, and the benchmark leaves the model out.

`models/manifest.json` records for each model whether its reference came from posteriordb or was built here, and `apin list` shows it. A local reference stores its CmdStan version, settings and checks in its `meta`.

### The test

For every reference variable x and each of the three moments f ∈ {x, x², (x − μ_ref)²}, the gate computes the standardised squared bias

    b²(f) = (mean_run(f) − E_ref[f])² / Var_ref[f].

This is the statistic of Robnik, Cohn-Gordon and Seljak (2025, "Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo", arXiv:2503.01707, section 7, eq. 15), the paper behind BlackJAX's MAMS sampler, who also take the worst parameter. They use x²; apin adds x for location errors and (x − μ)² for scale errors when |μ| is much larger than σ, where x² is dominated by the mean. The statistic is unit-free: b² = 0.01 means the expectation is off by 0.1 posterior standard deviations, the error of about 100 independent draws. The gate's threshold is b² = 0.01 for the worst (variable, moment) pair.

Both expectations in b² are estimates. A correct sampler with few effective draws has b² of order 1/ESS, which can exceed 0.01 by chance. With the naive rule (fail when b² > 0.01), 182 of 300 simulated correct runs with 400 to 2000 draws fail. So the gate tests a lower confidence bound instead. Let s = √(MCSE_run² + MCSE_ref²), where MCSE_run comes from the run's ESS for the mean of f (posterior's `mcse_mean`). With z the two-sided Bonferroni critical value for all m = 3 × (number of variables) tests at family-wise level α = 0.01,

    b²_lower = max(0, |mean_run(f) − E_ref[f]| − z·s)² / Var_ref[f],   and the rep fails when b²_lower > 0.01.

A correct sampler fails only if some difference exceeds z·s, so its false-fail rate is at most α = 1 % per run, whatever its ESS. `tests/test_gate.py::test_false_fail_rate_is_at_most_alpha` checks this on 300 simulated correct runs (AR(1) chains with autocorrelation 0, 0.5 and 0.9, five variables with means up to 10 standard deviations from zero, against a reference estimated from 10 000 draws). None of the 300 fails; the test allows at most 9. Bonferroni and the maximum over moments make the rate lower than 1 % in practice.

The price is power that depends on ESS. At an ESS of 400 per moment and 15 tests, the gate catches a bias of about 0.27 standard deviations and up; it approaches 0.1 standard deviations as ESS grows. This is the right way round for a speed benchmark: a sampler that claims a high ESS has a small MCSE and is checked tightly.

### The other checks

Before the bias test, the gate fails a rep for:

- a non-finite draw in any variable;
- a reference variable the run did not write;
- split R-hat (Vehtari et al. 2021) above 1.05 on any variable. Chains that disagree make the MCSE unreliable. A stuck chain inflates its MCSE so much that the bias test alone would pass it; R-hat catches it. The threshold is looser than the usual 1.01 because 4 chains of 1000 draws often exceed 1.01 on hard models without being wrong. The summary still counts models above 1.01.

The verdict is cached in `gate.json` next to the rep. It is recomputed when the rep or the reference changes.

## Deviation from the old campaign

The old `run_pathfinder.py` ran Pathfinder once per model at seed 20260819, wrote one init file per chain, and reused those inits for every rep. Its wall time excluded Pathfinder. The `pathfinder` adapter runs Pathfinder in every rep at the rep's seed and includes its time in the wall. The new numbers are therefore lower than the old `pathfinder_medians.csv` numbers, and the reps are independent. Two smoke runs on two small models (eight_schools_noncentered, blr; one rep) gave ESS/s ratios of 0.750 and 0.919 against plain CmdStan. Runs that short are too noisy to estimate the cost. The old figures in `results/tables/pathfinder_medians.csv` were not rerun with the new adapter.

The old walnutpie arms (W-63 to W-133) used Pathfinder inits listed in `scratch/w63/manifest.csv` and `--metric-window 50`. That manifest and its init files were never committed to the old repository, so the exact inits are lost; the `pathfinder` adapter can make new ones. The `walnutpie` adapter uses neither by default. Pass `--opt flags="--metric-window 50 --init-file ..."` to reproduce that posture.

## Measuring a patch

1. Pre-register. Before the first run, write down in the record the arms, the models, the metric, the expected effect and the result that would count as a failure. The old campaign retracted several claims (REPORT.md, section 7).
2. Build a baseline arm and a patched arm that differ only in the patch: same compiler, flags, submodule commits otherwise, and machine. Record each arm's identity. `run.json` keeps what the adapter's `describe()` reports.
3. Run the bit-identity canary. On a model the patch must not touch, the two arms must give byte-identical draws (compare `md5sum` of the chain CSVs). If they differ, a flag, a header or a stale object differs between the builds. Find it before you measure anything.
4. For a change to a gradient, check it against finite differences on a few random points (stan-math's `test/unit/math/mix` framework, or BridgeStan's `log_density_gradient` against a central difference). A patch that is faster and wrong looks like a win in ESS/s.
5. Run the grid with a new `--label` per arm and score with `--baseline`.
6. Write the record under `results/records/` and update `upstream/STATUS.md` in the same commit. If a later result contradicts a record, add a dated correction at the top of the old record and at the place where the claim was made. Do not delete the old text.

## The log-density seam

The `logdensity` adapter (`harness/adapters/logdensity.py`) runs a sampling algorithm written in Python against any core model, through BridgeStan. It is the interface of the benchmark in BENCH.md. A submission is one file that defines

```python
def sample(problem: Problem, budget: Budget, rng: numpy.random.Generator) -> numpy.ndarray:
    ...  # returns unconstrained draws of shape (budget.chains, budget.draws, problem.dim)
```

with these types from `harness/logdensity.py`:

```python
@dataclass(frozen=True)
class Problem:
    dim: int
    log_density: Callable[[np.ndarray], float]
    log_density_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]]
    initial_points: np.ndarray  # (chains, dim), one starting point per chain

@dataclass(frozen=True)
class Budget:
    chains: int           # independent chains to return
    draws: int            # draws per chain to return
    max_evaluations: int  # calls to log_density plus log_density_gradient, all chains, warmup included
    max_seconds: float    # wall time of the whole sample() call
```

The contract:

- The density is on the unconstrained space, with the Jacobian of the constraining transform and all constants (BridgeStan with `jacobian=True, propto=False`). theta must be a float array of shape (dim,).
- A point the model rejects (Stan raises a domain error) has log density −inf and a NaN gradient. This is how Stan's own samplers treat it. The submission never sees the exception.
- The harness counts every call. Density and gradient calls count the same, one evaluation each, against one limit. The call that would exceed `max_evaluations`, and any call after `max_seconds`, raises `BudgetExceeded`. From then on every call raises, and the run fails even if the submission catches the error.
- The defaults are `max_evaluations` = chains × (warmup + draws) × 1023, the most NUTS can spend at its default tree depth of 10, and `max_seconds` = 1800. `--opt max_evals=N` and `--opt max_seconds=S` change them. The submission is killed 300 s after `max_seconds`.
- `initial_points` are uniform on (−2, 2), Stan's default, from a generator seeded by the rep's seed. `rng` is a second, independent generator from the same seed. A submission that needs more streams spawns them from `rng`.
- The return value must have exactly the shape (chains, draws, dim). Every returned draw must be a point the submission passed to `log_density` or `log_density_gradient`, bit for bit; the harness keeps a hash of every evaluated point, in call order. A chain's draws must keep the order in which they were first evaluated, and no point may appear in two chains. These checks rule out draws the submission did not pay for, shuffled chains, which hide autocorrelation from ESS, and copied chains. They also rule out returning averages of draws. The harness maps every draw to the constrained space with BridgeStan (`param_constrain`, with transformed parameters and generated quantities) and writes one CSV per chain. A draw that cannot be mapped is written as NaN, which fails the gate.
- The submission may import only numpy, scipy and the standard modules math, typing, dataclasses, functools, itertools, collections and `__future__`, and may not name the builtins `open`, `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `input`, `globals` or `vars`. An AST check enforces both before the run. It is a lint, not a sandbox (see BENCH.md).
- The submission runs in a child process, `python -m harness.logdensity spec.json`, with `OMP_NUM_THREADS=1`. Its stdout and stderr go to `submission.log` in the rep directory.

`timing.wall_s` is the time inside `sample()`. Loading the model and constraining the draws are excluded. The chains share one process, so each `chain_s` entry is `wall_s / chains` and `ess_per_s_chain` is ESS per second of the single core the submission ran on. `ess_per_s_wall` of a CmdStan run uses 4 cores in parallel, so compare `ess_per_s_chain` or `ess_per_eval` between a submission and CmdStan, never `ess_per_s_wall`. `timing.evaluations` holds the count, and `timing.extra` holds the density and gradient counts apart and the limits in force.

`examples/samplers/` holds two submissions: `hmc.py`, HMC with a diagonal metric, a step size tuned by dual averaging in warmup and a jittered number of steps, and `ula_wrong.py`, the unadjusted Langevin algorithm with a large step, which is wrong on purpose. Run one with

```bash
uv run apin run --sampler logdensity --label hmc --opt sampler=examples/samplers/hmc.py --models easy/small
```

The first run of each model builds a BridgeStan library, which takes one to two minutes per model; later runs reuse it from `build/`. The BridgeStan tree is the one the `bridgestan` Python package keeps (it downloads it on first use) unless `--opt bridgestan=PATH` or `BRIDGESTAN` names another.

## Runner output schema

Every adapter writes the same layout, so `apin score` never knows which sampler ran:

```
runs/<label>/<model>/rep<r>/
    chain_<c>.csv   one file per chain, c = 0..chains-1
    run.json        the request, the sampler identity and the timing
    DONE | FAILED   marker
    ess.json        score cache, written by apin score
    gate.json       correctness verdict, written by apin score
    spec.json, evaluations.json, submission.log   logdensity adapter only
```

`chain_<c>.csv` may begin with `#` comment lines. The first other line is a header of variable names, flattened in Stan's column-major style (`theta.1`, `L.2.1`). Each further line is one post-warmup draw. Columns that end in `__` are diagnostics. CmdStan's own CSVs also hold the warmup draws, before the `# Adaptation terminated` comment; the reader keeps them apart.

`run.json` has these fields:

| field | content |
|---|---|
| `label`, `sampler`, `model`, `rep`, `seed` | the cell |
| `chains`, `warmup`, `draws` | the budget |
| `identity` | the adapter's `describe()`: binary paths, versions, flags |
| `host` | machine name |
| `timing.wall_s` | wall time of the rep, compile excluded |
| `timing.chain_s` | each chain's own time, warmup plus sampling |
| `timing.warmup_s`, `timing.sampling_s` | per chain, or null when the sampler does not report them |
| `timing.grads_sampling`, `timing.grads_warmup` | gradient evaluations summed over chains, or null |
| `timing.evaluations` | every density and gradient evaluation, when the harness counts them itself (logdensity), else null |
| `timing.divergences` | summed over chains, or null when the sampler does not report them |
| `timing.extra` | anything adapter-specific |

Use null for an unknown value. The old runners wrote -1, and one aggregation step treated it as a count.

## Hazards

The old campaign lost the most time to these. They apply to any compiled upstream, not only to Stan.

- Stale objects. Make and CMake builds can keep an object compiled from an old header or with old flags and link it into a new binary. After a change to flags, compiler or a submodule commit, clean the build (`make clean-all` for CmdStan, delete `bridgestan.o` for BridgeStan, `--clean-first` for CMake). The bit-identity canary catches the rest.
- A new option wired in one place. A new command-line option must reach every dispatch site. In the old campaign a flag was parsed, accepted and ignored by one sampler template, and a whole experiment (W-71) ran with it doing nothing. Test that the option changes the output before measuring.
- Shared state across threads. A library that evaluates a model may not be thread-safe. A BridgeStan model built without `STAN_THREADS=true` corrupts memory when several threads call it.
- One random stream per object. Each chain needs its own generator, and each distribution object must draw from one generator only. Sharing a distribution object between streams makes results depend on thread timing.
- Argument lists. Pass a command as a list of separate tokens (`["method=sample", "num_warmup=1000"]`), never as one string with spaces.
- Ragged output. A chain can stop early or write a partial last line. Cut all chains to the shortest complete length before scoring (`apin score` does this).
- Large output in git. `runs/`, `build/` and `.cache/` are in `.gitignore`. Add files by name, never with `git add -A`. The old repository once committed 44 GB of run output.
- Load from other work. Another process on the machine changes wall times by 10 to 25 percent. Compare arms that ran under the same load, or use `ess_per_s_chain` and `ess_per_grad`, which depend less on it.

## Using a patched build

Point the harness at a patched build with `--cmdstan` or the `CMDSTAN`, `BRIDGESTAN` and `WALNUTPIE_CLI` variables. The build notes for each upstream are in `upstream/<name>.md`. Give each build its own `--label`. The model cache in `build/` is keyed by the model source, the flags, the path of the CmdStan or BridgeStan tree and the stanc version. It cannot see a change inside a tree at the same path, such as a new branch checked out in `upstream/math`. After you switch a submodule branch, delete `build/` or pass a new `--build` directory per arm.
