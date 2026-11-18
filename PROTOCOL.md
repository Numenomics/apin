# Measurement protocol

This file says how apin measures a sampler or a patch, what a runner must write, and the mistakes that cost the most time in the past. `apin run` and `apin score` implement the protocol. Where they leave a choice to you, this file says what to choose.

## The grid

The core set is the 21 posteriordb models in `models/manifest.json`. Each has a family label (easy/small 4, GLM 3, hierarchical 5, stiff/funnel 5, GP/spatial 4). The set was fixed in the first phase of the old campaign (`results/records/PHASE0_SUMMARY.md`) and has not changed since. `apin list` prints them with their family. Pass a family name to `--models` to run one family.

Each cell is one sampler, one model and one rep:

- 4 chains, 1000 warmup iterations, and the draws per chain in the model's `budget` in `models/manifest.json`: 5000 for most models, 10 000 for lsat_model. These budgets are what stock NUTS needs to be proven right at the gate's tolerance (BENCH.md, "The budget"). `--chains`, `--warmup` and `--draws` override them for experiments, but the scorer marks a rep with another budget wrong. CmdStan saves its warmup draws so that their gradient count is known. `read_csv` keeps them apart, and they are never scored.
- 3 reps for development, 5 for a benchmark result. Rep r uses seed BASE + 1000·r, with BASE 20260819 unless `--base-seed` says otherwise. Chains inside a rep get the chain id or seed offset that their sampler uses (CmdStan `id=c+1`; walnutpie `seed+c`).
- At most 4 cores in use for the whole grid. The old campaign ran on a 12-core Ryzen 9 5900X and kept to 4 cores so that memory bandwidth and turbo states were comparable between arms. A rep's 4 chains run in parallel. Run one grid at a time.

A rep that finishes gets a `DONE` marker; a rep that raises gets `FAILED` with the traceback. A rerun reuses a rep with `DONE` only when its seed, base seed, sampler identity and budget match the new request. Otherwise it deletes the rep and runs it again, so an interrupted grid resumes where it stopped and a changed grid never keeps old rows.

## The score

`apin score --models M --reps R` scores the grid of the models in M that have a reference, with R reps each. It walks that grid, not the run directories: a rep the grid declares and a label lacks is wrong. It computes, for every rep, the rank-normalized bulk and tail ESS and split R-hat of every series the gate checks (Vehtari et al. 2021; `harness/ess.py` reproduces the R package posterior 1.7.0 to 10 significant digits). Columns whose names end in `__` are sampler diagnostics and are not scored; lp__ is recomputed by the harness. Chains of unequal length are cut to the shortest.

Every rep gets a verdict from the correctness gate (next section): right, wrong or unproven. The credited ESS of a rep is the smallest bulk or tail ESS over every series. Cost is reported two ways, and only for a rep that is right:

| column | meaning |
|---|---|
| `ess_per_eval` | credited ESS divided by every density and gradient evaluation, warmup included. It does not depend on the machine and does not let a sampler hide work in warmup. It is the primary speed. It needs complete counts: CmdStan's `n_leapfrog__` over every saved warmup and sampling row with thin = 1, or the logdensity adapter's own count. An adapter that cannot count all its evaluations says why in `timing.uncounted`, and the rep has no `ess_per_eval`. |
| `ess_per_s` | credited ESS divided by the sum of each chain's own time (warmup plus sampling). This removes the effect of parallel dispatch and machine load. W-109 and W-133 used this convention ("stanza sum"). It is the secondary speed. |

Never compare numbers that use different conventions. A report states which one it uses.

BENCH.md, "The score", defines the ranking: models right first, then the geometric mean of the speed ratio to the baseline over the models where both are right, with a bootstrap interval that resamples both. `apin score` writes `per_run.csv`, `per_model.csv`, `ranking.csv` and `table.md`.

## The correctness gate

ESS measures how fast a sampler forgets where it was. It says nothing about whether it converges to the right distribution. A sampler that is fast and wrong would win on ESS per evaluation. So the gate compares each rep with a reference posterior, and a rep earns speed credit only when it is shown to agree. The code is `harness/gate.py`; the references are `models/reference/<model>.json`. BENCH.md, rules 1 to 3, states what the gate decides. This section gives the statistics.

### References

A reference holds summary statistics only, never draws (`harness/reference.py`). It has a set of series: every standardised variable, projections on eigenvectors of the reference correlation, and lp__. For each series it stores four functions (the mean, the mean of the square and the fractions below the reference's 5 % and 95 % quantiles), each with its expected value, its SD and its Monte Carlo standard error (MCSE). It also stores the constants of the model and the correlation of the functions, which the calibration test needs. The largest file, bym2_offset_only, is 877 KB; the limit is 1 MB.

- 12 models use posteriordb's reference draws (`posterior_database/reference_posteriors/`). posteriordb builds them with Stan: 10 chains, thin 10, 10 000 draws, no divergences and R-hat below 1.01 (`doc/REFERENCE_POSTERIOR_DEFINITION.md` in posteriordb). eight_schools_centered uses the reference that posteriordb lists for it, the one of the non-centred model, which has the same posterior.
- The other 9 have no posteriordb reference. For 8 of them, `apin reference build` built one with posteriordb's recipe in CmdStan: 10 chains, 2000 warmup and 10 000 sampling iterations each, thin 10, adapt delta 0.95. It refuses a result with a divergence or an R-hat of 1.01 or more. A model can name an equivalent reparameterisation to sample from instead (`reference.stan` in the manifest). pilots does this: NUTS diverges on its centred funnel, so its reference comes from `models/reference/pilots.stan`, the same model non-centred.
- kronecker_gp has no reference. Its build did not finish in 1 hour 50 minutes on a loaded machine. It is not in the benchmark grid and is not scored.

`models/manifest.json` records for each model whether its reference came from posteriordb or was built here, and `apin list` shows it. A local reference stores its CmdStan version, settings and checks in its `meta`.

### The test

For each checked quantity j (a function of a series), with e_j the error of the run mean in reference SDs of the function and s_j its standard error (run and reference MCSE combined), the gate calls the rep

- wrong when some |e_j| − z_w·s_j exceeds the tolerance 0.1, z_w the Bonferroni value for all quantities of all reps at 1 % per model;
- right when every |e_j| + 1.645·s_j is below 0.1;
- unproven otherwise.

Both tests trust s_j. So they run only on chains that look mixed: split R-hat below 1.01 and bulk and tail ESS of at least 400 on every series, and no two chains near-identical. Chains that fail this are unproven, not wrong. With strong autocorrelation and few effective draws, the ESS estimate is itself noisy and understates the MCSE. The old gate called such correct runs wrong.

The rates, from `harness/gate_rates.py` (`results/tables/gate_rates.md`):

- a correct sampler whose chains have mixed: called wrong in 0 of 1100 runs (design: at most 1 % per model); right in 90 % of reps at an ESS of 4000 per variable and in all reps at 20 000;
- a sampler with an error of exactly the tolerance: called right in 0 to 4 % of runs (design: at most 5 % per rep);
- chains that have not mixed (AR(1) at ρ = 0.99 or 0.999 with too few draws): unproven in every run, correct or not.

BENCH.md, "Rule 1: the gate", has the table.

The power to call a bias wrong grows with ESS. A rep is wrong once |e| exceeds 0.1 + z_w·s. At an ESS of 400 per quantity s is about 0.05 (√(1/400) plus the reference's share), and z_w is about 4 for a model with 100 quantities and 5 reps, so the gate calls a bias wrong from about 0.3 SD. At an ESS of 4000, s is about 0.02 and the gate calls a bias wrong from about 0.18 SD. Between the two bounds the rep is unproven, which earns nothing. A rep earns credit only when its error is shown to be below 0.1.

The model's verdict is the worst of its reps (wrong, then unproven), and a model whose reps are all right must also pass the calibration test of BENCH.md, rule 2: the realized errors over all quantities and reps must agree with the claimed MCSEs at p ≥ 0.001.

### The other checks

Before the statistical tests, a rep is wrong for:

- a non-finite draw, or a non-finite lp__ (a draw outside the support, or one the model rejects);
- a reference variable the run did not write;
- a variable that never moves;
- a reference constant the run does not reproduce to 1e-6;
- a function whose reference value is exact and which the run misses by more than 1e-6.

The verdict is cached in `gate.json` next to the rep. It is recomputed when the rep, the reference, the number of reps or a gate constant changes.

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
- The harness counts every call. Density and gradient calls count the same, one evaluation each, against one limit. The call that would exceed `max_evaluations`, and any call after `max_seconds`, raises `BudgetExceeded`. From then on every call raises, and the run fails even if the submission catches the error. A run that overruns `max_seconds` after its last call fails at the end.
- The defaults are `max_evaluations` = chains × (warmup + draws) × 1023, the most NUTS can spend at its default tree depth of 10, and `max_seconds` = 1800. `--opt max_evals=N` and `--opt max_seconds=S` change them. The submission is killed 300 s after `max_seconds`.
- `initial_points` are uniform on (−2, 2), Stan's default, from a generator seeded by the rep's seed. `rng` is a second, independent generator from the same seed. A submission that needs more streams spawns them from `rng`.
- The return value must have exactly the shape (chains, draws, dim). Every returned draw must be a point the submission passed to `log_density` or `log_density_gradient`, bit for bit, and whose log density was finite; the harness keeps a hash of every evaluated point, in call order. A chain's draws must keep the order in which they were first evaluated, and no point may appear in two chains. These checks rule out draws the submission did not pay for, shuffled chains, which hide autocorrelation from ESS, and copied chains. They also rule out returning averages of draws. The harness maps every draw to the constrained space with BridgeStan (`param_constrain`, with transformed parameters and generated quantities) and writes one CSV per chain. A draw that cannot be mapped is written as NaN, which fails the gate.
- The submission may import only numpy, scipy and the standard modules math, typing, dataclasses, functools, itertools, collections and `__future__`, and may not name the builtins `open`, `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `input`, `globals` or `vars`. An AST check enforces both before the run. It is a lint, not a sandbox (see BENCH.md).
- The submission runs in a child process, `python -m harness.logdensity spec.json`, with `OMP_NUM_THREADS=1`. Its stdout and stderr go to `submission.log` in the rep directory.

`timing.wall_s` runs from loading the submission file to converting what `sample()` returns into an array. Loading the model and constraining the draws are excluded. The chains share one process, so each `chain_s` entry is `wall_s / chains` and `ess_per_s` is ESS per second of the single core the submission ran on. `ess_per_s_wall` of a CmdStan run uses 4 cores in parallel, so compare `ess_per_s` (chain time) or `ess_per_eval` between a submission and CmdStan, never a wall time. `timing.evaluations` holds the count, and `timing.extra` holds the density and gradient counts apart and the limits in force.

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
| `label`, `sampler`, `model`, `rep`, `seed`, `base_seed` | the cell |
| `chains`, `warmup`, `draws` | the budget |
| `identity` | the adapter's `describe()`: binary paths, versions, flags |
| `host` | machine name |
| `timing.wall_s` | wall time of the rep, compile excluded |
| `timing.chain_s` | each chain's own time, warmup plus sampling |
| `timing.warmup_s`, `timing.sampling_s` | per chain, or null when the sampler does not report them |
| `timing.grads_sampling`, `timing.grads_warmup` | gradient evaluations summed over chains, or null |
| `timing.evaluations` | every density and gradient evaluation, when the harness counts them itself (logdensity), else null |
| `timing.divergences` | summed over chains, or null when the sampler does not report them |
| `timing.uncounted` | why the counts above leave out some evaluations, or empty when they are complete |
| `timing.extra` | anything adapter-specific; `density_evals` and `gradient_evals` when the harness counts them |

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
- Load from other work. Another process on the machine changes wall times by 10 to 25 percent. Compare arms that ran under the same load, or use `ess_per_s` and `ess_per_eval`, which depend less on it.

## Using a patched build

Point the harness at a patched build with `--cmdstan` or the `CMDSTAN`, `BRIDGESTAN` and `WALNUTPIE_CLI` variables. The build notes for each upstream are in `upstream/<name>.md`. Give each build its own `--label`. The model cache in `build/` is keyed by the model source, the flags, the path of the CmdStan or BridgeStan tree and the stanc version. It cannot see a change inside a tree at the same path, such as a new branch checked out in `upstream/math`. After you switch a submodule branch, delete `build/` or pass a new `--build` directory per arm.
