# Measurement protocol

This file says how apin measures a sampler or a patch, what a runner must write, and the mistakes that cost the most time in the past. `apin run` and `apin score` implement the protocol. Where they leave a choice to you, this file says what to choose.

## The grid

The core set is the 21 posteriordb models in `models/manifest.json`. Each has a family label (easy/small 4, GLM 3, hierarchical 5, stiff/funnel 5, GP/spatial 4). The set was fixed in the first phase of the old campaign (`results/records/PHASE0_SUMMARY.md`) and has not changed since. `apin list` prints them with their family. Pass a family name to `--models` to run one family.

Each cell is one sampler, one model and one rep:

- 4 chains, 1000 warmup iterations, and the draws per chain in the model's `budget` in `models/manifest.json`: 5000 for most models, 10 000 for lsat_model and 20 000 for accel_gp. These budgets are what stock NUTS needs to be proven right at the gate's tolerance (BENCH.md, "The budget"). `--chains`, `--warmup` and `--draws` override them for experiments, but the scorer marks a rep with another budget wrong. CmdStan saves its warmup draws so that their gradient count is known. `read_csv` keeps them apart, and they are never scored.
- 3 reps for development, 5 for a benchmark result. Rep r uses seed BASE + 1000·r, with BASE 20260819 unless `--base-seed` says otherwise. Chains inside a rep get the chain id or seed offset that their sampler uses (CmdStan `id=c+1`; walnutpie `seed+c`).
- Use at most 4 cores for the whole grid. The harness does not enforce this. The old campaign ran on a 12-core Ryzen 9 5900X and kept to 4 cores so that memory bandwidth and turbo states were comparable between arms. A rep's 4 chains run in parallel. Run one grid at a time.

A rep that finishes gets a `DONE` marker; a rep that raises gets `FAILED` with the traceback. A rerun reuses a rep with `DONE` only when its seed, base seed, sampler identity and budget match the new request. Otherwise it deletes the rep and runs it again, so an interrupted grid resumes where it stopped and a rep whose request changed is not kept.

## The score

`apin score --models M --reps R` scores the grid of the models in M that have a reference, with R reps each. It walks that grid, not the run directories: a rep the grid declares and a label lacks is wrong, and so is a rep whose run.json names another model, rep, seed, base seed or budget than its cell. `--submission LABEL=PATH` makes every rep of that label match the file's sha256; a LABEL that is not scored is refused. It computes, for every rep, the rank-normalized bulk and tail ESS and split R-hat of every series the gate checks (Vehtari et al. 2021; `harness/ess.py` reproduces the R package posterior 1.7.0 to a relative 1e-9 on the fixture in `tests/test_ess.py`). Columns whose names end in `__` are sampler diagnostics and are not scored; lp__ is recomputed by the harness. A rep whose chains hold another number of draws than the budget, or are cut short, is wrong; the scorer does not cut chains to the shortest.

Every rep gets a verdict from the correctness gate (next section): right, wrong or unproven. The credited ESS of a rep is the smallest bulk or tail ESS over every series. Cost is reported two ways, and only for a rep that is right:

| column | meaning |
|---|---|
| `ess_per_cost` | **Primary efficiency:** credited ESS divided by target-calibrated cost `n_density + r_m * n_gradient (+ r_hvp * n_hvp + r_hess * n_hess)`, warmup included, where `r_m >= 1.0` is measured once per model through BridgeStan (`models/costs.json`). Missing `r_m` is a hard error. Checkpoint `cost_to_gate` (earliest prefix checkpoint passing the 0.1-SD gate) is also recorded per rep and compared via paired hierarchical bootstrap. |
| `ess_per_eval` | Diagnostic 1:1 efficiency: credited ESS divided by every density and gradient evaluation, warmup included. It does not depend on load, depends on the machine only through floating-point results, and does not let a sampler hide work in warmup. Needs complete counts; an adapter that cannot count all evaluations sets `timing.uncounted`. |
| `ess_per_s` | credited ESS divided by the sum of the record's `chain_s` entries. For CmdStan and `logdensity`, each entry is that chain's own time (warmup plus sampling), removing parallel dispatch. For nutpie every entry is the wall time of the whole batch. `overhead_ratio` (`overhead_s / eval_s`) is printed alongside. |

Never compare numbers that use different conventions. A report states which one it uses.

BENCH.md, "The score", defines the ranking: models right first, then the geometric mean of the speed ratio to the baseline over the models where both are right, with a bootstrap interval that resamples both. `apin score` writes `per_run.csv`, `per_model.csv`, `ranking.csv` and `table.md`.

## The correctness gate

ESS measures how fast a sampler forgets where it was. It says nothing about whether it converges to the right distribution. A sampler that is fast and wrong would win on ESS per evaluation. So the gate compares each rep with a reference posterior, and a rep earns speed credit only when it is shown to agree. The code is `harness/gate.py`; the references are `models/reference/<model>.json`. BENCH.md, rules 1 to 3, states what the gate decides. This section gives the statistics.

### References

A reference holds summary statistics only, never draws (`harness/reference.py`). It has a set of series: every standardised variable, projections on eigenvectors of the reference correlation, and lp__. For each series it stores four functions (the mean, the mean of the square and the fractions below the reference's 5 % and 95 % quantiles), each with its expected value, its SD and its Monte Carlo standard error (MCSE). It also stores the constants of the model and tr(C²), C the correlation of the functions, which the calibration test needs. Each half's estimate and MCSE of every function are in `models/reference/halves/<model>.json`. The largest file, bym2_offset_only.json, is 886 KB, and its halves file is 520 KB; the limit is 1 MB per file.

- 12 models use posteriordb's reference draws (`posterior_database/reference_posteriors/`). posteriordb builds them with Stan: 10 chains, thin 10, 10 000 draws, no divergences and R-hat below 1.01 (`doc/REFERENCE_POSTERIOR_DEFINITION.md` in posteriordb). eight_schools_centered uses the reference that posteriordb lists for it, the one of the non-centred model, which has the same posterior.
- The other 9 have no posteriordb reference. For 8 of them, `apin reference build` built one with posteriordb's recipe in CmdStan, but with 20 chains instead of 10: 2000 warmup and 10 000 sampling iterations each, thin 10, adapt delta 0.95. It refuses a result with a divergence or an R-hat of 1.01 or more. A model can name an equivalent reparameterisation to sample from instead (`reference.stan` in the manifest). pilots does this: NUTS diverges on its centred funnel, so its reference comes from `models/reference/pilots.stan`, the same model non-centred.
- kronecker_gp has no reference. Its build did not finish in 1 hour 50 minutes on a loaded machine. It is not in the benchmark grid and is not scored.

`models/manifest.json` records for each model whether its reference came from posteriordb or was built here, and `apin list` shows it. A local reference stores its CmdStan version, settings and checks in its `meta`; the version is the one named in the chain CSVs it was summarised from. A rebuild reuses the CSVs in `build/reference/<model>` only when every one is complete and its header names the requested settings and the version of the CmdStan it is given, release suffix included (2.39.0-rc1 is not 2.39.0); otherwise it samples again.

### The test

For each checked quantity j (a function of a series), with e_j the error of the run mean in reference SDs of the function and s_j its standard error (run and reference MCSE combined), the gate calls the rep

- wrong, before the tests below, when a function whose reference value is exact (its reference SD is 0) misses that value by more than 1e-6 (relative) and the halves agree on it. When the halves disagree, the miss only blocks right: the rep goes through the tests below, can still be wrong on another quantity, and is unproven where it would have been right;
- wrong when some |e_j| − z_w·s_j exceeds the tolerance 0.1 on a quantity the two halves of the reference agree on, z_w the Bonferroni value for all quantities of all reps at 1 % per model. The halves disagree when their estimates differ by more than 5 times the smaller of their two MCSEs. A half whose MCSE is 0 claims an exact estimate, so against it any difference above 1e-6 is a disagreement, relative to the larger magnitude of the two estimates when that is above 1 in absolute value; with both MCSEs positive, 5 times the smaller decides however small it is. A quantity the halves disagree on can keep the rep from right but cannot make it wrong (BENCH.md, rule 1);
- right when every |e_j| + 1.645·s_j is below 0.1 and no exact value is missed;
- unproven otherwise.

Both tests trust s_j. So they run only on chains that look mixed: split R-hat below 1.01 and bulk and tail ESS of at least 400 on every series, and no two chains near-identical. Chains that fail this are unproven, not wrong. With strong autocorrelation and few effective draws, the ESS estimate is itself noisy and understates the MCSE. The old gate called such correct runs wrong.

The rates, from `harness/gate_rates.py` (`results/tables/gate_rates.md`):

- a correct sampler whose chains have mixed: called wrong in 0 of 1100 runs (design: at most 1 % per model); right in 90 % of reps at an ESS of 4000 per variable and in all reps at 20 000. This was measured on Gaussian AR(1) chains against references whose MCSE is accurate. A reference that missed rare excursions understates its MCSE and breaks the rate: the halves rule removes the false wrongs where one half saw the excursions (5 of 120 trials before, 0 after) and not those where neither half did (3 of 150, with or without it);
- a sampler with an error of exactly the tolerance: called right in 0 to 4 % of runs (nominal: at most 5 % per rep, when the run's MCSE estimates its error; a sampler with rare jumps that its runs do not show is called right far more often, BENCH.md, rule 1);
- chains that have not mixed (AR(1) at ρ = 0.99 with 1000 draws per chain, and at ρ = 0.999 with 1000 or 50 000): unproven in every run, correct or not.

BENCH.md, "Rule 1: the gate", has the table.

The power to call a bias wrong grows with ESS. A rep is wrong once |e| exceeds 0.1 + z_w·s. At an ESS of 400 per quantity s is about 0.05 (√(1/400) plus the reference's share), and z_w is about 4 for a model with 100 quantities and 5 reps, so the gate calls a bias wrong from about 0.3 SD. At an ESS of 4000, s is about 0.02 and the gate calls a bias wrong from about 0.19 SD. Between the two bounds the rep is unproven, which earns nothing. A rep earns credit only when its error is shown to be below 0.1.

The model's verdict is the worst of its reps (wrong, then unproven), and a model whose reps are all right must also pass the calibration test of BENCH.md, rule 2: the realized errors over all quantities and reps must agree with the claimed MCSEs at p ≥ 0.001. The threshold is nominal: under an idealised Gaussian null the test refused 0.48 % of correct models with the correlations of eight_schools_noncentered and 0.16 % with those of hier_2pl (BENCH.md, rule 1, after the table of thresholds).

### The other checks

Before the statistical tests, a rep is wrong for:

- a non-finite draw, or a non-finite lp__ (a draw outside the support, or one the model rejects);
- a reference variable the run did not write;
- a variable that never moves;
- a reference constant the run does not reproduce to 1e-6 (relative, for constants above 1 in absolute value);
- a function whose reference value is exact and which the run misses by more than 1e-6, when the halves of the reference agree on it. Where they disagree, the reference saw something one half did not, and the miss makes the rep unproven instead.

The verdict is cached in `gate.json` next to the rep. It is recomputed when the rep's chain files, the reference, the number of reps or the code the verdict comes from changes. The key holds a sha256 of that code (`gate.py`, `ess.py`, `reference.py`, `cmdstan.py` and `score.py`), so a cache written by an older gate is not reused. The key does not cover the Stan model, its data or the BridgeStan build that compute lp__: after a change to those, delete the `gate.json` files.

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

## Slice 2 kernel ABI & cost unit (`harness/logdensity.py`, `harness/cost.py`)

- Submissions implement `adapt(oracle, budget, rng) -> (tuning, chain0, betas, aux_spec)` and `transition(tuning, chain, oracle, rng) -> chain_next`.
- Each chain runs `adapt`, `sample`, and unbilled `replay` in separate `OMP_NUM_THREADS=1` child processes under a hard wall-clock deadline; raw unconstrained draws are stored in `unconstrained.npz` with shape `(runs, walkers, draws, dim)` and cold walkers (`beta == 1.0`) are constrained to `chain_<c>.csv`.
- `timing.extra` records split `adapt_*` and `transition_*` counts (`density_evals`, `gradient_evals`, `hvp_evals`, `hess_evals`), `eval_s`, `overhead_s`, `overhead_ratio`, `walkers`, `betas`, `tempered`, `tuning_sha256`, `replay_singles`, `replay_window`, and `bridgestan_key`.
- `apin costs build` measures `r_grad` at raw reference draws into `models/costs.json`; `apin score` ranks by models passing the 0.1-SD gate, then by `ess_per_cost` against baseline (keeping 1:1 `ess_per_eval`, `ess_per_s`, and `overhead_ratio` alongside), and hashes Stan model, data, reference+halves, BridgeStan build, and gate source in `gate.json`.

---

## The log-density seam

The `logdensity` adapter (`harness/adapters/logdensity.py`) runs a Markov chain sampler written in Python against any core model, through BridgeStan, using the Slice 2 `adapt`/`transition` kernel ABI (`harness/logdensity.py`) over out-of-process brokered oracle IPC (`harness/ipc.py`). A submission is one file that defines

```python
def adapt(oracle: Oracle, budget: Budget, rng: numpy.random.Generator) -> tuple[tuning, chain0, betas, aux_spec]:
    ...

def transition(tuning, chain: Chain, oracle: Oracle, rng: numpy.random.Generator) -> Chain | tuple[np.ndarray, Any]:
    ...
```

with these types from `harness/logdensity.py`:

```python
class Oracle:
    dim: int
    block_shape: tuple[tuple[int, ...], ...] | None  # None unless capability 'block_shape' enabled
    log_density(theta) -> float
    log_density_gradient(theta) -> tuple[float, np.ndarray]
    hvp(theta, v) -> tuple[float, np.ndarray]        # requires priced 'hvp' capability
    hess(theta) -> tuple[float, np.ndarray, np.ndarray]  # requires priced 'hess' capability

@dataclass(frozen=True)
class Budget:
    chains: int           # independent runs in the rep (informational)
    draws: int            # post-warmup transition() calls for this run
    max_evaluations: int  # total oracle calls for this run (adapt + transition combined)
    max_seconds: float    # wall time limit for this run

@dataclass(frozen=True)
class Chain:
    position: np.ndarray  # (dim,) for single walker or (K, dim) for K walkers
    aux: Any = None       # typed auxiliary Markov state matching aux_spec
```

The contract:

- The density is on the unconstrained space, with the Jacobian of the constraining transform and all constants (BridgeStan with `jacobian=True, propto=False`). `theta` must be a float array of shape `(dim,)`.
- A point the model rejects (Stan raises a domain error) has log density `−inf` and a NaN gradient. The submission never sees the exception.
- The harness counts every call in the parent process over a Unix socketpair (`harness/ipc.py`), so neither `Counter` nor BridgeStan exists in the submission's process. Every call is charged before evaluation, at rejected points too, and `adapt` vs. `transition` counts are recorded separately (`adapt_*`, `transition_*`).
- The defaults are `max_evaluations` = chains × (warmup + draws) × 1023 and `max_seconds` = 1800. `--opt max_evals=N` and `--opt max_seconds=S` override them (non-grid limits are marked wrong by `apin score`).
- `adapt()` runs in one clean child process (`OMP_NUM_THREADS=1`), validates `(tuning, chain0, betas, aux_spec)`, checks `chain0` finite provenance, and serializes `tuning` canonically (no pickle) with a SHA-256 digest. `transition()` runs in a second clean child process with derived per-step RNGs, per-step finite provenance checking, and post-run `tuning` SHA-256 verification. A third clean child process replays ~20 random steps plus a 3-step contiguous window bit-for-bit (unbilled).
- The submission may import only `numpy`, `scipy`, and standard modules `math`, `typing`, `dataclasses`, `functools`, `itertools`, `collections`, and `__future__`, and may not name `open`, `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `input`, `globals`, or `vars`. An AST check enforces both before execution (a lint, not an OS container sandbox).

`timing.wall_s` is the sum of per-chain `adapt + sample` times (`chain_s` holds each chain's time); model compilation and post-run constraining are excluded. `timing.evaluations` holds total calls and `timing.extra` holds split `adapt_*`/`transition_*` counts, `eval_s`, `overhead_s`, `overhead_ratio`, `walkers`, `betas`, `tempered`, `tuning_sha256`, `replay_singles`, `replay_window`, and `bridgestan_key`.

CmdStan's gradient count (`adapters/cmdstan.py`, `gradients`) follows the CmdStan 2.39 source in `stan/src/stan/mcmc/hmc`:

- One gradient per leapfrog step (`n_leapfrog__`), and one at the start of every transition, warmup and sampling (`base_nuts.hpp`, `hamiltonian_.init`). `n_leapfrog__` leaves the second out.
- 2k + 4 gradients per step-size search (`base_hmc.hpp`, `init_stepsize`), where k is the number of times the search halved or doubled the step size. There is one search before warmup and one after each metric update, six in all at 1000 warmup iterations. CmdStan does not report k. The adapter finds it by replaying the dual averaging (`stepsize_adaptation.hpp`) from `accept_stat__` and `stepsize__` of the warmup rows. A trace that does not replay to 1e-4 in log2 leaves the run uncounted. `tests/test_adapters.py` checks the rebuilt k against a CmdStan chain.
- Not counted: the initialisation attempts (`services/util/initialize.hpp`). Each costs one density and, if that is finite, one gradient. There are at most 100, so 2 to 200 evaluations per chain. The rest of a chain costs at least 2 × (warmup + draws), 12 000 at the bench budget.

`examples/samplers/` holds two submissions: `hmc.py`, HMC with a diagonal metric, a step size tuned by dual averaging in warmup and a jittered number of steps, and `ula_wrong.py`, the unadjusted Langevin algorithm with a large step, which is wrong on purpose. Run one with

```bash
uv run apin run logdensity --label hmc --opt sampler=examples/samplers/hmc.py --models easy/small
```

The first run of each model builds a BridgeStan library, which takes one to two minutes per model; later runs reuse it from `build/`. The BridgeStan tree is the one the `bridgestan` Python package keeps (it downloads it on first use) unless `--opt bridgestan=PATH` or `BRIDGESTAN` names another.

## Runner output schema

Every adapter writes the same layout, so `apin score` reads every rep from the same files:

```
runs/<label>/<model>/rep<r>/
    chain_<c>.csv   one file per chain, c = 0..chains-1
    run.json        the request, the sampler identity and the timing
    DONE | FAILED   marker
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
| `timing.extra.step_size_searches` | CmdStan: per chain, the halvings or doublings of each step-size search |
| `timing.evaluations` | every density and gradient evaluation, when the harness counts them itself (logdensity), else null |
| `timing.divergences` | summed over chains, or null when the sampler does not report them |
| `timing.uncounted` | why the counts above leave out some evaluations, or empty when they are complete; a rep the harness metered must leave it empty |
| `timing.extra` | anything adapter-specific; `density_evals`, `gradient_evals`, `max_evaluations` and `max_seconds` when the harness counts them |

Use null for an unknown value. The old runners wrote -1, and one aggregation step treated it as a count.

## Hazards

The old campaign lost the most time to these. They apply to any compiled upstream, not only to Stan.

- Stale objects. Make and CMake builds can keep an object compiled from an old header or with old flags and link it into a new binary. After a change to flags, compiler or a submodule commit, clean the build (`make clean-all` for CmdStan, delete `bridgestan.o` for BridgeStan, `--clean-first` for CMake). The bit-identity canary catches the rest.
- A new option wired in one place. A new command-line option must reach every dispatch site. In the old campaign a flag was parsed, accepted and ignored by one sampler template, and a whole experiment (W-71) ran with it doing nothing. Test that the option changes the output before measuring.
- Shared state across threads. A library that evaluates a model may not be thread-safe. A BridgeStan model built without `STAN_THREADS=true` is not safe when several threads call it (`upstream/bridgestan.md`).
- One random stream per object. Each chain needs its own generator, and each distribution object must draw from one generator only. Sharing a distribution object between streams makes results depend on thread timing.
- Argument lists. Pass a command as a list of separate tokens (`["method=sample", "num_warmup=1000"]`), never as one string with spaces.
- Ragged output. A chain can stop early or write a partial last line. `apin score` calls such a rep wrong, since its draws fall short of the budget; rerun it.
- Large output in git. `runs/`, `build/` and `.cache/` are in `.gitignore`. Add files by name, never with `git add -A`. The old repository once committed 44 GB of run output.
- Load from other work. Other work on the machine changes wall times: in the old campaign, the cost per gradient rose 10 to 25 percent under 4-way concurrency (REPORT.md). Compare arms that ran under the same load, or use `ess_per_s` and `ess_per_eval`, which depend less on it.

## Using a patched build

Point the harness at a patched build with `--cmdstan` or the `CMDSTAN`, `BRIDGESTAN` and `WALNUTPIE_CLI` variables. The build notes for each upstream are in `upstream/<name>.md`. Give each build its own `--label`. The model cache in `build/` is keyed by the model source, the flags and the path of the tree, and for CmdStan also by the stanc version. It cannot see other changes inside a tree at the same path, such as a new branch checked out in `upstream/math`. After you switch a submodule branch, delete `build/` or pass a new `--build` directory per arm.
