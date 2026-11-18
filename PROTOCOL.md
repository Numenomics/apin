# Measurement protocol

This file says how apin measures a sampler or a patch, what a runner must write, and the mistakes that cost the most time in the past. `apin run` and `apin score` implement the protocol. Where they leave a choice to you, this file says what to choose.

## The grid

The core set is the 21 posteriordb models in `models/manifest.json`. Each has a family label (easy/small 4, GLM 3, hierarchical 5, stiff/funnel 5, GP/spatial 4). The set was fixed in the first phase of the old campaign (`results/records/PHASE0_SUMMARY.md`) and has not changed since. `apin list` prints them with their family. Pass a family name to `--models` to run one family.

Each cell is one sampler, one model and one rep:

- 4 chains, 1000 warmup iterations, 1000 draws per chain. Warmup draws are not saved.
- 3 reps. Rep r uses seed 20260819 + 1000·r. Chains inside a rep get the chain id or seed offset that their sampler uses (CmdStan `id=c+1`; walnutpie `seed+c`).
- At most 4 cores in use for the whole grid. The old campaign ran on a 12-core Ryzen 9 5900X and kept to 4 cores so that memory bandwidth and turbo states were comparable between arms. A rep's 4 chains run in parallel. Run one grid at a time.

A rep that finishes gets a `DONE` marker; a rep that raises gets `FAILED` with the traceback. A rerun skips reps with `DONE`, so an interrupted grid resumes where it stopped.

## The score

`apin score` computes, for every rep, the rank-normalized bulk and tail ESS and split R-hat of every variable (Vehtari et al. 2021; `harness/ess.py` reproduces the R package posterior 1.7.0 to 10 significant digits). Columns whose names end in `__` are sampler diagnostics and are not scored. Chains of unequal length are cut to the shortest before scoring.

The quality number of a rep is `ess_bulk_min`, the smallest bulk ESS over all variables. It is a pessimistic number with a large variance between reps, so the per-model value is the median over reps and the summary over models is a geometric mean of those medians. A model counts as failed when its median R-hat is above 1.01.

Cost is reported three ways:

| column | meaning |
|---|---|
| `ess_per_s_wall` | `ess_bulk_min` divided by the wall time of the rep, chains in parallel. This is what a user waits for. |
| `ess_per_s_chain` | `ess_bulk_min` divided by the sum of each chain's own sampler time (warmup plus sampling). This removes the effect of parallel dispatch and machine load. W-109 and W-133 used this convention ("stanza sum"). |
| `ess_per_grad` | `ess_bulk_min` divided by the gradient evaluations in the sampling phase. It does not depend on the machine. |

Never compare numbers that use different conventions. A report states which one it uses. With `--baseline LABEL`, `apin score` writes per-model ratios against that label and prints their geometric mean.

Agreement with the true posterior is not scored. posteriordb has reference draws for some core models (see `upstream/posteriordb.md`). A sampler that is fast and wrong gets a good ESS/s number from apin. Check correctness with the bit-identity and finite-difference checks below until this is added.

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

## Runner output schema

Every adapter writes the same layout, so `apin score` never knows which sampler ran:

```
runs/<label>/<model>/rep<r>/
    chain_<c>.csv   one file per chain, c = 0..chains-1
    run.json        the request, the sampler identity and the timing
    DONE | FAILED   marker
    ess.json        score cache, written by apin score
```

`chain_<c>.csv` may begin with `#` comment lines. The first other line is a header of variable names, flattened in Stan's column-major style (`theta.1`, `L.2.1`). Each further line is one post-warmup draw. Columns that end in `__` are diagnostics.

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
