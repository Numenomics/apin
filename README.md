# apin

apin is a workbench for measuring changes to MCMC samplers and the libraries under them. It pins the upstream projects as git submodules, keeps the owner's patches on fork branches, and runs any sampler over a fixed set of 21 posteriordb models with one protocol, so that a patched build and its baseline can be compared in effective samples per second. Every run is checked against a reference posterior, so a sampler that is fast and wrong is marked wrong. It also holds the records of a 2026 campaign on Stan's wall time and a ledger of every patch and where it stands upstream.

BENCH.md turns the same pieces into a benchmark. A Markov chain sampler written in Python implements `adapt(oracle, budget, rng)` and `transition(tuning, chain, oracle, rng)` behind out-of-process brokered oracle IPC (`harness/ipc.py`), clean per-phase child processes (`OMP_NUM_THREADS=1`), and bit-exact replay (OS-level container/network sandboxing for untrusted external PRs remains Slice 6). A model earns speed credit only when it passes the 0.1-SD bench gate; ties are broken by ESS per target-calibrated cost unit (`n_density + r_m * n_gradient`) against Stan's NUTS, with paired hierarchical bootstrap CIs and checkpoint cost-to-gate diagnostics reported alongside.

## Layout

| path | what it holds |
|---|---|
| `harness/` | The runner, scorer, and judge. `sampler.py` is the interface; `adapters/` has one file per sampler (cmdstan, pathfinder, pathfinder_only, walnutpie, nutpie, logdensity). `gate.py` is the 0.1-SD bench gate, `cost.py` measures target-calibrated `r_m` weights (`models/costs.json`), `reference.py` reads and builds reference posteriors, `logdensity.py` enforces the Slice 2 `adapt`/`transition` kernel ABI, `ipc.py` isolates submissions from `Counter` and models over a Unix socketpair, `shapes.py` + `invariance.py` run calibrated exact-draw invariance & dispersed-start mixing, `records.py` validates `records/` PR packages & paired hierarchical bootstraps, `judge.py` owns the read-only Lane A1 fixed HMC kernel (`adapt_metric` seam), `autoresearch.py` runs the scaffolded Karpathy ratchet + paired MAP-Elites loop, and `stan_library.py` builds BridgeStan libraries. |
| `codex/` | Schema-validated `INDEX.yaml` and ~40-line technique cards for cross-ecosystem sampler & optimizer tricks (`apin codex validate`). |
| `autoresearch/` | Lane specs (`lanes/A1/program.md`), pre-seeded `NEGATIVE_MEMORY.md`, and versioned anti-gaming `RULINGS.md`. |
| `models/` | The 21 core models and `manifest.json` (posteriordb name, family, data file, shipped or fetched, reference source, benchmark split). |
| `models/reference/` | One reference posterior per model, as summary statistics (`<model>.json`), each half's estimates in `halves/<model>.json`, and the reparameterised programs some references are built from. |
| `examples/samplers/` | Two submissions for the logdensity adapter: a working HMC and a sampler that is wrong on purpose. |
| `data/` | The small data files apin ships; `SOURCES.md` says where each data set comes from. |
| `upstream/` | One submodule per upstream at `upstream/<name>/`, its build notes in `upstream/<name>.md`, and the ledger `upstream/STATUS.md`. |
| `results/` | The old campaign: `records/` (one note per experiment), `tables/` (medians), `campaign/` (summary JSON). |
| `tests/` | Fast unit tests, and slow end-to-end tests marked `slow`. |
| `PROTOCOL.md` | How to measure a sampler or a patch, the correctness gate, the log-density seam, the runner output schema and the known hazards. |
| `BENCH.md` | The benchmark: the task, the rules, the budget, the score, the public and held-out split, and the defences against gaming it. |
| `REPORT.md` | What the campaign found, with the source of every headline number. |
| `ARCHIVE.md` | What the old repository held that apin leaves out. |
| `LICENSE`, `LICENSING.md` | The proposed licence and the licences of the upstreams. |

Nothing in the layout is specific to Stan or to C++. An upstream is a submodule with a build note and a ledger section. A sampler is one adapter file. The current upstreams are Stan projects because the first campaign was about Stan.

## Set up

You need git, [uv](https://docs.astral.sh/uv/) and Python 3.12 or later. For the cmdstan and pathfinder adapters you need a CmdStan tree.

```bash
git clone <apin repository> apin
cd apin
uv sync
uv run apin fetch-data     # downloads the data files that are not shipped, with a sha256 check
uv run apin list           # the samplers that can run here, and the models
```

The submodules are not checked out by `git clone`. Check out only the ones you need:

```bash
git submodule update --init upstream/math upstream/stan
```

None of the submodules needs its own nested submodules for apin, except as the build note for that upstream says (CmdStan and BridgeStan expect stan and math in their own `stan/` directories; `upstream/cmdstan.md` and `upstream/bridgestan.md` show how to point them at `upstream/stan` and `upstream/math`).

## Run a grid and score it

```bash
export CMDSTAN=/path/to/cmdstan        # or pass --cmdstan
uv run apin run cmdstan --models hierarchical --reps 3 --label stock
uv run apin run cmdstan --models hierarchical --reps 3 --label patched --cmdstan /path/to/patched-cmdstan
uv run apin score --models hierarchical --reps 3 --baseline stock
```

`--models` takes `all`, a family name, `public`, `held-out` or a comma list. Runs go to `runs/<label>/`, scores to `scores/`; both are ignored by git. A rerun skips the reps that finished with the same seed, sampler identity and budget, and runs the others again. Adapter options go through `--opt key=value`; each adapter documents its own at the top of its file. `--base-seed` changes the seeds of every rep. PROTOCOL.md says which budget, seeds and cost column to use.

`apin score` scores the grid that `--models` and `--reps` declare, so pass it the same values as `apin run`. A rep the grid declares and the label lacks is wrong, and so is a rep whose `run.json` names another model, rep, seed, base seed or budget. It gates every rep against its reference: right, wrong or unproven (BENCH.md, rule 1). It writes `per_run.csv`, `per_model.csv`, `ranking.csv` and `table.md`. With `--baseline` the ranking has the paired speed ratio and hierarchical bootstrap interval against that label.

Run a Python sampler through the log-density seam, and score it against CmdStan:

```bash
uv run apin run logdensity --label hmc --opt sampler=examples/samplers/hmc.py --models public
uv run apin run cmdstan --label cmdstan --models public
uv run apin score --models public --baseline cmdstan --submission hmc=examples/samplers/hmc.py
```

`--submission` makes the score check that every rep of `hmc` recorded that file's sha256.

Additional CLI commands (Slices 2–5):

```bash
uv run apin costs build --models all                      # measure target-calibrated r_grad into models/costs.json
uv run apin codex validate                                # validate codex/INDEX.yaml schema & card links
uv run apin invariance --sampler examples/samplers/hmc.py # calibrated exact-draw invariance & dispersed-start mixing
uv run apin records validate records/<release>/<track>/<slug>
uv run apin autoresearch eval --lane A1 --run-id my_run --repeats 2 --hypothesis "..."
uv run apin autoresearch eval --lane A4 --run-id my_a4 --dims 6,20 --repeats 2 --hypothesis "..."
uv run apin autoresearch eval-s1 --upstream walnutpie --allowed-path src/ --hypothesis "..." --dry-run
uv run apin autoresearch eval-s2 --upstream math --allowed-path stan/math/rev/ --hypothesis "..." --dry-run
uv run apin autoresearch eval-s3 --mode reference
uv run apin track run --track L3 --sampler examples/samplers/hmc.py --dims 6,12
uv run apin track run --track L4 --sampler examples/samplers/ula_wrong.py --dims 6
uv run apin track run --track L5 --sampler examples/samplers/elliptical_rwm_l5.py --dims 6,12
uv run apin score --models public --reps 3 --baseline cmdstan --track L2
# Validate Lane A1 (`adapt_metric`) or Lane A4 (`adapt_transport`) submissions on real Stan posteriors:
uv run apin run logdensity --label a1_cand --opt sampler=autoresearch/lanes/A1/submission.py --opt lane=A1 --models blr,gp_regr --reps 3
uv run apin run logdensity --label a4_cand --opt sampler=autoresearch/lanes/A4/submission.py --opt lane=A4 --models blr,eight_schools_centered --reps 3
```

Autoresearch lanes cover all algorithmic islands (**`A0`** ground-up full sampler, **`A1`** preconditioning/metric, **`A2`** multiscale/funnel dynamics, **`A3`** integrators/momentum/trajectories/PDMPs, **`A4`** warm-start & diffeomorphic nonlinear transport with automatic finite-diff Jacobian verification) plus systems & calibration lanes (**`S1`** upstream sampler patches, **`S2`** gradient/compiler speed with finite-diff + byte-identical draw canary, **`S3`** rewind calibration & agent grading via AUARC). Benchmark tracks cover **`L1`** (cost units), **`L2`** (wall-clock `ESS/s` + machine fingerprint), **`L3`** (Haar rotation + extreme `1e-2..1e2` scale stress), **`L4`** (unadjusted kernels, segregated from L1), **`L5`** (density-only `n_gradient == 0`), and **`S2`** (pure systems). Note: OS-level container sandboxing and a private sealed holdout remain Slice 6.

The first logdensity run of each model builds a BridgeStan library, one to two minutes per model.

posteriordb has no reference for 9 of the models. apin built 8 of them with CmdStan and posteriordb's recipe, and keeps them in git. kronecker_gp has none, so it is not in the benchmark grid. To rebuild one:

```bash
uv run apin reference build --models pilots   # needs CMDSTAN; minutes to an hour per model
```

## Tests

```bash
uv run pytest                                   # fast unit tests, no CmdStan or BridgeStan needed
CMDSTAN=/path/to/cmdstan uv run pytest -m slow  # end to end on real models
```

The slow tests run CmdStan on a few models, and the two example samplers on two models through BridgeStan. The first run builds a BridgeStan library per model, one to two minutes each. The CmdStan test is skipped when `CMDSTAN` is not set.

A smoke run of the command line, about a minute on first build:

```bash
export CMDSTAN=/path/to/cmdstan
uv run apin run cmdstan --models eight_schools_noncentered,blr --reps 1 --label smoke-cmdstan --runs /tmp/apin-smoke/runs
uv run apin run pathfinder --models eight_schools_noncentered,blr --reps 1 --label smoke-pf --runs /tmp/apin-smoke/runs
uv run apin score --runs /tmp/apin-smoke/runs --out /tmp/apin-smoke/scores --models eight_schools_noncentered,blr --reps 1 --baseline smoke-cmdstan
```

## Run against a patched build

Each upstream has a build note in `upstream/<name>.md` with the make or CMake command, the flags, and the hazards that cost time before. In short: check out the branch you want in the submodule, build, point the harness at the result (`--cmdstan`, `BRIDGESTAN` or `WALNUTPIE_CLI`), and give the run its own `--label`. The model cache in `build/` does not see a branch change inside a tree at the same path; delete `build/` or pass `--build` per arm.

## Make a patch

apin has one patch mechanism: a branch on the owner's fork of the upstream, pinned by the submodule.

1. In `upstream/<name>/`, start a branch from the upstream commit you want to change. The submodule's `origin` is the fork.
2. Make the change and build it. Run the upstream's own tests for the files you touched.
3. Measure it as PROTOCOL.md says: pre-register, build a matched baseline, run the bit-identity canary, check gradients against finite differences, then run both arms with their own labels.
4. Write the record in `results/records/`.
5. Push the branch to the fork, move the submodule pin if apin should use it, and add or update the row in `upstream/STATUS.md`, all in one apin commit.
6. Open a pull request on the fork if the change needs a description. The owner decides whether it goes upstream and sends it personally. Agents never post to stan-dev, flatironinstitute or cmdstanr repositories (`upstream/STATUS.md`).

Branches on a fork can be rebased. When a pin is not on any branch, push a tag for it (`apin/<name>-<label>`), or `git submodule update` fails once GitHub drops the commit.

## Add a sampler

Write one file in `harness/adapters/` that defines a class `Adapter`, a subclass of `harness.sampler.Sampler`, with a `name` and a `run(RunRequest) -> Timing` method. `run` writes one CSV per chain in the output directory; PROTOCOL.md, "Runner output schema", gives the format. The grid sets the seeds and budget, writes `run.json` and the markers, and the scorer reads only the CSVs and `run.json`. `apin list` finds the new adapter on its own. A sampler written in Python needs no adapter: the logdensity adapter runs it (PROTOCOL.md, "The log-density seam"). REPORT.md, section 9, describes what the seam still assumes (Stan models, one Stan-specific setting).

## Add an upstream

Add the submodule at `upstream/<name>/` over HTTPS, with its fork as the URL. Write `upstream/<name>.md` with the build commands and hazards. Add a row to the Upstreams table and a section to the patch tables in `upstream/STATUS.md`. Upstreams under consideration go in the Candidates table first.

## Upstreams

| name | why apin carries it |
|---|---|
| math | The autodiff library. Most gradient patches are here. |
| stan | The samplers and their adaptation. |
| stanc3 | The Stan compiler. Patches change the C++ it emits. |
| cmdstan | The command-line driver used by the cmdstan and pathfinder adapters. |
| bridgestan | The model library that walnutpie's `stan_cli` loads, and the way a sampler outside Stan reaches a Stan model. |
| walnutpie | A C++ WALNUTS sampler; the largest share of patches. |
| posteriordb | The source of the core models, the fetched data and reference posteriors. |

`upstream/STATUS.md` has the pins, forks, branches, licences and the state of every patch.

## Past results

`REPORT.md` summarises the 2026 campaign (Phase 0 to W-133) and traces each headline number to a file in `results/`. The records in `results/records/` are the measurements, with dated corrections where a later result changed them. The raw per-cell output and the worklog stayed in the old repository; ARCHIVE.md lists them with their old paths.

REPORT.md, section 1, says how the work was done.

## Licence

BSD-3-Clause (`LICENSE`), copyright Maximilian Scholz. `LICENSING.md` covers the upstreams, the translated ESS code, patches and data.
