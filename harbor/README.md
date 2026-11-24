# apin agent benchmark (Harbor suite)

This is a standardised, containerised benchmark for LLMs and coding agents, built on the [Harbor](https://harborframework.com) task format (`laude-institute/harbor`). Every model or agent gets the same frozen starting code, sealed container, network policy and hidden verifier seeds, so results within a release are comparable.

> Agent benchmark vs. open-ended autoresearch:
> - Agent benchmark (here, `harbor/`): evaluates LLMs and agent harnesses under fixed conditions (one model or agent, frozen starting code, a sealed container that can reach only the model provider). It is ranked by speed-up over the starting code.
> - Open-ended autoresearch (`autoresearch/` and [apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)): evaluates the best posterior approximator produced by any process (multi-LLM councils, custom harnesses, human-in-the-loop), with strict rules only on the final artefact.

## Tasks

The benchmark is five speed-up tasks, one per lane `A0` to `A4`. The agent starts from frozen starting code and makes it faster. The grader (`harness/grade.py`) scores the final file at the standard configuration (dimensions 6 and 20, both budgets `H=150` and `H=1000`, 8 repeats) on hidden seeds. A file that fails an invariance, mixing or replay check is `invalid` and scores `0.0`. A purely statistical rejection counts only after it fails again on a second seed. See [Scoring](#scoring) for the rules.

> How `soft_min_eff` works in A0 to A4: the grader combines per-class geometric means with a $p = -4$ power mean (`(mean_c eff_c^{-4})^{-1/4}`), so the worst target class dominates the score. For example, a small regression on `hybrid_rosenbrock` or `banana` can erase +50% gains on easier classes. Check `by_class` / `CLASS_METRIC` after every edit and improve the weakest class first.

| Task | Lane | Starting code (`autoresearch/lanes/<LANE>/start.py`) | Stan-style reference (`tests/reference.py`) | Frontier (`tests/frontier.py`) |
|---|---|---|---|---|
| `a0-ground-up` | `A0` | A 43-line random-walk Metropolis stub | Stan 2.37 NUTS, diagonal metric (`stan-nuts-diag-2.37`) | A0 incumbent: windowed dense-whitened HMC + RWM |
| `a1-preconditioning` | `A1` | Low-rank Fisher geometric-mean metric (`adapt_metric`) | Stan's warmup, its step size and windowed diagonal variance handed to the judge's HMC (`stan-windowed-diag-metric-2.37`) | A1 incumbent: robust dense covariance metric |
| `a2-funnel-dynamics` | `A2` | Diagonal HMC | Stan 2.37 NUTS (same file as A0) | A2 incumbent: persistent-momentum GHMC + Barker |
| `a3-integrator-trajectory` | `A3` | Diagonal HMC (same code as A2) | Stan 2.37 NUTS (same file as A0) | A3 incumbent: persistent-momentum GHMC |
| `a4-nonlinear-transport` | `A4` | Diagonal affine transport, median/MAD fit | Diagonal affine map fitted by Stan's warmup (`stan-windowed-diag-affine-2.37`) | A4 incumbent: whitened triangular + SAS transport |

## Scoring

One grader scores every sampler file: `harness/grade.py::score_file`. Three callers use it and differ only in the seed.

| Caller | Seeds | Divides by |
|---|---|---|
| The lane loop, `apin autoresearch eval` | public (`PUBLIC_SEED`, `PUBLIC_CONFIRMATION_SEED` in `harness/grade.py`) | the lane's yardstick, `autoresearch/lanes/<LANE>/yardstick.json` |
| The agent's own check inside a benchmark container | public, the same command | the same yardstick, shipped in the image |
| The verifier, `harbor/verify.py`, after the agent has finished | hidden (`HIDDEN_VERIFIER_SEED`, `HIDDEN_CONFIRMATION_SEED` in `harbor/verify.py`) | `tests/frozen_scores.json` |

The speed-up is the candidate's `soft_min_eff` divided by the frozen `soft_min_eff` of the starting code at the same configuration and seeds. It is deterministic for a given file. On the public seeds it is what the agent sees while it works. On the hidden seeds it is the leaderboard number. Tuning a sampler to the public seeds does not carry over.

The verifier scores only the candidate at run time. It divides the candidate's `soft_min_eff` by three denominators, each scored once on the hidden seeds and frozen in `tests/frozen_scores.json`:

| Field | Meaning |
|---|---|
| `reward` = `ref_speedup` | candidate / Stan-style reference. `0` if the candidate fails a check |
| `start_speedup` | candidate / starting code. The leaderboard ranks by this |
| `frontier_speedup` | candidate / the lane's incumbent at freeze time (`null` if the frontier fails a check on the hidden seeds) |
| `gap_closed` | `log(candidate/start) / log(frontier/start)`: 0 = no better than the start, 1 = as good as the frontier. `null` unless the frontier is valid and ahead of the start |
| `outcome` | `invalid` (candidate failed Tier 0, a non-statistical Tier 1 check, or a statistical Tier 1 check on both the verifier seed and the confirmation seed, see [Confirmation](#confirmation-of-statistical-rejections)), `unscorable` (candidate valid, but the starting code or the reference failed a check, so there is no ratio, reward 0, never a substituted 1x), `no_gain` (`start_speedup < 1.05`), `improved` (`start_speedup >= 1.05`, `ref_speedup < 1`), `solved` (`start_speedup >= 1.05`, `ref_speedup >= 1`) |

`reward.json` (numbers only, for Harbor) holds `reward`, `passed`, `solved`, `improved`, `unscorable`, the three speed-ups, `gap_closed` (left out when null), `soft_min_eff`, `start_soft_min`, `reference_soft_min`, `frontier_soft_min`, `short_horizon_soft_min`, `long_horizon_soft_min`, `b2_max`, `overhead_ratio`, `confirmed` (1 when a confirmation run ran and passed), `improvement_margin`, `repeats`, `candidate_wall_s` and the `solution_*` code statistics. `details.json` adds `outcome`, `lane`, `task`, `config`, `judge_digest`, `confirmation` (`ran`, `passed`, `seed`, `failed_targets_first_run`, `failed_targets`, `wall_s`, `efficiency_source`), `reference` (`name`, `file`, `source`: `frozen` or `live`, `valid`), `denominators` (per denominator: `valid`, `source`, `sha256`, soft-mins, `by_class`, `per_repeat_soft_min`, `confirmation`, `error`), the candidate's `by_class` and `per_repeat_soft_min`, and `error` / `failed_targets` / `tier0_info` for an invalid candidate. Soft-min values are rounded to 6 significant digits.

**Frozen scores.** `scripts/freeze_harbor_references.py` writes both kinds in one command. It scores each lane's `start.py` on the public seeds into `autoresearch/lanes/<LANE>/yardstick.json`, copies `autoresearch/lanes/<LANE>/submission.py` to `tests/frontier.py`, and scores `tests/baseline.py`, `tests/reference.py` and `tests/frontier.py` on the hidden seeds into `tests/frozen_scores.json`. Every score is keyed by lane, `judge_digest` (`harness.judge.judge_digest()`), the scoring config and the file's SHA-256. A yardstick also carries an id that changes when its key or its score changes, and every lane trial records that id. The verifier uses a frozen score only when the whole key matches and scores the file live otherwise (`source: "live"` in `details.json`), so a stale freeze only adds run time. The lane loop refuses to grade against a stale yardstick. Refreeze after any change to a module in the judge digest, to a `start.py`, to a reference or to a seed. `tests/test_harbor_scoring.py` fails until you do. The digest covers the modules that can change a score (`logdensity`, `ipc`, `sandbox`, `shapes`, `invariance`, `judge`, `grade`) and leaves out the lane loop's bookkeeping.

**Single source.** `autoresearch/lanes/<LANE>/start.py` is the starting code. `harbor/verify.py` is the verifier. `harbor/references/stan_core.py` plus one lane tail is the reference. `uv run python scripts/sync_harbor_tests.py` regenerates every `environment/baseline.py`, `tests/baseline.py`, `tests/verify.py` and `tests/reference.py` (`--check` reports drift, the scoring tests run it).

**Seeds.** The hidden seeds rotate with every release and are listed with the retired ones in `harbor/verify.py`. They are public once a release ships, because they are in that file. During a run the agent cannot read them: `harbor/` and `tests/` never enter the agent image, `tests/` reaches only the verifier container after the agent phase, and the agent's network is sealed. The public seeds are disjoint from every hidden one.

**Isolation.** `tests/test.sh` runs `/app/.venv/bin/python -I /tests/verify.py` in the separate verifier container. It reads the candidate from the path in `--submission` (the task's `artifacts` entry, e.g. `/app/autoresearch/lanes/A0/submission.py`), copies it into a private temporary directory and scores it from there. The harness and interpreter come from the image. The denominators and frozen scores come from `/tests`.

### Frozen scores of bench-v0.3.0

Soft-min efficiency at the standard configuration (dimensions 6 and 20, 8 repeats). The public yardstick is what `apin autoresearch eval` divides by. The other columns are the verifier's denominators on the hidden seeds.

| Lane | Public yardstick (id) | Starting code, hidden | Stan-style reference | reference / start | Frontier | frontier / start |
|---|---|---|---|---|---|---|
| A0 | 3.901e-4 (`A0-9f4329102727`) | 3.610e-4 | 3.805e-4 | 1.05 | 1.099e-3 | 3.05 |
| A1 | 4.268e-4 (`A1-bef4899b12b6`) | 4.759e-4 | 3.397e-4 | 0.71 | 1.095e-3 | 2.30 |
| A2 | 2.396e-4 (`A2-2d1d93ca843f`) | 2.677e-4 | 3.805e-4 | 1.42 | 3.205e-4 | 1.20 |
| A3 | 2.396e-4 (`A3-39e156326f6f`) | 2.677e-4 | 3.805e-4 | 1.42 | 1.729e-3 | 6.46 |
| A4 | 4.519e-4 (`A4-1798e590a9cc`) | 4.450e-4 | 3.397e-4 | 0.76 | fails a check (funnel d=6 and hybrid Rosenbrock d=6 stuck) | - |

Under this grader, the Stan-style NUTS beats the 43-line RWM stub and the plain diagonal HMC starting code only modestly (1.05x and 1.42x). Stan's warmup with a diagonal metric scores 0.71 to 0.76x of the A1 and A4 starting code, so unchanged starting code in A1 and A4 already has `ref_speedup` 1.31 to 1.40 (still `no_gain`). The A1 and A4 references share the same dynamics (an affine map is a diagonal metric) and score the same to six digits. The A0 starting code gets stuck on hybrid Rosenbrock d=20 on some seeds. It passes on the public and the hidden seeds of this release.

One grader run of a single file took 116 s (A0 starting code) to 430 s (A4 frontier) while four ran in parallel on a shared 24-core machine, and about 1,230 s for the NUTS reference. That is the time of one lane trial and of the verifier's candidate run.

### A/A calibration of repeats and margin

`scripts/calibrate_harbor_noise.py` scored each lane's starting code and 8 copies of it at 8 repeats. Each copy differs only in its random numbers (drawing k extra uniforms at the top of its adapt entry point and, in A0/A2/A3, of `transition()`). The score at R repeats is the geometric mean of the first R per-repeat soft-mins, which matches what the grader computes. Ratio = copy soft-min / starting-code soft-min. A3 has the same code as A2 and is counted once. Results: `harbor/calibration/summary.json` and one JSON per file.

| Repeats | A0 max (p95) | A1 max (p95) | A2 = A3 max (p95) | A4 max (p95) | Pooled p95 | Pooled max | Copies invalid |
|---|---|---|---|---|---|---|---|
| 2 | 0.99 (0.99) | 1.04 (1.02) | 1.39 (1.35) | 1.11 (1.07) | 1.20 | 1.39 | 0 / 32 |
| 4 | 1.03 (1.02) | 1.04 (1.01) | 1.23 (1.21) | 1.08 (1.06) | 1.13 | 1.23 | 4 / 32 |
| 8 | 1.00 (1.00) | 1.03 (1.01) | 1.02 (1.01) | 0.91 (0.89) | 1.01 | 1.03 | 6 / 32 |

The grader uses `REPEATS = 8` and `IMPROVEMENT_MARGIN = 1.05` (`harness/grade.py`), in the lane loop and in the verifier: at 8 repeats no copy exceeded 1.03x, while at 4 repeats a margin above the noise would have to be 1.25 and at 2 repeats 1.40. The copy medians at 8 repeats are 0.94 to 0.97 (A4 is 0.84 because its frozen starting code drew well), so a candidate equal in law to the starting code usually scores below 1.00x. The lane loop's tie band for the simplicity rule (2%, `TIE_BAND` in `harness/autoresearch.py`) sits inside this spread: two files within 2% of each other are indistinguishable, so the simpler one is kept.

The cost is the invalid column. The Tier 1 checks reject valid samplers by chance (each target check has a small false-rejection rate and the verifier runs 13 targets per repeat), so more repeats mean more checks. Of the 6 invalid copies at 8 repeats, 4 are chance rejections (|z| just above the threshold on Student-t, banana and mixture targets) and 2 are the A0 RWM stub getting stuck on hybrid Rosenbrock d=20. A single pass therefore marks about one valid submission in eight `invalid` (4/32, across five optimize tasks that is a false `invalid` in about half of all sweeps), which is why the verifier confirms statistical rejections.

### Confirmation of statistical rejections

When the candidate passes Tier 0 and every Tier 1 failure is a statistical rejection (`invariance rejected on …` or `rank test rejected on …`), the grader runs Tier 1 once more with the same configuration on the configuration's confirmation seed (`HIDDEN_CONFIRMATION_SEED` in the verifier, `PUBLIC_CONFIRMATION_SEED` in the lane loop). Its suites are disjoint from the primary seed's. The candidate is `invalid` only if that run fails too, on any check. Tier 0 failures (lint, replay and the invariance smoke test), non-statistical Tier 1 failures (stuck kernel, adapt-cost cap, exceptions and timeouts in `adapt()` or `transition()`) and a missing file are `invalid` at once.

After a confirmed pass, the grader keeps the first run's efficiency numbers. The judge reports every target's efficiency even when a check fails (zeroing only the suite's soft-min), so the grader rebuilds the soft-min from those target efficiencies with the judge's formulas (`scores_from_targets`, checked against the judge on every passing run). Those numbers stay paired with the frozen denominators (`efficiency_source: first_run_paired`). If a target reported no positive efficiency in the first run, there is nothing paired to report: the result is `unscorable` (`efficiency_source: none`), because a score from the confirmation seed cannot be divided by denominators scored on the verifier seed. The same rule applies to a denominator scored live or frozen.

| | Single pass | With confirmation |
|---|---|---|
| A/A copies `invalid` at 8 repeats (32 copies) | 6 (4 chance rejections, 2 stuck A0 stubs) | 3 (all the A0 RWM stub getting stuck: 2 at once, 1 in its confirmation run) |
| Chance rejections still `invalid` | 4 of 4 | 0 of 4 on their own account. The 4th copy then got stuck |
| False `invalid` rate of a valid sampler that does not get stuck | ≈ 12.5% per file (4/32), ≈ 49% chance of one per 5-task sweep | ≈ 1.6% per file (two independent rejections, 0.125², 8% at the 95% upper bound), ≈ 8% per sweep |

Data: `harbor/calibration/confirmation_rerun.json`. Confirmation adds one Tier 1 pass of the candidate and runs only after a statistical rejection: 195 to 420 s for the copies here and 420 s for the A1 frontier on this loaded machine (about the candidate-only time minus Tier 0). A sampler that gets stuck on some seeds, like the A0 RWM stub on hybrid Rosenbrock d=20, is still marked `invalid` because getting stuck is a real defect at that seed.

### How the reference compares with CmdStan

`scripts/compare_stan_reference.py` samples an ill-conditioned diagonal Gaussian (D=10, standard deviations 0.1 to 10) and Neal's funnel (D=10), written as Stan programs, with CmdStan 2.37.0 (on m0pad) and with the reference. Each run uses 64 chains × 1000 draws under three warmup settings: 100 warmup draws with the windows CmdStan prints for that budget passed explicitly (`windows`), plain `num_warmup=100` (`default`, where CmdStan never updates the metric) and 1000 warmup draws (`long`). Data: `harbor/calibration/cmdstan_comparison.json`.

| Target | Warmup | Sampler | Step size | Mean tree depth | Gradients / draw | ESS / gradient |
|---|---|---|---|---|---|---|
| Gaussian | windows | CmdStan (seed 2) | 0.659 (0.653) | 2.83 (2.83) | 6.7 (6.6) | 0.236 (0.254) |
| Gaussian | windows | reference | 0.635 | 2.87 | 6.8 | 0.205 |
| Gaussian | default | CmdStan (seed 2) | 0.114 (0.114) | 6.28 (6.26) | 132.1 (131.3) | 1.49e-3 (1.65e-3) |
| Gaussian | default | reference | 0.113 | 6.28 | 132.5 | 1.56e-3 |
| Gaussian | long | CmdStan (seed 2) | 0.710 (0.714) | 2.82 (2.81) | 6.5 (6.4) | 0.295 (0.293) |
| Gaussian | long | reference | 0.699 | 2.82 | 6.5 | 0.291 |
| Funnel | windows | CmdStan (seed 2) | 0.185 (0.161) | 4.09 (4.17) | 48.9 (55.3) | 1.38e-4 (1.04e-4) |
| Funnel | windows | reference | 0.170 | 3.99 | 45.2 | 9.6e-5 |
| Funnel | default | CmdStan (seed 2) | 0.364 (0.368) | 3.95 (3.96) | 38.2 (38.4) | 1.63e-4 (1.62e-4) |
| Funnel | default | reference | 0.371 | 3.99 | 39.3 | 2.92e-4 |
| Funnel | long | CmdStan (seed 2) | 0.114 (0.113) | 3.71 (3.84) | 30.7 (37.1) | 2.08e-4 (1.52e-4) |
| Funnel | long | reference | 0.121 | 3.81 | 33.9 | 1.69e-4 |

Step size, tree depth and gradients per draw agree within 4% on the Gaussian and within the spread of CmdStan's two seeds on the funnel. ESS per gradient on the Gaussian lies within CmdStan's seed spread with the default warmup and within 1.4% with the long one. With the 100-draw windows the reference is 13 to 19% below CmdStan's two seeds (which differ from each other by 8%). Funnel ESS (a few hundred effective draws from 64,000) varies by up to 1.8x between runs of either sampler. ESS per gradient is the min over parameters of bulk ESS divided by all sampling gradients.

**What is and is not Stan** (full list in the docstring of `harbor/references/stan_core.py`). Identical: multinomial NUTS with the generalised no-U-turn criterion and Stan's extra subtree checks, biased progressive sampling, max depth 10, divergence at ΔH > 1000, diagonal metric, `init_stepsize`, dual averaging (δ 0.8, γ 0.05, κ 0.75, t0 10), Welford variance with Stan's regularisation, window doubling, the 15%/75%/10% rescaling for short warmup, uniform(-2, 2) inits, jitter 0. Different: (1) CmdStan 2.37 and 2.39 never apply the rescaled short-warmup windows (`windowed_adaptation::set_window_params` returns before `restart()`), so real CmdStan at 100 warmup draws keeps a unit metric. The reference uses the windows Stan prints, which is what CmdStan does when they are passed explicitly (`SHORT_WARMUP_RESTART = False` reproduces the bug, the `default` rows above). (2) `adapt()` reuses the gradient it holds instead of recomputing it each warmup draw. (3) Uses `numpy` random numbers. (4) One chain per target. (5) In A1 and A4 only the warmup is Stan's. The judge's fixed-path HMC kernel samples.

## Running with Harbor

Install the Harbor CLI once (0.27.0 or later, it has the per-phase network policies the tasks use), build the images from repo root, then run any task or the full suite:

```bash
# 1. Install Harbor CLI (requires Docker daemon for containerized runs)
uv tool install harbor==0.27.0

# 2. Build the sealed base image apin-harbor-base:bench-v0.3.0 (--tasks also builds the 5 task images as a check)
uv run python scripts/build_harbor_base.py

# 3. Run one task or the full suite via Harbor CLI (e.g. OpenCode V2 with effort variant)
eval "$(grep -E '^export ZAI_API_KEY=' ~/.bashrc)"   # or export ZAI_API_KEY yourself
harbor run -p harbor/tasks/a1-preconditioning -a opencode-v2 -m zai-coding-plan/glm-5.3-flash \
  --ak reasoning_effort=high --api-key-name ZAI_API_KEY --allow-agent-host api.z.ai
harbor run -p harbor/tasks -a claude-code -m anthropic/claude-sonnet-4-5
```

### What the agent container holds

`scripts/build_harbor_base.py` stages a build context from an allowlist and builds the base image from it: `harness/` (with the grader and its public seeds), `models/`, `data/`, `codex/`, `autoresearch/RULINGS.md`, and of lanes A0 to A4 the files `program.md`, `NEGATIVE_MEMORY.md`, `start.py` and `yardstick.json`, plus `pyproject.toml`, `uv.lock` and `LICENSE`. Every lane's `submission.py` is a copy of its `start.py`, so `apin autoresearch eval` works offline and reports a speed-up of 1.0 for the unchanged file. Each task image then copies its own starting code. Evolved submissions, `examples/`, `tests/`, `results/`, `scripts/`, `harbor/` (the verifier, its hidden seeds and every `frozen_scores.json`), `autoresearch/lanes/S3` (fixtures and reference fixes), `runs/` and `.git` are not in the image. The image includes a complete virtual environment with `UV_OFFLINE=1` and `UV_NO_SYNC=1` set, so `uv run` never syncs or downloads. `tests/test_harbor_seal.py` checks the staged context and the task files: the public yardsticks are there, and no staged file contains a hidden seed, a `frozen_scores.json` or the verifier. `-m slow` also inspects built task images.

### Time limits and network

| Phase | Time limit | Network |
|---|---|---|
| Agent setup (Harbor installs the agent) | Harbor default | public (`[environment] network_mode`) |
| Agent | 86400 s (24 h) for all tasks | `allowlist`: only the model provider's API host |
| Verifier (separate container) | 7200 s | `no-network` |

Harbor enforces the allowlist with its Docker egress-control sidecar, which filters by hostname (TLS SNI) and connects to the named host itself, so a forged SNI or `Host` header does not reach another site. Harbor adds the inference host it can infer from `-m`. For `zai-coding-plan/...` it infers neither the host nor the key, so pass `--allow-agent-host api.z.ai` and `--api-key-name ZAI_API_KEY` as above. Without them the agent cannot reach the model. An explicit `--allow-agent-host` replaces the inferred host, so list every host the agent needs. Because the agent phase is restricted, Harbor also turns off the agents' own web tools (`disable_web_search`): for `opencode-v2` it writes deny policies for `websearch` and `webfetch` into `opencode.json` (and refuses OpenCode older than 2.0.29). For `claude-code` it passes `--disallowedTools WebSearch,WebFetch`. Those are the tools a host allowlist cannot stop, because the provider runs the search on its side.

The verifier runs in a separate container built from the same task image (`[verifier] environment_mode = "separate"`). Harbor carries over only the task's `artifacts` entry, the submission file, to the same path, so changes the agent makes to `harness/`, `.venv/` or anything else in its container never reach the verifier. A Docker host without nftables support cannot change policy between phases. Harbor then fails the trial instead of running it with an open network.

## Running the verifier locally (no Docker or Harbor CLI required)

Every `tests/test.sh` calls [harbor/verify.py](verify.py), which you can run directly on your host machine. `--out-dir` defaults to `./verifier_out` outside Docker (gitignored). A run scores only your candidate, on the hidden seeds at the standard configuration. It reads the denominators from `harbor/tasks/<task>/tests/frozen_scores.json` unless their key no longer matches, in which case it scores them live too.

```bash
# The starting code itself: outcome no_gain, start_speedup 1.0
uv run python harbor/verify.py optimize --lane A1 \
  --submission autoresearch/lanes/A1/start.py --out-dir /tmp/harbor-opt-check
```

Compare its `start_speedup` with the speed-up `apin autoresearch eval` printed for the same file. The two use different seeds, so they are close but not equal.

The verifier writes Harbor-standard `reward.txt` (scalar reward), `reward.json` (numbers only) and `details.json` (full breakdown), with the fields listed under [Scoring](#scoring), and archives `submission.py`.

## Publishing results to the leaderboard

The leaderboard is a page of the project website: [apin-flywheel.m0hawk.workers.dev/benchmark/](https://apin-flywheel.m0hawk.workers.dev/benchmark/). It reads one data file, `flywheel/public/data/benchmark.json`. Collect Harbor job directories into that file, then deploy the site:

```bash
uv run python scripts/collect_harbor_results.py --release bench-v0.3.0 --jobs jobs/
cd flywheel && npx wrangler deploy
```

The collector reads each task's title, lane, difficulty and time limit from its `task.toml`, so the page never restates them by hand. It records any run cut off by the model provider or infrastructure as `interrupted`, whatever the run produced, and any valid result the verifier could not score as `unscorable`. Neither counts as a result. The leaderboard covers the five speed-up tasks. Each row is ranked by the geometric mean of its speed-ups over the starting code, and a task with no valid attempt enters that mean at 0.1×. Attempts at bug-fix tasks of earlier releases stay in the data file as a record and appear in no row. Preview the page locally with `python3 -m http.server 8788 --directory flywheel/public` and open `http://localhost:8788/benchmark/`.
