# apin Agent Benchmark (Harbor Suite)

Standardized, containerized **LLM & coding-agent benchmark** built on the [Harbor](https://harborframework.com) task format (`laude-institute/harbor`). Every model/agent gets the exact same frozen starter code, sealed container environment, network policy, and hidden verification seeds so results are fair and apples-to-apples across releases.

> **Agent Benchmark vs. Open-Ended Autoresearch:**
> - **Agent Benchmark (here, `harbor/`):** Evaluates **LLMs & agent harnesses** under strict, fixed conditions (single model/agent, frozen baseline, no internet → pass rate + speedup over baseline).
> - **Open-Ended Autoresearch (`autoresearch/` + [apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)):** Evaluates the **best posterior approximator** ever produced — anything goes on process (multi-LLM councils, custom harnesses, human-in-the-loop), strict rules only on the final artifact.

---

## Task Categories

### Category 1 — Diagnostic & Invariance Rewinds (Binary Pass/Fail)
Tasks derived from real MCMC / transport bugs discovered in `apin` campaigns. Base fails the hidden invariance/Jacobian check; agent must diagnose and fix it without cheating.

| Task | Difficulty | Description | Metric |
|---|---|---|---|
| `s3-position-gated-mh` | easy | Fix position-dependent kernel gating breaking MH detailed balance | `reward ∈ {0, 1}` |
| `s3-missing-jacobian-a4` | easy | Fix missing `log_abs_det_jac` & `pullback_grad` in Lane A4 sinh-arcsinh transport | `reward ∈ {0, 1}` |
| `s3-unadjusted-ula` | easy | Convert biased unadjusted Langevin (ULA) into an exact Metropolis-adjusted kernel | `reward ∈ {0, 1}` |

### Category 2 — Open-Ended Optimization (Continuous Speedup)
KernelBench-style optimization tasks. The agent starts from frozen starting code and maximizes dual-horizon (`H=150/1000`, `dims=6,20`) verified efficiency on **hidden verifier seeds**. Any invariance, mixing or replay failure makes the run `invalid` and scores **`0.0`**. Scoring is described in [Scoring (optimize tasks)](#scoring-optimize-tasks).

> **How `soft_min_eff` works (critical for A0–A4):** Per-class geometric means are aggregated with a **$p = -4$ power mean** (`(mean_c eff_c^{-4})^{-1/4}`), so **the worst target class dominates the score** (e.g., a small regression on `hybrid_rosenbrock` or `banana` can erase +50% gains on easier classes). Check `by_class` / `CLASS_METRIC` after every edit and improve the weakest class first.

| Task | Lane | Starting code (`tests/baseline.py`) | Stan-style reference (`tests/reference.py`) | Frontier (`tests/frontier.py`) |
|---|---|---|---|---|
| `a0-ground-up` | `A0` | 43-LOC isotropic RWM stub | Stan 2.37 NUTS, diagonal metric (`stan-nuts-diag-2.37`) | A0 incumbent: windowed dense-whitened HMC + RWM |
| `a1-preconditioning` | `A1` | Low-rank Fisher geometric-mean metric (`adapt_metric`) | Stan's warmup, its step size and windowed diagonal variance handed to the judge's HMC (`stan-windowed-diag-metric-2.37`) | A1 incumbent: robust dense covariance metric |
| `a2-funnel-dynamics` | `A2` | Diagonal HMC starter | Stan 2.37 NUTS (same file as A0) | A2 incumbent: persistent-momentum GHMC + Barker |
| `a3-integrator-trajectory` | `A3` | Diagonal HMC starter (same code as A2) | Stan 2.37 NUTS (same file as A0) | A3 incumbent: persistent-momentum GHMC |
| `a4-nonlinear-transport` | `A4` | Diagonal affine transport, median/MAD fit | Diagonal affine map fitted by Stan's warmup (`stan-windowed-diag-affine-2.37`) | A4 incumbent: whitened triangular + SAS transport |

## Scoring (optimize tasks)

The verifier scores only the candidate at run time. It divides the candidate's `soft_min_eff` by three **denominators**, each scored once on the same verifier seeds and frozen in `tests/frozen_scores.json`:

| Field | Meaning |
|---|---|
| `reward` = `ref_speedup` | candidate / Stan-style reference; `0` if the candidate fails a check |
| `start_speedup` | candidate / starting code |
| `frontier_speedup` | candidate / the lane's incumbent at freeze time (`null` if the frontier fails a check on the verifier seeds) |
| `gap_closed` | `log(candidate/start) / log(frontier/start)`: 0 = no better than the start, 1 = as good as the frontier; `null` unless the frontier is valid and ahead of the start |
| `outcome` | `invalid` (candidate failed Tier 0, a non-statistical Tier 1 check, or a statistical Tier 1 check on both the verifier seed and the confirmation seed; see [Confirmation](#confirmation-of-statistical-rejections)), `unscorable` (candidate valid, but the starting code or the reference failed a check, so there is no ratio; reward 0, never a substituted 1x), `no_gain` (`start_speedup < 1.05`), `improved` (`start_speedup >= 1.05`, `ref_speedup < 1`), `solved` (`start_speedup >= 1.05`, `ref_speedup >= 1`) |

`reward.json` (numbers only, for Harbor) holds `reward`, `passed`, `solved`, `improved`, `unscorable`, the three speed-ups, `gap_closed` (left out when null), `soft_min_eff`, `start_soft_min`, `reference_soft_min`, `frontier_soft_min`, `short_horizon_soft_min`, `long_horizon_soft_min`, `b2_max`, `overhead_ratio`, `confirmed` (1 when a confirmation run ran and passed), `improvement_margin`, `repeats`, `candidate_wall_s` and the `solution_*` code statistics. `details.json` adds `outcome`, `lane`, `task`, `config`, `judge_digest`, `confirmation` (`ran`, `passed`, `seed`, `failed_targets_first_run`, `failed_targets`, `wall_s`, `efficiency_source`), `reference` (`name`, `file`, `source`: `frozen` or `live`, `valid`), `denominators` (per denominator: `valid`, `source`, `sha256`, soft-mins, `by_class`, `per_repeat_soft_min`, `confirmation`, `error`), the candidate's `by_class` and `per_repeat_soft_min`, and `error` / `failed_targets` / `tier0_info` for an invalid candidate. Soft-min values are rounded to 6 significant digits. Rewind tasks report `outcome` `solved` or `invalid` and a numeric `solved`.

**Frozen denominators.** `scripts/freeze_harbor_references.py` copies `autoresearch/lanes/<LANE>/submission.py` to `tests/frontier.py`, scores the three files and writes `tests/frozen_scores.json`, keyed by lane, `judge_digest` (`harness.judge.judge_digest()`), the scoring config (seed, dims, repeats, chains, steps, budgets) and each file's SHA-256. The verifier uses a frozen score only when the whole key matches and scores the file live otherwise (`source: "live"` in `details.json`), so a stale freeze costs time, not correctness. Rerun the freeze after any change to `harness/`, a denominator file or a scoring constant; `tests/test_harbor_scoring.py` fails until you do. Scoring is deterministic given the seeds: the freeze reproduced the calibration run's scores bit for bit, and the verifier run on each lane's own starting code gives `start_speedup` exactly 1.0.

**Single source.** `harbor/verify.py` is the verifier; `harbor/references/stan_core.py` plus one lane tail is the reference. `uv run python scripts/sync_harbor_tests.py` regenerates every `tests/verify.py` and `tests/reference.py` (`--check` reports drift; the scoring tests run it).

**Verifier seed.** `HIDDEN_VERIFIER_SEED` in `harbor/verify.py` is rotated every release (bench-v0.2.0 retired `88421077`). It is public once a release ships, because it is in this file. During a run the agent cannot read it: `tests/` reaches only the verifier container, after the agent phase, and the agent's network is sealed.

**Isolation.** `tests/test.sh` runs `/app/.venv/bin/python -I /tests/verify.py` in the separate verifier container. The candidate is read from the path in `--submission` (the task's `artifacts` entry, e.g. `/app/autoresearch/lanes/A0/submission.py`), copied into a private temporary directory and scored from there; the harness and interpreter come from the image, the denominators and frozen scores from `/tests`.

### Denominators on the verifier seeds (bench-v0.2.x, repeats 8)

| Lane | Starting code | Stan-style reference | reference / start | Frontier | frontier / start |
|---|---|---|---|---|---|
| A0 | 3.503e-4 | 3.840e-4 | 1.10 | 1.094e-3 | 3.12 |
| A1 | 4.623e-4 | 3.175e-4 | 0.69 | 8.739e-4 (valid after confirmation: a chance rejection on banana d=20, \|z\| 3.94 > 3.82, not repeated on the confirmation seed) | 1.89 |
| A2 | 2.913e-4 | 3.840e-4 | 1.32 | 2.748e-4 | 0.94 (behind the start: `gap_closed` null) |
| A3 | 2.913e-4 | 3.840e-4 | 1.32 | 1.130e-3 | 3.88 |
| A4 | 4.666e-4 | 3.175e-4 | 0.68 | fails a check (hybrid Rosenbrock d=6 stuck, repeats 3 and 8) | — |

Under this judge the Stan-style NUTS beats the 43-line RWM stub and the plain diagonal HMC starters only modestly (1.10x, 1.32x), and Stan's warmup with a diagonal metric is 0.68–0.69x of the A1 and A4 starting code, so in A1 and A4 unchanged starting code already has `ref_speedup` 1.46–1.47 (still `no_gain`). The A1 and A4 references are the same dynamics (an affine map is a diagonal metric) and score the same to five digits. Candidate-only wall time of the verifier, five lanes in parallel on a loaded 24-core machine: A0 173 s, A1 374 s, A2 421 s, A3 425 s, A4 396 s. Scoring one denominator live adds about the same per file (the NUTS reference about 1000 s).

### A/A calibration of repeats and margin

`scripts/calibrate_harbor_noise.py` scored each lane's starting code and 8 copies of it that differ only in their random numbers (each draws k extra uniforms at the top of its adapt entry point and, in A0/A2/A3, of `transition()`), at repeats 8; the score at R repeats is the geometric mean of the first R per-repeat soft-mins, exactly what the verifier computes. Ratio = copy soft-min / starting-code soft-min. A3 has the same code as A2 and is counted once. Results: `harbor/calibration/summary.json` and one JSON per file.

| Repeats | A0 max (p95) | A1 max (p95) | A2 = A3 max (p95) | A4 max (p95) | Pooled p95 | Pooled max | Copies invalid |
|---|---|---|---|---|---|---|---|
| 2 | 0.99 (0.99) | 1.04 (1.02) | 1.39 (1.35) | 1.11 (1.07) | 1.20 | 1.39 | 0 / 32 |
| 4 | 1.03 (1.02) | 1.04 (1.01) | 1.23 (1.21) | 1.08 (1.06) | 1.13 | 1.23 | 4 / 32 |
| 8 | 1.00 (1.00) | 1.03 (1.01) | 1.02 (1.01) | 0.91 (0.89) | 1.01 | 1.03 | 6 / 32 |

The verifier uses **`VERIFIER_REPEATS = 8`** and **`IMPROVEMENT_MARGIN = 1.05`**: at 8 repeats no copy reached 1.03x, while at 4 repeats a margin above the noise would have to be 1.25 and at 2 repeats 1.40. The copy medians at 8 repeats are 0.94–0.97 (A4: 0.84, its frozen start drew well), so a candidate equal in law to the starting code usually scores below 1.00x. The cost is the invalid column: the Tier 1 checks reject correct samplers by chance (each target check has a small false-rejection rate and the verifier runs 13 targets per repeat), and more repeats mean more checks. Of the 6 invalid copies at 8 repeats, 4 are such chance rejections (\|z\| just above the threshold on Student-t, banana and mixture targets) and 2 are the A0 RWM stub getting stuck on hybrid Rosenbrock d=20. A single pass therefore marks about one correct submission in eight `invalid` (4/32; with five optimize tasks that is a false `invalid` in about half of all sweeps), which is why the verifier confirms statistical rejections.

### Confirmation of statistical rejections

When the candidate passes Tier 0 and every Tier 1 failure is a statistical rejection (`invariance rejected on …` or `rank test rejected on …`), the verifier runs Tier 1 once more with the same configuration on `CONFIRMATION_SEED` (`78210474`, disjoint from the verifier seed's suites, rotated with it). The candidate is `invalid` only if that run fails too (any check). Tier 0 failures (lint, replay, the invariance smoke test), non-statistical Tier 1 failures (stuck kernel, adapt-cost cap, exceptions and timeouts in `adapt()` or `transition()`) and a missing file are `invalid` at once. After a confirmed pass the efficiency numbers are the first run's: the judge reports every target's efficiency even when a check fails (it only zeroes the suite's soft-min), so the verifier rebuilds the soft-min from those with the judge's formulas (`scores_from_targets`, checked against the judge on every passing run). They stay paired with the frozen denominators (`efficiency_source: first_run_paired`); only if a target reported no positive efficiency would it use the confirmation run's numbers and say `confirmation_run_unpaired`. The same rule applies to a denominator scored live or frozen.

| | Single pass (before) | With confirmation (now) |
|---|---|---|
| A/A copies `invalid` at 8 repeats (32 copies) | 6 (4 chance rejections, 2 stuck A0 stubs) | 3 (all the A0 RWM stub getting stuck: 2 at once, 1 in its confirmation run) |
| Chance rejections still `invalid` | 4 of 4 | 0 of 4 on their own account; the 4th copy then got stuck |
| False `invalid` rate of a correct sampler that does not get stuck | ≈ 12.5% per file (4/32), ≈ 49% chance of one per 5-task sweep | ≈ 1.6% per file (two independent rejections, 0.125²; 8% at the 95% upper bound), ≈ 8% per sweep |

Data: `harbor/calibration/confirmation_rerun.json`. A confirmation costs one more Tier 1 pass of the candidate, only after a statistical rejection: 195–420 s for the copies here and 420 s for the A1 frontier on this loaded machine (about the candidate-only time minus Tier 0). A sampler that gets stuck on some seeds, like the A0 RWM stub on hybrid Rosenbrock d=20, is still marked `invalid`: that is a real defect at that seed, not a chance rejection.

**Rewind tasks get no confirmation.** Their hidden check is one target on the fixed seed in `tests/tasks.json`, so it is deterministic for a given file. Scoring each reference fix on 20 other seeds passed 60 of 60 (and each buggy starting file failed 5 of 5); with one target per check instead of the 104 of an optimize file, the expected false-rejection rate is about 0.2%.

### How the reference compares with CmdStan

`scripts/compare_stan_reference.py` samples an ill-conditioned diagonal Gaussian (D=10, standard deviations 0.1 to 10) and Neal's funnel (D=10) written as Stan programs, with CmdStan 2.37.0 (on m0pad) and with the reference, 64 chains × 1000 draws each, after 100 warmup draws with the windows CmdStan prints for that budget passed explicitly (`windows`), after plain `num_warmup=100` (`default`, where CmdStan never updates the metric), and after 1000 warmup draws (`long`). Data: `harbor/calibration/cmdstan_comparison.json`.

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

Step size, tree depth and gradients per draw agree within 4% on the Gaussian and within the spread of CmdStan's two seeds on the funnel. ESS per gradient on the Gaussian lies within CmdStan's seed spread with the default warmup and within 1.4% with the long one; with the 100-draw windows the reference is 13–19% below CmdStan's two seeds (which differ from each other by 8%). Funnel ESS (a few hundred effective draws from 64,000) varies by up to 1.8x between runs of either sampler. ESS per gradient is the min over parameters of bulk ESS divided by all sampling gradients.

**What is and is not Stan** (full list in the docstring of `harbor/references/stan_core.py`). Identical: multinomial NUTS with the generalized no-U-turn criterion and Stan's extra subtree checks, biased progressive sampling, max depth 10, divergence at ΔH > 1000, diagonal metric, `init_stepsize`, dual averaging (δ 0.8, γ 0.05, κ 0.75, t0 10), Welford variance with Stan's regularization, window doubling, the 15%/75%/10% rescaling for short warmup, uniform(-2, 2) inits, jitter 0. Different: (1) CmdStan 2.37 and 2.39 never apply the rescaled short-warmup windows (`windowed_adaptation::set_window_params` returns before `restart()`), so real CmdStan at 100 warmup draws keeps a unit metric; the reference uses the windows Stan prints, which is what CmdStan does when they are passed explicitly (`SHORT_WARMUP_RESTART = False` reproduces the bug; the `default` rows above). (2) adapt() reuses the gradient it holds instead of recomputing it each warmup draw. (3) numpy random numbers, not Stan's. (4) One chain per target. (5) In A1 and A4 only the warmup is Stan's; the judge's fixed-path HMC kernel samples.

---
|---|---|---|
| `a0-ground-up` | `A0` | 43-LOC isotropic RWM stub | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a1-preconditioning` | `A1` | Low-rank Fisher geometric-mean metric (`adapt_metric`, 211 LOC) | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a2-funnel-dynamics` | `A2` | Diagonal HMC starter | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a3-integrator-trajectory` | `A3` | Diagonal HMC starter | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a4-nonlinear-transport` | `A4` | Diagonal affine transport starter | `soft_min_cand / soft_min_base` (0 if invariance/FD fails) |

---

## Running with Harbor

Install the Harbor CLI once (0.27.0 or later; it has the per-phase network policies the tasks use), build the images from repo root, then run any task or the full suite:

```bash
# 1. Install Harbor CLI (requires Docker daemon for containerized runs)
uv tool install harbor==0.27.0

# 2. Build the sealed base image apin-harbor-base:bench-v0.2.0 (--tasks also builds the 8 task images as a check)
uv run python scripts/build_harbor_base.py

# 3. Run one task or the full suite via Harbor CLI (e.g. OpenCode V2 with effort variant)
eval "$(grep -E '^export ZAI_API_KEY=' ~/.bashrc)"   # or export ZAI_API_KEY yourself
harbor run -p harbor/tasks/a1-preconditioning -a opencode-v2 -m zai-coding-plan/glm-5.3-flash \
  --ak reasoning_effort=high --api-key-name ZAI_API_KEY --allow-agent-host api.z.ai
harbor run -p harbor/tasks -a claude-code -m anthropic/claude-sonnet-4-5
```

### What the agent container holds

`scripts/build_harbor_base.py` stages a build context from an allowlist and builds the base image from it: `harness/`, `models/`, `data/`, `codex/`, `autoresearch/RULINGS.md`, `program.md` and `NEGATIVE_MEMORY.md` of lanes A0–A4, `pyproject.toml`, `uv.lock` and `LICENSE`. Every lane's `submission.py` is that lane's frozen starting code from `harbor/tasks/<task>/environment/baseline.py`; each task image then copies its own starting file. Evolved submissions, `examples/`, `tests/`, `results/`, `scripts/`, `harbor/` (the verifier), `autoresearch/lanes/S3` (hidden fixtures and reference fixes), `runs/` and `.git` are not in the image. The virtual environment is complete in the image and `UV_OFFLINE=1`, `UV_NO_SYNC=1` are set, so `uv run` never syncs or downloads. `tests/test_harbor_seal.py` checks the staged context and the task files; `-m slow` also inspects built task images.

### Time limits and network

| Phase | Time limit | Network |
|---|---|---|
| Agent setup (Harbor installs the agent) | Harbor default | public (`[environment] network_mode`) |
| Agent | 86400 s (24 h) for all tasks | `allowlist`: only the model provider's API host |
| Verifier (separate container) | 7200 s (optimize), 1800 s (rewind) | `no-network` |

Harbor enforces the allowlist with its Docker egress-control sidecar, which filters by hostname (TLS SNI) and connects to the named host itself, so a forged SNI or `Host` header does not reach another site. Harbor adds the inference host it can infer from `-m`. For `zai-coding-plan/...` it infers neither the host nor the key, so pass `--allow-agent-host api.z.ai` and `--api-key-name ZAI_API_KEY` as above; without them the agent cannot reach the model. An explicit `--allow-agent-host` replaces the inferred host, so list every host the agent needs. Because the agent phase is restricted, Harbor also turns off the agents' own web tools (`disable_web_search`): for `opencode-v2` it writes deny policies for `websearch` and `webfetch` into `opencode.json` (and refuses OpenCode older than 2.0.29), for `claude-code` it passes `--disallowedTools WebSearch,WebFetch`. Those are the tools a host allowlist cannot stop, because the provider runs the search on its side.

The verifier runs in a separate container built from the same task image (`[verifier] environment_mode = "separate"`). Harbor carries over only the task's `artifacts` entry, the submission file, to the same path, so changes the agent makes to `harness/`, `.venv/` or anything else in its container never reach the grader. A Docker host without nftables support cannot change policy between phases; Harbor then fails the trial instead of running it with open network.

## Running the Verifier Locally (No Docker or Harbor CLI Required)

Every `tests/test.sh` delegates to [harbor/verify.py](verify.py), which you can invoke directly on your host machine. Both hyphenated task slugs (e.g. `s3-position-gated-mh`) and internal IDs (`s3_position_gated_mh_bias`) are accepted, and `--out-dir` defaults to `./verifier_out` outside Docker (gitignored). A `verify.py optimize` run scores only your candidate (`dims=6,20 × repeats=8`, about 3–7 min); the denominators come from `harbor/tasks/<task>/tests/frozen_scores.json` unless their key no longer matches, in which case they are scored live too.

```bash
# Rewind task verification (reference fix -> reward 1.0)
uv run python harbor/verify.py rewind \
  --task-id s3-position-gated-mh \
  --submission autoresearch/lanes/S3/fixtures/position_gated_ref.py \
  --out-dir /tmp/harbor-check

# Optimization task verification (the starting code itself -> outcome no_gain, start_speedup 1.0)
uv run python harbor/verify.py optimize \
  --lane A1 \
  --submission harbor/tasks/a1-preconditioning/environment/baseline.py \
  --out-dir /tmp/harbor-opt-check
```

Outputs Harbor-standard `reward.txt` (scalar reward), `reward.json` (numbers only) and `details.json` (full breakdown), with the fields listed under [Scoring (optimize tasks)](#scoring-optimize-tasks), and archives `submission.py`.

## Publishing Results to the Leaderboard

The leaderboard is a page of the project website: [apin-flywheel.m0hawk.workers.dev/benchmark/](https://apin-flywheel.m0hawk.workers.dev/benchmark/). It reads one data file, `flywheel/public/data/benchmark.json`. Collect Harbor job directories into that file, then deploy the site:

```bash
uv run python scripts/collect_harbor_results.py --release bench-v0.2.0 --jobs jobs/
cd flywheel && npx wrangler deploy
```

The collector reads each task's title, lane, difficulty and time limit from its `task.toml`, so the page never restates them by hand. A run that the model provider or the infrastructure cut off is recorded as `interrupted`, whatever it had produced, and a valid result the verifier could not score is recorded as `unscorable`. Neither is counted as a pass or a fail. Preview the page locally with `python3 -m http.server 8788 --directory flywheel/public` and open `http://localhost:8788/benchmark/`.
