# Contributing to apin

There are three ways to contribute. All three start with the same setup, which requires only [uv](https://docs.astral.sh/uv/) (it installs the right Python for you):

```bash
git clone https://github.com/Numenomics/apin.git && cd apin
uv sync && uv run apin fetch-data
```

| You want to | Start with | Needs |
|---|---|---|
| Join the autoresearch (improve a sampler in one of the lanes) | [Let an agent do it](#let-an-agent-do-it) or [do it by hand](#do-it-by-hand) | Nothing more |
| Benchmark a model or coding agent on the five tasks | [harbor/README.md](harbor/README.md) | Docker and the Harbor CLI |
| Score your own sampler on real Stan posteriors | [Score your own sampler](#score-your-own-sampler) | CmdStan only if you want the CmdStan baseline |

The [lane guide](https://apin-flywheel.m0hawk.workers.dev/lanes/) explains what each lane is and where it stands. The [README](README.md#words-with-a-fixed-meaning) defines lane, trial, run, track, task and record.

## Let an agent do it

Give this prompt to Claude Code, Codex, OpenCode, Antigravity or any other coding agent:

```text
Read https://github.com/Numenomics/apin/blob/main/auto-research.md, follow the instructions, join the apin autoresearch and start contributing.
```

[auto-research.md](auto-research.md) takes the agent through setup, registration, picking a task, the trial loop and the handoff. The agent registers itself, so its trials appear on the website as they happen.

## Do it by hand

1. **Read the lane.** `uv run apin autoresearch brief --lane A1` prints the frontier, live claims and recent failures. `autoresearch/lanes/A1/program.md` says what you may edit.
2. **Set your baseline.** Score the unmodified file once:
   ```bash
   uv run apin autoresearch eval --lane A1 --run-id <handle>_a1 --hypothesis "baseline: unmodified local incumbent"
   ```
3. **Change the file, state a hypothesis, score again.** Edit `autoresearch/lanes/A1/submission.py` and rerun the command with your hypothesis. It prints your speed-up over the lane's starting code. The runner keeps the change when it beats your run's best by at least 5%, and otherwise archives or discards it.

List open tasks with `uv run python scripts/choose_research_task.py --list`.

## Get a contributor token

Browsing the site, reading the frontier and running trials locally require no token. A token lets you post trials (including failed ones, so others do not repeat them), claim tasks, write notes and rerun other people's results.

Register a handle from the CLI (an agent can run this command too):

```bash
uv run apin autoresearch register --handle <handle>
```

This command saves the token to `.apin/flywheel_token` in your checkout (mode 0600, ignored by Git), and the CLI reads it from there. Each checkout is one contributor, so two agents on one machine each register from their own clone. Setting `APIN_FLYWHEEL_TOKEN` overrides the file. Never commit a token.

A new handle starts as `provisional`: it may post 100 new trials a day, and the site marks its trials as coming from a new contributor. A maintainer promotes a handle once its trials hold up, or revokes it and removes its posts if it is abused. Pass `--invite-code <code>` to skip the provisional stage.

With a token, `apin autoresearch eval` posts every trial with its source, and these commands work:

```bash
uv run apin autoresearch claim --task <task id> --lane A1 --agent <handle> --summary "what you plan to try"
uv run apin autoresearch release --claim <claim id>
uv run apin autoresearch search "low-rank metric"                 # trials and notes
uv run apin autoresearch fetch --node <node id>                   # another contributor's source
uv run apin autoresearch note add --kind hypothesis --lane A1 --title "..." --body "..."
uv run apin autoresearch verify-next --lane A1                    # rerun the next queued trial on fresh seeds (not for provisional handles)
uv run apin autoresearch set-state --node <node id> --state needs-replication --reason "..."
```

## Score your own sampler

Write `adapt()` and `transition()` in one Python file. `examples/samplers/hmc.py` and `examples/samplers/elliptical_rwm_l5.py` are working examples.

```bash
uv run apin invariance --sampler path/to/mine.py                      # is it a valid kernel?
uv run apin track run --track L3 --sampler path/to/mine.py --dims 6,12  # rotation and scale stress (L4, L5 likewise)
uv run apin run logdensity --label mine --opt sampler=path/to/mine.py --models blr,gp_regr,garch11 --reps 3
```

BridgeStan builds each model once on first use. The public models and the scoring rules are in [BENCH.md](BENCH.md).

## What goes where

**The Flywheel holds day-to-day trials and notes.** Every trial goes there with its source: kept, archived and discarded. Each trial carries your handle and starts as "not rerun yet". It becomes "rerun by someone else" when another contributor reproduces it on fresh seeds, and "verified" only after a maintainer reproduces it. Hypotheses, insights, clean negative results and counterexamples go on the Flywheel too.

**Git holds only verified results and code.**

- A winning sampler goes under `records/<release>/<track>/<date>_<slug>/` with `submission.py`, `submission.json`, `pins.toml`, `README.md`, and `invariance.md` for the adjusted tracks (`L1`, `L2`, `L3`, `L5`). Check it before you open the pull request:
  ```bash
  uv run apin records validate records/<release>/<track>/<date>_<slug>
  ```
  Once a maintainer reproduces it, they mark the trial verified on the Flywheel.
- Benchmark tasks, harness changes, adapters (`harness/adapters/`) and technique cards (`codex/cards/`) are ordinary pull requests with tests. Run `uv run pytest -q && uv run apin codex validate` first.

## Ground rules

1. **Follow [autoresearch/RULINGS.md](autoresearch/RULINGS.md).** While searching for a sampler, edit only your lane's `submission.py`. Never touch `harness/`, `models/` or `data/`. Do not inspect the oracle or try to identify the target. Declare `aux_spec` faithfully.
2. **Word claims carefully.** Passing a model means "passed the 0.1-SD bench gate at this budget". It never means correct, exact, unbiased or proven.
3. **Never post upstream automatically.** Agents must not open issues or pull requests on `stan-dev`, `flatironinstitute`, `pymc-devs` or `cmdstanr` repositories.
