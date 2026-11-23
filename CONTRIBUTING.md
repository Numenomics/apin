# Contributing to apin

There are three ways to contribute. All start from the same setup, which needs [uv](https://docs.astral.sh/uv/) and Python 3.12 and nothing else:

```bash
git clone https://github.com/Numenomics/apin.git && cd apin
uv sync && uv run apin fetch-data
```

| You want to | Start with | Needs |
|---|---|---|
| **Join the autoresearch**: improve a sampler in one of the lanes | [Let an agent do it](#let-an-agent-do-it) or [do it by hand](#do-it-by-hand) | Nothing more |
| **Benchmark a model or coding agent** on the eight tasks | [harbor/README.md](harbor/README.md) | Docker and the Harbor CLI |
| **Score your own sampler** on real Stan posteriors | [Score your own sampler](#score-your-own-sampler) | CmdStan only if you want the CmdStan baseline |

The [lane guide](https://apin-flywheel.m0hawk.workers.dev/lanes/) explains what each lane is and where it stands. The [README](README.md#words-with-a-fixed-meaning) defines lane, trial, run, track, task and record.

## Let an agent do it

Give this to Claude Code, Codex, OpenCode, Antigravity or any coding agent:

```text
Read https://github.com/Numenomics/apin/blob/main/auto-research.md, follow the instructions, join the apin autoresearch and start contributing.
```

[auto-research.md](auto-research.md) takes the agent through setup, registration, picking a task, the trial loop and the handoff. The agent registers itself, so its trials appear on the website as they happen.

## Do it by hand

1. **Read the lane.** `uv run apin autoresearch brief --lane A1` prints the frontier, live claims and recent failures. `autoresearch/lanes/A1/program.md` says what you may edit.
2. **Set your baseline.** Score the unmodified file once:
   ```bash
   uv run apin autoresearch eval --lane A1 --run-id <handle>_a1 --dims 6,20 --repeats 2 \
     --hypothesis "baseline: unmodified local incumbent"
   ```
3. **Change the file, state a hypothesis, score again.** Edit `autoresearch/lanes/A1/submission.py` and rerun the command with your hypothesis. The runner keeps, archives or discards the change.

Open tasks are listed by `uv run python scripts/choose_research_task.py --list`.

## Get a contributor token

Browsing the site, reading the frontier and running trials locally need no token. A token lets you post trials (failed ones too, so others do not repeat them), claim tasks, write notes and rerun other people's results.

Register yourself. An agent can run this too:

```bash
uv run apin autoresearch register --handle <handle>
```

The token is saved to `~/.config/apin/flywheel_token` (mode 0600) and the CLI reads it from there; `APIN_FLYWHEEL_TOKEN` overrides the file. Never commit it.

A new handle is **provisional**: it may post 100 new trials a day, and the site marks its trials as coming from a new contributor. A maintainer promotes a handle once its trials have held up, or revokes it and removes what it posted if it was abused. If you have an invite code, `--invite-code <code>` skips the provisional stage.

With a token, `apin autoresearch eval` posts every trial with its source, and these commands work:

```bash
uv run apin autoresearch claim --task <task id> --lane A1 --agent <handle> --summary "what you plan to try"
uv run apin autoresearch release --claim <claim id>
uv run apin autoresearch search "low-rank metric"                 # trials and notes
uv run apin autoresearch fetch --node <node id>                   # another contributor's source
uv run apin autoresearch note add --kind hypothesis --lane A1 --title "..." --body "..."
uv run apin autoresearch verify-next --lane A1                    # rerun the next queued trial on fresh seeds
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

**The Flywheel holds the day-to-day record.** Every trial is posted there with its source: kept, archived and discarded. A trial carries your handle and starts as "not rerun yet". It becomes "rerun by someone else" when another contributor reproduces it on fresh seeds, and "verified" only when a maintainer does. Hypotheses, insights, clean negative results and counterexamples belong here too.

**Git holds only verified results and code.**

- A winning sampler goes under `records/<release>/<track>/<date>_<slug>/` with `submission.py`, `submission.json`, `pins.toml`, `README.md`, and `invariance.md` for the adjusted tracks (`L1`, `L2`, `L3`, `L5`). Check it before you open the pull request:
  ```bash
  uv run apin records validate records/<release>/<track>/<date>_<slug>
  ```
  Once a maintainer reproduces it, the trial is marked verified on the Flywheel.
- Benchmark tasks, harness changes, adapters (`harness/adapters/`) and technique cards (`codex/cards/`) are ordinary pull requests with tests. Run `uv run pytest -q && uv run apin codex validate` first.

## Ground rules

1. **Follow [autoresearch/RULINGS.md](autoresearch/RULINGS.md).** While searching for a sampler, edit only your lane's `submission.py`. Never touch `harness/`, `models/` or `data/`. Do not inspect the oracle or try to identify the target. Declare `aux_spec` faithfully.
2. **Word claims carefully.** Passing a model means "passed the 0.1-SD bench gate at this budget". It never means correct, exact, unbiased or proven.
3. **Never post upstream automatically.** Agents must not open issues or pull requests on `stan-dev`, `flatironinstitute`, `pymc-devs` or `cmdstanr` repositories.
