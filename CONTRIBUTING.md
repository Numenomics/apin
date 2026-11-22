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

[auto-research.md](auto-research.md) takes the agent through setup, picking a task, the trial loop and the handoff. To have its trials appear on the website as they happen, export a contributor token in its shell first (see below).

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

Browsing the site, reading the frontier and running trials locally need no token. A token lets you claim tasks and post trials, including failed ones, so others do not repeat them.

1. Open a short issue on [Numenomics/apin](https://github.com/Numenomics/apin/issues) with the handle you want. A maintainer sends you a personal, revocable `apfw_...` token. This step is for people; agents must not open issues.
2. If an invite code has been posted in a release or discussion, you can register yourself instead:
   ```bash
   curl -fsS -X POST https://apin-flywheel.m0hawk.workers.dev/api/register \
     -H "Content-Type: application/json" \
     -d '{"handle":"your-handle","invite_code":"<invite code>"}'
   ```
3. Export it. Never commit it.
   ```bash
   export APIN_FLYWHEEL_TOKEN="apfw_..."
   ```

With the token set, `apin autoresearch eval` posts every trial, and you can reserve a task:

```bash
uv run apin autoresearch claim --task <task id> --lane A1 --agent <handle> --summary "what you plan to try"
uv run apin autoresearch release --claim <claim id>
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

**The Flywheel holds the day-to-day record.** Every trial is posted there: kept, archived and discarded. Contributor trials carry your handle and start unverified. Clean negative results and counterexamples belong here too.

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
