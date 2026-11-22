# Contributing to `apin`

Welcome! You can contribute in **two ways**:
1. **Open-Ended Autoresearch (push the frontier):** Anything goes on process (single agent, council of 5 LLMs, custom harness, human math) — stream trials to the live **[apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)** and submit verified winners via PR.
2. **Agent Benchmark (fair LLM/agent evaluation):** Run models/agents through our standardized **[Harbor](harbor/README.md)** task containers (`harbor/tasks/`) to measure apples-to-apples capability and speedup over frozen baselines.

Everything runs locally in ~3 minutes with **no Cloudflare account, Docker, or CmdStan install** required for Python invariance, autoresearch, and local `harbor/verify.py` grading (Docker + `uv tool install harbor` is only needed for full containerized Harbor runs).

---

## Option A: Let Your Agent Set It Up (Copy-Ready Prompt)

Give this prompt to Claude Code, Codex, OpenCode, Antigravity, or any coding agent:

```text
Clone https://github.com/Numenomics/apin.git, run `uv sync && uv run apin fetch-data && uv run pytest -q`, then read AGENTS.md and auto-research.md. Check the live Flywheel frontier and active claims at https://apin-flywheel.m0hawk.workers.dev (GET /api/frontier and GET /api/claims), pick an unclaimed open task from `uv run python scripts/choose_research_task.py --list`, and run a continuous autoresearch loop using `uv run apin autoresearch eval` until exhaustion, keeping RESEARCH_HANDOFF.md updated along the way.
```

If you have a contributor Flywheel token (`apfw_...`), export `APIN_FLYWHEEL_TOKEN` in the agent's shell before starting so every trial streams live to the dashboard automatically.

---

## Option B: Manual Quickstart (3 Steps)

### 1. Local setup
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+:

```bash
git clone https://github.com/Numenomics/apin.git && cd apin
uv sync
uv run apin fetch-data
uv run pytest -q
```

### 2. *(Optional)* Get a Flywheel contributor token
Browsing **[apin-flywheel.m0hawk.workers.dev](https://apin-flywheel.m0hawk.workers.dev)**, pulling briefs (`uv run apin autoresearch brief --lane A1`), and running locally need **no token** — autonomous agents can run end-to-end read-only without one. To claim tasks and stream live trials (including discarded ideas so others don't repeat them):

1. *(Human step — agents must not open GitHub issues)* Open a short GitHub issue on [Numenomics/apin](https://github.com/Numenomics/apin/issues) with your desired handle (e.g. your GitHub username) to receive a personal revocable `apfw_...` token.
   *(If a community invite code is posted in a release or discussion, self-register via `POST https://apin-flywheel.m0hawk.workers.dev/api/register` with `{"handle":"...","invite_code":"..."}`.)*
2. Export it in your shell (never commit it to git):
   ```bash
   export APIN_FLYWHEEL_TOKEN="apfw_..."
   ```
   The URL defaults to `https://apin-flywheel.m0hawk.workers.dev` automatically.

### 3. Pick something to work on

| What you want to do | Where to start | Command |
|---|---|---|
| **Open-ended autoresearch (`A0`–`A4`)** | `uv run python scripts/choose_research_task.py --list`, then read `autoresearch/lanes/<LANE>/program.md` | `uv run apin autoresearch eval --lane A1 --run-id <handle>_a1 --dims 6,20 --repeats 2 --hypothesis "..."` |
| **Run the standardized Agent Benchmark** | [harbor/README.md](harbor/README.md) (8 Harbor tasks: 3 bug-fix rewinds + 5 optimization tasks) | `uv tool install harbor && docker build -f harbor/base.Dockerfile -t apin-harbor-base:latest . && harbor run -p harbor/tasks ...` (or local `uv run python harbor/verify.py ...` without Docker) |
| **Benchmark your own Python posterior approximator** | Implement `adapt`/`transition` (see `examples/samplers/hmc.py` or `elliptical_rwm_l5.py`) | `uv run apin invariance --sampler path/to/mine.py` · `uv run apin track run --track L3\|L4\|L5 --sampler path/to/mine.py` |
| **Test on real Stan posteriors** (BridgeStan builds once per model) | Public models in `BENCH.md` | `uv run apin run logdensity --label mine --opt sampler=path/to/mine.py --models blr,gp_regr,garch11 --reps 3` |
| **Add an external adapter, Harbor task, or codex card** | `harness/adapters/`, `harbor/tasks/`, or `codex/cards/` | `uv run pytest -q && uv run apin codex validate` |

---

## Where Things Go: Flywheel vs. GitHub PR

- **Flywheel (`https://apin-flywheel.m0hawk.workers.dev`) — all day-to-day bookkeeping:**
  - Every autoresearch trial (kept, archived, **and failed/discarded**) streams automatically when `APIN_FLYWHEEL_TOKEN` is set.
  - Contributor trials are tagged with your handle and `verified = 0`.
  - Clean negative results and counterexamples are first-class outcomes here — they keep others from repeating dead ends.
- **GitHub PR (`Numenomics/apin`) — only clean, verified artifacts:**
  - **Winning posterior approximators:** package under `records/<release>/<track>/<date>_<slug>/` (`submission.py`, `submission.json`, `pins.toml`, `README.md`, plus `invariance.md` for adjusted tracks `L1`/`L2`/`L3`/`L5`) and validate locally with:
    ```bash
    uv run apin records validate records/<release>/<track>/<date>_<slug>
    ```
    Once reproduced on the maintainer's harness, the corresponding Flywheel node is promoted to `verified = 1`.
  - **Harbor tasks, harness, adapters, or codex improvements:** standard PR with unit tests (`uv run pytest`).

---

## Ground Rules

1. **Follow [autoresearch/RULINGS.md](autoresearch/RULINGS.md):** During sampler/approximator searches, edit only your lane's `submission.py` (never touch `harness/`, `models/`, or `data/`); do not inspect oracle internals or fingerprint targets; declare `aux_spec` faithfully.
2. **Claim wording:** Passing a model means *"passed the 0.1-SD bench gate at this budget"* — never *"is correct"*, *"exact"*, *"unbiased"*, or *"proven"*.
3. **Never auto-post upstream:** Do not have agents open issues or PRs on `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr` repositories.
