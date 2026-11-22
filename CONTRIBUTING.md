# Contributing to `apin`

Welcome! You can contribute by hand or by pointing a coding agent (Claude Code, Codex, OpenCode, Antigravity, etc.) at this repo. Everything needed to run locally takes ~3 minutes and requires **no Cloudflare account and no CmdStan install** for the fast Python invariance & autoresearch tracks.

---

## Option A: Let Your Agent Set It Up (Copy-Paste Prompt)

Clone the repo (or let your agent clone it) and paste this prompt:

> Clone `https://github.com/Numenomics/apin.git`, run `uv sync && uv run apin fetch-data && uv run pytest -q`, then read `AGENTS.md` and `auto-research.md`. Check the live Flywheel frontier and active claims at `https://apin-flywheel.m0hawk.workers.dev` (`GET /api/frontier`, `GET /api/claims`), pick an unclaimed open task from `scripts/choose_research_task.py --list`, and run a 45–60 minute autoresearch session using `uv run apin autoresearch eval`.

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
Browsing **[apin-flywheel.m0hawk.workers.dev](https://apin-flywheel.m0hawk.workers.dev)** and running locally need **no token**. To claim tasks and stream live trials (including discarded ideas so others don't repeat them):

1. Open a short GitHub issue on [Numenomics/apin](https://github.com/Numenomics/apin/issues) with your desired handle (e.g. your GitHub username) to get a personal revocable `apfw_...` token.
   *(If a community invite code is posted, you can self-register immediately via `POST https://apin-flywheel.m0hawk.workers.dev/api/register` with `{"handle":"...","invite_code":"..."}`.)*
2. Export it in your shell (never commit it to git):
   ```bash
   export APIN_FLYWHEEL_TOKEN="apfw_..."
   ```
   The URL defaults to `https://apin-flywheel.m0hawk.workers.dev` automatically.

### 3. Pick something to work on

| What you want to do | Where to start | Command |
|---|---|---|
| **Evolve a sampler / approximator in an autoresearch lane** (`A0`–`A4`) | Run `uv run python scripts/choose_research_task.py --list`, then read `autoresearch/lanes/<LANE>/program.md` | `uv run apin autoresearch eval --lane A1 --run-id <handle>_a1 --dims 6,20 --repeats 2 --hypothesis "..."` |
| **Benchmark your own Python posterior approximator** | Implement `adapt`/`transition` (see `examples/samplers/hmc.py` or `elliptical_rwm_l5.py`) | `uv run apin invariance --sampler path/to/mine.py` · `uv run apin track run --track L3\|L4\|L5 --sampler path/to/mine.py` |
| **Test on real Stan posteriors** (requires BridgeStan; builds once per model) | Choose public models in `BENCH.md` | `uv run apin run logdensity --label mine --opt sampler=path/to/mine.py --models blr,gp_regr,garch11 --reps 3` |
| **Add a new external sampler adapter** (Rust, C++, Julia, JAX, etc.) | Subclass `harness.sampler.Sampler` in `harness/adapters/` | Auto-discovered by `uv run apin list` |
| **Add a literature technique card** | Check `codex/INDEX.yaml` & `codex/cards/` | Validate with `uv run apin codex validate` |

---

## Where Things Go: Flywheel vs. GitHub PR

- **Flywheel (`https://apin-flywheel.m0hawk.workers.dev`) — all day-to-day bookkeeping:**
  - Every trial (kept, archived, **and failed/discarded**) streams automatically when `APIN_FLYWHEEL_TOKEN` is set.
  - Contributor trials are tagged with your handle and `verified = 0`.
  - Clean negative results and counterexamples are genuinely valuable here — they stop the next person/agent from wasting compute.
- **GitHub PR (`Numenomics/apin`) — only clean, verified artifacts:**
  - **Winning sampler/approximator submissions:** package under `records/<release>/<track>/<date>_<slug>/` (must include `submission.py`, `submission.json`, `pins.toml`, `README.md`, and `invariance.md` for adjusted tracks `L1`/`L2`/`L3`/`L5`) and verify locally with:
    ```bash
    uv run apin records validate records/<release>/<track>/<date>_<slug>
    ```
    Once reproduced on the maintainer's harness, the corresponding Flywheel node is promoted to `verified = 1`.
  - **Harness, track, adapter, or codex improvements:** standard PR with unit tests (`uv run pytest`).

---

## Ground Rules

1. **Follow [autoresearch/RULINGS.md](autoresearch/RULINGS.md):** During sampler/approximator searches, edit only your lane's `submission.py` (never touch `harness/`, `models/`, or `data/`); do not inspect oracle internals or fingerprint targets; declare `aux_spec` faithfully.
2. **Claim wording:** Passing a model means *"passed the 0.1-SD bench gate at this budget"* — never *"is correct"*, *"exact"*, *"unbiased"*, or *"proven"*.
3. **Never auto-post upstream:** Do not have agents open issues or PRs on `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr` repositories.
