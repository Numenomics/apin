# Agent Entry Point (`AGENTS.md`)

For open-ended requests ("set this up", "contribute", "join", "use spare tokens", "run autoresearch", "improve posterior inference"), follow [auto-research.md](auto-research.md) from step 1: it covers setup, picking and claiming a task, the trial loop and the handoff. [README.md](README.md) defines lane, trial, run, track, task and record. Explicit user assignments take precedence over the default task picker.

When you finish, tell the user which commands you ran, what was kept, what failed, and anything in these instructions you could not follow.

## Standing Rules (Non-Negotiable)
1. **No upstream posts:** Never open issues, comments, PRs, or reviews on `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr` (`upstream/STATUS.md`). Write local drafts/records only; the owner submits upstream personally.
2. **Respect `autoresearch/RULINGS.md`:** `requires = ["grad"]` (or density-only) unless explicitly authorized; never touch `harness/`, `models/`, or `data/` during an autoresearch run; never inspect `oracle` internals or fingerprint targets.
3. **Validation bar & claim wording:** Every candidate must pass Tier 0/1 exact-draw global-balance invariance (`harness/invariance.py`, including declared `aux_spec` stationarity), dispersed-start mixing, bit-exact replay, and—before claiming a real-posterior win—the 0.1-SD bench gate (`BENCH.md`). Always say *"passed the 0.1-SD bench gate at this budget"*, never *"is correct"*, *"exact"*, or *"proven"*.
4. **Shared host safety:** Keep the built-in FIFO evaluation lock enabled (`harness/queue.py`, do not pass `--no-queue` when other agents run on the same host). Isolate per-run artifacts via `--run-id`.
5. **Flywheel bookkeeping vs. Git:** All trial-by-trial history (kept, archived, and discarded) and active task leases live on the live Flywheel at `https://apin-flywheel.m0hawk.workers.dev` (public reads via `GET /api/frontier`, `/api/claims`, `/api/nodes`; set `APIN_FLYWHEEL_TOKEN` to auto-stream every `apin autoresearch eval` trial, and reserve work with `apin autoresearch claim` / `release`). Keep Git light — commit only winning/promoted submissions (`records/` PRs or lane incumbents).
6. **Continuation & handoff:** Maintain a concise `RESEARCH_HANDOFF.md` in your run directory with commands, evidence, failures, and open next questions. For harnesses supporting explicit continuation (e.g. `rohanarun/Auto-research-agent`), finish research turns with `RESEARCH_STATUS: continue`, `RESEARCH_STATUS: needs_input`, or `RESEARCH_STATUS: finished`.
