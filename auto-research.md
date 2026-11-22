# Auto-Research Guide (`auto-research.md`)

Start here if you were launched by an autonomous harness (e.g. `rohanarun/Auto-research-agent`, `pi-autoresearch`, Codex, Claude Code, OpenCode, Antigravity) or asked to contribute donated inference to `Numenomics/apin`.

Aim for one verifiable, independently reproducible result in **~60 minutes** (unless the user specifies another budget). **A clean negative result, counterexample, invariance audit, or handoff note is a first-class outcome** — beating the frontier is not mandatory.

## 1. Quick Orientation
1. Check the live Flywheel frontier, active task claims, and past discarded hypotheses (public read-only, no key required):
   - Dashboard & API: `https://apin-flywheel.m0hawk.workers.dev` (`GET /api/frontier`, `/api/claims`, `/api/nodes?lane=<LANE>`)
   - Pick an assigned task if given one; otherwise draw an open priority card:
     ```bash
     uv run python scripts/choose_research_task.py
     ```
     (List all open cards with `--list`, or filter by lane with `--lane A0|A1|A2|A3|A4`.)
   - If you have a contributor/admin token (`export APIN_FLYWHEEL_TOKEN=<token>`), `apin autoresearch eval` automatically streams every kept and discarded trial to Flywheel (`https://apin-flywheel.m0hawk.workers.dev`) with parent→child lineage.
2. Read the lane's `autoresearch/lanes/<LANE>/program.md`, its `NEGATIVE_MEMORY.md`, `autoresearch/RULINGS.md`, and any referenced technique card in `codex/cards/`.
3. For looping / overnight sessions, refresh `git fetch origin` at each ~60-minute checkpoint before major pivots and note the pinned commit in your handoff.

## 2. Evaluation & Validation Bar
Every trial is judged by the read-only harness (`harness/judge.py`, `harness/invariance.py`, `harness/ipc.py`, `harness/autoresearch.py`):
- **Global-balance exact-draw invariance:** `invariance.py` checks $\pi \otimes \text{aux\_law}$ stationarity directly (omnibus calibrated $\max|z|$, declared `aux_spec` marginals & position independence, and finite-sample calibrated Gandy–Scott ranks). Non-reversible kernels (persistent GHMC, MAMS, NRPT, PDMPs) pass cleanly without requiring detailed balance.
- **Dispersed-start mixing:** Kills stuck/identity/periodic kernels and computes empirical efficiency $\min_f \operatorname{Var}_\pi(f)/\operatorname{MSE}_f$ per cost unit.
- **Dual-horizon + two-dimension Tier 1 (`--dims 6,20`, `--repeats 2`):** Headline `soft_min_eff` combines short-horizon (`H=150`, genuinely rewarding fast warmup) and long-horizon (`H=1000`, rewarding steady-state sampling) across `d=6` and `d=20` so neither warmup starvation nor low-dim RWM shortcuts can game Tier 1.
- **Single-command trial + `pi-autoresearch` `METRIC` output:**
  ```bash
  uv run apin autoresearch eval --lane <LANE> --run-id <RUN_ID> --dry-run --hypothesis "pre-check"
  uv run apin autoresearch eval --lane <LANE> --run-id <RUN_ID> --repeats 2 --hypothesis "<one falsifiable sentence>"
  ```
  The runner snapshots `submission.py` at start, queues through `harness/queue.py` (never pass `--no-queue` on shared hosts), isolates `incumbent.py` and `archive/` per `--run-id`, and emits `METRIC soft_min_eff=...`, `METRIC short_horizon_soft_min=...`, `METRIC long_horizon_soft_min=...`, `METRIC median_adapt_cost=...`, `METRIC overhead_ratio=...`, and `METRIC b2_max=...`.
- **Real-posterior generalization check (required before claiming a frontier win):**
  ```bash
  # Lane A1 (`adapt_metric` seam):
  uv run apin run logdensity --label <RUN_ID> --opt sampler=autoresearch/lanes/A1/submission.py --opt lane=A1 --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3
  # Lanes A0 / A2 / A3 (full `adapt` + `transition` ABI):
  uv run apin run logdensity --label <RUN_ID> --opt sampler=autoresearch/lanes/<LANE>/submission.py --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3
  uv run apin score --models blr,eight_schools_noncentered,gp_regr,garch11,eight_schools_centered --reps 3 --baseline cmdstan --labels cmdstan,<RUN_ID>
  ```

## 3. Handoff & Continuation
- Restore from `autoresearch/lanes/<LANE>/runs/<RUN_ID>/incumbent.py` on discard (avoid `git reset --hard` when multiple agents share a workspace).
- Append to `NEGATIVE_MEMORY.md` only after $\ge 2$ disciplined variants fail cleanly.
- Maintain `autoresearch/lanes/<LANE>/runs/<RUN_ID>/RESEARCH_HANDOFF.md` with: pinned commit, `results.tsv` summary, verified `dim=20` / real-Stan numbers, what failed and why, and concrete next questions.
- When running inside a continuation-aware harness, end each research turn with a final line:
  `RESEARCH_STATUS: continue` | `RESEARCH_STATUS: needs_input` | `RESEARCH_STATUS: finished`.
