# apin

**apin** is a benchmark and autoresearch workbench for MCMC samplers and the libraries under them. It evaluates any sampler — Python kernels or C++/Rust upstreams like CmdStan, nutpie, and walnutpie — over a fixed suite of 20 scored [posteriordb](https://github.com/stan-dev/posteriordb) models plus fresh affine-invariant analytic stress shapes. Every run is checked against a reference posterior under a strict **0.1-SD accuracy gate** and calibrated global-balance invariance checks: fast-and-wrong samplers earn zero credit, and passing samplers are compared by target-calibrated cost units (`ESS/cost` for algorithmic progress) alongside wall-clock speed (`ESS/s` for implementation progress).

The project separates **verified results in Git** from **live experiment bookkeeping**. Winning submissions, harness code, and lane specs live in this repository (keeping it light), while the live **[apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)** tracks every trial (kept, archived, and discarded), parent→child research lineage, active task leases, and synthetic-vs-real-Stan alignment. Reads on Flywheel are public; contributors use personal revocable tokens to stream trials and open GitHub PRs under `records/` for verified winners.

To get started locally, install [uv](https://docs.astral.sh/uv/) and Python 3.12+:

```bash
git clone https://github.com/Numenomics/apin.git && cd apin
uv sync && uv run apin fetch-data && uv run pytest
```

Read [BENCH.md](BENCH.md) for the scoring rules and anti-gaming defenses, [PROTOCOL.md](PROTOCOL.md) for the measurement contract, [REPORT.md](REPORT.md) for the 2026 Stan wall-time campaign, [flywheel/README.md](flywheel/README.md) for the live research DAG & contributor keys, and [auto-research.md](auto-research.md) if you are running an autonomous coding agent.

---

## Tracks & Lanes

**Benchmark Tracks** (published separately — never pooled across tiers):
- **Algorithmic:** `L1` (primary: adjusted kernels by calibrated cost units `n_density + r_m * n_gradient`), `L3` (Haar rotation + `1e-2..1e2` scale stress), `L4` (unadjusted kernels — measured bias & cost to `b²_max < 0.01`, segregated from L1), `L5` (density-only `n_gradient == 0`).
- **Implementation:** `L2` (wall-clock `ESS/s` + host fingerprint), `S2` (upstream autodiff/compiler patches with byte-identical draw canary).

**Autoresearch Lanes** (isolated search islands):
- **Algorithmic (`A0`–`A4`):** `A0` ground-up full sampler · `A1` preconditioning/metric (`adapt_metric` seam) · `A2` multiscale & funnel dynamics · `A3` integrators, momentum, trajectories & PDMPs · `A4` warm-start & diffeomorphic nonlinear transport (with automatic finite-diff Jacobian verification).
- **Systems & Calibration (`S1`–`S3`):** `S1` upstream sampler patches (`walnutpie`, `stan`, `nuts-rs`) · `S2` gradient/compiler speed (`stan-math`, `stanc3`, `bridgestan`) · `S3` rewind calibration & agent grading (AUARC).

---

## Repository Layout

| path | what it holds |
|---|---|
| `harness/` | Runner, 0.1-SD gate, calibrated cost weights, out-of-process oracle IPC (`ipc.py`), invariance suite (`shapes.py`, `invariance.py`), read-only lane judges (`judge.py`), tracks (`tracks.py`), systems/rewind runners, and Flywheel client (`flywheel.py`). |
| `autoresearch/` | Lane specs (`lanes/A0..A4`, `S1..S3`), task catalog, negative memory, and anti-gaming `RULINGS.md`. |
| `flywheel/` | Self-hosted Cloudflare Workers + D1 research DAG, dashboard, multi-contributor auth, and MCP endpoint ([live instance](https://apin-flywheel.m0hawk.workers.dev)). |
| `codex/` | Schema-validated `INDEX.yaml` and short technique cards across MCMC ecosystems (`apin codex validate`). |
| `models/` & `data/` | 21 core Stan models, `manifest.json` (10 public / 10 held-out scored + `kronecker_gp`), calibrated `costs.json`, and reference posterior summaries. |
| `examples/samplers/` | Reference Python submissions: HMC (`hmc.py`), intentionally biased ULA (`ula_wrong.py`), and density-only L5 baseline (`elliptical_rwm_l5.py`). |
| `upstream/` | Git submodules pinned to owner fork branches (`math`, `stan`, `stanc3`, `cmdstan`, `bridgestan`, `walnutpie`, `posteriordb`), build notes, and `STATUS.md`. |
| `results/` | Campaign records (`results/records/`), tables, and local upstream issue drafts. |
| `tests/` | Fast unit tests (`uv run pytest`) and slow end-to-end model tests (`-m slow`). |

---

<details>
<summary><strong>Running Benchmarks & Scoring (CLI Reference)</strong></summary>

### Run a sampler over Stan models and score it

```bash
export CMDSTAN=/path/to/cmdstan        # needed for cmdstan / pathfinder adapters
uv run apin list                       # available adapters, models, and r_grad cost weights

# Python sampler through the isolated adapt/transition log-density seam:
uv run apin run logdensity --label hmc --opt sampler=examples/samplers/hmc.py --models public --reps 3
uv run apin run cmdstan --label cmdstan --models public --reps 3

# Score on Track L1 (cost units) or Track L2 (wall-clock ESS/s + machine fingerprint):
uv run apin score --models public --reps 3 --baseline cmdstan --track L1 --submission hmc=examples/samplers/hmc.py
uv run apin score --models public --reps 3 --baseline cmdstan --track L2
```

`--models` accepts `all`, a family name, `public`, `held-out`, or a comma list. `runs/` and `scores/` are gitignored; reruns skip completed reps whose seed, submission digest, and budget match.

### Specialized Benchmark Tracks (`L3`, `L4`, `L5`) & Utilities

```bash
uv run apin invariance --sampler examples/samplers/hmc.py               # exact-draw invariance & mixing
uv run apin track run --track L3 --sampler examples/samplers/hmc.py --dims 6,12
uv run apin track run --track L4 --sampler examples/samplers/ula_wrong.py --dims 6
uv run apin track run --track L5 --sampler examples/samplers/elliptical_rwm_l5.py --dims 6,12
uv run apin costs build --models all                                    # measure r_grad into models/costs.json
uv run apin records validate records/<release>/<track>/<slug>           # validate a PR submission record
uv run apin codex validate                                              # validate codex/INDEX.yaml
uv run apin reference build --models pilots                             # rebuild local reference (needs CMDSTAN)
```

</details>

<details>
<summary><strong>Autoresearch & Autonomous Agents (A0–A4, S1–S3, Flywheel)</strong></summary>

Agents start at [AGENTS.md](AGENTS.md) and [auto-research.md](auto-research.md).

```bash
# Optional: stream every trial automatically to the live Flywheel DAG
export APIN_FLYWHEEL_TOKEN="<your-contributor-or-admin-token>"

# Pick an open research task
uv run python scripts/choose_research_task.py --list

# Evaluate an algorithmic lane trial (isolated IPC, FIFO host queue, dual-horizon dims=6,20)
uv run apin autoresearch eval --lane A1 --run-id my_run --dims 6,20 --repeats 2 --hypothesis "..."
uv run apin autoresearch eval --lane A4 --run-id my_a4  --dims 6,20 --repeats 2 --hypothesis "..."

# Validate Lane A1 (adapt_metric) or Lane A4 (adapt_transport) on real Stan posteriors
uv run apin run logdensity --label a1_cand --opt sampler=autoresearch/lanes/A1/submission.py --opt lane=A1 --models blr,gp_regr --reps 3
uv run apin run logdensity --label a4_cand --opt sampler=autoresearch/lanes/A4/submission.py --opt lane=A4 --models blr,eight_schools_centered --reps 3

# Systems & calibration lanes (S1 upstream sampler patches, S2 gradient speed, S3 rewind AUARC)
uv run apin autoresearch eval-s1 --upstream walnutpie --allowed-path src/ --hypothesis "..." --dry-run
uv run apin autoresearch eval-s2 --upstream math --allowed-path stan/math/rev/ --hypothesis "..." --dry-run
uv run apin autoresearch eval-s3 --mode reference
```

Note: OS-level container sandboxing and a private sealed holdout remain Slice 6.

</details>

<details>
<summary><strong>Submodules, Upstream Patching & Adding Samplers</strong></summary>

Submodules are not checked out by default; fetch only what you need:

```bash
git submodule update --init upstream/math upstream/stan
```

Each upstream has build instructions in `upstream/<name>.md` and patch status in `upstream/STATUS.md`. Patch workflow:
1. Branch inside `upstream/<name>/` from the target upstream commit (`origin` points to the owner's fork).
2. Build, run upstream unit tests, check finite-difference gradients / bit-identity canary (`PROTOCOL.md`), and benchmark against a matched baseline.
3. Record the result in `results/records/`, push the fork branch, update the submodule pin + `upstream/STATUS.md`.
4. **Never** open issues or PRs directly on `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr` from agents — the owner submits upstream personally.

**Adding a non-Python sampler:** add one adapter file in `harness/adapters/` subclassing `harness.sampler.Sampler` (auto-discovered by `apin list`). Python samplers use the built-in `logdensity` adapter directly.

**Slow end-to-end tests:**
```bash
CMDSTAN=/path/to/cmdstan uv run pytest -m slow
```

</details>

---

## Licence

BSD-3-Clause ([LICENSE](LICENSE)), copyright Maximilian Scholz. [LICENSING.md](LICENSING.md) covers upstreams, translated ESS code, patches, and data.
