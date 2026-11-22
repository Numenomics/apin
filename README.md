# apin

**apin** is a benchmark and autoresearch workbench for **Bayesian posterior approximators** and the libraries under them — covering adjusted MCMC & PDMP samplers, unadjusted/variational approximators (Pathfinder, VI, Laplace, ULA, normalizing-flow transports), and C++/Rust engines like CmdStan, nutpie, and walnutpie. Every method is checked over 20 scored [posteriordb](https://github.com/stan-dev/posteriordb) models plus fresh affine-invariant analytic stress shapes against a **0.1-SD accuracy gate** and calibrated global-balance invariance checks: fast-and-wrong methods earn zero credit, and passing methods are ranked by target-calibrated cost units (`ESS/cost`, algorithmic) alongside wall-clock speed (`ESS/s`, implementation).

This repository stays light on purpose: it holds the harness, lane specs, and **winning/verified submissions**, while the live **[apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)** stores day-to-day experiment bookkeeping — every trial (kept, archived, and discarded), parent→child lineage, active task claims, and synthetic-vs-real-Stan alignment. Anyone can browse Flywheel or run benchmarks locally with no account.

**Start here:**
- **Contribute (human or coding agent):** Read **[CONTRIBUTING.md](CONTRIBUTING.md)** (includes a copy-paste setup prompt for Claude Code / Codex / OpenCode / Antigravity, plus how to get a Flywheel token).
- **Benchmark rules & past results:** [BENCH.md](BENCH.md) (rules & anti-gaming), [PROTOCOL.md](PROTOCOL.md) (measurement contract), [REPORT.md](REPORT.md) (2026 Stan wall-time campaign).
- **Autonomous agents:** [AGENTS.md](AGENTS.md) & [auto-research.md](auto-research.md).
- **Maintainer / self-hosting:** [MAINTAINING.md](MAINTAINING.md) & [flywheel/README.md](flywheel/README.md).

---

## Tracks & Lanes

**Benchmark Tracks** (published separately — never pooled across tiers):
- **Algorithmic:** `L1` (adjusted Markov kernels by calibrated cost units `n_density + r_m * n_gradient`), `L3` (Haar rotation + `1e-2..1e2` scale stress), `L4` (unadjusted & approximate methods — ULA, unadjusted MCLMC, Foster ULD, VI/Pathfinder/Laplace — measured bias & cost to `b²_max < 0.01`), `L5` (density-only `n_gradient == 0`).
- **Implementation:** `L2` (wall-clock `ESS/s` + host fingerprint), `S2` (upstream autodiff/compiler patches with byte-identical draw canary).

**Autoresearch Lanes** (isolated search islands):
- **Algorithmic (`A0`–`A4`):** `A0` ground-up full sampler · `A1` preconditioning/metric (`adapt_metric` seam) · `A2` multiscale & funnel dynamics · `A3` integrators, momentum, trajectories & PDMPs · `A4` warm-start & diffeomorphic nonlinear transport (with automatic finite-diff Jacobian verification).
- **Systems & Calibration (`S1`–`S3`):** `S1` upstream sampler patches · `S2` gradient/compiler speed · `S3` rewind calibration & agent grading (AUARC).

---

## Repository Layout

| path | what it holds |
|---|---|
| `harness/` | Runner, 0.1-SD gate, cost weights, out-of-process oracle IPC (`ipc.py`), invariance suite (`shapes.py`, `invariance.py`), lane judges (`judge.py`), tracks (`tracks.py`), systems/rewind runners, and Flywheel client (`flywheel.py`). |
| `autoresearch/` | Lane specs (`lanes/A0..A4`, `S1..S3`), task catalog, negative memory, and anti-gaming `RULINGS.md`. |
| `flywheel/` | Self-hosted Cloudflare Workers + D1 research DAG, dashboard, multi-contributor auth, and MCP endpoint ([live instance](https://apin-flywheel.m0hawk.workers.dev)). |
| `codex/` | Schema-validated `INDEX.yaml` and short technique cards across inference ecosystems (`apin codex validate`). |
| `models/` & `data/` | 21 core Stan models, `manifest.json` (10 public / 10 held-out scored + `kronecker_gp`), calibrated `costs.json`, and reference posterior summaries. |
| `examples/samplers/` | Reference Python submissions: HMC (`hmc.py`), intentionally biased ULA (`ula_wrong.py`), and density-only L5 baseline (`elliptical_rwm_l5.py`). |
| `upstream/` | Submodules pinned to fork branches (`math`, `stan`, `stanc3`, `cmdstan`, `bridgestan`, `walnutpie`, `posteriordb`) and `STATUS.md`. |
| `results/` | Campaign records (`results/records/`), tables, and local upstream issue drafts. |
| `tests/` | Fast unit tests (`uv run pytest`) and slow end-to-end model tests (`-m slow`). |

---

<details>
<summary><strong>Quick CLI Reference</strong></summary>

```bash
# Setup (uv + Python 3.12+)
uv sync && uv run apin fetch-data && uv run pytest -q

# Run & score a Python approximator on Stan models (L1 cost units or L2 wall-clock ESS/s)
uv run apin run logdensity --label hmc --opt sampler=examples/samplers/hmc.py --models public --reps 3
uv run apin run cmdstan --label cmdstan --models public --reps 3
uv run apin score --models public --reps 3 --baseline cmdstan --track L1 --submission hmc=examples/samplers/hmc.py
uv run apin score --models public --reps 3 --baseline cmdstan --track L2

# Fast invariance & specialized tracks (no CmdStan required)
uv run apin invariance --sampler examples/samplers/hmc.py
uv run apin track run --track L3 --sampler examples/samplers/hmc.py --dims 6,12
uv run apin track run --track L4 --sampler examples/samplers/ula_wrong.py --dims 6
uv run apin track run --track L5 --sampler examples/samplers/elliptical_rwm_l5.py --dims 6,12

# Autoresearch trial (auto-streams to Flywheel when APIN_FLYWHEEL_TOKEN is set)
uv run apin autoresearch eval --lane A1 --run-id my_run --dims 6,20 --repeats 2 --hypothesis "..."
uv run apin autoresearch eval --lane A4 --run-id my_a4  --dims 6,20 --repeats 2 --hypothesis "..."
uv run apin autoresearch eval-s3 --mode reference
```

</details>

---

## Licence

BSD-3-Clause ([LICENSE](LICENSE)), copyright Maximilian Scholz. [LICENSING.md](LICENSING.md) covers upstreams, translated ESS code, patches, and data.
