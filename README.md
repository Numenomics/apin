# apin

**apin** is a workbench for **Bayesian posterior approximators** and the libraries under them — covering adjusted MCMC & PDMP samplers, unadjusted/variational methods (Pathfinder, VI, Laplace, ULA, normalizing-flow transports), and C++/Rust engines like CmdStan, nutpie, and walnutpie. Every method is evaluated over 20 scored [posteriordb](https://github.com/stan-dev/posteriordb) models plus fresh affine-invariant analytic stress shapes under a strict **0.1-SD accuracy gate** and calibrated global-balance invariance checks: fast-and-wrong methods earn zero credit.

The same verification harness powers **two distinct modes**:

1. **Agent Benchmark (`harbor/`):** A standardized, containerized [Harbor](https://harborframework.com) benchmark (in the spirit of KernelBench and Terminal-Bench) for fairly comparing **LLMs and coding agents**. Every model starts from the exact same frozen baseline, container, and time budget and is graded on hidden seeds by pass rate and speedup over baseline.
2. **Open-Ended Autoresearch (`autoresearch/` + [apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)):** Pushing the **posterior-approximation frontier** as far as possible — use a single agent, a council of 5 LLMs, custom harnesses, or human insight. Anything goes on *process*; strict verification applies to the *resulting approximator* (`L1`–`L5` & `S2` artifact tracks). Flywheel holds all day-to-day trial bookkeeping (kept, archived, and discarded) while this Git repo stays light with only the harness, lane specs, and verified winners.

**Start here:**
- **Contribute (human or coding agent):** [CONTRIBUTING.md](CONTRIBUTING.md) (includes a 1-click copyable agent setup prompt and how to get a Flywheel token).
- **Run the LLM/Agent Benchmark:** [harbor/README.md](harbor/README.md).
- **Artifact scoring rules & past results:** [BENCH.md](BENCH.md), [PROTOCOL.md](PROTOCOL.md), [REPORT.md](REPORT.md).
- **Autonomous agent loop spec:** [AGENTS.md](AGENTS.md) & [auto-research.md](auto-research.md).
- **Maintainer / Flywheel ops:** [MAINTAINING.md](MAINTAINING.md) & [flywheel/README.md](flywheel/README.md).

---

## Two Modes at a Glance

| Pillar | What is ranked | Rules on *process* | Where it lives |
|---|---|---|---|
| **Agent Benchmark** | LLMs & agent harnesses | **Strict & fair:** fixed Harbor container, frozen starter, fixed timeout, hidden verifier seeds | `harbor/tasks/` (3 bug-fix rewinds + 5 optimization tasks `A0`–`A4`) |
| **Open-Ended Autoresearch** | Posterior approximators & engines | **Anything goes:** multi-LLM councils, custom harnesses, humans + agents; strict artifact gate only | `autoresearch/lanes/` (`A0`–`A4`, `S1`–`S2`) + [apin Flywheel](https://apin-flywheel.m0hawk.workers.dev) |

**Artifact Leaderboard Tracks** (published separately — never pooled across tiers):
- **Algorithmic:** `L1` (adjusted Markov kernels by calibrated cost units `n_density + r_m * n_gradient`), `L3` (Haar rotation + `1e-2..1e2` scale stress), `L4` (unadjusted & approximate methods — ULA, unadjusted MCLMC, Foster ULD, VI/Pathfinder/Laplace — measured bias & cost to `b²_max < 0.01`), `L5` (density-only `n_gradient == 0`).
- **Implementation:** `L2` (wall-clock `ESS/s` + host fingerprint), `S2` (upstream autodiff/compiler patches with byte-identical draw canary).

---

## Repository Layout

| path | what it holds |
|---|---|
| `harbor/` | Standardized **Agent Benchmark** suite (`base.Dockerfile`, `verify.py`, and 8 `harbor/tasks/*` containers with frozen baselines). |
| `autoresearch/` | **Open-Ended Autoresearch** lane specs (`A0`–`A4` algorithmic, `S1`–`S2` systems, `S3` rewind calibration), task catalog, negative memory, and anti-gaming `RULINGS.md`. |
| `flywheel/` | Self-hosted Cloudflare Workers + D1 research DAG, dashboard, multi-contributor auth, and MCP endpoint ([live instance](https://apin-flywheel.m0hawk.workers.dev)). |
| `harness/` | Shared runner, 0.1-SD gate, cost weights, out-of-process oracle IPC (`ipc.py`), invariance suite (`shapes.py`, `invariance.py`), lane judges (`judge.py`), tracks (`tracks.py`), and Flywheel client (`flywheel.py`). |
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

# Agent Benchmark (Harbor — requires Docker + Harbor CLI)
uv tool install harbor
docker build -f harbor/base.Dockerfile -t apin-harbor-base:latest .
harbor run -p harbor/tasks -a claude-code -m anthropic/claude-sonnet-4-5

# Open-Ended Autoresearch trial (auto-streams to Flywheel when APIN_FLYWHEEL_TOKEN is set)
uv run apin autoresearch eval --lane A1 --run-id my_run --dims 6,20 --repeats 2 --hypothesis "..."
uv run apin autoresearch eval --lane A4 --run-id my_a4  --dims 6,20 --repeats 2 --hypothesis "..."

# Real-Stan Artifact Leaderboard (L1 cost units or L2 wall-clock ESS/s)
uv run apin run logdensity --label hmc --opt sampler=examples/samplers/hmc.py --models public --reps 3
uv run apin run cmdstan --label cmdstan --models public --reps 3
uv run apin score --models public --reps 3 --baseline cmdstan --track L1 --submission hmc=examples/samplers/hmc.py
uv run apin score --models public --reps 3 --baseline cmdstan --track L2

# Specialized tracks (L3 stress, L4 unadjusted, L5 density-only) & invariance check
uv run apin invariance --sampler examples/samplers/hmc.py
uv run apin track run --track L3 --sampler examples/samplers/hmc.py --dims 6,12
uv run apin track run --track L4 --sampler examples/samplers/ula_wrong.py --dims 6
uv run apin track run --track L5 --sampler examples/samplers/elliptical_rwm_l5.py --dims 6,12
```

</details>

---

## Licence

BSD-3-Clause ([LICENSE](LICENSE)), copyright Maximilian Scholz. [LICENSING.md](LICENSING.md) covers upstreams, translated ESS code, patches, and data.
