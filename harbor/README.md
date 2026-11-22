# apin Agent Benchmark (Harbor Suite)

Standardized, containerized **LLM & coding-agent benchmark** built on the [Harbor](https://harborframework.com) task format (`laude-institute/harbor`). Every model/agent gets the exact same frozen starter code, container environment, time budget, and hidden verification seeds so results are fair and apples-to-apples across releases.

> **Agent Benchmark vs. Open-Ended Autoresearch:**
> - **Agent Benchmark (here, `harbor/`):** Evaluates **LLMs & agent harnesses** under strict, fixed conditions (single model/agent, fixed budget, frozen baseline → pass rate + speedup over baseline).
> - **Open-Ended Autoresearch (`autoresearch/` + [apin Flywheel](https://apin-flywheel.m0hawk.workers.dev)):** Evaluates the **best posterior approximator** ever produced — anything goes on process (multi-LLM councils, custom harnesses, human-in-the-loop), strict rules only on the final artifact.

---

## Task Categories

### Category 1 — Diagnostic & Invariance Rewinds (Binary Pass/Fail, 15 min)
Tasks derived from real MCMC / transport bugs discovered in `apin` campaigns. Base fails the hidden invariance/Jacobian check; agent must diagnose and fix it without cheating.

| Task | Difficulty | Description | Metric |
|---|---|---|---|
| `s3-position-gated-mh` | easy | Fix position-dependent kernel gating breaking MH detailed balance | `reward ∈ {0, 1}` |
| `s3-missing-jacobian-a4` | easy | Fix missing `log_abs_det_jac` & `pullback_grad` in Lane A4 sinh-arcsinh transport | `reward ∈ {0, 1}` |
| `s3-unadjusted-ula` | easy | Convert biased unadjusted Langevin (ULA) into an exact Metropolis-adjusted kernel | `reward ∈ {0, 1}` |

### Category 2 — Open-Ended Optimization vs. Frozen Baseline (Continuous Speedup, 45–60 min)
KernelBench-style optimization tasks. Agent starts from a frozen baseline and maximizes dual-horizon (`H=150/1000`, `dims=6,20`) verified efficiency on **hidden held-out verification seeds** (unseen during agent dev runs). Any invariance or replay failure scores **`0.0`**.

| Task | Lane | Frozen Baseline | Reward |
|---|---|---|---|
| `a0-ground-up` | `A0` | 43-LOC isotropic RWM stub | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a1-preconditioning` | `A1` | Diagonal dual-averaging HMC (`adapt_metric`) | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a2-funnel-dynamics` | `A2` | Diagonal HMC starter | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a3-integrator-trajectory` | `A3` | Diagonal HMC starter | `soft_min_cand / soft_min_base` (0 if invariance fails) |
| `a4-nonlinear-transport` | `A4` | Diagonal affine transport starter | `soft_min_cand / soft_min_base` (0 if invariance/FD fails) |

---

## Running with Harbor

Build the shared base image once from repo root, then run any task or the full suite with Harbor:

```bash
# 1. Build shared base image
docker build -f harbor/base.Dockerfile -t apin-harbor-base:latest .

# 2. Run one task or the full suite via Harbor CLI
harbor run -p harbor/tasks/a1-preconditioning -a claude-code -m anthropic/claude-sonnet-4-5
harbor run -p harbor/tasks -a codex -m openai/gpt-5
```

## Running the Verifier Locally (No Docker Required)

Every `tests/test.sh` delegates to [harbor/verify.py](verify.py), which you can also invoke directly:

```bash
# Rewind task verification
uv run python harbor/verify.py rewind \
  --task-id s3_position_gated_mh_bias \
  --submission autoresearch/lanes/S3/fixtures/position_gated_ref.py \
  --out-dir /tmp/harbor-check

# Optimization task verification (paired hidden seeds vs frozen baseline)
uv run python harbor/verify.py optimize \
  --lane A1 \
  --submission autoresearch/lanes/A1/submission.py \
  --baseline harbor/tasks/a1-preconditioning/environment/baseline.py \
  --dims 6,20 --repeats 2 \
  --out-dir /tmp/harbor-opt-check
```

Outputs Harbor-standard `/logs/verifier/reward.txt` (scalar float reward) and `/logs/verifier/reward.json` (structured breakdown: `reward`, `passed`, `speedup`, `soft_min_eff`, `baseline_soft_min`, `b2_max`, `overhead_ratio`).
