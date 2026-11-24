# Improve the warmup metric of a fixed HMC kernel

The file `/app/autoresearch/lanes/A1/submission.py` defines `adapt_metric(oracle, budget, rng)`, which returns a step size, a mass matrix and a starting point. The judge in `harness/judge.py` owns the HMC kernel that samples with them. Make the metric and the warmup better, so that the fixed kernel mixes faster on rotated Gaussians, funnels, bananas, hybrid Rosenbrock ridges and heavy tails in 6 and 20 dimensions. A warmup that costs more gradient calls is charged for them.

## Rules

- Edit only `/app/autoresearch/lanes/A1/submission.py`.
- Do not change `harness/`, `models/`, `data/` or any other file in `/app/autoresearch/lanes/A1/`.
- Use only the log density and gradient that the oracle gives you. Do not try to identify the target.
- The sampler must leave the target distribution exactly invariant. A biased kernel fails the checks.
- Read `/app/autoresearch/lanes/A1/program.md`, `/app/autoresearch/RULINGS.md` and `/app/codex/INDEX.yaml` before you start.

## How to measure

Run this after every change:

```bash
uv run apin autoresearch eval --lane A1 --run-id harbor_dev --no-queue --hypothesis "<one sentence>"
```

It prints `METRIC speedup=<x>`, the speed-up of your file over the starting code on public seeds. The unchanged file scores 1.0. A difference under 5% is noise. The tool says whether a change beat your best kept version by at least 5% and keeps that version in `/app/autoresearch/lanes/A1/runs/harbor_dev/incumbent.py`.

The score is a soft minimum over the target classes, so the weakest class sets it. A large gain on an easy class can vanish behind a small loss on the hardest one. Check the `CLASS_METRIC` lines after every run and work on the lowest class.

## How you are graded

When you finish, the same grader scores the file left at `/app/autoresearch/lanes/A1/submission.py`, with hidden seeds in place of the public ones. Tuning to the public seeds will not carry over. The hidden seeds draw different targets, so expect the two speed-ups to differ by about a tenth. A sampler that fails a validity check scores zero, so the last file you leave in place must be one the tool reported as valid. The leaderboard ranks your speed-up over the starting code. The reward is your file's score divided by the score of a Stan-style reference sampler on the same hidden seeds, or 0 if the file fails a check.

There is no time or token limit to save. Keep trying new ideas until several different ideas in a row bring no gain. Then stop with the best valid version in place.
