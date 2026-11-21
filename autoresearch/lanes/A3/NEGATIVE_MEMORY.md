# Negative Result Memory (`autoresearch/lanes/A3/NEGATIVE_MEMORY.md`)

Pre-seeded from `autoresearch/lanes/A1/NEGATIVE_MEMORY.md` (`NM-001`..`NM-005`):
- **NM-001:** Naive single-chain windowed warmup early-exit degrades marginal models.
- **NM-003:** Naive late-warmup truncation harms steady-state sampling unless replaced by a faster-converging estimator.
- **NM-004:** ML step-size schedules (Adam / Prodigy / D-Adaptation) oscillate and underperform dual averaging.

Append new Lane A3 falsifications (`NM-A3-001`, ...) after $\ge 2$ disciplined trials fail cleanly.
