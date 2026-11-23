#!/usr/bin/env bash
# Runs in a clean verifier container (task.toml: [verifier] environment_mode = "separate").
# Harbor restores only the agent's submission file, at the path below; the harness and the
# interpreter come from the image, the denominators and frozen scores from /tests.
set -euo pipefail
cd /app
exec /app/.venv/bin/python -I /tests/verify.py optimize \
  --lane A0 \
  --submission /app/autoresearch/lanes/A0/submission.py \
  --tests-dir /tests \
  --out-dir /logs/verifier
