#!/usr/bin/env bash
# Runs in a clean verifier container (task.toml: [verifier] environment_mode = "separate").
# Harbor restores only the agent's submission file, at the path below.
set -euo pipefail
cd /app
exec /app/.venv/bin/python -I /tests/verify.py rewind \
  --task-id s3_unadjusted_ula_bias \
  --tasks /tests/tasks.json \
  --submission /app/submission.py \
  --out-dir /logs/verifier
