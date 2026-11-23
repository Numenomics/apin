#!/usr/bin/env bash
# Runs in a clean verifier container (task.toml: [verifier] environment_mode = "separate").
# Harbor restores only the agent's submission file, at the path below.
set -euo pipefail
cd /app
exec /app/.venv/bin/python -I /tests/verify.py rewind \
  --task-id s3_missing_jacobian_transport \
  --tasks /tests/tasks.json \
  --submission /app/autoresearch/lanes/A4/submission.py \
  --out-dir /logs/verifier
