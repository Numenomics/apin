#!/usr/bin/env bash
set -euo pipefail
cd /app
/app/.venv/bin/python /tests/verify.py rewind \
  --task-id s3_position_gated_mh_bias \
  --tasks /tests/tasks.json \
  --submission /app/submission.py \
  --out-dir /logs/verifier
