#!/usr/bin/env bash
set -euo pipefail
cd /app
/app/.venv/bin/python /tests/verify.py rewind \
  --task-id s3_missing_jacobian_transport \
  --tasks /tests/tasks.json \
  --submission /app/autoresearch/lanes/A4/submission.py \
  --out-dir /logs/verifier
