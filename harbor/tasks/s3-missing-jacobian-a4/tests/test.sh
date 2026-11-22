#!/usr/bin/env bash
set -euo pipefail
cd /app
uv run python /app/harbor/verify.py rewind \
  --task-id s3_missing_jacobian_transport \
  --submission /app/autoresearch/lanes/A4/submission.py \
  --out-dir /logs/verifier
