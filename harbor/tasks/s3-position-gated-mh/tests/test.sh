#!/usr/bin/env bash
set -euo pipefail
cd /app
uv run python /app/harbor/verify.py rewind \
  --task-id s3_position_gated_mh_bias \
  --submission /app/submission.py \
  --out-dir /logs/verifier
