#!/usr/bin/env bash
set -euo pipefail
cd /app
uv run python /app/harbor/verify.py optimize \
  --lane A4 \
  --submission /app/autoresearch/lanes/A4/submission.py \
  --baseline /app/harbor_baseline.py \
  --dims 6,20 --repeats 2 \
  --out-dir /logs/verifier
