#!/usr/bin/env bash
set -euo pipefail
cd /app
uv run python /app/harbor/verify.py optimize \
  --lane A0 \
  --submission /app/autoresearch/lanes/A0/submission.py \
  --baseline /app/harbor_baseline.py \
  --dims 6,20 --repeats 2 \
  --out-dir /logs/verifier
