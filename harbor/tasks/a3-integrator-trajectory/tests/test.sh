#!/usr/bin/env bash
set -euo pipefail
cd /app
/app/.venv/bin/python /tests/verify.py optimize \
  --lane A3 \
  --submission /app/autoresearch/lanes/A3/submission.py \
  --baseline /tests/baseline.py \
  --dims 6,20 --repeats 2 \
  --out-dir /logs/verifier
