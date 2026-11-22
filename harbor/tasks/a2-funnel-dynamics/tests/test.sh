#!/usr/bin/env bash
set -euo pipefail
cd /app
/app/.venv/bin/python /tests/verify.py optimize \
  --lane A2 \
  --submission /app/autoresearch/lanes/A2/submission.py \
  --baseline /tests/baseline.py \
  --dims 6,20 --repeats 2 \
  --out-dir /logs/verifier
