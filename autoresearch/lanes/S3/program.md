# Autoresearch Lane S3: Rewind Calibration & Agent Grading (`program.md`)

> **Calibration Lane S3 (`ARCHITECTURE.md` §2.2):** Rewind tasks where `base` fails a hidden invariance/efficiency check and `reference` passes.
> Grades agent configurations, models, and harnesses by pass rate and **AUARC** (Area Under Accuracy–Resource Curve).

## Calibration Check

Verify that all `base` snapshots fail and all `reference` snapshots pass:

```bash
uv run apin autoresearch eval-s3 --mode base
uv run apin autoresearch eval-s3 --mode reference
```

Grade an agent's solved patches:

```bash
uv run apin autoresearch eval-s3 --mode custom --submission s3_position_gated_mh_bias=path/to/fix.py
```
