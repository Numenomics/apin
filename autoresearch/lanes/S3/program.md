# Autoresearch Lane S3: Rewind Calibration & Agent Grading (`program.md`)

> **Calibration Lane S3:** Rewind tasks where `base` fails a hidden invariance/efficiency check and `reference` passes.
> Grades agent configurations, models, and harnesses by pass rate and **AUARC** (Area Under Accuracy–Resource Curve).

## Calibration Check

Verify that all `base` snapshots fail and all `reference` snapshots pass:

```bash
uv run apin autoresearch eval-s3 --mode base
uv run apin autoresearch eval-s3 --mode reference
```

## Contributing a Fix or New Rewind Task

Inspect `autoresearch/lanes/S3/tasks.json` and each task's broken `base_submission` under `autoresearch/lanes/S3/fixtures/` (without peeking at `*_ref.py` when diagnosing a bug yourself). Write your repaired sampler(s) under `autoresearch/lanes/S3/runs/<RUN_ID>/` and grade them:

```bash
uv run apin autoresearch eval-s3 --mode custom --run-id <RUN_ID> \
  --submission s3_position_gated_mh_bias=autoresearch/lanes/S3/runs/<RUN_ID>/fix_gated.py \
  --submission s3_missing_jacobian_transport=autoresearch/lanes/S3/runs/<RUN_ID>/fix_jacobian.py \
  --submission s3_unadjusted_ula_bias=autoresearch/lanes/S3/runs/<RUN_ID>/fix_ula.py
```

Share your AUARC / pass-rate report, root-cause diagnostics, or newly proposed rewind tasks on the Flywheel via:

```bash
uv run apin autoresearch note add --kind insight --lane S3 --title "<summary>" --body "<results and root causes>"
```
