# Record Template (`examples/record_template`)

Copy this directory to `records/<release>/<track>/<date>_<slug>/` when submitting a verified result by pull request.

## Checklist

1. Replace `submission.py` with your frozen candidate file (no further edits after scoring).
2. Update `submission.json` (`entry_id`, `author`, `date`, `track`, `capabilities_requested`, `kernel_class`, `aux_spec_summary`, `reproduce_cmd`).
3. Pin the exact commit SHA in `pins.toml` (`git rev-parse HEAD`).
4. Complete `invariance.md` for all adjusted tracks (`L1`, `L2`, `L3`, `L5`).
5. Validate before opening a PR:
   ```bash
   uv run apin records validate records/<release>/<track>/<date>_<slug>
   ```
6. Report results conservatively: say *"passed the 0.1-SD bench gate at this budget"*, never *"is correct"*, *"exact"*, *"unbiased"*, or *"proven"*.
