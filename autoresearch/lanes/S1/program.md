# Autoresearch Lane S1: Upstream Sampler Patches (`program.md`)

> **Systems Lane S1 (`ARCHITECTURE.md` §2.2):** Autonomous patches to upstream sampler modules (`upstream/walnutpie`, `upstream/stan`, `upstream/nuts-rs`).
> - **Guardrails:** `allowed_paths` enforced via `git diff --name-only`; upstream unit tests + `apin` statistical gate must pass.
> - **Upstream policy:** Agents push ONLY to the owner's fork branches and NEVER post issues or PRs to `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr` (`upstream/STATUS.md`).

## Evaluation

```bash
uv run apin autoresearch eval-s1 \
  --upstream walnutpie \
  --allowed-path src/ \
  --build-cmd "cmake --build build -j" \
  --test-cmd "ctest --test-dir build --output-on-failure" \
  --bench-cmd "uv run apin run walnutpie --models blr,eight_schools_noncentered --reps 3 --label s1_cand" \
  --run-id s1_dev \
  --hypothesis "..."
```

Use `--dry-run` first to verify `allowed_paths` and clean patch hashing before building.
