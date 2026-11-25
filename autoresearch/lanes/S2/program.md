# Autoresearch Lane S2: Gradient & Compiler Speed (`program.md`)

> **Systems Lane S2 (`RULINGS.md` R-009):** Autonomous gradient, autodiff, and compiler optimizations in `upstream/math`, `upstream/stan`, `upstream/stanc3`, and `upstream/bridgestan`.
> - **Exact-math waiver (R-009):** Statistical parity testing is waived *only* when draws at fixed seed are **byte-identical** (`check_byte_identical_draws` SHA-256 parity). Any non-byte-identical numerical change (e.g. `-ffast-math`, reordered reductions) must pass the full `apin` statistical gate.
> - **Hard ban on mixed builds (`NM-S2-001` / `W-27`):** Any ISA/SIMD flag experiment must clean-rebuild ALL translation units from scratch and pass finite-difference gradient checks (`max_rel_err < 1e-5`).
> - **Never post upstream** (`AGENTS.md` Rule 1): do not open PRs, issues, or comments on `stan-dev/*`, `flatironinstitute/*`, `pymc-devs/*`, or `cmdstanr`.

## Setup

Upstream repos live as git submodules under `upstream/`. Read `upstream/STATUS.md`, `upstream/CANDIDATES.md`, and the target repo's `.md` note (e.g. `upstream/math.md`), then initialize only the submodule(s) you need:

```bash
git submodule update --init upstream/math
```

## Evaluation

```bash
uv run apin autoresearch eval-s2 \
  --upstream math \
  --allowed-path stan/math/rev/ \
  --clean-build-cmd "make clean-all && make -j4" \
  --test-cmd "./runTests.py test/unit/math/rev/fun/..." \
  --run-id s2_dev \
  --hypothesis "..."
```

Use `--dry-run` first to verify `allowed_paths` and patch digest. `eval-s2` writes its ledger locally under `autoresearch/lanes/S2/runs/<run-id>/`; share validated findings, timings, and open hypotheses on the Flywheel with:

```bash
uv run apin autoresearch note add --kind insight --lane S2 --title "<what you found>" --body "<evidence>"
```
