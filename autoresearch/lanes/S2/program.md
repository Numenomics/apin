# Autoresearch Lane S2: Gradient & Compiler Speed (`program.md`)

> **Systems Lane S2 (`RULINGS.md` R-008):** Autonomous gradient, autodiff, and compiler optimizations in `upstream/math`, `upstream/stan`, `upstream/stanc3`, and `upstream/bridgestan`.
> - **Exact-math waiver (R-008):** Statistical parity testing is waived *only* when draws at fixed seed are **byte-identical** (`check_byte_identical_draws` SHA-256 parity). Any non-byte-identical numerical change (e.g. `-ffast-math`, reordered reductions) must pass the full `apin` statistical gate.
> - **Hard ban on mixed builds (`NM-A1-002` / `W-27`):** Any ISA/SIMD flag experiment must clean-rebuild ALL translation units from scratch and pass finite-difference gradient checks (`max_rel_err < 1e-5`).

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

Use `--dry-run` first to verify `allowed_paths` and patch digest.
