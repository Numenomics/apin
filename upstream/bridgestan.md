# bridgestan

Compiles a Stan model into a shared library with a C interface, and ships bindings for Python, Julia, R and Rust. Fork: sims1253/bridgestan, branch `isa-option`, which adds a build switch for the instruction set (the W-105 AVX2 builds). Licence: BSD-3-Clause for code, CC BY 4.0 for documentation.

The walnutpie adapter needs it, because walnutpie's `stan_cli` loads a BridgeStan model library. Any sampler written outside C++ can reach the same model libraries through one of the bindings.

```bash
git submodule update --init upstream/bridgestan
git -C upstream/bridgestan submodule update --init --recursive stan
export BRIDGESTAN=$PWD/upstream/bridgestan
```

Use BridgeStan's own nested `stan` submodule (do **not** override `MATH=` / `STAN=` to the top-level `upstream/math` and `upstream/stan` pins, which are on different Stan/Eigen revisions and fail to compile together). If you need custom `make/local` flags (e.g. `STANC=...` or ISA flags), run `mkdir -p upstream/bridgestan/make` first because `make/` does not exist in a fresh checkout.

The walnutpie adapter builds each model with `make -C $BRIDGESTAN <dir>/<model>_model.so` and caches it under `build/` by source and flags. Note that `make clean` removes `bin/stanc` (which re-downloads on the next build).

Two hazards cost time in the campaign:

- After a change to math or the flags, delete `upstream/bridgestan/src/bridgestan.o`. BridgeStan reuses it otherwise.
- A model library built without `STAN_THREADS=true` is unsafe when several threads call it. Upstream (roualdes/bridgestan#336) documents this as intended. Build with `STAN_THREADS=true` for any sampler that evaluates chains in threads.
