# bridgestan

Compiles a Stan model into a shared library with a C interface, and ships bindings for Python, Julia, R and Rust. Fork: sims1253/bridgestan, branch `isa-option`, which adds a build switch for the instruction set (the W-105 AVX2 builds). Licence: BSD-3-Clause for code, CC BY 4.0 for documentation.

The walnutpie adapter needs it, because walnutpie's `stan_cli` loads a BridgeStan model library. Any sampler written outside C++ can reach the same model libraries through one of the bindings.

```bash
git submodule update --init upstream/bridgestan upstream/math upstream/stanc3
cat > upstream/bridgestan/make/local <<MK
MATH=$PWD/upstream/math/
STANC=$PWD/upstream/stanc3/_build/default/src/stanc/stanc.exe
MK
export BRIDGESTAN=$PWD/upstream/bridgestan
```

Build stanc3 first ([stanc3.md](stanc3.md)). The walnutpie adapter builds each model with `make -C $BRIDGESTAN <dir>/<model>_model.so` and caches it under `build/` by source and flags.

Two hazards cost time in the campaign:

- After a change to math or the flags, delete `upstream/bridgestan/src/bridgestan.o`. BridgeStan reuses it otherwise.
- A model library built without `STAN_THREADS=true` is unsafe when several threads call it. Upstream (roualdes/bridgestan#336) documents this as intended. Build with `STAN_THREADS=true` for any sampler that evaluates chains in threads.
