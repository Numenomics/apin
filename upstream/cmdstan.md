# cmdstan

C++ command-line driver for Stan, built with make. Fork: sims1253/cmdstan. Licence: BSD-3-Clause.

apin builds CmdStan from the sibling submodules, so that a patch to stan, math or stanc3 reaches the binary without copying files.

```bash
git submodule update --init upstream/cmdstan upstream/stan upstream/math upstream/stanc3
cat > upstream/cmdstan/make/local <<'MK'
STAN=../stan/
MATH=../math/
STANC3=../stanc3
MK
make -C upstream/cmdstan -j4 build
export CMDSTAN=$PWD/upstream/cmdstan
```

Do not initialise the nested submodules inside `upstream/cmdstan` (`stan`, and `lib/stan_math` inside it). `make/local` points past them.

With `STANC3` set, CmdStan builds `bin/stanc` from the stanc3 checkout with `dune build @install`, so the stanc3 build tools must be installed first (see [stanc3.md](stanc3.md)).

After you change compiler, flags or any submodule commit, run `make -C upstream/cmdstan clean-all` before the next build. CmdStan reuses `main.o` and the precompiled header across such changes and can link objects from two builds into one binary. Fork PR sims1253/cmdstan#1 adds a guard against this. The harness keys its model cache on the CmdStan version and flags, but it cannot see a stale `main.o`.

A stock CmdStan release (for example 2.39.0 from the Stan releases page) also works: set `CMDSTAN` to its directory. Use one as the baseline when you measure a patch.
