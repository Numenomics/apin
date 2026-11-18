# stan

C++ library with the samplers, I/O and services that CmdStan calls. Fork: sims1253/stan. Licence: BSD-3-Clause.

The pin d13c50c is a stan-dev/stan develop commit with no fork patch. The fork pull requests (scratch-hoist-base-nuts, gather-elimination) are on their own branches; see STATUS.md.

stan is header-only from apin's point of view. It has no separate build step: CmdStan includes it through `STAN=../stan/` in `upstream/cmdstan/make/local` (see [cmdstan.md](cmdstan.md)). Do not initialise its nested `lib/stan_math` submodule; `MATH=../math/` replaces it.

To measure a stan patch, check out the patch branch in `upstream/stan`, run `make -C upstream/cmdstan clean-all build`, and run the grid with a new `--label`.
