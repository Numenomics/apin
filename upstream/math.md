# math

The Stan Math library: C++ automatic differentiation and the probability functions. Fork: sims1253/math. Licence: BSD-3-Clause.

math is header-only apart from the TBB and SUNDIALS libraries, which CmdStan and BridgeStan build from inside it. It reaches the two builds through make variables:

- CmdStan: `MATH=../math/` in `upstream/cmdstan/make/local` ([cmdstan.md](cmdstan.md)).
- BridgeStan: `MATH=<absolute path>/upstream/math/` in `upstream/bridgestan/make/local` ([bridgestan.md](bridgestan.md)).

Unit tests run from the math checkout with its own runner, for example:

```bash
cd upstream/math
python3 runTests.py -j4 test/unit/math/rev/fun/eigenvectors_sym_test.cpp
make -j4 test-headers
```

The pin e355b14 is the commit the W-133 benchmark measured. It is no longer on a fork branch (STATUS.md, note 1).
