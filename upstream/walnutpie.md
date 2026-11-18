# walnutpie

C++ implementation of the WALNUTS sampler, built with CMake. Fork: sims1253/walnutpie, branch `exp/safe-adapt-defaults`. Licence: MIT.

The pin 43b6435 is the integration branch that the W-36 to W-133 measurements used. The fork has 47 pull requests, eight of them merged upstream; see STATUS.md.

```bash
git submodule update --init upstream/walnutpie
cmake -S upstream/walnutpie -B upstream/walnutpie/build -DCMAKE_BUILD_TYPE=Release
cmake --build upstream/walnutpie/build -j4 --target stan_cli
```

CMake downloads Eigen at configure time. The BridgeStan header is in the checkout (`thirdparty/bridgestan`), so `stan_cli` builds without the bridgestan submodule, but it runs only on a model library built by BridgeStan ([bridgestan.md](bridgestan.md)).

The adapter finds the binary at `upstream/walnutpie/build/examples/stan_cli`, or at `$WALNUTPIE_CLI`, or at `--opt cli=<path>`. Pass extra sampler flags with `--opt flags="--metric-window 50"`.

After an edit to a header, rebuild with `--clean-first`. Incremental builds in the campaign skipped header changes more than once.
