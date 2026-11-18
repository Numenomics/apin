# stanc3

The Stan compiler, written in OCaml and built with dune. Fork: sims1253/stanc3, pinned on branch `gathered-pcm-emit`, which carries the gathered-family emission. Licence: BSD-3-Clause.

Install the OCaml tool chain once, from inside the checkout:

```bash
git submodule update --init upstream/stanc3
cd upstream/stanc3
bash scripts/install_build_deps.sh
eval $(opam env)
dune build @install
```

The compiler is then `upstream/stanc3/_build/default/src/stanc/stanc.exe`. CmdStan builds it itself when `STANC3=../stanc3` is in its `make/local`. BridgeStan takes it through `STANC=` (see [bridgestan.md](bridgestan.md)).

The gathered-family passes run at `--O1`. Pass compiler flags to a run with `--opt stancflags=--O1`. Compiler tests: `dune runtest`.
