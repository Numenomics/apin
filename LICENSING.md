# Licensing

Status: proposed. The owner has not confirmed this proposal. Until they do, treat every statement below as a draft.

## apin's own files

The harness, tests, documents and the measurement records under `results/` are offered under the BSD-3-Clause licence in `LICENSE`. BSD-3-Clause is the licence of Stan, Stan Math, stanc3, CmdStan and BridgeStan, so code can move from apin into those projects without a licence change. The copyright line in `LICENSE` says "the apin authors" until the owner names the holder.

## Submodules

Each submodule under `upstream/` keeps its own licence. apin does not relicense any of them. The licences below were read from each project's licence file at the pinned commit (or, for posteriordb, from the database's info files).

| upstream | licence | checked |
|---|---|---|
| stan-dev/math | BSD-3-Clause | LICENSE.md at e355b14 |
| stan-dev/stan | BSD-3-Clause | LICENSE.md at d13c50c |
| stan-dev/stanc3 | BSD-3-Clause | LICENSE.txt at d2081e6 |
| stan-dev/cmdstan | BSD-3-Clause | LICENSE at 6380837 |
| roualdes/bridgestan | BSD-3-Clause for code; CC BY 4.0 for documentation | LICENSE-CODE and LICENSE-DOC at 16e7b3e |
| flatironinstitute/walnutpie | MIT | LICENSE at 43b6435 |
| stan-dev/posteriordb | BSD-3-Clause for the database code; each model info file has its own `licence` field | LICENCE.md and `posterior_database/models/info/*.info.json` at 28f8d3d |
| pymc-devs/nutpie (Python dependency, not a submodule) | MIT | PyPI metadata for 0.16.11 |

Stan Math bundles third-party libraries (Eigen 5.0.1 under MPL-2.0, Boost 1.87.0 under BSL-1.0, SUNDIALS 6.1.1 under BSD-3-Clause, TBB 2020.3 under Apache-2.0; see `licenses/` in the math checkout). They reach apin only through the submodule and keep their own terms.

## Translated code in apin

`harness/ess.py` is a line-by-line translation of the ESS and R-hat functions of the R package posterior 1.7.0 (stan-dev/posterior). posterior is BSD-3-Clause, copyright 2021 the posterior package authors, Stan Developers and their Assignees, and the Trustees of Columbia University (checked in its DESCRIPTION and LICENSE files). The file states this in its docstring and keeps that copyright line. The licences match, so the file can stay in a BSD-3-Clause repository.

## Patches

A patch to an upstream is offered under that upstream's licence: BSD-3-Clause for the Stan projects and BridgeStan, MIT for walnutpie. A fork pull request states this in its body. The owner sends each patch personally (see `upstream/STATUS.md`).

An algorithm or idea taken from another project is reimplemented from its paper or its behaviour and credited in the ledger; code translated line by line from another project carries that project's licence, which matters when the source is Apache-2.0 or GPL and the destination upstream is BSD-3-Clause.

## Builds

apin does not ship binaries now. If it ever ships a build of a patched upstream (a CmdStan tree, a BridgeStan model library, a `stan_cli`), the build:

- carries the licence and notice files of every upstream it contains, including the third-party libraries inside Stan Math;
- is named unofficial, for example "CmdStan 2.39.0 with apin patches, unofficial", and does not use the Stan name or logo in a way that suggests the Stan project made it.

## Models and data

The 21 `.stan` files in `models/` come from posteriordb. Each model's info file at commit 28f8d3d gives `"licence": "BSD3"`, and `models/manifest.json` records the posteriordb posterior each one belongs to.

posteriordb data info files have no licence field. Some data sets are old textbook or public-domain data; others came from papers. `data/SOURCES.md` lists each file with its origin. apin ships nine small data files in `data/` and fetches the other twelve at a pinned commit with a sha256 check. Owner decision: confirm that shipping the nine is acceptable, or move them to fetched as well (one flag per file in `models/manifest.json`).
