# posteriordb

Stan models, data sets and reference posterior draws, maintained by stan-dev. Licence: BSD-3-Clause for the database; each model's info file has a `licence` field (BSD3 for all 21 core models); data info files have no licence field. apin carries no patch to it, so the submodule points at stan-dev/posteriordb directly.

The submodule is optional. `apin fetch-data` reads data from it when it is checked out and downloads the same files from GitHub at the pinned commit when it is not. Either way the harness checks each file's sha256 against `models/manifest.json`.

```bash
git submodule update --init upstream/posteriordb
```

posteriordb also has reference posteriors for some models. A future score that measures agreement with the true posterior, not only ESS, would read them from here.
