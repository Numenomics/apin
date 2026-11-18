"""nutpie (pymc-devs/nutpie), through its Stan frontend.

Needs the optional extra: `uv sync --extra nutpie`. nutpie compiles models
with its own bundled BridgeStan, so a patched Stan Math does not reach it.

nutpie reports no per-chain timings; chain_s is the wall time for each chain.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..sampler import RunRequest, Sampler, Timing


def _nutpie():
    try:
        import nutpie
    except ImportError as e:
        raise RuntimeError("nutpie is not installed; run `uv sync --extra nutpie`") from e
    return nutpie


class Adapter(Sampler):
    name = "nutpie"

    def check(self) -> None:
        _nutpie()

    def describe(self) -> dict:
        return {"nutpie": _nutpie().__version__}

    def run(self, req: RunRequest) -> Timing:
        nutpie = _nutpie()
        data = json.loads(req.data_file.read_text())
        compiled = nutpie.compile_stan_model(filename=str(req.model.stan_file)).with_data(**data)
        t0 = time.perf_counter()
        res = nutpie.sample(compiled, chains=req.chains, tune=req.warmup, draws=req.draws, seed=req.seed,
                            cores=req.chains, save_warmup=False, progress_bar=False)
        wall = time.perf_counter() - t0
        post, stats = res["posterior"], res["sample_stats"]
        columns, per_chain = flatten_posterior(post)
        n_steps = stats["n_steps"].values if "n_steps" in stats else None
        diverging = stats["diverging"].values if "diverging" in stats else None
        for c in range(req.chains):
            cols = columns + (["n_leapfrog__"] if n_steps is not None else [])
            mat = per_chain[c] if n_steps is None else np.column_stack([per_chain[c], n_steps[c]])
            write_csv(req.out_dir / f"chain_{c}.csv", cols, mat)
        return Timing(wall_s=wall, chain_s=[wall] * req.chains, warmup_s=[None] * req.chains,
                      sampling_s=[None] * req.chains,
                      grads_sampling=int(n_steps.sum()) if n_steps is not None else None, grads_warmup=None,
                      divergences=int(diverging.sum()) if diverging is not None else None,
                      extra={"chain_s_is_wall": True})


def flatten_posterior(post) -> tuple[list[str], list[np.ndarray]]:
    """xarray posterior (chain, draw, *dims) -> Stan-style columns and one
    (draws, columns) matrix per chain."""
    columns, blocks = [], []
    for name, da in post.data_vars.items():
        v = np.asarray(da.values)
        shape = v.shape[2:]
        if not shape:
            columns.append(name)
            blocks.append(v[:, :, None])
            continue
        for idx in np.ndindex(*shape):
            columns.append(name + "." + ".".join(str(i + 1) for i in idx))
        blocks.append(v.reshape(v.shape[0], v.shape[1], -1))
    stacked = np.concatenate(blocks, axis=2)
    return columns, [stacked[c] for c in range(stacked.shape[0])]


def write_csv(path, columns: list[str], values: np.ndarray) -> None:
    with open(path, "w") as fh:
        fh.write(",".join(columns) + "\n")
        for row in values:
            fh.write(",".join(repr(float(x)) for x in row) + "\n")
