"""Adapter discovery, argument building and output parsing, without running samplers."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from harness import cmdstan, models, sampler
from harness.adapters import nutpie as nutpie_adapter
from harness.adapters import pathfinder, walnutpie


def test_registry_lists_every_adapter():
    assert {"cmdstan", "pathfinder", "nutpie", "walnutpie"} <= set(sampler.registry())


def test_manifest_and_shipped_data():
    ms = models.load()
    assert len(ms) == 21
    for m in ms.values():
        assert m.stan_file.is_file()
        if m.data_shipped:
            assert m.data_file(fetch=False).is_file()


def test_cmdstan_path_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("CMDSTAN", raising=False)
    with pytest.raises(cmdstan.CmdStanError, match="--cmdstan"):
        cmdstan.locate()
    monkeypatch.setenv("CMDSTAN", str(tmp_path))
    with pytest.raises(cmdstan.CmdStanError, match="makefile"):
        cmdstan.locate()


def test_sample_argv():
    argv = cmdstan.sample_argv(Path("/m/x"), Path("/d.json"), 42, 2, Path("/o/chain_2.csv"), 10, 20,
                               init=Path("/i.json"), extra=["adapt", "delta=0.9"])
    assert argv == ["/m/x", "id=3", "data", "file=/d.json", "random", "seed=42", "init=/i.json", "output",
                    "file=/o/chain_2.csv", "method=sample", "num_warmup=10", "num_samples=20", "save_warmup=0",
                    "adapt", "delta=0.9"]


def test_read_cmdstan_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("# model = x\nlp__,n_leapfrog__,theta.1\n-1,3,0.5\n-2,7,0.25\n"
                 "#  Elapsed Time: 0.012 seconds (Warm-up)\n#                0.034 seconds (Sampling)\n"
                 "#                0.046 seconds (Total)\n")
    f = cmdstan.read_csv(p)
    assert (f.warmup_s, f.sampling_s, f.total_s) == (0.012, 0.034, 0.046)
    assert f.column("n_leapfrog__").sum() == 10


def test_unflatten_for_pathfinder_inits():
    flat = {"lp__": 0.0, "mu": 1.0, "v.1": 2.0, "v.2": 3.0, "m.1.1": 4.0, "m.2.1": 5.0, "m.1.2": 6.0,
            "m.2.2": 7.0, "tp": 9.0}
    assert pathfinder.unflatten(flat, ["mu", "v", "m"]) == {"mu": 1.0, "v": [2.0, 3.0], "m": [[4.0, 6.0], [5.0, 7.0]]}


def test_walnutpie_stanzas_and_argv(tmp_path):
    log = ("warmup\ntotal time: 1.5s\nlogp_grad time: 1.0s\nlogp_grad fraction: 0.66\n  logp_grad calls: 300\n"
           "  time per call: 3.3e-3s\nsampling\ntotal time: 2.5s\nlogp_grad time: 2.0s\nlogp_grad fraction: 0.8\n"
           "  logp_grad calls: 700\n  time per call: 2.8e-3s\n")
    w, s = walnutpie.parse_stanzas(log)
    assert (w["total_s"], w["grad_calls"], s["total_s"], s["grad_calls"]) == (1.5, 300, 2.5, 700)
    cli = tmp_path / "stan_cli"
    a = walnutpie.Adapter(sampler.Settings(options={"cli": str(cli), "flags": "--metric-window 50"}))
    req = sampler.RunRequest(models.load()["blr"], Path("/d.json"), 100, 4, 10, 20, tmp_path)
    assert a.argv(Path("/m.so"), req, 3) == [str(cli), "/m.so", "/d.json", "--seed", "103", "--warmup",
                                             "10", "--samples", "20", "--output", str(tmp_path / "chain_3.csv"),
                                             "--metric-window", "50"]


def test_walnutpie_check_reports_missing_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("WALNUTPIE_CLI", raising=False)
    a = walnutpie.Adapter(sampler.Settings(options={"cli": str(tmp_path / "nope"), "bridgestan": str(tmp_path)}))
    with pytest.raises(RuntimeError, match="stan_cli not found"):
        a.check()


def test_nutpie_flatten():
    class DA:
        def __init__(self, v):
            self.values = v

    class Post:
        data_vars = {"mu": DA(np.zeros((2, 3))), "b": DA(np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2))}

    cols, chains = nutpie_adapter.flatten_posterior(Post())
    assert cols == ["mu", "b.1", "b.2"]
    assert chains[1].shape == (3, 3)
    assert list(chains[1][0]) == [0.0, 6.0, 7.0]


def test_nutpie_check_explains_missing_extra():
    try:
        import nutpie  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="extra nutpie"):
            nutpie_adapter.Adapter(sampler.Settings()).check()
    else:
        pytest.skip("nutpie is installed")
