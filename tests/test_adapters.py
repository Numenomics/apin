"""Adapter discovery, argument building and output parsing, without running samplers."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from harness import cmdstan, models, sampler, score
from harness.adapters import cmdstan as cmdstan_adapter
from harness.adapters import nutpie as nutpie_adapter
from harness.adapters import pathfinder, pathfinder_only, walnutpie


def test_registry_lists_every_adapter():
    assert {"cmdstan", "pathfinder", "pathfinder_only", "nutpie", "walnutpie"} <= set(sampler.registry())


def test_parse_pathfinder_only_footer(tmp_path):
    p = tmp_path / "chain_0.csv"
    p.write_text(
        "lp_approx__,lp__,path__,x\n0.0,-1.0,1,0.5\n"
        "# Elapsed Time: 0.009000 seconds (Pathfinders)\n"
        "#               0.010000 seconds (PSIS)\n"
        "#               0.019000 seconds (Total)\n"
    )
    assert pathfinder_only.parse_pathfinder_time(p) == (0.009, 0.01, 0.019)


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
                    "file=/o/chain_2.csv", "method=sample", "num_warmup=10", "num_samples=20", "save_warmup=1",
                    "adapt", "delta=0.9"]


def test_read_cmdstan_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("# model = x\nlp__,n_leapfrog__,theta.1\n-1,3,0.5\n-2,7,0.25\n"
                 "#  Elapsed Time: 0.012 seconds (Warm-up)\n#                0.034 seconds (Sampling)\n"
                 "#                0.046 seconds (Total)\n")
    f = cmdstan.read_csv(p)
    assert (f.warmup_s, f.sampling_s, f.total_s) == (0.012, 0.034, 0.046)
    assert f.column("n_leapfrog__").sum() == 10


def test_read_csv_keeps_warmup_rows_apart(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("lp__,n_leapfrog__,x\n-1,7,0.1\n-1,15,0.2\n# Adaptation terminated\n# Step size = 0.5\n"
                 "-2,3,0.3\n-2,3,0.4\n-2,1,0.5\n")
    f = cmdstan.read_csv(p)
    assert f.values.shape == (3, 3) and f.warmup_values.shape == (2, 3)
    t = cmdstan_adapter.timing_from_csvs([p], 1.0)
    assert t.grads_sampling == 7 + 3  # one gradient per transition start, too
    assert t.grads_warmup is None and "step size" in t.uncounted  # no stepsize__ trace to rebuild
    q = tmp_path / "d.csv"
    q.write_text("lp__,n_leapfrog__,x\n-2,3,0.3\n")
    assert cmdstan.read_csv(q).warmup_values is None
    assert cmdstan_adapter.timing_from_csvs([q], 1.0).grads_warmup is None


def test_read_csv_keeps_only_the_columns_asked_for(tmp_path):
    """A wide model writes thousands of generated quantities the gate never reads; parsing only the
    columns it needs keeps memory to what it uses. Cut rows are still caught."""
    p = tmp_path / "c.csv"
    p.write_text("lp__,n_leapfrog__,x,y\n-1,7,0.1,9\n# Adaptation terminated\n-2,3,0.3,8\n-2,1,0.5,7\n")
    f = cmdstan.read_csv(p, columns=["x", "n_leapfrog__", "absent"])
    assert f.columns == ["n_leapfrog__", "x"] and f.values.shape == (2, 2) and f.warmup_values.shape == (1, 2)
    assert f.column("x").tolist() == [0.3, 0.5] and f.column("y") is None
    p.write_text("lp__,x\n-1,0.1\n-2\n")
    with pytest.raises(cmdstan.CmdStanError, match="fields"):
        cmdstan.read_csv(p, columns=["x"])


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


def _csv(path, thin=None, leapfrog=True):
    head = "# method = sample\n" + (f"#     thin = {thin}\n" if thin else "#     thin = 1 (Default)\n")
    cols = "lp__,n_leapfrog__,x" if leapfrog else "lp__,x"
    row = (lambda a, b: f"-1,{a},{b}") if leapfrog else (lambda a, b: f"-1,{b}")
    path.write_text(head + cols + "\n" + row(7, 0.1) + "\n# Adaptation terminated\n" + row(3, 0.3) + "\n")
    return path


def test_thinned_cmdstan_runs_have_no_evaluation_count(tmp_path):
    """S12: with thin=10 only every tenth transition's leapfrogs are written."""
    t = cmdstan_adapter.timing_from_csvs([_csv(tmp_path / "a.csv", thin=10)], 1.0)
    assert score.evaluation_counts(t.to_dict()) == (None, None, None) and "thin" in t.uncounted
    t1 = cmdstan_adapter.timing_from_csvs([FIXTURE], 1.0)
    assert score.evaluation_counts(t1.to_dict()) == (18111 + 128, 0, 18111 + 128) and t1.uncounted == ""


def test_a_csv_without_leapfrogs_has_no_evaluation_count(tmp_path):
    t = cmdstan_adapter.timing_from_csvs([_csv(tmp_path / "a.csv", leapfrog=False)], 1.0)
    assert score.evaluation_counts(t.to_dict())[0] is None


def test_pathfinder_cost_is_unavailable(tmp_path, monkeypatch):
    """S12: the NUTS leapfrogs leave out Pathfinder's own gradients."""
    from harness.sampler import Settings, Timing

    monkeypatch.setattr(cmdstan_adapter.Adapter, "run",
                        lambda self, req, inits=None: Timing(1.0, [1.0], [0.5], [0.5], 100, 50, 0))
    monkeypatch.setattr(cmdstan_adapter.Adapter, "build", lambda self, req: tmp_path / "exe")
    monkeypatch.setattr(cmdstan_adapter.Adapter, "root", tmp_path)
    monkeypatch.setattr(pathfinder.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stderr": ""})())
    monkeypatch.setattr(pathfinder.cmdstan, "parameter_names", lambda *a: ["x"])
    monkeypatch.setattr(pathfinder.cmdstan, "read_csv", lambda p: None)
    monkeypatch.setattr(pathfinder, "pathfinder_inits", lambda *a: [])
    req = type("Req", (), {"out_dir": tmp_path, "data_file": tmp_path / "d.json", "seed": 1, "chains": 1,
                           "model": type("M", (), {"stan_file": tmp_path / "m.stan"})()})()
    t = pathfinder.Adapter(Settings()).run(req)
    assert "pathfinder" in t.uncounted and score.evaluation_counts(t.to_dict())[0] is None


FIXTURE = Path(__file__).parent / "fixtures" / "nuts" / "blr_chain.csv"  # a CmdStan 2.39 chain, cut to 10 draws


def test_cmdstan_gradients_count_transition_starts_and_step_size_searches():
    """S12: n_leapfrog__ leaves out the gradient at the start of every transition (base_nuts.hpp)
    and the step-size searches (base_hmc.hpp init_stepsize), 2k + 4 gradients for k halvings or
    doublings: k = 13, 7, 0, 2, 0, 1 in this chain, rebuilt from stepsize__ and accept_stat__."""
    t = cmdstan_adapter.timing_from_csvs([FIXTURE], 1.0)
    assert t.uncounted == ""
    assert t.grads_warmup == 17041 + 1000 + 70 and t.grads_sampling == 118 + 10
    assert t.extra["step_size_searches"] == [[13, 7, 0, 2, 0, 1]]


def test_a_step_size_trace_that_does_not_rebuild_is_uncounted(tmp_path):
    lines = FIXTURE.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("lp__"))
    f = lines[start + 300].split(",")
    f[2] = str(float(f[2]) * 1.5)  # stepsize__ of one warmup row
    lines[start + 300] = ",".join(f)
    p = tmp_path / "c.csv"
    p.write_text("".join(lines))
    t = cmdstan_adapter.timing_from_csvs([p], 1.0)
    assert score.evaluation_counts(t.to_dict()) == (None, None, None) and "step size" in t.uncounted
