"""Reference summaries: names, statistics, storage, and the manifest's record of sources."""

import json

import numpy as np
import pytest

from harness import models, reference


def test_stan_name():
    assert reference.stan_name("theta[1]") == "theta.1"
    assert reference.stan_name("L[2,1]") == "L.2.1"
    assert reference.stan_name("L[2, 1]") == "L.2.1"
    assert reference.stan_name("sigma") == "sigma"


def test_summarise_moments_and_mcse():
    rng = np.random.default_rng(0)
    x = 2.0 + 3.0 * rng.standard_normal((4000, 4))
    names, st = reference.summarise({"a": x, "const": np.ones((4000, 4))})
    assert names == ["a"]  # constant variables are left out
    assert st["mean"][0] == pytest.approx(2.0, abs=0.15)
    assert st["sd"][0] == pytest.approx(3.0, rel=0.03)
    assert st["sq_mean"][0] == pytest.approx(13.0, rel=0.05)
    assert st["var"][0] == pytest.approx(9.0, rel=0.05)
    assert st["var_sd"][0] == pytest.approx(9.0 * 2**0.5, rel=0.06)  # sd of (x-mu)^2 for a normal is sqrt(2) sigma^2
    assert st["mcse_mean"][0] == pytest.approx(3.0 / np.sqrt(16000), rel=0.1)


def test_json_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "REF_DIR", tmp_path)
    names, st = reference.summarise({"a": np.arange(40.0).reshape(10, 4) % 7})
    ref = reference.Reference("m", "local", names, st, {"k": 1})
    reference.path("m").write_text(ref.to_json())
    back = reference.load("m")
    assert back.names == ["a"] and back.source == "local"
    assert back.stats["mean"] == pytest.approx(st["mean"], rel=1e-8)
    assert reference.load("absent") is None


def test_manifest_records_a_source_for_every_model():
    raw = json.loads((models.ROOT / "models" / "manifest.json").read_text())
    for e in raw:
        r = e["reference"]
        assert r["source"] in {"posteriordb", "local", "none"}
        assert (r["posteriordb"] is not None) == (r["source"] == "posteriordb")
        assert models.load()[e["model"]].reference == r["source"]


def test_shipped_references_are_small_and_match_the_manifest():
    ms = models.load()
    for p in reference.REF_DIR.glob("*.json"):
        assert p.stat().st_size < reference.MAX_BYTES
        ref = reference.load(p.stem)
        assert ref.source == ms[p.stem].reference
        assert ref.names and all(np.all(ref.stats["sd"] > 0) for _ in [0])
        assert all(np.all(np.isfinite(ref.stats[k])) for k in reference.STATS)
