"""Reference summaries: series, statistics, storage, and the manifest's record of sources."""

import json

import numpy as np
import pytest

from harness import models, reference


def test_stan_name():
    assert reference.stan_name("theta[1]") == "theta.1"
    assert reference.stan_name("L[2,1]") == "L.2.1"
    assert reference.stan_name("L[2, 1]") == "L.2.1"
    assert reference.stan_name("sigma") == "sigma"


def _normal_ref(rng, n=1000, chains=10, rho=0.0):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    x = rng.multivariate_normal([0, 0], cov, size=(n, chains))
    draws = {"a": 2.0 + 3.0 * x[..., 0], "b": x[..., 1], "const": np.ones((n, chains))}
    return draws, reference.summarise(draws, lp=-0.5 * (x[..., 0] ** 2 + x[..., 1] ** 2))


def test_summarise_series_and_moments():
    ref = _normal_ref(np.random.default_rng(0))[1]
    assert ref.names == ["a", "b"] and ref.constants == {"const": 1.0}
    assert ref.series == ["a", "b", "proj.1", "proj.2", "lp__"] and ref.has_lp
    assert ref.loc[0] == pytest.approx(2.0, abs=0.15) and ref.scale[0] == pytest.approx(3.0, rel=0.03)
    st = ref.stats
    assert st["m1"][:2] == pytest.approx(0, abs=1e-6) and st["sd1"][:2] == pytest.approx(1, abs=1e-6)
    assert st["sd2"][0] == pytest.approx(2**0.5, rel=0.06)  # sd of z^2 for a normal
    assert st["mcse1"][0] == pytest.approx(1 / np.sqrt(10000), rel=0.1)
    assert st["p05"] == pytest.approx(0.05, abs=1e-3) and st["q95"][0] == pytest.approx(1.645, abs=0.06)
    assert st["mcse_p05"][0] == pytest.approx(np.sqrt(0.05 * 0.95 / 10000), rel=0.15)


def test_projections_see_the_correlation():
    ref = _normal_ref(np.random.default_rng(1), rho=0.9)[1]
    u = ref.directions
    assert u @ u.T == pytest.approx(np.eye(2), abs=1e-3)
    assert np.abs(u[0]) == pytest.approx([0.7071, 0.7071], abs=1e-3)  # largest eigenvalue first
    # the projections are standardised on the second half of the chains: their SD there is 1
    assert ref.stats["sd1"][2:4] == pytest.approx(1, abs=1e-5)


def test_standardise_repeats_the_reference_arithmetic():
    draws, ref = _normal_ref(np.random.default_rng(2), rho=0.5)
    lp = -0.5 * (((draws["a"] - 2) / 3) ** 2 + draws["b"] ** 2)
    s = ref.standardise({**draws, "lp__": lp})
    assert s.shape == (5, 1000, 10)
    assert s[0].mean() == pytest.approx(ref.stats["m1"][0], abs=1e-6)
    assert (s[2][:, 5:] ** 2).mean() == pytest.approx(ref.stats["m2"][2], rel=1e-5)


def test_calibration_counts_functions():
    ref = _normal_ref(np.random.default_rng(3))[1]
    cal = ref.calibration
    assert cal["functions"] == 4 * 5
    # J <= tr(C^2) <= J^2; independent functions would give J
    assert cal["functions"] <= cal["trC2"] < cal["functions"] ** 2


def test_json_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "REF_DIR", tmp_path)
    ref = _normal_ref(np.random.default_rng(4), rho=0.3)[1]
    ref.model, ref.source = "m", "local"
    reference.path("m").write_text(ref.to_json())
    back = reference.load("m")
    assert back.names == ["a", "b"] and back.source == "local" and back.series == ref.series
    for k in reference.SERIES_STATS:
        assert back.stats[k] == pytest.approx(ref.stats[k], rel=1e-6, abs=1e-12)
    assert back.directions == pytest.approx(ref.directions) and back.constants == ref.constants
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
        assert ref.names and ref.has_lp, p.stem
        assert np.all(ref.scale > 0) and np.all(ref.stats["scale"] > 0)
        assert all(np.all(np.isfinite(ref.stats[k])) for k in reference.SERIES_STATS), p.stem
        assert ref.calibration["functions"] > 0
