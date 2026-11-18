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
    reference.write(ref)
    back = reference.load("m")
    for k in ("mean", "mcse"):
        assert back.halves[k] == pytest.approx(ref.halves[k], rel=1e-3, abs=1e-12)
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
        assert reference.halves_path(p.stem).stat().st_size < reference.MAX_BYTES, p.stem
        ref = reference.load(p.stem)
        assert ref.halves["mean"].shape == (2, len(ref.series), len(reference.FUNCTIONS)), p.stem
        assert ref.source == ms[p.stem].reference
        assert ref.names and ref.has_lp, p.stem
        assert np.all(ref.scale > 0) and np.all(ref.stats["scale"] > 0)
        assert all(np.all(np.isfinite(ref.stats[k])) for k in reference.SERIES_STATS), p.stem
        assert ref.calibration["functions"] > 0


def test_a_model_can_raise_its_reference_settings_in_the_manifest():
    """A local reference is built with LOCAL, then the model's manifest settings, then the
    command line; so a rebuild from the repository repeats the reference that is shipped."""
    import dataclasses
    m = dataclasses.replace(models.load()["blr"], reference_settings={"samples": 20000, "adapt_delta": 0.9})
    s = reference.settings_for(m, {"adapt_delta": 0.99})
    assert s == {**reference.LOCAL, "samples": 20000, "adapt_delta": 0.99}
    assert reference.settings_for(models.load()["blr"]) == reference.LOCAL


def test_local_references_measure_projections_on_ten_chains():
    """The projections are chosen on half the chains and summarised on the other half. With 10
    chains that half was 5, and its error put two models' projections 3 to 5 MCSE away from
    independent runs, so a correct sampler could not be proven right (BENCH.md, rule 3)."""
    assert reference.LOCAL["chains"] == 20
    for m in models.load().values():
        if m.reference == "local" and reference.path(m.name).is_file():
            assert reference.load(m.name).meta["settings"]["chains"] >= 20, m.name


def _fake_cmdstan_csv(path, s, chain, rng, stanc="stanc3 v2.39.0"):
    head = (f"# stanc_version = {stanc}\n# method = sample (Default)\n#   sample\n#     num_samples = {s['samples']}\n"
            f"#     num_warmup = {s['warmup']}\n#     thin = {s['thin']}\n#     adapt\n#       delta = {s['adapt_delta']}\n"
            f"# id = {chain + 1}\n# random\n#   seed = {s['seed']}\n")
    x = rng.normal(size=s["samples"] // s["thin"])
    path.write_text(head + "lp__,divergent__,x\n" + "".join(f"{-v * v / 2},0,{v}\n" for v in x))


def test_a_rebuild_reuses_only_csvs_made_with_the_requested_settings(tmp_path, monkeypatch):
    """T9: CSVs made with other settings, or cut short, are a cache miss, and the reference
    records the settings its draws were made with."""
    from harness import cmdstan
    m = models.load()["wells_dist100_model"]
    base = {"chains": 4, "warmup": 20, "samples": 400, "thin": 1, "adapt_delta": 0.95, "seed": 4711}
    assert reference.settings_for(m, base) == base
    rng = np.random.default_rng(3)
    runs = []

    def build(settings):
        def run_chains(argv_for_chain, chains):  # stands in for CmdStan: writes chains for `settings`
            runs.append(chains)
            for c in range(chains):
                _fake_cmdstan_csv(tmp_path / f"ref_{c}.csv", settings, c, rng)
            return 1.0
        monkeypatch.setattr(cmdstan, "run_chains", run_chains)
        return reference.from_cmdstan(m, tmp_path, tmp_path, tmp_path, settings=settings, log=lambda *a: None)

    monkeypatch.setattr(cmdstan, "parameter_names", lambda root, f: ["x"])
    monkeypatch.setattr(cmdstan, "version", lambda root: "stanc3 v2.39.0 (Unix)")
    monkeypatch.setattr(cmdstan, "compile_model", lambda *a, **k: tmp_path / "exe")
    monkeypatch.setattr(reference, "log_density", lambda model, draws, build_dir, **k: -draws["x"] ** 2 / 2)
    for c in range(4):
        _fake_cmdstan_csv(tmp_path / f"ref_{c}.csv", base, c, rng)
    assert build(base).meta["settings"] == base and runs == []  # matching CSVs are reused

    other = {**base, "seed": 991, "adapt_delta": 0.99}
    assert build(other).meta["settings"] == other and runs == [4]  # other settings: sampled again

    text = (tmp_path / "ref_2.csv").read_text()
    (tmp_path / "ref_2.csv").write_text(text[: len(text) - 7])  # a cut-off last row
    assert build(other).meta["settings"] == other and runs == [4, 4]


def test_a_rebuild_with_another_cmdstan_samples_again(tmp_path, monkeypatch):
    """U7: CSVs made by CmdStan 2.39 were reused for a build with 2.40 and the reference recorded
    2.40. A CSV whose header names another version than the requested CmdStan is a cache miss,
    and the reference records the version in the CSVs it was summarised from."""
    from harness import cmdstan
    m = models.load()["wells_dist100_model"]
    s = {"chains": 4, "warmup": 20, "samples": 400, "thin": 1, "adapt_delta": 0.95, "seed": 4711}
    rng = np.random.default_rng(4)
    installed, runs = ["stanc3 v2.39.0 (Unix)"], []

    def run_chains(argv_for_chain, chains):
        runs.append(chains)
        for c in range(chains):
            _fake_cmdstan_csv(tmp_path / f"ref_{c}.csv", s, c, rng, stanc=installed[0].removesuffix(" (Unix)"))
        return 1.0

    monkeypatch.setattr(cmdstan, "run_chains", run_chains)
    monkeypatch.setattr(cmdstan, "parameter_names", lambda root, f: ["x"])
    monkeypatch.setattr(cmdstan, "version", lambda root: installed[0])
    monkeypatch.setattr(cmdstan, "compile_model", lambda *a, **k: tmp_path / "exe")
    monkeypatch.setattr(reference, "log_density", lambda model, draws, build_dir, **k: -draws["x"] ** 2 / 2)
    for c in range(4):
        _fake_cmdstan_csv(tmp_path / f"ref_{c}.csv", s, c, rng)

    def build():
        return reference.from_cmdstan(m, tmp_path, tmp_path, tmp_path, settings=s, log=lambda *a: None)

    assert build().meta["cmdstan"] == "stanc3 v2.39.0" and runs == []
    installed[0] = "stanc3 v2.40.0 (Unix)"
    assert build().meta["cmdstan"] == "stanc3 v2.40.0" and runs == [4]
    assert build().meta["cmdstan"] == "stanc3 v2.40.0" and runs == [4]


def test_to_json_raises_no_numpy_warning():
    """`_round` with np.vectorize raised "invalid value encountered" on finite input now and then."""
    import warnings
    ref = reference.load("bym2_offset_only")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(5):
            ref.to_json()
    v = np.array([[1.23456789, -2.5e-12], [np.nan, np.inf]])
    assert np.array_equal(reference._round(v, 4), [[1.235, -2.5e-12], [np.nan, np.inf]], equal_nan=True)
