"""Slice 3 tests: records/ PR validation, paired hierarchical bootstrap, and Blum-Hardt Ladder."""

import json

import pytest

from harness import records


def _make_record(tmp_path, *, track="L1", license_id="BSD-3-Clause", with_inv=True):
    d = tmp_path / "2026-11-20_demo"
    d.mkdir(parents=True)
    (d / "submission.py").write_text(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng): ...\n"
        "def transition(tuning, chain, oracle, rng): ...\n"
    )
    (d / "submission.json").write_text(json.dumps({
        "entry_id": "demo-01",
        "author": "tester",
        "date": "2026-11-20",
        "release": "R1",
        "track": track,
        "license": license_id,
        "capabilities_requested": ["grad"],
        "kernel_class": "reversible",
        "aux_spec_summary": "none",
        "reproduce_cmd": "uv run apin run logdensity --opt sampler=submission.py",
    }))
    (d / "pins.toml").write_text('[apin]\nharness_sha = "deadbeef"\n')
    (d / "README.md").write_text("# Demo submission\n")
    if with_inv:
        (d / "invariance.md").write_text(
            "## State space and aux\nPosition x in R^d, aux = none.\n\n"
            "## Moves and composition\nSingle reversible HMC involution + MH acceptance.\n"
        )
    return d


def test_validate_record_dir_accepts_valid_and_rejects_violations(tmp_path):
    good = _make_record(tmp_path / "ok")
    meta = records.validate_record_dir(good)
    assert meta["entry_id"] == "demo-01"

    bad_lic = _make_record(tmp_path / "gpl", license_id="GPL-3.0")
    with pytest.raises(ValueError, match="license"):
        records.validate_record_dir(bad_lic)

    no_inv = _make_record(tmp_path / "no_inv", track="L1", with_inv=False)
    with pytest.raises(ValueError, match="invariance.md"):
        records.validate_record_dir(no_inv)

    # L4 unadjusted does not require invariance.md
    l4 = _make_record(tmp_path / "l4", track="L4", with_inv=False)
    assert records.validate_record_dir(l4)["track"] == "L4"


def test_paired_hierarchical_bootstrap():
    # Candidate is consistently 1.5x better across 6 models x 5 reps -> p < 0.01
    pairs = [([1.48, 1.52, 1.49, 1.51, 1.50], [1.0, 1.01, 0.99, 1.0, 1.0]) for _ in range(6)]
    res = records.paired_hierarchical_bootstrap(pairs, n_boot=1000, seed=0, margin=1.05)
    assert res["ratio"] == pytest.approx(1.5, rel=0.02)
    assert res["ci_lo"] > 1.4 and res["p_value"] < 0.01

    # Tie -> high p-value
    tie = [([1.0, 1.02, 0.98], [1.01, 0.99, 1.0]) for _ in range(5)]
    res_tie = records.paired_hierarchical_bootstrap(tie, n_boot=500, seed=0, margin=1.05)
    assert res_tie["p_value"] > 0.5


def test_blum_hardt_ladder():
    ladder = records.BlumHardtLadder(step=0.05, initial_best=1.0)
    assert ladder.query(1.03) == (False, 1.0)  # noise below ladder step not revealed
    assert ladder.query(1.12) == (True, 1.10)  # quantized to 1.0 + 2*0.05
    assert ladder.query(1.13) == (False, 1.10)
    assert ladder.queries == 3
