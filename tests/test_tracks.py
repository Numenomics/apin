"""Unit tests for Benchmark Tracks L2, L3, L4, and L5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness import score, tracks
from harness.models import ROOT


def test_machine_fingerprint_populated():
    fp = tracks.machine_fingerprint()
    assert fp["cpu_count"] >= 1
    assert fp["python"]
    assert fp["numpy"]
    assert fp["scipy"]


def test_track_l3_haar_rotation_and_scale_stress(tmp_path):
    hmc = ROOT / "examples" / "samplers" / "hmc.py"
    rep = tracks.run_track_l3(
        hmc, dims=(4,), scales=(1e-1, 1.0, 1e1), seed=2026,
        n_chains=32, k_steps=10, adapt_draws=60, out_dir=tmp_path / "l3",
    )
    assert rep["track"] == "L3"
    assert (tmp_path / "l3" / "track_L3.json").is_file()
    assert (tmp_path / "l3" / "track_L3.md").is_file()
    assert set(rep["scale_soft_min"].keys()) == {"0.1", "1", "10"}
    assert len(rep["targets"]) == 15


def test_track_l4_unadjusted_records_bias_without_hard_failing(tmp_path):
    ula = ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "ula_base.py"
    rep = tracks.run_track_l4(
        ula, dims=(4,), seed=2026, n_chains=32, checkpoints=(5, 10),
        adapt_draws=30, out_dir=tmp_path / "l4",
    )
    assert rep["track"] == "L4"
    assert "never ranked or pooled against Track L1" in rep["pool_policy"]
    assert rep["targets_total"] == 5
    assert (tmp_path / "l4" / "track_L4.json").is_file()
    assert (tmp_path / "l4" / "track_L4.md").is_file()
    assert all(len(t["checkpoints"]) == 2 for t in rep["targets"])


def test_track_l5_enforces_density_only_and_passes_rwm(tmp_path):
    # Gradient-using sampler MUST fail density-only enforcement immediately
    hmc = ROOT / "examples" / "samplers" / "hmc.py"
    rep_bad = tracks.run_track_l5(
        hmc, dims=(4,), seed=2026, n_chains=24, k_steps=10, adapt_draws=40, out_dir=tmp_path / "l5_bad",
    )
    assert rep_bad["passed_density_only"] is False
    assert "density_only" in rep_bad.get("error", "")

    # Density-only baseline sampler MUST pass L5 with 0 gradient calls
    rwm = ROOT / "examples" / "samplers" / "elliptical_rwm_l5.py"
    rep_ok = tracks.run_track_l5(
        rwm, dims=(4,), seed=2026, n_chains=32, k_steps=12, adapt_draws=80, out_dir=tmp_path / "l5_ok",
    )
    assert rep_ok["passed_density_only"] is True
    assert rep_ok["gradient_evals"] == 0
    assert (tmp_path / "l5_ok" / "track_L5.json").is_file()


def test_score_rank_track_l2_orders_by_wall_clock_ess_per_s():
    models = [
        {"label": "base", "model": "m1", "status": "right", "reason": "", "cost_u": [1.0, 1.0],
         "eval": [1.0, 1.0], "sec": [10.0, 10.0], "gate_eff": [1.0, 1.0], "overhead_ratio": 1.0,
         "ess_per_cost": 1.0, "ess_per_eval": 1.0, "ess_per_s": 10.0},
        {"label": "fast_wall", "model": "m1", "status": "right", "reason": "", "cost_u": [0.8, 0.8],
         "eval": [0.8, 0.8], "sec": [30.0, 30.0], "gate_eff": [0.8, 0.8], "overhead_ratio": 0.2,
         "ess_per_cost": 0.8, "ess_per_eval": 0.8, "ess_per_s": 30.0},
        {"label": "fast_cost", "model": "m1", "status": "right", "reason": "", "cost_u": [2.0, 2.0],
         "eval": [2.0, 2.0], "sec": [5.0, 5.0], "gate_eff": [2.0, 2.0], "overhead_ratio": 10.0,
         "ess_per_cost": 2.0, "ess_per_eval": 2.0, "ess_per_s": 5.0},
    ]
    r_l1 = score.rank(models, "base", ["m1"], track="L1")
    r_l2 = score.rank(models, "base", ["m1"], track="L2")
    assert r_l1[0]["label"] == "fast_cost"
    assert r_l2[0]["label"] == "fast_wall"
    tab_l2 = score.table(r_l2, models, "base", ["m1"], track="L2", machine=tracks.machine_fingerprint())
    assert "Track L2 (wall-clock ESS/s)" in tab_l2


def test_track_out_json_file_writes_directly_and_l3_markdown_header(tmp_path):
    ula = ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "ula_base.py"
    out_file = tmp_path / "custom_l4.json"
    rep = tracks.run_track_l4(
        ula, dims=(4,), seed=2026, n_chains=24, checkpoints=(5,),
        adapt_draws=20, out_dir=out_file,
    )
    assert out_file.is_file()
    assert (tmp_path / "custom_l4.md").is_file()
    assert not out_file.is_dir()

    md_l3 = tracks.format_track_markdown({
        "track": "L3", "submission": "hmc.py", "passed_all_invariance": True,
        "soft_min_eff": 0.123, "scale_invariance_ratio": 1.0, "worst_b2_max": 0.001,
        "targets": [{
            "scale_stress": 1.0, "dim": 6, "family": "gaussian", "passed": True,
            "emp_eff": 0.123, "b2_max": 0.001, "max_z": 1.1, "z_threshold": 3.5, "reason": "ok",
        }],
    })
    assert "dual_eff" in md_l3 and "emp_eff" not in md_l3
    assert rep["track"] == "L4"
