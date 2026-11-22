"""Unit tests for Lane A4 (Warm-Start & Nonlinear Transport)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness import autoresearch, invariance, ipc, judge, logdensity, shapes
from harness.models import ROOT


def test_a4_starter_passes_isolated_abi_and_invariance(tmp_path):
    sub = ROOT / "autoresearch" / "lanes" / "A4" / "submission.py"
    tgt = shapes.IllCondGaussian(dim=4, seed=42, rotated=True, kappa=16.0)
    draws, summary = ipc.execute_run_isolated(
        tgt, sub, logdensity.Budget(1, 12, 100_000, 30.0),
        run_seed=99, work_dir=tmp_path / "abi", replay_count=4, lane="A4",
    )
    assert draws.shape == (1, 12, 4)
    assert summary["replay"]["replayed_singles"] == 4

    res = invariance.check_target_invariance(
        tgt, sub, seed=7, n_chains=48, k_steps=12, adapt_draws=60, lane="A4", isolated=True,
    )
    assert res.passed, res.reason
    assert res.empirical_efficiency > 0.0


def test_a4_nonlinear_sinh_arcsinh_passes_and_broken_jacobian_fails(tmp_path):
    good = ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "missing_jacobian_a4_ref.py"
    bad = ROOT / "autoresearch" / "lanes" / "S3" / "fixtures" / "missing_jacobian_a4_base.py"
    tgt = shapes.StudentT(dim=4, seed=101, rotated=False, nu=3.0)

    # Good nonlinear transport passes automatic Jacobian/round-trip verification & invariance
    res_good = invariance.check_target_invariance(
        tgt, good, seed=11, n_chains=48, k_steps=12, adapt_draws=40, lane="A4", isolated=True,
    )
    assert res_good.passed, res_good.reason

    # Missing log-det-Jacobian is caught immediately during adapt_transport verification
    with pytest.raises(logdensity.ContractError, match="log_abs_det_jac"):
        ipc.run_adapt_phase_isolated(
            tgt, bad, logdensity.Budget(1, 20, 50_000, 30.0),
            run_seed=12, bundle_dir=tmp_path / "bad_adapt", lane="A4",
        )


def test_a4_forbids_defining_transition_or_adapt(tmp_path):
    bad = tmp_path / "bad_a4.py"
    bad.write_text(
        "def adapt(oracle, budget, rng): pass\n"
        "def transition(tuning, chain, oracle, rng): pass\n"
    )
    with pytest.raises(logdensity.ContractError, match="may not define transition"):
        judge.load_a4_submission(bad)


def test_a4_autoresearch_dry_run_works(tmp_path):
    ar_dir = tmp_path / "ar"
    (ar_dir / "lanes" / "A4").mkdir(parents=True)
    sub = ar_dir / "lanes" / "A4" / "submission.py"
    sub.write_text((ROOT / "autoresearch" / "lanes" / "A4" / "submission.py").read_text())
    row = autoresearch.evaluate_lane_trial(
        "A4", "dry run smoke", autoresearch_dir=ar_dir, run_id="test", dry_run=True, use_queue=False,
    )
    assert row["decision"] == "dry_run_ok"
    assert row["lint_ok"] is True


def test_a4_parent_on_nonzero_axis_preserves_jacobian_contract():
    """Regression: when _parent_scan picks c > 0, tri_a[0]/tri_b[0] must stay 0 so ldj matches FD."""
    sub_path = ROOT / "autoresearch" / "lanes" / "A4" / "submission.py"
    mod = judge.load_a4_submission(sub_path)
    rng = np.random.default_rng(123)
    n, d = 300, 6
    # Put the funnel parent on axis c = 2 with larger variance so PCA/direct scan lands on c > 0.
    v = rng.normal(0.0, 2.0, size=n)
    S = np.zeros((n, d))
    S[:, 2] = v
    for j in range(d):
        if j != 2:
            S[:, j] = rng.normal(0.0, np.exp(0.5 * v))
    t = mod._fit_affine(S)
    t = mod._fit_nonlinear(t, S)
    assert t["tri_a"][0] == 0.0
    assert t["tri_b"][0] == 0.0
    assert np.any(t["tri_a"][1:] > 0.0)
    judge.validate_a4_transport(mod, t, d, np.random.default_rng(7))

