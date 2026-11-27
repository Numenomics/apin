"""Unit tests for systems lane S2 and rewind calibration lane S3 (AUARC)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from harness import rewind, systems_lanes


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "src").mkdir()
    (path / "other").mkdir()
    (path / "src" / "a.cpp").write_text("int f() { return 1; }\n")
    (path / "other" / "b.cpp").write_text("int g() { return 2; }\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_s2_fd_check_byte_identical_canary_and_timing(tmp_path):
    repo = tmp_path / "math_repo"
    _init_git_repo(repo)
    ar_dir = tmp_path / "ar"
    (repo / "src" / "a.cpp").write_text("// fast path\nint f() { return 1; }\n")

    def good_grad(x):
        return -0.5 * float(x @ x), -x

    def bad_grad(x):
        return -0.5 * float(x @ x), -2.0 * x

    pts = np.random.default_rng(0).standard_normal((8, 4))
    assert systems_lanes.finite_diff_gradient_check(good_grad, pts)["passed"] is True
    assert systems_lanes.finite_diff_gradient_check(bad_grad, pts)["passed"] is False

    b_csv = tmp_path / "base_chain_0.csv"
    c_csv = tmp_path / "cand_chain_0.csv"
    b_csv.write_text("x\n1.0\n2.0\n")
    c_csv.write_text("x\n1.0\n2.0\n")
    b_dir_f = tmp_path / "b" / "chain_0.csv"
    c_dir_f = tmp_path / "c" / "chain_0.csv"
    b_dir_f.parent.mkdir()
    c_dir_f.parent.mkdir()
    b_dir_f.write_text("x\n1.0\n2.0\n")
    c_dir_f.write_text("x\n1.0\n2.0\n")

    canary_eq = systems_lanes.check_byte_identical_draws([b_dir_f], [c_dir_f])
    assert canary_eq["byte_identical"] is True
    assert canary_eq["waiver_granted"] is True

    r_ok = systems_lanes.evaluate_s2_trial(
        "math", "test s2 speedup", allowed_paths=["src/"],
        clean_build_cmd="true", test_cmd="true",
        baseline_draws=[b_dir_f], candidate_draws=[c_dir_f],
        baseline_grad_fn=good_grad, candidate_grad_fn=good_grad,
        probe_points=pts, min_speedup=0.0,
        upstream_dir=repo, autoresearch_dir=ar_dir, run_id="t2",
    )
    assert r_ok["decision"] == "keep"
    assert r_ok["fd_ok"] is True
    assert r_ok["byte_identical"] is True
    assert r_ok["speedup_ratio"] is not None

    # Non-byte-identical draws trigger statistical gate requirement
    c_dir_f.write_text("x\n1.0\n2.0001\n")
    r_stat = systems_lanes.evaluate_s2_trial(
        "math", "non identical", allowed_paths=["src/"],
        baseline_draws=[b_dir_f], candidate_draws=[c_dir_f],
        upstream_dir=repo, autoresearch_dir=ar_dir, run_id="t2",
    )
    assert r_stat["decision"] == "needs_statistical_gate"


def test_s3_auarc_and_rewind_calibration(tmp_path):
    # AUARC math: instant solve -> 1.0; solve halfway -> 0.5; fail -> 0.0
    assert rewind.compute_auarc([{"passed": True, "resource_used": 0.0}], max_resource=1.0) == 1.0
    assert rewind.compute_auarc([{"passed": True, "resource_used": 0.5}], max_resource=1.0) == 0.5
    assert rewind.compute_auarc([{"passed": False, "resource_used": 0.1}], max_resource=1.0) == 0.0

    rep_base = rewind.evaluate_s3_suite(mode="base", out_dir=tmp_path / "s3")
    rep_ref = rewind.evaluate_s3_suite(mode="reference", out_dir=tmp_path / "s3")

    # Every base fixture MUST fail its hidden check, and every reference fix MUST pass!
    assert rep_base["total"] == 3
    assert rep_base["passed"] == 0, rep_base["results"]
    assert rep_ref["total"] == 3
    assert rep_ref["passed"] == 3, rep_ref["results"]
    assert rep_ref["auarc"] == 1.0


def test_s2_directory_draws_timing_override_and_s3_cli_exit_code(tmp_path, monkeypatch):
    from harness import cli

    repo = tmp_path / "bs_repo"
    _init_git_repo(repo)
    ar_dir = tmp_path / "ar"
    (repo / "src" / "a.cpp").write_text("// tuned\nint f() { return 1; }\n")

    b_dir = tmp_path / "base_draws"
    c_dir = tmp_path / "cand_draws"
    b_dir.mkdir()
    c_dir.mkdir()
    (b_dir / "chain_0.csv").write_text("x\n1.0\n2.0\n")
    (c_dir / "chain_0.csv").write_text("x\n1.0\n2.0\n")

    # Directory inputs expand cleanly and timing_override drives keep/discard
    r_keep = systems_lanes.evaluate_s2_trial(
        "bridgestan", "dir draws + timing", allowed_paths=["src/"],
        baseline_draws=[b_dir], candidate_draws=[c_dir],
        timing_override={"fd_ok": True, "baseline_median_us": 100.0, "candidate_median_us": 90.0},
        min_speedup=1.05, upstream_dir=repo, autoresearch_dir=ar_dir, run_id="t_dir",
    )
    assert r_keep["decision"] == "keep"
    assert r_keep["fd_ok"] is True
    assert r_keep["byte_identical"] is True
    assert r_keep["speedup_ratio"] == round(100.0 / 90.0, 4)

    r_fd_fail = systems_lanes.evaluate_s2_trial(
        "bridgestan", "fd fail", allowed_paths=["src/"],
        timing_override={"fd_ok": False, "speedup_ratio": 2.0},
        upstream_dir=repo, autoresearch_dir=ar_dir, run_id="t_dir",
    )
    assert r_fd_fail["decision"] == "discard_fd_fail"

    # CLI eval-s3 custom mode must return non-zero when any task fails
    monkeypatch.setattr(
        rewind, "evaluate_s3_suite",
        lambda *a, **kw: {
            "passed": 1, "total": 3, "pass_rate": 0.333, "auarc": 0.333,
            "results": [{"task_id": "t1", "passed": False, "emp_eff": 0.0, "reason": "fail"}],
        },
    )
    assert cli.main(["autoresearch", "eval-s3", "--mode", "custom"]) == 1
    assert cli.main(["autoresearch", "eval-s3", "--mode", "base"]) == 1
