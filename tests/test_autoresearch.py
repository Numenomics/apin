"""Slice 5 tests: Karpathy 3-file autoresearch runner, Lane A1 narrow `adapt_metric` seam,
AST-normalized novelty filter, Tier 0/1 isolated gates, paired MAP-Elites archive,
crash resilience, and ledger.jsonl -> results.tsv generation."""

import json
import shutil
from pathlib import Path

from harness import autoresearch, shapes, invariance

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "samplers"
A1_SUB = ROOT / "autoresearch" / "lanes" / "A1" / "submission.py"


def test_autoresearch_a1_narrow_contract_and_novelty_filter(tmp_path):
    """Lane A1 enforces narrow adapt_metric() seam, rejects transition overrides, and AST-dedupes comments."""
    lane_dir = tmp_path / "lanes" / "A1"
    lane_dir.mkdir(parents=True)
    shutil.copy2(A1_SUB, lane_dir / "submission.py")

    # Trial 1: baseline A1 adapt_metric establishes incumbent + populates MAP-Elites archive
    r1 = autoresearch.evaluate_lane_trial(
        "A1", "Baseline diagonal adapt_metric", codex_refs=["fisher-low-rank"],
        seed=42, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r1["tier0_ok"] is True and r1["tier1_ok"] is True, r1
    assert r1["decision"] == "keep_baseline"
    run_dir = lane_dir / "runs" / "test_run"
    assert (run_dir / "incumbent.py").is_file()
    assert not (lane_dir / "incumbent.py").exists()  # isolated per run_id
    assert len(r1["archive_cells"]) >= 1
    assert r1["dim"] == 4 and r1["repeats"] == 1 and r1["horizon_steps"] == 500
    assert r1["magic_constants"] > 0

    # Dry-run duplicate check does NOT write a ledger row
    dry = autoresearch.evaluate_lane_trial(
        "A1", "Dry run duplicate", seed=43, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run", dry_run=True,
    )
    assert dry["decision"] == "dry_run_duplicate" and dry["is_duplicate"] is True
    assert len((run_dir / "ledger.jsonl").read_text().splitlines()) == 1

    # Trial 2: exact duplicate rejected immediately
    r2 = autoresearch.evaluate_lane_trial(
        "A1", "Duplicate proposal", seed=43, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r2["decision"] == "discard_duplicate"

    # Trial 3: comment-only change rejected by AST-normalized novelty filter
    (lane_dir / "submission.py").write_text(A1_SUB.read_text() + "\n# cosmetic comment only\n")
    r3 = autoresearch.evaluate_lane_trial(
        "A1", "Comment-only bypass attempt", seed=44, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r3["decision"] == "discard_duplicate"

    # Trial 4: trying to override transition() in Lane A1 is rejected at Tier 0!
    shutil.copy2(EXAMPLES / "hmc.py", lane_dir / "submission.py")
    r4 = autoresearch.evaluate_lane_trial(
        "A1", "Attempt to replace fixed A1 transition kernel", seed=45, dim=4, n_chains=36, k_steps=10,
        adapt_draws=60, autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r4["tier0_ok"] is False and r4["decision"] == "discard_tier0"
    assert "adapt_metric" in r4["tier0_info"]["error"]

    # Verify ledger.jsonl and auto-generated results.tsv have all 4 rows + config columns
    ledger_lines = [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines()]
    tsv_lines = (run_dir / "results.tsv").read_text().splitlines()
    assert len(ledger_lines) == 4 and len(tsv_lines) == 5  # header + 4 rows
    header = tsv_lines[0].split("\t")
    for col in ("dim", "seed", "repeats", "k_steps", "adapt_draws", "horizon_steps", "magic_constants"):
        assert col in header


def test_autoresearch_full_kernel_lane_catches_stuck_and_logs_errors(tmp_path, monkeypatch):
    """General lane A0 accepts full (adapt, transition), rejects stuck kernels, and logs unexpected errors."""
    lane_dir = tmp_path / "lanes" / "A0"
    lane_dir.mkdir(parents=True)
    shutil.copy2(EXAMPLES / "hmc.py", lane_dir / "submission.py")

    r1 = autoresearch.evaluate_lane_trial(
        "A0", "Baseline full HMC", seed=50, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="a0_run",
    )
    assert r1["tier0_ok"] is True and r1["tier1_ok"] is True, r1
    assert r1["decision"] == "keep_baseline"

    # Stuck/identity kernel fails Tier 0
    (lane_dir / "submission.py").write_text(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng):\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    return chain.position, None\n"
    )
    r2 = autoresearch.evaluate_lane_trial(
        "A0", "Identity kernel cheat", seed=51, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="a0_run",
    )
    assert r2["tier0_ok"] is False and r2["decision"] == "discard_tier0"

    # Unexpected crash in Tier 1 still writes a discard_error ledger row (Review 3 #4)
    (lane_dir / "submission.py").write_text(EXAMPLES.joinpath("hmc.py").read_text() + "\nX_TAG = 1\n")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated tier1 failure")

    monkeypatch.setattr(autoresearch, "_eval_tier1_suite", boom)
    r3 = autoresearch.evaluate_lane_trial(
        "A0", "Crash resilience check", seed=52, dim=4, n_chains=36, k_steps=10, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="a0_run",
    )
    assert r3["decision"] == "discard_error" and "simulated tier1 failure" in r3["error"]
    ledger_lines = (lane_dir / "runs" / "a0_run" / "ledger.jsonl").read_text().splitlines()
    assert len(ledger_lines) == 3


def test_default_seed_schedule_no_overflow_on_banana_and_rosenbrock():
    """Review 3 #4: shipped HMC/A1 must not crash with OverflowError across default seed schedule."""
    for i in range(6):
        seed = 20261120 + i * 97
        for tgt in (shapes.Banana(4, seed + 4, rotated=True), shapes.HybridRosenbrock(4, seed + 5, b=5.0)):
            res = invariance.check_target_invariance(
                tgt, A1_SUB, seed=seed, n_chains=24, k_steps=8, adapt_draws=40, lane="A1", isolated=False,
            )
            assert "OverflowError" not in res.reason


def test_horizon_amortization_and_simplicity_and_queue(tmp_path, monkeypatch):
    """Verify horizon_steps amortization, Rule 2.5 simplicity regularizer, and FIFO queue lock."""
    import threading
    import time
    from harness import queue

    # 1. Horizon amortization keeps adapt_cost proportional to k_steps / horizon_steps
    tgt = shapes.IllCondGaussian(4, seed=7, rotated=True, kappa=10.0)
    r_amort = invariance.check_target_invariance(
        tgt, A1_SUB, seed=7, n_chains=24, k_steps=10, adapt_draws=40,
        lane="A1", isolated=False, horizon_steps=500,
    )
    assert r_amort.horizon_steps == 500
    assert abs(r_amort.adapt_cost_amortized - r_amort.adapt_cost * (10.0 / 500.0)) < 1e-6

    # 2. Rule 2.5 simplicity regularizer credits fewer magic constants and >=15% lower overhead
    assert autoresearch._is_simpler(100, 100, 5, 6, 4.0, 4.0) is True
    assert autoresearch._is_simpler(100, 100, 6, 6, 3.0, 4.0) is True
    assert autoresearch._is_simpler(100, 100, 6, 6, 3.8, 4.0) is False
    assert autoresearch._is_simpler(105, 100, 2, 10, 1.0, 4.0) is False  # larger LOC not simpler

    # 3. Queue lock serializes concurrent eval slots cleanly
    monkeypatch.setenv("APIN_FORCE_QUEUE", "1")
    lock_file = tmp_path / "q.lock"
    events: list[tuple[str, float]] = []

    def worker(tag: str, hold: float):
        with queue.eval_slot(tag, enabled=True, lock_path=lock_file):
            events.append((f"{tag}:enter", time.perf_counter()))
            time.sleep(hold)
            events.append((f"{tag}:exit", time.perf_counter()))

    t1 = threading.Thread(target=worker, args=("a", 0.15))
    t2 = threading.Thread(target=worker, args=("b", 0.05))
    t1.start()
    time.sleep(0.03)
    t2.start()
    t1.join()
    t2.join()
    assert len(events) == 4
    # Second worker must enter after first worker exits
    assert events[1][0].endswith(":exit") and events[2][0].endswith(":enter")
    assert events[2][1] >= events[1][1] - 1e-4

