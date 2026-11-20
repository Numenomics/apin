"""Slice 5 tests: Karpathy 3-file autoresearch runner, novelty filter, Tier 0/1 gates,
MAP-Elites archive, and ledger.jsonl -> results.tsv generation."""

import json
import shutil
from pathlib import Path

from harness import autoresearch

EXAMPLES = Path(__file__).parent.parent / "examples" / "samplers"


def test_autoresearch_mvp_end_to_end(tmp_path):
    lane_dir = tmp_path / "lanes" / "A1"
    lane_dir.mkdir(parents=True)
    shutil.copy2(EXAMPLES / "hmc.py", lane_dir / "submission.py")

    # Trial 1: baseline HMC establishes incumbent + populates MAP-Elites archive
    r1 = autoresearch.evaluate_lane_trial(
        "A1", "Baseline diagonal HMC", codex_refs=["fisher-low-rank"],
        seed=42, dim=4, n_chains=40, k_steps=12, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r1["tier0_ok"] is True and r1["tier1_ok"] is True
    assert r1["decision"] == "keep_baseline"
    assert (lane_dir / "incumbent.py").is_file()
    assert len(r1["archive_cells"]) >= 1

    # Trial 2: identical file rejected immediately by pre-eval novelty filter
    r2 = autoresearch.evaluate_lane_trial(
        "A1", "Duplicate proposal", seed=43, dim=4, n_chains=40, k_steps=12, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r2["decision"] == "discard_duplicate"

    # Trial 3: broken/stuck submission dies at Tier 0 in seconds
    (lane_dir / "submission.py").write_text(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng):\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    return chain.position, None\n"
    )
    r3 = autoresearch.evaluate_lane_trial(
        "A1", "Identity kernel cheat", seed=44, dim=4, n_chains=40, k_steps=12, adapt_draws=60,
        autoresearch_dir=tmp_path, run_id="test_run",
    )
    assert r3["tier0_ok"] is False and r3["decision"] == "discard_tier0"

    # Verify ledger.jsonl and auto-generated results.tsv have all 3 rows
    run_dir = lane_dir / "runs" / "test_run"
    ledger_lines = [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines()]
    tsv_lines = (run_dir / "results.tsv").read_text().splitlines()
    assert len(ledger_lines) == 3 and len(tsv_lines) == 4  # header + 3 rows
