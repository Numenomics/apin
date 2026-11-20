"""Slice 3 submission records (`records/`), paired hierarchical bootstrap, and Blum-Hardt Ladder.

Implements Council Rulings B3, B4, D4, and LEADERBOARD.md §2:
  - Self-contained PR record validation under records/<release>/<track>/<date>_<slug>/
  - Permissive license enforcement (BSD-3-Clause, MIT, Apache-2.0 only)
  - Structured involutive-MCMC invariance.md check for adjusted tracks (L1, L2, L3, L5)
  - Paired hierarchical bootstrap (resampling models, then reps) with paired p-value
  - Blum-Hardt Ladder holdout promotion guard (prevents adaptive holdout overfitting)
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import numpy as np

from . import logdensity

PERMISSIVE_LICENSES = {"BSD-3-Clause", "MIT", "Apache-2.0"}
VALID_TRACKS = {"L1", "L2", "L3", "L4", "L5", "S2"}
ADJUSTED_TRACKS = {"L1", "L2", "L3", "L5"}
REQUIRED_JSON_FIELDS = {
    "entry_id", "author", "date", "release", "track", "license",
    "capabilities_requested", "kernel_class", "aux_spec_summary", "reproduce_cmd",
}
REQUIRED_INVARIANCE_MARKERS = ("state space", "aux", "composition")


def validate_record_dir(record_dir: Path) -> dict:
    """Validate a PR record directory under records/<release>/<track>/<date>_<slug>/.
    Returns parsed submission.json metadata on success; raises ValueError on any violation."""
    d = Path(record_dir)
    if not d.is_dir():
        raise ValueError(f"record directory not found: {d}")
    for fname in ("submission.py", "submission.json", "pins.toml", "README.md"):
        if not (d / fname).is_file():
            raise ValueError(f"{d.name}: missing required file {fname}")

    meta = json.loads((d / "submission.json").read_text())
    missing = REQUIRED_JSON_FIELDS - set(meta.keys())
    if missing:
        raise ValueError(f"{d.name}/submission.json missing fields: {sorted(missing)}")
    if meta["track"] not in VALID_TRACKS:
        raise ValueError(f"{d.name}: unknown track {meta['track']!r}")
    if meta["license"] not in PERMISSIVE_LICENSES:
        raise ValueError(f"{d.name}: license {meta['license']!r} not in {sorted(PERMISSIVE_LICENSES)}")

    # Lint submission.py
    logdensity.check_imports((d / "submission.py").read_text())

    # Validate pins.toml parses cleanly and names base SHAs / harness commit
    pins = tomllib.loads((d / "pins.toml").read_text())
    if "apin" not in pins or "harness_sha" not in pins["apin"]:
        raise ValueError(f"{d.name}/pins.toml must include [apin] with harness_sha")

    # Adjusted tracks require structured involutive-MCMC invariance.md
    if meta["track"] in ADJUSTED_TRACKS:
        inv_path = d / "invariance.md"
        if not inv_path.is_file():
            raise ValueError(f"{d.name}: track {meta['track']} requires invariance.md")
        inv_text = inv_path.read_text().lower()
        absent = [m for m in REQUIRED_INVARIANCE_MARKERS if m not in inv_text]
        if absent:
            raise ValueError(
                f"{d.name}/invariance.md must follow involutive-MCMC format (missing sections: {absent})"
            )
    return meta


def paired_hierarchical_bootstrap(
    pairs: list[tuple[list[float], list[float]]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    margin: float = 1.0,
) -> dict:
    """Paired hierarchical bootstrap over models, then reps within each model.

    Each element of `pairs` is (candidate_reps, baseline_reps) for one model (positive efficiencies).
    Returns point geometric mean ratio, 95% CI [lo, hi], and one-sided p-value for H0: ratio <= margin.
    """
    if not pairs:
        return {"ratio": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p_value": math.nan}
    cleaned = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in pairs]
    log_ratios = np.array([math.log(float(np.mean(a)) / float(np.mean(b))) for a, b in cleaned])
    point = float(math.exp(np.mean(log_ratios)))

    m = len(cleaned)
    rng = np.random.default_rng(seed)
    boot_logs = np.empty(n_boot, dtype=float)
    for b_idx in range(n_boot):
        m_pick = rng.integers(0, m, size=m)
        acc = 0.0
        for j in m_pick:
            a, b = cleaned[j]
            if len(a) == len(b):
                # Resample matched seed pairs together so paired comparisons preserve seed blocking
                idx = rng.integers(0, len(a), size=len(a))
                ma = float(np.mean(a[idx]))
                mb = float(np.mean(b[idx]))
            else:
                ma = float(np.mean(a[rng.integers(0, len(a), size=len(a))]))
                mb = float(np.mean(b[rng.integers(0, len(b), size=len(b))]))
            acc += math.log(ma / mb)
        boot_logs[b_idx] = acc / m

    lo, hi = np.percentile(boot_logs, [2.5, 97.5])
    # One-sided bootstrap p-value against null ratio <= margin
    log_margin = math.log(margin)
    p_val = float(np.mean(boot_logs <= log_margin))
    return {
        "ratio": point,
        "ci_lo": float(math.exp(lo)),
        "ci_hi": float(math.exp(hi)),
        "p_value": p_val,
    }


class BlumHardtLadder:
    """Ladder mechanism (Blum & Hardt 2015) guarding sealed-holdout promotion queries.

    Only reveals and updates the running holdout high-water mark when a candidate clears
    `best_reported + step`, preventing adaptive overfitting to the holdout across releases.
    """

    def __init__(self, step: float = 0.05, initial_best: float = 1.0):
        if step <= 0:
            raise ValueError("Ladder step must be positive")
        self.step = float(step)
        self.best_reported = float(initial_best)
        self.queries = 0

    def query(self, true_holdout_score: float) -> tuple[bool, float]:
        """Return (promoted, reported_score)."""
        self.queries += 1
        if math.isfinite(true_holdout_score) and true_holdout_score >= self.best_reported + self.step:
            # Quantize update to ladder step grid
            k = math.floor((true_holdout_score - self.best_reported) / self.step)
            self.best_reported = round(self.best_reported + k * self.step, 6)
            return True, self.best_reported
        return False, self.best_reported
