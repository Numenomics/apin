"""One grader for every caller: the judge digest covers what can change a score, and the lane loop
and the benchmark verifier compute the same efficiency for the same file, lane, configuration and
seed."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from harness import autoresearch, grade, judge
from harness.models import ROOT


def _verify():
    spec = importlib.util.spec_from_file_location("apin_harbor_verify_grade", ROOT / "harbor" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _edited_copy(module, tmp_path: Path, edit: str) -> str:
    path = tmp_path / Path(module.__file__).name
    path.write_text(Path(module.__file__).read_text() + edit)
    return str(path)


def test_digest_ignores_lane_loop_bookkeeping_and_tracks_the_grader(tmp_path, monkeypatch):
    before = judge.judge_digest()
    monkeypatch.setattr(autoresearch, "__file__", _edited_copy(autoresearch, tmp_path, "\nTIE_BAND = 0.5\n"))
    assert judge.judge_digest() == before
    monkeypatch.setattr(grade, "__file__", _edited_copy(grade, tmp_path, "\n# a comment\n"))
    assert judge.judge_digest() != before


def test_lane_loop_and_verifier_give_the_same_efficiency(tmp_path, monkeypatch):
    """Same file, lane, configuration and seed through both callers: identical soft-min. The
    verifier's denominators are frozen stand-ins keyed to this configuration, so only the
    candidate is scored in each caller."""
    cfg = grade.config(dims=(4,), repeats=1, n_chains=24, k_steps=8, adapt_draws=40)
    start = ROOT / "autoresearch" / "lanes" / "A2" / "start.py"
    cand = tmp_path / "cand.py"
    cand.write_text(start.read_text() + "\n_EDIT = 1\n")

    lane_dir = tmp_path / "ar" / "lanes" / "A2"
    lane_dir.mkdir(parents=True)
    shutil.copy2(cand, lane_dir / "submission.py")
    row = autoresearch.evaluate_lane_trial("A2", "same efficiency", cfg=cfg, autoresearch_dir=tmp_path / "ar",
                                           run_id="r", use_queue=False)
    assert row["tier1_ok"] and row["decision"] == "exploratory", row

    verify = _verify()
    tests = tmp_path / "tests"
    tests.mkdir()
    digest = judge.judge_digest()
    stand_in = {"valid": True, "tier0_ok": True, "tier1_ok": True, "soft_min_eff": 1e-4}
    scores = {}
    for role, name in verify.DENOMINATOR_FILES.items():
        shutil.copy2(start, tests / name)
        scores[role] = grade.frozen_entry("A2", role, tests / name, cfg, digest, stand_in)
    (tests / verify.FROZEN_FILE).write_text(json.dumps({"schema": verify.FROZEN_SCHEMA, "lane": "A2", "scores": scores}))
    verify.verify_optimize("A2", cand, tmp_path / "out", tests_dir=tests, cfg=cfg)
    details = json.loads((tmp_path / "out" / "details.json").read_text())
    assert all(d["source"] == "frozen" for d in details["denominators"].values())
    assert details["passed"] == 1 and details["config"] == cfg
    assert details["soft_min_eff"] == row["soft_min_eff"]
    assert details["by_class"] == row["by_class"]


def test_speedup_is_candidate_over_start():
    good, start = {"valid": True, "soft_min_eff": 3.0}, {"valid": True, "soft_min_eff": 2.0}
    assert grade.speedup(good, start) == 1.5
    assert grade.speedup({"valid": False, "soft_min_eff": 3.0}, start) is None
    assert grade.speedup({"valid": True, "unpaired": True, "soft_min_eff": 0.0}, start) is None
    assert grade.speedup(good, {"valid": False}) is None


def test_yardstick_id_changes_with_any_key_or_score_field(tmp_path):
    f = tmp_path / "start.py"
    f.write_text("x = 1\n")
    score = {"valid": True, "soft_min_eff": 2e-4, "by_class": {"banana": 2e-4}}
    base = grade.yardstick_entry("A2", f, grade.config(), "d" * 16, score)
    assert base["id"].startswith("A2-") and base["id"] == grade.yardstick_id(base)
    variants = [
        grade.yardstick_entry("A3", f, grade.config(), "d" * 16, score),
        grade.yardstick_entry("A2", f, grade.config(repeats=4), "d" * 16, score),
        grade.yardstick_entry("A2", f, grade.config(), "e" * 16, score),
        grade.yardstick_entry("A2", f, grade.config(), "d" * 16, {**score, "soft_min_eff": 3e-4}),
    ]
    f.write_text("x = 2\n")
    variants.append(grade.yardstick_entry("A2", f, grade.config(), "d" * 16, score))
    assert len({v["id"] for v in variants} | {base["id"]}) == len(variants) + 1
