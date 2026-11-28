"""The real-posterior gate (harness/real_gate.py) and `apin autoresearch gate`, with the run and
score layers stubbed: counting, partial runs, the lane option, the CLI and the wording."""

from __future__ import annotations

import contextlib
import json
import math
import re
from pathlib import Path

import pytest

from harness import cli, flywheel, grid, queue, real_gate, sampler, score

GATE = {"id": "test-gate-v1", "required": ["m1", "m2", "m3"], "reported": ["m4"], "reps": 3, "reference": "test"}
FORBIDDEN = re.compile(r"\b(correct|exact|unbiased|proven)\b", re.I)


def _rows(statuses: dict[str, str], ess: float = 0.02) -> list[dict]:
    """Per-model rows as score.per_model returns them: ess_per_cost is NaN unless the model passed."""
    return [{"model": m, "status": s, "reason": f"{s}: because", "ess_per_cost": ess if s == "right" else math.nan}
            for m, s in statuses.items()]


def test_the_shipped_gate_file_loads_and_matches_the_owner_decision():
    gate = real_gate.load()
    assert gate["id"] == "nuts-ref-v1"
    assert gate["required"] == ["blr", "eight_schools_noncentered", "gp_regr", "garch11"]
    assert gate["reported"] == ["eight_schools_centered"] and gate["reps"] == 3
    assert "2026-11-28" in gate["reference"] and "bump" in gate["reference"]


@pytest.mark.parametrize("bad", [
    {**GATE, "id": "has space"},
    {**GATE, "id": ""},
    {**GATE, "required": []},
    {**GATE, "reported": ["m1"]},  # listed twice
    {**GATE, "reps": 0},
    {**GATE, "reps": True},
])
def test_load_refuses_a_malformed_gate(tmp_path: Path, bad: dict):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        real_gate.load(path)


def test_every_required_model_passing_is_ok_and_reported_models_do_not_count():
    res = real_gate.summarize(GATE, _rows({"m1": "right", "m2": "right", "m3": "right", "m4": "wrong"}, ess=0.04))
    assert (res["required_passed"], res["required_total"], res["reported_passed"], res["reported_total"]) == (3, 3, 0, 1)
    assert res["ok"] and not res["partial"] and res["gate_id"] == "test-gate-v1"
    assert res["real_ess_per_cost"] == pytest.approx(0.04)
    by = {p["model"]: p for p in res["models"]}
    assert by["m4"]["required"] is False and by["m4"]["passed"] is False and by["m4"]["ess_per_cost"] is None
    assert real_gate.flywheel_fields(res) == {"real_stan_passed": 3, "real_stan_total": 3, "real_gate": "test-gate-v1",
                                              "real_ess_per_cost": pytest.approx(0.04)}


def test_a_required_model_that_fails_or_is_unproven_or_missing_is_not_ok():
    res = real_gate.summarize(GATE, _rows({"m1": "right", "m2": "unproven", "m4": "right"}))
    assert (res["required_passed"], res["required_total"]) == (1, 3)
    assert not res["ok"] and not res["partial"] and res["real_ess_per_cost"] is None
    missing = next(p for p in res["models"] if p["model"] == "m3")
    assert missing["status"] == "missing" and "reference" in missing["reason"]
    # The verifier still posts it: k of n tells the Flywheel the trial is not a frontier.
    assert real_gate.flywheel_fields(res) == {"real_stan_passed": 1, "real_stan_total": 3, "real_gate": "test-gate-v1"}


def test_the_ess_per_cost_is_the_geometric_mean_over_required_models():
    rows = _rows({"m1": "right", "m2": "right", "m3": "right"})
    for r, v in zip(rows, (0.01, 0.1, 1.0)):
        r["ess_per_cost"] = v
    res = real_gate.summarize(GATE, rows)
    assert res["ok"] and res["real_ess_per_cost"] == pytest.approx(0.1)


def test_a_subset_or_other_reps_is_partial_never_ok_and_never_posted():
    rows = _rows({"m1": "right", "m2": "right"})
    res = real_gate.summarize(GATE, rows, selected=["m1", "m2"])
    assert res["partial"] and not res["ok"]
    # The total stays the gate's: a subset can never report n of n.
    assert (res["required_passed"], res["required_total"], res["required_run"]) == (2, 3, 2)
    assert real_gate.flywheel_fields(res) is None
    line = real_gate.summary_line(res)
    assert line.startswith("2 of 3 required models passed the 0.1-SD bench gate at this budget")
    assert "partial" in line and "cannot qualify a frontier" in line
    full = _rows({"m1": "right", "m2": "right", "m3": "right", "m4": "right"})
    other_reps = real_gate.summarize(GATE, full, reps=1)
    assert other_reps["partial"] and not other_reps["ok"] and other_reps["required_passed"] == 3
    # Models outside the gate (re-measuring a candidate set) make it partial as well.
    extra = real_gate.summarize(GATE, full + _rows({"m5": "right"}), selected=["m1", "m2", "m3", "m4", "m5"])
    assert extra["partial"] and not extra["ok"]


def test_the_wording_never_claims_more_than_the_gate_shows():
    for statuses in ({"m1": "right", "m2": "right", "m3": "right", "m4": "right"}, {"m1": "wrong", "m4": "unproven"}):
        res = real_gate.summarize(GATE, _rows(statuses))
        for text in (real_gate.summary_line(res), real_gate.table(res)):
            # The gate's reasons are its own; the harness's words must not overclaim.
            assert not FORBIDDEN.search(text.replace("because", "")), text
        assert "passed the 0.1-SD bench gate at this budget" in real_gate.summary_line(res)
        assert "[gate test-gate-v1]" in real_gate.summary_line(res)


class _Recorder:
    """Stands in for the run and score layers and records what step 8 would have run."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, statuses: dict[str, str]):
        self.calls: dict = {}
        self.statuses = statuses

        def fake_get(name, settings):
            self.calls["sampler"] = (name, dict(settings.options), settings.build_dir)
            return object()

        def fake_grid(s_obj, models, reps, runs_dir, label, **kw):
            self.calls["grid"] = ([m.name for m in models], reps, runs_dir, label, kw)
            return []

        def fake_declared(spec, reps, base_seed, build_dir):
            self.calls["declared"] = (spec, reps, base_seed)
            return "grid"

        def fake_score(runs_dir, out_dir, g, baseline=None, labels=None, **kw):
            self.calls["score"] = (runs_dir, out_dir, g, baseline, labels, kw.get("submissions"))
            return {"per_model": _rows(self.statuses)}

        @contextlib.contextmanager
        def fake_slot(job, *, enabled=True):
            self.calls["queue"] = (job, enabled)
            yield 0.0

        monkeypatch.setattr(sampler, "get", fake_get)
        monkeypatch.setattr(grid, "run_grid", fake_grid)
        monkeypatch.setattr(score, "declared", fake_declared)
        monkeypatch.setattr(score, "score", fake_score)
        monkeypatch.setattr(queue, "eval_slot", fake_slot)


@pytest.mark.parametrize("lane, lane_opt", [("A1", "A1"), ("A4", "A4"), ("A2", None), (None, None)])
def test_run_is_step_8_with_the_lane_option_for_judge_kernel_lanes(monkeypatch, tmp_path, lane, lane_opt):
    gate = real_gate.load()
    rec = _Recorder(monkeypatch, {m: "right" for m in gate["required"] + gate["reported"]})
    sub = tmp_path / "submission.py"
    sub.write_text("def adapt(): pass\n")
    res = real_gate.run(sub, lane, work_dir=tmp_path / "work")
    name, options, _ = rec.calls["sampler"]
    assert name == "logdensity" and options.get("sampler") == str(sub.resolve())
    assert options.get("lane") == lane_opt
    models, reps, runs_dir, label, kw = rec.calls["grid"]
    assert models == gate["required"] + gate["reported"] and reps == 3
    assert runs_dir == tmp_path / "work" / "runs" and label == res["label"]
    assert rec.calls["declared"] == (",".join(models), 3, grid.BASE_SEED)
    # The score refuses runs made with a different file (`apin score --submission`).
    assert rec.calls["score"][4] == [label] and rec.calls["score"][5] == {label: sub.resolve()}
    assert rec.calls["queue"][1] is True and label in rec.calls["queue"][0]
    assert res["ok"] and res["lane"] == lane and res["work_dir"] == str(tmp_path / "work")
    assert re.fullmatch(r"[0-9a-f]{64}", res["submission_sha256"])


def test_run_labels_by_content_and_lane_and_honours_models_reps_and_no_queue(monkeypatch, tmp_path):
    rec = _Recorder(monkeypatch, {"blr": "right"})
    sub = tmp_path / "s.py"
    sub.write_text("x = 1\n")
    res = real_gate.run(sub, "A1", models=["blr"], reps=1, use_queue=False)
    assert res["label"] == real_gate.default_label(sub, "A1") and res["label"].startswith("gate-A1-")
    assert real_gate.default_label(sub, "A2").startswith("gate-") and "A2" not in real_gate.default_label(sub, "A2")
    assert Path(res["work_dir"]) == real_gate.DEFAULT_WORK / res["label"]
    assert rec.calls["grid"][0] == ["blr"] and rec.calls["grid"][1] == 1 and rec.calls["queue"][1] is False
    assert res["partial"] and not res["ok"] and res["required_passed"] == 1
    sub.write_text("x = 2\n")
    assert real_gate.default_label(sub, "A1") != res["label"]
    with pytest.raises(ValueError):
        real_gate.run(sub, "A1", label="../escape")


def test_cli_gate_prints_the_table_and_the_line_or_one_json_object(monkeypatch, tmp_path, capsys):
    seen: dict = {}
    sub = tmp_path / "s.py"
    sub.write_text("x = 1\n")

    def fake_run(path, lane, **kw):
        seen.update(path=path, lane=lane, **kw)
        kw["log"]("progress line")
        return {**real_gate.summarize(GATE, _rows({"m1": "right", "m2": "wrong", "m3": "right", "m4": "unproven"})),
                "work_dir": "/tmp/w", "label": "L"}

    monkeypatch.setattr(real_gate, "run", fake_run)
    assert cli.main(["autoresearch", "gate", "--submission", str(sub), "--lane", "A1"]) == 0
    out = capsys.readouterr().out
    assert "2 of 3 required models passed the 0.1-SD bench gate at this budget" in out
    assert "m2" in out and "failed" in out and "unproven" in out and "/tmp/w" in out
    assert seen["path"] == sub and seen["lane"] == "A1" and seen["models"] is None and seen["reps"] is None
    assert seen["use_queue"] is True and seen["label"] is None

    assert cli.main(["autoresearch", "gate", "--submission", str(sub), "--models", "blr,gp_regr", "--models", "garch11",
                     "--reps", "1", "--label", "quick", "--json", "--no-queue"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["required_passed"] == 2  # stdout is one JSON object
    assert "progress line" in captured.err
    assert seen["models"] == ["blr", "gp_regr", "garch11"] and seen["reps"] == 1 and seen["label"] == "quick"
    assert seen["use_queue"] is False and seen["lane"] is None

    # Errors exit nonzero; a finished run exits 0 whatever it found.
    assert cli.main(["autoresearch", "gate", "--submission", str(tmp_path / "none.py")]) == 2
    assert cli.main(["autoresearch", "gate", "--submission", str(sub), "--label", "a/b"]) == 2
    monkeypatch.setattr(real_gate, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no bridgestan")))
    assert cli.main(["autoresearch", "gate", "--submission", str(sub)]) == 1
    assert "no bridgestan" in capsys.readouterr().err


def test_verify_result_posts_the_gate_fields_only_with_a_gate_id(monkeypatch):
    sent: list[tuple] = []
    monkeypatch.setattr(flywheel, "_request_json", lambda method, path, payload=None, **kw: sent.append((method, path, payload)) or {})
    flywheel.verify_result("A1:r:T1", "vlease_x", True, metrics={"speedup": 1.2}, note="reproduced",
                           real_stan_passed=4, real_stan_total=4, real_gate="nuts-ref-v1", real_ess_per_cost=0.03)
    flywheel.verify_result("A1:r:T1", "vlease_x", True, real_stan_passed=3, real_stan_total=4, real_gate="nuts-ref-v1")
    flywheel.verify_result("A1:r:T1", "vlease_x", False, note="mismatch")
    assert [s[:2] for s in sent] == [("POST", "/api/verify/A1%3Ar%3AT1/result")] * 3
    assert sent[0][2] == {"lease_token": "vlease_x", "passed": True, "metrics": {"speedup": 1.2}, "note": "reproduced",
                          "real_stan_passed": 4, "real_stan_total": 4, "real_gate": "nuts-ref-v1", "real_ess_per_cost": 0.03}
    assert sent[1][2] == {"lease_token": "vlease_x", "passed": True, "real_stan_passed": 3, "real_stan_total": 4,
                          "real_gate": "nuts-ref-v1"}
    assert sent[2][2] == {"lease_token": "vlease_x", "passed": False, "note": "mismatch"}

