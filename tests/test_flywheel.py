from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from harness import flywheel

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_seed_module():
    spec = importlib.util.spec_from_file_location(
        "seed_from_apin", REPO_ROOT / "flywheel" / "seed_from_apin.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolated_token_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's real ~/.config/apin/flywheel_token."""
    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN_FILE", str(tmp_path / "no-token-here"))


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_unconfigured_client_cleanly_noops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("APIN_FLYWHEEL_URL", raising=False)
    monkeypatch.delenv("APIN_FLYWHEEL_TOKEN", raising=False)

    assert flywheel.configured() is False
    assert flywheel.get_frontier() is None
    assert flywheel.claim_task("A1:t1", "A1", "agent") is None
    assert flywheel.release_claim("claim-1", status="done") is None
    assert flywheel.post_node({"lane": "A1", "run_id": "r1", "decision": "keep"}) is None

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"trial_id":"T0001","decision":"keep_baseline"}\n', encoding="utf-8")
    assert flywheel.sync_ledger(ledger, lane="A1", run_id="r1") == []


def test_client_http_calls_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APIN_FLYWHEEL_URL", "https://flywheel.example.dev/")
    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN", "secret-123")
    assert flywheel.configured() is True

    calls: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=10.0):
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        calls.append(
            {
                "method": req.get_method(),
                "url": req.full_url,
                "headers": dict(req.header_items()),
                "body": body,
                "timeout": timeout,
            }
        )
        if req.full_url.endswith("/api/frontier"):
            return _FakeResponse({"frontier": {"A1": {"id": "A1:r1:T0001"}}, "count": 1})
        if req.full_url.endswith("/api/claims"):
            return _FakeResponse({"ok": True, "claim": {"id": "c-1", "status": "active"}})
        if "/api/claims/" in req.full_url:
            return _FakeResponse({"ok": True, "claim": {"id": "c-1", "status": "done"}})
        return _FakeResponse({"ok": True, "node": body})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    frontier = flywheel.get_frontier()
    assert frontier == {"frontier": {"A1": {"id": "A1:r1:T0001"}}, "count": 1}

    claim = flywheel.claim_task("A3:pdmp", "A3", "GLM-5.3", summary="Try zigzag", ttl_minutes=45)
    assert claim["claim"]["id"] == "c-1"

    released = flywheel.release_claim("c-1", status="done", summary="Completed")
    assert released["claim"]["status"] == "done"

    posted = flywheel.post_node({"lane": "A3", "run_id": "r1", "decision": "keep"})
    assert posted["ok"] is True

    assert [c["method"] for c in calls] == ["GET", "POST", "PATCH", "POST"]
    assert calls[0]["url"] == "https://flywheel.example.dev/api/frontier"
    assert calls[1]["url"] == "https://flywheel.example.dev/api/claims"
    assert calls[1]["body"]["ttl_minutes"] == 45
    assert calls[2]["url"] == "https://flywheel.example.dev/api/claims/c-1"
    assert calls[2]["body"] == {"status": "done", "summary": "Completed"}
    assert calls[3]["url"] == "https://flywheel.example.dev/api/nodes"
    for call in calls:
        assert call["headers"]["Authorization"] == "Bearer secret-123"


def test_client_http_error_raises_flywheel_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout=10.0):
        raise urllib.error.HTTPError(
            req.full_url,
            409,
            "Conflict",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"already_claimed"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(flywheel.FlywheelError) as exc_info:
        flywheel.claim_task("dup-task", "A1", "bot", url="https://flywheel.example.dev")

    assert exc_info.value.status == 409
    assert exc_info.value.body == {"error": "already_claimed"}


def test_sync_ledger_links_incumbent_lineage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        {
            "trial_id": "T0001",
            "timestamp": "2026-11-21T10:00:00+00:00",
            "lane": "A2",
            "candidate_sha": "sha1",
            "dim": 6,
            "hypothesis": "Baseline",
            "codex_refs": ["ref1"],
            "soft_min_eff": 0.0002,
            "ratio_vs_incumbent": 1.0,
            "decision": "keep_baseline",
        },
        {
            "trial_id": "T0002",
            "timestamp": "2026-11-21T10:05:00+00:00",
            "lane": "A2",
            "candidate_sha": "sha2",
            "dim": 6,
            "hypothesis": "Discarded idea",
            "soft_min_eff": 0.0001,
            "ratio_vs_incumbent": 0.5,
            "decision": "discard_no_gain",
        },
        {
            "trial_id": "T0003",
            "timestamp": "2026-11-21T10:10:00+00:00",
            "lane": "A2",
            "candidate_sha": "sha3",
            "dims": [6, 20],
            "hypothesis": "Breakthrough keep",
            "short_horizon_soft_min": 0.0003,
            "long_horizon_soft_min": 0.0006,
            "soft_min_eff": 0.0005,
            "ratio_vs_incumbent": 2.5,
            "decision": "keep",
        },
        {
            "trial_id": "T0004",
            "timestamp": "2026-11-21T10:15:00+00:00",
            "lane": "A2",
            "candidate_sha": "sha4",
            "hypothesis": "Follow-up off T0003",
            "soft_min_eff": 0.0004,
            "ratio_vs_incumbent": 0.8,
            "decision": "archive_only",
        },
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    posted_nodes: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=10.0):
        payload = json.loads(req.data.decode("utf-8"))
        posted_nodes.append(payload)
        return _FakeResponse({"ok": True, "node": payload})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = flywheel.sync_ledger(
        ledger,
        lane="A2",
        run_id="demo_run",
        agent_model="Muse-1.3",
        url="https://flywheel.example.dev",
        token="tok",
    )
    assert len(out) == 4
    assert [n["parent_id"] for n in posted_nodes] == [
        None,
        "A2:demo_run:T0001",
        "A2:demo_run:T0001",
        "A2:demo_run:T0003",
    ]
    assert posted_nodes[2]["dims"] == [6, 20]
    assert posted_nodes[2]["short_horizon_eff"] == 0.0003
    assert posted_nodes[2]["long_horizon_eff"] == 0.0006


def test_seed_from_apin_generates_valid_sql(tmp_path: Path) -> None:
    seed_mod = _load_seed_module()

    # Build a synthetic mini-repo
    run_dir = tmp_path / "autoresearch" / "lanes" / "A1" / "runs" / "glm53_a1_1h"
    run_dir.mkdir(parents=True)
    ledger_rows = [
        {
            "trial_id": "T0001",
            "timestamp": "2026-11-21T12:00:00+00:00",
            "lane": "A1",
            "candidate_sha": "sha_base",
            "dim": 6,
            "hypothesis": "Baseline A1",
            "codex_refs": ["fisher-low-rank"],
            "soft_min_eff": 0.0004,
            "ratio_vs_incumbent": 1.0,
            "overhead_ratio": 3.5,
            "b2_max": 0.1,
            "decision": "keep_baseline",
        },
        {
            "trial_id": "T0002",
            "timestamp": "2026-11-21T12:05:00+00:00",
            "lane": "A1",
            "candidate_sha": "sha_keep",
            "dims": [6, 20],
            "hypothesis": "Improved A1 with 'quoted' note",
            "codex_refs": ["fisher-low-rank"],
            "soft_min_eff": 0.0011,
            "speedup": 1.27,
            "yardstick": "A1:judge-1:r8",
            "ratio_vs_incumbent": 2.75,
            "overhead_ratio": 5.1,
            "b2_max": 0.12,
            "decision": "keep",
        },
        {
            "trial_id": "T0003",
            "timestamp": "2026-11-21T12:10:00+00:00",
            "lane": "A1",
            "candidate_sha": "sha_arch",
            "dim": 16,
            "hypothesis": "Archive check",
            "soft_min_eff": 0.0009,
            "ratio_vs_incumbent": 0.82,
            "decision": "archive_only",
        },
    ]
    (run_dir / "ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ledger_rows) + "\n", encoding="utf-8"
    )

    scores_dir = tmp_path / "scores" / "multilane_1h_all"
    scores_dir.mkdir(parents=True)
    (scores_dir / "per_model.csv").write_text(
        "label,model,status,reason,calibration,calibration_p,ess_per_cost,ess_per_eval,ess_per_s,cost_to_gate,overhead_ratio\n"
        "a1_glm53_1h,blr,right,,0.9,1.0,0.02,0.04,400,1000,5.1\n"
        "a1_glm53_1h,gp_regr,right,,0.8,0.9,0.04,0.08,700,800,4.9\n"
        "a1_glm53_1h,eight_schools_centered,unproven,R-hat,nan,nan,nan,nan,nan,nan,6.0\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "flywheel"
    summary = seed_mod.generate_seed_artifacts(tmp_path, out_dir)
    assert summary["nodes"] == 3
    assert summary["edges"] == 2
    assert summary["lanes"] == ["A1"]
    # The site reads the live API only: no offline snapshot is written any more.
    assert sorted(p.name for p in out_dir.iterdir()) == ["seed.sql"]

    # Verify generated seed.sql executes cleanly against flywheel/schema.sql + migrations in sqlite3
    schema_sql = (REPO_ROOT / "flywheel" / "schema.sql").read_text(encoding="utf-8")
    seed_sql = (out_dir / "seed.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    for mig in sorted((REPO_ROOT / "flywheel" / "migrations").glob("*.sql")):
        conn.executescript(mig.read_text(encoding="utf-8"))
    conn.executescript(seed_sql)
    # A trial disputed on the live Flywheel stays disputed when the seed is re-applied.
    conn.execute("UPDATE nodes SET state = 'disputed' WHERE id = 'A1:glm53_a1_1h:T0003'")
    # Re-run to verify idempotency
    conn.executescript(seed_sql)
    count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert count == 3
    verified_row = conn.execute(
        "SELECT id, real_stan_passed, real_stan_total, verified FROM nodes WHERE verified = 1"
    ).fetchone()
    assert verified_row == ("A1:glm53_a1_1h:T0002", 2, 3, 1)
    states = dict(conn.execute("SELECT id, state FROM nodes").fetchall())
    assert states["A1:glm53_a1_1h:T0002"] == "verified"
    assert states["A1:glm53_a1_1h:T0003"] == "disputed"
    assert conn.execute("SELECT COUNT(*) FROM node_parents").fetchone()[0] == 2
    # Re-applying the seed upserts in place, so the trigger-kept run counters stay exact.
    assert conn.execute("SELECT lane, run_id, trials, kept, archived FROM run_stats").fetchall() == [
        ("A1", "glm53_a1_1h", 3, 2, 1)
    ]
    # seed.sql ends with migration 0005's backfill: only T0002 was graded, so it is the lane's
    # one frontier step, and the lane counters agree.
    flags = conn.execute("SELECT id, yardstick IS NOT NULL, frontier_step, speedup FROM nodes ORDER BY id").fetchall()
    assert flags == [
        ("A1:glm53_a1_1h:T0001", 0, 0, None),
        ("A1:glm53_a1_1h:T0002", 1, 1, 1.27),
        ("A1:glm53_a1_1h:T0003", 0, 0, None),
    ]
    assert conn.execute("SELECT standard, steps FROM lane_stats WHERE lane = 'A1'").fetchone() == (1, 1)
    assert seed_mod.flag_backfill_sql() in seed_sql
    conn.close()


def test_seed_from_apin_writes_the_flag_backfill_for_the_rollout_repair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--backfill-sql writes the BACKFILL block alone. It recomputes the step flags of trials
    written without them, with the 5% margin, and it is idempotent."""
    seed_mod = _load_seed_module()
    out = tmp_path / "backfill.sql"
    seed_mod.main(["--backfill-sql", str(out)])
    assert str(out) in capsys.readouterr().out
    sql = out.read_text(encoding="utf-8")
    assert seed_mod.flag_backfill_sql() in sql
    assert "ALTER TABLE" not in sql and "CREATE TRIGGER" not in sql

    conn = sqlite3.connect(":memory:")
    conn.executescript((REPO_ROOT / "flywheel" / "schema.sql").read_text(encoding="utf-8"))
    for mig in sorted((REPO_ROOT / "flywheel" / "migrations").glob("*.sql")):
        conn.executescript(mig.read_text(encoding="utf-8"))
    # Trials written straight into the table: no step flag set.
    rows = [
        ("A1:r:T1", "A1", "keep", 1.0, "y", "2026-01-01T00:01:00Z"),
        ("A1:r:T2", "A1", "keep", 1.03, "y", "2026-01-01T00:02:00Z"),
        ("A1:r:T3", "A1", "keep", 1.05, "y", "2026-01-01T00:03:00Z"),
        ("A1:r:T4", "A1", "keep", 9.0, None, "2026-01-01T00:04:00Z"),
        ("A1:r:T5", "A1", "keep", None, "y", "2026-01-01T00:05:00Z"),
        ("S1:r:T1", "S1", "discard_tier1", 5.0, "y", "2026-01-01T00:01:00Z"),
        ("S1:r:T2", "S1", "keep_simpler", 2.0, "y", "2026-01-01T00:02:00Z"),
    ]
    conn.executemany(
        "INSERT INTO nodes (id, lane, run_id, decision, speedup, yardstick, created_at) VALUES (?, ?, 'r', ?, ?, ?, ?)",
        rows,
    )
    assert conn.execute("SELECT SUM(frontier_step) FROM nodes").fetchone() == (0,)
    for _ in range(2):
        conn.executescript(sql)
        flags = conn.execute("SELECT id, yardstick IS NOT NULL, frontier_step FROM nodes ORDER BY id").fetchall()
        assert flags == [
            ("A1:r:T1", 1, 1),
            ("A1:r:T2", 1, 0),
            ("A1:r:T3", 1, 1),
            ("A1:r:T4", 0, 0),
            ("A1:r:T5", 1, 0),
            ("S1:r:T1", 1, 0),
            ("S1:r:T2", 1, 1),
        ]
        stats = conn.execute("SELECT lane, standard, steps, steps_dirty FROM lane_stats ORDER BY lane").fetchall()
        assert stats == [("A1", 4, 2, 0), ("S1", 2, 1, 0)]
    conn.close()


def test_token_only_defaults_to_public_url_and_post_trial_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APIN_FLYWHEEL_URL", raising=False)
    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN", "apfw_contributor_tok")

    assert flywheel.configured() is True
    calls: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=10.0):
        calls.append({
            "url": req.full_url,
            "auth": req.get_header("Authorization"),
            "body": json.loads(req.data.decode("utf-8")) if req.data else None,
        })
        return _FakeResponse({"ok": True, "node": {"id": "A4:r1:T0002"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    resp = flywheel.post_trial_row(
        {
            "trial_id": "T0002",
            "lane": "A4",
            "hypothesis": "Try sinh warp",
            "decision": "keep",
            "speedup": 1.27,
            "yardstick": "A4-0123456789ab",
            "soft_min_eff": 0.0008,
            "dims": [6, 20],
            "candidate_sha": "abc123",
            "timestamp": "2026-11-22T00:00:00+00:00",
        },
        run_id="r1",
        parent_id="A4:r1:T0001",
    )
    assert resp is not None
    assert calls[0]["url"] == f"{flywheel.DEFAULT_PUBLIC_URL}/api/nodes"
    assert calls[0]["auth"] == "Bearer apfw_contributor_tok"
    assert calls[0]["body"]["id"] == "A4:r1:T0002"
    assert calls[0]["body"]["parent_id"] == "A4:r1:T0001"
    assert calls[0]["body"]["verified"] == 0
    assert calls[0]["body"]["speedup"] == 1.27 and calls[0]["body"]["yardstick"] == "A4-0123456789ab"
    assert calls[0]["body"]["soft_min_eff"] == 0.0008


def test_cli_claim_and_release_report_next_step(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`apin autoresearch claim/release` wrap the claim ledger and say what to do next."""
    from harness import cli

    seen: dict[str, Any] = {}

    def fake_claim(task_id, lane, agent, summary="", ttl_minutes=60, **_):
        seen["claim"] = (task_id, lane, agent, summary, ttl_minutes)
        return {"claim": {"id": "c-1", "expires_at": "2026-11-23T01:00:00Z"}}

    monkeypatch.setattr(flywheel, "claim_task", fake_claim)
    monkeypatch.setattr(flywheel, "release_claim", lambda claim_id, status="done", **_: {"claim": {"id": claim_id, "status": status}})

    assert cli.main(["autoresearch", "claim", "--task", "a1-heavy-tail-robust-fisher", "--lane", "A1", "--agent", "me", "--summary", "MAD scale"]) == 0
    assert seen["claim"] == ("a1-heavy-tail-robust-fisher", "A1", "me", "MAD scale", 60)
    out = capsys.readouterr().out
    assert "2026-11-23T01:00:00Z" in out and "release --claim c-1" in out

    assert cli.main(["autoresearch", "release", "--claim", "c-1", "--status", "abandoned"]) == 0
    assert "abandoned" in capsys.readouterr().out

    def taken(*_, **__):
        raise flywheel.FlywheelError("conflict", status=409)

    monkeypatch.setattr(flywheel, "claim_task", taken)
    assert cli.main(["autoresearch", "claim", "--task", "t", "--lane", "A1", "--agent", "me"]) == 1
    assert "already claimed" in capsys.readouterr().err

    monkeypatch.setattr(flywheel, "claim_task", lambda *_, **__: None)
    assert cli.main(["autoresearch", "claim", "--task", "t", "--lane", "A1", "--agent", "me"]) == 1
    assert "APIN_FLYWHEEL_TOKEN" in capsys.readouterr().err


def test_cli_renew_claim_and_at_handle_stripping(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from harness import cli

    seen: dict[str, Any] = {}

    def fake_renew(claim_id, ttl_minutes=60, summary=None, **_):
        seen["renew"] = (claim_id, ttl_minutes, summary)
        return {"claim": {"id": claim_id, "expires_at": "2026-11-23T03:00:00Z"}}

    def fake_register(handle, agent=None, model=None, invite_code=None, **_):
        seen["register"] = (handle, agent, model)
        return {"handle": handle, "role": "provisional", "hourly_limit": 60, "token": "apfw_test_tok"}

    monkeypatch.setattr(flywheel, "renew_claim", fake_renew)
    monkeypatch.setattr(flywheel, "register", fake_register)
    monkeypatch.setattr(flywheel, "save_token", lambda tok, force=True: Path("/tmp/tok"))
    monkeypatch.setattr(flywheel, "save_identity", lambda *a, **kw: None)

    assert cli.main(["autoresearch", "renew", "--claim", "c-9", "--ttl-minutes", "120", "--summary", "still running"]) == 0
    assert seen["renew"] == ("c-9", 120, "still running")
    assert "2026-11-23T03:00:00Z" in capsys.readouterr().out

    assert cli.main(["autoresearch", "register", "--handle", "@m0hawk", "--model", "glm-5.3-flash", "--force"]) == 0
    assert seen["register"] == ("m0hawk", None, "glm-5.3-flash")
