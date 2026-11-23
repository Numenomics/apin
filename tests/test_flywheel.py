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


def test_seed_from_apin_generates_valid_snapshot_and_sql(tmp_path: Path) -> None:
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
            "dim": 6,
            "hypothesis": "Improved A1 with 'quoted' note",
            "codex_refs": ["fisher-low-rank"],
            "soft_min_eff": 0.0011,
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
    assert summary["alignment"] == 1

    snapshot = json.loads((out_dir / "public" / "snapshot.json").read_text(encoding="utf-8"))
    assert set(snapshot.keys()) == {"nodes", "edges", "frontier", "claims", "alignment"}
    assert snapshot["frontier"]["A1"]["id"] == "A1:glm53_a1_1h:T0002"
    assert snapshot["frontier"]["A1"]["real_stan_passed"] == 2
    assert snapshot["frontier"]["A1"]["real_stan_total"] == 3
    assert snapshot["frontier"]["A1"]["verified"] == 1
    assert snapshot["alignment"][0]["real_stan_pass_rate"] == pytest.approx(2 / 3)

    assert all(e["relation"] == "incumbent" for e in snapshot["edges"])
    assert snapshot["frontier"]["A1"]["state"] == "verified"
    assert all(n["has_source"] is False and "state" in n for n in snapshot["nodes"])

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
