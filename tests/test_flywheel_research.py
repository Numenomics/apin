"""Client and CLI tests for Flywheel registration, sources, parents, notes, search and verification."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import pytest

from harness import autoresearch, cli, flywheel


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    token_file = tmp_path / "config" / "apin" / "flywheel_token"
    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("APIN_FLYWHEEL_TOKEN", raising=False)
    monkeypatch.setenv("APIN_FLYWHEEL_URL", "https://flywheel.example.dev")
    return token_file


class _Resp:
    def __init__(self, payload: Any) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeFlywheel:
    """Routes urllib requests to handlers keyed by (method, path); records every call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], Callable[[dict[str, Any]], Any]] = {}
        monkeypatch.setattr("urllib.request.urlopen", self._urlopen)

    def on(self, method: str, path: str, handler: Callable[[dict[str, Any]], Any] | Any) -> None:
        self.routes[(method, path)] = handler if callable(handler) else (lambda _c, h=handler: h)

    def _urlopen(self, req, timeout=10.0):
        parsed = urllib.parse.urlsplit(req.full_url)
        call = {
            "method": req.get_method(),
            "path": urllib.parse.unquote(parsed.path),
            "query": dict(urllib.parse.parse_qsl(parsed.query)),
            "auth": req.get_header("Authorization"),
            "body": json.loads(req.data.decode("utf-8")) if req.data else None,
        }
        self.calls.append(call)
        handler = self.routes.get((call["method"], call["path"]))
        if handler is None:
            raise AssertionError(f"unexpected request {call['method']} {call['path']}")
        result = handler(call)
        if isinstance(result, Exception):
            raise result
        return _Resp(result)


def _http_error(status: int, body: dict[str, Any]):
    import io
    import urllib.error

    return urllib.error.HTTPError("u", status, "err", hdrs=None, fp=io.BytesIO(json.dumps(body).encode()))


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def test_token_comes_from_env_first_then_the_0600_file(_isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert flywheel._resolve_token() is None
    saved = flywheel.save_token("apfw_from_file")
    assert saved == _isolated
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert stat.S_IMODE(saved.parent.stat().st_mode) == 0o700
    assert flywheel._resolve_token() == "apfw_from_file"
    with pytest.raises(FileExistsError):
        flywheel.save_token("apfw_other")

    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN", "apfw_from_env")
    assert flywheel._resolve_token() == "apfw_from_env"
    assert flywheel._resolve_token("") is None  # explicit "no token" still wins


def test_token_lives_in_the_checkout_and_the_user_file_is_only_a_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # tests/conftest.py points every test at an empty token file and clears the environment token.
    assert "APIN_FLYWHEEL_TOKEN" not in os.environ
    assert not flywheel.token_path().exists() and flywheel._resolve_token() is None

    # Without the override the token belongs to the checkout, so two clones are two contributors.
    monkeypatch.delenv("APIN_FLYWHEEL_TOKEN_FILE", raising=False)
    repo_token = Path(flywheel.__file__).resolve().parent.parent / ".apin" / "flywheel_token"
    assert flywheel.token_path() == repo_token

    # A token saved in the user's config directory before this change still works as a fallback...
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "apin").mkdir()
    (tmp_path / "apin" / "flywheel_token").write_text("apfw_user_token\n")
    if not repo_token.exists():
        assert flywheel._resolve_token() == "apfw_user_token"
    # ...but never when a token file is named explicitly.
    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN_FILE", str(tmp_path / "none"))
    assert flywheel._resolve_token() is None


def test_registered_model_is_sent_with_trials(_isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APIN_AGENT_MODEL", raising=False)
    flywheel.save_token("apfw_x")
    flywheel.save_identity("ada", "claude-code", "claude-opus-5-5")
    assert flywheel.read_identity()["model"] == "claude-opus-5-5"
    sent: dict[str, Any] = {}
    monkeypatch.setattr(flywheel, "post_node", lambda payload, **kw: sent.update(payload) or {"ok": True})
    flywheel.post_trial_row({"trial_id": "T0001", "lane": "A1", "decision": "keep_baseline"}, run_id="r1")
    assert sent["agent_model"] == "claude-opus-5-5"


def test_cli_register_saves_token_once_and_refuses_to_clobber(
    _isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFlywheel(monkeypatch)
    fake.on("POST", "/api/register", {
        "ok": True, "handle": "ada", "role": "provisional", "hourly_limit": 60,
        "daily_write_limit": 300, "token": "apfw_secret_once",
    })
    assert cli.main(["autoresearch", "register", "--handle", "ada", "--agent", "claude-code", "--model", "opus"]) == 0
    assert fake.calls[0]["auth"] is None
    assert fake.calls[0]["body"] == {"handle": "ada", "agent": "claude-code", "model": "opus"}
    assert _isolated.read_text().strip() == "apfw_secret_once"
    assert stat.S_IMODE(_isolated.stat().st_mode) == 0o600
    out = capsys.readouterr().out
    assert out.count("apfw_secret_once") == 1
    assert "provisional" in out

    assert cli.main(["autoresearch", "register", "--handle", "ada2"]) == 1
    assert "--force" in capsys.readouterr().err
    assert len(fake.calls) == 1

    fake.on("POST", "/api/register", lambda c: _http_error(429, {"error": "registration_ip_cap", "limit": 5}))
    assert cli.main(["autoresearch", "register", "--handle", "ada3", "--force"]) == 1
    err = capsys.readouterr().err
    assert "registration_ip_cap" in err
    assert _isolated.read_text().strip() == "apfw_secret_once"


# ---------------------------------------------------------------------------
# Posting trials with source and parents
# ---------------------------------------------------------------------------


def test_post_trial_row_uploads_source_and_parents(_isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flywheel.save_token("apfw_tok")
    fake = FakeFlywheel(monkeypatch)
    fake.on("POST", "/api/nodes", lambda c: {"ok": True, "node": {"id": c["body"]["id"]}})
    links = flywheel.parent_links("A1:r:T0001", combines=["A3:x:T0004"], fork_of=["A1:y:T0002"])
    row = {"trial_id": "T0002", "lane": "A1", "decision": "keep", "candidate_sha": "abc", "hypothesis": "h"}

    assert flywheel.post_trial_row(row, run_id="r", parent_id="A1:r:T0001", parents=links, source="print(1)\n")
    body = fake.calls[0]["body"]
    assert fake.calls[0]["auth"] == "Bearer apfw_tok"
    assert body["source"] == "print(1)\n"
    assert body["parent_id"] == "A1:r:T0001"
    assert body["parent_ids"] == [
        {"id": "A1:r:T0001", "relation": "incumbent"},
        {"id": "A3:x:T0004", "relation": "combines"},
        {"id": "A1:y:T0002", "relation": "fork"},
    ]

    # Oversized sources are skipped client-side; the trial still posts.
    flywheel.post_trial_row(row, run_id="r", source="#" * (flywheel.MAX_SOURCE_BYTES + 1))
    assert "source" not in fake.calls[1]["body"]


def test_sync_ledger_attaches_trial_snapshots_and_ledger_parents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "trials" / "T0001").mkdir(parents=True)
    (run_dir / "trials" / "T0001" / "submission.py").write_text("print('t1')\n")
    rows = [
        {"trial_id": "T0001", "lane": "A2", "decision": "keep_baseline"},
        {"trial_id": "T0002", "lane": "A2", "decision": "keep", "parents": [{"id": "A3:o:T0009", "relation": "combines"}]},
    ]
    (run_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fake = FakeFlywheel(monkeypatch)
    fake.on("POST", "/api/nodes", {"ok": True})
    flywheel.sync_ledger(run_dir / "ledger.jsonl", lane="A2", run_id="r", token="tok")
    first, second = (c["body"] for c in fake.calls)
    assert first["source"] == "print('t1')\n"
    assert "source" not in second
    assert second["parent_ids"] == [
        {"id": "A2:r:T0001", "relation": "incumbent"},
        {"id": "A3:o:T0009", "relation": "combines"},
    ]


def test_append_row_posts_source_and_parents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(flywheel, "post_trial_row", lambda row, **kw: seen.update(kw))
    autoresearch._append_row(
        tmp_path / "ledger.jsonl", tmp_path / "results.tsv", {"trial_id": "T0001", "decision": "keep"},
        run_id="r", parent_id="A1:r:T0000", parents=[{"id": "A1:r:T0000", "relation": "incumbent"}], source="src",
    )
    assert seen == {"run_id": "r", "parent_id": "A1:r:T0000",
                    "parents": [{"id": "A1:r:T0000", "relation": "incumbent"}], "source": "src"}


def test_cli_eval_passes_combines_and_fork_of(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    def fake_eval(lane, hypothesis, **kw):
        seen.update(kw)
        return {"trial_id": "T0001", "lane": lane, "decision": "keep", "tier0_ok": True, "tier1_ok": True,
                "wall_s": 1.0}

    monkeypatch.setattr(autoresearch, "evaluate_lane_trial", fake_eval)
    rc = cli.main(["autoresearch", "eval", "--lane", "A1", "--hypothesis", "h",
                   "--combines", "A3:x:T0004,A2:y:T0001", "--combines", "A4:z:T0002", "--fork-of", "A1:w:T0007"])
    assert rc == 0
    assert seen["combines"] == ["A3:x:T0004", "A2:y:T0001", "A4:z:T0002"]
    assert seen["fork_of"] == ["A1:w:T0007"]


def test_evaluate_lane_trial_records_extra_parents_in_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A duplicate submission is rejected before any evaluation, so this runs instantly."""
    lane_dir = tmp_path / "lanes" / "A1"
    run_dir = lane_dir / "runs" / "r"
    run_dir.mkdir(parents=True)
    sub = lane_dir / "submission.py"
    sub.write_text("x = 1\n")
    sha = hashlib.sha256(sub.read_text().encode()).hexdigest()[:16]
    (run_dir / "ledger.jsonl").write_text(json.dumps({"trial_id": "T0001", "decision": "keep", "candidate_sha": sha}) + "\n")
    posted: dict[str, Any] = {}
    monkeypatch.setattr(flywheel, "post_trial_row", lambda row, **kw: posted.update(row=row, **kw))
    row = autoresearch.evaluate_lane_trial("A1", "dup", autoresearch_dir=tmp_path, run_id="r",
                                           combines=["A3:x:T0004"], use_queue=False)
    assert row["decision"] == "discard_duplicate"
    assert row["parents"] == [{"id": "A3:x:T0004", "relation": "combines"}]
    assert posted["parent_id"] == "A1:r:T0001"
    assert posted["parents"] == [{"id": "A1:r:T0001", "relation": "incumbent"},
                                 {"id": "A3:x:T0004", "relation": "combines"}]
    assert posted["source"] == "x = 1\n"


# ---------------------------------------------------------------------------
# fetch, notes, state, search
# ---------------------------------------------------------------------------


def _source_payload(node_id: str, source: str) -> dict[str, Any]:
    return {"node_id": node_id, "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "size": len(source.encode()), "code_sha": None, "source": source}


def test_cli_fetch_writes_a_candidate_never_over_submission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "autoresearch" / "lanes" / "A1").mkdir(parents=True, exist_ok=True)  # the lane must exist locally
    fake = FakeFlywheel(monkeypatch)
    node_id = "A1:run:T0004"
    fake.on("GET", f"/api/nodes/{node_id}/source", _source_payload(node_id, "def adapt(): pass\n"))
    fake.on("GET", f"/api/nodes/{node_id}", {"node": {"id": node_id, "lane": "A1"}})

    assert cli.main(["autoresearch", "fetch", "--node", node_id]) == 0
    out = tmp_path / "autoresearch" / "lanes" / "A1" / "runs" / "default" / "candidates" / "A1_run_T0004.py"
    assert out.read_text() == "def adapt(): pass\n"
    assert "--fork-of A1:run:T0004" in capsys.readouterr().out
    assert fake.calls[0]["auth"] is None  # public read

    sub = tmp_path / "submission.py"
    sub.write_text("mine\n")
    assert cli.main(["autoresearch", "fetch", "--node", node_id, "--out", str(sub)]) == 1
    assert sub.read_text() == "mine\n"
    assert cli.main(["autoresearch", "fetch", "--node", node_id, "--out", str(sub), "--force"]) == 0
    assert sub.read_text() == "def adapt(): pass\n"

    tampered = _source_payload(node_id, "x\n")
    tampered["source"] = "evil\n"
    fake.on("GET", f"/api/nodes/{node_id}/source", tampered)
    assert cli.main(["autoresearch", "fetch", "--node", node_id, "--out", str(tmp_path / "t.py")]) == 1
    assert "sha256" in capsys.readouterr().err
    assert not (tmp_path / "t.py").exists()


def test_cli_note_add_list_set_state_and_trial_set_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = FakeFlywheel(monkeypatch)
    note = {"id": "note_1", "kind": "hypothesis", "lane": "A1", "title": "Robust Fisher", "body": "MAD",
            "state": "open", "author": "ada", "node_ids": ["A1:r:T0002"], "updated_at": "2026-11-23T00:00:00Z"}
    fake.on("POST", "/api/notes", lambda c: {"ok": True, "note": {**note, **c["body"]}})
    fake.on("GET", "/api/notes", {"notes": [note], "count": 1})
    fake.on("PATCH", "/api/notes/note_1", lambda c: {"ok": True, "note": {**note, **c["body"]}})
    fake.on("POST", "/api/nodes/A1:r:T0002/state", lambda c: {"ok": True, "node": {"id": "A1:r:T0002", **c["body"]}})

    # Writes need a token; reads do not.
    assert cli.main(["autoresearch", "note", "add", "--kind", "hypothesis", "--title", "t"]) == 1
    assert "register" in capsys.readouterr().err
    flywheel.save_token("apfw_tok")

    body_file = tmp_path / "body.md"
    body_file.write_text("long body")
    assert cli.main(["autoresearch", "note", "add", "--kind", "hypothesis", "--lane", "A1", "--title", "Robust Fisher",
                     "--body-file", str(body_file), "--node", "A1:r:T0002,A1:r:T0003"]) == 0
    assert fake.calls[-1]["body"] == {"kind": "hypothesis", "title": "Robust Fisher", "lane": "A1",
                                      "body": "long body", "node_ids": ["A1:r:T0002", "A1:r:T0003"]}
    assert "note_1" in capsys.readouterr().out

    assert cli.main(["autoresearch", "note", "list", "--lane", "A1", "--kind", "hypothesis", "--state", "open"]) == 0
    assert fake.calls[-1]["query"] == {"lane": "A1", "kind": "hypothesis", "state": "open", "limit": "20"}
    assert "Robust Fisher" in capsys.readouterr().out

    assert cli.main(["autoresearch", "note", "set-state", "--note", "note_1", "--state", "refuted", "--reason", "T0007"]) == 0
    assert fake.calls[-1]["body"] == {"state": "refuted", "reason": "T0007"}
    assert "refuted" in capsys.readouterr().out

    assert cli.main(["autoresearch", "set-state", "--node", "A1:r:T0002", "--state", "retracted", "--reason", "bug"]) == 0
    assert fake.calls[-1]["body"] == {"state": "retracted", "reason": "bug"}
    assert fake.calls[-1]["auth"] == "Bearer apfw_tok"


def test_cli_search_prints_hits_and_paging(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeFlywheel(monkeypatch)
    fake.on("GET", "/api/search", {
        "results": [
            {"kind": "node", "id": "A3:p:T0001", "title": "T0001: zigzag", "snippet": "[zigzag] PDMP", "author": "ada",
             "decision": "keep", "state": "unverified", "soft_min_eff": 0.002, "has_source": True},
            {"kind": "note", "id": "note_9", "lane": "A3", "title": "Refresh", "snippet": "[zigzag] needs",
             "author": "bo", "note_kind": "insight", "state": "open"},
        ],
        "next_offset": 2,
    })
    assert cli.main(["autoresearch", "search", "zigzag", "--lane", "A3", "--limit", "2"]) == 0
    assert fake.calls[0]["query"] == {"q": "zigzag", "lane": "A3", "limit": "2"}
    out = capsys.readouterr().out
    assert "A3:p:T0001" in out and "[source]" in out and "note_9" in out and "--offset 2" in out


# ---------------------------------------------------------------------------
# verify-next and brief
# ---------------------------------------------------------------------------


def test_cli_verify_next_leases_reruns_and_posts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "autoresearch" / "lanes" / "A1").mkdir(parents=True, exist_ok=True)  # the lane must exist locally
    flywheel.save_token("apfw_verifier")
    fake = FakeFlywheel(monkeypatch)
    node_id = "A1:run:T0004"
    fake.on("POST", "/api/verify/claim", {
        "ok": True,
        "lease": {"node_id": node_id, "lane": "A1", "lease_token": "vlease_abc", "lease_expires_at": "2026-11-23T02:00:00Z"},
        "node": {"id": node_id, "lane": "A1", "soft_min_eff": 0.002, "dims": [6, 20]},
    })
    fake.on("GET", f"/api/nodes/{node_id}/source", _source_payload(node_id, "def adapt(): pass\n"))
    fake.on("POST", f"/api/verify/{node_id}/result", lambda c: {"ok": True, "result": "pass" if c["body"]["passed"] else "fail"})
    seen: dict[str, Any] = {}

    def fake_rerun(path, lane, **kw):
        seen.update(path=path, lane=lane, **kw)
        return {"passed": True, "metrics": {"soft_min_eff": 0.0019, "seed": 31}, "reason": "reproduced"}

    monkeypatch.setattr(autoresearch, "rerun_for_verification", fake_rerun)
    assert cli.main(["autoresearch", "verify-next", "--lane", "A1", "--repeats", "1"]) == 0
    assert fake.calls[0]["body"] == {"lane": "A1", "ttl_minutes": 120}
    assert seen["path"].read_text() == "def adapt(): pass\n"
    assert seen["path"].parent == tmp_path / "autoresearch" / "lanes" / "A1" / "runs" / "verify" / "A1_run_T0004"
    assert seen["claimed_soft_min"] == 0.002 and seen["dims"] == (6, 20) and seen["repeats"] == 1
    result = fake.calls[-1]
    assert result["body"] == {"lease_token": "vlease_abc", "passed": True,
                              "metrics": {"soft_min_eff": 0.0019, "seed": 31}, "note": "reproduced"}
    assert "replicated" in capsys.readouterr().out

    # A crash in the evaluator posts nothing.
    monkeypatch.setattr(autoresearch, "rerun_for_verification", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    n_calls = len(fake.calls)
    assert cli.main(["autoresearch", "verify-next"]) == 1
    assert all(c["path"] != f"/api/verify/{node_id}/result" for c in fake.calls[n_calls:])

    fake.on("POST", "/api/verify/claim", lambda c: _http_error(404, {"error": "queue_empty"}))
    assert cli.main(["autoresearch", "verify-next"]) == 0
    assert "Nothing to verify" in capsys.readouterr().out


def test_brief_lists_open_hypotheses_and_recent_insights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(flywheel, "get_frontier", lambda **_: {"frontier": {}})
    monkeypatch.setattr(flywheel, "get_nodes", lambda **_: {"nodes": []})
    monkeypatch.setattr(flywheel, "get_claims", lambda **_: {"claims": []})
    seen: list[dict[str, Any]] = []

    def fake_notes(**kw):
        seen.append(kw)
        if kw["kind"] == "hypothesis":
            return {"notes": [{"id": "note_h", "title": "Robust Fisher helps funnels", "author": "ada",
                               "state": "open", "body": "MAD", "node_ids": ["A1:r:T0002"]}]}
        return {"notes": [{"id": "note_i", "title": "Warmup starvation games Tier 1", "author": "bo", "state": "open"},
                          {"id": "note_x", "title": "withdrawn", "author": "bo", "state": "retracted"}]}

    monkeypatch.setattr(flywheel, "list_notes", fake_notes)
    brief = autoresearch.build_research_brief("A1", autoresearch_dir=tmp_path, flywheel_url="https://f.example")
    assert "## Open Hypotheses on A1" in brief
    assert "Robust Fisher helps funnels" in brief and "`A1:r:T0002`" in brief
    assert "Warmup starvation games Tier 1" in brief and "withdrawn" not in brief
    assert seen[0]["lane"] == "A1" and seen[0]["state"] == "open"

    def broken(**_):
        raise flywheel.FlywheelError("HTTP 404", status=404)

    monkeypatch.setattr(flywheel, "list_notes", broken)
    brief = autoresearch.build_research_brief("A1", autoresearch_dir=tmp_path, flywheel_url="https://f.example")
    assert "Hypotheses and Insights on A1" in brief and "unavailable" in brief


def test_admin_helpers_hit_handle_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeFlywheel(monkeypatch)
    for action in ("promote", "revoke", "purge"):
        fake.on("POST", f"/api/admin/handles/ada/{action}", {"ok": True})
    flywheel.promote_handle("ada", token="root")
    flywheel.revoke_handle("ada", token="root")
    flywheel.purge_handle("ada", token="root")
    assert [c["path"] for c in fake.calls] == [
        "/api/admin/handles/ada/promote", "/api/admin/handles/ada/revoke", "/api/admin/handles/ada/purge",
    ]
    assert all(c["auth"] == "Bearer root" for c in fake.calls)
    assert os.environ["APIN_FLYWHEEL_URL"] == "https://flywheel.example.dev"


def test_non_finite_numbers_are_sent_as_null() -> None:
    # ratio = inf when the candidate passes and the incumbent fails; bare `Infinity` is not JSON
    # and the Worker would reject the whole trial.
    body = flywheel.strict_json({"ratio_vs_incumbent": float("inf"), "b2_max": float("nan"), "by_class": [1.0, float("-inf")]})
    assert json.loads(body, parse_constant=lambda name: pytest.fail(name)) == {"b2_max": None, "by_class": [1.0, None], "ratio_vs_incumbent": None}


def test_cli_refuses_a_lane_name_that_would_move_a_write(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # The lane comes from the server. `../..` or an absolute path must never become a directory.
    from harness import cli

    for bad in ("../../../tmp/x", "/tmp/x", "A1/../../x", "", "ZZ"):
        with pytest.raises(ValueError):
            cli._local_lane(bad)
    assert cli._local_lane("A1") == "A1"
    # A trial id made only of dots must not become `.` or `..` in a path.
    assert cli._safe_name("..") not in ("", ".", "..") and cli._safe_name("...").startswith("trial_")
    assert cli._safe_name("A1:run:T0001") == "A1_run_T0001"

    monkeypatch.setattr(flywheel, "get_source", lambda node_id, **kw: {"source": "x = 1\n", "sha256": __import__("hashlib").sha256(b"x = 1\n").hexdigest(), "size": 6})
    monkeypatch.setattr(flywheel, "get_node", lambda node_id, **kw: {"node": {"id": node_id, "lane": "../../../../tmp/apin_escape"}})
    rc = cli.main(["autoresearch", "fetch", "--node", "A1:run:T0001"])
    assert rc == 1 and "does not have" in capsys.readouterr().err
    assert not Path("/tmp/apin_escape").exists()
