"""Lightweight stdlib-only Python client for apin Flywheel.

The token comes from ``APIN_FLYWHEEL_TOKEN`` first, then from the token file that
``apin autoresearch register`` writes (``$XDG_CONFIG_HOME/apin/flywheel_token``, by
default ``~/.config/apin/flywheel_token``, mode 0600; ``APIN_FLYWHEEL_TOKEN_FILE``
overrides the path). ``APIN_FLYWHEEL_URL`` picks the server; with a token and no URL
the public Flywheel is used. Writes cleanly no-op when no token is configured, so
autoresearch runs work identically offline.
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ENV_URL = "APIN_FLYWHEEL_URL"
ENV_TOKEN = "APIN_FLYWHEEL_TOKEN"
ENV_TOKEN_FILE = "APIN_FLYWHEEL_TOKEN_FILE"
DEFAULT_PUBLIC_URL = "https://apin-flywheel.m0hawk.workers.dev"
KEPT_DECISIONS = {"keep", "keep_baseline", "keep_simpler", "keep_incumbent_failed"}
#: The Worker refuses sources above this size; the client skips the upload instead.
MAX_SOURCE_BYTES = 256 * 1024
PARENT_RELATIONS = ("incumbent", "combines", "fork")

_METRIC_KEYS = (
    "loc", "magic_constants", "wall_s", "queue_wait_s", "archive_cells",
    "tier0_ok", "tier1_ok", "tier0_info", "incumbent_soft_min",
    "median_adapt_cost", "judge_digest", "seed", "repeats", "by_class",
)


class FlywheelError(RuntimeError):
    """Raised when a configured Flywheel endpoint returns an HTTP or network error."""

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


# ---------------------------------------------------------------------------
# Configuration: URL and token
# ---------------------------------------------------------------------------


def strict_json(value: Any) -> str:
    """Serialise for the Flywheel: NaN and infinities become null.

    Python writes them as bare `NaN` / `Infinity`, which is not JSON. The Worker and every
    browser reject such a body, so a trial with an infinite ratio (candidate passed, incumbent
    failed) or an infinite bias would never be posted.
    """

    def clean(v: Any) -> Any:
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    return json.dumps(clean(value), sort_keys=True, allow_nan=False)


def token_path() -> Path:
    """Where `apin autoresearch register` saves the personal token."""
    override = os.environ.get(ENV_TOKEN_FILE)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "apin" / "flywheel_token"


def _read_token_file() -> str | None:
    path = token_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    cleaned = text.strip().splitlines()[0].strip() if text.strip() else ""
    return cleaned or None


def save_token(token: str, *, path: Path | None = None, force: bool = False) -> Path:
    """Write the token to the user config file with mode 0600 (directory 0700)."""
    target = path or token_path()
    if target.exists() and not force:
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (token.strip() + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    return target


def _resolve_url(url: str | None = None, token: str | None = None) -> str | None:
    raw = url if url is not None else os.environ.get(ENV_URL)
    if raw:
        cleaned = raw.strip().rstrip("/")
        if cleaned:
            return cleaned
    # When a token is configured without APIN_FLYWHEEL_URL, default to the public Flywheel.
    if _resolve_token(token) is not None:
        return DEFAULT_PUBLIC_URL
    return None


def _resolve_token(token: str | None = None) -> str | None:
    """Explicit argument (``""`` means none), then APIN_FLYWHEEL_TOKEN, then the token file."""
    if token is not None:
        return token.strip() or None
    env = (os.environ.get(ENV_TOKEN) or "").strip()
    if env:
        return env
    return _read_token_file()


def configured(url: str | None = None, token: str | None = None) -> bool:
    """Return True when a Flywheel base URL (or a token targeting the default public URL) is configured."""
    return _resolve_url(url, token) is not None


def read_url(url: str | None = None) -> str:
    """Base URL for public reads: explicit, then APIN_FLYWHEEL_URL, then the public Flywheel."""
    return _resolve_url(url, "") or DEFAULT_PUBLIC_URL


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    base_url = _resolve_url(url, token)
    if not base_url:
        return None

    full_url = f"{base_url}{path}"
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": "apin-flywheel-client/0.2 (+https://github.com/Numenomics/apin)",
    }
    data: bytes | None = None

    if payload is not None:
        data = strict_json(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    resolved_token = _resolve_token(token)
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    req = urllib.request.Request(full_url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_err: Any = json.loads(raw_err) if raw_err else None
        except json.JSONDecodeError:
            parsed_err = raw_err
        raise FlywheelError(
            f"Flywheel {method} {path} failed with HTTP {exc.code}: {raw_err}",
            status=exc.code,
            body=parsed_err,
        ) from exc
    except urllib.error.URLError as exc:
        raise FlywheelError(f"Flywheel {method} {path} network error: {exc.reason}") from exc


def _q(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _qs(params: dict[str, Any]) -> str:
    clean = {k: str(v) for k, v in params.items() if v is not None and v != ""}
    return f"?{urllib.parse.urlencode(clean)}" if clean else ""


# ---------------------------------------------------------------------------
# Registration and maintainer operations
# ---------------------------------------------------------------------------


def register(
    handle: str,
    *,
    agent: str | None = None,
    model: str | None = None,
    invite_code: str | None = None,
    url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Register a handle. Returns the server response, whose ``token`` is shown only once.

    Without an invite code the token is ``provisional``; a valid invite code gives ``contributor``.
    """
    payload: dict[str, Any] = {"handle": handle}
    for key, val in (("agent", agent), ("model", model), ("invite_code", invite_code)):
        if val:
            payload[key] = val
    resp = _request_json("POST", "/api/register", payload, url=read_url(url), token="", timeout=timeout)
    return resp or {}


def register_contributor(
    handle: str,
    invite_code: str,
    *,
    url: str = DEFAULT_PUBLIC_URL,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Self-register a personal revocable contributor token using a community invite code."""
    return register(handle, invite_code=invite_code, url=url, timeout=timeout)


def mint_key(
    handle: str,
    *,
    role: str = "contributor",
    hourly_limit: int = 120,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Admin-only: mint a personal revocable contributor/admin/provisional key."""
    return _request_json(
        "POST", "/api/admin/keys", {"handle": handle, "role": role, "hourly_limit": hourly_limit},
        url=url, token=token, timeout=timeout,
    )


def revoke_key(
    key_id: str,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Admin-only: revoke an issued API key by its key_id."""
    return _request_json("DELETE", f"/api/admin/keys/{_q(key_id)}", url=url, token=token, timeout=timeout)


def _handle_action(action: str, handle: str, url: str | None, token: str | None, timeout: float) -> dict[str, Any] | None:
    return _request_json("POST", f"/api/admin/handles/{_q(handle)}/{action}", {}, url=url, token=token, timeout=timeout)


def promote_handle(handle: str, *, url: str | None = None, token: str | None = None, timeout: float = 10.0) -> dict[str, Any] | None:
    """Admin-only: move every provisional key of ``handle`` to the contributor tier."""
    return _handle_action("promote", handle, url, token, timeout)


def revoke_handle(handle: str, *, url: str | None = None, token: str | None = None, timeout: float = 10.0) -> dict[str, Any] | None:
    """Admin-only: revoke every key of ``handle``."""
    return _handle_action("revoke", handle, url, token, timeout)


def purge_handle(handle: str, *, url: str | None = None, token: str | None = None, timeout: float = 10.0) -> dict[str, Any] | None:
    """Admin-only: revoke every key of ``handle`` and delete all its trials, notes, claims and sources."""
    return _handle_action("purge", handle, url, token, timeout)


def verify_node(
    node_id: str,
    *,
    verified: bool = True,
    real_stan_passed: int | None = None,
    real_stan_total: int | None = None,
    real_ess_per_cost: float | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Admin-only: mark a node verified=1 (optionally attaching reproduced real-Stan metrics)."""
    payload: dict[str, Any] = {"verified": verified}
    if real_stan_passed is not None:
        payload["real_stan_passed"] = int(real_stan_passed)
    if real_stan_total is not None:
        payload["real_stan_total"] = int(real_stan_total)
    if real_ess_per_cost is not None:
        payload["real_ess_per_cost"] = float(real_ess_per_cost)
    return _request_json("POST", f"/api/admin/nodes/{_q(node_id)}/verify", payload, url=url, token=token, timeout=timeout)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_frontier(
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch the current frontier per lane. Returns None if Flywheel is unconfigured."""
    return _request_json("GET", "/api/frontier", url=url, token=token, timeout=timeout)


def get_nodes(
    *,
    lane: str | None = None,
    limit: int = 30,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch recent trial nodes (optionally filtered by lane). Returns None if unconfigured."""
    return _request_json("GET", f"/api/nodes{_qs({'limit': int(limit), 'lane': lane})}", url=url, token=token, timeout=timeout)


def get_node(node_id: str, *, url: str | None = None, timeout: float = 10.0) -> dict[str, Any] | None:
    """One trial with ``parents`` (id, relation), ``children``, ``edges`` and ``verification``."""
    return _request_json("GET", f"/api/nodes/{_q(node_id)}", url=read_url(url), token="", timeout=timeout)


def get_source(node_id: str, *, url: str | None = None, timeout: float = 20.0) -> dict[str, Any] | None:
    """``{node_id, sha256, size, code_sha, source}`` for a trial that uploaded its source."""
    return _request_json("GET", f"/api/nodes/{_q(node_id)}/source", url=read_url(url), token="", timeout=timeout)


def get_claims(
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch active task claims. Returns None if Flywheel is unconfigured."""
    return _request_json("GET", "/api/claims", url=url, token=token, timeout=timeout)


def list_notes(
    *,
    lane: str | None = None,
    kind: str | None = None,
    state: str | None = None,
    author: str | None = None,
    limit: int = 50,
    offset: int = 0,
    url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Hypotheses and insights, newest first."""
    qs = _qs({"lane": lane, "kind": kind, "state": state, "author": author, "limit": limit, "offset": offset or None})
    return _request_json("GET", f"/api/notes{qs}", url=read_url(url), token="", timeout=timeout)


def search(
    q: str,
    *,
    lane: str | None = None,
    kind: str | None = None,
    limit: int = 20,
    offset: int = 0,
    url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Search trials and notes; every term must match. Page with ``offset`` / ``next_offset``."""
    qs = _qs({"q": q, "lane": lane, "kind": kind, "limit": limit, "offset": offset or None})
    return _request_json("GET", f"/api/search{qs}", url=read_url(url), token="", timeout=timeout)


def verify_queue(
    *,
    lane: str | None = None,
    status: str | None = None,
    limit: int = 50,
    url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Trials waiting for an independent rerun (``status``: open|queued|leased|done|all)."""
    qs = _qs({"lane": lane, "status": status, "limit": limit})
    return _request_json("GET", f"/api/verify/queue{qs}", url=read_url(url), token="", timeout=timeout)


# ---------------------------------------------------------------------------
# Writes (need a token)
# ---------------------------------------------------------------------------


def claim_task(
    task_id: str,
    lane: str,
    agent: str,
    summary: str = "",
    ttl_minutes: int = 60,
    *,
    force: bool = False,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Claim a research task on Flywheel with a TTL lease. Returns None if unconfigured."""
    payload: dict[str, Any] = {
        "task_id": task_id,
        "lane": lane,
        "agent": agent,
        "summary": summary,
        "ttl_minutes": ttl_minutes,
        "force": force,
    }
    return _request_json("POST", "/api/claims", payload, url=url, token=token, timeout=timeout)


def release_claim(
    claim_id: str,
    status: str = "done",
    *,
    summary: str | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Mark an active claim as 'done' or 'abandoned'. Returns None if unconfigured."""
    payload: dict[str, Any] = {"status": status}
    if summary is not None:
        payload["summary"] = summary
    return _request_json("PATCH", f"/api/claims/{_q(claim_id)}", payload, url=url, token=token, timeout=timeout)


def post_node(
    node: dict[str, Any],
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Post a single trial node to Flywheel. Returns None if unconfigured."""
    return _request_json("POST", "/api/nodes", node, url=url, token=token, timeout=timeout)


def set_node_state(
    node_id: str,
    state: str,
    reason: str | None = None,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Retract your own trial, or flag another's as needs-replication / disputed (reason required)."""
    payload: dict[str, Any] = {"state": state}
    if reason:
        payload["reason"] = reason
    return _request_json("POST", f"/api/nodes/{_q(node_id)}/state", payload, url=url, token=token, timeout=timeout)


def add_note(
    kind: str,
    title: str,
    *,
    lane: str | None = None,
    body: str | None = None,
    node_ids: list[str] | None = None,
    state: str | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Record a ``hypothesis`` or an ``insight``."""
    payload: dict[str, Any] = {"kind": kind, "title": title}
    for key, val in (("lane", lane), ("body", body), ("state", state)):
        if val:
            payload[key] = val
    if node_ids:
        payload["node_ids"] = list(node_ids)
    return _request_json("POST", "/api/notes", payload, url=url, token=token, timeout=timeout)


def update_note(
    note_id: str,
    *,
    state: str | None = None,
    reason: str | None = None,
    title: str | None = None,
    body: str | None = None,
    node_ids: list[str] | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Change a note's state (with an optional reason) or, as its author, its text and links."""
    payload: dict[str, Any] = {}
    for key, val in (("state", state), ("reason", reason), ("title", title), ("body", body)):
        if val is not None:
            payload[key] = val
    if node_ids is not None:
        payload["node_ids"] = list(node_ids)
    return _request_json("PATCH", f"/api/notes/{_q(note_id)}", payload, url=url, token=token, timeout=timeout)


def verify_claim(
    *,
    lane: str | None = None,
    node_id: str | None = None,
    ttl_minutes: int | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Lease the next queued trial (never your own). The response's ``lease.lease_token`` posts the result."""
    payload: dict[str, Any] = {}
    if lane:
        payload["lane"] = lane
    if node_id:
        payload["node_id"] = node_id
    if ttl_minutes:
        payload["ttl_minutes"] = int(ttl_minutes)
    return _request_json("POST", "/api/verify/claim", payload, url=url, token=token, timeout=timeout)


def verify_result(
    node_id: str,
    lease_token: str,
    passed: bool,
    *,
    metrics: dict[str, Any] | None = None,
    note: str | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Post a rerun result: passed sets the trial ``replicated``, failed sets it ``disputed``."""
    payload: dict[str, Any] = {"lease_token": lease_token, "passed": bool(passed)}
    if metrics is not None:
        payload["metrics"] = metrics
    if note:
        payload["note"] = note[:1000]
    return _request_json("POST", f"/api/verify/{_q(node_id)}/result", payload, url=url, token=token, timeout=timeout)


# ---------------------------------------------------------------------------
# Trial rows -> nodes
# ---------------------------------------------------------------------------


def parent_links(
    incumbent: str | None = None,
    *,
    combines: list[str] | None = None,
    fork_of: list[str] | None = None,
) -> list[dict[str, str]]:
    """``parent_ids`` payload: the incumbent the trial was scored against plus reused mechanisms."""
    links: list[dict[str, str]] = []
    if incumbent:
        links.append({"id": incumbent, "relation": "incumbent"})
    links += [{"id": i, "relation": "combines"} for i in (combines or []) if i]
    links += [{"id": i, "relation": "fork"} for i in (fork_of or []) if i]
    return links


def _source_for_upload(source: str | None) -> str | None:
    if not source or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return None
    return source


def trial_payload(
    row: dict[str, Any],
    *,
    run_id: str,
    lane: str | None = None,
    parent_id: str | None = None,
    parents: list[dict[str, str]] | None = None,
    agent_model: str | None = None,
    author: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the ``POST /api/nodes`` body for one ledger row."""
    row_lane = str(row.get("lane") or lane or "A1")
    trial_id = str(row.get("trial_id") or "T0001")
    hypothesis = str(row.get("hypothesis") or "").strip() or None
    dims = row.get("dims")
    if dims is None and row.get("dim") is not None:
        dims = [int(row["dim"])]
    title = (
        f"{trial_id}: {hypothesis[:73]}..." if hypothesis and len(hypothesis) > 76
        else (f"{trial_id}: {hypothesis}" if hypothesis else f"{trial_id}: {run_id}")
    )
    links = list(parents) if parents is not None else parent_links(parent_id)
    for extra in row.get("parents") or []:
        if isinstance(extra, dict) and extra.get("id") and extra not in links:
            links.append({"id": str(extra["id"]), "relation": str(extra.get("relation") or "combines")})
    payload: dict[str, Any] = {
        "id": f"{row_lane}:{run_id}:{trial_id}",
        "parent_id": parent_id,
        "lane": row_lane,
        "run_id": run_id,
        "trial_id": trial_id,
        "agent_model": agent_model or row.get("agent_model"),
        "author": author or row.get("author"),
        "title": title,
        "hypothesis": hypothesis,
        "decision": str(row.get("decision") or "unknown"),
        "soft_min_eff": row.get("soft_min_eff"),
        "ratio_vs_incumbent": row.get("ratio_vs_incumbent"),
        "short_horizon_eff": row.get("short_horizon_soft_min"),
        "long_horizon_eff": row.get("long_horizon_soft_min"),
        "overhead_ratio": row.get("overhead_ratio"),
        "b2_max": row.get("b2_max"),
        "dims": dims,
        "codex_refs": row.get("codex_refs") or [],
        "code_sha": row.get("candidate_sha") or row.get("code_sha"),
        "metrics_json": strict_json({k: row[k] for k in _METRIC_KEYS if k in row}),
        "verified": 0,
        "created_at": row.get("timestamp"),
    }
    if links:
        payload["parent_ids"] = links
    upload = _source_for_upload(source)
    if upload is not None:
        payload["source"] = upload
    return payload


def _trial_source(run_dir: Path, trial_id: str) -> str | None:
    for name in ("submission.py", "submission_snapshot.py"):
        path = run_dir / "trials" / trial_id / name
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None
    return None


def sync_ledger(
    ledger_path: str | Path,
    lane: str,
    run_id: str,
    agent_model: str | None = None,
    *,
    author: str | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Sync every trial row from a local ledger.jsonl to Flywheel, preserving incumbent lineage.

    Each trial's snapshot under ``trials/<trial_id>/`` is uploaded as its source when present.
    Re-syncing is idempotent. Cleanly no-ops and returns [] when Flywheel is unconfigured.
    """
    if not configured(url, token):
        return []

    path = Path(ledger_path)
    if not path.exists():
        return []

    results: list[dict[str, Any]] = []
    incumbent_id: str | None = None

    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            trial_id = str(row.get("trial_id") or f"T{idx:04d}")
            row = {**row, "trial_id": trial_id}
            payload = trial_payload(
                row,
                run_id=run_id,
                lane=lane,
                parent_id=incumbent_id,
                agent_model=agent_model,
                author=author or agent_model,
                source=_trial_source(path.parent, trial_id),
            )
            resp = post_node(payload, url=url, token=token, timeout=timeout)
            if resp is not None:
                results.append(resp)
            if payload["decision"] in KEPT_DECISIONS:
                incumbent_id = payload["id"]

    return results


def post_trial_row(
    row: dict[str, Any],
    *,
    run_id: str,
    parent_id: str | None = None,
    parents: list[dict[str, str]] | None = None,
    source: str | None = None,
    agent_model: str | None = None,
    author: str | None = None,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    """Best-effort publish of one autoresearch trial row (with its source) to Flywheel.

    No-ops when no token is configured (argument, APIN_FLYWHEEL_TOKEN or the token file),
    so offline or read-only runs never fail or stall.
    """
    resolved_token = _resolve_token(token)
    if resolved_token is None and url is None:
        return None
    if not configured(url, resolved_token):
        return None
    payload = trial_payload(
        row,
        run_id=run_id,
        parent_id=parent_id,
        parents=parents,
        agent_model=agent_model or os.environ.get("APIN_AGENT_MODEL"),
        author=author or os.environ.get("APIN_AUTHOR"),
        source=source,
    )
    try:
        return post_node(payload, url=url, token=resolved_token, timeout=timeout)
    except FlywheelError as exc:
        # An outage is not the contributor's problem and stays quiet. A refusal is: the trial ran
        # but is missing from the shared record, so say why (bad id, locked trial, daily cap).
        if exc.status is not None and 400 <= exc.status < 500:
            print(f"Flywheel did not accept trial {payload.get('id')}: {exc}", file=sys.stderr)
        return None
