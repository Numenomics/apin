"""Lightweight stdlib-only Python client for apin Flywheel.

Reads ``APIN_FLYWHEEL_URL`` and ``APIN_FLYWHEEL_TOKEN`` from the environment and
cleanly no-ops when unconfigured so autoresearch runs work identically offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ENV_URL = "APIN_FLYWHEEL_URL"
ENV_TOKEN = "APIN_FLYWHEEL_TOKEN"
KEPT_DECISIONS = {"keep", "keep_baseline", "keep_simpler"}


class FlywheelError(RuntimeError):
    """Raised when a configured Flywheel endpoint returns an HTTP or network error."""

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _resolve_url(url: str | None = None) -> str | None:
    raw = url if url is not None else os.environ.get(ENV_URL)
    if not raw:
        return None
    cleaned = raw.strip().rstrip("/")
    return cleaned or None


def _resolve_token(token: str | None = None) -> str | None:
    raw = token if token is not None else os.environ.get(ENV_TOKEN)
    if not raw:
        return None
    cleaned = raw.strip()
    return cleaned or None


def configured(url: str | None = None) -> bool:
    """Return True when a Flywheel base URL is configured."""
    return _resolve_url(url) is not None


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    base_url = _resolve_url(url)
    if not base_url:
        return None

    full_url = f"{base_url}{path}"
    headers: dict[str, str] = {"Accept": "application/json"}
    data: bytes | None = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
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


def get_frontier(
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch the current frontier per lane. Returns None if Flywheel is unconfigured."""
    return _request_json("GET", "/api/frontier", url=url, token=token, timeout=timeout)


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
    encoded_id = urllib.parse.quote(claim_id, safe="")
    return _request_json(
        "PATCH", f"/api/claims/{encoded_id}", payload, url=url, token=token, timeout=timeout
    )


def post_node(
    node: dict[str, Any],
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Post a single trial node to Flywheel. Returns None if unconfigured."""
    return _request_json("POST", "/api/nodes", node, url=url, token=token, timeout=timeout)


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

    Cleanly no-ops and returns [] when Flywheel is unconfigured.
    """
    if not configured(url):
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
            row_lane = str(row.get("lane") or lane)
            node_id = f"{row_lane}:{run_id}:{trial_id}"
            decision = str(row.get("decision") or "unknown")
            hypothesis = str(row.get("hypothesis") or "").strip() or None

            dims = row.get("dims")
            if dims is None and row.get("dim") is not None:
                dims = [int(row["dim"])]

            extra_metrics = {
                k: row[k]
                for k in (
                    "loc",
                    "magic_constants",
                    "wall_s",
                    "queue_wait_s",
                    "archive_cells",
                    "tier0_ok",
                    "tier1_ok",
                    "tier0_info",
                    "incumbent_soft_min",
                    "median_adapt_cost",
                    "judge_digest",
                    "seed",
                    "repeats",
                )
                if k in row
            }

            title = f"{trial_id}: {hypothesis[:73]}..." if hypothesis and len(hypothesis) > 76 else (
                f"{trial_id}: {hypothesis}" if hypothesis else f"{trial_id}: {run_id}"
            )

            node_payload: dict[str, Any] = {
                "id": node_id,
                "parent_id": incumbent_id,
                "lane": row_lane,
                "run_id": run_id,
                "trial_id": trial_id,
                "agent_model": agent_model or row.get("agent_model"),
                "author": author or agent_model or row.get("author"),
                "title": title,
                "hypothesis": hypothesis,
                "decision": decision,
                "soft_min_eff": row.get("soft_min_eff"),
                "ratio_vs_incumbent": row.get("ratio_vs_incumbent"),
                "short_horizon_eff": row.get("short_horizon_soft_min"),
                "long_horizon_eff": row.get("long_horizon_soft_min"),
                "overhead_ratio": row.get("overhead_ratio"),
                "b2_max": row.get("b2_max"),
                "dims": dims,
                "codex_refs": row.get("codex_refs") or [],
                "code_sha": row.get("candidate_sha") or row.get("code_sha"),
                "metrics_json": json.dumps(extra_metrics, sort_keys=True),
                "verified": 0,
                "created_at": row.get("timestamp"),
            }

            resp = post_node(node_payload, url=url, token=token, timeout=timeout)
            if resp is not None:
                results.append(resp)

            if decision in KEPT_DECISIONS:
                incumbent_id = node_id

    return results
